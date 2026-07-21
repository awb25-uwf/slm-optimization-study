#!/usr/bin/env python3
"""
Shared resource monitoring, used both for one-off batch jobs (e.g. building
the vector DB) and for live metrics inside the API server.

For batch jobs: use ResourceMonitor as a context manager. It samples CPU/RAM
(via psutil, portable) and Jetson GPU load (best-effort, via sysfs) on a
background thread, then writes a Prometheus textfile-collector-compatible
.prom file on exit -- this is the standard way to get one-off job metrics
into Prometheus without needing a Pushgateway: point node_exporter's
--collector.textfile.directory at the same folder these files are written to.

For the live API server: use the exported prometheus_client Gauges directly
(GPU_UTIL_GAUGE, CPU_PERCENT_GAUGE, etc.) and start_background_sampler() to
keep them updated continuously while the server runs.

For high-frequency per-trial use (e.g. wrapping thousands of individual
benchmark generations) where writing a .prom file per call would spam the
filesystem with no Prometheus server to scrape them: pass
write_prom_file=False to ResourceMonitor. summary computation and console
printing still happen; only the textfile write is skipped. Default is True,
so existing callers (e.g. build_vector_db.py) are unaffected.
"""
import time
import threading
import psutil
from prometheus_client import Gauge, CollectorRegistry, write_to_textfile

# Jetson GPU load sysfs path -- confirmed location varies by L4T/JetPack version.
# This is the common path for recent Orin-based JetPack 6 releases; if it
# doesn't exist, GPU monitoring silently falls back to unavailable rather
# than crashing the whole job.
JETSON_GPU_LOAD_PATHS = [
    "/sys/devices/platform/host1x/17000000.gpu/load",
    "/sys/devices/gpu.0/load",
]

# --- Live gauges for use inside the API server (module-level, default registry) ---
CPU_PERCENT_GAUGE = Gauge("resource_cpu_percent", "CPU utilization percent")
RAM_USED_MB_GAUGE = Gauge("resource_ram_used_mb", "RAM used in MB")
RAM_PERCENT_GAUGE = Gauge("resource_ram_percent", "RAM utilization percent")
GPU_PERCENT_GAUGE = Gauge("resource_gpu_percent", "GPU utilization percent (Jetson, best-effort)")


def read_jetson_gpu_load():
    """Returns GPU load 0-100, or None if the sysfs path isn't found on this
    device -- verify the correct path with: cat /sys/devices/platform/host1x/17000000.gpu/load
    and adjust JETSON_GPU_LOAD_PATHS above if it differs on your JetPack version."""
    for path in JETSON_GPU_LOAD_PATHS:
        try:
            with open(path, "r") as f:
                raw = int(f.read().strip())
                return raw / 10.0  # Jetson reports in tenths of a percent
        except (FileNotFoundError, ValueError, PermissionError):
            continue
    return None


def sample_once():
    """Returns a dict of current CPU/RAM/GPU readings."""
    mem = psutil.virtual_memory()
    return {
        "cpu_percent": psutil.cpu_percent(interval=0.1),
        "ram_used_mb": mem.used / (1024 * 1024),
        "ram_percent": mem.percent,
        "gpu_percent": read_jetson_gpu_load(),
    }


def update_live_gauges():
    """Updates the module-level Gauges from one sample -- call this
    periodically (e.g. from a background thread) while the API server runs."""
    reading = sample_once()
    CPU_PERCENT_GAUGE.set(reading["cpu_percent"])
    RAM_USED_MB_GAUGE.set(reading["ram_used_mb"])
    RAM_PERCENT_GAUGE.set(reading["ram_percent"])
    if reading["gpu_percent"] is not None:
        GPU_PERCENT_GAUGE.set(reading["gpu_percent"])


def start_background_sampler(interval_seconds=2.0):
    """Starts a daemon thread that keeps the live gauges updated. Call once
    at API server startup."""
    def _loop():
        while True:
            try:
                update_live_gauges()
            except Exception:
                pass
            time.sleep(interval_seconds)

    thread = threading.Thread(target=_loop, daemon=True)
    thread.start()
    return thread


class ResourceMonitor:
    """Context manager for one-off batch jobs. Samples resource usage on a
    background thread for the duration of the `with` block, then writes a
    Prometheus textfile-collector .prom file with summary stats (peak/mean)
    plus job duration.

    Usage:
        with ResourceMonitor("vector_db_build", output_dir="./prom_textfile") as mon:
            build_the_thing()
        # mon.summary is also available as a dict after the block exits

    Set write_prom_file=False for high-frequency per-call use (e.g. one
    instance per benchmark trial across thousands of trials) where writing
    an individual .prom file per call would spam the filesystem with no
    Prometheus server present to scrape them. Summary computation and
    console printing are unaffected by this flag -- only the textfile
    write is skipped.
    """

    def __init__(self, job_name: str, output_dir: str = ".", sample_interval: float = 1.0,
                 write_prom_file: bool = True, verbose: bool = True):
        self.job_name = job_name
        self.output_dir = output_dir
        self.sample_interval = sample_interval
        self.write_prom_file = write_prom_file
        self.verbose = verbose
        self.samples = []
        self._stop_event = threading.Event()
        self._thread = None
        self.start_time = None
        self.end_time = None
        self.summary = {}

    def _sample_loop(self):
        while not self._stop_event.is_set():
            try:
                self.samples.append(sample_once())
            except Exception:
                pass
            self._stop_event.wait(self.sample_interval)

    def __enter__(self):
        self.start_time = time.time()
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        self._stop_event.set()
        self._thread.join(timeout=5)
        self._compute_summary()
        if self.verbose:
            self._print_summary()
        if self.write_prom_file:
            self._write_prom_file()
        return False  # don't suppress exceptions

    def _compute_summary(self):
        duration = self.end_time - self.start_time
        cpu_vals = [s["cpu_percent"] for s in self.samples]
        ram_vals = [s["ram_used_mb"] for s in self.samples]
        gpu_vals = [s["gpu_percent"] for s in self.samples if s["gpu_percent"] is not None]

        self.summary = {
            "job_name": self.job_name,
            "duration_seconds": round(duration, 2),
            "n_samples": len(self.samples),
            "cpu_percent_mean": round(sum(cpu_vals) / len(cpu_vals), 2) if cpu_vals else None,
            "cpu_percent_peak": round(max(cpu_vals), 2) if cpu_vals else None,
            "ram_used_mb_mean": round(sum(ram_vals) / len(ram_vals), 2) if ram_vals else None,
            "ram_used_mb_peak": round(max(ram_vals), 2) if ram_vals else None,
            "gpu_percent_mean": round(sum(gpu_vals) / len(gpu_vals), 2) if gpu_vals else None,
            "gpu_percent_peak": round(max(gpu_vals), 2) if gpu_vals else None,
        }

    def _print_summary(self):
        print(f"\n[ResourceMonitor] {self.job_name} summary:")
        for k, v in self.summary.items():
            print(f"  {k}: {v}")

    def _write_prom_file(self):
        # Write Prometheus textfile-collector format
        registry = CollectorRegistry()
        duration_gauge = Gauge(f"batch_job_duration_seconds", "Batch job duration",
                                ["job_name"], registry=registry)
        duration_gauge.labels(job_name=self.job_name).set(self.end_time - self.start_time)

        cpu_mean_gauge = Gauge("batch_job_cpu_percent_mean", "Mean CPU percent during job",
                                ["job_name"], registry=registry)
        if self.summary["cpu_percent_mean"] is not None:
            cpu_mean_gauge.labels(job_name=self.job_name).set(self.summary["cpu_percent_mean"])

        cpu_peak_gauge = Gauge("batch_job_cpu_percent_peak", "Peak CPU percent during job",
                                ["job_name"], registry=registry)
        if self.summary["cpu_percent_peak"] is not None:
            cpu_peak_gauge.labels(job_name=self.job_name).set(self.summary["cpu_percent_peak"])

        ram_mean_gauge = Gauge("batch_job_ram_used_mb_mean", "Mean RAM used (MB) during job",
                                ["job_name"], registry=registry)
        if self.summary["ram_used_mb_mean"] is not None:
            ram_mean_gauge.labels(job_name=self.job_name).set(self.summary["ram_used_mb_mean"])

        ram_peak_gauge = Gauge("batch_job_ram_used_mb_peak", "Peak RAM used (MB) during job",
                                ["job_name"], registry=registry)
        if self.summary["ram_used_mb_peak"] is not None:
            ram_peak_gauge.labels(job_name=self.job_name).set(self.summary["ram_used_mb_peak"])

        if self.summary["gpu_percent_mean"] is not None:
            gpu_mean_gauge = Gauge("batch_job_gpu_percent_mean", "Mean GPU percent during job",
                                    ["job_name"], registry=registry)
            gpu_mean_gauge.labels(job_name=self.job_name).set(self.summary["gpu_percent_mean"])

            gpu_peak_gauge = Gauge("batch_job_gpu_percent_peak", "Peak GPU percent during job",
                                    ["job_name"], registry=registry)
            gpu_peak_gauge.labels(job_name=self.job_name).set(self.summary["gpu_percent_peak"])

        import os
        os.makedirs(self.output_dir, exist_ok=True)
        output_path = os.path.join(self.output_dir, f"{self.job_name}.prom")
        write_to_textfile(output_path, registry)
        if self.verbose:
            print(f"[ResourceMonitor] Wrote Prometheus textfile metrics to {output_path}")
