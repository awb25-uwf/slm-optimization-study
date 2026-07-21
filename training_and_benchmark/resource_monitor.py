#!/usr/bin/env python3
"""
Shared resource monitoring.

For batch jobs: use ResourceMonitor as a context manager by pointing node_exporter's
--collector.textfile.directory at the same folder these files are written to.

For the live API server: use the exported prometheus_client Gauges directly
and start_background_sampler() to keep them updated continuously while the
server runs.

For high-frequency per-trial use: pass write_prom_file=False to ResourceMonitor.
summary computation and console printing still happen; only the textfile write is
skipped. Default is True.
"""
import time
import threading
import psutil
from prometheus_client import Gauge, CollectorRegistry, write_to_textfile

# jetson-stats (jtop) is optional -- provides power draw, temperatures, and
# clock frequencies that the plain psutil/sysfs path below cannot. Falls
# back gracefully (all jtop-sourced fields = None) if the package isn't
# installed or the jtop service isn't reachable, so this module still works
# unmodified on non-Jetson hosts (e.g. LAPTOP-F7IOVI6T running rag_api.py /
# kg_api.py) and doesn't hard-require jetson-stats as a dependency there.
try:
    from jtop import jtop
    _JTOP_IMPORT_OK = True
except ImportError:
    _JTOP_IMPORT_OK = False

_jtop_instance = None
_jtop_failed = False
_jtop_lock = threading.Lock()


def _get_jtop():
    """Returns a live  jtop connection, creating it on first use and
    reusing it thereafter.."""
    global _jtop_instance, _jtop_failed
    if not _JTOP_IMPORT_OK or _jtop_failed:
        return None
    with _jtop_lock:
        if _jtop_instance is None:
            try:
                _jtop_instance = jtop()
                _jtop_instance.start()
            except Exception as e:
                print(f"[resource_monitor] Could not start jtop connection ({e}). "
                      f"Falling back to psutil/sysfs-only sampling. Check: "
                      f"'sudo systemctl status jtop.service' and that your user "
                      f"is in the jtop group (may require logout/reboot after install).")
                _jtop_failed = True
                return None
    return _jtop_instance


def read_jtop_stats():
    """Returns a dict of Jetson-specific readings sourced from jtop
    or a dict of all-None values if jtop is unavailable for any reason.

    Fields names were verified against jetson-stats 7.2.0 for AGX Orin.

    Known caveats from that verification:
      - power_cpu_mw is sourced from the "VDD_CPU_CV" rail, which combines
        CPU + CV power on this board. Similarly power_gpu_mw is sourced
        from "VDD_GPU_SOC". power_total_mw is unaffected by this and is
        accurate as a whole-module reading.
      - temp_gpu_c: this board's GPU temperature sensor reports the
        sentinel value -256.
    """
    empty = {
        "power_total_mw": None,
        "power_cpu_mw": None,
        "power_gpu_mw": None,
        "temp_cpu_c": None,
        "temp_gpu_c": None,
        "gpu_percent_jtop": None,  # prefer this over sysfs read_jetson_gpu_load() when available
    }
    jetson = _get_jtop()
    if jetson is None:
        return empty
    try:
        stats = jetson.stats
        power = jetson.power  # structured dict: {"tot": {"power": mW}, "rail": {name: {"power": mW, ...}}}

        reading = dict(empty)

        if isinstance(power, dict):
            tot = power.get("tot", {})
            reading["power_total_mw"] = tot.get("power")
            rails = power.get("rail", {})
            for rail_name, rail_data in rails.items():
                rn = rail_name.upper()
                if reading["power_cpu_mw"] is None and "CPU" in rn:
                    reading["power_cpu_mw"] = rail_data.get("power")
                if reading["power_gpu_mw"] is None and "GPU" in rn:
                    reading["power_gpu_mw"] = rail_data.get("power")

        if isinstance(stats, dict):
            temp_cpu = stats.get("Temp cpu")
            reading["temp_cpu_c"] = temp_cpu if temp_cpu is not None and temp_cpu > -200 else None

            temp_gpu = stats.get("Temp gpu")
            reading["temp_gpu_c"] = temp_gpu if temp_gpu is not None and temp_gpu > -200 else None

            reading["gpu_percent_jtop"] = stats.get("GPU")

        return reading
    except Exception:
        return empty

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
    """Returns a dict of current CPU/RAM/GPU readings, plus Jetson power
    draw/temperature fields from jtop when available (all None if jtop
    isn't installed/connected). GPU utilization prefers jtop's directly-reported
    value over the raw sysfs read when jtop is connected."""
    mem = psutil.virtual_memory()
    jtop_reading = read_jtop_stats()

    gpu_percent = jtop_reading.get("gpu_percent_jtop")
    if gpu_percent is None:
        gpu_percent = read_jetson_gpu_load()

    reading = {
        "cpu_percent": psutil.cpu_percent(interval=0.1),
        "ram_used_mb": mem.used / (1024 * 1024),
        "ram_percent": mem.percent,
        "gpu_percent": gpu_percent,
    }
    reading.update(jtop_reading)
    return reading


def update_live_gauges():
    """Updates the module-level Gauges from one sample"""
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

    Set write_prom_file=False for high-frequency per-call use where writing
    an individual .prom file per call would spam the filesystem with no
    Prometheus server present to scrape them. Summary computation and
    console printing are unaffected by this flag.
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
        power_tot_vals = [s["power_total_mw"] for s in self.samples if s.get("power_total_mw") is not None]
        power_cpu_vals = [s["power_cpu_mw"] for s in self.samples if s.get("power_cpu_mw") is not None]
        power_gpu_vals = [s["power_gpu_mw"] for s in self.samples if s.get("power_gpu_mw") is not None]
        temp_cpu_vals = [s["temp_cpu_c"] for s in self.samples if s.get("temp_cpu_c") is not None]
        temp_gpu_vals = [s["temp_gpu_c"] for s in self.samples if s.get("temp_gpu_c") is not None]

        def _mean_peak(vals):
            if not vals:
                return None, None
            return round(sum(vals) / len(vals), 2), round(max(vals), 2)

        cpu_mean, cpu_peak = _mean_peak(cpu_vals)
        ram_mean, ram_peak = _mean_peak(ram_vals)
        gpu_mean, gpu_peak = _mean_peak(gpu_vals)
        power_tot_mean, power_tot_peak = _mean_peak(power_tot_vals)
        power_cpu_mean, power_cpu_peak = _mean_peak(power_cpu_vals)
        power_gpu_mean, power_gpu_peak = _mean_peak(power_gpu_vals)
        temp_cpu_mean, temp_cpu_peak = _mean_peak(temp_cpu_vals)
        temp_gpu_mean, temp_gpu_peak = _mean_peak(temp_gpu_vals)

        self.summary = {
            "job_name": self.job_name,
            "duration_seconds": round(duration, 2),
            "n_samples": len(self.samples),
            "cpu_percent_mean": cpu_mean,
            "cpu_percent_peak": cpu_peak,
            "ram_used_mb_mean": ram_mean,
            "ram_used_mb_peak": ram_peak,
            "gpu_percent_mean": gpu_mean,
            "gpu_percent_peak": gpu_peak,
            # Jetson-only (jtop) fields -- None if jtop unavailable on this host.
            "power_total_mw_mean": power_tot_mean,
            "power_total_mw_peak": power_tot_peak,
            "power_cpu_mw_mean": power_cpu_mean,
            "power_cpu_mw_peak": power_cpu_peak,
            "power_gpu_mw_mean": power_gpu_mean,
            "power_gpu_mw_peak": power_gpu_peak,
            "temp_cpu_c_mean": temp_cpu_mean,
            "temp_cpu_c_peak": temp_cpu_peak,
            "temp_gpu_c_mean": temp_gpu_mean,
            "temp_gpu_c_peak": temp_gpu_peak,
        }

        # Convenience field for Objective 4 (accuracy / energy in watt-hours):
        # energy (Wh) ~= mean power (W) x duration (h). This is an
        # approximation using the mean of discrete samples rather than a
        # true trapezoidal integral over continuous power draw -- fine at
        # the INFERENCE_SAMPLE_INTERVAL_SECONDS=0.5 sampling rate used for
        # per-trial generation calls (short, relatively stable duration),
        # but treat it as an estimate, not a metered reading.
        if power_tot_mean is not None:
            energy_wh = (power_tot_mean / 1000.0) * (duration / 3600.0)
            self.summary["energy_wh_estimate"] = round(energy_wh, 6)
        else:
            self.summary["energy_wh_estimate"] = None

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

        jtop_gauge_fields = [
            ("power_total_mw_mean", "batch_job_power_total_mw_mean", "Mean total power draw (mW)"),
            ("power_total_mw_peak", "batch_job_power_total_mw_peak", "Peak total power draw (mW)"),
            ("temp_cpu_c_mean", "batch_job_temp_cpu_c_mean", "Mean CPU temperature (C)"),
            ("temp_gpu_c_mean", "batch_job_temp_gpu_c_mean", "Mean GPU temperature (C)"),
            ("energy_wh_estimate", "batch_job_energy_wh_estimate", "Estimated energy used (Wh)"),
        ]
        for summary_key, gauge_name, gauge_help in jtop_gauge_fields:
            value = self.summary.get(summary_key)
            if value is not None:
                g = Gauge(gauge_name, gauge_help, ["job_name"], registry=registry)
                g.labels(job_name=self.job_name).set(value)

        import os
        os.makedirs(self.output_dir, exist_ok=True)
        output_path = os.path.join(self.output_dir, f"{self.job_name}.prom")
        write_to_textfile(output_path, registry)
        if self.verbose:
            print(f"[ResourceMonitor] Wrote Prometheus textfile metrics to {output_path}")
