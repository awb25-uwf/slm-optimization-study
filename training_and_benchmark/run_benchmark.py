#!/usr/bin/env python3
"""
Fault-tolerant, checkpointed benchmark runner across all combinations of:
    model size:    phi4-mini | phi4-14b
    model state:   base | lora
    RAG:           on | off
    KG:            on | off
  -> 4 retrieval combos x 4 model states (2 sizes x base/lora) = 16 configs
  -> x 10 scenarios x N_TRIALS trials = 4800 generations at N_TRIALS=30

Greedy decoding (temperature 0, top_k 1, top_p 1, repeat_penalty 1.0) matches
the deployed system's intended deterministic behavior for this compliance-audit
use case.

Generation is served via Ollama. Each of the four model states was pre-converted
to GGUF (bf16, unquantized) and registered with `ollama create` before this script
runs -- see merge_lora_adapters.py, convert_to_gguf.py, and modelfiles/.
A one-time warm-up call precedes each model state's trial loop so the
model-load cost doesn't land inside a scored trial's resource sample.

IMPORTANT COMPARABILITY NOTE: base+adapter (transformers/PEFT, used by
earlier runs of this study), merged-then-saved (transformers, merged
weights), and Ollama-served (llama.cpp via GGUF, used by this version) were
empirically confirmed to NOT produce byte-identical outputs from each other
under greedy decoding on the same prompt -- verified and documented during
setup. This is expected (LoRA merging changes floating-point computation
order; llama.cpp is a separate inference engine with different kernels),
not a bug.

Trial ordering / interleaving strategy: mini and 14B use different interleaving
granularity, because 14B's model-swap cost is roughly 7x mini's.
  - phi4mini: trials interleave finely (scenario x combo x lora-state x trial).
    Stopping anywhere gives balanced coverage across essentially all 8 mini configs.
  - phi4_14b: lora/base is the outer, rare-swap loop (2 swaps total), with
    scenario x combo interleaved within each block. Balanced within a
    block; not balanced across base-vs-lora if stopped mid-block.

RAG and KG return the same number of citations to facilitate comparison.

Fault tolerance: each retrieval call is retried up to RETRY_ATTEMPTS times
with backoff. If still failing, the trial is written with
retrieval_failed=True and excluded from quality aggregates.

Checkpointing: every row is appended to CSV immediately (flush + fsync),
and a JSON checkpoint tracks completed (config, scenario_id, trial_num)
tuples.

Resource monitoring:
  - Inference host (this machine): each trial's Ollama generation call is
    wrapped in a resource_monitor.ResourceMonitor, sampling this machine's
    CPU/RAM/GPU every INFERENCE_SAMPLE_INTERVAL_SECONDS for that trial's
    duration. Summary stats are written to OUTPUT_CSV.
  - RAG/KG service hosts + containers: polled independently on a
    background thread for the full run duration, via each service's /metrics
    endpoint and a single shared cAdvisor instance. Written as
    timestamped rows to SERVICE_RESOURCE_LOG_CSV; correlate with a given
    trial via that trial's trial_start_ts/trial_end_ts columns in
    OUTPUT_CSV.

REQUIRES: resource_monitor.py must be present in the same directory as
this script.
"""
import json
import csv
import os
import sys
import time
import threading
import argparse
import requests
from prometheus_client.parser import text_string_to_metric_families

from benchmark_scorer import score_response, aggregate_scores
from resource_monitor import ResourceMonitor

# --- CONFIGURATION ---
BENCHMARK_FILE = "nerc_benchmark_seed.jsonl"
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_KEEP_ALIVE = "60m"  # allows the process to continue if retrieval retries
                            # ever create an unusually long gap between calls
OLLAMA_TIMEOUT_SECONDS = 120  # generation can take longer than retrieval calls
OLLAMA_SYSTEM_MESSAGE = "You are a NERC CIP compliance audit assistant."

RAG_SERVICE_URL = "http://192.168.0.100:8001"
KG_SERVICE_URL = "http://192.168.0.100:8002"

RETRIEVAL_TIMEOUT_SECONDS = 10
RETRY_ATTEMPTS = 3
RETRY_BACKOFF_SECONDS = 2.0  # doubled each retry

MAX_CITATIONS = 3

N_TRIALS = 30

OUTPUT_CSV = "benchmark_results_full.csv"
CHECKPOINT_FILE = "benchmark_full_checkpoint.json"

# --- Resource monitoring configuration -----------------------------------
# Inference-host (this machine -- the Jetson) resource use is sampled
# per-trial via ResourceMonitor, wrapped directly around each Ollama
# generation call, and written as extra columns on the same row in
# OUTPUT_CSV.
INFERENCE_SAMPLE_INTERVAL_SECONDS = 0.5

# RAG/KG service-host and cAdvisor resource use is polled independently on
# a background thread for the full duration of the run. Poll results are
# timestamped and written to a separate CSV; correlate with trial rows via
# trial_start_ts/trial_end_ts (added to OUTPUT_CSV) by matching timestamp
# ranges after the fact.
RESOURCE_POLL_INTERVAL_SECONDS = 2.0
SERVICE_RESOURCE_LOG_CSV = "service_resource_log.csv"

# Both RAG and KG services, plus their shared cAdvisor instance, run on
# the laptop. cAdvisor mounts the host's /rootfs, /var/lib/docker, and
# /sys, so one cAdvisor instance sees every container on that host --
# including neo4j/kg-api even though they're started from kg_service's own
# docker-compose.yml with no cadvisor service of its own. Keep the
# rag_service cAdvisor container running whenever the KG stack is also up.
CADVISOR_URL = "http://192.168.0.100:8080"
CADVISOR_CONTAINER_NAMES = ["nerc_cip_rag", "nerc_cip_chromadb", "nerc_cip_kg", "nerc_cip_neo4j"]

# Retrieval combos to sweep under each loaded model state.
RETRIEVAL_COMBOS = [
    {"rag": False, "kg": False, "label": "no_retrieval"},
    {"rag": True, "kg": False, "label": "with_rag"},
    {"rag": False, "kg": True, "label": "with_kg"},
    {"rag": True, "kg": True, "label": "with_rag_and_kg"},
]

PROMPT_TEMPLATE = """Context:
{context}

Instruction:
{instruction}

Respond with your compliance determination and reasoning. End your response with exactly these two lines:
Citation: <the specific standard and requirement/part/section you are citing>
Compliance Status: <compliant|partial|non_compliant>"""


# --- Retrieval with retry -----------------------------------------------

def _call_with_retry(url, payload, label):
    """POSTs to a retrieval service with retry+backoff. Returns
    (context, retrieval_time_sec, failed_bool)."""
    backoff = RETRY_BACKOFF_SECONDS
    last_error = None
    total_start = time.time()
    for attempt in range(1, RETRY_ATTEMPTS + 1):
        try:
            response = requests.post(url, json=payload, timeout=RETRIEVAL_TIMEOUT_SECONDS)
            response.raise_for_status()
            elapsed = time.time() - total_start
            return response.json().get("context", ""), elapsed, False
        except requests.exceptions.RequestException as e:
            last_error = e
            if attempt < RETRY_ATTEMPTS:
                print(f"[Warning] {label} call failed (attempt {attempt}/{RETRY_ATTEMPTS}): {e} "
                      f"-- retrying in {backoff:.1f}s...")
                time.sleep(backoff)
                backoff *= 2
    elapsed = time.time() - total_start
    print(f"[Error] {label} call failed after {RETRY_ATTEMPTS} attempts: {last_error} "
          f"-- marking trial retrieval_failed.")
    return "", elapsed, True


def retrieve_rag_context(scenario):
    return _call_with_retry(
        f"{RAG_SERVICE_URL}/retrieve",
        {
            "instruction": scenario.get("instruction", ""),
            "scenario_input": scenario.get("scenario_input", ""),
            "top_k": MAX_CITATIONS,
        },
        "RAG",
    )


def retrieve_kg_context(scenario):
    return _call_with_retry(
        f"{KG_SERVICE_URL}/resolve",
        {
            "instruction": scenario.get("instruction", ""),
            "scenario_input": scenario.get("scenario_input", ""),
            "max_citations": MAX_CITATIONS,
        },
        "KG",
    )


# --- Ollama generation -----------------------------------------------------

def generate_via_ollama(ollama_model, prompt_text):
    """POSTs to Ollama's /api/generate, with retry+backoff. Decoding parameters
    are passed explicitly on every request:
      - temperature 0, top_k 1, top_p 1 -> deterministic/greedy
      - repeat_penalty 1.0
      - num_predict 300

    Returns (generated_text, wall_elapsed_sec, ollama_timings_dict, failed_bool).
    ollama_timings_dict contains Ollama's self-reported nanosecond timings
    (total_duration, load_duration, prompt_eval_duration, eval_duration,
    prompt_eval_count, eval_count) when available, empty dict on failure.
    """
    payload = {
        "model": ollama_model,
        "system": OLLAMA_SYSTEM_MESSAGE,
        "prompt": prompt_text,
        "stream": False,
        "keep_alive": OLLAMA_KEEP_ALIVE,
        "options": {
            "temperature": 0,
            "top_k": 1,
            "top_p": 1,
            "repeat_penalty": 1.0,
            "num_predict": 300,
        },
    }

    backoff = RETRY_BACKOFF_SECONDS
    last_error = None
    total_start = time.time()
    for attempt in range(1, RETRY_ATTEMPTS + 1):
        try:
            response = requests.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json=payload,
                timeout=OLLAMA_TIMEOUT_SECONDS,
            )
            response.raise_for_status()
            data = response.json()
            elapsed = time.time() - total_start
            timings = {
                "ollama_total_duration_ns": data.get("total_duration"),
                "ollama_load_duration_ns": data.get("load_duration"),
                "ollama_prompt_eval_duration_ns": data.get("prompt_eval_duration"),
                "ollama_eval_duration_ns": data.get("eval_duration"),
                "ollama_prompt_eval_count": data.get("prompt_eval_count"),
                "ollama_eval_count": data.get("eval_count"),
            }
            return data.get("response", ""), elapsed, timings, False
        except requests.exceptions.RequestException as e:
            last_error = e
            if attempt < RETRY_ATTEMPTS:
                print(f"[Warning] Ollama generate call failed (attempt {attempt}/{RETRY_ATTEMPTS}): "
                      f"{e} -- retrying in {backoff:.1f}s...")
                time.sleep(backoff)
                backoff *= 2
    elapsed = time.time() - total_start
    print(f"[Error] Ollama generate call failed after {RETRY_ATTEMPTS} attempts: {last_error}")
    return "", elapsed, {}, True


def warm_up_ollama_model(ollama_model):
    """Force Ollama to load the model with a throwaway inference call."""
    print(f"Warming up {ollama_model} (discarding this call's output/timing)...")
    _, elapsed, timings, failed = generate_via_ollama(ollama_model, "Warm-up request; ignore.")
    if failed:
        print(f"[Warning] Warm-up call for {ollama_model} failed -- first real trial "
              f"may absorb the model-load cost instead.")
    else:
        load_ns = timings.get("ollama_load_duration_ns")
        load_sec = f"{load_ns / 1e9:.1f}s" if load_ns is not None else "unknown"
        print(f"Warm-up complete ({elapsed:.1f}s wall, {load_sec} model load).")


# --- Service/container resource polling ----------------------------------

def _scrape_service_gauges(url, service_label):
    """GETs a Prometheus text-exposition /metrics endpoint and returns a
    list of flat dict rows, one per gauge of interest."""
    wanted = {
        "resource_cpu_percent": "cpu_percent",
        "resource_ram_used_mb": "ram_used_mb",
        "resource_ram_percent": "ram_percent",
        "resource_gpu_percent": "gpu_percent",
    }
    rows = []
    try:
        resp = requests.get(f"{url}/metrics", timeout=5)
        resp.raise_for_status()
        ts = time.time()
        for family in text_string_to_metric_families(resp.text):
            if family.name in wanted:
                for sample in family.samples:
                    rows.append({
                        "timestamp": ts,
                        "source": service_label,
                        "container": "",
                        "metric": wanted[family.name],
                        "value": sample.value,
                    })
    except requests.exceptions.RequestException as e:
        print(f"[Warning] Resource poll of {service_label} ({url}/metrics) failed: {e}")
    return rows


def _scrape_cadvisor(url, container_names):
    """GETs cAdvisor's /metrics and extracts CPU (cumulative seconds)
    and memory usage (bytes) for the given container names."""
    wanted = {
        "container_cpu_usage_seconds_total": "cpu_usage_seconds_total",
        "container_memory_usage_bytes": "memory_usage_bytes",
    }
    rows = []
    try:
        resp = requests.get(f"{url}/metrics", timeout=5)
        resp.raise_for_status()
        ts = time.time()
        for family in text_string_to_metric_families(resp.text):
            if family.name not in wanted:
                continue
            for sample in family.samples:
                name = sample.labels.get("name", "")
                if name not in container_names:
                    continue
                rows.append({
                    "timestamp": ts,
                    "source": "cadvisor",
                    "container": name,
                    "metric": wanted[family.name],
                    "value": sample.value,
                })
    except requests.exceptions.RequestException as e:
        print(f"[Warning] cAdvisor poll ({url}/metrics) failed: {e}")
    return rows


def _service_resource_poller(stop_event, csv_path):
    """Runs on a background daemon thread for the full duration of the
    benchmark run. Every RESOURCE_POLL_INTERVAL_SECONDS, scrapes RAG's and
    KG's /metrics plus cAdvisor's /metrics (filtered to the four containers
    of interest), and appends timestamped rows to csv_path. Independent of
    the per-trial main loop.
    """
    fieldnames = ["timestamp", "source", "container", "metric", "value"]
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
            f.flush()
            os.fsync(f.fileno())

        while not stop_event.is_set():
            rows = []
            rows.extend(_scrape_service_gauges(RAG_SERVICE_URL, "rag_service"))
            rows.extend(_scrape_service_gauges(KG_SERVICE_URL, "kg_service"))
            rows.extend(_scrape_cadvisor(CADVISOR_URL, CADVISOR_CONTAINER_NAMES))

            for row in rows:
                writer.writerow(row)
            if rows:
                f.flush()
                os.fsync(f.fileno())

            stop_event.wait(RESOURCE_POLL_INTERVAL_SECONDS)


# --- Checkpointing --------------------------------------------------------

def load_checkpoint():
    """Returns the set of completed (config_label, scenario_id, trial_num)
    tuples from a prior run, or an empty set if none exists / unreadable."""
    if not os.path.exists(CHECKPOINT_FILE):
        return set()
    try:
        with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {tuple(item) for item in data.get("completed", [])}
    except (json.JSONDecodeError, OSError):
        print("[Warning] Checkpoint file unreadable/corrupt. Starting fresh.")
        return set()


def save_checkpoint(completed):
    with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
        json.dump({"completed": sorted(list(t) for t in completed)}, f)
        f.flush()
        os.fsync(f.fileno())


def load_benchmark(path):
    scenarios = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                scenarios.append(json.loads(line))
    return scenarios


# --- Main run ---------------------------------------------------------

def run_single_trial(config_label, model_state_label, ollama_model, combo, scenario, trial_num,
                      completed, csv_writer, csv_file, all_scores_by_config):
    """Runs exactly one trial: retrieval (if enabled) + generation + scoring
    + CSV row write + checkpoint update. Shared by both the mini (finely
    interleaved) and 14B (blocked by lora/base) orchestration functions."""
    key = (config_label, scenario["id"], trial_num)
    if key in completed:
        return  # already done in a prior interrupted run

    rag_context, rag_time, rag_failed = ("", 0.0, False)
    kg_context, kg_time, kg_failed = ("", 0.0, False)

    if combo["rag"]:
        rag_context, rag_time, rag_failed = retrieve_rag_context(scenario)
    if combo["kg"]:
        kg_context, kg_time, kg_failed = retrieve_kg_context(scenario)

    retrieval_failed = (combo["rag"] and rag_failed) or (combo["kg"] and kg_failed)

    retrieved_parts = [p for p in [rag_context, kg_context] if p]
    combined_context = f"{scenario['scenario_input']}\n\n{chr(10).join(retrieved_parts)}".strip()

    # Ollama applies the GGUF's embedded chat template itself, given
    # separate "system" and "prompt" fields.
    prompt_text = PROMPT_TEMPLATE.format(
        context=combined_context,
        instruction=scenario["instruction"],
    )

    trial_start_ts = time.time()
    with ResourceMonitor(
        job_name=f"{config_label}_{scenario['id']}_t{trial_num}",
        sample_interval=INFERENCE_SAMPLE_INTERVAL_SECONDS,
        write_prom_file=False,
        verbose=False,          
    ) as res_mon:
        generated_text, gen_elapsed, ollama_timings, gen_failed = generate_via_ollama(
            ollama_model, prompt_text
        )
    trial_end_ts = time.time()

    retrieval_failed = retrieval_failed or gen_failed

    score = score_response(scenario, generated_text)

    if not retrieval_failed:
        all_scores_by_config.setdefault(config_label, []).append(score)

    status_icon = "PASS" if score.full_pass else "FAIL"
    fail_tags = []
    if rag_failed or kg_failed:
        fail_tags.append("RETRIEVAL_FAILED")
    if gen_failed:
        fail_tags.append("GENERATION_FAILED")
    fail_tag = f" [{', '.join(fail_tags)}]" if fail_tags else ""
    print(f"[{config_label}] {scenario['id']} trial {trial_num}/{N_TRIALS}: "
          f"citation={'Y' if score.citation_correct else 'N'} "
          f"status={'Y' if score.status_correct else 'N'} "
          f"facts={score.key_fact_recall:.2f} -> {status_icon} "
          f"({gen_elapsed:.1f}s){fail_tag}")

    row = {
        "config_label": config_label,
        "model_state": model_state_label,
        "rag_enabled": combo["rag"],
        "kg_enabled": combo["kg"],
        "scenario_id": scenario["id"],
        "trial_num": trial_num,
        "standard_family": scenario.get("standard_family", ""),
        "retrieval_failed": retrieval_failed,
        "citation_correct": score.citation_correct,
        "status_correct": score.status_correct,
        "status_found": score.status_found,
        "key_fact_recall": round(score.key_fact_recall, 3),
        "full_pass": score.full_pass,
        "extracted_citation": score.extracted_citation,
        "extracted_status": score.extracted_status,
        "expected_citation": scenario["ground_truth_citation"],
        "expected_status": scenario["ground_truth_compliance_status"],
        "rag_context": rag_context,
        "rag_retrieval_time_sec": round(rag_time, 3),
        "kg_context": kg_context,
        "kg_retrieval_time_sec": round(kg_time, 3),
        "model_output": generated_text,
        "generation_time_sec": round(gen_elapsed, 2),
        "generation_failed": gen_failed,
        "trial_start_ts": round(trial_start_ts, 3),
        "trial_end_ts": round(trial_end_ts, 3),
        "gen_cpu_percent_mean": res_mon.summary.get("cpu_percent_mean"),
        "gen_cpu_percent_peak": res_mon.summary.get("cpu_percent_peak"),
        "gen_ram_used_mb_mean": res_mon.summary.get("ram_used_mb_mean"),
        "gen_ram_used_mb_peak": res_mon.summary.get("ram_used_mb_peak"),
        "gen_gpu_percent_mean": res_mon.summary.get("gpu_percent_mean"),
        "gen_gpu_percent_peak": res_mon.summary.get("gpu_percent_peak"),
        "gen_resource_n_samples": res_mon.summary.get("n_samples"),
        "gen_power_total_mw_mean": res_mon.summary.get("power_total_mw_mean"),
        "gen_power_total_mw_peak": res_mon.summary.get("power_total_mw_peak"),
        "gen_power_cpu_mw_mean": res_mon.summary.get("power_cpu_mw_mean"),
        "gen_power_gpu_mw_mean": res_mon.summary.get("power_gpu_mw_mean"),
        "gen_temp_cpu_c_mean": res_mon.summary.get("temp_cpu_c_mean"),
        "gen_temp_cpu_c_peak": res_mon.summary.get("temp_cpu_c_peak"),
        "gen_temp_gpu_c_mean": res_mon.summary.get("temp_gpu_c_mean"),
        "gen_temp_gpu_c_peak": res_mon.summary.get("temp_gpu_c_peak"),
        "gen_energy_wh_estimate": res_mon.summary.get("energy_wh_estimate"),
        "ollama_total_duration_ns": ollama_timings.get("ollama_total_duration_ns"),
        "ollama_load_duration_ns": ollama_timings.get("ollama_load_duration_ns"),
        "ollama_prompt_eval_duration_ns": ollama_timings.get("ollama_prompt_eval_duration_ns"),
        "ollama_eval_duration_ns": ollama_timings.get("ollama_eval_duration_ns"),
        "ollama_prompt_eval_count": ollama_timings.get("ollama_prompt_eval_count"),
        "ollama_eval_count": ollama_timings.get("ollama_eval_count"),
    }
    csv_writer.writerow(row)
    csv_file.flush()
    os.fsync(csv_file.fileno())

    completed.add(key)
    save_checkpoint(completed)


def run_mini_interleaved(scenarios, completed, csv_writer, csv_file, all_scores_by_config):
    print(f"\n{'#' * 70}")
    print("# phi4mini -- fine-grained interleave (scenario x combo x lora-state x trial)")
    print(f"{'#' * 70}")

    warm_up_ollama_model("phi4mini_base")
    warm_up_ollama_model("phi4mini_lora")

    for scenario in scenarios:
        for combo in RETRIEVAL_COMBOS:
            for lora_suffix in ["base", "lora"]:
                model_state_label = f"phi4mini_{lora_suffix}"
                ollama_model = model_state_label
                config_label = f"{model_state_label}_{combo['label']}"
                for trial_num in range(1, N_TRIALS + 1):
                    run_single_trial(
                        config_label, model_state_label, ollama_model, combo,
                        scenario, trial_num, completed, csv_writer, csv_file,
                        all_scores_by_config,
                    )


def run_14b_blocked(scenarios, completed, csv_writer, csv_file, all_scores_by_config):
    print(f"\n{'#' * 70}")
    print("# phi4_14b -- blocked by lora-state (swap cost ~7min), interleaved within each block")
    print(f"{'#' * 70}")

    for lora_suffix in ["base", "lora"]:
        model_state_label = f"phi4_14b_{lora_suffix}"
        ollama_model = model_state_label

        print(f"\n{'=' * 70}")
        print(f"[Block] {model_state_label}")
        print(f"{'=' * 70}")

        warm_up_ollama_model(ollama_model)

        for scenario in scenarios:
            for combo in RETRIEVAL_COMBOS:
                config_label = f"{model_state_label}_{combo['label']}"
                for trial_num in range(1, N_TRIALS + 1):
                    run_single_trial(
                        config_label, model_state_label, ollama_model, combo,
                        scenario, trial_num, completed, csv_writer, csv_file,
                        all_scores_by_config,
                    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fresh", action="store_true",
                         help="Ignore any existing checkpoint/CSV and start over.")
    parser.add_argument("--sizes", nargs="+", choices=["phi4mini", "phi4_14b"],
                         default=["phi4mini", "phi4_14b"],
                         help="Restrict this run to specific model sizes, e.g. "
                              "'--sizes phi4mini' to run only the mini configs (8 configs) "
                              "and defer phi4_14b to a later invocation. Checkpointing means "
                              "a later run with a different --sizes selection adds to the "
                              "same results file rather than conflicting.")
    args = parser.parse_args()

    completed = set()
    write_header = True

    if args.fresh:
        for f in (OUTPUT_CSV, CHECKPOINT_FILE):
            if os.path.exists(f):
                backup = f + ".bak"
                print(f"[--fresh] Backing up {f} -> {backup}")
                os.replace(f, backup)
    else:
        completed = load_checkpoint()
        if completed:
            print(f"[Resume] Found checkpoint with {len(completed)} trial(s) already completed.")
            write_header = not os.path.exists(OUTPUT_CSV)
        else:
            print("[Start] No checkpoint found -- fresh run.")

    scenarios = load_benchmark(BENCHMARK_FILE)
    n_configs = len(args.sizes) * 2 * len(RETRIEVAL_COMBOS)  # 2 = base/lora per size
    total_generations = len(scenarios) * N_TRIALS * n_configs
    print(f"[Plan] Sizes: {args.sizes}")
    print(f"[Plan] {len(scenarios)} scenarios x {N_TRIALS} trials x "
          f"{len(RETRIEVAL_COMBOS)} retrieval combos x {len(args.sizes) * 2} model states = "
          f"{total_generations} total generations planned.")
    print(f"[Plan] MAX_CITATIONS={MAX_CITATIONS} (matched across RAG top_k and KG max_citations)")
    if "phi4mini" in args.sizes:
        print("[Plan] phi4mini: fine-grained interleave (scenario x combo x lora-state x trial) "
              "-- balanced coverage even if stopped early.")
    if "phi4_14b" in args.sizes:
        print("[Plan] phi4_14b: blocked by lora-state (~7min swap cost) -- balanced across "
              "scenario/combo within each block, but base-vs-lora coverage is unbalanced if "
              "stopped mid-block.")

    fieldnames = [
        "config_label", "model_state", "rag_enabled", "kg_enabled",
        "scenario_id", "trial_num", "standard_family", "retrieval_failed",
        "citation_correct", "status_correct", "status_found", "key_fact_recall",
        "full_pass", "extracted_citation", "extracted_status",
        "expected_citation", "expected_status",
        "rag_context", "rag_retrieval_time_sec", "kg_context", "kg_retrieval_time_sec",
        "model_output", "generation_time_sec", "generation_failed",
        "trial_start_ts", "trial_end_ts",
        "gen_cpu_percent_mean", "gen_cpu_percent_peak",
        "gen_ram_used_mb_mean", "gen_ram_used_mb_peak",
        "gen_gpu_percent_mean", "gen_gpu_percent_peak", "gen_resource_n_samples",
        "gen_power_total_mw_mean", "gen_power_total_mw_peak",
        "gen_power_cpu_mw_mean", "gen_power_gpu_mw_mean",
        "gen_temp_cpu_c_mean", "gen_temp_cpu_c_peak",
        "gen_temp_gpu_c_mean", "gen_temp_gpu_c_peak",
        "gen_energy_wh_estimate",
        "ollama_total_duration_ns", "ollama_load_duration_ns",
        "ollama_prompt_eval_duration_ns", "ollama_eval_duration_ns",
        "ollama_prompt_eval_count", "ollama_eval_count",
    ]

    all_scores_by_config = {}

    print(f"[Plan] Polling RAG/KG services + cAdvisor every "
          f"{RESOURCE_POLL_INTERVAL_SECONDS}s -> {SERVICE_RESOURCE_LOG_CSV}")
    poller_stop_event = threading.Event()
    poller_thread = threading.Thread(
        target=_service_resource_poller,
        args=(poller_stop_event, SERVICE_RESOURCE_LOG_CSV),
        daemon=True,
    )
    poller_thread.start()

    try:
        # Append mode is essential for resuming; header only written once.
        with open(OUTPUT_CSV, "a", newline="", encoding="utf-8") as csv_file:
            csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            if write_header:
                csv_writer.writeheader()
                csv_file.flush()
                os.fsync(csv_file.fileno())

            for size in args.sizes:
                if size == "phi4mini":
                    run_mini_interleaved(scenarios, completed, csv_writer, csv_file, all_scores_by_config)
                elif size == "phi4_14b":
                    run_14b_blocked(scenarios, completed, csv_writer, csv_file, all_scores_by_config)
    finally:
        poller_stop_event.set()
        poller_thread.join(timeout=RESOURCE_POLL_INTERVAL_SECONDS + 5)

    print(f"\n{'#' * 70}")
    print("FINAL CROSS-CONFIG SUMMARY")
    print(f"{'#' * 70}")
    for config_label, scores in all_scores_by_config.items():
        if not scores:
            print(f"\n{config_label}: no successful trials")
            continue
        summary = aggregate_scores(scores)
        print(f"\n{config_label}  (n={len(scores)}):")
        for k, v in summary.items():
            print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")

    print(f"\n[Done] Full results in {OUTPUT_CSV}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[Terminated] Stopped by user. Progress saved -- rerun to resume.")
        sys.exit(0)
