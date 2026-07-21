#!/usr/bin/env python3
import json
import csv
import os
import sys
import time
import threading
import argparse
import torch
import requests
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from prometheus_client.parser import text_string_to_metric_families

from benchmark_scorer import score_response, aggregate_scores
from resource_monitor import ResourceMonitor

# --- CONFIGURATION ---
BENCHMARK_FILE = "nerc_benchmark_seed.jsonl"

MODEL_SIZES = [
    {
        "size_label": "phi4mini",
        "base_model_name": "microsoft/Phi-4-mini-instruct",
        "adapter_dir": "/mnt/ollama_repo/phi4_nerc_cip_lora",
    },
    {
        "size_label": "phi4_14b",
        "base_model_name": "microsoft/phi-4",
        "adapter_dir": "/mnt/ollama_repo/phi4_14b_nerc_cip_lora",
    },
]

RAG_SERVICE_URL = "http://192.168.0.100:8001"
KG_SERVICE_URL = "http://192.168.0.100:8002"

RETRIEVAL_TIMEOUT_SECONDS = 10
RETRY_ATTEMPTS = 3
RETRY_BACKOFF_SECONDS = 2.0

MAX_CITATIONS = 3

N_TRIALS = 30

OUTPUT_CSV = "benchmark_results_full.csv"
CHECKPOINT_FILE = "benchmark_full_checkpoint.json"

# --- Resource monitoring configuration -----------------------------------
INFERENCE_SAMPLE_INTERVAL_SECONDS = 0.5

RESOURCE_POLL_INTERVAL_SECONDS = 2.0
SERVICE_RESOURCE_LOG_CSV = "service_resource_log.csv"

CADVISOR_URL = "http://192.168.0.100:8080"
CADVISOR_CONTAINER_NAMES = ["nerc_cip_rag", "nerc_cip_chromadb", "nerc_cip_kg", "nerc_cip_neo4j"]

MODEL_STATES = []
for _size in MODEL_SIZES:
    MODEL_STATES.append({
        "label": f"{_size['size_label']}_base",
        "base_model_name": _size["base_model_name"],
        "adapter_dir": None,
    })
    MODEL_STATES.append({
        "label": f"{_size['size_label']}_lora",
        "base_model_name": _size["base_model_name"],
        "adapter_dir": _size["adapter_dir"],
    })

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
    (context, retrieval_time_sec, failed_bool). Never raises -- a failure
    after all retries is reported via the failed flag, not an exception,
    so the outer loop can keep going and log it cleanly."""
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


# --- Service/container resource polling ----------------------------------

def _scrape_service_gauges(url, service_label):
    """GETs a Prometheus text-exposition /metrics endpoint (like rag_api.py's
    or kg_api.py's, backed by resource_monitor's live gauges) and returns a
    list of flat dict rows, one per gauge of interest. Never raises -- a
    scrape failure is logged and returns an empty list, so a momentarily
    unreachable service doesn't kill the polling thread."""
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
    """GETs cAdvisor's /metrics and extracts CPU (cumulative seconds -- we
    report the raw counter; rate/delta can be computed post-hoc from
    consecutive timestamped rows) and memory usage (bytes) for the given
    container names only. cAdvisor exposes many metrics per container;
    filtering keeps the log focused on what's relevant here."""
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

def run_model_state(model_state, scenarios, completed, csv_writer, csv_file, all_scores_by_config):
    """Loads one model state (base or LoRA) ONCE, then runs all four
    retrieval combos x all scenarios x all trials against it before
    unloading. This is the "load once per model state" grouping."""
    print(f"\n{'#' * 70}")
    print(f"# Loading model state: {model_state['label']}")
    print(f"{'#' * 70}")

    print(f"Loading base model ({model_state['base_model_name']})...")
    model = AutoModelForCausalLM.from_pretrained(
        model_state["base_model_name"],
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer_source = model_state["adapter_dir"] if model_state["adapter_dir"] else model_state["base_model_name"]
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=True)

    if model_state["adapter_dir"]:
        print("Loading LoRA adapter...")
        model = PeftModel.from_pretrained(model, model_state["adapter_dir"])
    model.eval()

    for combo in RETRIEVAL_COMBOS:
        config_label = f"{model_state['label']}_{combo['label']}"
        print(f"\n{'=' * 70}")
        print(f"[Config] {config_label}  (RAG={combo['rag']}, KG={combo['kg']})")
        print(f"{'=' * 70}")

        config_scores = []

        for scenario in scenarios:
            for trial_num in range(1, N_TRIALS + 1):
                key = (config_label, scenario["id"], trial_num)
                if key in completed:
                    continue  # already done in a prior interrupted run

                rag_context, rag_time, rag_failed = ("", 0.0, False)
                kg_context, kg_time, kg_failed = ("", 0.0, False)

                if combo["rag"]:
                    rag_context, rag_time, rag_failed = retrieve_rag_context(scenario)
                if combo["kg"]:
                    kg_context, kg_time, kg_failed = retrieve_kg_context(scenario)

                retrieval_failed = (combo["rag"] and rag_failed) or (combo["kg"] and kg_failed)

                retrieved_parts = [p for p in [rag_context, kg_context] if p]
                combined_context = f"{scenario['scenario_input']}\n\n{chr(10).join(retrieved_parts)}".strip()

                prompt_text = PROMPT_TEMPLATE.format(
                    context=combined_context,
                    instruction=scenario["instruction"],
                )
                messages = [
                    {"role": "system", "content": "You are a NERC CIP compliance audit assistant."},
                    {"role": "user", "content": prompt_text},
                ]
                prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

                trial_start_ts = time.time()
                gen_start = time.time()
                with ResourceMonitor(
                    job_name=f"{config_label}_{scenario['id']}_t{trial_num}",
                    sample_interval=INFERENCE_SAMPLE_INTERVAL_SECONDS,
                    write_prom_file=False,  # no Prometheus server; capture summary into CSV row instead
                    verbose=False,          # avoid a print-summary block per trial x 4800
                ) as res_mon:
                    with torch.no_grad():
                        output_ids = model.generate(
                            **inputs,
                            max_new_tokens=300,
                            do_sample=False,
                            temperature=None,
                            top_p=None,
                            pad_token_id=tokenizer.eos_token_id,
                        )
                gen_elapsed = time.time() - gen_start
                trial_end_ts = time.time()

                generated_text = tokenizer.decode(
                    output_ids[0][inputs["input_ids"].shape[1]:],
                    skip_special_tokens=True,
                )

                score = score_response(scenario, generated_text)

                if not retrieval_failed:
                    config_scores.append(score)

                status_icon = "PASS" if score.full_pass else "FAIL"
                fail_tag = " [RETRIEVAL_FAILED]" if retrieval_failed else ""
                print(f"[{config_label}] {scenario['id']} trial {trial_num}/{N_TRIALS}: "
                      f"citation={'Y' if score.citation_correct else 'N'} "
                      f"status={'Y' if score.status_correct else 'N'} "
                      f"facts={score.key_fact_recall:.2f} -> {status_icon} "
                      f"({gen_elapsed:.1f}s){fail_tag}")

                row = {
                    "config_label": config_label,
                    "model_state": model_state["label"],
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
                }
                csv_writer.writerow(row)
                csv_file.flush()
                os.fsync(csv_file.fileno())

                completed.add(key)
                save_checkpoint(completed)

        all_scores_by_config[config_label] = config_scores

        if config_scores:
            summary = aggregate_scores(config_scores)
            print(f"\n{'-' * 70}")
            print(f"SUBTOTAL -- {config_label}  "
                  f"({len(config_scores)}/{len(scenarios) * N_TRIALS} trials counted; "
                  f"excluded trials had failed retrieval)")
            print(f"{'-' * 70}")
            for k, v in summary.items():
                print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")
        else:
            print(f"[Warning] No successful trials for {config_label} -- all retrieval failed?")

    # Free GPU memory before loading the next model state.
    del model
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fresh", action="store_true",
                         help="Ignore any existing checkpoint/CSV and start over.")
    parser.add_argument("--sizes", nargs="+", choices=[s["size_label"] for s in MODEL_SIZES],
                         default=[s["size_label"] for s in MODEL_SIZES],
                         help="Restrict this run to specific model sizes, e.g. "
                              "'--sizes phi4mini' to run only the mini configs (8 configs) "
                              "and defer phi4_14b to a later invocation. Checkpointing means "
                              "a later run with a different --sizes selection adds to the "
                              "same results file rather than conflicting.")
    args = parser.parse_args()

    global MODEL_STATES
    MODEL_STATES = [s for s in MODEL_STATES
                     if any(s["label"].startswith(sz + "_") for sz in args.sizes)]

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
    print(f"[Plan] Model states: {[s['label'] for s in MODEL_STATES]}")
    print(f"[Plan] {len(scenarios)} scenarios x {N_TRIALS} trials x "
          f"{len(RETRIEVAL_COMBOS)} retrieval combos x {len(MODEL_STATES)} model states = "
          f"{len(scenarios) * N_TRIALS * len(RETRIEVAL_COMBOS) * len(MODEL_STATES)} total generations planned.")
    print(f"[Plan] MAX_CITATIONS={MAX_CITATIONS} (matched across RAG top_k and KG max_citations)")

    fieldnames = [
        "config_label", "model_state", "rag_enabled", "kg_enabled",
        "scenario_id", "trial_num", "standard_family", "retrieval_failed",
        "citation_correct", "status_correct", "status_found", "key_fact_recall",
        "full_pass", "extracted_citation", "extracted_status",
        "expected_citation", "expected_status",
        "rag_context", "rag_retrieval_time_sec", "kg_context", "kg_retrieval_time_sec",
        "model_output", "generation_time_sec",
        "trial_start_ts", "trial_end_ts",
        "gen_cpu_percent_mean", "gen_cpu_percent_peak",
        "gen_ram_used_mb_mean", "gen_ram_used_mb_peak",
        "gen_gpu_percent_mean", "gen_gpu_percent_peak", "gen_resource_n_samples",
        "gen_power_total_mw_mean", "gen_power_total_mw_peak",
        "gen_power_cpu_mw_mean", "gen_power_gpu_mw_mean",
        "gen_temp_cpu_c_mean", "gen_temp_cpu_c_peak",
        "gen_temp_gpu_c_mean", "gen_temp_gpu_c_peak",
        "gen_energy_wh_estimate",
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

            for model_state in MODEL_STATES:
                run_model_state(model_state, scenarios, completed, csv_writer, csv_file, all_scores_by_config)
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
