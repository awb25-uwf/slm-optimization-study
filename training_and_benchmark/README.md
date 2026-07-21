# Training & Benchmark Pipeline

Scripts for building a NERC CIP compliance fine-tuning dataset, training
LoRA adapters on Phi-4 (mini and 14B), converting/serving the result, and
running a fault-tolerant benchmark across model size / LoRA / RAG / KG
combinations. Developed and run on a Jetson/Ubuntu machine (uses
`jetson-stats` for GPU telemetry).

## Runtime dependencies (not Python packages)

- **[Ollama](https://ollama.com/)**, running locally with `deepseek-r1:32b` pulled — used by the dataset-generation scripts and as the inference backend for `run_benchmark.py` (`http://localhost:11434`).
- **[llama.cpp](https://github.com/ggml-org/llama.cpp)**, cloned locally with `convert_hf_to_gguf.py` — used by `convert_to_gguf.py`. Update `CONVERT_SCRIPT_PATH` in that script to your clone's path.
- Source NERC CIP standards as PDFs (for `extract_and_chunk.py`) — not included in this repo.

## Pipeline stages

**1. Data extraction & dataset build**

    1. extract_and_chunk.py
    2. build_citation_lookup.py
    3. generate_lora_data.py
    4. augment_dataset.py/boost_underrepresented.py (targeted expansion)
    5. balance_dataset.py
    6. decontaminate_dataset.py
    7. disambiguate_citations.py
    
  Helper/QA scripts:
  * `check_citation_collisions.py`
  * `clean_version_history.py`

**2. Training**
`train_phi4mini_lora.py` (mini) and `train_phi4_14b_lora.py` (14B) — LoRA
fine-tuning via `trl.SFTTrainer`. `check_prompt_tokens.py` estimates a safe
context window beforehand.

**3. Post-training**

    1. merge_lora_adapters.py (merge adapter into base weights)
    2. convert_to_gguf.py (convert to GGUF for Ollama)
    3. bias_regression_test_mini.py / bias_regression_test_14b.py

**4. Benchmarking**
* `run_benchmark.py` / `run_full_benchmark.py` — checkpointed runs across
model size × base/LoRA × RAG on/off × KG on/off.
* `benchmark_scorer.py`
mechanically scores the output against ground truth.

**5. Serving**
`serve_api.py` — hosts the fine-tuned model behind a FastAPI endpoint with
Prometheus metrics. `resource_monitor.py` (shared with the other two
services) provides CPU/RAM/GPU telemetry.

## Install & run

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Then run whichever stage you need, e.g.:

```bash
python extract_and_chunk.py
python build_citation_lookup.py
python generate_lora_data.py
...
python train_phi4_lora.py
...
uvicorn serve_api:app --host 0.0.0.0 --port 8000
```

Most scripts read/write to hardcoded filenames in the working directory
(e.g. `nerc_cip_phi4_dataset.jsonl`) — check the top of each script for its
`INPUT_FILE`/`OUTPUT_FILE` constants before running.

## Dependency Notes

- **`fastapi`, `uvicorn`, `python-docx`, `pypdf`** are imported by
  `serve_api.py` and `extract_and_chunk.py` respectively, but weren't
  present in the `pip freeze` you provided for this machine. They're
  included in `requirements.txt` — install and confirm versions, then pin
  them once known-good.
- **`jetson-stats`** is used only as an optional, try/except-guarded import
  in `resource_monitor.py` for Jetson-specific GPU telemetry; the script
  falls back to plain `psutil` if it's missing. Commented out in
  `requirements.txt` — uncomment if deploying on Jetson hardware.
