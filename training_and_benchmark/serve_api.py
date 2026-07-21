#!/usr/bin/env python3
"""
API server: hosts the fine-tuned model behind an HTTP endpoint, with live
Prometheus metrics at /metrics for scraping.

Run with:
    uvicorn serve_api:app --host 0.0.0.0 --port 8000

Prometheus scrape config (prometheus.yml):
    scrape_configs:
      - job_name: 'nerc_cip_api'
        static_configs:
          - targets: ['<jetson-ip>:8000']
"""
import time
import torch
import requests
from fastapi import FastAPI
from fastapi.responses import Response
from pydantic import BaseModel
from prometheus_client import Counter, Histogram, CONTENT_TYPE_LATEST, generate_latest
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from resource_monitor import start_background_sampler

# --- CONFIGURATION ---
BASE_MODEL_NAME = "microsoft/Phi-4-mini-instruct"
ADAPTER_DIR = "/mnt/ollama_repo/phi4_nerc_cip_lora"
USE_RAG_DEFAULT = False  # per-request override available via the request body

RAG_SERVICE_URL = "http://192.168.1.100:8001"  # <-- set to your laptop's actual LAN IP
RAG_TIMEOUT_SECONDS = 10

# --- Prometheus metrics ---
REQUEST_COUNT = Counter("api_requests_total", "Total API requests", ["endpoint", "status"])
REQUEST_LATENCY = Histogram("api_request_latency_seconds", "Request latency", ["endpoint"])
TOKENS_GENERATED = Counter("api_tokens_generated_total", "Total tokens generated")
RAG_RETRIEVAL_LATENCY = Histogram("rag_retrieval_latency_seconds", "RAG retrieval latency")

app = FastAPI(title="NERC CIP Compliance Audit API")

model = None
tokenizer = None


class AuditRequest(BaseModel):
    scenario_input: str
    instruction: str
    use_rag: bool = USE_RAG_DEFAULT


class AuditResponse(BaseModel):
    output: str
    generation_time_sec: float
    used_rag: bool


@app.on_event("startup")
def load_model():
    global model, tokenizer
    print("Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_DIR, trust_remote_code=True)
    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(model, ADAPTER_DIR)
    model.eval()
    print("Model ready.")

    start_background_sampler(interval_seconds=2.0)
    print("Resource sampler started.")


def get_rag_context(scenario: dict) -> str:
    """Calls the RAG service running as a separate container (e.g. on your
    laptop) over HTTP. This is what makes RAG genuinely independent per your
    mix-and-match design -- the Jetson has zero RAG-specific dependencies
    (no chromadb, no sentence-transformers), just an HTTP client. If the RAG
    service is down or slow, this fails safe by returning empty context
    rather than crashing the whole request."""
    with RAG_RETRIEVAL_LATENCY.time():
        try:
            response = requests.post(
                f"{RAG_SERVICE_URL}/retrieve",
                json={"instruction": scenario.get("instruction", ""),
                      "scenario_input": scenario.get("scenario_input", "")},
                timeout=RAG_TIMEOUT_SECONDS,
            )
            response.raise_for_status()
            return response.json().get("context", "")
        except requests.exceptions.RequestException as e:
            print(f"[Warning] RAG service call failed ({e}) -- proceeding without RAG context.")
            return ""


PROMPT_TEMPLATE = """Context:
{context}

Instruction:
{instruction}

Respond with your compliance determination and reasoning. End your response with exactly these two lines:
Citation: <the specific standard and requirement/part/section you are citing>
Compliance Status: <compliant|partial|non_compliant>"""


@app.post("/audit", response_model=AuditResponse)
def audit(request: AuditRequest):
    endpoint = "/audit"
    start = time.time()
    try:
        scenario = {"instruction": request.instruction, "scenario_input": request.scenario_input}

        rag_context = get_rag_context(scenario) if request.use_rag else ""
        combined_context = f"{request.scenario_input}\n\n{rag_context}".strip()

        prompt_text = PROMPT_TEMPLATE.format(context=combined_context, instruction=request.instruction)
        messages = [
            {"role": "system", "content": "You are a NERC CIP compliance audit assistant."},
            {"role": "user", "content": prompt_text},
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=300,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=tokenizer.eos_token_id,
            )

        n_new_tokens = output_ids.shape[1] - inputs["input_ids"].shape[1]
        generated_text = tokenizer.decode(
            output_ids[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )

        elapsed = time.time() - start
        TOKENS_GENERATED.inc(n_new_tokens)
        REQUEST_COUNT.labels(endpoint=endpoint, status="success").inc()
        REQUEST_LATENCY.labels(endpoint=endpoint).observe(elapsed)

        return AuditResponse(output=generated_text, generation_time_sec=round(elapsed, 2), used_rag=request.use_rag)

    except Exception as e:
        REQUEST_COUNT.labels(endpoint=endpoint, status="error").inc()
        raise


@app.get("/metrics")
def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}
