#!/usr/bin/env python3
"""
Standalone RAG retrieval service. Called over HTTP by the Jetson's
inference API.

Run standalone with:
    uvicorn rag_api:app --host 0.0.0.0 --port 8001

Endpoints:
    POST /retrieve  -- {"instruction": "...", "scenario_input": "..."} -> {"context": "..."}
    GET  /metrics    -- Prometheus scrape target
    GET  /health
"""
import os
import time
import chromadb
from sentence_transformers import SentenceTransformer
from fastapi import FastAPI
from fastapi.responses import Response
from pydantic import BaseModel
from prometheus_client import Counter, Histogram, CONTENT_TYPE_LATEST, generate_latest

from resource_monitor import start_background_sampler

CHROMA_HOST = os.environ.get("CHROMA_HOST", "host.docker.internal")
CHROMA_PORT = int(os.environ.get("CHROMA_PORT", 8000))
COLLECTION_NAME = "nerc_cip_citations"
EMBEDDING_MODEL = "all-mpnet-base-v2"
TOP_K = 1

REQUEST_COUNT = Counter("rag_requests_total", "Total RAG retrieval requests", ["status"])
REQUEST_LATENCY = Histogram("rag_request_latency_seconds", "RAG retrieval request latency")

app = FastAPI(title="NERC CIP RAG Retrieval Service")

_client = None
_collection = None
_embedder = None


class RetrieveRequest(BaseModel):
    instruction: str
    scenario_input: str
    top_k: int = TOP_K


class RetrieveResponse(BaseModel):
    context: str
    n_results: int
    retrieval_time_sec: float


@app.on_event("startup")
def load_index():
    global _client, _collection, _embedder
    print("Loading embedding model...")
    _embedder = SentenceTransformer(EMBEDDING_MODEL)
    print(f"Connecting to Chroma server at {CHROMA_HOST}:{CHROMA_PORT}...")
    _client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
    _collection = _client.get_collection(COLLECTION_NAME)
    print(f"RAG service ready. Collection has {_collection.count()} entries.")

    start_background_sampler(interval_seconds=2.0)
    print("Resource sampler started.")


@app.post("/retrieve", response_model=RetrieveResponse)
def retrieve(request: RetrieveRequest):
    start = time.time()
    try:
        query_text = f"{request.instruction} {request.scenario_input}".strip()
        query_embedding = _embedder.encode([query_text]).tolist()

        results = _collection.query(query_embeddings=query_embedding, n_results=request.top_k)

        if not results["documents"] or not results["documents"][0]:
            context = ""
            n_results = 0
        else:
            chunks = []
            for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
                citation = meta.get("citation", "unknown")
                chunks.append(f"[Retrieved -- {citation}]: {doc}")
            context = "\n\n".join(chunks)
            n_results = len(chunks)

        elapsed = time.time() - start
        REQUEST_COUNT.labels(status="success").inc()
        REQUEST_LATENCY.observe(elapsed)

        return RetrieveResponse(context=context, n_results=n_results, retrieval_time_sec=round(elapsed, 3))

    except Exception:
        REQUEST_COUNT.labels(status="error").inc()
        raise


@app.get("/metrics")
def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/health")
def health():
    ready = _collection is not None
    return {"status": "ok" if ready else "not_ready", "collection_count": _collection.count() if ready else 0}
