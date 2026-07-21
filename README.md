# NERC CIP Compliance LLM

Three components for a NERC CIP compliance assistant: two retrieval
services (vector + graph) and a training/benchmarking pipeline for a
fine-tuned Phi-4 model that uses them.

## Components

| Directory | Functionality | Platform |
|---|---|---|
| [`rag_service/`](rag_service/) | FastAPI + ChromaDB retrieval service | laptop |
| [`kg_service/`](kg_service/) | FastAPI + Neo4j knowledge-graph service | laptop |
| [`training_and_benchmark/`](training_and_benchmark/) | Dataset build, LoRA fine-tuning, GGUF conversion, benchmarking, and model serving | Jetson/Ubuntu |

Each directory has its own `README.md` with install and run instructions,
and its own `requirements.txt`. Set up a **separate virtual environment per
directory** — the training stack and the two API services have very
different (and non-overlapping) dependencies, and mixing them into one
venv is unnecessary.

## Shared piece

`resource_monitor.py` (CPU/RAM/GPU metrics, Prometheus-compatible) is
duplicated across all three directories rather than shared as a package,
matching how it's deployed today. If you want a single source of truth,
it's a good candidate to pull into a small shared package later.

## Data

Source documents, generated datasets, the vector index, and the knowledge
graph JSON are not included in this repo (see `.gitignore`). Each
component's README explains how to regenerate them from the corresponding
build script.

## Getting started

Pick the component you need and follow its README:

- [`rag_service/README.md`](rag_service/README.md)
- [`kg_service/README.md`](kg_service/README.md)
- [`training_and_benchmark/README.md`](training_and_benchmark/README.md)
