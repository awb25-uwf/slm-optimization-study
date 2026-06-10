# SLM Optimization Study

An automated edge-computing simulation program designed to benchmark small language models (SLMs) on localized hardware. This repository coordinates an orchestration layer, vector database, and telemetry pipeline across an isolated Docker network on a host PC, communicating directly with an external NVIDIA Jetson AGX Orin edge device.

## Requirements
- Ubuntu 26.04
- [Docker Engine](https://docs.docker.com/engine/install/) (Linux) or [Docker Desktop](https://docs.docker.com/desktop/install/windows/) (Windows with WSL)
- Python Modules:
  - Requests
  - ChromaDB
  - Prometheus Client

## Execution
After cloning the repository and preparing the host, run `docker compose up` in the project root to instantiate the runtime environment. Running `python app/main.py` will perform the trial runs.
