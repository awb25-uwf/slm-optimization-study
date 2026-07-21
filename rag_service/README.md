# RAG Service

FastAPI service that answers NERC CIP questions using retrieval-augmented
generation: embeds a query with `sentence-transformers`, retrieves nearby
chunks from a Chroma vector store, and returns the matched context.

## Files

| File | Purpose |
|---|---|
| `rag_api.py` | FastAPI app — the running service (port 8001) |
| `build_vector_db.py` | One-off/rebuildable script that embeds `citation_lookup.jsonl` into the Chroma vector store |
| `resource_monitor.py` | Shared CPU/RAM/GPU metrics helper, used by both the batch build script and the live API's `/metrics` endpoint |

## Runtime dependencies (not Python packages)

- **ChromaDB server** — runs as its own container (`chromadb/chroma:latest` in `docker-compose.yml`); `rag_api.py` and `build_vector_db.py` connect to it as a client, they don't embed it.
- **`citation_lookup.jsonl`** — the source data `build_vector_db.py` indexes. Not included in this repo; mount it or place it alongside these scripts before running the build step.

## Install & run — Docker (recommended, matches production setup)

```bash
docker compose up -d --build
```

This starts three containers:
* `chromadb` (vector store, port 8000)
* `rag-api` (this service, port 8001)
* `cadvisor` (container metrics, port 8080)

The Dockerfile installs a CPU-only build of `torch` before the rest of
`requirements.txt`, since this service only runs a small embedding model on
CPU — GPU acceleration isn't needed here.

Once the containers are up, build/refresh the vector index:

```bash
docker compose exec rag-api python build_vector_db.py
```

## Install & run — local Python

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

# Requires a Chroma server reachable at CHROMA_HOST:CHROMA_PORT
# (defaults used in docker-compose.yml: localhost:8000)
export CHROMA_HOST=localhost
export CHROMA_PORT=8000

python build_vector_db.py         # build the index (run once, or after data changes)
uvicorn rag_api:app --host 0.0.0.0 --port 8001
```

## Metrics

`resource_monitor.py` exposes Prometheus gauges and writes a
textfile-collector-compatible `.prom` file for one-off jobs. Point
`node_exporter`'s `--collector.textfile.directory` at `./prom_textfile`
(created automatically) to pick these up, or scrape `/metrics` on the live
API directly.
