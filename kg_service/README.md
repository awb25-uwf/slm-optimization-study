# Knowledge Graph Service

FastAPI service that answers NERC CIP questions using a Neo4j knowledge
graph built from the standards' citations and relationships.

## Files

| File | Purpose |
|---|---|
| `kg_api.py` | FastAPI app — the running service (port 8002) |
| `build_knowledge_graph.py` | One-off/rebuildable script that parses source text into `knowledge_graph.json` |
| `load_graph.py` | Loads `knowledge_graph.json` into the running Neo4j instance (bolt protocol) |
| `resource_monitor.py` | Shared CPU/RAM/GPU metrics helper, used by both batch scripts and the live API's `/metrics` endpoint |

## Runtime dependencies (not Python packages)

- **Neo4j server** — runs as its own container (`neo4j:5-community` in `docker-compose.yml`). Default credentials in that file are `neo4j` / `nerc_cip_kg_pass` — must match the `NEO4J_PASSWORD` used in `load_graph.py` / `kg_api.py`.
- **`citation_lookup.jsonl`** — source data for `build_knowledge_graph.py`. Not included in this repo.

## Install & run — Docker (recommended, matches production setup)

```bash
docker compose up -d --build
```

Starts `neo4j` (browser UI at http://localhost:7474, bolt at 7687) and
`kg-api` (this service, port 8002).

Then build and load the graph:

```bash
docker compose exec kg-api python build_knowledge_graph.py
docker compose exec kg-api python load_graph.py
```

## Install & run — local Python

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Requires a Neo4j server reachable at the bolt URL configured in
# load_graph.py / kg_api.py (defaults to the docker-compose values above)
python build_knowledge_graph.py
python load_graph.py
uvicorn kg_api:app --host 0.0.0.0 --port 8002
```

## Metrics

Same pattern as `rag_service`: `resource_monitor.py` exposes Prometheus
gauges and writes to `./prom_textfile` for one-off jobs; the live API
also serves `/metrics` directly.
