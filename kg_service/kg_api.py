#!/usr/bin/env python3
"""
KG resolution service. This API extracts structured facts from the scenario text
and resolves the applicable citation via graph traversal in Neo4j.

Run with:
    uvicorn kg_api:app --host 0.0.0.0 --port 8002

Endpoints:
    POST /resolve  -- {"instruction": "...", "scenario_input": "..."} -> {"context": "...", "resolved_citations": [...]}
    GET  /metrics
    GET  /health
"""
import re
import json
import time
from typing import Optional
from neo4j import GraphDatabase
from fastapi import FastAPI
from fastapi.responses import Response
from pydantic import BaseModel
from prometheus_client import Counter, Histogram, CONTENT_TYPE_LATEST, generate_latest

from resource_monitor import start_background_sampler

NEO4J_URI = "bolt://neo4j:7687"  # Docker Compose service name -- resolves within the compose network
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "nerc_cip_kg_pass"
CITATION_LOOKUP_FILE = "citation_lookup.jsonl"

ENTITY_ROLES = {
    "Balancing Authority": r"\bBalancing Authorit(?:y|ies)\b|\bBA\b",
    "Distribution Provider": r"\bDistribution Provider(?:s)?\b|\bDP\b",
    "Generator Owner": r"\bGenerator Owner(?:s)?\b|\bGO\b",
    "Generator Operator": r"\bGenerator Operator(?:s)?\b|\bGOP\b",
    "Reliability Coordinator": r"\bReliability Coordinator(?:s)?\b|\bRC\b",
    "Reserve Sharing Group": r"\bReserve Sharing Group(?:s)?\b|\bRSG\b",
    "Transmission Operator": r"\bTransmission Operator(?:s)?\b|\bTOP\b",
    "Transmission Owner": r"\bTransmission Owner(?:s)?\b|\bTO\b",
    "Transmission Planner": r"\bTransmission Planner(?:s)?\b|\bTP\b",
    "Transmission Service Provider": r"\bTransmission Service Provider(?:s)?\b|\bTSP\b",
}

IMPACT_LEVELS = {
    "High Impact": r"\bhigh[\s-]?impact\b",
    "Medium Impact": r"\bmedium[\s-]?impact\b",
    "Low Impact": r"\blow[\s-]?impact\b",
}

# Must match build_knowledge_graph.py's CONCEPTS dict
CONCEPTS = {
    "BES Cyber System": r"\bBES Cyber System(?:s)?\b|\bBCS\b",
    "BES Cyber Asset": r"\bBES Cyber Asset(?:s)?\b|\bBCA\b",
    "Electronic Access Control or Monitoring System": r"\bEACMS\b|\bElectronic Access Control",
    "Physical Access Control System": r"\bPACS\b|\bPhysical Access Control System",
    "Protected Cyber Asset": r"\bPCA\b|\bProtected Cyber Asset",
    "Control Center": r"\bControl Center(?:s)?\b",
    "Cyber Security Incident": r"\bCyber Security Incident(?:s)?\b",
    "Electronic Access Point": r"\bEAP\b|\bElectronic Access Point",
    "Physical Security Perimeter": r"\bPhysical Security Perimeter(?:s)?\b|\bPSP\b",
    "Transmission Station": r"\bTransmission station(?:s)?\b|\bTransmission substation(?:s)?\b",
    "Encryption": r"\bencrypt(?:ed|ion)?\b|\bHTTPS\b|\bSSH\b",
    "Personnel Risk Assessment": r"\bpersonnel risk assessment\b|\bcriminal history\b",
    "Access Revocation": r"\brevoke\b|\brevocation\b|\btermination\b",
    "Vendor Remote Access": r"\bvendor\s+(?:remote\s+)?access\b",
    "Removable Media": r"\bRemovable Media\b",
    "Transient Cyber Asset": r"\bTransient Cyber Asset(?:s)?\b|\bTCA\b",
    "Supply Chain Risk Management": r"\bsupply chain\b",
    "Internal Network Security Monitoring": r"\binternal network security monitoring\b",
    "Recovery Plan": r"\brecovery plan(?:s)?\b",
    "Visitor Control Program": r"\bvisitor control program\b|\bvisitor(?:s)?\b",
}

REQUEST_COUNT = Counter("kg_requests_total", "Total KG resolution requests", ["status"])
REQUEST_LATENCY = Histogram("kg_request_latency_seconds", "KG resolution request latency")

app = FastAPI(title="NERC CIP Knowledge Graph Resolution Service")

_driver = None
_citation_lookup = {}


class ResolveRequest(BaseModel):
    instruction: str
    scenario_input: str
    max_citations: int = 3


class ResolveResponse(BaseModel):
    context: str
    resolved_citations: list
    detected_entity_role: Optional[str] = None
    detected_impact_level: Optional[str] = None
    detected_concepts: list = []
    resolution_time_sec: float


@app.on_event("startup")
def startup():
    global _driver, _citation_lookup
    print("Connecting to Neo4j...")
    _driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    print("Loading citation_lookup.jsonl for grounding text...")
    with open(CITATION_LOOKUP_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entry = json.loads(line)
                _citation_lookup[entry["citation"].lower().strip()] = entry
    print(f"Loaded {len(_citation_lookup)} citation contexts.")

    start_background_sampler(interval_seconds=2.0)
    print("Resource sampler started.")


def detect_entity_role(text: str):
    for role_name, pattern in ENTITY_ROLES.items():
        if re.search(pattern, text, re.IGNORECASE):
            return role_name
    return None


def detect_impact_level(text: str):
    for impact_name, pattern in IMPACT_LEVELS.items():
        if re.search(pattern, text, re.IGNORECASE):
            return impact_name
    return None


def detect_concepts(text: str) -> list:
    """Unlike entity role/impact level (mutually exclusive, first-match-wins),
    a scenario can genuinely mention multiple concepts -- return all matches."""
    return [name for name, pattern in CONCEPTS.items() if re.search(pattern, text, re.IGNORECASE)]


def resolve_citations(entity_role: str, impact_level: str, concepts: list, max_citations: int):
    """Scores citations using SPECIFICITY-WEIGHTED matching: each matched
    signal (entity role, impact level, concept) contributes 1/citation_count
    to the score, where citation_count is how many citations that node
    connects to overall (precomputed in load_graph.py). This is the same
    principle as IDF weighting in text retrieval -- a signal connected to few
    citations is highly discriminating and should dominate the ranking; a
    signal connected to many citations (a common entity role, or an
    unusually well-connected 'hub' citation) is nearly uninformative and
    should barely move the score. This directly addresses the failure mode
    where a generic match like 'Transmission Owner' or a high-connectivity
    citation like 'CIP-004-8, R2' won by raw match count regardless of
    actual topical relevance."""
    if not entity_role and not impact_level and not concepts:
        return []

    query = """
        MATCH (c:Citation)
        OPTIONAL MATCH (c)-[:APPLIES_TO]->(r:EntityRole {label: $role})
        OPTIONAL MATCH (c)-[:DETERMINES]->(i:ImpactLevel {label: $impact})
        OPTIONAL MATCH (c)-[:MENTIONS]->(concept:Concept)
        WHERE concept.label IN $concepts
        WITH c,
             max(CASE WHEN r IS NOT NULL THEN 1.0 / r.citation_count ELSE 0.0 END) AS role_score,
             max(CASE WHEN i IS NOT NULL THEN 1.0 / i.citation_count ELSE 0.0 END) AS impact_score,
             sum(CASE WHEN concept IS NOT NULL THEN 1.0 / concept.citation_count ELSE 0.0 END) AS concept_score
        WITH c, role_score + impact_score + concept_score AS score
        WHERE score > 0
        RETURN c.label AS citation, score
        ORDER BY score DESC
        LIMIT $limit
    """
    with _driver.session() as session:
        result = session.run(
            query,
            role=entity_role,
            impact=impact_level,
            concepts=concepts,
            limit=max_citations,
        )
        return [record["citation"] for record in result]


@app.post("/resolve", response_model=ResolveResponse)
def resolve(request: ResolveRequest):
    start = time.time()
    try:
        combined_text = f"{request.instruction} {request.scenario_input}"
        entity_role = detect_entity_role(combined_text)
        impact_level = detect_impact_level(combined_text)
        concepts = detect_concepts(combined_text)

        citations = resolve_citations(entity_role, impact_level, concepts, request.max_citations)

        chunks = []
        for citation in citations:
            entry = _citation_lookup.get(citation.lower().strip())
            if entry:
                chunks.append(f"[KG-resolved -- {citation}]: {entry['context']}")
            else:
                chunks.append(f"[KG-resolved -- {citation}]: (no grounding text found for this citation)")

        context = "\n\n".join(chunks)
        elapsed = time.time() - start

        REQUEST_COUNT.labels(status="success").inc()
        REQUEST_LATENCY.observe(elapsed)

        return ResolveResponse(
            context=context,
            resolved_citations=citations,
            detected_entity_role=entity_role,
            detected_impact_level=impact_level,
            detected_concepts=concepts,
            resolution_time_sec=round(elapsed, 3),
        )
    except Exception:
        REQUEST_COUNT.labels(status="error").inc()
        raise


@app.get("/metrics")
def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/health")
def health():
    ready = _driver is not None
    return {"status": "ok" if ready else "not_ready", "citations_loaded": len(_citation_lookup)}
