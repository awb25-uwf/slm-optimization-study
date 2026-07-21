#!/usr/bin/env python3
"""
Builds the NERC CIP knowledge graph structure from source docs.

Node types: Standard, Citation, EntityRole, ImpactLevel
Edge types: CONTAINS (Standard->Citation), APPLIES_TO (Citation->EntityRole),
            DETERMINES (Citation->ImpactLevel)

Outputs knowledge_graph.json: {"nodes": [...], "edges": [...]}
"""
import os
import re
import json
import glob

INPUT_DIR = "extracted_txt"
OUTPUT_FILE = "knowledge_graph.json"
CONTEXT_WINDOW = 600
RELATIONSHIP_WINDOW = 150

CITATION_PATTERN = re.compile(
    r"(Attachment\s+\d+,?\s+Section\s+[\d.]+|Section\s+[\d.]+|"
    r"R\d+(?:\.\d+)*\s+Part\s+[\d.]+|Part\s+[\d.]+|R\d+(?:\.\d+)*|M\d+(?:\.\d+)*|Requirement\s+\d+(?:\.\d+)*)",
    re.IGNORECASE
)
STANDARD_PATTERN = re.compile(r"(CIP-\d{3}-\d+(?:\.\d+)?)", re.IGNORECASE)
PARTIAL_STANDARD_PATTERN = re.compile(r"\bCIP-\d{3}\b", re.IGNORECASE)

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

# Keywords and domain-specific topics
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


def disambiguate_citation(citation: str, source_file: str) -> str:
    citation = citation.strip()
    if STANDARD_PATTERN.search(citation) or PARTIAL_STANDARD_PATTERN.search(citation):
        return citation
    match = STANDARD_PATTERN.search(source_file)
    if not match:
        return citation
    return f"{match.group(1).upper()}, {citation}"


def node_id_for(prefix: str, name: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", name.strip().lower()).strip("_")
    return f"{prefix}:{slug}"


class KnowledgeGraph:
    def __init__(self):
        self.nodes = {}
        self.edges = []
        self._edge_seen = set()

    def add_node(self, node_id, node_type, label):
        if node_id not in self.nodes:
            self.nodes[node_id] = {"id": node_id, "type": node_type, "label": label}
        return node_id

    def add_edge(self, source, target, edge_type):
        key = (source, target, edge_type)
        if key not in self._edge_seen:
            self._edge_seen.add(key)
            self.edges.append({"source": source, "target": target, "type": edge_type})


def build():
    kg = KnowledgeGraph()
    all_files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.txt")))

    for file_path in all_files:
        filename = os.path.basename(file_path)
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        standard_match = STANDARD_PATTERN.search(filename)
        standard_name = standard_match.group(1).upper() if standard_match else filename
        standard_id = kg.add_node(node_id_for("standard", standard_name), "Standard", standard_name)

        citation_count = 0
        for match in CITATION_PATTERN.finditer(content):
            raw_citation = match.group(0).strip()
            citation_text = disambiguate_citation(raw_citation, filename)
            citation_id = kg.add_node(node_id_for("citation", citation_text), "Citation", citation_text)
            kg.add_edge(standard_id, citation_id, "CONTAINS")
            citation_count += 1

            start = max(0, match.start() - RELATIONSHIP_WINDOW // 2)
            end = min(len(content), match.end() + RELATIONSHIP_WINDOW // 2)
            window = content[start:end]

            for role_name, pattern in ENTITY_ROLES.items():
                if re.search(pattern, window, re.IGNORECASE):
                    role_id = kg.add_node(node_id_for("role", role_name), "EntityRole", role_name)
                    kg.add_edge(citation_id, role_id, "APPLIES_TO")

            for impact_name, pattern in IMPACT_LEVELS.items():
                if re.search(pattern, window, re.IGNORECASE):
                    impact_id = kg.add_node(node_id_for("impact", impact_name), "ImpactLevel", impact_name)
                    kg.add_edge(citation_id, impact_id, "DETERMINES")

            for concept_name, pattern in CONCEPTS.items():
                if re.search(pattern, window, re.IGNORECASE):
                    concept_id = kg.add_node(node_id_for("concept", concept_name), "Concept", concept_name)
                    kg.add_edge(citation_id, concept_id, "MENTIONS")

        print(f"[{filename}] -> {standard_name}, {citation_count} citation mention(s)")

    graph = {"nodes": list(kg.nodes.values()), "edges": kg.edges}
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(graph, f, indent=2)

    from collections import Counter
    edge_type_counts = Counter(e["type"] for e in kg.edges)
    print(f"\n[Done] {len(kg.nodes)} nodes, {len(kg.edges)} edges written to {OUTPUT_FILE}")
    print(f"[Done] Edge breakdown: {dict(edge_type_counts)}")


if __name__ == "__main__":
    build()
