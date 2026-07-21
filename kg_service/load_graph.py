#!/usr/bin/env python3
"""
Merges knowledge_graph.json into a the KG API container's Neo4j instance. Resource
consumption during the load is logged the same way as the RAG service's vector DB
build.
"""
import json
from neo4j import GraphDatabase

from resource_monitor import ResourceMonitor

GRAPH_FILE = "knowledge_graph.json"
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "nerc_cip_kg_pass"  # must match NEO4J_AUTH in docker-compose.yml
PROM_TEXTFILE_DIR = "./prom_textfile"


def load():
    with open(GRAPH_FILE, "r", encoding="utf-8") as f:
        graph = json.load(f)

    nodes = graph["nodes"]
    edges = graph["edges"]
    print(f"[Plan] {len(nodes)} nodes, {len(edges)} edges to load.")

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    with ResourceMonitor("kg_neo4j_load", output_dir=PROM_TEXTFILE_DIR, sample_interval=1.0):
        with driver.session() as session:
            print("Clearing existing graph...")
            session.run("MATCH (n) DETACH DELETE n")

            print("Creating index on node id...")
            session.run("CREATE INDEX node_id_index IF NOT EXISTS FOR (n:Node) ON (n.id)")

            print("Loading nodes...")
            for node in nodes:
                session.run(
                    f"MERGE (n:Node:{node['type']} {{id: $id}}) SET n.label = $label",
                    id=node["id"], label=node["label"]
                )

            print("Loading edges...")
            for edge in edges:
                session.run(
                    f"MATCH (a:Node {{id: $source}}), (b:Node {{id: $target}}) "
                    f"MERGE (a)-[:{edge['type']}]->(b)",
                    source=edge["source"], target=edge["target"]
                )

            print("Computing node specificity (citation_count) for hub-bias-resistant scoring...")
            # Inverse document frequency (IDF) weighting
            session.run("""
                MATCH (c:Citation)-[:APPLIES_TO]->(r:EntityRole)
                WITH r, count(DISTINCT c) AS cnt
                SET r.citation_count = cnt
            """)
            session.run("""
                MATCH (c:Citation)-[:DETERMINES]->(i:ImpactLevel)
                WITH i, count(DISTINCT c) AS cnt
                SET i.citation_count = cnt
            """)
            session.run("""
                MATCH (c:Citation)-[:MENTIONS]->(concept:Concept)
                WITH concept, count(DISTINCT c) AS cnt
                SET concept.citation_count = cnt
            """)

    driver.close()
    print(f"\n[Done] Loaded {len(nodes)} nodes and {len(edges)} edges into Neo4j.")


if __name__ == "__main__":
    load()
