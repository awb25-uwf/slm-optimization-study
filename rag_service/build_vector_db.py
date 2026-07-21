#!/usr/bin/env python3
"""
Builds a persistent Chroma vector database from citation_lookup.jsonl for use
as the RAG module's retrieval backend. Wraps the embedding + indexing step in
ResourceMonitor to capture CPU/RAM/GPU consumption during the vectorization step,
written as Prometheus textfile-collector metrics for later scraping.
"""
import json
import chromadb
from sentence_transformers import SentenceTransformer

from resource_monitor import ResourceMonitor

LOOKUP_FILE = "citation_lookup.jsonl"
CHROMA_HOST = "localhost" 
CHROMA_PORT = 8000  # matches the existing chromadb container's published port
COLLECTION_NAME = "nerc_cip_citations"
EMBEDDING_MODEL = "all-mpnet-base-v2"  # good balance between speed and accuracy
PROM_TEXTFILE_DIR = "./prom_textfile"


def load_jsonl(path):
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def build():
    entries = load_jsonl(LOOKUP_FILE)
    print(f"[Plan] {len(entries)} citation entries to embed and index.")

    with ResourceMonitor("rag_vector_db_build", output_dir=PROM_TEXTFILE_DIR, sample_interval=1.0):
        print(f"Loading embedding model ({EMBEDDING_MODEL})...")
        embedder = SentenceTransformer(EMBEDDING_MODEL)

        print(f"Connecting to Chroma server at {CHROMA_HOST}:{CHROMA_PORT}...")
        client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass
        collection = client.create_collection(COLLECTION_NAME)

        print("Embedding and indexing...")
        # Prepend the citation to the text being embedded to distinguish
        # among unrelated chunks.
        documents = [f"{e['citation']}: {e['context']}" for e in entries]
        metadatas = [{"citation": e["citation"], "source_file": e.get("source_file", "")} for e in entries]
        ids = [f"citation_{i}" for i in range(len(entries))]

        embeddings = embedder.encode(documents, show_progress_bar=True, batch_size=32).tolist()

        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

    print(f"\n[Done] Indexed {len(entries)} citation entries into Chroma collection "
          f"'{COLLECTION_NAME}' on server {CHROMA_HOST}:{CHROMA_PORT}")


if __name__ == "__main__":
    build()
