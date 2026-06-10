import os
import time
import requests
from chromadb import HttpClient
from prometheus_client import CollectorRegistry, Gauge, push_to_gateway

# 1. Environment & Network Pointers
CHROMA_HOST = os.getenv("CHROMA_DB_HOST", "localhost")
CHROMA_PORT = int(os.getenv("CHROMA_DB_PORT", 8000))
JETSON_OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434") 

def run_benchmark(query_text):
    print(f"\n🚀 Evaluating query: '{query_text}'")
    
    # --- Step 1: Query ChromaDB Vector database ---
    try:
        chroma = HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
        collection = chroma.get_or_create_collection("study_docs")
        
        # Pulling context logs
        db_results = collection.query(query_texts=[query_text], n_results=1)
        context = db_results['documents'][0][0] if db_results['documents'] else "No context found."
        print("📊 Context retrieved from ChromaDB.")
    except Exception as e:
        print(f"❌ ChromaDB Connection Failed: {e}")
        context = "Fallback data: context unavailable."

    # --- Step 2: Time the Jetson Orin's execution speed ---
    payload = {
        "model": "phi4",
        "prompt": f"Context: {context}\n\nQuestion: {query_text}\nAnswer:",
        "stream": False
    }
    
    try:
        start_time = time.time()
        response = requests.post(f"{JETSON_OLLAMA_URL}/api/generate", json=payload, timeout=45)
        latency = time.time() - start_time
        
        if response.status_code == 200:
            print(f"✅ Jetson processing complete in {latency:.4f} seconds.")
        else:
            print(f"⚠️ Jetson returned status code: {response.status_code}")
            return
    except Exception as e:
        print(f"❌ Failed to communicate with Jetson Orin Ollama node: {e}")
        return

    # --- Step 3: Stream Latency data into Prometheus ---
    try:
        registry = CollectorRegistry()
        gauge = Gauge('slm_latency_seconds', 'Inference speed tracking of edge device', registry=registry)
        gauge.set(latency)
        
        # Push to the prometheus container instance
        push_to_gateway('localhost:9090', job='headless_benchmark_job', registry=registry)
        print("📈 Performance telemetry pushed to Prometheus.")
    except Exception as e:
        print(f"⚠️ Prometheus metric push skipped: {e}")

if __name__ == "__main__":
    # Test suite queries
    test_queries = [
        "What are the thermal threshold parameters for edge inference?",
        "How do memory constraints affect Phi-4 precision parameters?"
    ]
    
    for query in test_queries:
        run_benchmark(query)
