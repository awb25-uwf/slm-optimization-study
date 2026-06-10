import chromadb

print("Connecting to local ChromaDB instance...")
# 1. Connect to the database running on your host machine port
client = chromadb.HttpClient(host="localhost", port=8000)

print("Creating database collection...")
# 2. Create a brand new vector container structure named 'study_docs'
# If it already exists, this command will safely open it instead
collection = client.get_or_create_collection(name="study_docs")

# 3. Add your initial baseline documentation snippets
print("Seeding text embeddings into the database...")
collection.add(
    documents=[
        "Optimization Phase 1: Small Language Models like Phi-4 require strict quantized INT4 or FP16 memory maps to run efficiently on edge hardware constraints.",
        "NVIDIA Jetson AGX Orin power profiles can be dynamically adjusted between 15W, 30W, and 50W to balance inference latency against structural thermal ceilings.",
        "Retrieval-Augmented Generation (RAG) loops enhance SLM accuracy by passing local context windows directly into the system prompt parameters."
    ],
    metadatas=[
        {"source": "slm_manual"}, 
        {"source": "hardware_specs"}, 
        {"source": "rag_framework"}
    ],
    ids=["doc_001", "doc_002", "doc_003"]
)

print("🎉 Success! Your ChromaDB vector database is built, seeded, and ready for your simulation runs.")
