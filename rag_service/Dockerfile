FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
# Install CPU-only torch, better than GPU for running the small embedding model
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

COPY resource_monitor.py .
COPY build_vector_db.py .
COPY rag_api.py .

EXPOSE 8001

CMD ["uvicorn", "rag_api:app", "--host", "0.0.0.0", "--port", "8001"]
