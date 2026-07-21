FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY resource_monitor.py .
COPY kg_api.py .

# citation_lookup.jsonl provided via volume mount (see docker-compose.yml)

EXPOSE 8002

CMD ["uvicorn", "kg_api:app", "--host", "0.0.0.0", "--port", "8002"]
