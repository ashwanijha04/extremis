FROM python:3.11-slim

WORKDIR /app

# system deps for psycopg2
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq-dev gcc \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md LICENSE ./
COPY src/ src/

RUN pip install --no-cache-dir ".[server,postgres,openai]"

# pre-download the default embedding model so first request isn't slow
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')" || true

ENV EXTREMIS_SERVER_HOME=/data/server
ENV EXTREMIS_HOME=/data

# Default to postgres when EXTREMIS_POSTGRES_URL is set (Railway injects this).
# Falls back to sqlite if not set — useful for local Docker testing.
ENV EXTREMIS_STORE=sqlite

VOLUME ["/data"]

# Railway sets $PORT dynamically. Use it if present, otherwise 8000.
CMD extremis-server serve --host 0.0.0.0 --port ${PORT:-8000}
