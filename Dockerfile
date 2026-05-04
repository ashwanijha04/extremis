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

ENV LORE_SERVER_HOME=/data/server
ENV LORE_STORE=sqlite
ENV LORE_FRIDAY_HOME=/data

VOLUME ["/data"]
EXPOSE 8000

CMD ["lore-server", "serve", "--host", "0.0.0.0", "--port", "8000"]
