# Supabase

Supabase is the easiest hosted Postgres option for extremis. Your project already has pgvector — no manual setup needed.

## Install

```bash
pip3.11 install "extremis[postgres]"
```

## Configure

=== "Environment variables"

    ```bash
    EXTREMIS_STORE=supabase
    SUPABASE_DB_URL=postgresql://postgres.[ID]:[PASS]@aws-0-us-east-1.pooler.supabase.com:5432/postgres
    ```

=== "Python"

    ```python
    from extremis import Extremis, Config

    mem = Extremis(config=Config(
        store="supabase",
        # OR set SUPABASE_DB_URL env var — extremis reads it automatically
    ))
    ```

extremis automatically reads `SUPABASE_DB_URL` or `DATABASE_URL` — no extra config needed if those are already in your environment.

## Get your connection string

1. Go to your Supabase project → **Settings** → **Database**
2. Under **Connection string**, select **URI** mode
3. Use the **Transaction pooler** URL (recommended for serverless)

!!! tip "Which URL to use?"
    - **Transaction pooler** (`aws-0-*.pooler.supabase.com:5432`) — for serverless / short-lived connections
    - **Direct connection** (`db.*.supabase.co:5432`) — for long-running servers

## pgvector

Supabase includes pgvector in all projects. extremis enables it automatically on first start — no manual SQL needed.

## With the hosted server

```bash
# In your Render/Railway environment variables:
EXTREMIS_STORE=supabase
SUPABASE_DB_URL=postgresql://postgres.[ID]:[PASS]@...
```

## With the wrap

```python
import os
os.environ["EXTREMIS_STORE"] = "supabase"
os.environ["SUPABASE_DB_URL"] = "postgresql://..."

from extremis.wrap import Anthropic
from extremis import Extremis

client = Anthropic(api_key="sk-ant-...", memory=Extremis())
```

## Free tier

Supabase free tier includes:
- 500 MB database storage
- pgvector (no extra charge)
- Unlimited API requests

Sufficient for most personal and small-team use cases.
