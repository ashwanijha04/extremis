# Deploy to Render

One-click deployment with automatic Postgres provisioning.

## Deploy

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/ashwanijha04/extremis)

This provisions:
- **extremis web service** — the API server
- **Postgres database** — where memories and API keys live (persistent across restarts)

## Get your API key

After deploy, open your extremis service → **Logs** tab. Look for:

```
============================================================
  extremis — FIRST START
============================================================
  No API keys found. Generated your first key:

  extremis_sk_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

  Store this key — it will NOT be shown again.
============================================================
```

Copy the key. It's stored in Postgres and will survive future restarts and redeploys.

## Connect

```python
from extremis import HostedClient

mem = HostedClient(
    api_key="extremis_sk_...",
    base_url="https://your-app.onrender.com",
)
```

Or with the wrap:

```python
from extremis.wrap import Anthropic
from extremis import HostedClient

client = Anthropic(
    api_key="sk-ant-...",
    memory=HostedClient(
        api_key="extremis_sk_...",
        base_url="https://your-app.onrender.com",
    ),
    session_id="user_123",
)
```

## Creating additional keys

Use Render's **Shell** tab for the extremis service:

```bash
extremis-server create-key --namespace alice --label "alice prod"
# → extremis_sk_...

extremis-server list-keys
```

## Disable auto-deploy

By default, Render redeploys on every push to `main`. Since the landing page and docs are also in `main`, this triggers unnecessary rebuilds.

Go to Render → extremis service → **Settings** → **Auto-Deploy** → set to **No**.

Deploy manually after tagging a new version: **Manual Deploy** → **Deploy latest commit**.

## Environment variables

Set in Render dashboard → extremis service → **Environment**:

| Variable | Description |
|----------|-------------|
| `EXTREMIS_STORE` | Set to `postgres` (auto-configured from `render.yaml`) |
| `EXTREMIS_POSTGRES_URL` | Auto-injected from the Postgres plugin |
| `EXTREMIS_EMBEDDER` | Override embedding model (default: `all-MiniLM-L6-v2`) |
| `OPENAI_API_KEY` | Required if using `text-embedding-*` embedder |
| `ANTHROPIC_API_KEY` | Required for consolidation |

## Costs

Render free tier includes:
- 1 web service (750 hours/month)
- 1 Postgres instance (1 GB storage)
- Services spin down after 15 min inactivity

Free tier is sufficient for personal use and testing.
