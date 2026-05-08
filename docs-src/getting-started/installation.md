# Installation

## Requirements

- Python 3.11 or later
- No API key required for the core library

!!! tip "Check your Python version"
    ```bash
    python3 --version
    ```
    If it says 3.9 or 3.10, see [below](#older-python).

## Install

```bash
pip3.11 install extremis
```

### With extras

Install only what you need:

| Extra | What it adds | Install |
|-------|-------------|---------|
| `wrap-anthropic` | Drop-in Claude client wrapper | `pip3.11 install "extremis[wrap-anthropic]"` |
| `wrap-openai` | Drop-in OpenAI client wrapper | `pip3.11 install "extremis[wrap-openai]"` |
| `mcp` | MCP server for Claude Desktop / Code | `pip3.11 install "extremis[mcp]"` |
| `postgres` | Postgres + pgvector backend | `pip3.11 install "extremis[postgres]"` |
| `chroma` | ChromaDB backend | `pip3.11 install "extremis[chroma]"` |
| `pinecone` | Pinecone backend | `pip3.11 install "extremis[pinecone]"` |
| `openai` | OpenAI embeddings (replaces local model) | `pip3.11 install "extremis[openai]"` |
| `server` | REST API server (FastAPI + uvicorn) | `pip3.11 install "extremis[server]"` |
| `client` | HTTP client for hosted server | `pip3.11 install "extremis[client]"` |
| `all` | Everything | `pip3.11 install "extremis[all]"` |

## First run note

On first `recall()`, extremis downloads `all-MiniLM-L6-v2` (~90 MB) from HuggingFace. This is a one-time download cached to `~/.cache/huggingface/`.

To skip the download entirely, use OpenAI embeddings:

```bash
pip3.11 install "extremis[openai]"
```
```bash
EXTREMIS_EMBEDDER=text-embedding-3-small
OPENAI_API_KEY=sk-...
```

## Older Python

=== "macOS"

    ```bash
    brew install python@3.11
    /opt/homebrew/bin/pip3.11 install extremis
    ```

=== "Linux"

    ```bash
    sudo apt install python3.11 python3.11-pip
    pip3.11 install extremis
    ```

=== "Windows"

    Download Python 3.11+ from [python.org/downloads](https://python.org/downloads).
    Then:
    ```bash
    pip install extremis
    ```

## Verify

```python
import extremis
mem = extremis.Extremis()
mem.remember("test", conversation_id="verify")
results = mem.recall("test")
print(results[0].memory.content)  # → "test"
print("✓ extremis working")
```

Or run the interactive demo:

```bash
extremis-demo
```
