# extremis

**Memory that gets smarter the more your agent uses it.**

extremis is an open-source memory layer for AI agents. It handles embedding, storage, retrieval ranking, and consolidation — so you don't have to.

---

## The fastest path

Change one import. Get persistent, learning memory for free.

=== "Claude (Anthropic)"

    ```python
    from extremis.wrap import Anthropic  # (1)
    from extremis import Extremis

    client = Anthropic(api_key="sk-ant-...", memory=Extremis())

    # Your existing code — unchanged
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        messages=[{"role": "user", "content": "What's my name?"}]
    )
    # Memory recalled before call, saved after. Nothing else to do.
    ```

    1. Drop-in for `anthropic.Anthropic`. Every other method passes through unchanged.

=== "OpenAI"

    ```python
    from extremis.wrap import OpenAI
    from extremis import Extremis

    client = OpenAI(api_key="sk-...", memory=Extremis())
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "What did we discuss last time?"}]
    )
    ```

=== "Manual API"

    ```python
    from extremis import Extremis

    mem = Extremis()

    # Store
    mem.remember("User is building a WhatsApp AI", conversation_id="c1")

    # Recall — every result explains why it ranked
    results = mem.recall("what is the user building?")
    for r in results:
        print(r.memory.content)
        print(r.reason)  # "similarity 0.91 · score +2.0 · used 5× · 3d old"

    # Feedback — useful memories surface first over time
    mem.report_outcome([results[0].memory.id], success=True)
    ```

---

## What makes it different

| | extremis | Mem0 | LangChain | Raw vectors |
|---|---|---|---|---|
| **RL-scored retrieval** | ✅ | ❌ | ❌ | ❌ |
| **Memory explains itself** | ✅ | ❌ | ❌ | ❌ |
| **Knowledge graph** | ✅ | ❌ | ❌ | ❌ |
| **Drop-in LLM wrapper** | ✅ | ❌ | ❌ | ❌ |
| **Backend-agnostic** | ✅ 4 backends | Cloud only | Manual | — |
| **MCP server (Claude)** | ✅ | ❌ | ❌ | ❌ |
| **Self-hostable** | ✅ | ❌ | ✅ | ✅ |
| **Open source (MIT)** | ✅ | Partial | ✅ | ✅ |

---

## Install

=== "Python"

    ```bash
    pip3.11 install extremis                    # core
    pip3.11 install "extremis[wrap-anthropic]"  # + Claude wrapper
    pip3.11 install "extremis[wrap-openai]"     # + OpenAI wrapper
    pip3.11 install "extremis[mcp]"             # + Claude Desktop MCP
    pip3.11 install "extremis[verification]"    # + local NLI for hallucination detection
    ```

    !!! warning "Python 3.11+ required"
        If `pip install` says "no matching distribution found", your `pip` points to Python 3.9.
        Run `python3 --version` to check. Fix: `brew install python@3.11` then use `pip3.11`.

=== "TypeScript"

    ```bash
    npm install @extremis/sdk
    ```

    Zero runtime dependencies. Works on Node 18+, Bun, Deno, Cloudflare Workers, browsers. See the [TypeScript SDK guide](sdks/typescript.md).

---

## Next steps

<div class="grid cards" markdown>

-   :material-clock-fast: **[Quickstart](getting-started/quickstart.md)**

    Get memory working in 5 minutes

-   :material-swap-horizontal: **[Wrap your existing app](wrap/index.md)**

    One import change for Claude or OpenAI

-   :material-language-typescript: **[TypeScript SDK](sdks/typescript.md)**

    `npm install @extremis/sdk` — same surface as Python

-   :material-shield-check: **[Hallucination detection](concepts/hallucination-detection.md)**

    Runtime NLI + judge + self-consistency on every consolidation

-   :material-brain: **[Core concepts](concepts/memory-layers.md)**

    How layers, RL scoring, and consolidation work

-   :material-server: **[Deploy to Render](deployment/render.md)**

    Hosted server with persistent Postgres memory

</div>
