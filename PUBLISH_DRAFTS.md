# Publication Drafts

Ready-to-post content. Copy-paste to publish.

---

## 1. awesome-mcp-servers PR

**Repo:** https://github.com/punkpeye/awesome-mcp-servers
**Action:** Fork → edit README.md → add to Memory section → open PR

**Add this line under the Memory section:**
```markdown
- [extremis](https://github.com/ashwanijha04/extremis) - Layered, learning memory with RL scoring, knowledge graph, and consolidation. SQLite/Postgres/Chroma/Pinecone backends. `pip install "extremis[mcp]"`
```

**PR title:** `Add extremis — RL-scored memory with knowledge graph`

**PR body:**
```
extremis is an open-source memory server that goes beyond cosine search —
it uses reinforcement learning to learn which memories are actually useful
over time. Negative signals apply 1.5× weight (asymmetric, like human
threat-memory).

Features:
- RL-scored retrieval (unique)
- Knowledge graph alongside vectors
- 5 memory layers (episodic/semantic/procedural/identity/working)
- 4 backends (SQLite/Postgres/Chroma/Pinecone)
- 10 MCP tools
- Every recalled memory explains why it ranked

pip install "extremis[mcp]" → works in Claude Desktop in 30 seconds
```

---

## 2. Dev.to article

**Title:** Add persistent memory to Claude in one line of Python

**Tags:** claude, python, ai, mcp

**Body:**

---

Claude forgets everything the moment a conversation ends. If you're building anything with Claude — a chatbot, an agent, a personal assistant — this is your biggest friction point.

I spent a few months building **extremis**, an open-source memory layer that fixes this. Here's what it looks like:

```python
# Before
import anthropic
client = anthropic.Anthropic(api_key="sk-ant-...")

# After — one import change
from extremis.wrap import Anthropic
from extremis import Extremis

client = Anthropic(api_key="sk-ant-...", memory=Extremis())

# Your existing code is unchanged
response = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=1024,
    messages=[{"role": "user", "content": "What's my name?"}]
)
# Memory recalled before call, saved after. That's it.
```

That's the entire change. Every `client.messages.create()` call now:
1. Recalls relevant past context before the LLM call
2. Saves the conversation after

**What makes it different from just storing messages in a database:**

Most memory systems are cosine search — the most *similar* memory wins. extremis adds RL scoring: memories that actually helped get a higher score. Ones that didn't help fade out, with 1.5× weight on negative signals (same asymmetry human threat-learning uses).

Every recalled memory also tells you why it ranked:
```
"similarity 0.91 · score +4.0 · used 8× · 3d old"
```

**It also has a knowledge graph:**

```python
mem.kg_add_entity("Alice", EntityType.PERSON)
mem.kg_add_relationship("Alice", "Acme Corp", "works_at")
mem.kg_add_attribute("Alice", "timezone", "Asia/Dubai")

result = mem.kg_query("Alice")
# → works_at Acme Corp, timezone: Asia/Dubai
```

Vectors can't answer "who does Alice work for?" — the graph can.

**Claude Desktop (no code):**

```bash
pip3.11 install "extremis[mcp]"
```

Add to `claude_desktop_config.json`:
```json
{
  "mcpServers": {
    "extremis": {
      "command": "/opt/homebrew/bin/extremis-mcp"
    }
  }
}
```

Restart Claude. Done. 10 memory tools appear automatically.

**Try it:**

```bash
pip3.11 install extremis
extremis-demo
```

GitHub: https://github.com/ashwanijha04/extremis
Docs: https://ashwanijha04.github.io/extremis/docs/

Happy to answer questions about the RL scoring architecture or anything else.

---

## 3. Product Hunt listing

**Name:** extremis

**Tagline:** Memory that gets smarter the more your AI agent uses it

**Description:**
extremis is an open-source memory layer for AI agents. Change one import and your Claude or OpenAI app remembers everything across sessions — no vector database to run, no RAG pipeline to build.

What makes it different: most memory systems rank by cosine similarity (similar = relevant). extremis adds reinforcement learning — memories that actually helped get scored up, ones that didn't fade out. Over time, the most useful context surfaces first.

Features:
→ Drop-in wrapper: `from extremis.wrap import Anthropic` — one import change
→ RL-scored retrieval with asymmetric weighting (1.5× on negative signals)
→ Knowledge graph alongside vectors
→ 5 memory layers (episodic/semantic/procedural/identity/working)
→ MCP server for Claude Desktop (10 tools, zero code)
→ 4 backends: SQLite locally, Postgres/Chroma/Pinecone for production
→ Deploy to Render in one click (free Postgres included)
→ MIT license, zero telemetry

**First comment:**
"Hey PH! 👋

I built extremis after getting frustrated that every AI tool I built had to re-explain context every session.

The quickest way to see it: `pip3.11 install extremis && extremis-demo`

For Claude Desktop users: two lines of config and Claude remembers everything. No code.

For developers: `from extremis.wrap import Anthropic` — one import change, memory works.

Would love feedback on the RL scoring design and what use cases people have. What would you build if your AI never forgot anything?"

**Links:**
- Website: https://ashwanijha04.github.io/extremis
- Docs: https://ashwanijha04.github.io/extremis/docs/
- GitHub: https://github.com/ashwanijha04/extremis

---

## 4. SEO/AEO keyword targets

Pages to write / topics to target:

| Query | Target page |
|-------|------------|
| "Claude memory MCP" | Landing page + MCP docs |
| "persistent memory for Claude" | Landing page |
| "add memory to ChatGPT" | wrap/openai.md |
| "AI agent memory Python" | getting-started/quickstart.md |
| "LangChain memory alternative" | README comparison table |
| "Mem0 alternative open source" | README comparison table |
| "extremis AI memory" | All pages |

**Answer Engine targets (things AI assistants answer):**
- "How do I add memory to Claude?" → our quickstart
- "What is the best memory system for AI agents?" → our README
- "How does extremis work?" → our docs index

To get indexed by AI: ensure the first paragraph of every major page answers the question directly. AI training data weights early, clear answers heavily.
