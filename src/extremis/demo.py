"""
extremis demo — run with: extremis-demo

Shows memory storage, semantic recall, RL scoring, knowledge graph,
and attention scoring in ~20 seconds. No API key needed.
"""

from __future__ import annotations

import sys

# ── ANSI colours (no extra deps) ─────────────────────────────────────────────
BOLD = "\033[1m"
DIM = "\033[2m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
CYAN = "\033[36m"
RED = "\033[31m"
MAGENTA = "\033[35m"
RESET = "\033[0m"

BAR = f"{DIM}{'─' * 54}{RESET}"


def p(text: str = "") -> None:
    print(text)


def header(title: str) -> None:
    p()
    p(f"{BOLD}{CYAN}{title}{RESET}")
    p(BAR)


def ok(msg: str) -> None:
    print(f"  {GREEN}✓{RESET} {msg}")


def arrow(msg: str) -> None:
    print(f"  {YELLOW}→{RESET} {msg}")


def dim(msg: str) -> None:
    print(f"  {DIM}{msg}{RESET}")


def run_demo() -> None:
    p()
    p(f"{BOLD}🧠  extremis demo{RESET}")
    p(BAR)
    p(f"{DIM}Layered memory · RL scoring · Knowledge graph · Attention{RESET}")

    # ── import ────────────────────────────────────────────────────────────────
    import os
    import tempfile

    # Suppress model-loading noise — irrelevant for a demo
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("HF_HUB_VERBOSITY", "error")
    import logging as _logging
    import warnings as _warnings

    _warnings.filterwarnings("ignore")
    _logging.getLogger("huggingface_hub").setLevel(_logging.ERROR)
    _logging.getLogger("sentence_transformers").setLevel(_logging.ERROR)
    _logging.getLogger("transformers").setLevel(_logging.ERROR)

    p()
    print("Loading…", end="", flush=True)

    from extremis import Extremis, MemoryLayer
    from extremis.config import Config
    from extremis.types import EntityType

    # Use a fresh temp dir so the demo is isolated from existing ~/.extremis/ data
    _demo_home = tempfile.mkdtemp(prefix="extremis-demo-")
    mem = Extremis(config=Config(extremis_home=_demo_home))

    # Warm up the embedder silently (suppresses the tqdm loading bar)
    import io as _io
    import sys as _sys

    _old_stderr = _sys.stderr
    _sys.stderr = _io.StringIO()
    try:
        mem._embedder.embed("warmup")
    finally:
        _sys.stderr = _old_stderr

    print(f"\r{GREEN}✓{RESET} extremis ready" + " " * 20)

    # ── 1. Store memories ─────────────────────────────────────────────────────
    header("1 / 4  Storing memories")

    facts = [
        ("User is a Python developer with 10 years experience", MemoryLayer.SEMANTIC),
        ("User prefers concise answers, hates filler words", MemoryLayer.SEMANTIC),
        ("User likes dark mode in all tools", MemoryLayer.SEMANTIC),
        ("Always ask about deadlines before suggesting solutions", MemoryLayer.PROCEDURAL),
    ]
    memories = []
    for content, layer in facts:
        m = mem.remember_now(content, layer=layer)
        memories.append(m)
        ok(f"[{layer.value}]  {content[:55]}")

    # ── 2. Recall ─────────────────────────────────────────────────────────────
    header('2 / 4  Semantic recall  →  "how should I respond to this user?"')

    results = mem.recall("how should I respond to this user?", limit=4)
    for i, r in enumerate(results, 1):
        layer_tag = f"{DIM}[{r.memory.layer.value}]{RESET}"
        p(f"  {i}. {layer_tag} {r.memory.content[:52]}")
        dim(f"   rank={r.final_rank:.3f}  {r.reason}")

    # ── 3. RL scoring ─────────────────────────────────────────────────────────
    header("3 / 4  RL scoring  →  apply feedback, watch ranking shift")

    # Recall all memories without limit so we can find our demo targets
    all_results = mem.recall("how should I respond to this user?", limit=20)

    concise = next((r for r in all_results if "concise" in r.memory.content), None)
    dark = next((r for r in all_results if "dark mode" in r.memory.content), None)

    if concise:
        p(f'  {GREEN}+1  +1{RESET}  "concise answers" confirmed useful twice')
        mem.report_outcome([concise.memory.id], success=True, weight=1.0)
        mem.report_outcome([concise.memory.id], success=True, weight=1.0)

    if dark:
        p(f'  {RED}-1     {RESET}  "dark mode" not relevant to this query')
        mem.report_outcome([dark.memory.id], success=False, weight=1.0)
        p(f"  {DIM}(negative weight applied: 1.0 × 1.5 = -1.5){RESET}")

    p()
    p(f"  {BOLD}Ranking after feedback:{RESET}")
    results2 = mem.recall("how should I respond to this user?", limit=4)
    for r in results2:
        score_str = ""
        if r.memory.score > 0:
            score_str = f"  {GREEN}score {r.memory.score:+.1f}{RESET}"
        elif r.memory.score < 0:
            score_str = f"  {RED}score {r.memory.score:+.1f}{RESET}"
        p(f"  rank={CYAN}{r.final_rank:.3f}{RESET}{score_str}  {r.memory.content[:52]}")

    # ── 4. Knowledge graph ────────────────────────────────────────────────────
    header("4 / 4  Knowledge graph  →  structured entities + relationships")

    mem.kg_add_entity("User", EntityType.PERSON)
    mem.kg_add_entity("Extremis", EntityType.PROJECT)
    mem.kg_add_relationship("User", "Extremis", "building", weight=1.0)
    mem.kg_add_attribute("User", "timezone", "Asia/Dubai")
    mem.kg_add_attribute("User", "language", "Python")
    ok("Added: User → [building] → Extremis")
    ok("Added: User.timezone = Asia/Dubai,  User.language = Python")

    result = mem.kg_query("User")
    p()
    p(f'  {BOLD}kg_query("User"):{RESET}')
    if result:
        for rel in result.relationships:
            arrow(f"{rel.from_entity} —[{rel.rel_type}]→ {rel.to_entity}")
        for attr in result.attributes:
            arrow(f"{attr.entity}.{attr.key} = {attr.value}")

    # ── Attention scorer (bonus) ──────────────────────────────────────────────
    p()
    p(f"{BOLD}Bonus: attention scoring{RESET}  {DIM}(should I even respond?){RESET}")
    p(BAR)
    cases = [
        ("URGENT: production is down, need help NOW!", "dm", "full"),
        ("can you check the deployment logs?", "dm", "standard"),
        ("haha nice one 👍", "group", "minimal/ignore"),
    ]
    for msg, channel, expected in cases:
        score = mem.score_attention(msg, channel=channel)
        color = GREEN if score.level in ("full", "standard") else YELLOW if score.level == "minimal" else DIM
        p(f'  {color}{score.score:>3}/100  {score.level:<10}{RESET}  "{msg[:45]}"')

    # ── Done ──────────────────────────────────────────────────────────────────
    p()
    p(BAR)
    p(f"{BOLD}{GREEN}✅  extremis is working.{RESET}")
    p()
    p(f"  {BOLD}pip install extremis{RESET}          local SQLite, no infra")
    p(f'  {BOLD}pip install "extremis[mcp]"{RESET}   plug into Claude Desktop')
    p(f"  {BOLD}github.com/ashwanijha04/extremis{RESET}")
    p()


def main() -> None:
    try:
        run_demo()
    except KeyboardInterrupt:
        print("\nCancelled.")
        sys.exit(0)


if __name__ == "__main__":
    main()
