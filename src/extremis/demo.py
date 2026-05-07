"""
extremis demo — run with: extremis-demo

Shows memory storage, semantic recall, RL scoring, knowledge graph,
and attention scoring in ~20 seconds. No API key needed.
"""

from __future__ import annotations

import sys
import threading
import time

# ── ANSI colours (no extra deps) ─────────────────────────────────────────────
BOLD = "\033[1m"
DIM = "\033[2m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
CYAN = "\033[36m"
RED = "\033[31m"
RESET = "\033[0m"

BAR = f"{DIM}{'─' * 54}{RESET}"

SPINNER_FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"


def p(text: str = "") -> None:
    print(text, flush=True)


def header(title: str) -> None:
    p()
    p(f"{BOLD}{CYAN}{title}{RESET}")
    p(BAR)


def ok(msg: str) -> None:
    print(f"  {GREEN}✓{RESET} {msg}", flush=True)


def arrow(msg: str) -> None:
    print(f"  {YELLOW}→{RESET} {msg}", flush=True)


def dim(msg: str) -> None:
    print(f"  {DIM}{msg}{RESET}", flush=True)


class Spinner:
    """Show a spinning indicator while a slow operation runs."""

    def __init__(self, label: str) -> None:
        self._label = label
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._spin, daemon=True)

    def _spin(self) -> None:
        i = 0
        while not self._stop.is_set():
            frame = SPINNER_FRAMES[i % len(SPINNER_FRAMES)]
            print(f"\r  {DIM}{frame}{RESET}  {self._label}", end="", flush=True)
            time.sleep(0.08)
            i += 1

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *_):
        self._stop.set()
        self._thread.join()
        # Clear the spinner line
        print("\r" + " " * (len(self._label) + 8) + "\r", end="", flush=True)


def run_demo() -> None:
    p()
    p(f"{BOLD}🧠  extremis demo{RESET}")
    p(BAR)
    p(f"{DIM}Layered memory · RL scoring · Knowledge graph · Attention{RESET}")

    # ── suppress noise ────────────────────────────────────────────────────────
    import io as _io
    import logging as _logging
    import os
    import sys as _sys
    import tempfile
    import warnings as _warnings

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("HF_HUB_VERBOSITY", "error")
    _warnings.filterwarnings("ignore")
    _logging.getLogger("huggingface_hub").setLevel(_logging.ERROR)
    _logging.getLogger("sentence_transformers").setLevel(_logging.ERROR)
    _logging.getLogger("transformers").setLevel(_logging.ERROR)

    # ── load model ────────────────────────────────────────────────────────────
    p()
    with Spinner("Importing extremis…"):
        from extremis import Extremis, MemoryLayer
        from extremis.config import Config
        from extremis.types import EntityType

    ok("extremis imported")

    with Spinner("Loading embedding model (all-MiniLM-L6-v2)…"):
        _demo_home = tempfile.mkdtemp(prefix="extremis-demo-")
        mem = Extremis(config=Config(extremis_home=_demo_home))
        # warm up silently
        _old_stderr = _sys.stderr
        _sys.stderr = _io.StringIO()
        try:
            mem._embedder.embed("warmup")
        finally:
            _sys.stderr = _old_stderr

    ok(f"Model ready — {mem._embedder.dim}d vectors, runs locally, no API key")

    # ── 1. Store memories ─────────────────────────────────────────────────────
    header("1 / 4  Storing memories")

    facts = [
        ("User is a Python developer with 10 years experience", MemoryLayer.SEMANTIC),
        ("User prefers concise answers, hates filler words", MemoryLayer.SEMANTIC),
        ("User likes dark mode in all tools", MemoryLayer.SEMANTIC),
        ("Always ask about deadlines before suggesting solutions", MemoryLayer.PROCEDURAL),
    ]
    for content, layer in facts:
        with Spinner(f"Embedding + storing [{layer.value}]…"):
            mem.remember_now(content, layer=layer)
        ok(f"[{layer.value}]  {content[:52]}")

    # ── 2. Recall ─────────────────────────────────────────────────────────────
    header('2 / 4  Semantic recall  →  "how should I respond to this user?"')

    with Spinner("Embedding query…"):
        mem._embedder.embed("how should I respond to this user?")
    ok("Query embedded")

    with Spinner("Searching memories (cosine × RL score × recency)…"):
        results = mem.recall("how should I respond to this user?", limit=4)
    ok(f"Found {len(results)} memories")
    p()

    for i, r in enumerate(results, 1):
        layer_tag = f"{DIM}[{r.memory.layer.value}]{RESET}"
        p(f"  {i}. {layer_tag} {r.memory.content[:52]}")
        dim(f"     rank={r.final_rank:.3f}  {r.reason}")

    # ── 3. RL scoring ─────────────────────────────────────────────────────────
    header("3 / 4  RL scoring  →  apply feedback, watch ranking shift")

    all_results = mem.recall("how should I respond to this user?", limit=20)
    concise = next((r for r in all_results if "concise" in r.memory.content), None)
    dark = next((r for r in all_results if "dark mode" in r.memory.content), None)

    if concise:
        print(f'  {GREEN}+1{RESET}  Applying positive signal to "concise answers"…', end="", flush=True)
        mem.report_outcome([concise.memory.id], success=True, weight=1.0)
        time.sleep(0.15)
        print(f"  score: {GREEN}+1.0{RESET}", flush=True)

        print(f"  {GREEN}+1{RESET}  Applying again (confirmed twice)…", end="", flush=True)
        mem.report_outcome([concise.memory.id], success=True, weight=1.0)
        time.sleep(0.15)
        print(f"  score: {GREEN}+2.0{RESET}", flush=True)

    if dark:
        print(f'  {RED}-1{RESET}  Applying negative signal to "dark mode"…', end="", flush=True)
        mem.report_outcome([dark.memory.id], success=False, weight=1.0)
        time.sleep(0.15)
        print(f"  score: {RED}-1.5{RESET}  {DIM}(×1.5 asymmetric weight){RESET}", flush=True)

    p()
    with Spinner("Re-ranking memories with updated scores…"):
        results2 = mem.recall("how should I respond to this user?", limit=4)
        time.sleep(0.3)
    ok("Re-ranked")
    p()

    for r in results2:
        score_str = ""
        if r.memory.score > 0:
            score_str = f"  {GREEN}score {r.memory.score:+.1f} ↑{RESET}"
        elif r.memory.score < 0:
            score_str = f"  {RED}score {r.memory.score:+.1f} ↓{RESET}"
        p(f"  rank={CYAN}{r.final_rank:.3f}{RESET}{score_str}  {r.memory.content[:50]}")

    # ── 4. Knowledge graph ────────────────────────────────────────────────────
    header("4 / 4  Knowledge graph  →  entities + relationships + attributes")

    with Spinner("Writing entities and relationships…"):
        mem.kg_add_entity("User", EntityType.PERSON)
        mem.kg_add_entity("Extremis", EntityType.PROJECT)
        mem.kg_add_relationship("User", "Extremis", "building", weight=1.0)
        mem.kg_add_attribute("User", "timezone", "Asia/Dubai")
        mem.kg_add_attribute("User", "language", "Python")
        time.sleep(0.2)

    ok("User → [building] → Extremis")
    ok("User.timezone = Asia/Dubai  ·  User.language = Python")

    with Spinner("Querying graph…"):
        result = mem.kg_query("User")
        time.sleep(0.15)

    p()
    p(f'  {BOLD}kg_query("User"):{RESET}')
    if result:
        for rel in result.relationships:
            arrow(f"{rel.from_entity} —[{rel.rel_type}]→ {rel.to_entity}")
        for attr in result.attributes:
            arrow(f"{attr.entity}.{attr.key} = {attr.value}")

    # ── Attention scorer ──────────────────────────────────────────────────────
    p()
    p(f"{BOLD}Bonus: attention scoring{RESET}  {DIM}(should I even respond?){RESET}")
    p(BAR)
    cases = [
        ("URGENT: production is down, need help NOW!", "dm"),
        ("can you check the deployment logs?", "dm"),
        ("haha nice one 👍", "group"),
    ]
    for msg, channel in cases:
        print(f'  {DIM}scoring:{RESET} "{msg[:42]}"…', end="", flush=True)
        time.sleep(0.1)
        score = mem.score_attention(msg, channel=channel)
        color = GREEN if score.level in ("full", "standard") else YELLOW if score.level == "minimal" else DIM
        print(f'\r  {color}{score.score:>3}/100  {score.level:<10}{RESET}  "{msg[:42]}"', flush=True)

    # ── Done ──────────────────────────────────────────────────────────────────
    p()
    p(BAR)
    p(f"{BOLD}{GREEN}✅  extremis is working.{RESET}")
    p()
    p(f"  {BOLD}pip3.11 install extremis{RESET}          local SQLite, no infra")
    p(f'  {BOLD}pip3.11 install "extremis[mcp]"{RESET}   plug into Claude Desktop')
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
