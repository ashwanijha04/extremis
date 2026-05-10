"""extremis CLI — stats, search, doctor."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _stats(args: argparse.Namespace) -> None:
    from extremis import Extremis
    from extremis.config import Config
    from extremis.types import MemoryLayer

    config = Config()
    mem = Extremis(config=config)
    store = mem.get_local_store()
    log = mem.get_log()

    counts: dict[str, int] = {layer.value: 0 for layer in MemoryLayer}
    all_memories = store.search(
        query_embedding=[0.0] * config.embedding_dim,
        limit=10_000,
        min_score=0.0,
    )
    for r in all_memories:
        counts[r.memory.layer.value] += 1

    log_entries = log.read_since(None)
    checkpoint = log.get_checkpoint()

    ns = config.namespace

    print(f"\nNamespace: {ns}")
    print("─" * 36)
    for layer in MemoryLayer:
        label = f"  {layer.value:<12}"
        print(f"{label}: {counts[layer.value]:>4} memories")
    print("─" * 36)
    total = sum(counts.values())
    print(f"  {'total':<12}: {total:>4} memories")
    print(f"  {'log entries':<12}: {len(log_entries):>4}")
    if checkpoint:
        print(f"  last consolidation checkpoint: {checkpoint[:19]}")
    else:
        print("  last consolidation: never run")

    top = sorted(all_memories, key=lambda r: r.memory.score, reverse=True)[:5]
    if top:
        print("\nTop memories by RL score:")
        for r in top:
            score = f"{r.memory.score:+.1f}"
            layer = r.memory.layer.value
            content = r.memory.content[:72]
            print(f"  {score:>5}  [{layer}] {content}")
    print()


def _doctor(args: argparse.Namespace) -> None:

    ok = "✅"
    warn = "⚠️ "
    fail = "❌"
    issues = 0

    print("\nextremi doctor\n")

    # Python version
    major, minor = sys.version_info[:2]
    if (major, minor) >= (3, 11):
        print(f"{ok}  Python {sys.version.split()[0]}")
    else:
        print(f"{fail}  Python {sys.version.split()[0]} — requires 3.11+")
        issues += 1

    # extremis importable
    try:
        import extremis

        print(f"{ok}  extremis {extremis.__version__ if hasattr(extremis, '__version__') else 'installed'}")
    except ImportError:
        print(f"{fail}  extremis not importable")
        issues += 1

    # SQLite writable
    from extremis.config import Config

    config = Config()
    db_path = Path(config.resolved_local_db_path())
    try:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        db_path.touch(exist_ok=True)
        print(f"{ok}  SQLite store writable ({db_path})")
    except Exception as e:
        print(f"{fail}  SQLite store not writable: {e}")
        issues += 1

    # sentence-transformers model cached
    try:
        cache = Path.home() / ".cache" / "huggingface" / "hub"
        cached = any(cache.glob("**/all-MiniLM*")) if cache.exists() else False
        if cached:
            print(f"{ok}  sentence-transformers model cached")
        else:
            print(f"{warn} sentence-transformers model not cached — will download ~90 MB on first use")
    except Exception:
        print(f"{warn} could not check model cache")

    # ANTHROPIC_API_KEY
    import os

    if os.environ.get("ANTHROPIC_API_KEY"):
        print(f"{ok}  ANTHROPIC_API_KEY set")
    else:
        print(f"{warn} ANTHROPIC_API_KEY not set — required for consolidation")

    # claude mcp config
    mcp_config = Path.home() / "Library" / "Application Support" / "Claude" / "claude_desktop_config.json"
    if mcp_config.exists():
        import json

        try:
            data = json.loads(mcp_config.read_text())
            if "extremis" in data.get("mcpServers", {}):
                print(f"{ok}  Claude Desktop MCP config found")
            else:
                print(f"{warn} Claude Desktop config exists but extremis MCP not configured")
                print(f"       Run: pip install 'extremis[mcp]' then add to {mcp_config}")
        except Exception:
            print(f"{warn} Claude Desktop config found but could not parse")
    else:
        print(f"{warn} Claude Desktop MCP config not found")
        print("       Run: claude mcp add extremis extremis-mcp \\")
        print("                --env EXTREMIS_FRIDAY_HOME=~/.extremis \\")
        print("                --env ANTHROPIC_API_KEY=sk-ant-...")

    print()
    if issues == 0:
        print("All checks passed.\n")
    else:
        print(f"{issues} issue(s) found.\n")
        sys.exit(1)


def _traces(args: argparse.Namespace) -> None:
    import json
    from pathlib import Path

    from extremis.config import Config

    config = Config()
    traces_path = Path(config.resolved_traces_path())

    if not traces_path.exists():
        print(f"\nNo traces file found at {traces_path}")
        print("Enable observability: EXTREMIS_OBSERVE=true or Config(observe=True)\n")
        return

    spans = []
    for line in traces_path.read_text().splitlines():
        if line.strip():
            try:
                spans.append(json.loads(line))
            except json.JSONDecodeError:
                pass

    if not spans:
        print("\nNo spans recorded yet.\n")
        return

    total_input = total_output = 0
    llm_calls: dict[str, dict] = {}
    total_duration_ms = 0.0

    for span in spans:
        attrs = span.get("attributes", {})
        name = span.get("name", "unknown")
        dur = span.get("duration_ms") or 0.0
        inp = attrs.get("tokens_input", 0)
        out = attrs.get("tokens_output", 0)
        total_input += inp
        total_output += out
        total_duration_ms += dur
        if inp or out:
            key = f"{name} ({attrs.get('model', '?')})"
            if key not in llm_calls:
                llm_calls[key] = {"calls": 0, "input": 0, "output": 0, "duration_ms": 0.0}
            llm_calls[key]["calls"] += 1
            llm_calls[key]["input"] += inp
            llm_calls[key]["output"] += out
            llm_calls[key]["duration_ms"] += dur

    haiku_cost = (total_input / 1_000_000 * 0.80) + (total_output / 1_000_000 * 4.00)

    print(f"\nTraces: {traces_path}")
    print(f"Total spans     : {len(spans)}")
    print(f"Total duration  : {total_duration_ms / 1000:.1f}s")
    print("─" * 52)

    if llm_calls:
        print("\nLLM call breakdown:")
        for name, stats in sorted(llm_calls.items(), key=lambda x: -x[1]["input"]):
            cost = (stats["input"] / 1_000_000 * 0.80) + (stats["output"] / 1_000_000 * 4.00)
            print(
                f"  {name}\n"
                f"    calls={stats['calls']}  "
                f"in={stats['input']:,}  out={stats['output']:,}  "
                f"cost=${cost:.4f}  avg={stats['duration_ms'] / max(stats['calls'], 1):.0f}ms"
            )
        print("─" * 52)
        print(f"\nTotal tokens    : {total_input + total_output:,} ({total_input:,} in / {total_output:,} out)")
        print(f"Estimated cost  : ${haiku_cost:.4f} (Haiku rates)\n")
    else:
        print("\nNo LLM calls traced yet.\n")


def main() -> None:
    parser = argparse.ArgumentParser(prog="extremis", description="extremis CLI")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("stats", help="Show memory counts, top memories, and log stats")
    sub.add_parser("doctor", help="Validate setup and diagnose common issues")
    sub.add_parser("traces", help="Show LLM cost and timing summary from peekr traces")

    args = parser.parse_args()

    if args.command == "stats":
        _stats(args)
    elif args.command == "doctor":
        _doctor(args)
    elif args.command == "traces":
        _traces(args)
    else:
        parser.print_help()
