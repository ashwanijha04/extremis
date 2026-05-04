# Contributing

Thanks for taking the time to contribute.

## Getting started

```bash
git clone https://github.com/ashwanijha04/lore-ai
cd lore-ai
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

Run the tests:
```bash
pytest tests/ -v
```

Run lint and type checks:
```bash
ruff check src/ tests/
mypy src/lore_ai --ignore-missing-imports
```

## What we're looking for

**Good contributions:**
- Bug fixes with a test that would have caught the bug
- New storage backends (e.g. Redis, DynamoDB) that implement `MemoryStore`
- New embedder implementations (e.g. OpenAI, Cohere) that implement `Embedder`
- Performance improvements to the SQLite cosine search
- Documentation improvements

**Please discuss first (open an issue):**
- New memory layers or significant changes to `types.py`
- Changes to the consolidation prompt strategy
- Anything that changes the public API surface

## How to add a new storage backend

Implement the `MemoryStore` protocol from `lore_ai.interfaces`:

```python
from lore_ai.interfaces import MemoryStore
from lore_ai.types import Memory, MemoryLayer, RecallResult

class MyStore:
    def store(self, memory: Memory) -> Memory: ...
    def get(self, memory_id: UUID) -> Optional[Memory]: ...
    def search(self, query_embedding, layers=None, limit=10, min_score=0.0) -> list[RecallResult]: ...
    def update_score(self, memory_id: UUID, delta: float) -> None: ...
    def supersede(self, old_id: UUID, new_memory: Memory) -> None: ...
    def list_recent(self, layer=None, limit=50) -> list[Memory]: ...
```

Add tests in `tests/test_<backend>.py`. Use the existing `test_core.py` as a reference.

## Pull request checklist

- [ ] Tests pass (`pytest tests/ -v`)
- [ ] Lint passes (`ruff check src/ tests/`)
- [ ] New behaviour has test coverage
- [ ] No secrets or personal data in code or tests

## Reporting bugs

Open an issue with:
1. Python version
2. OS
3. Minimal reproduction script
4. Expected vs actual behaviour
