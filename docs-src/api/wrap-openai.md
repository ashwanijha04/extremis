# wrap.OpenAI

See [wrap → OpenAI](../wrap/openai.md) for full documentation.

## Quick reference

```python
from extremis.wrap import OpenAI
from extremis import Extremis

client = OpenAI(
    api_key="sk-...",
    memory=Extremis(),
    session_id="user_123",
)

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "..."}],
)
```

All methods other than `chat.completions.create()` pass through unchanged.
