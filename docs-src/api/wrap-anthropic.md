# wrap.Anthropic

See [wrap → Anthropic / Claude](../wrap/anthropic.md) for full documentation.

## Quick reference

```python
from extremis.wrap import Anthropic
from extremis import Extremis

client = Anthropic(
    api_key="sk-ant-...",
    memory=Extremis(),           # or HostedClient(...)
    session_id="user_123",       # optional, defaults to UUID
    # any other anthropic.Anthropic() kwargs
)

response = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=1024,
    messages=[{"role": "user", "content": "..."}],
)
```

All methods other than `messages.create()` pass through to the underlying `anthropic.Anthropic` client unchanged.
