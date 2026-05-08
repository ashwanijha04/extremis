# Claude Desktop MCP setup

Add persistent memory to Claude Desktop in 30 seconds.

## Install

```bash
pip3.11 install "extremis[mcp]"
```

## Configure

Add to `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "extremis": {
      "command": "/opt/homebrew/bin/extremis-mcp",
      "env": {
        "EXTREMIS_HOME": "~/.extremis"
      }
    }
  }
}
```

!!! tip "Full path required"
    Claude Desktop doesn't inherit your shell PATH. Use the full path to `extremis-mcp`:
    ```bash
    which extremis-mcp
    # → /opt/homebrew/bin/extremis-mcp
    ```

## Restart Claude Desktop

Quit (⌘Q) and reopen. The 10 memory tools appear in the tool picker.

## Test it

Paste this into Claude:

> Use `memory_remember_now` to store that my name is [your name] in the semantic layer. Then use `memory_recall` with query "what is my name?" to verify it was stored.

## Claude Code

```bash
claude mcp add extremis /opt/homebrew/bin/extremis-mcp \
  --env EXTREMIS_HOME=~/.extremis
```

## SSE / HTTP mode

Run extremis as an HTTP server instead of stdio — useful for non-Claude integrations:

```bash
extremis-mcp --transport sse --port 8765
```

## Prompt

The server instructs Claude to:

- Call `memory_recall` at the start of every conversation
- Call `memory_remember` after learning something durable
- Call `memory_report_outcome` when the user rates a response
- Call `memory_consolidate` every ~20 conversations

You don't need to instruct Claude manually — these are in the server instructions.

## Tools reference

See [Tools reference](tools.md) for all 10 tools.
