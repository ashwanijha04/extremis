# Security Policy

## Supported versions

| Version | Supported |
|---------|-----------|
| 0.1.x   | ✅        |

## Reporting a vulnerability

**Please do not report security vulnerabilities via public GitHub issues.**

Email **ashwanijha04@gmail.com** with:
- A description of the vulnerability
- Steps to reproduce
- Potential impact
- Any suggested fix (optional)

You will receive a response within 48 hours. If the issue is confirmed, a patch will
be released as quickly as possible — typically within 7 days for critical issues.

## Scope

Things we care about:
- **Namespace isolation bypasses** — one user reading another user's memories
- **Prompt injection via stored memories** — malicious content in memories affecting agent behaviour
- **Path traversal in the log store** — e.g. writing outside the configured log directory
- **SQL injection in SQLite/Postgres stores** — all queries should use parameterised statements

Out of scope:
- The security of your Anthropic API key (that's between you and Anthropic)
- Attacks that require physical access to the machine running the server
- Denial of service via large memory stores (no current rate limiting)

## Data handling notes

- All memory data is stored locally by default (`~/.friday/`)
- No data is sent anywhere except Anthropic's API during consolidation (opt-in)
- The JSONL log files are plaintext — protect them with filesystem permissions if sensitive
