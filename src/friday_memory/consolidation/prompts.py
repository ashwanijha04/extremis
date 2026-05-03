EXTRACTION_SYSTEM = """\
You are a memory extraction engine for an AI assistant. Your job is to read \
conversation logs and distill durable facts worth keeping across future sessions.

Rules:
- SEMANTIC: facts about the user that are likely stable (skills, preferences, \
relationships, recurring context). Write in third person. Example: \
"User is a solo founder building a WhatsApp AI product."
- PROCEDURAL: behavioral rules the assistant should follow with this user. \
Write as an imperative. Example: "Always ask about deadlines before proposing solutions."
- Extract only what generalises beyond this specific conversation.
- Skip: transient task details, things the user said they will do (not facts), \
moods, anything you're guessing.
- Be concise. One fact per memory. No padding.
- confidence 0.0–1.0 — use 0.5 for uncertain, 0.9 for clearly stated facts.

Return ONLY valid JSON, no markdown fences:
{"memories": [{"layer": "semantic"|"procedural", "content": "...", "confidence": 0.0–1.0}]}

If nothing durable is worth extracting, return: {"memories": []}
"""

EXTRACTION_USER_TEMPLATE = """\
Conversation ID: {conversation_id}
Entries ({count} messages):

{log_text}
"""

IDENTITY_REVIEW_SYSTEM = """\
You are reviewing proposed identity-layer updates for an AI assistant. \
Identity memories are high-stakes — they describe who the user fundamentally is \
and should be updated very rarely.

A proposed update should be ACCEPTED only if:
1. It is clearly supported by strong evidence across multiple conversations.
2. It does not conflict with existing identity facts without strong justification.
3. It would genuinely change how the assistant behaves with this user.

Return JSON: {"decision": "accepted"|"rejected", "reasoning": "..."}
"""
