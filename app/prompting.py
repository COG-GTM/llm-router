"""Prompt templates and conversation-history flattening."""

# Real LLM latency grows with prompt length, so `/v1/chat` only sends the
# most recent slice of a conversation rather than its entire history.
HISTORY_MAX_MESSAGES = 40
HISTORY_TOKEN_BUDGET = 2000

SYSTEM_PREAMBLE = (
    "You are a concise, helpful assistant. Answer directly and keep "
    "responses focused on the question asked."
)


def flatten_history(messages: list[dict], new_message: str) -> str:
    lines = [SYSTEM_PREAMBLE, ""]
    for message in messages:
        speaker = "User" if message["role"] == "user" else "Assistant"
        lines.append(f"{speaker}: {message['content']}")
    lines.append(f"User: {new_message}")
    lines.append("Assistant:")
    return "\n".join(lines)


def token_estimate(text: str) -> int:
    """Rough token count: ~0.75 words per token. An estimate, not a tokenizer."""
    return max(1, round(len(text.split()) / 0.75))


def trim_history(messages: list[dict], max_messages: int = HISTORY_MAX_MESSAGES,
                 token_budget: int = HISTORY_TOKEN_BUDGET) -> list[dict]:
    """Keep the most recent messages that fit in `max_messages` and
    `token_budget` (per `token_estimate`), preserving chronological order."""
    kept: list[dict] = []
    used = 0
    for message in reversed(messages[-max_messages:] if max_messages > 0 else []):
        cost = token_estimate(message["content"])
        if used + cost > token_budget:
            break
        kept.append(message)
        used += cost
    kept.reverse()
    return kept
