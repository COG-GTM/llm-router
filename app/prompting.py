"""Prompt templates and conversation-history flattening."""

import os

DEFAULT_HISTORY_TURNS = 10

SYSTEM_PREAMBLE = (
    "You are a concise, helpful assistant. Answer directly and keep "
    "responses focused on the question asked."
)


def history_message_limit() -> int:
    """Max prior messages sent upstream: CHAT_HISTORY_TURNS turns (user + assistant)."""
    try:
        turns = int(os.environ.get("CHAT_HISTORY_TURNS", DEFAULT_HISTORY_TURNS))
    except ValueError:
        turns = DEFAULT_HISTORY_TURNS
    if turns < 1:
        turns = DEFAULT_HISTORY_TURNS
    return turns * 2


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
