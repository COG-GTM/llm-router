import os

_DEFAULT = "http://localhost:9001,http://localhost:9002,http://localhost:9003"


def upstream_urls() -> list[str]:
    raw = os.environ.get("LLM_SERVICE_URLS", _DEFAULT)
    return [u.strip() for u in raw.split(",") if u.strip()]


_DEFAULT_CHAT_HISTORY_TURNS = 10


def chat_history_turns() -> int:
    """How many prior user/assistant turns /v1/chat sends upstream as context.

    Unset or non-integer values fall back to the default; values < 1 clamp to 1.
    """
    try:
        return max(1, int(os.environ.get("CHAT_HISTORY_TURNS", "")))
    except ValueError:
        return _DEFAULT_CHAT_HISTORY_TURNS
