import os

_DEFAULT = "http://localhost:9001,http://localhost:9002,http://localhost:9003"


def upstream_urls() -> list[str]:
    raw = os.environ.get("LLM_SERVICE_URLS", _DEFAULT)
    return [u.strip() for u in raw.split(",") if u.strip()]


DEFAULT_MAX_TOKENS = 64
_DEFAULT_MAX_TOKENS_LIMIT = 1024


def max_tokens_limit() -> int:
    """Upper bound on client-supplied `max_tokens` (env `MAX_TOKENS_LIMIT`).

    Non-numeric or non-positive values fall back to the default.
    """
    try:
        limit = int(os.environ.get("MAX_TOKENS_LIMIT", _DEFAULT_MAX_TOKENS_LIMIT))
    except ValueError:
        return _DEFAULT_MAX_TOKENS_LIMIT
    return limit if limit >= 1 else _DEFAULT_MAX_TOKENS_LIMIT
