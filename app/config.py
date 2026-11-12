import os

_DEFAULT = "http://localhost:9001,http://localhost:9002,http://localhost:9003"


def upstream_urls() -> list[str]:
    raw = os.environ.get("LLM_SERVICE_URLS", _DEFAULT)
    return [u.strip() for u in raw.split(",") if u.strip()]


def upstream_timeout() -> float:
    """Seconds allowed for an upstream attempt that can still be retried elsewhere."""
    return float(os.environ.get("LLM_UPSTREAM_TIMEOUT_S", "0.5"))


def upstream_retry_timeout() -> float:
    """Seconds allowed for the final (retry) attempt; a safety cap, not a latency target."""
    return float(os.environ.get("LLM_UPSTREAM_RETRY_TIMEOUT_S", "10"))
