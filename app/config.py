import math
import os

_DEFAULT = "http://localhost:9001,http://localhost:9002,http://localhost:9003"


def upstream_urls() -> list[str]:
    raw = os.environ.get("LLM_SERVICE_URLS", _DEFAULT)
    return [u.strip() for u in raw.split(",") if u.strip()]


def _env_number(name: str, default, cast):
    try:
        value = cast(os.environ.get(name, default))
    except ValueError:
        return default
    return value if value > 0 and math.isfinite(value) else default


def upstream_attempt_timeout() -> float:
    """Seconds before an attempt is abandoned and retried on another replica."""
    return _env_number("UPSTREAM_ATTEMPT_TIMEOUT_S", 0.25, float)


def upstream_timeout() -> float:
    """Seconds allowed for the final attempt, when no replica is left to fail over to."""
    return _env_number("UPSTREAM_TIMEOUT_S", 10.0, float)


def upstream_cache_size() -> int:
    return _env_number("UPSTREAM_CACHE_SIZE", 1024, int)
