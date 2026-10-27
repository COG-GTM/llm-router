import math
import os

_DEFAULT = "http://localhost:9001,http://localhost:9002,http://localhost:9003"
_DEFAULT_TIMEOUT_S = 0.25
_DEFAULT_CONNECT_TIMEOUT_S = 0.1


def upstream_urls() -> list[str]:
    raw = os.environ.get("LLM_SERVICE_URLS", _DEFAULT)
    return [u.strip() for u in raw.split(",") if u.strip()]


def _env_seconds(name: str, default: float) -> float:
    try:
        value = float(os.environ.get(name, default))
    except ValueError:
        return default
    return value if math.isfinite(value) and value > 0 else default


def upstream_timeout_s() -> float:
    return _env_seconds("UPSTREAM_TIMEOUT_S", _DEFAULT_TIMEOUT_S)


def upstream_connect_timeout_s() -> float:
    return _env_seconds("UPSTREAM_CONNECT_TIMEOUT_S", _DEFAULT_CONNECT_TIMEOUT_S)


def upstream_max_retries(default: int) -> int:
    try:
        value = int(os.environ.get("UPSTREAM_MAX_RETRIES", default))
    except ValueError:
        return default
    return value if value >= 0 else default
