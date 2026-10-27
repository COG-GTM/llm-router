import math
import os

_DEFAULT = "http://localhost:9001,http://localhost:9002,http://localhost:9003"
_DEFAULT_TIMEOUT_S = 0.25
_DEFAULT_CONNECT_TIMEOUT_S = 0.1


def upstream_urls() -> list[str]:
    raw = os.environ.get("LLM_SERVICE_URLS", _DEFAULT)
    return [u.strip() for u in raw.split(",") if u.strip()]


def _env_float(name: str, default: float) -> float:
    try:
        value = float(os.environ.get(name, default))
    except ValueError:
        return default
    return value if math.isfinite(value) and value > 0 else default


def upstream_slow_s() -> float:
    return _env_float("UPSTREAM_SLOW_S", 1.0)


def cb_failure_threshold() -> int:
    return int(_env_float("CB_FAILURE_THRESHOLD", 3)) or 3


def cb_cooldown_s() -> float:
    return _env_float("CB_COOLDOWN_S", 5.0)


def cb_max_cooldown_s() -> float:
    return _env_float("CB_MAX_COOLDOWN_S", 60.0)


def upstream_timeout_s() -> float:
    return _env_float("UPSTREAM_TIMEOUT_S", _DEFAULT_TIMEOUT_S)


def upstream_connect_timeout_s() -> float:
    return _env_float("UPSTREAM_CONNECT_TIMEOUT_S", _DEFAULT_CONNECT_TIMEOUT_S)


def upstream_max_retries(default: int) -> int:
    try:
        value = int(os.environ.get("UPSTREAM_MAX_RETRIES", default))
    except ValueError:
        return default
    return value if value >= 0 else default
