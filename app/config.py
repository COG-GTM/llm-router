import os

_DEFAULT = "http://localhost:9001,http://localhost:9002,http://localhost:9003"


def upstream_urls() -> list[str]:
    raw = os.environ.get("LLM_SERVICE_URLS", _DEFAULT)
    return [u.strip() for u in raw.split(",") if u.strip()]


def hedge_delay_s() -> float:
    return float(os.environ.get("UPSTREAM_HEDGE_DELAY_MS", 150)) / 1000


def max_attempts() -> int:
    return max(1, int(os.environ.get("UPSTREAM_MAX_ATTEMPTS", 4)))


def upstream_timeout_s() -> float:
    return float(os.environ.get("UPSTREAM_TIMEOUT_S", 30))
