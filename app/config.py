import os

_DEFAULT = "http://localhost:9001,http://localhost:9002,http://localhost:9003"


def upstream_urls() -> list[str]:
    raw = os.environ.get("LLM_SERVICE_URLS", _DEFAULT)
    return [u.strip() for u in raw.split(",") if u.strip()]


def cache_max_size() -> int:
    return int(os.environ.get("UPSTREAM_CACHE_SIZE", "1024"))


def cache_ttl_s() -> float:
    return float(os.environ.get("UPSTREAM_CACHE_TTL_S", "300"))
