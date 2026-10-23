import os
from dataclasses import dataclass

_DEFAULT = "http://localhost:9001,http://localhost:9002,http://localhost:9003"


def upstream_urls() -> list[str]:
    raw = os.environ.get("LLM_SERVICE_URLS", _DEFAULT)
    return [u.strip() for u in raw.split(",") if u.strip()]


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class UpstreamSettings:
    attempt_timeout_s: float
    final_timeout_s: float
    max_attempts: int  # 0 = one attempt per backend
    breaker_threshold: int
    breaker_cooldown_s: float
    breaker_max_cooldown_s: float
    reprobe_after_s: float


def upstream_settings() -> UpstreamSettings:
    return UpstreamSettings(
        attempt_timeout_s=_env_int("UPSTREAM_ATTEMPT_TIMEOUT_MS", 250) / 1000,
        final_timeout_s=_env_int("UPSTREAM_FINAL_TIMEOUT_MS", 30000) / 1000,
        max_attempts=_env_int("UPSTREAM_MAX_ATTEMPTS", 0),
        breaker_threshold=max(1, _env_int("UPSTREAM_BREAKER_THRESHOLD", 3)),
        breaker_cooldown_s=_env_int("UPSTREAM_BREAKER_COOLDOWN_MS", 1000) / 1000,
        breaker_max_cooldown_s=_env_int("UPSTREAM_BREAKER_MAX_COOLDOWN_MS", 30000) / 1000,
        reprobe_after_s=_env_int("UPSTREAM_REPROBE_AFTER_MS", 10000) / 1000,
    )
