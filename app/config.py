import os

_DEFAULT = "http://localhost:9001,http://localhost:9002,http://localhost:9003"


def upstream_urls() -> list[str]:
    raw = os.environ.get("LLM_SERVICE_URLS", _DEFAULT)
    return [u.strip() for u in raw.split(",") if u.strip()]


def hedge_delay_s() -> float:
    """Delay before a hedged request is sent to a second backend; <=0 disables."""
    return float(os.environ.get("LLM_HEDGE_DELAY_MS", "300")) / 1000
