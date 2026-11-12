import asyncio
import random
from collections import Counter

import httpx

from app.config import upstream_urls
from app.upstream import UpstreamPool


def test_upstream_urls_default(monkeypatch):
    monkeypatch.delenv("LLM_SERVICE_URLS", raising=False)
    assert upstream_urls() == [
        "http://localhost:9001",
        "http://localhost:9002",
        "http://localhost:9003",
    ]


def test_upstream_urls_env(monkeypatch):
    monkeypatch.setenv("LLM_SERVICE_URLS", "http://a:1, http://b:2")
    assert upstream_urls() == ["http://a:1", "http://b:2"]


class _FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def _latency_transport(clock: _FakeClock, latency_ms: dict[str, float],
                       hits: list[str], status: dict[str, int] | None = None) -> httpx.MockTransport:
    async def handler(request: httpx.Request) -> httpx.Response:
        host = request.url.host
        hits.append(host)
        clock.now += latency_ms[host] / 1000
        code = (status or {}).get(host, 200)
        return httpx.Response(code, json={"completion": host} if code == 200 else {"detail": "boom"})

    return httpx.MockTransport(handler)


def _pool(clock, latency, hits, status=None, **kw) -> UpstreamPool:
    urls = [f"http://{h}:9000" for h in latency]
    return UpstreamPool(urls=urls, transport=_latency_transport(clock, latency, hits, status),
                        clock=clock, rng=random.Random(0), **kw)


def _drive(pool: UpstreamPool, n: int) -> list[tuple[int, dict]]:
    async def run():
        return [await pool.forward({"prompt": "p"}) for _ in range(n)]

    return asyncio.run(run())


def test_passthrough_and_spreads_over_equal_backends():
    clock, hits = _FakeClock(), []
    pool = _pool(clock, {"u1": 100, "u2": 100, "u3": 100}, hits)
    results = _drive(pool, 300)
    asyncio.run(pool.aclose())
    assert all(status == 200 and body["completion"] in {"u1", "u2", "u3"}
               for status, body in results)
    counts = Counter(hits)
    assert set(counts) == {"u1", "u2", "u3"}
    assert min(counts.values()) > 50


def test_degraded_backend_is_ejected_and_mostly_avoided():
    clock, hits = _FakeClock(), []
    pool = _pool(clock, {"u1": 120, "u2": 140, "u3": 1350}, hits)
    _drive(pool, 600)
    counts = Counter(hits)
    # Only a handful of initial samples + periodic probes ever reach u3.
    assert counts["u3"] < 15
    assert next(s for s in pool.stats() if "u3" in s["url"])["ejected"]


def test_ejected_backend_is_reprobed_and_reinstated_after_recovery():
    clock, hits = _FakeClock(), []
    latency = {"u1": 120, "u2": 140, "u3": 1350}
    pool = _pool(clock, latency, hits)
    _drive(pool, 200)
    assert next(s for s in pool.stats() if "u3" in s["url"])["ejected"]

    latency["u3"] = 110
    clock.now += 120  # longer than the max ejection window
    hits.clear()
    _drive(pool, 300)
    assert not next(s for s in pool.stats() if "u3" in s["url"])["ejected"]
    assert Counter(hits)["u3"] > 50


def test_consecutive_5xx_ejects_backend():
    clock, hits = _FakeClock(), []
    pool = _pool(clock, {"u1": 100, "u2": 100, "u3": 100}, hits, status={"u2": 503})
    results = _drive(pool, 300)
    assert Counter(hits)["u2"] < 15
    assert sum(1 for status, _ in results if status == 503) == Counter(hits)["u2"]


def test_never_ejects_more_than_allowed_fraction():
    clock, hits = _FakeClock(), []
    pool = _pool(clock, {"u1": 100, "u2": 2000, "u3": 2000}, hits)
    _drive(pool, 200)
    assert sum(s["ejected"] for s in pool.stats()) <= 1


def test_single_backend_always_used():
    clock, hits = _FakeClock(), []
    pool = _pool(clock, {"u1": 5000}, hits)
    _drive(pool, 20)
    assert hits == ["u1"] * 20
