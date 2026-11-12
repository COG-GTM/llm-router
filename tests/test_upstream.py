import asyncio
import random
from collections import Counter

import httpx
import pytest

from app.config import hedge_delay_s, upstream_urls
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


def test_hedge_delay_env(monkeypatch):
    monkeypatch.delenv("LLM_HEDGE_DELAY_MS", raising=False)
    assert hedge_delay_s() == 0.3
    monkeypatch.setenv("LLM_HEDGE_DELAY_MS", "150")
    assert hedge_delay_s() == 0.15


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
    kw.setdefault("hedge_delay", 0)
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
    assert counts["u3"] < 15
    assert next(s for s in pool.stats() if "u3" in s["url"])["ejected"]


def test_ejected_backend_is_reprobed_and_reinstated_after_recovery():
    clock, hits = _FakeClock(), []
    latency = {"u1": 120, "u2": 140, "u3": 1350}
    pool = _pool(clock, latency, hits)
    _drive(pool, 200)
    assert next(s for s in pool.stats() if "u3" in s["url"])["ejected"]

    latency["u3"] = 110
    clock.now += 120
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


class _ReverseFixedRng:
    def sample(self, values, count):
        assert count == 2
        return [values[-1], values[0]]


def _delayed_transport(delays: dict[str, float], hits: list[str],
                       cancelled: list[str] | None = None,
                       failing: set[str] | dict[str, str] = frozenset()) -> httpx.MockTransport:
    async def handler(request: httpx.Request) -> httpx.Response:
        host = request.url.host
        hits.append(host)
        try:
            await asyncio.sleep(delays.get(host, 0))
        except asyncio.CancelledError:
            if cancelled is not None:
                cancelled.append(host)
            raise
        if host in failing:
            message = failing[host] if isinstance(failing, dict) else "boom"
            raise httpx.ConnectError(message, request=request)
        return httpx.Response(200, json={"completion": host})

    return httpx.MockTransport(handler)


URLS = ["http://u1:9000", "http://u2:9000", "http://u3:9000"]


def test_fast_primary_is_not_hedged():
    hits: list[str] = []
    pool = UpstreamPool(urls=URLS, transport=_delayed_transport({}, hits),
                        rng=_ReverseFixedRng(), hedge_delay=0.05)

    async def run():
        result = await pool.forward({"prompt": "p"})
        await pool.aclose()
        return result

    result = asyncio.run(run())
    assert result == (200, {"completion": hits[0]})
    assert len(hits) == 1
    assert pool.hedge_stats() == {"requests": 1, "hedged": 0, "hedge_wins": 0}


def test_primary_5xx_is_passed_through_without_hedging():
    hits: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        hits.append(request.url.host)
        return httpx.Response(503, json={"detail": "unavailable"})

    pool = UpstreamPool(urls=URLS, transport=httpx.MockTransport(handler),
                        rng=_ReverseFixedRng(), hedge_delay=0.05)

    async def run():
        result = await pool.forward({"prompt": "p"})
        await pool.aclose()
        return result

    assert asyncio.run(run()) == (503, {"detail": "unavailable"})
    assert len(hits) == 1
    assert pool.hedge_stats() == {"requests": 1, "hedged": 0, "hedge_wins": 0}


def test_slow_primary_is_hedged_to_an_alternative_and_loser_cancelled():
    hits: list[str] = []
    cancelled: list[str] = []
    pool = UpstreamPool(
        urls=URLS,
        transport=_delayed_transport({"u1": 0.01, "u2": 0.01, "u3": 0.2},
                                     hits, cancelled),
        rng=_ReverseFixedRng(),
        hedge_delay=0.05,
    )

    async def run():
        result = await pool.forward({"prompt": "p"})
        await pool.aclose()
        return result

    assert asyncio.run(run()) == (200, {"completion": "u1"})
    assert hits == ["u3", "u1"]
    assert cancelled == ["u3"]
    assert pool.hedge_stats() == {"requests": 1, "hedged": 1, "hedge_wins": 1}


def test_primary_can_win_after_hedge_starts():
    hits: list[str] = []
    cancelled: list[str] = []
    pool = UpstreamPool(
        urls=URLS,
        transport=_delayed_transport({"u1": 0.2, "u3": 0.05}, hits, cancelled),
        rng=_ReverseFixedRng(),
        hedge_delay=0.02,
    )

    async def run():
        result = await pool.forward({"prompt": "p"})
        await pool.aclose()
        return result

    assert asyncio.run(run()) == (200, {"completion": "u3"})
    assert hits == ["u3", "u1"]
    assert cancelled == ["u1"]
    assert (pool.hedged, pool.hedge_wins) == (1, 0)


def test_cancelled_loser_records_elapsed_sample_and_releases_inflight():
    hits: list[str] = []
    cancelled: list[str] = []
    pool = UpstreamPool(
        urls=URLS,
        transport=_delayed_transport({"u1": 0.01, "u3": 0.2}, hits, cancelled),
        rng=_ReverseFixedRng(),
        hedge_delay=0.03,
    )

    async def run():
        await pool.forward({"prompt": "p"})
        await pool.aclose()

    asyncio.run(run())
    slow = next(b for b in pool._backends if b.url.endswith("u3:9000"))
    assert cancelled == ["u3"]
    assert slow.samples == 1
    assert slow.ewma_ms is not None and 25 <= slow.ewma_ms <= 150
    assert slow.inflight == 0


def test_cancelled_slow_backend_is_ejected_after_repeated_hedge_losses():
    hits: list[str] = []
    pool = UpstreamPool(
        urls=URLS,
        transport=_delayed_transport({"u1": 0.02, "u2": 0.02, "u3": 0.5}, hits),
        rng=_ReverseFixedRng(),
        hedge_delay=0.05,
        outlier_ratio=2,
        outlier_floor_ms=10,
        min_samples=3,
        stale_after_s=0.1,
    )
    for backend in pool._backends[:2]:
        backend.ewma_ms = 20
        backend.samples = 3
    for backend in pool._backends:
        backend.last_picked = pool._clock()

    async def run():
        for _ in range(10):
            await pool.forward({"prompt": "p"})
        await pool.aclose()

    asyncio.run(run())
    slow = next(b for b in pool._backends if b.url.endswith("u3:9000"))
    assert slow.samples >= 3
    assert slow.ejected_until > pool._clock()
    assert Counter(hits)["u3"] == 3


def test_hedge_target_excludes_primary_ejected_and_probing_and_picks_lowest_score():
    hits: list[str] = []
    urls = [f"http://u{i}:9000" for i in range(1, 6)]
    pool = UpstreamPool(
        urls=urls,
        transport=_delayed_transport({"u1": 0.2, "u4": 0.01}, hits),
        rng=_ReverseFixedRng(),
        hedge_delay=0.02,
        min_samples=100,
    )
    now = pool._clock()
    for backend, score in zip(pool._backends, [1, 2, 3, 20, 30]):
        backend.ewma_ms = score
        backend.last_picked = now
    pool._backends[1].ejected_until = now + 10
    pool._backends[2].probing = True

    async def run():
        result = await pool.forward({"prompt": "p"})
        await pool.aclose()
        return result

    assert asyncio.run(run()) == (200, {"completion": "u4"})
    assert hits == ["u1", "u4"]
    assert pool._backends[2].probing


def test_does_not_hedge_when_no_eligible_alternative_exists():
    hits: list[str] = []
    urls = ["http://u1:9000", "http://u2:9000"]
    pool = UpstreamPool(
        urls=urls,
        transport=_delayed_transport({"u1": 0.08}, hits),
        rng=_ReverseFixedRng(),
        hedge_delay=0.02,
    )
    pool._backends[1].ejected_until = pool._clock() + 10

    async def run():
        result = await pool.forward({"prompt": "p"})
        await pool.aclose()
        return result

    assert asyncio.run(run()) == (200, {"completion": "u1"})
    assert hits == ["u1"]
    assert (pool.hedged, pool.hedge_wins) == (0, 0)


def test_failed_hedge_falls_back_to_primary():
    hits: list[str] = []
    pool = UpstreamPool(
        urls=URLS,
        transport=_delayed_transport({"u1": 0.01, "u3": 0.08}, hits,
                                     failing={"u1"}),
        rng=_ReverseFixedRng(),
        hedge_delay=0.02,
    )

    async def run():
        result = await pool.forward({"prompt": "p"})
        await pool.aclose()
        return result

    assert asyncio.run(run()) == (200, {"completion": "u3"})
    assert hits == ["u3", "u1"]


def test_failed_primary_falls_back_to_hedge():
    hits: list[str] = []
    pool = UpstreamPool(
        urls=URLS,
        transport=_delayed_transport({"u1": 0.01, "u3": 0.1}, hits,
                                     failing={"u3"}),
        rng=_ReverseFixedRng(),
        hedge_delay=0.02,
    )

    async def run():
        result = await pool.forward({"prompt": "p"})
        await pool.aclose()
        return result

    assert asyncio.run(run()) == (200, {"completion": "u1"})
    assert hits == ["u3", "u1"]


def test_both_attempts_failing_raises_primary_error():
    hits: list[str] = []
    pool = UpstreamPool(
        urls=URLS,
        transport=_delayed_transport({"u1": 0.03, "u3": 0.08}, hits,
                                     failing={"u1": "hedge", "u3": "primary"}),
        rng=_ReverseFixedRng(),
        hedge_delay=0.02,
    )

    async def run():
        try:
            await pool.forward({"prompt": "p"})
        finally:
            await pool.aclose()

    with pytest.raises(httpx.ConnectError, match="primary"):
        asyncio.run(run())


def test_pool_hedge_stats_do_not_change_backend_stats_shape():
    clock, hits = _FakeClock(), []
    pool = _pool(clock, {"u1": 100, "u2": 100}, hits)
    _drive(pool, 1)
    assert set(pool.stats()[0]) == {"url", "ewma_ms", "inflight", "ejected"}
    assert pool.hedge_stats() == {"requests": 1, "hedged": 0, "hedge_wins": 0}
