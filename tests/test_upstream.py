import asyncio
import time

import httpx
from fastapi.testclient import TestClient

from app import main
from app.config import upstream_urls
from app.generate_cache import ResponseCache
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


def _recording_transport(hits: list[str]) -> httpx.MockTransport:
    async def handler(request: httpx.Request) -> httpx.Response:
        hits.append(f"{request.url.scheme}://{request.url.host}:{request.url.port}")
        return httpx.Response(200, json={"completion": "ok"})

    return httpx.MockTransport(handler)


def test_round_robin_and_passthrough():
    hits: list[str] = []
    urls = ["http://u1:9000", "http://u2:9000", "http://u3:9000"]
    pool = UpstreamPool(urls=urls, transport=_recording_transport(hits))

    async def run():
        results = [await pool.forward({"prompt": "p"}) for _ in range(4)]
        await pool.aclose()
        return results

    results = asyncio.run(run())
    assert hits[:3] == urls
    assert hits[3] in urls
    assert all(r == (200, {"completion": "ok"}) for r in results)


def _delayed_transport(delays: dict[str, float], hits: list[str],
                       statuses: dict[str, int] | None = None) -> httpx.MockTransport:
    async def handler(request: httpx.Request) -> httpx.Response:
        host = request.url.host
        hits.append(host)
        await asyncio.sleep(delays[host])
        return httpx.Response((statuses or {}).get(host, 200), json={"completion": host})

    return httpx.MockTransport(handler)


def test_round_robin_fallback_until_every_replica_has_a_sample():
    hits: list[str] = []
    urls = ["http://u1:9000", "http://u2:9000", "http://u3:9000"]
    transport = _delayed_transport({"u1": 0.0, "u2": 0.0, "u3": 0.0}, hits)
    pool = UpstreamPool(urls=urls, transport=transport)

    async def run():
        for _ in range(3):
            await pool.forward({"prompt": "p"})
        await pool.aclose()

    asyncio.run(run())
    assert hits == ["u1", "u2", "u3"]


def test_prefers_fastest_replica_once_stats_exist():
    hits: list[str] = []
    urls = ["http://u1:9000", "http://u2:9000", "http://u3:9000"]
    transport = _delayed_transport({"u1": 0.05, "u2": 0.001, "u3": 0.1}, hits)
    pool = UpstreamPool(urls=urls, transport=transport)

    async def run():
        results = [await pool.forward({"prompt": "p"}) for _ in range(8)]
        await pool.aclose()
        return results

    results = asyncio.run(run())
    assert hits[:3] == ["u1", "u2", "u3"]
    assert hits[3:] == ["u2"] * 5
    assert results[-1] == (200, {"completion": "u2"})


def test_failed_replica_is_penalized_and_avoided():
    hits: list[str] = []
    urls = ["http://u1:9000", "http://u2:9000"]
    transport = _delayed_transport({"u1": 0.02, "u2": 0.0}, hits, statuses={"u2": 503})
    pool = UpstreamPool(urls=urls, transport=transport, failure_penalty=5.0)

    async def run():
        results = [await pool.forward({"prompt": "p"}) for _ in range(4)]
        await pool.aclose()
        return results

    results = asyncio.run(run())
    assert results[1] == (503, {"completion": "u2"})
    assert pool._ewma["http://u2:9000"] == 5.0
    assert hits == ["u1", "u2", "u1", "u1"]


def test_transport_error_is_penalized_and_reraised():
    async def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("down", request=request)

    pool = UpstreamPool(urls=["http://u1:9000"], transport=httpx.MockTransport(handler),
                        failure_penalty=5.0)

    async def run():
        try:
            await pool.forward({"prompt": "p"})
        finally:
            await pool.aclose()

    try:
        asyncio.run(run())
    except httpx.ConnectError:
        pass
    else:
        raise AssertionError("expected ConnectError")
    assert pool._ewma["http://u1:9000"] == 5.0


def test_ewma_update_math():
    pool = UpstreamPool(urls=["http://u1:9000"], alpha=0.25)
    pool._record("http://u1:9000", 1.0)
    assert pool._ewma["http://u1:9000"] == 1.0
    pool._record("http://u1:9000", 3.0)
    assert pool._ewma["http://u1:9000"] == 0.25 * 3.0 + 0.75 * 1.0
    pool._record("http://u1:9000", 0.0)
    assert abs(pool._ewma["http://u1:9000"] - 0.75 * 1.5) < 1e-12
    asyncio.run(pool.aclose())


def test_unsampled_replica_gets_one_probe_while_others_route_by_ewma():
    hits: list[str] = []
    urls = ["http://u1:9000", "http://u2:9000"]
    transport = _delayed_transport({"u1": 0.0, "u2": 0.2}, hits)
    pool = UpstreamPool(urls=urls, transport=transport)

    async def run():
        await pool.forward({"prompt": "p"})  # u1 sampled
        slow_probe = asyncio.create_task(pool.forward({"prompt": "p"}))  # u2 probe in flight
        await asyncio.sleep(0.01)
        for _ in range(3):
            await pool.forward({"prompt": "p"})
        await slow_probe
        await pool.aclose()

    asyncio.run(run())
    assert hits == ["u1", "u2", "u1", "u1", "u1"]


def test_stale_replica_is_reprobed_after_probe_interval():
    hits: list[str] = []
    statuses = {"u2": 503}
    urls = ["http://u1:9000", "http://u2:9000"]
    transport = _delayed_transport({"u1": 0.01, "u2": 0.0}, hits, statuses=statuses)
    pool = UpstreamPool(urls=urls, transport=transport, probe_interval=0.1)

    async def run():
        for _ in range(3):
            await pool.forward({"prompt": "p"})
        statuses.clear()  # u2 recovers
        await asyncio.sleep(0.15)
        for _ in range(2):
            await pool.forward({"prompt": "p"})
        await pool.aclose()

    asyncio.run(run())
    assert hits[:3] == ["u1", "u2", "u1"]
    assert "u2" in hits[3:]
    assert pool._ewma["http://u2:9000"] < 5.0


class _CountingPool:
    def __init__(self, status=200):
        self.status = status
        self.calls: list[dict] = []

    async def forward(self, payload):
        self.calls.append(payload)
        if self.status != 200:
            return self.status, {"detail": "model overloaded"}
        return 200, {
            "completion": f"answer {len(self.calls)}",
            "signature": f"{len(self.calls):064x}",
        }

    async def aclose(self):
        pass


def _generate_client(pool) -> TestClient:
    c = TestClient(main.app)
    c.__enter__()
    main.app.state.pool = pool
    return c


def test_generate_cache_hit_skips_upstream_and_is_byte_identical():
    pool = _CountingPool()
    c = _generate_client(pool)
    r1 = c.post("/v1/generate", json={"prompt": "hi", "max_tokens": 8})
    r2 = c.post("/v1/generate", json={"prompt": "hi", "max_tokens": 8})
    c.__exit__(None, None, None)
    assert r1.status_code == r2.status_code == 200
    assert r1.content == r2.content
    assert r2.headers["content-type"] == r1.headers["content-type"]
    assert len(pool.calls) == 1


def test_generate_cache_keys_on_max_tokens():
    pool = _CountingPool()
    c = _generate_client(pool)
    r1 = c.post("/v1/generate", json={"prompt": "hi", "max_tokens": 8})
    r2 = c.post("/v1/generate", json={"prompt": "hi", "max_tokens": 16})
    c.__exit__(None, None, None)
    assert r1.content != r2.content
    assert pool.calls == [{"prompt": "hi", "max_tokens": 8}, {"prompt": "hi", "max_tokens": 16}]


def test_generate_cache_skips_non_200():
    pool = _CountingPool(status=503)
    c = _generate_client(pool)
    r1 = c.post("/v1/generate", json={"prompt": "hi"})
    pool.status = 200
    r2 = c.post("/v1/generate", json={"prompt": "hi"})
    r3 = c.post("/v1/generate", json={"prompt": "hi"})
    c.__exit__(None, None, None)
    assert r1.status_code == 503
    assert r1.json() == {"detail": "model overloaded"}
    assert r2.status_code == 200
    assert r3.content == r2.content
    assert len(pool.calls) == 2


def test_response_cache_evicts_least_recently_used():
    cache = ResponseCache(maxsize=2)
    cache.put(("a", 1), b"A")
    cache.put(("b", 1), b"B")
    assert cache.get(("a", 1)) == b"A"
    cache.put(("c", 1), b"C")
    assert len(cache) == 2
    assert ("b", 1) not in cache
    assert cache.get(("a", 1)) == b"A"
    assert cache.get(("c", 1)) == b"C"


def test_generate_cache_bounded_at_1000_entries():
    pool = _CountingPool()
    c = _generate_client(pool)
    for i in range(1001):
        c.post("/v1/generate", json={"prompt": f"p{i}"})
    cache = main.app.state.generate_cache
    c.post("/v1/generate", json={"prompt": "p0"})
    c.__exit__(None, None, None)
    assert cache.maxsize == 1000
    assert len(cache) == 1000
    assert len(pool.calls) == 1002


def _seed(pool: UpstreamPool, ewma: dict[str, float]) -> None:
    now = time.monotonic()
    for url, latency in ewma.items():
        pool._ewma[url] = latency
        pool._sampled_at[url] = now


def _hedge_transport(delays: dict[str, float], hits: list[str], cancelled: list[str],
                     statuses: dict[str, int] | None = None,
                     errors: set[str] | None = None) -> httpx.MockTransport:
    async def handler(request: httpx.Request) -> httpx.Response:
        host = request.url.host
        hits.append(host)
        try:
            await asyncio.sleep(delays[host])
        except asyncio.CancelledError:
            cancelled.append(host)
            raise
        if host in (errors or set()):
            raise httpx.ConnectError("down", request=request)
        return httpx.Response((statuses or {}).get(host, 200), json={"completion": host})

    return httpx.MockTransport(handler)


HEDGE_URLS = ["http://u1:9000", "http://u2:9000", "http://u3:9000"]


def _run_hedged(delays, ewma, hedge_delay=0.05, **kw):
    hits: list[str] = []
    cancelled: list[str] = []
    transport = _hedge_transport(delays, hits, cancelled, **kw)
    pool = UpstreamPool(urls=HEDGE_URLS, transport=transport, hedge_delay=hedge_delay)
    _seed(pool, ewma)

    async def run():
        start = asyncio.get_running_loop().time()
        try:
            result = await pool.forward({"prompt": "p"})
        finally:
            elapsed = asyncio.get_running_loop().time() - start
            await pool.aclose()
        return result, elapsed

    result, elapsed = asyncio.run(run())
    return pool, result, elapsed, hits, cancelled


def test_hedge_fires_after_delay_on_slow_primary():
    pool, result, elapsed, hits, _cancelled = _run_hedged(
        {"u1": 1.0, "u2": 0.01, "u3": 0.01},
        {"http://u1:9000": 0.1, "http://u2:9000": 0.2, "http://u3:9000": 0.3})
    assert hits == ["u1", "u2"]
    assert result == (200, {"completion": "u2"})
    assert 0.05 <= elapsed < 0.5
    assert (pool.forwarded, pool.hedges, pool.hedge_wins) == (1, 1, 1)


def test_no_hedge_when_primary_is_fast():
    pool, result, _, hits, cancelled = _run_hedged(
        {"u1": 0.0, "u2": 0.0, "u3": 0.0},
        {"http://u1:9000": 0.1, "http://u2:9000": 0.2, "http://u3:9000": 0.3},
        hedge_delay=0.2)
    assert hits == ["u1"]
    assert cancelled == []
    assert result == (200, {"completion": "u1"})
    assert (pool.hedges, pool.hedge_wins) == (0, 0)


def test_hedge_loser_is_cancelled_and_recorded_as_slow():
    pool, result, _, _hits, cancelled = _run_hedged(
        {"u1": 1.0, "u2": 0.01, "u3": 0.01},
        {"http://u1:9000": 0.01, "http://u2:9000": 0.02, "http://u3:9000": 0.3})
    assert cancelled == ["u1"]
    assert result == (200, {"completion": "u2"})
    # loser's elapsed time (>= hedge delay) is folded into its EWMA
    assert pool._ewma["http://u1:9000"] > 0.01
    assert pool._ewma["http://u2:9000"] < pool._ewma["http://u1:9000"]


def test_primary_wins_if_it_finishes_before_hedge():
    pool, result, _, hits, cancelled = _run_hedged(
        {"u1": 0.08, "u2": 1.0, "u3": 1.0},
        {"http://u1:9000": 0.1, "http://u2:9000": 0.2, "http://u3:9000": 0.3})
    assert hits == ["u1", "u2"]
    assert cancelled == ["u2"]
    assert result == (200, {"completion": "u1"})
    assert (pool.hedges, pool.hedge_wins) == (1, 0)


def test_hedge_contract_prefers_success_over_5xx():
    _pool, result, _, hits, _cancelled = _run_hedged(
        {"u1": 0.15, "u2": 0.0, "u3": 0.0},
        {"http://u1:9000": 0.1, "http://u2:9000": 0.2, "http://u3:9000": 0.3},
        statuses={"u2": 503})
    assert hits == ["u1", "u2"]
    assert result == (200, {"completion": "u1"})
    assert isinstance(result[0], int) and isinstance(result[1], dict)


def test_hedge_contract_returns_5xx_when_both_fail():
    _, result, _, _, _ = _run_hedged(
        {"u1": 0.1, "u2": 0.0, "u3": 0.0},
        {"http://u1:9000": 0.1, "http://u2:9000": 0.2, "http://u3:9000": 0.3},
        statuses={"u1": 503, "u2": 502})
    assert result == (502, {"completion": "u2"})


def test_hedge_survives_primary_transport_error():
    _, result, _, hits, _ = _run_hedged(
        {"u1": 0.1, "u2": 0.0, "u3": 0.0},
        {"http://u1:9000": 0.1, "http://u2:9000": 0.2, "http://u3:9000": 0.3},
        errors={"u1"}, statuses={"u2": 503})
    assert hits == ["u1", "u2"]
    assert result == (503, {"completion": "u2"})


def test_hedging_disabled_with_none():
    pool, result, _, hits, _cancelled = _run_hedged(
        {"u1": 0.1, "u2": 0.0, "u3": 0.0},
        {"http://u1:9000": 0.1, "http://u2:9000": 0.2, "http://u3:9000": 0.3},
        hedge_delay=None)
    assert hits == ["u1"]
    assert result == (200, {"completion": "u1"})
    assert pool.hedges == 0


def test_hedge_loser_never_lowers_its_ewma():
    pool, result, _, _, cancelled = _run_hedged(
        {"u1": 5.0, "u2": 0.01, "u3": 0.01},
        {"http://u1:9000": 2.0, "http://u2:9000": 3.0, "http://u3:9000": 4.0})
    assert cancelled == ["u1"]
    assert result == (200, {"completion": "u2"})
    assert pool._ewma["http://u1:9000"] == 2.0
