import asyncio

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
