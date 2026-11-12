import asyncio

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


def _recording_transport(hits: list[str]) -> httpx.MockTransport:
    async def handler(request: httpx.Request) -> httpx.Response:
        hits.append(f"{request.url.scheme}://{request.url.host}:{request.url.port}")
        return httpx.Response(200, json={"completion": "ok"})

    return httpx.MockTransport(handler)


def test_round_robin_and_passthrough():
    hits: list[str] = []
    urls = ["http://u1:9000", "http://u2:9000", "http://u3:9000"]
    pool = UpstreamPool(urls=urls, transport=_recording_transport(hits), hedge_delay=0)

    async def run():
        results = [await pool.forward({"prompt": "p"}) for _ in range(4)]
        await pool.aclose()
        return results

    results = asyncio.run(run())
    assert hits == ["http://u1:9000", "http://u2:9000", "http://u3:9000", "http://u1:9000"]
    assert all(r == (200, {"completion": "ok"}) for r in results)


def test_hedge_delay_env(monkeypatch):
    monkeypatch.delenv("LLM_HEDGE_DELAY_MS", raising=False)
    assert hedge_delay_s() == 0.3
    monkeypatch.setenv("LLM_HEDGE_DELAY_MS", "150")
    assert hedge_delay_s() == 0.15


def _delayed_transport(delays: dict[str, float], hits: list[str],
                       cancelled: list[str] | None = None,
                       failing: set[str] = frozenset()) -> httpx.MockTransport:
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
            raise httpx.ConnectError("boom", request=request)
        return httpx.Response(200, json={"completion": host})

    return httpx.MockTransport(handler)


URLS = ["http://u1:9000", "http://u2:9000", "http://u3:9000"]


def test_fast_primary_is_not_hedged():
    hits: list[str] = []
    pool = UpstreamPool(urls=URLS, transport=_delayed_transport({}, hits), hedge_delay=0.05)

    async def run():
        result = await pool.forward({"prompt": "p"})
        await pool.aclose()
        return result

    assert asyncio.run(run()) == (200, {"completion": "u1"})
    assert hits == ["u1"]
    assert (pool.requests, pool.hedged, pool.hedge_wins) == (1, 0, 0)


def test_slow_primary_is_hedged_to_next_backend_and_loser_cancelled():
    hits: list[str] = []
    cancelled: list[str] = []
    pool = UpstreamPool(urls=URLS, transport=_delayed_transport({"u1": 5}, hits, cancelled),
                        hedge_delay=0.05)

    async def run():
        result = await pool.forward({"prompt": "p"})
        await asyncio.sleep(0)
        await pool.aclose()
        return result

    assert asyncio.run(run()) == (200, {"completion": "u2"})
    assert hits == ["u1", "u2"]
    assert cancelled == ["u1"]
    assert (pool.requests, pool.hedged, pool.hedge_wins) == (1, 1, 1)


def test_hedging_keeps_round_robin_for_primary():
    hits: list[str] = []
    pool = UpstreamPool(urls=URLS, transport=_delayed_transport({"u1": 5}, hits),
                        hedge_delay=0.05)

    async def run():
        results = [await pool.forward({"prompt": "p"}) for _ in range(3)]
        await pool.aclose()
        return results

    results = asyncio.run(run())
    assert [r[1]["completion"] for r in results] == ["u2", "u2", "u3"]
    assert hits == ["u1", "u2", "u2", "u3"]


def test_primary_wins_if_it_finishes_before_hedge():
    hits: list[str] = []
    pool = UpstreamPool(urls=URLS, transport=_delayed_transport({"u1": 0.1, "u2": 5}, hits),
                        hedge_delay=0.05)

    async def run():
        result = await pool.forward({"prompt": "p"})
        await pool.aclose()
        return result

    assert asyncio.run(run()) == (200, {"completion": "u1"})
    assert (pool.hedged, pool.hedge_wins) == (1, 0)


def test_failed_hedge_falls_back_to_slow_primary():
    hits: list[str] = []
    pool = UpstreamPool(urls=URLS,
                        transport=_delayed_transport({"u1": 0.2}, hits, failing={"u2"}),
                        hedge_delay=0.05)

    async def run():
        result = await pool.forward({"prompt": "p"})
        await pool.aclose()
        return result

    assert asyncio.run(run()) == (200, {"completion": "u1"})


def test_both_attempts_failing_raises():
    hits: list[str] = []
    pool = UpstreamPool(urls=URLS,
                        transport=_delayed_transport({"u1": 0.2}, hits, failing={"u1", "u2"}),
                        hedge_delay=0.05)

    async def run():
        try:
            await pool.forward({"prompt": "p"})
        finally:
            await pool.aclose()

    with pytest.raises(httpx.ConnectError):
        asyncio.run(run())
