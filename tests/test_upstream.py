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
    assert hits == ["http://u1:9000", "http://u2:9000", "http://u3:9000", "http://u1:9000"]
    assert all(r == (200, {"completion": "ok"}) for r in results)


def _delayed_transport(delays: dict[str, float], hits: list[str]) -> httpx.MockTransport:
    async def handler(request: httpx.Request) -> httpx.Response:
        host = request.url.host
        hits.append(host)
        await asyncio.sleep(delays.get(host, 0.0))
        return httpx.Response(200, json={"completion": host})

    return httpx.MockTransport(handler)


def test_hedges_slow_attempt_to_next_upstream():
    hits: list[str] = []
    transport = _delayed_transport({"u1": 1.0, "u2": 0.0}, hits)
    pool = UpstreamPool(urls=["http://u1:9000", "http://u2:9000"], transport=transport,
                        hedge_delay=0.02, attempts=2)

    async def run():
        result = await pool.forward({"prompt": "p"})
        await pool.aclose()
        return result

    assert asyncio.run(run()) == (200, {"completion": "u2"})
    assert hits == ["u1", "u2"]


def test_persistently_slow_upstream_leaves_rotation():
    hits: list[str] = []
    transport = _delayed_transport({"u3": 0.3}, hits)
    urls = ["http://u1:9000", "http://u2:9000", "http://u3:9000"]
    pool = UpstreamPool(urls=urls, transport=transport, hedge_delay=0.1, attempts=2)

    async def run():
        for _ in range(6):
            await pool.forward({"prompt": "p"})
        hits.clear()
        for _ in range(6):
            assert (await pool.forward({"prompt": "p"}))[0] == 200
        await pool.aclose()

    asyncio.run(run())
    assert "u3" not in hits


def test_fails_over_on_connection_error():
    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "u1":
            raise httpx.ConnectError("down")
        return httpx.Response(200, json={"completion": "ok"})

    pool = UpstreamPool(urls=["http://u1:9000", "http://u2:9000"],
                        transport=httpx.MockTransport(handler), hedge_delay=1.0)

    async def run():
        result = await pool.forward({"prompt": "p"})
        await pool.aclose()
        return result

    assert asyncio.run(run()) == (200, {"completion": "ok"})
