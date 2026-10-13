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


def test_client_timeout_is_bounded():
    pool = UpstreamPool(urls=["http://u1:9000"], transport=_recording_transport([]))
    timeout = pool._client.timeout
    assert timeout.connect == 5.0
    assert timeout.read == 60.0
    assert timeout.write == 60.0
    assert timeout.pool == 60.0
    asyncio.run(pool.aclose())


def test_client_connection_limits():
    pool = UpstreamPool(urls=["http://u1:9000"])
    limits = pool._client._transport._pool
    assert limits._max_connections == 100
    assert limits._max_keepalive_connections == 50
    assert limits._keepalive_expiry == 30.0
    asyncio.run(pool.aclose())


def test_timeout_propagates_to_requests():
    seen: list[dict] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.extensions["timeout"])
        return httpx.Response(200, json={"completion": "ok"})

    pool = UpstreamPool(urls=["http://u1:9000"], transport=httpx.MockTransport(handler))

    async def run():
        result = await pool.forward({"prompt": "p"})
        await pool.aclose()
        return result

    assert asyncio.run(run()) == (200, {"completion": "ok"})
    assert seen == [{"connect": 5.0, "read": 60.0, "write": 60.0, "pool": 60.0}]
