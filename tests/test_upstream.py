import asyncio

import httpx

from app.config import upstream_timeout, upstream_urls
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


def test_upstream_timeout_default(monkeypatch):
    monkeypatch.delenv("UPSTREAM_TIMEOUT_S", raising=False)
    assert upstream_timeout() == 0.4


def test_upstream_timeout_env(monkeypatch):
    monkeypatch.setenv("UPSTREAM_TIMEOUT_S", "1.25")
    assert upstream_timeout() == 1.25


def test_client_uses_configured_timeout(monkeypatch):
    monkeypatch.setenv("UPSTREAM_TIMEOUT_S", "0.3")
    pool = UpstreamPool(urls=["http://u1:9000"], transport=_recording_transport([]))
    assert pool._client.timeout == httpx.Timeout(0.3)
    explicit = UpstreamPool(urls=["http://u1:9000"], transport=_recording_transport([]),
                            timeout=2.0)
    assert explicit._client.timeout == httpx.Timeout(2.0)
    asyncio.run(pool.aclose())
    asyncio.run(explicit.aclose())


def test_timeout_propagates_without_retry():
    hits: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        hits.append(request.url.host)
        raise httpx.ReadTimeout("slow", request=request)

    pool = UpstreamPool(urls=["http://u1:9000", "http://u2:9000"],
                        transport=httpx.MockTransport(handler))

    async def run():
        try:
            await pool.forward({"prompt": "p"})
        finally:
            await pool.aclose()

    try:
        asyncio.run(run())
    except httpx.ReadTimeout:
        pass
    else:
        raise AssertionError("expected ReadTimeout")
    assert hits == ["u1"]
