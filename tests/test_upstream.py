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


URLS = ["http://u1:9000", "http://u2:9000", "http://u3:9000"]


def _host(request: httpx.Request) -> str:
    return f"{request.url.scheme}://{request.url.host}:{request.url.port}"


def _forward_once(pool: UpstreamPool):
    async def run():
        try:
            return await pool.forward({"prompt": "p"})
        finally:
            await pool.aclose()

    return asyncio.run(run())


def test_5xx_retried_on_another_url():
    hits: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        hits.append(_host(request))
        if _host(request) == URLS[0]:
            return httpx.Response(503, json={"error": "busy"})
        return httpx.Response(200, json={"completion": "ok"})

    pool = UpstreamPool(urls=URLS, transport=httpx.MockTransport(handler))
    assert _forward_once(pool) == (200, {"completion": "ok"})
    assert hits == [URLS[0], URLS[1]]


def test_transport_error_retried_on_another_url():
    hits: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        hits.append(_host(request))
        if _host(request) == URLS[0]:
            raise httpx.ConnectError("refused", request=request)
        return httpx.Response(200, json={"completion": "ok"})

    pool = UpstreamPool(urls=URLS, transport=httpx.MockTransport(handler))
    assert _forward_once(pool) == (200, {"completion": "ok"})
    assert hits == [URLS[0], URLS[1]]


def test_timeout_retried_on_another_url():
    hits: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        hits.append(_host(request))
        if _host(request) == URLS[0]:
            raise httpx.ReadTimeout("slow", request=request)
        return httpx.Response(200, json={"completion": "ok"})

    pool = UpstreamPool(urls=URLS, transport=httpx.MockTransport(handler))
    assert _forward_once(pool) == (200, {"completion": "ok"})
    assert hits == [URLS[0], URLS[1]]


def test_2xx_and_4xx_not_retried():
    hits: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        hits.append(_host(request))
        if len(hits) == 1:
            return httpx.Response(200, json={"completion": "ok"})
        return httpx.Response(422, json={"error": "bad"})

    pool = UpstreamPool(urls=URLS, transport=httpx.MockTransport(handler))

    async def run():
        results = [await pool.forward({"prompt": "p"}) for _ in range(2)]
        await pool.aclose()
        return results

    assert asyncio.run(run()) == [(200, {"completion": "ok"}), (422, {"error": "bad"})]
    assert hits == [URLS[0], URLS[1]]


def test_both_5xx_returns_retry_response():
    hits: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        hits.append(_host(request))
        return httpx.Response(500 + len(hits), json={"error": len(hits)})

    pool = UpstreamPool(urls=URLS, transport=httpx.MockTransport(handler))
    assert _forward_once(pool) == (502, {"error": 2})
    assert hits == [URLS[0], URLS[1]]


def test_retry_transport_error_propagates():
    hits: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        hits.append(_host(request))
        if len(hits) == 1:
            return httpx.Response(503, json={"error": "busy"})
        raise httpx.ConnectError("refused", request=request)

    pool = UpstreamPool(urls=URLS, transport=httpx.MockTransport(handler))
    try:
        _forward_once(pool)
    except httpx.ConnectError:
        pass
    else:
        raise AssertionError("expected ConnectError")
    assert hits == [URLS[0], URLS[1]]


def test_single_url_not_retried():
    hits: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        hits.append(_host(request))
        return httpx.Response(503, json={"error": "busy"})

    pool = UpstreamPool(urls=URLS[:1], transport=httpx.MockTransport(handler))
    assert _forward_once(pool) == (503, {"error": "busy"})
    assert hits == [URLS[0]]
