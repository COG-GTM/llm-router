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


def _host(request: httpx.Request) -> str:
    return f"{request.url.scheme}://{request.url.host}:{request.url.port}"


def test_concurrent_requests_avoid_busy_slow_backend():
    hits: list[str] = []
    slow_started = asyncio.Event()

    async def handler(request: httpx.Request) -> httpx.Response:
        host = _host(request)
        hits.append(host)
        if host == "http://u1:9000":
            slow_started.set()
            await asyncio.sleep(0.3)
        return httpx.Response(200, json={"completion": host})

    urls = ["http://u1:9000", "http://u2:9000", "http://u3:9000"]
    pool = UpstreamPool(urls=urls, transport=httpx.MockTransport(handler))

    async def run():
        slow = asyncio.create_task(pool.forward({"prompt": "p"}))
        await slow_started.wait()
        fast = [await pool.forward({"prompt": "p"}) for _ in range(6)]
        await slow
        await pool.aclose()
        return fast

    fast = asyncio.run(run())
    assert hits[0] == "http://u1:9000"
    assert "http://u1:9000" not in hits[1:]
    assert {r[1]["completion"] for r in fast} == {"http://u2:9000", "http://u3:9000"}


def test_slow_backend_is_avoided_after_it_completes():
    hits: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        host = _host(request)
        hits.append(host)
        if host == "http://u2:9000":
            await asyncio.sleep(0.2)
        return httpx.Response(200, json={"completion": "ok"})

    urls = ["http://u1:9000", "http://u2:9000", "http://u3:9000"]
    pool = UpstreamPool(urls=urls, transport=httpx.MockTransport(handler))

    async def run():
        for _ in range(9):
            await pool.forward({"prompt": "p"})
        await pool.aclose()

    asyncio.run(run())
    assert hits[:3] == urls
    assert "http://u2:9000" not in hits[3:]
    assert set(hits[3:]) == {"http://u1:9000", "http://u3:9000"}


def test_in_flight_returns_to_zero_after_errors():
    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "u1":
            raise httpx.ConnectError("boom", request=request)
        return httpx.Response(503, json={"detail": "overloaded"})

    urls = ["http://u1:9000", "http://u2:9000"]
    pool = UpstreamPool(urls=urls, transport=httpx.MockTransport(handler))

    async def run():
        results = await asyncio.gather(
            *(pool.forward({"prompt": "p"}) for _ in range(6)), return_exceptions=True
        )
        await pool.aclose()
        return results

    results = asyncio.run(run())
    assert any(isinstance(r, httpx.ConnectError) for r in results)
    assert (503, {"detail": "overloaded"}) in results
    assert [b.in_flight for b in pool._backends] == [0, 0]


def test_hung_unmeasured_backend_does_not_absorb_traffic():
    hits: list[str] = []
    release = asyncio.Event()
    hung_started = asyncio.Event()

    async def handler(request: httpx.Request) -> httpx.Response:
        host = _host(request)
        hits.append(host)
        if host == "http://u2:9000":
            hung_started.set()
            await release.wait()
        else:
            await asyncio.sleep(0.01)
        return httpx.Response(200, json={"completion": "ok"})

    urls = ["http://u1:9000", "http://u2:9000", "http://u3:9000"]
    pool = UpstreamPool(urls=urls, transport=httpx.MockTransport(handler))

    async def run():
        await pool.forward({"prompt": "p"})
        hung = asyncio.create_task(pool.forward({"prompt": "p"}))
        await hung_started.wait()
        for _ in range(8):
            await pool.forward({"prompt": "p"})
        release.set()
        await hung
        await pool.aclose()

    asyncio.run(run())
    assert hits.count("http://u2:9000") == 1


def test_fast_failures_are_penalized():
    hits: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        host = _host(request)
        hits.append(host)
        if host == "http://u1:9000":
            return httpx.Response(429, json={"detail": "rate limited"})
        if host == "http://u2:9000":
            return httpx.Response(200, text="not json")
        await asyncio.sleep(0.05)
        return httpx.Response(200, json={"completion": "ok"})

    urls = ["http://u1:9000", "http://u2:9000", "http://u3:9000"]
    pool = UpstreamPool(urls=urls, transport=httpx.MockTransport(handler))

    async def run():
        for _ in range(6):
            try:
                await pool.forward({"prompt": "p"})
            except ValueError:
                pass
        await pool.aclose()

    asyncio.run(run())
    assert hits[:3] == urls
    assert set(hits[3:]) == {"http://u3:9000"}
