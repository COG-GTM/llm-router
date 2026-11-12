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


def _scripted_transport(hits: list[str], behaviour: dict) -> httpx.MockTransport:
    """behaviour maps host -> 'ok' | 'timeout' | 'connect' | int status."""
    async def handler(request: httpx.Request) -> httpx.Response:
        host = request.url.host
        hits.append(host)
        action = behaviour.get(host, "ok")
        if action == "timeout":
            raise httpx.ReadTimeout("slow", request=request)
        if action == "connect":
            raise httpx.ConnectError("down", request=request)
        if isinstance(action, int):
            return httpx.Response(action, json={"detail": f"err {action}"})
        return httpx.Response(200, json={"completion": host})

    return httpx.MockTransport(handler)


def _forward_n(pool: UpstreamPool, n: int) -> list:
    async def run():
        results = [await pool.forward({"prompt": "p"}) for _ in range(n)]
        await pool.aclose()
        return results

    return asyncio.run(run())


URLS = ["http://u1:9000", "http://u2:9000", "http://u3:9000"]


def test_timeout_env_config(monkeypatch):
    from app.config import upstream_retry_timeout, upstream_timeout

    monkeypatch.delenv("LLM_UPSTREAM_TIMEOUT_S", raising=False)
    monkeypatch.delenv("LLM_UPSTREAM_RETRY_TIMEOUT_S", raising=False)
    assert upstream_timeout() == 0.5
    assert upstream_retry_timeout() == 10.0
    monkeypatch.setenv("LLM_UPSTREAM_TIMEOUT_S", "0.75")
    monkeypatch.setenv("LLM_UPSTREAM_RETRY_TIMEOUT_S", "3")
    assert upstream_timeout() == 0.75
    assert upstream_retry_timeout() == 3.0


def test_client_has_finite_timeout():
    pool = UpstreamPool(urls=URLS, timeout=0.5, retry_timeout=10)
    assert pool._client.timeout.read == 10
    asyncio.run(pool.aclose())


def test_retries_once_on_different_backend_after_timeout():
    hits: list[str] = []
    pool = UpstreamPool(urls=URLS, transport=_scripted_transport(hits, {"u1": "timeout"}))
    results = _forward_n(pool, 1)
    assert hits == ["u1", "u2"]
    assert results == [(200, {"completion": "u2"})]


def test_retries_on_connect_error_and_5xx():
    hits: list[str] = []
    pool = UpstreamPool(urls=URLS,
                        transport=_scripted_transport(hits, {"u1": "connect", "u3": 503}))
    results = _forward_n(pool, 3)
    assert hits == ["u1", "u2", "u3", "u1", "u2"]
    assert results[0] == (200, {"completion": "u2"})
    assert results[1][0] == 502
    assert results[2] == (200, {"completion": "u2"})


def test_no_retry_on_4xx():
    hits: list[str] = []
    pool = UpstreamPool(urls=URLS, transport=_scripted_transport(hits, {"u1": 422}))
    results = _forward_n(pool, 1)
    assert hits == ["u1"]
    assert results == [(422, {"detail": "err 422"})]


def test_retry_failure_maps_to_gateway_errors():
    hits: list[str] = []
    pool = UpstreamPool(urls=URLS, transport=_scripted_transport(
        hits, {"u1": "timeout", "u2": "timeout"}))
    assert _forward_n(pool, 1) == [(504, {"detail": "upstream timeout"})]
    assert hits == ["u1", "u2"]

    hits = []
    pool = UpstreamPool(urls=URLS, transport=_scripted_transport(
        hits, {"u1": "connect", "u2": "connect"}))
    assert _forward_n(pool, 1) == [(502, {"detail": "upstream unavailable"})]
    assert hits == ["u1", "u2"]


def test_retry_passes_through_final_5xx():
    hits: list[str] = []
    pool = UpstreamPool(urls=URLS, transport=_scripted_transport(hits, {"u1": 500, "u2": 503}))
    results = _forward_n(pool, 1)
    assert hits == ["u1", "u2"]
    assert results == [(503, {"detail": "err 503"})]


def test_single_backend_retries_same_url():
    hits: list[str] = []
    calls = {"n": 0}

    async def handler(request: httpx.Request) -> httpx.Response:
        hits.append(request.url.host)
        calls["n"] += 1
        if calls["n"] == 1:
            raise httpx.ReadTimeout("slow", request=request)
        return httpx.Response(200, json={"completion": "ok"})

    pool = UpstreamPool(urls=["http://only:9000"], transport=httpx.MockTransport(handler))
    assert _forward_n(pool, 1) == [(200, {"completion": "ok"})]
    assert hits == ["only", "only"]


def test_real_timeout_triggers_retry():
    hits: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        hits.append(request.url.host)
        if request.url.host == "u1":
            await asyncio.sleep(1)
        return httpx.Response(200, json={"completion": request.url.host})

    pool = UpstreamPool(urls=URLS, transport=httpx.MockTransport(handler), timeout=0.05)
    assert _forward_n(pool, 1) == [(200, {"completion": "u2"})]
