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


def _scripted_transport(behaviour: dict, hits: list[str]) -> httpx.MockTransport:
    async def handler(request: httpx.Request) -> httpx.Response:
        host = request.url.host
        hits.append(host)
        action = behaviour.get(host, 200)
        if action == "connect_error":
            raise httpx.ConnectError("refused", request=request)
        if action == "timeout":
            raise httpx.ReadTimeout("slow", request=request)
        return httpx.Response(action, json={"completion": host, "status": action})

    return httpx.MockTransport(handler)


def _forward_once(behaviour: dict, hits: list[str]):
    urls = ["http://u1:9000", "http://u2:9000", "http://u3:9000"]
    pool = UpstreamPool(urls=urls, transport=_scripted_transport(behaviour, hits))

    async def run():
        try:
            return await pool.forward({"prompt": "p"})
        finally:
            await pool.aclose()

    return asyncio.run(run())


def test_failover_on_transport_error():
    hits: list[str] = []
    status, body = _forward_once({"u1": "connect_error"}, hits)
    assert (status, body["completion"]) == (200, "u2")
    assert hits == ["u1", "u2"]


def test_failover_on_timeout():
    hits: list[str] = []
    status, body = _forward_once({"u1": "timeout", "u2": "timeout"}, hits)
    assert (status, body["completion"]) == (200, "u3")
    assert hits == ["u1", "u2", "u3"]


def test_failover_on_5xx():
    hits: list[str] = []
    status, body = _forward_once({"u1": 503}, hits)
    assert (status, body["completion"]) == (200, "u2")
    assert hits == ["u1", "u2"]


def test_4xx_returned_without_retry():
    hits: list[str] = []
    status, body = _forward_once({"u1": 422}, hits)
    assert status == 422
    assert hits == ["u1"]


def test_all_transport_errors_returns_502():
    hits: list[str] = []
    behaviour = {"u1": "connect_error", "u2": "timeout", "u3": "connect_error"}
    assert _forward_once(behaviour, hits) == (502, {"error": "all upstreams unavailable"})
    assert hits == ["u1", "u2", "u3"]


def test_all_5xx_returns_last_upstream_error():
    hits: list[str] = []
    status, body = _forward_once({"u1": 500, "u2": 502, "u3": 503}, hits)
    assert (status, body["completion"]) == (503, "u3")
    assert hits == ["u1", "u2", "u3"]


def test_failover_keeps_round_robin_rotation():
    hits: list[str] = []
    urls = ["http://u1:9000", "http://u2:9000", "http://u3:9000"]
    pool = UpstreamPool(urls=urls,
                        transport=_scripted_transport({"u2": "connect_error"}, hits))

    async def run():
        results = [await pool.forward({"prompt": "p"}) for _ in range(2)]
        await pool.aclose()
        return results

    results = asyncio.run(run())
    assert [b["completion"] for _, b in results] == ["u1", "u3"]
    assert hits == ["u1", "u2", "u3"]
