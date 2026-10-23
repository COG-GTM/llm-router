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


def _scripted_transport(hits: list[str], behaviour: dict[str, str]) -> httpx.MockTransport:
    async def handler(request: httpx.Request) -> httpx.Response:
        host = request.url.host
        hits.append(host)
        mode = behaviour.get(host, "ok")
        if mode == "hang":
            await asyncio.sleep(5)
        if mode == "503":
            return httpx.Response(503, json={"detail": "model overloaded"})
        if mode == "down":
            raise httpx.ConnectError("connection refused", request=request)
        return httpx.Response(200, json={"completion": f"from {host}"})

    return httpx.MockTransport(handler)


def _forward(behaviour: dict[str, str], n: int = 1, **kwargs):
    hits: list[str] = []
    urls = ["http://u1:9000", "http://u2:9000", "http://u3:9000"]
    pool = UpstreamPool(urls=urls, transport=_scripted_transport(hits, behaviour),
                        attempt_timeout=0.05, timeout=0.2, **kwargs)

    async def run():
        results = [await pool.forward({"prompt": "p"}) for _ in range(n)]
        await pool.aclose()
        return results

    return asyncio.run(run()), hits


def test_timeout_fails_over_to_next_backend():
    results, hits = _forward({"u1": "hang"})
    assert hits == ["u1", "u2"]
    assert results == [(200, {"completion": "from u2"})]


def test_5xx_fails_over_to_next_backend():
    results, hits = _forward({"u1": "503"})
    assert hits == ["u1", "u2"]
    assert results == [(200, {"completion": "from u2"})]


def test_connection_error_fails_over_to_next_backend():
    results, hits = _forward({"u1": "down"})
    assert hits == ["u1", "u2"]
    assert results == [(200, {"completion": "from u2"})]


def test_retry_skips_failing_backends_until_healthy_one():
    results, hits = _forward({"u1": "hang", "u2": "503"})
    assert hits == ["u1", "u2", "u3"]
    assert results == [(200, {"completion": "from u3"})]


def test_failover_does_not_shift_round_robin():
    results, hits = _forward({"u1": "503"}, n=3)
    assert hits == ["u1", "u2", "u2", "u3"]
    assert [r[1]["completion"] for r in results] == ["from u2", "from u2", "from u3"]


def test_4xx_is_passed_through_without_retry():
    async def handler(request: httpx.Request) -> httpx.Response:
        hits.append(request.url.host)
        return httpx.Response(422, json={"detail": "bad prompt"})

    hits: list[str] = []
    pool = UpstreamPool(urls=["http://u1:9000", "http://u2:9000"],
                        transport=httpx.MockTransport(handler))

    async def run():
        result = await pool.forward({"prompt": "p"})
        await pool.aclose()
        return result

    assert asyncio.run(run()) == (422, {"detail": "bad prompt"})
    assert hits == ["u1"]


def test_all_backends_failing_returns_last_upstream_error():
    results, hits = _forward({"u1": "503", "u2": "503", "u3": "503"}, max_attempts=3)
    assert hits == ["u1", "u2", "u3"]
    assert results == [(503, {"detail": "model overloaded"})]


def test_all_backends_hanging_is_bounded_and_returns_504():
    results, hits = _forward({"u1": "hang", "u2": "hang", "u3": "hang"})
    assert hits == ["u1", "u2", "u3", "u1"]
    assert results == [(504, {"detail": "upstream timeout"})]


def test_last_attempt_gets_full_timeout():
    async def handler(request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(0.1)
        return httpx.Response(200, json={"completion": request.url.host})

    pool = UpstreamPool(urls=["http://u1:9000", "http://u2:9000"],
                        transport=httpx.MockTransport(handler),
                        attempt_timeout=0.05, timeout=1.0, max_attempts=2)

    async def run():
        result = await pool.forward({"prompt": "p"})
        await pool.aclose()
        return result

    assert asyncio.run(run()) == (200, {"completion": "u2"})
