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


def _scripted_transport(hits: list[str], responses: dict) -> httpx.MockTransport:
    async def handler(request: httpx.Request) -> httpx.Response:
        host = f"{request.url.scheme}://{request.url.host}:{request.url.port}"
        hits.append(host)
        action = responses.get(host, 200)
        if isinstance(action, Exception):
            raise action
        return httpx.Response(action, json={"from": host})

    return httpx.MockTransport(handler)


def _forward_once(pool: UpstreamPool, payload=None):
    async def run():
        try:
            return await pool.forward(payload or {"prompt": "p"})
        finally:
            await pool.aclose()

    return asyncio.run(run())


def test_client_has_bounded_timeout_and_limits():
    pool = UpstreamPool(urls=URLS, transport=_recording_transport([]))
    timeout = pool._client.timeout
    assert timeout.connect == 5.0
    assert timeout.read == 60.0
    asyncio.run(pool.aclose())

    real = UpstreamPool(urls=URLS)
    http_pool = real._client._transport._pool
    assert http_pool._max_connections == 200
    assert http_pool._max_keepalive_connections == 50
    asyncio.run(real.aclose())


def test_retry_on_5xx_uses_different_upstream():
    hits: list[str] = []
    pool = UpstreamPool(urls=URLS, transport=_scripted_transport(hits, {URLS[0]: 503}))
    status, body = _forward_once(pool)
    assert hits == [URLS[0], URLS[1]]
    assert (status, body) == (200, {"from": URLS[1]})


def test_retry_on_transport_error_uses_different_upstream():
    hits: list[str] = []
    responses = {URLS[0]: httpx.ConnectError("refused")}
    pool = UpstreamPool(urls=URLS, transport=_scripted_transport(hits, responses))
    assert _forward_once(pool) == (200, {"from": URLS[1]})
    assert hits == [URLS[0], URLS[1]]


def test_retry_on_timeout_uses_different_upstream():
    hits: list[str] = []
    responses = {URLS[0]: httpx.ReadTimeout("slow")}
    pool = UpstreamPool(urls=URLS, transport=_scripted_transport(hits, responses))
    assert _forward_once(pool) == (200, {"from": URLS[1]})
    assert hits == [URLS[0], URLS[1]]


def test_no_retry_on_success_or_4xx():
    hits: list[str] = []
    pool = UpstreamPool(urls=URLS, transport=_scripted_transport(hits, {URLS[0]: 422}))
    assert _forward_once(pool) == (422, {"from": URLS[0]})
    assert hits == [URLS[0]]

    hits.clear()
    pool = UpstreamPool(urls=URLS, transport=_scripted_transport(hits, {}))
    assert _forward_once(pool) == (200, {"from": URLS[0]})
    assert hits == [URLS[0]]


def test_both_5xx_returns_last_status_and_body():
    hits: list[str] = []
    responses = {URLS[0]: 503, URLS[1]: 502}
    pool = UpstreamPool(urls=URLS, transport=_scripted_transport(hits, responses))
    assert _forward_once(pool) == (502, {"from": URLS[1]})
    assert hits == [URLS[0], URLS[1]]


def test_5xx_then_transport_error_returns_first_response():
    hits: list[str] = []
    responses = {URLS[0]: 503, URLS[1]: httpx.ConnectError("refused")}
    pool = UpstreamPool(urls=URLS, transport=_scripted_transport(hits, responses))
    assert _forward_once(pool) == (503, {"from": URLS[0]})


def test_both_transport_errors_raise():
    hits: list[str] = []
    err = httpx.ConnectError("refused")
    pool = UpstreamPool(urls=URLS, transport=_scripted_transport(hits, {URLS[0]: err, URLS[1]: err}))
    try:
        _forward_once(pool)
    except httpx.ConnectError:
        pass
    else:
        raise AssertionError("expected ConnectError")
    assert hits == [URLS[0], URLS[1]]


def test_single_upstream_does_not_retry():
    hits: list[str] = []
    pool = UpstreamPool(urls=URLS[:1], transport=_scripted_transport(hits, {URLS[0]: 503}))
    assert _forward_once(pool) == (503, {"from": URLS[0]})
    assert hits == [URLS[0]]


def test_round_robin_continues_after_retry():
    hits: list[str] = []
    pool = UpstreamPool(urls=URLS, transport=_scripted_transport(hits, {URLS[0]: 503}))

    async def run():
        results = [await pool.forward({"prompt": "p"}) for _ in range(2)]
        await pool.aclose()
        return results

    results = asyncio.run(run())
    assert hits == [URLS[0], URLS[1], URLS[1]]
    assert results == [(200, {"from": URLS[1]}), (200, {"from": URLS[1]})]


# -- /v1/generate cache ------------------------------------------------------


class _CountingPool:
    def __init__(self, status=200):
        self.status = status
        self.calls: list[dict] = []

    async def forward(self, payload):
        self.calls.append(payload)
        n = len(self.calls)
        return self.status, {"completion": f"c{n}", "signature": "ab" * 32, "n": n}

    async def aclose(self):
        pass


def _generate_client(pool):
    from fastapi.testclient import TestClient

    from app import main

    c = TestClient(main.app)
    c.__enter__()
    main.app.state.pool = pool
    return c


def test_generate_cache_hit_returns_identical_bytes_without_upstream_call():
    pool = _CountingPool()
    c = _generate_client(pool)
    r1 = c.post("/v1/generate", json={"prompt": "hi", "max_tokens": 8})
    r2 = c.post("/v1/generate", json={"prompt": "hi", "max_tokens": 8})
    c.__exit__(None, None, None)
    assert r1.status_code == r2.status_code == 200
    assert r1.content == r2.content
    assert r1.headers["content-type"] == r2.headers["content-type"]
    assert len(pool.calls) == 1


def test_generate_cache_key_includes_max_tokens():
    pool = _CountingPool()
    c = _generate_client(pool)
    r1 = c.post("/v1/generate", json={"prompt": "hi", "max_tokens": 8})
    r2 = c.post("/v1/generate", json={"prompt": "hi", "max_tokens": 16})
    c.__exit__(None, None, None)
    assert r1.content != r2.content
    assert pool.calls == [{"prompt": "hi", "max_tokens": 8}, {"prompt": "hi", "max_tokens": 16}]


def test_generate_cache_skips_non_200():
    pool = _CountingPool(status=503)
    c = _generate_client(pool)
    r1 = c.post("/v1/generate", json={"prompt": "hi"})
    r2 = c.post("/v1/generate", json={"prompt": "hi"})
    c.__exit__(None, None, None)
    assert r1.status_code == r2.status_code == 503
    assert len(pool.calls) == 2


def test_lru_cache_is_bounded_and_evicts_least_recently_used():
    from app.cache import LRUCache

    cache = LRUCache(maxsize=2)
    cache.put("a", b"1")
    cache.put("b", b"2")
    assert cache.get("a") == b"1"
    cache.put("c", b"3")
    assert len(cache) == 2
    assert cache.get("b") is None
    assert cache.get("a") == b"1"
    assert cache.get("c") == b"3"


def test_generate_cache_default_size_is_bounded():
    from app import main

    pool = _CountingPool()
    c = _generate_client(pool)
    cache = main.app.state.generate_cache
    for i in range(main.GENERATE_CACHE_SIZE + 5):
        c.post("/v1/generate", json={"prompt": f"p{i}"})
    c.__exit__(None, None, None)
    assert main.GENERATE_CACHE_SIZE == 1000
    assert len(cache) == 1000
