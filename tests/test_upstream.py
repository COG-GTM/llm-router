import asyncio

import httpx
from fastapi.testclient import TestClient

from app import main
from app.config import upstream_urls
from app.generate_cache import ResponseCache
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


class _CountingPool:
    def __init__(self, status=200):
        self.status = status
        self.calls: list[dict] = []

    async def forward(self, payload):
        self.calls.append(payload)
        if self.status != 200:
            return self.status, {"detail": "model overloaded"}
        return 200, {
            "completion": f"answer {len(self.calls)}",
            "signature": f"{len(self.calls):064x}",
        }

    async def aclose(self):
        pass


def _generate_client(pool) -> TestClient:
    c = TestClient(main.app)
    c.__enter__()
    main.app.state.pool = pool
    return c


def test_generate_cache_hit_skips_upstream_and_is_byte_identical():
    pool = _CountingPool()
    c = _generate_client(pool)
    r1 = c.post("/v1/generate", json={"prompt": "hi", "max_tokens": 8})
    r2 = c.post("/v1/generate", json={"prompt": "hi", "max_tokens": 8})
    c.__exit__(None, None, None)
    assert r1.status_code == r2.status_code == 200
    assert r1.content == r2.content
    assert r2.headers["content-type"] == r1.headers["content-type"]
    assert len(pool.calls) == 1


def test_generate_cache_keys_on_max_tokens():
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
    pool.status = 200
    r2 = c.post("/v1/generate", json={"prompt": "hi"})
    r3 = c.post("/v1/generate", json={"prompt": "hi"})
    c.__exit__(None, None, None)
    assert r1.status_code == 503
    assert r1.json() == {"detail": "model overloaded"}
    assert r2.status_code == 200
    assert r3.content == r2.content
    assert len(pool.calls) == 2


def test_response_cache_evicts_least_recently_used():
    cache = ResponseCache(maxsize=2)
    cache.put(("a", 1), b"A")
    cache.put(("b", 1), b"B")
    assert cache.get(("a", 1)) == b"A"
    cache.put(("c", 1), b"C")
    assert len(cache) == 2
    assert ("b", 1) not in cache
    assert cache.get(("a", 1)) == b"A"
    assert cache.get(("c", 1)) == b"C"


def test_generate_cache_bounded_at_1000_entries():
    pool = _CountingPool()
    c = _generate_client(pool)
    for i in range(1001):
        c.post("/v1/generate", json={"prompt": f"p{i}"})
    cache = main.app.state.generate_cache
    c.post("/v1/generate", json={"prompt": "p0"})
    c.__exit__(None, None, None)
    assert cache.maxsize == 1000
    assert len(cache) == 1000
    assert len(pool.calls) == 1002
