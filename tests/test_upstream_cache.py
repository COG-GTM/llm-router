import asyncio
import json

import httpx

from app.cache import ResponseCache
from app.upstream import UpstreamPool

URLS = ["http://u1:9000", "http://u2:9000"]


def _recording_transport(hits: list[dict], status: int = 200,
                         delay_s: float = 0.0) -> httpx.MockTransport:
    async def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        hits.append(payload)
        if delay_s:
            await asyncio.sleep(delay_s)
        return httpx.Response(status, json={"completion": f"re: {payload['prompt']}"})

    return httpx.MockTransport(handler)


def _pool(hits, cache=None, **kwargs) -> UpstreamPool:
    return UpstreamPool(urls=URLS, transport=_recording_transport(hits, **kwargs),
                        cache=cache or ResponseCache())


def test_identical_payload_served_from_cache():
    hits: list[dict] = []
    pool = _pool(hits)

    async def run():
        first = await pool.forward({"prompt": "p", "max_tokens": 8})
        second = await pool.forward({"prompt": "p", "max_tokens": 8})
        await pool.aclose()
        return first, second

    first, second = asyncio.run(run())
    assert hits == [{"prompt": "p", "max_tokens": 8}]
    assert first == second == (200, {"completion": "re: p"})


def test_distinct_payloads_hit_upstream():
    hits: list[dict] = []
    pool = _pool(hits)
    payloads = [{"prompt": "a", "max_tokens": 8}, {"prompt": "b", "max_tokens": 8},
                {"prompt": "a", "max_tokens": 16}]

    async def run():
        for p in payloads:
            await pool.forward(p)
        await pool.aclose()

    asyncio.run(run())
    assert hits == payloads


def test_non_200_not_cached():
    hits: list[dict] = []
    pool = _pool(hits, status=503)

    async def run():
        results = [await pool.forward({"prompt": "p", "max_tokens": 8}) for _ in range(2)]
        await pool.aclose()
        return results

    results = asyncio.run(run())
    assert len(hits) == 2
    assert all(status == 503 for status, _ in results)


def test_concurrent_identical_misses_coalesce():
    hits: list[dict] = []
    pool = _pool(hits, delay_s=0.05)

    async def run():
        results = await asyncio.gather(
            *(pool.forward({"prompt": "p", "max_tokens": 8}) for _ in range(10)))
        await pool.aclose()
        return results

    results = asyncio.run(run())
    assert len(hits) == 1
    assert all(r == (200, {"completion": "re: p"}) for r in results)


def test_returned_bodies_are_copies():
    hits: list[dict] = []
    pool = _pool(hits)

    async def run():
        _, body = await pool.forward({"prompt": "p", "max_tokens": 8})
        body["completion"] = "mutated"
        _, again = await pool.forward({"prompt": "p", "max_tokens": 8})
        await pool.aclose()
        return again

    assert asyncio.run(run()) == {"completion": "re: p"}


def test_ttl_expiry_and_lru_eviction(monkeypatch):
    now = [1000.0]
    monkeypatch.setattr("app.cache.time.monotonic", lambda: now[0])
    hits: list[dict] = []
    pool = _pool(hits, cache=ResponseCache(max_size=2, ttl_s=10))

    async def call(prompt):
        await pool.forward({"prompt": prompt, "max_tokens": 8})

    async def run():
        await call("a")
        await call("b")
        await call("a")  # hit; "a" becomes most recent
        await call("c")  # evicts "b"
        await call("b")  # miss
        now[0] += 11
        await call("b")  # expired
        await pool.aclose()

    asyncio.run(run())
    assert [h["prompt"] for h in hits] == ["a", "b", "c", "b", "b"]


def test_disabled_cache_always_forwards():
    hits: list[dict] = []
    pool = _pool(hits, cache=ResponseCache(max_size=0))

    async def run():
        for _ in range(2):
            await pool.forward({"prompt": "p", "max_tokens": 8})
        await pool.aclose()

    asyncio.run(run())
    assert len(hits) == 2
