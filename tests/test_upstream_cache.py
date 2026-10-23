import asyncio

import httpx

from app.upstream import UpstreamPool

URLS = ["http://u1:9000", "http://u2:9000", "http://u3:9000"]


def _transport(hits: list[dict], status: int = 200, delay: float = 0.0) -> httpx.MockTransport:
    async def handler(request: httpx.Request) -> httpx.Response:
        hits.append(request.read())
        if delay:
            await asyncio.sleep(delay)
        return httpx.Response(status, json={"completion": "ok", "usage": {"tokens": 1}})

    return httpx.MockTransport(handler)


def _pool(transport: httpx.MockTransport, **kwargs) -> UpstreamPool:
    kwargs.setdefault("cache_size", 16)
    kwargs.setdefault("cache_ttl", 60)
    return UpstreamPool(urls=URLS, transport=transport, **kwargs)


def test_repeat_identical_call_is_served_from_cache():
    hits: list = []
    pool = _pool(_transport(hits))

    async def run():
        first = await pool.forward({"prompt": "p", "max_tokens": 64})
        second = await pool.forward({"prompt": "p", "max_tokens": 64})
        await pool.aclose()
        return first, second

    first, second = asyncio.run(run())
    assert len(hits) == 1
    assert first == second == (200, {"completion": "ok", "usage": {"tokens": 1}})


def test_distinct_payloads_hit_transport():
    hits: list = []
    pool = _pool(_transport(hits))

    async def run():
        await pool.forward({"prompt": "a", "max_tokens": 64})
        await pool.forward({"prompt": "b", "max_tokens": 64})
        await pool.forward({"prompt": "a", "max_tokens": 32})
        await pool.aclose()

    asyncio.run(run())
    assert len(hits) == 3


def test_non_200_responses_are_not_cached():
    hits: list = []
    pool = _pool(_transport(hits, status=503))

    async def run():
        results = [await pool.forward({"prompt": "p", "max_tokens": 64}) for _ in range(2)]
        await pool.aclose()
        return results

    results = asyncio.run(run())
    # each uncached request fails over across every backend
    assert len(hits) == 2 * len(URLS)
    assert all(status == 503 for status, _ in results)


def test_failed_shared_fetch_retries_another_replica():
    hits: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        hits.append(f"{request.url.host}")
        if request.url.host == "u1":
            return httpx.Response(503, json={"error": "down"})
        return httpx.Response(200, json={"completion": "ok", "usage": {"tokens": 1}})

    pool = _pool(httpx.MockTransport(handler), cache_ttl=60)

    async def run():
        results = await asyncio.gather(
            *(pool.forward({"prompt": "p", "max_tokens": 64}) for _ in range(5))
        )
        await pool.aclose()
        return results

    results = asyncio.run(run())
    assert hits == ["u1", "u2"]
    assert all(r == (200, {"completion": "ok", "usage": {"tokens": 1}}) for r in results)


def test_shared_fetch_tries_each_backend_once():
    hits: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        hits.append(f"{request.url.host}")
        if request.url.host == "u1":
            raise httpx.ConnectError("refused", request=request)
        return httpx.Response(503, json={"error": "down"})

    pool = _pool(httpx.MockTransport(handler), cache_ttl=60)

    async def run():
        result = await pool.forward({"prompt": "p", "max_tokens": 64})
        await pool.aclose()
        return result

    status, _ = asyncio.run(run())
    assert hits == ["u1", "u2", "u3"]
    assert status == 503


def test_client_error_is_not_retried():
    hits: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        hits.append(f"{request.url.host}")
        return httpx.Response(400, json={"error": "invalid max_tokens"})

    pool = _pool(httpx.MockTransport(handler), cache_ttl=60)

    async def run():
        result = await pool.forward({"prompt": "p", "max_tokens": 64})
        await pool.aclose()
        return result

    status, body = asyncio.run(run())
    assert hits == ["u1"]
    assert status == 400
    assert body == {"error": "invalid max_tokens"}


def test_cached_body_cannot_be_mutated_by_callers():
    hits: list = []
    pool = _pool(_transport(hits))

    async def run():
        _, first = await pool.forward({"prompt": "p", "max_tokens": 64})
        first["completion"] = "mutated"
        first["usage"]["tokens"] = 99
        _, second = await pool.forward({"prompt": "p", "max_tokens": 64})
        await pool.aclose()
        return second

    assert asyncio.run(run()) == {"completion": "ok", "usage": {"tokens": 1}}
    assert len(hits) == 1


def test_concurrent_identical_misses_are_coalesced():
    hits: list = []
    pool = _pool(_transport(hits, delay=0.05))

    async def run():
        results = await asyncio.gather(
            *(pool.forward({"prompt": "p", "max_tokens": 64}) for _ in range(5))
        )
        await pool.aclose()
        return results

    results = asyncio.run(run())
    assert len(hits) == 1
    assert all(r == (200, {"completion": "ok", "usage": {"tokens": 1}}) for r in results)
    bodies = [body for _, body in results]
    assert len({id(b) for b in bodies}) == len(bodies)


def test_entries_expire_after_ttl(monkeypatch):
    hits: list = []
    pool = _pool(_transport(hits), cache_ttl=10)
    now = [1000.0]
    monkeypatch.setattr("app.upstream.time.monotonic", lambda: now[0])

    async def run():
        await pool.forward({"prompt": "p", "max_tokens": 64})
        now[0] += 11
        await pool.forward({"prompt": "p", "max_tokens": 64})
        await pool.aclose()

    asyncio.run(run())
    assert len(hits) == 2


def test_lru_evicts_least_recently_used():
    hits: list = []
    pool = _pool(_transport(hits), cache_size=2)

    async def run():
        for prompt in ["a", "b", "a", "c", "a", "b"]:
            await pool.forward({"prompt": prompt, "max_tokens": 64})
        await pool.aclose()

    asyncio.run(run())
    # a, b miss; a hits; c misses and evicts b; a hits; b misses again
    assert len(hits) == 4


def test_cache_disabled_by_default_for_injected_transport():
    hits: list = []
    pool = UpstreamPool(urls=URLS, transport=_transport(hits))

    async def run():
        for _ in range(2):
            await pool.forward({"prompt": "p", "max_tokens": 64})
        await pool.aclose()

    asyncio.run(run())
    assert len(hits) == 2
