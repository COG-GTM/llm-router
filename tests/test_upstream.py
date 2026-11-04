import asyncio
import hashlib
import json
import time

import httpx

from app.config import upstream_attempt_timeout, upstream_timeout, upstream_urls
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
        results = [await pool.forward({"prompt": f"p{i}"}) for i in range(4)]
        await pool.aclose()
        return results

    results = asyncio.run(run())
    assert hits == ["http://u1:9000", "http://u2:9000", "http://u3:9000", "http://u1:9000"]
    assert all(r == (200, {"completion": "ok"}) for r in results)


def _replica(request: httpx.Request) -> str:
    return f"{request.url.scheme}://{request.url.host}:{request.url.port}"


def _signed_transport(hits: list[str], slow: set[str] = frozenset(),
                      slow_s: float = 5.0) -> httpx.MockTransport:
    async def handler(request: httpx.Request) -> httpx.Response:
        replica = _replica(request)
        hits.append(replica)
        if replica in slow:
            await asyncio.sleep(slow_s)
        prompt = json.loads(request.content)["prompt"]
        return httpx.Response(200, json={
            "completion": f"echo {prompt}",
            "signature": hashlib.sha256(prompt.encode()).hexdigest(),
            "replica": replica,
        })

    return httpx.MockTransport(handler)


def test_slow_upstream_times_out_and_retries_on_another_replica():
    hits: list[str] = []
    urls = ["http://u1:9000", "http://u2:9000", "http://u3:9000"]
    pool = UpstreamPool(urls=urls, transport=_signed_transport(hits, slow={"http://u1:9000"}),
                        attempt_timeout=0.1)

    async def run():
        start = time.monotonic()
        result = await pool.forward({"prompt": "p", "max_tokens": 64})
        elapsed = time.monotonic() - start
        await pool.aclose()
        return result, elapsed

    (status, body), elapsed = asyncio.run(run())
    assert status == 200
    assert body["replica"] == "http://u2:9000"
    assert hits == ["http://u1:9000", "http://u2:9000"]
    assert elapsed < 1.0


def test_all_replicas_slow_returns_504_within_timeout():
    hits: list[str] = []
    urls = ["http://u1:9000", "http://u2:9000"]
    pool = UpstreamPool(urls=urls, transport=_signed_transport(hits, slow=set(urls)),
                        attempt_timeout=0.05, timeout=0.2)

    async def run():
        start = time.monotonic()
        result = await pool.forward({"prompt": "p", "max_tokens": 64})
        elapsed = time.monotonic() - start
        await pool.aclose()
        return result, elapsed

    (status, body), elapsed = asyncio.run(run())
    assert status == 504
    assert "detail" in body
    assert sorted(hits) == urls
    assert elapsed < 1.0


def test_5xx_is_retried_on_another_replica():
    hits: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        hits.append(_replica(request))
        if len(hits) == 1:
            return httpx.Response(503, json={"detail": "model overloaded"})
        return httpx.Response(200, json={"completion": "ok"})

    urls = ["http://u1:9000", "http://u2:9000"]
    pool = UpstreamPool(urls=urls, transport=httpx.MockTransport(handler))

    async def run():
        result = await pool.forward({"prompt": "p", "max_tokens": 64})
        await pool.aclose()
        return result

    assert asyncio.run(run()) == (200, {"completion": "ok"})
    assert hits == urls


def test_persistent_5xx_is_passed_through_and_not_cached():
    hits: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        hits.append(_replica(request))
        return httpx.Response(503, json={"detail": "model overloaded"})

    urls = ["http://u1:9000", "http://u2:9000"]
    pool = UpstreamPool(urls=urls, transport=httpx.MockTransport(handler))

    async def run():
        results = [await pool.forward({"prompt": "p", "max_tokens": 64}) for _ in range(2)]
        await pool.aclose()
        return results

    results = asyncio.run(run())
    assert results == [(503, {"detail": "model overloaded"})] * 2
    assert len(hits) == 4


def test_identical_payloads_are_served_from_cache():
    hits: list[str] = []
    pool = UpstreamPool(urls=["http://u1:9000", "http://u2:9000"],
                        transport=_signed_transport(hits))

    async def run():
        results = [await pool.forward({"prompt": "same", "max_tokens": 64}) for _ in range(5)]
        await pool.aclose()
        return results

    results = asyncio.run(run())
    assert len(hits) == 1
    assert all(r == results[0] for r in results)


def test_cached_response_preserves_signature():
    hits: list[str] = []
    pool = UpstreamPool(urls=["http://u1:9000", "http://u2:9000"],
                        transport=_signed_transport(hits))
    expected = hashlib.sha256(b"sig prompt").hexdigest()

    async def run():
        first = await pool.forward({"prompt": "sig prompt", "max_tokens": 64})
        first[1]["signature"] = "tampered"
        second = await pool.forward({"prompt": "sig prompt", "max_tokens": 64})
        await pool.aclose()
        return second

    status, body = asyncio.run(run())
    assert status == 200
    assert body["signature"] == expected
    assert len(hits) == 1


def test_distinct_payloads_still_reach_upstreams():
    hits: list[str] = []
    pool = UpstreamPool(urls=["http://u1:9000", "http://u2:9000"],
                        transport=_signed_transport(hits))
    payloads = [
        {"prompt": "a", "max_tokens": 64},
        {"prompt": "b", "max_tokens": 64},
        {"prompt": "a", "max_tokens": 32},
    ]

    async def run():
        results = [await pool.forward(p) for p in payloads]
        await pool.aclose()
        return results

    results = asyncio.run(run())
    assert len(hits) == 3
    assert [r[1]["signature"] for r in results] == [
        hashlib.sha256(b"a").hexdigest(),
        hashlib.sha256(b"b").hexdigest(),
        hashlib.sha256(b"a").hexdigest(),
    ]


def test_cache_is_bounded():
    hits: list[str] = []
    pool = UpstreamPool(urls=["http://u1:9000"], transport=_signed_transport(hits),
                        cache_size=2)

    async def run():
        for prompt in ["a", "b", "c", "a"]:
            await pool.forward({"prompt": prompt, "max_tokens": 64})
        await pool.aclose()

    asyncio.run(run())
    assert len(hits) == 4


def test_prefers_faster_replicas_over_a_consistently_slow_one():
    hits: list[str] = []
    urls = ["http://u1:9000", "http://u2:9000", "http://u3:9000"]
    pool = UpstreamPool(urls=urls,
                        transport=_signed_transport(hits, slow={"http://u3:9000"}, slow_s=0.2),
                        attempt_timeout=1.0)

    async def run():
        for i in range(12):
            await pool.forward({"prompt": f"p{i}", "max_tokens": 64})
        await pool.aclose()

    asyncio.run(run())
    assert hits.count("http://u3:9000") == 1
    assert hits[-6:].count("http://u1:9000") == 3
    assert hits[-6:].count("http://u2:9000") == 3


def test_non_json_5xx_is_retried_on_another_replica():
    hits: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        hits.append(_replica(request))
        if len(hits) == 1:
            return httpx.Response(502, text="<html>bad gateway</html>")
        return httpx.Response(200, json={"completion": "ok"})

    urls = ["http://u1:9000", "http://u2:9000"]
    pool = UpstreamPool(urls=urls, transport=httpx.MockTransport(handler))

    async def run():
        result = await pool.forward({"prompt": "p", "max_tokens": 64})
        await pool.aclose()
        return result

    assert asyncio.run(run()) == (200, {"completion": "ok"})
    assert hits == urls


def test_non_finite_timeout_env_falls_back_to_default(monkeypatch):
    monkeypatch.setenv("UPSTREAM_ATTEMPT_TIMEOUT_S", "inf")
    monkeypatch.setenv("UPSTREAM_TIMEOUT_S", "nan")
    assert upstream_attempt_timeout() == 0.25
    assert upstream_timeout() == 10.0
