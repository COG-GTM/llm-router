import asyncio

import httpx
import pytest

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


def _transport(delays: dict[str, float], hits: list[str],
               fail: set[str] = frozenset()) -> httpx.MockTransport:
    """Per-host artificial latency; hosts in `fail` refuse connections."""
    async def handler(request: httpx.Request) -> httpx.Response:
        host = request.url.host
        hits.append(host)
        if host in fail:
            raise httpx.ConnectError("refused", request=request)
        await asyncio.sleep(delays.get(host, 0))
        return httpx.Response(200, json={"completion": "ok", "host": host})

    return httpx.MockTransport(handler)


URLS = ["http://u1:9000", "http://u2:9000", "http://u3:9000"]


def test_first_call_uses_first_url_and_passes_through():
    hits: list[str] = []
    pool = UpstreamPool(urls=URLS, transport=_transport({}, hits))

    async def run():
        result = await pool.forward({"prompt": "p"})
        await pool.aclose()
        return result

    assert asyncio.run(run()) == (200, {"completion": "ok", "host": "u1"})
    assert hits == ["u1"]


def test_fast_upstream_preferred_over_slow():
    hits: list[str] = []
    delays = {"u1": 0.05, "u2": 0.001, "u3": 0.03}
    pool = UpstreamPool(urls=URLS, transport=_transport(delays, hits))

    async def run():
        for _ in range(10):
            await pool.forward({"prompt": "p"})
        await pool.aclose()

    asyncio.run(run())
    assert hits[:3] == ["u1", "u2", "u3"]  # unseen replicas are explored first
    assert hits[3:] == ["u2"] * 7


def test_timeout_retries_on_next_best_upstream_and_penalizes():
    hits: list[str] = []
    delays = {"u1": 5.0, "u2": 0.001}
    pool = UpstreamPool(urls=URLS[:2], transport=_transport(delays, hits),
                        attempt_timeout=0.05)

    async def run():
        result = await pool.forward({"prompt": "p"})
        follow_up = await pool.forward({"prompt": "p"})
        await pool.aclose()
        return result, follow_up

    result, follow_up = asyncio.run(run())
    assert result == (200, {"completion": "ok", "host": "u2"})
    assert follow_up[1]["host"] == "u2"
    assert hits == ["u1", "u2", "u2"]
    assert pool.score("http://u1:9000") > pool.score("http://u2:9000")


def test_connection_error_retries_once_then_propagates():
    hits: list[str] = []
    pool = UpstreamPool(urls=URLS, transport=_transport({}, hits, fail={"u1", "u2", "u3"}))

    async def run():
        try:
            with pytest.raises(httpx.ConnectError):
                await pool.forward({"prompt": "p"})
        finally:
            await pool.aclose()

    asyncio.run(run())
    assert hits == ["u1", "u2"]


def test_ewma_recovers_over_time():
    now = [0.0]
    pool = UpstreamPool(urls=URLS[:2], transport=_transport({}, []),
                        decay_half_life=5.0, clock=lambda: now[0])
    slow, fast = URLS[:2]
    pool._record(slow, 2.0)
    pool._record(fast, 0.15)
    assert pool._choose()[0][0] == fast

    now[0] = 20.0  # four half-lives idle: 2.0s decays to 0.125s
    assert pool.score(slow) == pytest.approx(0.125)
    for _ in range(10):  # meanwhile the fast replica keeps serving traffic
        pool._record(fast, 0.15)
    assert pool._choose()[0][0] == slow  # idle replica gets probed again

    pool._record(slow, 0.1)  # healthy probe keeps it in rotation
    assert pool.score(slow) < pool.score(fast)
    asyncio.run(pool.aclose())


def test_single_replica_gets_full_deadline():
    hits: list[str] = []
    pool = UpstreamPool(urls=URLS[:1], transport=_transport({"u1": 0.1}, hits),
                        attempt_timeout=0.05, retry_timeout=1.0)

    async def run():
        result = await pool.forward({"prompt": "p"})
        await pool.aclose()
        return result

    assert asyncio.run(run())[0] == 200
    assert hits == ["u1"]


def test_retry_prefers_measured_replica_over_unmeasured():
    pool = UpstreamPool(urls=URLS, transport=_transport({}, []))
    pool._record("http://u3:9000", 0.1)
    pool._last_ok["http://u3:9000"] = True
    first, retry = pool._choose()
    assert first[0] == "http://u1:9000"  # unmeasured: explored first
    assert retry[0] == "http://u3:9000"  # retry goes to the proven replica
    asyncio.run(pool.aclose())


def test_retry_prefers_unmeasured_over_failed_replica():
    urls = [f"http://u{i}:9000" for i in range(1, 5)]
    pool = UpstreamPool(urls=urls, transport=_transport({}, []))
    pool._record(urls[0], 1.0)
    pool._last_ok[urls[0]] = False
    pool._record(urls[1], 1.0)
    pool._last_ok[urls[1]] = False
    first, retry = pool._choose()
    assert first[0] == "http://u3:9000"
    assert retry[0] == "http://u4:9000"
    asyncio.run(pool.aclose())


def test_retry_falls_back_to_failed_replica_when_all_failed():
    pool = UpstreamPool(urls=URLS[:2], transport=_transport({}, []))
    for u, penalty in ((URLS[0], 1.0), (URLS[1], 2.0)):
        pool._record(u, penalty)
        pool._last_ok[u] = False
    first, retry = pool._choose()
    assert (first[0], retry[0]) == (URLS[0], URLS[1])
    asyncio.run(pool.aclose())
