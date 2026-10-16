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


URLS = ["http://u1:9000", "http://u2:9000", "http://u3:9000"]


def _origin(request: httpx.Request) -> str:
    return f"{request.url.scheme}://{request.url.host}:{request.url.port}"


def _transport(hits: list[str], delays: dict[str, float] | None = None,
               failing: dict[str, str] | None = None) -> httpx.MockTransport:
    """Mock fleet: per-upstream simulated latency and failure modes ("503" or "timeout")."""
    delays = {} if delays is None else delays
    failing = {} if failing is None else failing

    async def handler(request: httpx.Request) -> httpx.Response:
        origin = _origin(request)
        hits.append(origin)
        await asyncio.sleep(delays.get(origin, 0.0))
        mode = failing.get(origin)
        if mode == "timeout":
            raise httpx.ReadTimeout("simulated timeout", request=request)
        if mode == "503":
            return httpx.Response(503, json={"detail": "model overloaded"})
        return httpx.Response(200, json={"completion": f"ok from {origin}"})

    return httpx.MockTransport(handler)


def _run(pool: UpstreamPool, calls):
    async def run():
        try:
            return [await pool.forward({"prompt": "p"}, key=k) for k in calls]
        finally:
            await pool.aclose()

    return asyncio.run(run())


def test_client_has_finite_timeout():
    pool = UpstreamPool(urls=URLS)
    assert pool._client.timeout.read == 30.0
    asyncio.run(pool.aclose())


def test_same_key_routes_to_same_upstream():
    hits: list[str] = []
    pool = UpstreamPool(urls=URLS, transport=_transport(hits))
    results = _run(pool, ["conv-42"] * 10)

    assert len(set(hits)) == 1
    assert all(status == 200 for status, _ in results)
    assert results[0][1] == {"completion": f"ok from {hits[0]}"}


def test_different_keys_distribute_across_upstreams():
    hits: list[str] = []
    pool = UpstreamPool(urls=URLS, transport=_transport(hits))
    keys = [f"conv-{i}" for i in range(300)]
    _run(pool, keys)

    counts = {u: hits.count(u) for u in URLS}
    assert all(n >= 50 for n in counts.values()), counts
    # Routing is a pure function of the key and the URL set.
    other = UpstreamPool(urls=list(reversed(URLS)))
    assert [other.candidates(k)[0] for k in keys] == hits
    asyncio.run(other.aclose())


def test_keyless_requests_pick_lowest_latency_upstream():
    hits: list[str] = []
    delays = {"http://u1:9000": 0.08, "http://u2:9000": 0.0, "http://u3:9000": 0.04}
    pool = UpstreamPool(urls=URLS, transport=_transport(hits, delays))
    _run(pool, [None] * 8)

    # First pass measures each upstream once, then traffic sticks to the fastest.
    assert sorted(hits[:3]) == URLS
    assert hits[3:] == ["http://u2:9000"] * 5


def test_keyless_failover_skips_cooling_upstream():
    hits: list[str] = []
    pool = UpstreamPool(urls=URLS, transport=_transport(hits, failing={"http://u1:9000": "503"}))
    results = _run(pool, [None, None])

    assert hits == ["http://u1:9000", "http://u2:9000", "http://u3:9000"]
    assert [status for status, _ in results] == [200, 200]


def test_keyed_failover_on_5xx_and_timeout():
    for mode in ("503", "timeout"):
        probe = UpstreamPool(urls=URLS)
        primary, secondary, _ = probe.candidates("conv-7")
        asyncio.run(probe.aclose())

        hits: list[str] = []
        pool = UpstreamPool(urls=URLS, transport=_transport(hits, failing={primary: mode}))
        results = _run(pool, ["conv-7", "conv-7"])

        # First request fails over; the second skips the cooling primary.
        assert hits == [primary, secondary, secondary], mode
        assert results == [(200, {"completion": f"ok from {secondary}"})] * 2


def test_primary_reused_after_cooldown():
    hits: list[str] = []
    failing = {}
    pool = UpstreamPool(urls=URLS, transport=_transport(hits, failing=failing), cooldown_s=0.0)
    primary, secondary, _ = pool.candidates("conv-7")
    failing[primary] = "503"

    async def run():
        await pool.forward({"prompt": "p"}, key="conv-7")
        failing.clear()
        await pool.forward({"prompt": "p"}, key="conv-7")
        await pool.aclose()

    asyncio.run(run())
    assert hits == [primary, secondary, primary]


def test_failover_is_bounded_and_returns_last_error():
    for mode, expected in (("503", 503), ("timeout", 504)):
        hits: list[str] = []
        pool = UpstreamPool(urls=URLS, transport=_transport(hits, failing={u: mode for u in URLS}))
        [(status, body)] = _run(pool, ["conv-1"])

        assert sorted(hits) == URLS
        assert status == expected
        assert "detail" in body


def test_4xx_is_passed_through_without_failover():
    hits: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        hits.append(_origin(request))
        return httpx.Response(422, json={"detail": "bad request"})

    pool = UpstreamPool(urls=URLS, transport=httpx.MockTransport(handler))
    assert _run(pool, [None]) == [(422, {"detail": "bad request"})]
    assert len(hits) == 1
