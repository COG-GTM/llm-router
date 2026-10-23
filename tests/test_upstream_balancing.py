import asyncio

import httpx

from app.upstream import UpstreamPool

URLS = ["http://u1:9000", "http://u2:9000", "http://u3:9000"]


def _pool(hits: list[str], behaviour: dict) -> UpstreamPool:
    """Per-host behaviour: a latency in seconds (times out past the read timeout)
    or an asyncio.Event the response waits for."""

    async def handler(request: httpx.Request) -> httpx.Response:
        host = request.url.host
        hits.append(host)
        action = behaviour.get(host, 0.0)
        if isinstance(action, asyncio.Event):
            await action.wait()
        else:
            if action > request.extensions["timeout"]["read"]:
                raise httpx.ReadTimeout("timed out", request=request)
            await asyncio.sleep(action)
        return httpx.Response(200, json={"completion": host})

    return UpstreamPool(urls=URLS, transport=httpx.MockTransport(handler),
                        attempt_timeout=0.25, final_timeout=30.0)


def test_slow_backend_is_avoided_once_measured():
    hits: list[str] = []
    pool = _pool(hits, {"u1": 0.05})

    async def run():
        results = [await pool.forward({"prompt": "p"}) for _ in range(9)]
        await pool.aclose()
        return results

    results = asyncio.run(run())
    assert all(status == 200 for status, _ in results)
    assert hits[:3] == ["u1", "u2", "u3"]
    assert "u1" not in hits[3:]
    assert {"u2", "u3"} <= set(hits[3:])


def test_busy_backend_is_skipped_while_in_flight():
    hits: list[str] = []

    async def run():
        release = asyncio.Event()
        pool = _pool(hits, {"u1": release})
        blocked = asyncio.create_task(pool.forward({"prompt": "p"}))
        while not hits:
            await asyncio.sleep(0)
        others = [await pool.forward({"prompt": "p"}) for _ in range(4)]
        release.set()
        first = await blocked
        await pool.aclose()
        return first, others

    first, others = asyncio.run(run())
    assert first == (200, {"completion": "u1"})
    assert [body["completion"] for _, body in others] == ["u2", "u3", "u2", "u3"]
    assert hits.count("u1") == 1


def test_retry_picks_fastest_untried_and_timeout_is_penalised():
    hits: list[str] = []
    behaviour = {"u2": 0.05}
    pool = _pool(hits, behaviour)

    async def run():
        warmup = [await pool.forward({"prompt": "p"}) for _ in range(3)]
        behaviour["u1"] = 1.5
        retried = await pool.forward({"prompt": "p"})
        after = [await pool.forward({"prompt": "p"}) for _ in range(3)]
        await pool.aclose()
        return warmup, retried, after

    warmup, retried, after = asyncio.run(run())
    assert [body["completion"] for _, body in warmup] == ["u1", "u2", "u3"]
    # u1 times out; the retry skips u2 (next in rotation but measured slower) for u3.
    assert retried == (200, {"completion": "u3"})
    assert hits[3:5] == ["u1", "u3"]
    assert "u1" not in hits[5:]
    assert all(status == 200 for status, _ in after)
