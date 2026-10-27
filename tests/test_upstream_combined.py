import asyncio
import time

import httpx
from fastapi.testclient import TestClient

from app import main
from app.upstream import CLOSED, OPEN, UpstreamPool

URLS = ["http://u1:9000", "http://u2:9000", "http://u3:9000"]
TIMEOUT_S = 0.05


def _host(request: httpx.Request) -> str:
    return f"{request.url.scheme}://{request.url.host}:{request.url.port}"


def _pool(bad: dict[str, str], hits: list[str], **kw) -> UpstreamPool:
    """bad maps url -> "hang" (outlives the attempt timeout) | "500"."""

    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/healthz":
            return httpx.Response(200, json={"ok": True})
        url = _host(request)
        hits.append(url)
        mode = bad.get(url)
        if mode == "hang":
            await asyncio.sleep(TIMEOUT_S * 10)
        if mode == "500":
            return httpx.Response(500, json={"detail": "boom"})
        return httpx.Response(200, json={"completion": url})

    kw.setdefault("slow_s", 1.0)
    kw.setdefault("failure_threshold", 3)
    return UpstreamPool(urls=URLS, transport=httpx.MockTransport(handler), timeout_s=TIMEOUT_S,
                        cooldown_s=5, probe_interval_s=3600, **kw)


def _backend(pool: UpstreamPool, url: str):
    return next(b for b in pool.backends if b.url == url)


async def _send(pool: UpstreamPool, n: int):
    return [await pool.forward({"prompt": "p"}) for _ in range(n)]


def test_timing_out_backend_is_tripped_then_skipped_by_picks_and_retries():
    hits: list[str] = []
    bad = {URLS[2]: "hang"}
    pool = _pool(bad, hits)

    async def run():
        results = await _send(pool, 9)
        assert all(status == 200 for status, _ in results)
        assert _backend(pool, URLS[2]).state == OPEN
        hits.clear()
        bad[URLS[1]] = "500"
        results = await _send(pool, 2)
        await pool.aclose()
        return results

    results = asyncio.run(run())
    assert [status for status, _ in results] == [200, 200]
    # u1 first pick; u2 first pick fails and the retry skips open u3 for u1.
    assert hits == [URLS[0], URLS[1], URLS[0]]


def test_failed_first_attempt_retries_and_counts_toward_breaker():
    hits: list[str] = []
    pool = _pool({URLS[0]: "500"}, hits)

    async def run():
        result = await pool.forward({"prompt": "p"})
        await pool.aclose()
        return result

    assert asyncio.run(run()) == (200, {"completion": URLS[1]})
    assert hits == URLS[:2]
    assert _backend(pool, URLS[0]).failures == 1
    assert _backend(pool, URLS[1]).state == CLOSED and _backend(pool, URLS[1]).failures == 0


def test_watchdog_and_timeout_count_one_failure_per_attempt():
    hits: list[str] = []
    pool = _pool({URLS[0]: "hang"}, hits, slow_s=TIMEOUT_S / 2)

    async def run():
        result = await pool.forward({"prompt": "p"})
        await pool.aclose()
        return result

    assert asyncio.run(run())[0] == 200
    assert _backend(pool, URLS[0]).failures == 1


def test_all_backends_open_returns_503_fast():
    hits: list[str] = []
    with TestClient(main.app) as c:
        main.app.state.pool = _pool(dict.fromkeys(URLS, "500"), hits, failure_threshold=1)
        r = c.post("/v1/generate", json={"prompt": "hi"})
        assert r.status_code == 502
        assert hits == URLS  # each attempt trips its backend; nothing left to retry
        start = time.monotonic()
        r = c.post("/v1/generate", json={"prompt": "hi"})
        elapsed = time.monotonic() - start
    assert r.status_code == 503
    assert r.json() == {"detail": "no healthy upstream"}
    assert elapsed < 0.1
    assert len(hits) == len(URLS)
