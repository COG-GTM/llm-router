import asyncio
import time

import httpx
import pytest
from fastapi.testclient import TestClient

from app import main
from app.upstream import CLOSED, HALF_OPEN, OPEN, NoHealthyUpstream, UpstreamPool

URLS = ["http://u1:9000", "http://u2:9000", "http://u3:9000"]


class FakeClock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def _host(request: httpx.Request) -> str:
    return f"{request.url.scheme}://{request.url.host}:{request.url.port}"


def _transport(hits: list[str], bad: dict[str, str], stall_s: float = 0.2):
    async def handler(request: httpx.Request) -> httpx.Response:
        host = _host(request)
        if request.url.path == "/healthz":
            return httpx.Response(200, json={"ok": True})
        hits.append(host)
        mode = bad.get(host)
        if mode == "500":
            return httpx.Response(500, json={"detail": "boom"})
        if mode == "stall":
            await asyncio.sleep(stall_s)
        return httpx.Response(200, json={"completion": "ok"})

    return httpx.MockTransport(handler)


def _pool(hits, bad, clock=None, **kw) -> UpstreamPool:
    kw.setdefault("slow_s", 0.05)
    return UpstreamPool(urls=URLS, transport=_transport(hits, bad), clock=clock or FakeClock(),
                        failure_threshold=3, cooldown_s=5, max_cooldown_s=60,
                        probe_interval_s=3600, **kw)


def _state(pool: UpstreamPool, url: str):
    return next(b for b in pool.backends if b.url == url)


async def _send(pool: UpstreamPool, n: int):
    return [await pool.forward({"prompt": "p"}) for _ in range(n)]


def test_failing_backend_is_excluded_after_threshold():
    hits: list[str] = []
    pool = _pool(hits, {URLS[1]: "500"})

    async def run():
        await _send(pool, 9)
        assert _state(pool, URLS[1]).state == OPEN
        hits.clear()
        results = await _send(pool, 6)
        await pool.aclose()
        return results

    results = asyncio.run(run())
    assert hits.count(URLS[1]) == 0
    assert hits.count(URLS[0]) == 3 and hits.count(URLS[2]) == 3
    assert all(r[0] == 200 for r in results)


def test_backend_rejoins_after_cooldown_and_probe():
    hits: list[str] = []
    bad = {URLS[1]: "500"}
    clock = FakeClock()
    pool = _pool(hits, bad, clock)

    async def run():
        await _send(pool, 9)
        backend = _state(pool, URLS[1])
        assert backend.state == OPEN
        await pool.probe()
        assert backend.state == OPEN
        clock.now += 5.1
        bad.clear()
        await pool.probe()
        assert backend.state == HALF_OPEN
        hits.clear()
        await _send(pool, 3)
        assert URLS[1] in hits
        assert backend.state == CLOSED and backend.trips == 0
        await pool.aclose()

    asyncio.run(run())


def test_stalling_backend_is_circuit_broken():
    hits: list[str] = []
    pool = _pool(hits, {URLS[2]: "stall"})

    async def run():
        results = await _send(pool, 9)
        assert _state(pool, URLS[2]).state == OPEN
        assert all(r[0] == 200 for r in results)
        hits.clear()
        await _send(pool, 6)
        await pool.aclose()

    asyncio.run(run())
    assert URLS[2] not in hits


def test_all_backends_open_raises_promptly():
    hits: list[str] = []
    pool = _pool(hits, dict.fromkeys(URLS, "500"))

    async def run():
        await _send(pool, 9)
        assert all(b.state == OPEN for b in pool.backends)
        start = time.monotonic()
        with pytest.raises(NoHealthyUpstream):
            await pool.forward({"prompt": "p"})
        assert time.monotonic() - start < 0.1
        await pool.aclose()

    asyncio.run(run())


def test_routes_return_503_when_no_healthy_upstream():
    with TestClient(main.app) as c:
        main.app.state.pool = UpstreamPool(
            urls=URLS[:1], transport=_transport([], {URLS[0]: "500"}), failure_threshold=1,
            probe_interval_s=3600)
        assert c.post("/v1/generate", json={"prompt": "hi"}).status_code == 500
        r = c.post("/v1/generate", json={"prompt": "hi"})
        assert r.status_code == 503
        assert r.json() == {"detail": "no healthy upstream"}
        cid = c.post("/v1/conversations", json={}).json()["id"]
        r = c.post("/v1/chat", json={"conversation_id": cid, "message": "hi"})
        assert r.status_code == 503
        assert r.json() == {"detail": "no healthy upstream"}


def test_half_open_failure_reopens_with_doubled_cooldown():
    hits: list[str] = []
    bad = {URLS[0]: "500"}
    clock = FakeClock()
    pool = _pool(hits, bad, clock)

    async def run():
        await _send(pool, 9)
        backend = _state(pool, URLS[0])
        assert backend.state == OPEN
        assert backend.open_until == pytest.approx(clock.now + 5)
        clock.now += 5.1
        await pool.probe()
        assert backend.state == HALF_OPEN
        await _send(pool, 1)
        assert backend.state == OPEN
        assert backend.open_until == pytest.approx(clock.now + 10)
        await pool.aclose()

    asyncio.run(run())
