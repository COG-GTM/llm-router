import asyncio
import json
import time

import httpx
from fastapi.testclient import TestClient

from app import main
from app.config import (
    upstream_connect_timeout_s,
    upstream_max_retries,
    upstream_timeout_s,
)
from app.upstream import UpstreamPool

URLS = ["http://u1:9000", "http://u2:9000", "http://u3:9000"]
TIMEOUT_S = 0.05


def _host(request: httpx.Request) -> str:
    return f"{request.url.scheme}://{request.url.host}:{request.url.port}"


def _pool(behaviour: dict, hits: list[str], **kwargs) -> UpstreamPool:
    """behaviour maps url -> "timeout" | "slow" | "connect" | int status (default 200)."""

    async def handler(request: httpx.Request) -> httpx.Response:
        url = _host(request)
        hits.append(url)
        action = behaviour.get(url, 200)
        if action == "timeout":
            raise httpx.ReadTimeout("read timed out", request=request)
        if action == "connect":
            raise httpx.ConnectError("connection refused", request=request)
        if action == "slow":
            await asyncio.sleep(TIMEOUT_S * 10)
            action = 200
        return httpx.Response(action, json={"completion": url})

    kwargs.setdefault("timeout_s", TIMEOUT_S)
    return UpstreamPool(urls=URLS, transport=httpx.MockTransport(handler), **kwargs)


def _forward(pool: UpstreamPool) -> tuple[int, dict]:
    async def run():
        try:
            return await pool.forward({"prompt": "p"})
        finally:
            await pool.aclose()

    return asyncio.run(run())


def test_timeout_retries_on_next_backend():
    hits: list[str] = []
    status, body = _forward(_pool({URLS[0]: "timeout"}, hits))
    assert (status, body) == (200, {"completion": URLS[1]})
    assert hits == URLS[:2]


def test_all_timeouts_return_504_after_max_retries_plus_one_attempts():
    hits: list[str] = []
    status, body = _forward(_pool({u: "timeout" for u in URLS}, hits))
    assert status == 504
    assert "timed out" in body["detail"]
    assert hits == URLS + URLS[:1]  # default max_retries = len(urls)


def test_all_5xx_return_502_with_explicit_max_retries():
    hits: list[str] = []
    status, body = _forward(_pool({u: 503 for u in URLS}, hits, max_retries=1))
    assert status == 502
    assert "HTTP 503" in body["detail"]
    assert hits == URLS[:2]


def test_retries_walk_from_own_start_despite_concurrent_requests():
    hits: list[tuple[str, str]] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        hits.append((json.loads(request.content)["prompt"], _host(request)))
        await asyncio.sleep(0.005)  # let the other requests advance the shared rotation
        raise httpx.ReadTimeout("read timed out", request=request)

    # High breaker threshold: this test is about walk order, not tripping.
    pool = UpstreamPool(urls=URLS, transport=httpx.MockTransport(handler), timeout_s=TIMEOUT_S,
                        failure_threshold=100)

    async def run():
        try:
            return await asyncio.gather(*(pool.forward({"prompt": str(i)}) for i in range(3)))
        finally:
            await pool.aclose()

    results = asyncio.run(run())
    assert [status for status, _ in results] == [504] * 3
    assert [p for p, _ in hits[:3]] == ["0", "1", "2"]  # attempts really interleave
    for i in range(3):
        mine = [url for p, url in hits if p == str(i)]
        assert mine == [URLS[(i + k) % len(URLS)] for k in range(len(URLS) + 1)]


def test_last_failure_decides_status():
    hits: list[str] = []
    behaviour = {URLS[0]: "timeout", URLS[1]: 503, URLS[2]: "timeout"}
    assert _forward(_pool(behaviour, hits, max_retries=2))[0] == 504
    behaviour = {URLS[0]: "timeout", URLS[1]: "timeout", URLS[2]: "connect"}
    assert _forward(_pool(behaviour, hits, max_retries=2))[0] == 502


def test_slow_backend_is_cut_off_by_attempt_bound():
    hits: list[str] = []
    start = time.monotonic()
    status, body = _forward(_pool({URLS[0]: "slow"}, hits))
    elapsed = time.monotonic() - start
    assert (status, body) == (200, {"completion": URLS[1]})
    assert hits == URLS[:2]
    assert elapsed < TIMEOUT_S + 0.1


def test_connect_error_retries_on_next_backend():
    hits: list[str] = []
    status, body = _forward(_pool({URLS[0]: "connect"}, hits))
    assert (status, body) == (200, {"completion": URLS[1]})
    assert hits == URLS[:2]


def test_4xx_is_returned_without_retry():
    hits: list[str] = []
    status, body = _forward(_pool({URLS[0]: 422}, hits))
    assert (status, body) == (422, {"completion": URLS[0]})
    assert hits == URLS[:1]


def test_generate_route_surfaces_504():
    hits: list[str] = []
    with TestClient(main.app) as c:
        main.app.state.pool = _pool({u: "timeout" for u in URLS}, hits)
        r = c.post("/v1/generate", json={"prompt": "hi", "max_tokens": 8})
    assert r.status_code == 504
    assert len(hits) == len(URLS) + 1


def test_env_knobs_fall_back_to_defaults(monkeypatch):
    monkeypatch.setenv("UPSTREAM_TIMEOUT_S", "0.5")
    monkeypatch.setenv("UPSTREAM_CONNECT_TIMEOUT_S", "0.2")
    monkeypatch.setenv("UPSTREAM_MAX_RETRIES", "1")
    assert (upstream_timeout_s(), upstream_connect_timeout_s(), upstream_max_retries(3)) == (
        0.5, 0.2, 1)
    monkeypatch.setenv("UPSTREAM_TIMEOUT_S", "abc")
    monkeypatch.setenv("UPSTREAM_CONNECT_TIMEOUT_S", "-1")
    monkeypatch.setenv("UPSTREAM_MAX_RETRIES", "-2")
    assert (upstream_timeout_s(), upstream_connect_timeout_s(), upstream_max_retries(3)) == (
        0.25, 0.1, 3)
