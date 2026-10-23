import asyncio
import time

import httpx

from app.config import UpstreamSettings, upstream_settings, upstream_urls
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


URLS = ["http://u1:9000", "http://u2:9000", "http://u3:9000"]


def _settings(**overrides) -> UpstreamSettings:
    base = dict(attempt_timeout_s=0.25, final_timeout_s=30.0, max_attempts=0,
                breaker_threshold=3, breaker_cooldown_s=1.0, breaker_max_cooldown_s=30.0,
                reprobe_after_s=10.0)
    base.update(overrides)
    return UpstreamSettings(**base)


def _host(request: httpx.Request) -> str:
    return f"{request.url.scheme}://{request.url.host}:{request.url.port}"


def _run(pool: UpstreamPool, n: int = 1) -> list[tuple[int, dict]]:
    async def go():
        try:
            return [await pool.forward({"prompt": "p"}) for _ in range(n)]
        finally:
            await pool.aclose()
    return asyncio.run(go())


def test_settings_from_env_fall_back_on_bad_values(monkeypatch):
    monkeypatch.setenv("UPSTREAM_ATTEMPT_TIMEOUT_MS", "400")
    monkeypatch.setenv("UPSTREAM_FINAL_TIMEOUT_MS", "soon")
    monkeypatch.setenv("UPSTREAM_MAX_ATTEMPTS", "")
    s = upstream_settings()
    assert s.attempt_timeout_s == 0.4
    assert s.final_timeout_s == 30.0
    assert s.max_attempts == 0


def test_client_has_explicit_timeout():
    pool = UpstreamPool(urls=URLS, settings=_settings())
    assert pool._client.timeout.read == 30.0
    asyncio.run(pool.aclose())


def test_timeout_fails_over_to_next_backend_with_short_attempt_timeout():
    calls: list[tuple[str, float]] = []

    async def handler(request):
        calls.append((_host(request), request.extensions["timeout"]["read"]))
        if request.url.host == "u1":
            raise httpx.ReadTimeout("slow", request=request)
        return httpx.Response(200, json={"completion": "ok"})

    pool = UpstreamPool(urls=URLS, transport=httpx.MockTransport(handler), settings=_settings())
    assert _run(pool) == [(200, {"completion": "ok"})]
    assert calls == [("http://u1:9000", 0.25), ("http://u2:9000", 0.25)]


def test_final_attempt_uses_long_timeout_when_every_backend_is_slow():
    calls: list[float] = []

    async def handler(request):
        read = request.extensions["timeout"]["read"]
        calls.append(read)
        if read < 1.3:  # every backend takes longer than the per-attempt cap
            raise httpx.ReadTimeout("slow", request=request)
        return httpx.Response(200, json={"completion": "late"})

    pool = UpstreamPool(urls=URLS, transport=httpx.MockTransport(handler), settings=_settings())
    assert _run(pool) == [(200, {"completion": "late"})]
    assert calls == [0.25, 0.25, 30.0]


def test_5xx_fails_over_and_attempts_are_bounded():
    hits: list[str] = []

    async def handler(request):
        hits.append(_host(request))
        return httpx.Response(503, json={"detail": "overloaded"})

    pool = UpstreamPool(urls=URLS, transport=httpx.MockTransport(handler), settings=_settings())
    assert _run(pool) == [(503, {"detail": "overloaded"})]
    assert len(hits) == 3


def test_connection_errors_everywhere_return_502():
    async def handler(request):
        raise httpx.ConnectError("refused", request=request)

    pool = UpstreamPool(urls=URLS, transport=httpx.MockTransport(handler),
                        settings=_settings(max_attempts=5))
    assert _run(pool) == [(502, {"detail": "upstream unavailable"})]


def test_4xx_is_passed_through_without_retry():
    hits: list[str] = []

    async def handler(request):
        hits.append(_host(request))
        return httpx.Response(422, json={"detail": "bad"})

    pool = UpstreamPool(urls=URLS, transport=httpx.MockTransport(handler), settings=_settings())
    assert _run(pool) == [(422, {"detail": "bad"})]
    assert hits == ["http://u1:9000"]


def test_slow_backend_is_avoided_by_latency_aware_selection():
    hits: list[str] = []

    async def handler(request):
        hits.append(_host(request))
        if request.url.host == "u3":
            await asyncio.sleep(0.05)
        return httpx.Response(200, json={"completion": "ok"})

    pool = UpstreamPool(urls=URLS, transport=httpx.MockTransport(handler), settings=_settings())
    _run(pool, 9)
    assert hits[:3] == URLS
    assert "http://u3:9000" not in hits[3:]
    assert hits[3:] == ["http://u1:9000", "http://u2:9000"] * 3


def test_fast_5xx_does_not_look_fast():
    hits: list[str] = []
    fail = {"u1": True}

    async def handler(request):
        hits.append(_host(request))
        if fail.get(request.url.host):
            fail["u1"] = False
            return httpx.Response(500, json={"detail": "boom"})
        return httpx.Response(200, json={"completion": "ok"})

    pool = UpstreamPool(urls=URLS[:2], transport=httpx.MockTransport(handler),
                        settings=_settings())
    _run(pool, 4)
    assert hits[0] == "http://u1:9000"
    assert hits[1:] == ["http://u2:9000"] * 4


def test_unmeasured_backend_gets_one_probe_at_a_time():
    release = asyncio.Event()
    hits: list[str] = []

    async def handler(request):
        hits.append(_host(request))
        if request.url.host == "u1":
            await release.wait()
        return httpx.Response(200, json={"completion": "ok"})

    pool = UpstreamPool(urls=URLS[:2], transport=httpx.MockTransport(handler),
                        settings=_settings(attempt_timeout_s=5.0))
    pool._backends[1].record(0.1, time.monotonic())

    async def go():
        probe = asyncio.create_task(pool.forward({"prompt": "p"}))
        await asyncio.sleep(0.01)
        others = [await pool.forward({"prompt": "p"}) for _ in range(3)]
        release.set()
        await probe
        await pool.aclose()
        return others

    asyncio.run(go())
    assert hits == ["http://u1:9000"] + ["http://u2:9000"] * 3


def test_circuit_breaker_skips_failing_backend_then_probes_again():
    hits: list[str] = []
    down = {"u1": True}

    async def handler(request):
        hits.append(_host(request))
        if down.get(request.url.host):
            raise httpx.ConnectError("refused", request=request)
        return httpx.Response(200, json={"completion": "ok"})

    pool = UpstreamPool(urls=URLS[:2], transport=httpx.MockTransport(handler),
                        settings=_settings(breaker_threshold=1, breaker_cooldown_s=0.05))

    async def go():
        assert await pool.forward({"prompt": "p"}) == (200, {"completion": "ok"})
        assert hits == ["http://u1:9000", "http://u2:9000"]
        down["u1"] = False
        u1 = pool._backends[0]
        u1.ewma = 0.0  # even when it looks cheapest, an open breaker skips it
        await pool.forward({"prompt": "p"})
        assert hits[-1] == "http://u2:9000"
        await asyncio.sleep(0.06)
        u1.ewma = 0.0
        await pool.forward({"prompt": "p"})
        assert hits[-1] == "http://u1:9000"
        assert u1.failures == 0
        await pool.aclose()

    asyncio.run(go())
