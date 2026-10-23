import asyncio

import httpx

from app.upstream import UpstreamPool

URLS = ["http://u1:9000", "http://u2:9000", "http://u3:9000"]


def _transport(hits: list[tuple[str, float]], behaviour: dict) -> httpx.MockTransport:
    """Per-host behaviour: a latency in seconds (times out past the read timeout),
    an HTTP status code, or an exception to raise."""

    async def handler(request: httpx.Request) -> httpx.Response:
        host = request.url.host
        read_timeout = request.extensions["timeout"]["read"]
        hits.append((host, read_timeout))
        action = behaviour.get(host, 0.0)
        if isinstance(action, Exception):
            raise action
        if isinstance(action, int):
            return httpx.Response(action, json={"detail": f"{host} failed"})
        if read_timeout is not None and action > read_timeout:
            raise httpx.ReadTimeout("timed out", request=request)
        return httpx.Response(200, json={"completion": host})

    return httpx.MockTransport(handler)


def _forward(behaviour: dict, n: int = 1, **kwargs):
    hits: list[tuple[str, float]] = []
    kwargs.setdefault("attempt_timeout", 0.25)
    kwargs.setdefault("final_timeout", 30.0)
    pool = UpstreamPool(urls=URLS, transport=_transport(hits, behaviour), **kwargs)

    async def run():
        results = [await pool.forward({"prompt": "p"}) for _ in range(n)]
        await pool.aclose()
        return results

    return asyncio.run(run()), hits


def test_timeout_fails_over_to_next_backend():
    results, hits = _forward({"u1": 1.5})
    assert results == [(200, {"completion": "u2"})]
    assert hits == [("u1", 0.25), ("u2", 0.25)]


def test_healthy_backend_serves_retry_and_round_robin_continues():
    results, hits = _forward({"u1": 1.5}, n=3)
    assert results == [(200, {"completion": c}) for c in ("u2", "u2", "u3")]
    assert [h for h, _ in hits] == ["u1", "u2", "u2", "u3"]


def test_last_backend_gets_final_timeout():
    results, hits = _forward({"u1": 1.5, "u2": 1.5, "u3": 1.5})
    assert results == [(200, {"completion": "u3"})]
    assert hits == [("u1", 0.25), ("u2", 0.25), ("u3", 30.0)]


def test_5xx_fails_over_but_4xx_passes_through():
    results, hits = _forward({"u1": 503})
    assert results == [(200, {"completion": "u2"})]
    assert [h for h, _ in hits] == ["u1", "u2"]

    results, hits = _forward({"u1": 422})
    assert results == [(422, {"detail": "u1 failed"})]
    assert [h for h, _ in hits] == ["u1"]


def test_connection_error_fails_over():
    results, _ = _forward({"u1": httpx.ConnectError("refused")})
    assert results == [(200, {"completion": "u2"})]


def test_all_backends_failing_returns_error_without_raising():
    results, hits = _forward({"u1": 500, "u2": 500, "u3": 503})
    assert results == [(503, {"detail": "u3 failed"})]
    assert [h for h, _ in hits] == ["u1", "u2", "u3"]

    results, _ = _forward({"u1": 1.5, "u2": 1.5, "u3": 60.0})
    assert results == [(504, {"detail": "upstream timeout"})]

    refused = httpx.ConnectError("refused")
    results, _ = _forward({"u1": refused, "u2": refused, "u3": refused})
    assert results == [(502, {"detail": "upstream unavailable"})]


def test_failed_attempts_are_not_cached_but_failover_success_is():
    results, hits = _forward({"u1": 503}, n=2, cache_size=16, cache_ttl=60)
    assert results == [(200, {"completion": "u2"})] * 2
    assert [h for h, _ in hits] == ["u1", "u2"]

    results, hits = _forward({"u1": 503, "u2": 503, "u3": 503}, n=2, cache_size=16, cache_ttl=60)
    assert all(status == 503 for status, _ in results)
    assert len(hits) == 6
