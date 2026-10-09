"""Latency-aware load balancing across llm-service replicas."""
import asyncio
import logging
import math
import time
from contextvars import ContextVar

import httpx

from .config import upstream_urls

logger = logging.getLogger("llm-router.upstream")

# RequestLogMiddleware installs a fresh dict per request; forward() records the
# upstream that served it so the request log line can attribute latency.
upstream_attribution: ContextVar[dict | None] = ContextVar("upstream_attribution", default=None)


class UpstreamPool:
    """Routes each request to the replica with the lowest latency EWMA.

    Each attempt is bounded; on timeout or connection error the request is
    retried once on the next-best replica and the failed one is penalized,
    doubling per consecutive failure so a persistently slow replica backs off.
    Scores decay toward zero while a replica is idle, so a penalized or slow
    replica is eventually probed again and can recover.
    """

    def __init__(self, urls: list[str] | None = None,
                 transport: httpx.AsyncBaseTransport | None = None, *,
                 alpha: float = 0.3,
                 attempt_timeout: float = 0.5,
                 retry_timeout: float = 5.0,
                 failure_penalty: float = 2.0,
                 max_backoff_doublings: int = 4,
                 decay_half_life: float = 10.0,
                 clock=time.monotonic):
        self._urls = list(urls or upstream_urls())
        self._alpha = alpha
        self._attempt_timeout = attempt_timeout
        self._retry_timeout = retry_timeout
        self._failure_penalty = failure_penalty
        self._max_doublings = max_backoff_doublings
        self._half_life = decay_half_life
        self._clock = clock
        self._ewma: dict[str, float] = {u: 0.0 for u in self._urls}
        self._updated: dict[str, float] = {}
        self._failures: dict[str, int] = {u: 0 for u in self._urls}
        self._client = httpx.AsyncClient(timeout=None, transport=transport)

    def score(self, url: str) -> float:
        """Current EWMA latency (seconds), decayed by time since last update."""
        updated = self._updated.get(url)
        if updated is None:
            return 0.0
        idle = self._clock() - updated
        return self._ewma[url] * math.pow(0.5, idle / self._half_life)

    def _record(self, url: str, sample: float) -> None:
        prev = self.score(url) if url in self._updated else sample
        self._ewma[url] = self._alpha * sample + (1 - self._alpha) * prev
        self._updated[url] = self._clock()

    def _choose(self) -> list[tuple[str, float]]:
        # The first attempt explores via decayed scores; the retry goes to the
        # replica with the best observed latency so it is never a probe.
        first = min(self._urls, key=self.score)
        attempts = [(first, self._attempt_timeout)]
        rest = [u for u in self._urls if u != first]
        if rest:
            attempts.append((min(rest, key=lambda u: self._ewma[u]), self._retry_timeout))
        return attempts

    async def _attempt(self, url: str, payload: dict, timeout: float) -> httpx.Response:
        start = self._clock()
        try:
            async with asyncio.timeout(timeout):
                resp = await self._client.post(f"{url}/v1/completions", json=payload)
        except (TimeoutError, httpx.TransportError) as exc:
            doublings = min(self._failures[url], self._max_doublings)
            self._failures[url] += 1
            self._record(url, timeout * self._failure_penalty * 2 ** doublings)
            logger.warning("upstream failed url=%s error=%s", url, type(exc).__name__)
            raise
        elapsed = self._clock() - start
        if elapsed <= self._attempt_timeout:
            self._failures[url] = 0
        self._record(url, elapsed)
        return resp

    async def forward(self, payload: dict) -> tuple[int, dict]:
        attempts = self._choose()
        attribution = upstream_attribution.get()
        for i, (url, timeout) in enumerate(attempts, start=1):
            if attribution is not None:
                attribution.update(upstream=url, attempts=i)
            try:
                resp = await self._attempt(url, payload, timeout)
            except (TimeoutError, httpx.TransportError):
                if i == len(attempts):
                    raise
                continue
            return resp.status_code, resp.json()
        raise AssertionError("unreachable")

    async def aclose(self) -> None:
        await self._client.aclose()
