import asyncio
import time
from itertools import cycle, islice

import httpx

from .config import hedge_delay_s, max_attempts, upstream_timeout_s, upstream_urls

_EWMA_ALPHA = 0.2
_SLOW_FACTOR = 1.5      # upstream is "slow" if its EWMA exceeds best * factor ...
_SLOW_SLACK_S = 0.05    # ... and best + slack
_SLOW_DECAY = 0.98      # skipped slow upstreams drift back so they get re-probed


class UpstreamPool:
    """Round-robin over healthy upstreams with latency-aware ejection and hedged requests.

    If an attempt hasn't answered within `hedge_delay`, a backup attempt is sent to the
    next healthy upstream (up to `max_attempts`); the first response wins and the rest
    are cancelled. Per-upstream latency EWMAs (including censored samples from cancelled
    attempts) move persistently slow upstreams to the back of the rotation.
    """

    def __init__(self, urls: list[str] | None = None,
                 transport: httpx.AsyncBaseTransport | None = None,
                 hedge_delay: float | None = None,
                 attempts: int | None = None):
        self._urls = list(urls or upstream_urls())
        self._next = 0
        self._ewma: dict[str, float] = {}
        self._hedge_delay = hedge_delay_s() if hedge_delay is None else hedge_delay
        self._max_attempts = max_attempts() if attempts is None else attempts
        self._client = httpx.AsyncClient(timeout=upstream_timeout_s(), transport=transport)

    def _plan(self) -> list[str]:
        start = self._next
        self._next = (self._next + 1) % len(self._urls)
        rotated = self._urls[start:] + self._urls[:start]
        if not self._ewma:
            healthy, slow = rotated, []
        else:
            best = min(self._ewma.values())
            cutoff = max(best * _SLOW_FACTOR, best + _SLOW_SLACK_S)
            healthy = [u for u in rotated if self._ewma.get(u, 0.0) <= cutoff]
            slow = sorted((u for u in rotated if u not in healthy), key=self._ewma.__getitem__)
            for u in slow:
                self._ewma[u] *= _SLOW_DECAY
        # Distinct healthy upstreams first, then re-tries on healthy ones, with slow
        # upstreams kept as the last hedges so they are still reachable.
        repeats = max(0, self._max_attempts - len(healthy) - len(slow))
        targets = healthy + list(islice(cycle(healthy), repeats)) + slow
        return targets[:self._max_attempts]

    def _record(self, url: str, seconds: float) -> None:
        prev = self._ewma.get(url)
        self._ewma[url] = seconds if prev is None else prev + _EWMA_ALPHA * (seconds - prev)

    async def _attempt(self, url: str, payload: dict) -> tuple[int, dict]:
        resp = await self._client.post(f"{url}/v1/completions", json=payload)
        return resp.status_code, resp.json()

    async def forward(self, payload: dict) -> tuple[int, dict]:
        targets = self._plan()
        pending: dict[asyncio.Task, tuple[str, float]] = {}
        launched = 0
        last_exc: BaseException | None = None

        def launch() -> None:
            nonlocal launched
            url = targets[launched]
            launched += 1
            task = asyncio.create_task(self._attempt(url, payload))
            pending[task] = (url, time.monotonic())

        launch()
        try:
            while pending:
                can_hedge = launched < len(targets)
                done, _ = await asyncio.wait(
                    pending, timeout=self._hedge_delay if can_hedge else None,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if not done:
                    launch()
                    continue
                for task in done:
                    url, started = pending.pop(task)
                    if task.exception() is None:
                        self._record(url, time.monotonic() - started)
                        return task.result()
                    last_exc = task.exception()
                    self._record(url, upstream_timeout_s())
                if launched < len(targets):
                    launch()
            assert last_exc is not None
            raise last_exc
        finally:
            now = time.monotonic()
            for task, (url, started) in pending.items():
                task.cancel()
                self._record(url, now - started)

    async def aclose(self) -> None:
        await self._client.aclose()
