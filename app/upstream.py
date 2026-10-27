import asyncio
import contextlib
import time
from collections.abc import Callable

import httpx

from .config import (
    cb_cooldown_s,
    cb_failure_threshold,
    cb_max_cooldown_s,
    upstream_connect_timeout_s,
    upstream_max_retries,
    upstream_slow_s,
    upstream_timeout_s,
    upstream_urls,
)

CLOSED, OPEN, HALF_OPEN = "closed", "open", "half-open"


class NoHealthyUpstream(Exception):
    pass


class Backend:
    def __init__(self, url: str):
        self.url = url
        self.state = CLOSED
        self.failures = 0
        self.open_until = 0.0
        self.trips = 0


class UpstreamPool:
    def __init__(self, urls: list[str] | None = None,
                 transport: httpx.AsyncBaseTransport | None = None, *,
                 clock: Callable[[], float] = time.monotonic,
                 slow_s: float | None = None,
                 failure_threshold: int | None = None,
                 cooldown_s: float | None = None,
                 max_cooldown_s: float | None = None,
                 probe_interval_s: float = 1.0,
                 timeout_s: float | None = None,
                 max_retries: int | None = None):
        self.backends = [Backend(u) for u in (urls or upstream_urls())]
        self._next = 0
        self._timeout_s = timeout_s if timeout_s is not None else upstream_timeout_s()
        self._max_retries = (max_retries if max_retries is not None
                             else upstream_max_retries(len(self.backends)))
        connect_s = min(upstream_connect_timeout_s(), self._timeout_s)
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(self._timeout_s, connect=connect_s), transport=transport
        )
        self._clock = clock
        self._slow_s = slow_s or upstream_slow_s()
        self._threshold = failure_threshold or cb_failure_threshold()
        self._cooldown_s = cooldown_s or cb_cooldown_s()
        self._max_cooldown_s = max_cooldown_s or cb_max_cooldown_s()
        self._probe_interval_s = probe_interval_s
        self._probe_task: asyncio.Task | None = None

    def _pick(self) -> Backend:
        n = len(self.backends)
        for i in range(n):
            backend = self.backends[(self._next + i) % n]
            if backend.state != OPEN:
                self._next = (self._next + i + 1) % n
                return backend
        raise NoHealthyUpstream("all upstream circuits are open")

    def _retry_pick(self, failed: Backend, tried: set[int]) -> Backend | None:
        # Walk on from the backend that just failed, skipping open circuits;
        # prefer backends this request hasn't tried yet.
        n = len(self.backends)
        i = self.backends.index(failed)
        ring = [self.backends[(i + k) % n] for k in range(1, n + 1)]
        healthy = [b for b in ring if b.state != OPEN]
        fresh = [b for b in healthy if id(b) not in tried]
        return (fresh or healthy or [None])[0]

    def _trip(self, backend: Backend) -> None:
        cooldown = min(self._cooldown_s * 2 ** min(backend.trips, 30), self._max_cooldown_s)
        backend.trips += 1
        backend.state = OPEN
        backend.failures = 0
        backend.open_until = self._clock() + cooldown

    def _record_failure(self, backend: Backend) -> None:
        if backend.state == OPEN:
            return
        backend.failures += 1
        if backend.state == HALF_OPEN or backend.failures >= self._threshold:
            self._trip(backend)

    def _record_success(self, backend: Backend) -> None:
        if backend.state != OPEN:
            backend.state = CLOSED
            backend.failures = 0
            backend.trips = 0

    def _ensure_probe(self) -> None:
        if self._probe_task is None or self._probe_task.done():
            self._probe_task = asyncio.get_running_loop().create_task(self._probe_loop())

    async def _probe_loop(self) -> None:
        while True:
            await asyncio.sleep(self._probe_interval_s)
            await self.probe()

    async def probe(self) -> None:
        now = self._clock()
        due = [b for b in self.backends if b.state == OPEN and now >= b.open_until]
        await asyncio.gather(*(self._probe_one(b) for b in due))

    async def _probe_one(self, backend: Backend) -> None:
        try:
            resp = await self._client.get(f"{backend.url}/healthz", timeout=self._slow_s)
            healthy = resp.is_success
        except httpx.HTTPError:
            healthy = False
        if backend.state != OPEN:
            return
        if healthy:
            backend.state = HALF_OPEN
            backend.failures = 0
        else:
            self._trip(backend)

    async def _attempt(self, backend: Backend, payload: dict) -> tuple[int, dict | None]:
        slow = False

        def on_slow() -> None:
            nonlocal slow
            slow = True
            self._record_failure(backend)

        watchdog = asyncio.get_running_loop().call_later(self._slow_s, on_slow)
        try:
            # httpx timeouts are per phase/chunk; bound the whole attempt as well.
            async with asyncio.timeout(self._timeout_s):
                resp = await self._client.post(f"{backend.url}/v1/completions", json=payload)
        except (httpx.RequestError, TimeoutError):
            if not slow:
                self._record_failure(backend)
            raise
        finally:
            watchdog.cancel()
        if resp.status_code >= 500:
            if not slow:
                self._record_failure(backend)
            return resp.status_code, None
        if not slow:
            self._record_success(backend)
        return resp.status_code, resp.json()

    async def forward(self, payload: dict) -> tuple[int, dict]:
        self._ensure_probe()
        backend: Backend | None = self._pick()
        tried: set[int] = set()
        last_failure = "unavailable"
        for _ in range(self._max_retries + 1):
            if backend is None:
                break
            tried.add(id(backend))
            try:
                status, body = await self._attempt(backend, payload)
            except (httpx.TimeoutException, TimeoutError):
                last_failure = "timeout"
            except httpx.RequestError:
                last_failure = "unavailable"
            else:
                if status < 500:
                    return status, body
                last_failure = f"HTTP {status}"
            backend = self._retry_pick(backend, tried)
        attempts = len(tried)
        if last_failure == "timeout":
            return 504, {"detail": f"upstream timed out after {attempts} attempts"}
        return 502, {"detail": f"upstream failed after {attempts} attempts ({last_failure})"}

    async def aclose(self) -> None:
        if self._probe_task is not None:
            self._probe_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, RuntimeError):
                await self._probe_task
        await self._client.aclose()
