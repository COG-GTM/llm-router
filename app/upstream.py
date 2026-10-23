import logging
import time

import httpx

from .config import UpstreamSettings, upstream_settings, upstream_urls

log = logging.getLogger(__name__)

_EWMA_ALPHA = 0.3
_COST_FLOOR_S = 0.01  # latencies below this are noise; keeps fast backends tied
_TIE_RATIO = 1.25  # costs within 25% of the best are ties, broken round-robin
_FAILURE_PENALTY_S = 1.0  # a failed attempt never looks fast, however quickly it failed


class _Backend:
    def __init__(self, url: str):
        self.url = url
        self.in_flight = 0
        self.ewma: float | None = None
        self.last_sample = 0.0
        self.failures = 0
        self.open_until = 0.0

    def measured(self, now: float, reprobe_after: float) -> bool:
        return self.ewma is not None and now - self.last_sample < reprobe_after

    def cost(self, now: float, reprobe_after: float) -> float:
        latency = self.ewma if self.measured(now, reprobe_after) else 0.0
        return max(latency, _COST_FLOOR_S) * (self.in_flight + 1)

    def record(self, sample: float, now: float) -> None:
        self.ewma = sample if self.ewma is None else (
            _EWMA_ALPHA * sample + (1 - _EWMA_ALPHA) * self.ewma)
        self.last_sample = now


class _AttemptFailed(Exception):
    def __init__(self, reason: str, response: tuple[int, dict] | None = None):
        super().__init__(reason)
        self.response = response


class UpstreamPool:
    """Least-loaded routing with per-attempt timeouts, failover and a passive breaker."""

    def __init__(self, urls: list[str] | None = None,
                 transport: httpx.AsyncBaseTransport | None = None,
                 settings: UpstreamSettings | None = None):
        self._cfg = settings or upstream_settings()
        self._backends = [_Backend(u) for u in (urls or upstream_urls())]
        self._rr = 0
        self._max_attempts = self._cfg.max_attempts or len(self._backends)
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(self._cfg.final_timeout_s), transport=transport)

    def _pick(self, tried: set[int], final: bool) -> int:
        now = time.monotonic()
        reprobe = self._cfg.reprobe_after_s
        idx = range(len(self._backends))
        closed = [i for i in idx if self._backends[i].open_until <= now]
        if final:
            # The final attempt may run long: prefer backends with a known good
            # latency, even ones already tried, over unmeasured or failing ones.
            pool = closed or list(idx)
            good = [i for i in pool if self._backends[i].measured(now, reprobe)
                    and self._backends[i].ewma < _FAILURE_PENALTY_S]
            pool = good or pool
        else:
            pool = [i for i in closed if i not in tried] or closed or [
                i for i in idx if i not in tried] or list(idx)
            # Only one in-flight probe per unmeasured backend, so a hanging
            # first probe doesn't soak up traffic priced at the cost floor.
            probing = [i for i in pool if not self._backends[i].measured(now, reprobe)
                       and self._backends[i].in_flight > 0]
            pool = [i for i in pool if i not in probing] or pool
        costs = {i: self._backends[i].cost(now, reprobe) for i in pool}
        best = min(costs.values())
        n = len(self._backends)
        for k in range(n):
            i = (self._rr + k) % n
            if i in costs and costs[i] <= best * _TIE_RATIO:
                self._rr = i + 1
                return i
        raise AssertionError("unreachable")

    async def _attempt(self, b: _Backend, payload: dict, timeout: float) -> tuple[int, dict]:
        start = time.monotonic()
        b.in_flight += 1
        try:
            resp = await self._client.post(f"{b.url}/v1/completions", json=payload,
                                           timeout=httpx.Timeout(timeout))
            if resp.status_code >= 500:
                try:
                    body = resp.json()
                except ValueError:
                    body = {"detail": "upstream error"}
                raise _AttemptFailed(f"HTTP {resp.status_code}", (resp.status_code, body))
            try:
                body = resp.json()
            except ValueError:
                raise _AttemptFailed("invalid JSON", (502, {"detail": "invalid upstream response"}))
        except httpx.TimeoutException:
            self._failed(b, start)
            raise _AttemptFailed("timeout", (504, {"detail": "upstream timeout"}))
        except httpx.TransportError as e:
            self._failed(b, start)
            raise _AttemptFailed(type(e).__name__, (502, {"detail": "upstream unavailable"}))
        except _AttemptFailed:
            self._failed(b, start)
            raise
        finally:
            b.in_flight -= 1
        now = time.monotonic()
        elapsed = now - start
        b.record(elapsed, now)
        # A slow success (only possible on the long final attempt) doesn't
        # prove the backend healthy, so it doesn't reset the breaker backoff.
        if elapsed <= self._cfg.attempt_timeout_s:
            b.failures = 0
        return resp.status_code, body

    def _failed(self, b: _Backend, start: float) -> None:
        now = time.monotonic()
        b.record(max(now - start, _FAILURE_PENALTY_S), now)
        b.failures += 1
        excess = b.failures - self._cfg.breaker_threshold
        if excess >= 0:
            cooldown = min(self._cfg.breaker_cooldown_s * 2 ** min(excess, 16),
                           self._cfg.breaker_max_cooldown_s)
            b.open_until = now + cooldown

    async def forward(self, payload: dict) -> tuple[int, dict]:
        tried: set[int] = set()
        failure = _AttemptFailed("no attempt")
        for attempt in range(self._max_attempts):
            final = attempt == self._max_attempts - 1
            i = self._pick(tried, final)
            tried.add(i)
            b = self._backends[i]
            timeout = self._cfg.final_timeout_s if final else self._cfg.attempt_timeout_s
            try:
                return await self._attempt(b, payload, timeout)
            except _AttemptFailed as e:
                log.info("upstream attempt %d/%d to %s failed: %s",
                         attempt + 1, self._max_attempts, b.url, e)
                failure = e
        return failure.response or (502, {"detail": "upstream unavailable"})

    async def aclose(self) -> None:
        await self._client.aclose()
