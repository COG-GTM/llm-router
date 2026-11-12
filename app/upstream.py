import random
import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass

import httpx

from .config import upstream_urls


@dataclass
class _Backend:
    url: str
    ewma_ms: float | None = None
    samples: int = 0
    inflight: int = 0
    consecutive_failures: int = 0
    ejected_until: float = 0.0
    ejections: int = 0
    probing: bool = False
    stale_probe: bool = False
    last_picked: float = 0.0


class UpstreamPool:
    """Latency-aware pool: EWMA + power-of-two-choices with outlier ejection.

    Each request picks two random eligible backends and sends to the one with
    the lower `ewma_ms * (inflight + 1)`. A backend whose EWMA is far above the
    fleet median (or that keeps returning 5xx) is ejected for a while; when the
    ejection expires a single probe request decides whether it is reinstated
    or ejected again with exponential backoff. A backend that P2C has avoided
    for `stale_after_s` gets one request so its EWMA doesn't go stale.
    """

    def __init__(self, urls: list[str] | None = None,
                 transport: httpx.AsyncBaseTransport | None = None, *,
                 alpha: float = 0.3,
                 outlier_ratio: float = 3.0,
                 outlier_floor_ms: float = 250.0,
                 min_samples: int = 3,
                 failure_threshold: int = 3,
                 base_ejection_s: float = 5.0,
                 max_ejection_s: float = 60.0,
                 max_ejection_fraction: float = 0.5,
                 stale_after_s: float = 5.0,
                 clock: Callable[[], float] = time.monotonic,
                 rng: random.Random | None = None):
        self._backends = [_Backend(u) for u in (urls or upstream_urls())]
        self._client = httpx.AsyncClient(timeout=None, transport=transport)
        self._alpha = alpha
        self._outlier_ratio = outlier_ratio
        self._outlier_floor_ms = outlier_floor_ms
        self._min_samples = min_samples
        self._failure_threshold = failure_threshold
        self._base_ejection_s = base_ejection_s
        self._max_ejection_s = max_ejection_s
        self._max_ejection_fraction = max_ejection_fraction
        self._stale_after_s = stale_after_s
        self._clock = clock
        self._rng = rng or random.Random()

    async def forward(self, payload: dict) -> tuple[int, dict]:
        backend = self._pick()
        backend.last_picked = self._clock()
        backend.inflight += 1
        start = self._clock()
        try:
            resp = await self._client.post(f"{backend.url}/v1/completions", json=payload)
        except httpx.HTTPError:
            self._record(backend, (self._clock() - start) * 1000, failed=True)
            raise
        finally:
            backend.inflight -= 1
        self._record(backend, (self._clock() - start) * 1000,
                     failed=resp.status_code >= 500)
        return resp.status_code, resp.json()

    async def aclose(self) -> None:
        await self._client.aclose()

    def stats(self) -> list[dict]:
        now = self._clock()
        return [{"url": b.url, "ewma_ms": b.ewma_ms, "inflight": b.inflight,
                 "ejected": b.ejected_until > now} for b in self._backends]

    def _pick(self) -> _Backend:
        now = self._clock()
        for b in self._backends:
            if b.ejected_until and b.ejected_until <= now and not b.probing and b.inflight == 0:
                # Ejection expired: send exactly one probe to decide its fate.
                b.probing = True
                return b
        for b in self._backends:
            if (not b.ejected_until and b.ewma_ms is not None and b.inflight == 0
                    and now - b.last_picked >= self._stale_after_s):
                b.stale_probe = True
                return b
        eligible = [b for b in self._backends if b.ejected_until <= now and not b.probing]
        if not eligible:
            eligible = self._backends
        if len(eligible) == 1:
            return eligible[0]
        a, b = self._rng.sample(eligible, 2)
        return a if self._score(a) <= self._score(b) else b

    @staticmethod
    def _score(b: _Backend) -> float:
        return (b.ewma_ms or 0.0) * (b.inflight + 1)

    def _record(self, b: _Backend, latency_ms: float, failed: bool) -> None:
        now = self._clock()
        if b.probing:
            b.probing = False
            baseline = self._baseline(exclude=b)
            slow = baseline is not None and latency_ms > max(
                self._outlier_floor_ms, self._outlier_ratio * baseline)
            if failed or slow:
                self._eject(b, now)
            else:
                b.ejected_until = 0.0
                b.ejections = 0
                b.consecutive_failures = 0
                b.ewma_ms, b.samples = latency_ms, 1
            return

        b.samples += 1
        if b.stale_probe:
            b.stale_probe = False
            b.ewma_ms = None
        b.ewma_ms = latency_ms if b.ewma_ms is None else (
            self._alpha * latency_ms + (1 - self._alpha) * b.ewma_ms)
        b.consecutive_failures = b.consecutive_failures + 1 if failed else 0

        if b.ejected_until > now:
            return
        if b.consecutive_failures >= self._failure_threshold or self._is_outlier(b):
            if self._can_eject(now):
                self._eject(b, now)

    def _baseline(self, exclude: _Backend | None = None) -> float | None:
        now = self._clock()
        vals = [b.ewma_ms for b in self._backends
                if b is not exclude and b.ewma_ms is not None
                and b.ejected_until <= now and not b.probing]
        return statistics.median(vals) if vals else None

    def _is_outlier(self, b: _Backend) -> bool:
        if b.samples < self._min_samples or b.ewma_ms is None:
            return False
        baseline = self._baseline()
        if baseline is None:
            return False
        return b.ewma_ms > max(self._outlier_floor_ms, self._outlier_ratio * baseline)

    def _can_eject(self, now: float) -> bool:
        ejected = sum(1 for b in self._backends if b.ejected_until > now or b.probing)
        return ejected + 1 <= int(len(self._backends) * self._max_ejection_fraction)

    def _eject(self, b: _Backend, now: float) -> None:
        b.ejections += 1
        duration = min(self._max_ejection_s,
                       self._base_ejection_s * 2 ** (b.ejections - 1))
        b.ejected_until = now + duration
        b.consecutive_failures = 0
