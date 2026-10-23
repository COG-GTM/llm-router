import time

import httpx

from .config import upstream_urls

# Weight of the newest latency sample in each backend's moving average.
_EWMA_ALPHA = 0.2
# Floor for all latency estimates, so in-flight load always counts.
_MIN_LATENCY_S = 0.001
# Backends whose cost is within this factor of the cheapest are treated as
# equivalent and picked in round-robin order, keeping selection stable.
_TIE_TOLERANCE = 1.25
# Latency charged for a failed request so a backend that fails fast never
# looks attractive.
_ERROR_PENALTY_S = 1.0
_RATE_LIMITED = 429


class _Backend:
    __slots__ = ("ewma", "in_flight", "url")

    def __init__(self, url: str):
        self.url = url
        self.in_flight = 0
        self.ewma: float | None = None

    def cost(self, prior: float) -> float:
        latency = max(prior if self.ewma is None else self.ewma, _MIN_LATENCY_S)
        return latency * (self.in_flight + 1)

    def observe(self, seconds: float) -> None:
        if self.ewma is None:
            self.ewma = seconds
        else:
            self.ewma += _EWMA_ALPHA * (seconds - self.ewma)


class UpstreamPool:
    """Routes each request to the backend with the lowest expected wait.

    Expected wait is the latency EWMA times (in-flight + 1), so busy or slow
    replicas are avoided. EWMAs only change on completed requests (no idle
    decay), so a slow backend never looks fast just because it went unused.
    """

    def __init__(self, urls: list[str] | None = None,
                 transport: httpx.AsyncBaseTransport | None = None):
        self._backends = [_Backend(u) for u in urls or upstream_urls()]
        self._next = 0
        self._client = httpx.AsyncClient(timeout=None, transport=transport)

    def _pick(self) -> _Backend:
        n = len(self._backends)
        order = [self._backends[(self._next + i) % n] for i in range(n)]
        # Untried backends are assumed as fast as the best measured one, so
        # they get probed but pending probes still count as load.
        prior = min((b.ewma for b in order if b.ewma is not None), default=0.0)
        costs = [b.cost(prior) for b in order]
        best = min(costs)
        chosen = next(b for b, c in zip(order, costs) if c <= best * _TIE_TOLERANCE)
        self._next = (self._backends.index(chosen) + 1) % n
        return chosen

    async def forward(self, payload: dict) -> tuple[int, dict]:
        backend = self._pick()
        backend.in_flight += 1
        start = time.monotonic()
        failed = True
        try:
            resp = await self._client.post(f"{backend.url}/v1/completions", json=payload)
            body = resp.json()
            failed = resp.status_code >= 500 or resp.status_code == _RATE_LIMITED
            return resp.status_code, body
        finally:
            backend.in_flight -= 1
            elapsed = time.monotonic() - start
            backend.observe(max(elapsed, _ERROR_PENALTY_S) if failed else elapsed)

    async def aclose(self) -> None:
        await self._client.aclose()
