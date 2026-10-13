import time
from itertools import cycle

import httpx

from .config import upstream_urls


class UpstreamPool:
    """Routes each request to the replica with the lowest EWMA latency.

    Replicas without a latency sample (or whose sample is older than
    ``probe_interval`` seconds) get a single probe request, in round-robin
    order; everything else goes to the lowest-EWMA replica. Failed (5xx /
    transport error) requests count as ``failure_penalty`` seconds so a bad
    replica is avoided until a later probe shows it has recovered.
    """

    def __init__(self, urls: list[str] | None = None,
                 transport: httpx.AsyncBaseTransport | None = None,
                 alpha: float = 0.2, failure_penalty: float = 5.0,
                 probe_interval: float = 10.0):
        self._url_list = list(urls or upstream_urls())
        self._urls = cycle(self._url_list)
        self._alpha = alpha
        self._failure_penalty = failure_penalty
        self._probe_interval = probe_interval
        self._ewma: dict[str, float] = {}
        self._sampled_at: dict[str, float] = {}
        self._probing: set[str] = set()
        self._client = httpx.AsyncClient(timeout=None, transport=transport)

    def _needs_probe(self, url: str, now: float) -> bool:
        if url in self._probing:
            return False
        sampled_at = self._sampled_at.get(url)
        return sampled_at is None or now - sampled_at >= self._probe_interval

    def _pick(self) -> str:
        now = time.monotonic()
        for _ in range(len(self._url_list)):
            url = next(self._urls)
            if self._needs_probe(url, now):
                self._probing.add(url)
                return url
        if not self._ewma:
            return next(self._urls)
        return min(self._ewma, key=self._ewma.__getitem__)

    def _record(self, url: str, latency: float) -> None:
        prev = self._ewma.get(url)
        self._ewma[url] = latency if prev is None else (
            self._alpha * latency + (1 - self._alpha) * prev)
        self._sampled_at[url] = time.monotonic()
        self._probing.discard(url)

    async def forward(self, payload: dict) -> tuple[int, dict]:
        url = self._pick()
        start = time.perf_counter()
        try:
            resp = await self._client.post(f"{url}/v1/completions", json=payload)
        except httpx.HTTPError:
            self._record(url, max(time.perf_counter() - start, self._failure_penalty))
            raise
        finally:
            self._probing.discard(url)
        latency = time.perf_counter() - start
        if resp.status_code >= 500:
            latency = max(latency, self._failure_penalty)
        self._record(url, latency)
        return resp.status_code, resp.json()

    async def aclose(self) -> None:
        await self._client.aclose()
