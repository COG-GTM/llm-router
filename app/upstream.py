import asyncio
import logging
import time
from itertools import cycle

import httpx

from .config import upstream_urls

logger = logging.getLogger("app.upstream")


class UpstreamPool:
    """Routes each request to the replica with the lowest EWMA latency.

    Replicas without a latency sample (or whose sample is older than
    ``probe_interval`` seconds) get a single probe request, in round-robin
    order; everything else goes to the lowest-EWMA replica. Failed (5xx /
    transport error) requests count as ``failure_penalty`` seconds so a bad
    replica is avoided until a later probe shows it has recovered.

    If the chosen replica has not answered within ``hedge_delay`` seconds, one
    duplicate request is sent to the next-best replica; the first successful
    response wins and the other request is cancelled. A cancelled loser
    records its elapsed time (a lower bound) as a sample, never lowering its
    EWMA. Pass
    ``hedge_delay=None`` to disable hedging.
    """

    def __init__(self, urls: list[str] | None = None,
                 transport: httpx.AsyncBaseTransport | None = None,
                 alpha: float = 0.2, failure_penalty: float = 5.0,
                 probe_interval: float = 10.0,
                 hedge_delay: float | None = 0.25):
        self._url_list = list(urls or upstream_urls())
        self._urls = cycle(self._url_list)
        self._alpha = alpha
        self._failure_penalty = failure_penalty
        self._probe_interval = probe_interval
        self._ewma: dict[str, float] = {}
        self._sampled_at: dict[str, float] = {}
        self._probing: set[str] = set()
        self._hedge_delay = hedge_delay
        self.forwarded = 0
        self.hedges = 0
        self.hedge_wins = 0
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

    def _pick_hedge(self, exclude: str) -> str:
        others = [u for u in self._url_list if u != exclude]
        sampled = [u for u in others if u in self._ewma]
        if sampled:
            return min(sampled, key=self._ewma.__getitem__)
        return others[0]

    async def _attempt(self, url: str, payload: dict) -> tuple[int, dict]:
        start = time.perf_counter()
        try:
            resp = await self._client.post(f"{url}/v1/completions", json=payload)
            body = resp.json()
        except asyncio.CancelledError:
            elapsed = time.perf_counter() - start
            self._record(url, max(elapsed, self._ewma.get(url, elapsed)))
            raise
        except httpx.HTTPError:
            self._record(url, max(time.perf_counter() - start, self._failure_penalty))
            raise
        finally:
            self._probing.discard(url)
        latency = time.perf_counter() - start
        if resp.status_code >= 500:
            latency = max(latency, self._failure_penalty)
        self._record(url, latency)
        return resp.status_code, body

    async def forward(self, payload: dict) -> tuple[int, dict]:
        self.forwarded += 1
        primary = self._pick()
        tasks = [asyncio.create_task(self._attempt(primary, payload))]
        try:
            if self._hedge_delay is not None and len(self._url_list) > 1:
                done, _ = await asyncio.wait(tasks, timeout=self._hedge_delay)
                if not done:
                    self.hedges += 1
                    hedge_url = self._pick_hedge(exclude=primary)
                    tasks.append(asyncio.create_task(self._attempt(hedge_url, payload)))
            pending = set(tasks)
            fallback = None
            while pending:
                done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
                for task in tasks:
                    if task not in done:
                        continue
                    if task.exception() is None and task.result()[0] < 500:
                        if task is not tasks[0]:
                            self.hedge_wins += 1
                        return task.result()
                    fallback = fallback or task
            return fallback.result()
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    async def aclose(self) -> None:
        logger.info("upstream forwarded=%d hedges=%d hedge_wins=%d",
                    self.forwarded, self.hedges, self.hedge_wins)
        await self._client.aclose()
