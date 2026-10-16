import bisect
import hashlib
import time

import httpx

from .config import upstream_urls


def _hash(value: str) -> int:
    return int.from_bytes(hashlib.sha256(value.encode()).digest()[:8], "big")


def _json(resp: httpx.Response) -> dict:
    try:
        return resp.json()
    except ValueError:
        return {"detail": resp.text or "upstream error"}


class UpstreamPool:
    """Routes requests across upstream replicas.

    Keyed requests (e.g. a conversation id) are pinned to a replica via a
    consistent-hash ring; keyless requests go to the replica with the lowest
    EWMA latency. Timeouts, transport errors and 5xx responses fail over to the
    next candidate and put the failed replica in a short cooldown.
    """

    def __init__(self, urls: list[str] | None = None,
                 transport: httpx.AsyncBaseTransport | None = None,
                 timeout: float = 30.0,
                 alpha: float = 0.3,
                 cooldown_s: float = 5.0,
                 vnodes: int = 100):
        self._urls = list(urls or upstream_urls())
        self._alpha = alpha
        self._cooldown_s = cooldown_s
        self._ewma: dict[str, float | None] = {u: None for u in self._urls}
        self._down_until: dict[str, float] = {u: 0.0 for u in self._urls}
        ring = sorted((_hash(f"{u}#{i}"), u) for u in self._urls for i in range(vnodes))
        self._ring_hashes = [h for h, _ in ring]
        self._ring_urls = [u for _, u in ring]
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(timeout), transport=transport)

    def _ring_order(self, key: str) -> list[str]:
        start = bisect.bisect(self._ring_hashes, _hash(key)) % len(self._ring_urls)
        order: list[str] = []
        for i in range(len(self._ring_urls)):
            url = self._ring_urls[(start + i) % len(self._ring_urls)]
            if url not in order:
                order.append(url)
                if len(order) == len(self._urls):
                    break
        return order

    def _latency_order(self) -> list[str]:
        # Unsampled replicas sort first so each one gets measured.
        return sorted(self._urls, key=lambda u: self._ewma[u] or 0.0)

    def candidates(self, key: str | None = None) -> list[str]:
        order = self._ring_order(key) if key is not None else self._latency_order()
        now = time.monotonic()
        healthy = [u for u in order if self._down_until[u] <= now]
        return healthy + [u for u in order if u not in healthy]

    def _record_latency(self, url: str, seconds: float) -> None:
        prev = self._ewma[url]
        self._ewma[url] = seconds if prev is None else self._alpha * seconds + (1 - self._alpha) * prev

    def _mark_failed(self, url: str) -> None:
        self._down_until[url] = time.monotonic() + self._cooldown_s

    async def forward(self, payload: dict, key: str | None = None) -> tuple[int, dict]:
        last: tuple[int, dict] = (502, {"detail": "no upstream available"})
        for url in self.candidates(key):
            start = time.monotonic()
            try:
                resp = await self._client.post(f"{url}/v1/completions", json=payload)
            except httpx.TimeoutException:
                self._record_latency(url, time.monotonic() - start)
                self._mark_failed(url)
                last = (504, {"detail": "upstream timeout"})
                continue
            except httpx.TransportError:
                self._mark_failed(url)
                last = (502, {"detail": "upstream unavailable"})
                continue
            if resp.status_code >= 500:
                self._mark_failed(url)
                last = (resp.status_code, _json(resp))
                continue
            self._record_latency(url, time.monotonic() - start)
            return resp.status_code, _json(resp)
        return last

    async def aclose(self) -> None:
        await self._client.aclose()
