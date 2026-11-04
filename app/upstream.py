import asyncio
import copy
import json
import time
from collections import OrderedDict

import httpx

from .config import (
    upstream_attempt_timeout,
    upstream_cache_size,
    upstream_timeout,
    upstream_urls,
)

_EWMA_ALPHA = 0.2
_FAILURE_PENALTY = 4.0
_FAST_RATIO = 2.0
_FAST_SLACK_S = 0.05


class UpstreamPool:
    def __init__(self, urls: list[str] | None = None,
                 transport: httpx.AsyncBaseTransport | None = None,
                 attempt_timeout: float | None = None,
                 timeout: float | None = None,
                 cache_size: int | None = None):
        self._urls = list(urls or upstream_urls())
        self._attempt_timeout = attempt_timeout or upstream_attempt_timeout()
        self._timeout = max(timeout or upstream_timeout(), self._attempt_timeout)
        self._cache_size = cache_size or upstream_cache_size()
        self._cache: OrderedDict[str, tuple[int, dict]] = OrderedDict()
        self._latency: dict[str, float] = {}
        self._turn = 0
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(self._timeout),
                                         transport=transport)

    async def forward(self, payload: dict) -> tuple[int, dict]:
        key = json.dumps(payload, sort_keys=True)
        cached = self._cache.get(key)
        if cached is not None:
            self._cache.move_to_end(key)
            return cached[0], copy.deepcopy(cached[1])

        status, body = await self._forward_with_retry(payload)
        if 200 <= status < 300:
            self._cache[key] = (status, copy.deepcopy(body))
            if len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)
        return status, body

    async def _forward_with_retry(self, payload: dict) -> tuple[int, dict]:
        order = self._replica_order()
        result: tuple[int, dict] = (504, {"detail": "upstream timeout"})
        for i, url in enumerate(order):
            last = i == len(order) - 1
            timeout = self._timeout if last else self._attempt_timeout
            start = time.monotonic()
            try:
                async with asyncio.timeout(timeout):
                    resp = await self._client.post(f"{url}/v1/completions", json=payload)
            except (TimeoutError, httpx.TimeoutException):
                self._record_failure(url, start)
                result = (504, {"detail": "upstream timeout"})
                continue
            except httpx.TransportError:
                self._record_failure(url, start)
                result = (502, {"detail": "upstream unavailable"})
                continue
            status = resp.status_code
            try:
                body = resp.json()
            except ValueError:
                self._record_failure(url, start)
                result = (status if status >= 500 else 502,
                          {"detail": "invalid upstream response"})
                continue
            if status >= 500:
                self._record_failure(url, start)
                result = (status, body)
                continue
            self._record(url, time.monotonic() - start)
            return status, body
        return result

    def _replica_order(self) -> list[str]:
        """Rotate among replicas close to the fastest observed latency; slower ones go last."""
        known = [self._latency[u] for u in self._urls if u in self._latency]
        cutoff = (min(known) * _FAST_RATIO + _FAST_SLACK_S) if known else 0.0
        fast = [u for u in self._urls if self._latency.get(u, 0.0) <= cutoff]
        slow = sorted((u for u in self._urls if u not in fast), key=self._latency.__getitem__)
        start = self._turn % len(fast)
        self._turn += 1
        return fast[start:] + fast[:start] + slow

    def _record_failure(self, url: str, start: float) -> None:
        elapsed = time.monotonic() - start
        self._record(url, max(elapsed, self._attempt_timeout) * _FAILURE_PENALTY)

    def _record(self, url: str, seconds: float) -> None:
        prev = self._latency.get(url)
        self._latency[url] = seconds if prev is None else (
            _EWMA_ALPHA * seconds + (1 - _EWMA_ALPHA) * prev)

    async def aclose(self) -> None:
        await self._client.aclose()
