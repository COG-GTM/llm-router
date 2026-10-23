import asyncio
import copy
import os
import time
from collections import OrderedDict

import httpx

from .config import upstream_urls

_CacheKey = tuple[str | None, int | None]

# Latencies below this count as equal, and costs within _TIE_RATIO of the cheapest are ties.
_MIN_COST_S = 0.01
_TIE_RATIO = 1.25


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


class _ResponseCache:
    """Bounded LRU of successful upstream bodies, each entry expiring after `ttl` seconds."""

    def __init__(self, maxsize: int, ttl: float):
        self._maxsize = maxsize
        self._ttl = ttl
        self._entries: OrderedDict[_CacheKey, tuple[float, dict]] = OrderedDict()

    def get(self, key: _CacheKey) -> dict | None:
        entry = self._entries.get(key)
        if entry is None:
            return None
        expires_at, body = entry
        if expires_at <= time.monotonic():
            del self._entries[key]
            return None
        self._entries.move_to_end(key)
        return body

    def put(self, key: _CacheKey, body: dict) -> None:
        self._entries[key] = (time.monotonic() + self._ttl, copy.deepcopy(body))
        self._entries.move_to_end(key)
        while len(self._entries) > self._maxsize:
            self._entries.popitem(last=False)


class UpstreamPool:
    def __init__(self, urls: list[str] | None = None,
                 transport: httpx.AsyncBaseTransport | None = None,
                 cache_size: int | None = None,
                 cache_ttl: float | None = None,
                 attempt_timeout: float | None = None,
                 final_timeout: float | None = None):
        self._urls = list(urls or upstream_urls())
        self._next = 0
        self._in_flight = [0] * len(self._urls)
        self._ewma: list[float | None] = [None] * len(self._urls)
        self._ewma_alpha = _env_float("UPSTREAM_EWMA_ALPHA", 0.3)
        # A failed attempt counts as at least this slow, so one timeout keeps traffic away for a while.
        self._failure_penalty = _env_float("UPSTREAM_FAILURE_PENALTY_S", 1.0)
        # Every attempt but the last is cut short so a slow or spiking backend fails over
        # quickly; the last backend tried gets the long timeout so the request still succeeds.
        if attempt_timeout is None:
            attempt_timeout = _env_float("UPSTREAM_ATTEMPT_TIMEOUT_S", 0.25)
        if final_timeout is None:
            final_timeout = _env_float("UPSTREAM_FINAL_TIMEOUT_S", 30.0)
        self._attempt_timeout = attempt_timeout
        self._final_timeout = final_timeout
        limits = httpx.Limits(
            max_connections=_env_int("UPSTREAM_MAX_CONNECTIONS", 256),
            max_keepalive_connections=_env_int("UPSTREAM_MAX_KEEPALIVE", 64),
        )
        self._client = httpx.AsyncClient(timeout=final_timeout, limits=limits, transport=transport)
        # Injected transports (tests) default to no caching so they observe every call.
        if cache_size is None:
            cache_size = 0 if transport else _env_int("UPSTREAM_CACHE_SIZE", 1024)
        if cache_ttl is None:
            cache_ttl = _env_float("UPSTREAM_CACHE_TTL_S", 300.0)
        enabled = cache_size > 0 and cache_ttl > 0
        self._cache = _ResponseCache(cache_size, cache_ttl) if enabled else None
        self._inflight: dict[_CacheKey, asyncio.Task] = {}

    async def forward(self, payload: dict) -> tuple[int, dict]:
        if self._cache is None:
            return await self._post(payload)
        key = (payload.get("prompt"), payload.get("max_tokens"))
        cached = self._cache.get(key)
        if cached is not None:
            return 200, copy.deepcopy(cached)
        task = self._inflight.get(key)
        if task is None:
            task = asyncio.ensure_future(self._fetch(key, payload))
            self._inflight[key] = task
            task.add_done_callback(lambda _: self._inflight.pop(key, None))
        status, body = await asyncio.shield(task)
        return status, copy.deepcopy(body)

    async def _fetch(self, key: _CacheKey, payload: dict) -> tuple[int, dict]:
        # _post already fails over across every backend, so coalesced callers are
        # never pinned to one failing replica.
        status, body = await self._post(payload)
        if status == 200:
            self._cache.put(key, body)
        return status, body

    def _pick(self, start: int, tried: set[int]) -> int:
        """Cheapest untried backend by latency EWMA x (in-flight + 1); near-ties go round-robin."""
        n = len(self._urls)
        known = [e for e in self._ewma if e is not None]
        prior = min(known) if known else 0.0
        order = [i for i in ((start + k) % n for k in range(n)) if i not in tried]
        costs = {}
        for i in order:
            ewma = self._ewma[i] if self._ewma[i] is not None else prior
            costs[i] = max(ewma, _MIN_COST_S) * (self._in_flight[i] + 1)
        cheapest = min(costs.values())
        return next(i for i in order if costs[i] <= cheapest * _TIE_RATIO)

    def _observe(self, i: int, began: float, failed: bool = False) -> None:
        sample = time.monotonic() - began
        if failed:
            sample = max(sample, self._failure_penalty)
        prev = self._ewma[i]
        a = self._ewma_alpha
        self._ewma[i] = sample if prev is None else (1 - a) * prev + a * sample

    async def _post(self, payload: dict) -> tuple[int, dict]:
        """Least-loaded backend first, then fail over to the cheapest untried one, each at most once."""
        n = len(self._urls)
        start = self._next
        tried: set[int] = set()
        result = (502, {"detail": "no upstream available"})
        for attempt in range(n):
            i = self._pick(start, tried)
            if attempt == 0:
                self._next = (i + 1) % n
            tried.add(i)
            timeout = self._final_timeout if attempt == n - 1 else self._attempt_timeout
            began = time.monotonic()
            self._in_flight[i] += 1
            try:
                # asyncio.timeout bounds total elapsed time: httpx's own timeout is
                # per-phase and resets on each received chunk, so a replica dribbling
                # the response would otherwise never fail over.
                async with asyncio.timeout(timeout):
                    resp = await self._client.post(f"{self._urls[i]}/v1/completions", json=payload,
                                                   timeout=timeout)
            except (httpx.TimeoutException, TimeoutError):
                self._observe(i, began, failed=True)
                result = (504, {"detail": "upstream timeout"})
                continue
            except httpx.RequestError:
                # Transport errors plus decode failures (e.g. corrupt gzip body).
                self._observe(i, began, failed=True)
                result = (502, {"detail": "upstream unavailable"})
                continue
            finally:
                self._in_flight[i] -= 1
            try:
                body = resp.json()
            except ValueError:
                self._observe(i, began, failed=True)
                body = {"detail": "invalid upstream response"}
                if resp.status_code < 500:
                    return 502, body
                result = (resp.status_code, body)
                continue
            self._observe(i, began, failed=resp.status_code >= 500)
            if resp.status_code < 500:
                return resp.status_code, body
            result = (resp.status_code, body)
        return result

    async def aclose(self) -> None:
        await self._client.aclose()
