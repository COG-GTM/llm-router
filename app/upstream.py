import asyncio
import copy
import os
import time
from collections import OrderedDict

import httpx

from .config import upstream_urls

_CacheKey = tuple[str | None, int | None]

# EWMA weight of the latest attempt in a backend's failure score, and the score at or
# above which the backend is demoted behind every healthy one.
_FAILURE_ALPHA = 0.3
_UNHEALTHY_SCORE = 0.5


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


def _rotate(items: list[int], k: int) -> list[int]:
    if not items:
        return items
    k %= len(items)
    return items[k:] + items[:k]


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
        self._failure = [0.0] * len(self._urls)
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

    def _attempt_order(self) -> list[int]:
        """Round-robin over healthy backends, then round-robin over unhealthy ones.

        Rotating the unhealthy tail too means each of them still periodically gets the
        last slot (and its long timeout), so a slow-but-working backend is not starved by
        a dead one.
        """
        healthy = [i for i, f in enumerate(self._failure) if f < _UNHEALTHY_SCORE]
        unhealthy = [i for i, f in enumerate(self._failure) if f >= _UNHEALTHY_SCORE]
        start = self._next
        self._next += 1
        return _rotate(healthy, start) + _rotate(unhealthy, start)

    def _observe(self, idx: int, failed: bool) -> None:
        self._failure[idx] += _FAILURE_ALPHA * (float(failed) - self._failure[idx])

    async def _post(self, payload: dict) -> tuple[int, dict]:
        """Try backends in health-aware round-robin order, failing over to each at most once."""
        order = self._attempt_order()
        result = (502, {"detail": "no upstream available"})
        for i, idx in enumerate(order):
            url = self._urls[idx]
            last = i == len(order) - 1
            timeout = self._final_timeout if last else self._attempt_timeout
            started = time.monotonic()
            try:
                # asyncio.timeout bounds total elapsed time: httpx's own timeout is
                # per-phase and resets on each received chunk, so a replica dribbling
                # the response would otherwise never fail over.
                async with asyncio.timeout(timeout):
                    resp = await self._client.post(f"{url}/v1/completions", json=payload,
                                                   timeout=timeout)
            except (httpx.TimeoutException, TimeoutError):
                self._observe(idx, True)
                result = (504, {"detail": "upstream timeout"})
                continue
            except httpx.RequestError:
                # Transport errors plus decode failures (e.g. corrupt gzip body).
                self._observe(idx, True)
                result = (502, {"detail": "upstream unavailable"})
                continue
            # A response slower than the attempt budget counts against the backend too.
            slow = time.monotonic() - started > self._attempt_timeout
            self._observe(idx, resp.status_code >= 500 or slow)
            try:
                body = resp.json()
            except ValueError:
                body = {"detail": "invalid upstream response"}
                if resp.status_code < 500:
                    return 502, body
            if resp.status_code < 500:
                return resp.status_code, body
            result = (resp.status_code, body)
        return result

    async def aclose(self) -> None:
        await self._client.aclose()
