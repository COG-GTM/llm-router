import asyncio
import copy
import os
import time
from collections import OrderedDict
from itertools import cycle

import httpx

from .config import upstream_urls

_CacheKey = tuple[str | None, int | None]


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
                 cache_ttl: float | None = None):
        self._urls = cycle(urls or upstream_urls())
        self._client = httpx.AsyncClient(timeout=None, transport=transport)
        # Injected transports (tests) default to no caching so they observe every call.
        if cache_size is None:
            cache_size = 0 if transport else int(os.environ.get("UPSTREAM_CACHE_SIZE", "1024"))
        if cache_ttl is None:
            cache_ttl = float(os.environ.get("UPSTREAM_CACHE_TTL_S", "300"))
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
        status, body = await self._post(payload)
        if status == 200:
            self._cache.put(key, body)
        return status, body

    async def _post(self, payload: dict) -> tuple[int, dict]:
        url = next(self._urls)
        resp = await self._client.post(f"{url}/v1/completions", json=payload)
        return resp.status_code, resp.json()

    async def aclose(self) -> None:
        await self._client.aclose()
