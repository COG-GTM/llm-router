import asyncio
import copy
import json
import time
from collections import OrderedDict
from collections.abc import Awaitable, Callable

Result = tuple[int, dict]


class ResponseCache:
    """Bounded LRU/TTL cache of successful upstream responses.

    Concurrent misses for the same payload share one upstream call. Callers
    always get a deep copy, so mutating a result never touches the cache.
    """

    def __init__(self, max_size: int = 1024, ttl_s: float = 300.0):
        self.max_size = max_size
        self.ttl_s = ttl_s
        self._entries: OrderedDict[str, tuple[float, dict]] = OrderedDict()
        self._inflight: dict[str, asyncio.Task] = {}

    @property
    def enabled(self) -> bool:
        return self.max_size > 0 and self.ttl_s > 0

    async def get_or_fetch(self, payload: dict,
                           fetch: Callable[[], Awaitable[Result]]) -> Result:
        if not self.enabled:
            return await fetch()
        key = json.dumps(payload, sort_keys=True)
        body = self._get(key)
        if body is not None:
            return 200, copy.deepcopy(body)
        task = self._inflight.get(key)
        if task is None:
            task = asyncio.ensure_future(self._fetch_and_store(key, fetch))
            self._inflight[key] = task
        # shield: a cancelled waiter must not cancel the call others share
        status, body = await asyncio.shield(task)
        return status, copy.deepcopy(body)

    async def _fetch_and_store(self, key: str,
                               fetch: Callable[[], Awaitable[Result]]) -> Result:
        try:
            status, body = await fetch()
            if status == 200:
                self._put(key, body)
            return status, body
        finally:
            self._inflight.pop(key, None)

    def _get(self, key: str) -> dict | None:
        entry = self._entries.get(key)
        if entry is None:
            return None
        expires_at, body = entry
        if expires_at <= time.monotonic():
            del self._entries[key]
            return None
        self._entries.move_to_end(key)
        return body

    def _put(self, key: str, body: dict) -> None:
        self._entries[key] = (time.monotonic() + self.ttl_s, copy.deepcopy(body))
        self._entries.move_to_end(key)
        if len(self._entries) > self.max_size:
            now = time.monotonic()
            for k in [k for k, (exp, _) in self._entries.items() if exp <= now]:
                del self._entries[k]
        while len(self._entries) > self.max_size:
            self._entries.popitem(last=False)
