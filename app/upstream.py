from itertools import cycle

import httpx

from .cache import ResponseCache
from .config import upstream_urls


class UpstreamPool:
    def __init__(self, urls: list[str] | None = None,
                 transport: httpx.AsyncBaseTransport | None = None,
                 cache: ResponseCache | None = None):
        self._urls = cycle(urls or upstream_urls())
        self._client = httpx.AsyncClient(timeout=None, transport=transport)
        self._cache = cache

    async def forward(self, payload: dict) -> tuple[int, dict]:
        if self._cache is None:
            return await self._post(payload)
        return await self._cache.get_or_fetch(payload, lambda: self._post(payload))

    async def _post(self, payload: dict) -> tuple[int, dict]:
        url = next(self._urls)
        resp = await self._client.post(f"{url}/v1/completions", json=payload)
        return resp.status_code, resp.json()

    async def aclose(self) -> None:
        await self._client.aclose()
