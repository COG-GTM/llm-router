from itertools import count

import httpx

from .config import upstream_urls

TIMEOUT = httpx.Timeout(60.0, connect=5.0)
LIMITS = httpx.Limits(max_connections=200, max_keepalive_connections=50)


class UpstreamPool:
    def __init__(self, urls: list[str] | None = None,
                 transport: httpx.AsyncBaseTransport | None = None):
        self._urls = list(urls or upstream_urls())
        self._next = count()
        self._client = httpx.AsyncClient(timeout=TIMEOUT, limits=LIMITS, transport=transport)

    async def _post(self, url: str, payload: dict) -> tuple[int, dict]:
        resp = await self._client.post(f"{url}/v1/completions", json=payload)
        return resp.status_code, resp.json()

    async def forward(self, payload: dict) -> tuple[int, dict]:
        i = next(self._next) % len(self._urls)
        url, fallback = self._urls[i], self._urls[(i + 1) % len(self._urls)]
        try:
            first = await self._post(url, payload)
        except httpx.TransportError:
            if fallback == url:
                raise
            return await self._post(fallback, payload)
        if first[0] < 500 or fallback == url:
            return first
        try:
            return await self._post(fallback, payload)
        except httpx.TransportError:
            return first

    async def aclose(self) -> None:
        await self._client.aclose()
