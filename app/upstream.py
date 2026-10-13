from itertools import cycle

import httpx

from .config import upstream_urls


class UpstreamPool:
    def __init__(self, urls: list[str] | None = None,
                 transport: httpx.AsyncBaseTransport | None = None):
        self._url_list = list(urls or upstream_urls())
        self._indices = cycle(range(len(self._url_list)))
        self._client = httpx.AsyncClient(timeout=None, transport=transport)

    async def _post(self, url: str, payload: dict) -> httpx.Response:
        return await self._client.post(f"{url}/v1/completions", json=payload)

    async def forward(self, payload: dict) -> tuple[int, dict]:
        idx = next(self._indices)
        can_retry = len(self._url_list) > 1
        try:
            resp = await self._post(self._url_list[idx], payload)
        except httpx.TransportError:
            if not can_retry:
                raise
            resp = None
        if resp is not None and (resp.status_code < 500 or not can_retry):
            return resp.status_code, resp.json()
        retry_url = self._url_list[(idx + 1) % len(self._url_list)]
        resp = await self._post(retry_url, payload)
        return resp.status_code, resp.json()

    async def aclose(self) -> None:
        await self._client.aclose()
