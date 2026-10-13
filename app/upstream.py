from itertools import cycle

import httpx

from .config import upstream_urls

UPSTREAM_TIMEOUT = httpx.Timeout(60.0, connect=5.0)
UPSTREAM_LIMITS = httpx.Limits(
    max_connections=100,
    max_keepalive_connections=50,
    keepalive_expiry=30.0,
)


class UpstreamPool:
    def __init__(self, urls: list[str] | None = None,
                 transport: httpx.AsyncBaseTransport | None = None):
        self._urls = cycle(urls or upstream_urls())
        self._client = httpx.AsyncClient(
            timeout=UPSTREAM_TIMEOUT, limits=UPSTREAM_LIMITS, transport=transport,
        )

    async def forward(self, payload: dict) -> tuple[int, dict]:
        url = next(self._urls)
        resp = await self._client.post(f"{url}/v1/completions", json=payload)
        return resp.status_code, resp.json()

    async def aclose(self) -> None:
        await self._client.aclose()
