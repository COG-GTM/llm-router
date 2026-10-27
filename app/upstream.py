import asyncio
from itertools import cycle

import httpx

from .config import (
    upstream_connect_timeout_s,
    upstream_max_retries,
    upstream_timeout_s,
    upstream_urls,
)


class UpstreamPool:
    def __init__(self, urls: list[str] | None = None,
                 transport: httpx.AsyncBaseTransport | None = None,
                 timeout_s: float | None = None,
                 max_retries: int | None = None):
        self._urls = list(urls or upstream_urls())
        self._starts = cycle(range(len(self._urls)))
        self._timeout_s = timeout_s if timeout_s is not None else upstream_timeout_s()
        self._max_retries = (max_retries if max_retries is not None
                             else upstream_max_retries(len(self._urls)))
        connect_s = min(upstream_connect_timeout_s(), self._timeout_s)
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(self._timeout_s, connect=connect_s), transport=transport
        )

    async def _attempt(self, url: str, payload: dict) -> tuple[int, dict | None]:
        # httpx timeouts are per phase/chunk; bound the whole attempt as well.
        async with asyncio.timeout(self._timeout_s):
            resp = await self._client.post(f"{url}/v1/completions", json=payload)
        if resp.status_code >= 500:
            return resp.status_code, None
        return resp.status_code, resp.json()

    async def forward(self, payload: dict) -> tuple[int, dict]:
        last_failure = "unavailable"
        # One shared round-robin step picks the start; retries walk on from the
        # upstream that just failed, independent of concurrent requests.
        start = next(self._starts)
        for k in range(self._max_retries + 1):
            url = self._urls[(start + k) % len(self._urls)]
            try:
                status, body = await self._attempt(url, payload)
            except (httpx.TimeoutException, TimeoutError):
                last_failure = "timeout"
                continue
            except httpx.RequestError:
                last_failure = "unavailable"
                continue
            if status < 500:
                return status, body
            last_failure = f"HTTP {status}"
        attempts = self._max_retries + 1
        if last_failure == "timeout":
            return 504, {"detail": f"upstream timed out after {attempts} attempts"}
        return 502, {"detail": f"upstream failed after {attempts} attempts ({last_failure})"}

    async def aclose(self) -> None:
        await self._client.aclose()
