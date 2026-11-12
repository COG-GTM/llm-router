import asyncio
import logging
from itertools import cycle

import httpx

from .config import upstream_retry_timeout, upstream_timeout, upstream_urls

log = logging.getLogger(__name__)


class UpstreamPool:
    """Round-robin pool; a failed attempt is retried once on a different backend."""

    def __init__(self, urls: list[str] | None = None,
                 transport: httpx.AsyncBaseTransport | None = None,
                 timeout: float | None = None,
                 retry_timeout: float | None = None):
        self._url_list = list(urls or upstream_urls())
        self._urls = cycle(self._url_list)
        self._timeout = upstream_timeout() if timeout is None else timeout
        self._retry_timeout = upstream_retry_timeout() if retry_timeout is None else retry_timeout
        self._client = httpx.AsyncClient(timeout=self._retry_timeout, transport=transport)

    def _next_url(self, avoid: str | None = None) -> str:
        url = next(self._urls)
        if url == avoid and len(self._url_list) > 1:
            url = next(self._urls)
        return url

    async def _attempt(self, url: str, payload: dict, timeout: float) -> tuple[int, dict]:
        # httpx timeouts are per-phase; wait_for enforces a wall-clock deadline per attempt.
        try:
            resp = await asyncio.wait_for(
                self._client.post(f"{url}/v1/completions", json=payload,
                                  timeout=httpx.Timeout(timeout)),
                timeout,
            )
        except TimeoutError as exc:
            raise httpx.TimeoutException(f"no response within {timeout}s") from exc
        return resp.status_code, resp.json()

    async def forward(self, payload: dict) -> tuple[int, dict]:
        url = self._next_url()
        try:
            status, body = await self._attempt(url, payload, self._timeout)
            if status < 500:
                return status, body
            reason = f"status {status}"
        except httpx.TimeoutException:
            reason = "timeout"
        except httpx.TransportError as exc:
            reason = type(exc).__name__

        retry_url = self._next_url(avoid=url)
        log.info("upstream %s failed (%s); retrying on %s", url, reason, retry_url)
        try:
            return await self._attempt(retry_url, payload, self._retry_timeout)
        except httpx.TimeoutException:
            return 504, {"detail": "upstream timeout"}
        except httpx.TransportError:
            return 502, {"detail": "upstream unavailable"}

    async def aclose(self) -> None:
        await self._client.aclose()
