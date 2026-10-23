import asyncio
import os
from itertools import cycle

import httpx

from .config import upstream_urls


def _env(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


class UpstreamPool:
    """Round-robin pool that fails over to the next replica on timeout, 5xx or connection error.

    Every attempt but the last is cut off after ``attempt_timeout`` seconds so one slow
    replica can't stall the request; the last attempt gets the full ``timeout``.
    """

    def __init__(self, urls: list[str] | None = None,
                 transport: httpx.AsyncBaseTransport | None = None,
                 attempt_timeout: float | None = None,
                 timeout: float | None = None,
                 max_attempts: int | None = None):
        self._ring = urls or upstream_urls()
        self._starts = cycle(range(len(self._ring)))
        self._attempt_timeout = (attempt_timeout if attempt_timeout is not None
                                 else _env("UPSTREAM_ATTEMPT_TIMEOUT_S", 0.25))
        self._timeout = timeout if timeout is not None else _env("UPSTREAM_TIMEOUT_S", 10.0)
        self._max_attempts = max_attempts or int(
            _env("UPSTREAM_MAX_ATTEMPTS", len(self._ring) + 1))
        limits = httpx.Limits(
            max_connections=int(_env("UPSTREAM_MAX_CONNECTIONS", 512)),
            max_keepalive_connections=int(_env("UPSTREAM_MAX_KEEPALIVE", 128)),
        )
        self._client = httpx.AsyncClient(timeout=self._timeout, limits=limits,
                                         transport=transport)

    async def forward(self, payload: dict) -> tuple[int, dict]:
        start = next(self._starts)
        status, body = 502, {"detail": "upstream unavailable"}
        for attempt in range(self._max_attempts):
            url = self._ring[(start + attempt) % len(self._ring)]
            last = attempt == self._max_attempts - 1
            try:
                async with asyncio.timeout(self._timeout if last else self._attempt_timeout):
                    resp = await self._client.post(f"{url}/v1/completions", json=payload)
            except (TimeoutError, httpx.TimeoutException):
                status, body = 504, {"detail": "upstream timeout"}
                continue
            except httpx.TransportError:
                status, body = 502, {"detail": "upstream unavailable"}
                continue
            status, body = resp.status_code, resp.json()
            if status < 500:
                break
        return status, body

    async def aclose(self) -> None:
        await self._client.aclose()
