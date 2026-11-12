import asyncio
import logging

import httpx

from .config import hedge_delay_s, upstream_urls

log = logging.getLogger(__name__)


class UpstreamPool:
    def __init__(self, urls: list[str] | None = None,
                 transport: httpx.AsyncBaseTransport | None = None,
                 hedge_delay: float | None = None):
        self._urls = urls or upstream_urls()
        self._next = 0
        self._hedge_delay = hedge_delay_s() if hedge_delay is None else hedge_delay
        self._client = httpx.AsyncClient(timeout=None, transport=transport)
        self.requests = 0
        self.hedged = 0
        self.hedge_wins = 0

    async def _post(self, url: str, payload: dict) -> tuple[int, dict]:
        resp = await self._client.post(f"{url}/v1/completions", json=payload)
        return resp.status_code, resp.json()

    async def forward(self, payload: dict) -> tuple[int, dict]:
        idx = self._next
        self._next = (idx + 1) % len(self._urls)
        self.requests += 1

        primary = asyncio.create_task(self._post(self._urls[idx], payload))
        if self._hedge_delay <= 0 or len(self._urls) < 2:
            return await primary

        done, _ = await asyncio.wait({primary}, timeout=self._hedge_delay)
        if done:
            return primary.result()

        self.hedged += 1
        hedge_url = self._urls[(idx + 1) % len(self._urls)]
        hedge = asyncio.create_task(self._post(hedge_url, payload))
        pending = {primary, hedge}
        try:
            while pending:
                done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
                for task in done:
                    if task.exception() is None:
                        if task is hedge:
                            self.hedge_wins += 1
                        return task.result()
            return primary.result()
        finally:
            for task in pending:
                task.cancel()

    async def aclose(self) -> None:
        if self.requests:
            log.info("hedging stats: requests=%d hedged=%d (%.1f%%) hedge_wins=%d",
                     self.requests, self.hedged, 100 * self.hedged / self.requests,
                     self.hedge_wins)
        await self._client.aclose()
