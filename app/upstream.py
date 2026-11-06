from itertools import cycle

import httpx

from .config import upstream_timeout, upstream_urls


class UpstreamPool:
    def __init__(self, urls: list[str] | None = None,
                 transport: httpx.AsyncBaseTransport | None = None,
                 timeout: float | None = None):
        self._urls_list = list(urls or upstream_urls())
        self._starts = cycle(range(len(self._urls_list)))
        self._client = httpx.AsyncClient(timeout=timeout or upstream_timeout(),
                                         transport=transport)

    async def forward(self, payload: dict) -> tuple[int, dict]:
        last_5xx: httpx.Response | None = None
        start = next(self._starts)
        for url in self._urls_list[start:] + self._urls_list[:start]:
            try:
                resp = await self._client.post(f"{url}/v1/completions", json=payload)
            except (httpx.TransportError, httpx.TimeoutException):
                continue
            if resp.status_code >= 500:
                last_5xx = resp
                continue
            return resp.status_code, resp.json()
        if last_5xx is not None:
            return last_5xx.status_code, _json_or_detail(last_5xx)
        return 502, {"error": "all upstreams unavailable"}

    async def aclose(self) -> None:
        await self._client.aclose()


def _json_or_detail(resp: httpx.Response) -> dict:
    try:
        return resp.json()
    except ValueError:
        return {"detail": resp.text or "upstream error"}
