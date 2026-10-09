"""Request-ID tagging and structured request logging."""
import logging
import time
import uuid

from starlette.middleware.base import BaseHTTPMiddleware

from ..upstream import upstream_attribution

logger = logging.getLogger("llm-router.requests")


class RequestLogMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        request_id = uuid.uuid4().hex[:12]
        request.state.request_id = request_id
        attribution: dict = {}
        token = upstream_attribution.set(attribution)
        start = time.monotonic()
        try:
            response = await call_next(request)
        finally:
            upstream_attribution.reset(token)
        duration_ms = (time.monotonic() - start) * 1000
        logger.info(
            "request method=%s path=%s status=%s duration_ms=%.1f request_id=%s"
            " upstream=%s attempts=%s",
            request.method,
            request.url.path,
            response.status_code,
            duration_ms,
            request_id,
            attribution.get("upstream", "-"),
            attribution.get("attempts", 0),
        )
        response.headers["x-request-id"] = request_id
        return response
