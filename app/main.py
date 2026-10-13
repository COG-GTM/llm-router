import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from .generate_cache import ResponseCache
from .middleware.logging import RequestLogMiddleware
from .routes.chat import router as chat_router
from .routes.conversations import router as conversations_router
from .routes.info import router as info_router
from .schemas import GenerateRequest
from .store import Store
from .upstream import UpstreamPool

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.pool = UpstreamPool()
    app.state.generate_cache = ResponseCache(maxsize=1000)
    app.state.store = Store(os.environ.get("APP_DB_PATH", "data/app.db"))
    yield
    await app.state.pool.aclose()
    app.state.store.close()


app = FastAPI(lifespan=lifespan)
app.add_middleware(RequestLogMiddleware)


@app.post("/v1/generate")
async def generate(req: GenerateRequest):
    cache: ResponseCache = app.state.generate_cache
    key = (req.prompt, req.max_tokens)
    cached = cache.get(key)
    if cached is not None:
        return Response(content=cached, media_type="application/json")
    status, body = await app.state.pool.forward(req.model_dump())
    resp = JSONResponse(status_code=status, content=body)
    if status == 200:
        cache.put(key, resp.body)
    return resp


app.include_router(conversations_router)
app.include_router(chat_router)
app.include_router(info_router)

_DIST = os.path.join(os.path.dirname(__file__), "..", "frontend", "dist")
if os.path.isdir(_DIST):
    app.mount("/", StaticFiles(directory=_DIST, html=True), name="ui")
