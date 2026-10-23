import time

from fastapi import APIRouter, HTTPException, Request
from starlette.concurrency import run_in_threadpool

from ..prompting import flatten_history
from ..schemas import ChatRequest, ChatTurnResponse

router = APIRouter(tags=["chat"])

_AUTO_TITLE_LIMIT = 48


def _load_context(store, conversation_id: str) -> tuple[dict | None, list[dict]]:
    conversation = store.get_conversation(conversation_id)
    if conversation is None:
        return None, []
    return conversation, store.list_messages(conversation_id)


@router.post("/v1/chat", response_model=ChatTurnResponse)
async def chat(body: ChatRequest, request: Request):
    store = request.app.state.store
    # Store calls are blocking sqlite3 I/O; run them in worker threads so they
    # don't stall the event loop (Store serializes them internally).
    conversation, history = await run_in_threadpool(_load_context, store, body.conversation_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail="conversation not found")

    prompt = flatten_history(history, body.message)

    start = time.monotonic()
    status, upstream = await request.app.state.pool.forward(
        {"prompt": prompt, "max_tokens": body.max_tokens}
    )
    latency_ms = int((time.monotonic() - start) * 1000)

    if status != 200:
        raise HTTPException(status_code=status,
                            detail=upstream.get("detail", "upstream error"))

    title = None
    if not history and conversation["title"] == "New conversation":
        title = body.message[:_AUTO_TITLE_LIMIT]

    _, assistant = await run_in_threadpool(
        store.add_turn, body.conversation_id, body.message, upstream["completion"],
        latency_ms=latency_ms, title=title,
    )
    return {
        "message": assistant,
        "latency_ms": latency_ms,
        "model": upstream.get("model"),
        "usage": upstream.get("usage"),
    }
