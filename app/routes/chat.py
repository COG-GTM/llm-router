import time

from fastapi import APIRouter, HTTPException, Request
from starlette.concurrency import run_in_threadpool

from ..prompting import HISTORY_MAX_MESSAGES, flatten_history, trim_history
from ..schemas import ChatRequest, ChatTurnResponse

router = APIRouter(tags=["chat"])

_AUTO_TITLE_LIMIT = 48


@router.post("/v1/chat", response_model=ChatTurnResponse)
async def chat(body: ChatRequest, request: Request):
    store = request.app.state.store
    conversation = await run_in_threadpool(store.get_conversation, body.conversation_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail="conversation not found")

    recent = await run_in_threadpool(
        store.list_messages, body.conversation_id, limit=HISTORY_MAX_MESSAGES
    )
    prompt = flatten_history(trim_history(recent), body.message)

    start = time.monotonic()
    status, upstream = await request.app.state.pool.forward(
        {"prompt": prompt, "max_tokens": body.max_tokens}
    )
    latency_ms = int((time.monotonic() - start) * 1000)

    if status != 200:
        raise HTTPException(status_code=status,
                            detail=upstream.get("detail", "upstream error"))

    def persist() -> dict:
        if not recent and conversation["title"] == "New conversation":
            store.update_conversation(body.conversation_id,
                                      title=body.message[:_AUTO_TITLE_LIMIT])
        store.add_message(body.conversation_id, "user", body.message)
        return store.add_message(
            body.conversation_id, "assistant", upstream["completion"], latency_ms=latency_ms
        )

    assistant = await run_in_threadpool(persist)
    return {
        "message": assistant,
        "latency_ms": latency_ms,
        "model": upstream.get("model"),
        "usage": upstream.get("usage"),
    }
