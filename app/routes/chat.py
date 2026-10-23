import asyncio
import time

from fastapi import APIRouter, BackgroundTasks, HTTPException, Request

from ..prompting import flatten_history
from ..schemas import ChatRequest, ChatTurnResponse
from ..store import new_message

router = APIRouter(tags=["chat"])

_AUTO_TITLE_LIMIT = 48


def _load_conversation(store, conversation_id: str) -> tuple[dict | None, list[dict]]:
    conversation = store.get_conversation(conversation_id)
    if conversation is None:
        return None, []
    return conversation, store.list_messages(conversation_id)


@router.post("/v1/chat", response_model=ChatTurnResponse)
async def chat(body: ChatRequest, request: Request, background_tasks: BackgroundTasks):
    store = request.app.state.store
    # sqlite calls block, so keep them off the event loop.
    conversation, history = await asyncio.to_thread(
        _load_conversation, store, body.conversation_id
    )
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

    user = new_message(body.conversation_id, "user", body.message)
    assistant = new_message(
        body.conversation_id, "assistant", upstream["completion"], latency_ms=latency_ms
    )
    # Persist after the response is sent; any store read flushes pending
    # turns first, so follow-up requests still see these messages.
    store.defer_turn([user, assistant], title=title)
    background_tasks.add_task(store.flush)
    return {
        "message": assistant,
        "latency_ms": latency_ms,
        "model": upstream.get("model"),
        "usage": upstream.get("usage"),
    }
