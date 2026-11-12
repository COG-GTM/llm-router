from fastapi import APIRouter, HTTPException, Query, Request, Response
from starlette.concurrency import run_in_threadpool

from ..export import render_markdown, slugify
from ..schemas import ConversationCreate, ConversationOut, ConversationUpdate, MessageOut

router = APIRouter(prefix="/v1/conversations", tags=["conversations"])

MAX_MESSAGES_LIMIT = 200
MAX_SEARCH_QUERY_LENGTH = 200


@router.get("", response_model=list[ConversationOut])
async def list_conversations(
    request: Request,
    q: str | None = Query(
        default=None,
        max_length=MAX_SEARCH_QUERY_LENGTH,
        description="Case-insensitive substring filter on conversation title.",
    ),
):
    return await run_in_threadpool(request.app.state.store.list_conversations, q=q)


@router.post("", status_code=201, response_model=ConversationOut)
async def create_conversation(body: ConversationCreate, request: Request):
    title = body.title or "New conversation"
    return await run_in_threadpool(request.app.state.store.create_conversation, title=title)


@router.patch("/{conversation_id}", response_model=ConversationOut)
async def update_conversation(conversation_id: str, body: ConversationUpdate, request: Request):
    conversation = await run_in_threadpool(
        request.app.state.store.update_conversation,
        conversation_id, title=body.title, pinned=body.pinned,
    )
    if conversation is None:
        raise HTTPException(status_code=404, detail="conversation not found")
    return conversation


@router.delete("/{conversation_id}", status_code=204)
async def delete_conversation(conversation_id: str, request: Request):
    if not await run_in_threadpool(request.app.state.store.delete_conversation, conversation_id):
        raise HTTPException(status_code=404, detail="conversation not found")
    return Response(status_code=204)


@router.get("/{conversation_id}/messages", response_model=list[MessageOut])
async def list_messages(
    conversation_id: str,
    request: Request,
    limit: int | None = Query(
        default=None,
        gt=0,
        le=MAX_MESSAGES_LIMIT,
        description="Cap the number of messages returned, most recent first.",
    ),
    before: str | None = Query(
        default=None,
        description="Message id cursor: only return messages strictly before this one.",
    ),
):
    store = request.app.state.store
    if await run_in_threadpool(store.get_conversation, conversation_id) is None:
        raise HTTPException(status_code=404, detail="conversation not found")
    return await run_in_threadpool(store.list_messages, conversation_id, limit=limit, before=before)


@router.get("/{conversation_id}/export")
async def export_conversation(conversation_id: str, request: Request):
    store = request.app.state.store
    conversation = await run_in_threadpool(store.get_conversation, conversation_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail="conversation not found")
    messages = await run_in_threadpool(store.list_messages, conversation_id)
    body = render_markdown(conversation, messages)
    filename = f"{slugify(conversation['title'])}.md"
    return Response(
        content=body,
        media_type="text/markdown",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
