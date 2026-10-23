from typing import Annotated

from pydantic import AfterValidator, BaseModel, Field, model_validator

from .config import DEFAULT_MAX_TOKENS, max_tokens_limit


def _check_max_tokens_limit(value: int) -> int:
    # Out-of-range values are rejected (422) rather than clamped: silently
    # shortening a requested generation would hand the client a truncated
    # completion with no signal why. The limit is read per request so
    # MAX_TOKENS_LIMIT can be tuned without code changes.
    limit = max_tokens_limit()
    if value > limit:
        raise ValueError(f"max_tokens must be <= {limit}")
    return value


MaxTokens = Annotated[int, Field(ge=1), AfterValidator(_check_max_tokens_limit)]


class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: MaxTokens = DEFAULT_MAX_TOKENS


class ChatRequest(BaseModel):
    conversation_id: str
    message: str
    max_tokens: MaxTokens = DEFAULT_MAX_TOKENS
    model: str | None = None


class ConversationCreate(BaseModel):
    title: str | None = None


class ConversationUpdate(BaseModel):
    """PATCH body for a conversation: rename it, pin/unpin it, or both."""

    title: str | None = None
    pinned: bool | None = None

    @model_validator(mode="after")
    def _require_at_least_one_field(self) -> "ConversationUpdate":
        if self.title is None and self.pinned is None:
            raise ValueError("at least one of title or pinned must be provided")
        return self


# -- response models ---------------------------------------------------------
# These describe the shape of what the store already returns; wiring them in
# as `response_model=` gets us request-time validation plus accurate OpenAPI
# docs for the conversation/chat/info surface without changing any payloads.


class ConversationOut(BaseModel):
    id: str
    title: str
    created_at: float
    updated_at: float
    pinned: bool


class MessageOut(BaseModel):
    id: str
    conversation_id: str
    role: str
    content: str
    latency_ms: int | None = None
    created_at: float


class ChatTurnResponse(BaseModel):
    message: MessageOut
    latency_ms: int
    model: str | None = None
    # `usage` is passed through opaquely: it comes straight from the upstream
    # backend, whose exact field set this service doesn't own, so it isn't
    # worth (or safe) pinning to a strict shape here.
    usage: dict | None = None


class InfoResponse(BaseModel):
    name: str
    version: str
    models: list[str]
    uptime_s: float
