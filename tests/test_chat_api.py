import asyncio

import httpx
from fastapi.testclient import TestClient

from app import main
from app.upstream import UpstreamPool


class FakePool:
    def __init__(self, status=200):
        self.status = status
        self.calls: list[dict] = []
        self.keys: list[str | None] = []

    async def forward(self, payload, key=None):
        self.calls.append(payload)
        self.keys.append(key)
        return self.status, (
            {"completion": "mock answer", "model": "mock-large",
             "usage": {"prompt_tokens": 5, "completion_tokens": 2}}
            if self.status == 200 else {"detail": "model overloaded"}
        )

    async def aclose(self):
        pass


def chat_client(pool):
    c = TestClient(main.app)
    c.__enter__()
    main.app.state.pool = pool
    return c


def test_chat_roundtrip_persists_and_titles():
    pool = FakePool()
    c = chat_client(pool)
    cid = c.post("/v1/conversations", json={}).json()["id"]

    resp = c.post("/v1/chat", json={"conversation_id": cid, "message": "What is a hash map?"})
    body = resp.json()
    c.__exit__(None, None, None)

    assert resp.status_code == 200
    assert body["message"]["role"] == "assistant"
    assert body["message"]["content"] == "mock answer"
    assert isinstance(body["latency_ms"], int)
    assert body["model"] == "mock-large"

    # exactly one upstream call, carrying flattened history
    assert len(pool.calls) == 1
    assert pool.calls[0]["prompt"].endswith("User: What is a hash map?\nAssistant:")
    assert pool.calls[0]["max_tokens"] == 64
    assert pool.keys == [cid]


def test_chat_persists_both_turns_and_uses_history():
    pool = FakePool()
    c = chat_client(pool)
    cid = c.post("/v1/conversations", json={}).json()["id"]
    c.post("/v1/chat", json={"conversation_id": cid, "message": "first"})
    c.post("/v1/chat", json={"conversation_id": cid, "message": "second"})
    msgs = c.get(f"/v1/conversations/{cid}/messages").json()
    convo_title = [x for x in c.get("/v1/conversations").json() if x["id"] == cid][0]["title"]
    c.__exit__(None, None, None)

    assert [m["role"] for m in msgs] == ["user", "assistant", "user", "assistant"]
    assert "User: first" in pool.calls[1]["prompt"]
    assert "Assistant: mock answer" in pool.calls[1]["prompt"]
    assert convo_title == "first"  # auto-titled from first message


def test_chat_forwards_non_default_max_tokens_to_pool():
    pool = FakePool()
    c = chat_client(pool)
    cid = c.post("/v1/conversations", json={}).json()["id"]

    c.post("/v1/chat", json={"conversation_id": cid, "message": "hi", "max_tokens": 16})
    c.__exit__(None, None, None)

    assert pool.calls[0]["max_tokens"] == 16


def test_chat_unknown_conversation_404():
    c = chat_client(FakePool())
    resp = c.post("/v1/chat", json={"conversation_id": "nope", "message": "hi"})
    c.__exit__(None, None, None)
    assert resp.status_code == 404


def test_chat_upstream_error_passthrough_and_no_persist():
    pool = FakePool(status=503)
    c = chat_client(pool)
    cid = c.post("/v1/conversations", json={}).json()["id"]
    resp = c.post("/v1/chat", json={"conversation_id": cid, "message": "hi"})
    msgs = c.get(f"/v1/conversations/{cid}/messages").json()
    c.__exit__(None, None, None)
    assert resp.status_code == 503
    assert msgs == []  # failed exchanges are not persisted


def test_chat_latency_includes_failover_retries():
    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "bad":
            await asyncio.sleep(0.2)
            return httpx.Response(503, json={"detail": "overloaded"})
        return httpx.Response(200, json={"completion": "ok", "model": "m"})

    pool = UpstreamPool(urls=["http://bad:1", "http://good:1"],
                        transport=httpx.MockTransport(handler))
    c = chat_client(pool)
    cid = c.post("/v1/conversations", json={}).json()["id"]
    # Force the failing replica to be tried first for this conversation.
    pool._ring_order = lambda key: ["http://bad:1", "http://good:1"]

    resp = c.post("/v1/chat", json={"conversation_id": cid, "message": "hi"})
    c.__exit__(None, None, None)

    assert resp.status_code == 200
    assert resp.json()["message"]["content"] == "ok"
    assert resp.json()["latency_ms"] >= 200
