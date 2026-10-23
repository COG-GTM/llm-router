import asyncio
import os
import sqlite3
from unittest.mock import patch

import httpx
from fastapi.testclient import TestClient

from app import main


class FakePool:
    def __init__(self, status=200):
        self.status = status
        self.calls: list[dict] = []

    async def forward(self, payload):
        self.calls.append(payload)
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


def test_chat_returns_full_turn_and_persists_it():
    pool = FakePool()
    c = chat_client(pool)
    cid = c.post("/v1/conversations", json={}).json()["id"]
    resp = c.post("/v1/chat", json={"conversation_id": cid, "message": "hello"})
    body = resp.json()
    store = main.app.state.store
    store.flush()
    # read through a separate connection: the turn must be on disk, not just queued
    with sqlite3.connect(os.environ["APP_DB_PATH"]) as raw:
        rows = raw.execute(
            "SELECT id, role, content, latency_ms FROM messages"
            " WHERE conversation_id = ? ORDER BY rowid", (cid,)
        ).fetchall()
        title = raw.execute("SELECT title FROM conversations WHERE id = ?", (cid,)).fetchone()[0]
    msgs = c.get(f"/v1/conversations/{cid}/messages").json()
    c.__exit__(None, None, None)

    assert resp.status_code == 200
    message = body["message"]
    assert set(message) == {"id", "conversation_id", "role", "content", "latency_ms", "created_at"}
    assert message["conversation_id"] == cid
    assert message["role"] == "assistant"
    assert message["latency_ms"] == body["latency_ms"]
    assert isinstance(message["created_at"], float)
    assert body["model"] == "mock-large"
    assert body["usage"] == {"prompt_tokens": 5, "completion_tokens": 2}

    assert [(r[1], r[2]) for r in rows] == [("user", "hello"), ("assistant", "mock answer")]
    assert rows[1][0] == message["id"]
    assert rows[1][3] == body["latency_ms"]
    assert msgs[-1] == message  # the returned message is exactly what was stored
    assert title == "hello"


def test_chat_auto_titles_only_first_turn():
    c = chat_client(FakePool())
    cid = c.post("/v1/conversations", json={}).json()["id"]
    c.post("/v1/chat", json={"conversation_id": cid, "message": "x" * 100})
    c.post("/v1/chat", json={"conversation_id": cid, "message": "second turn"})
    convo = next(x for x in c.get("/v1/conversations").json() if x["id"] == cid)
    c.__exit__(None, None, None)
    assert convo["title"] == "x" * 48


def test_chat_pending_turn_visible_to_immediate_reads():
    """A turn that is still queued (background flush not yet run) must be
    visible to the very next read and included in the next turn's prompt."""
    pool = FakePool()
    c = chat_client(pool)
    store = main.app.state.store
    cid = c.post("/v1/conversations", json={}).json()["id"]
    with patch.object(store, "flush"):  # suppress the post-response flush
        c.post("/v1/chat", json={"conversation_id": cid, "message": "first"})
        assert store._pending
        msgs = c.get(f"/v1/conversations/{cid}/messages").json()
        convo = next(x for x in c.get("/v1/conversations").json() if x["id"] == cid)
        c.post("/v1/chat", json={"conversation_id": cid, "message": "second"})
    c.__exit__(None, None, None)

    assert [m["content"] for m in msgs] == ["first", "mock answer"]
    assert convo["title"] == "first"
    assert "User: first\nAssistant: mock answer" in pool.calls[1]["prompt"]


def test_chat_concurrent_turns_all_persisted():
    async def run():
        async with main.app.router.lifespan_context(main.app):
            main.app.state.pool = FakePool()
            transport = httpx.ASGITransport(app=main.app)
            async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
                cids = [(await client.post("/v1/conversations", json={})).json()["id"]
                        for _ in range(3)]
                resps = await asyncio.gather(*(
                    client.post("/v1/chat", json={"conversation_id": cids[i % 3],
                                                  "message": f"m{i}"})
                    for i in range(30)
                ))
                counts = [len((await client.get(f"/v1/conversations/{cid}/messages")).json())
                          for cid in cids]
        return resps, counts

    resps, counts = asyncio.run(run())
    assert all(r.status_code == 200 for r in resps)
    assert counts == [20, 20, 20]
