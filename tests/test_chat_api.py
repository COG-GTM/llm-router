import pytest
from fastapi.testclient import TestClient

from app import main
from app.config import chat_history_turns


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


class NumberedPool(FakePool):
    """Returns a distinct completion per call so prompt ordering is checkable."""

    async def forward(self, payload):
        self.calls.append(payload)
        return 200, {"completion": f"answer {len(self.calls)}", "model": "mock-large"}


def run_turns(c, cid, n):
    for i in range(1, n + 1):
        c.post("/v1/chat", json={"conversation_id": cid, "message": f"question {i}"})


def test_chat_prompt_includes_only_last_n_turns_in_order(monkeypatch):
    monkeypatch.setenv("CHAT_HISTORY_TURNS", "3")
    pool = NumberedPool()
    c = chat_client(pool)
    cid = c.post("/v1/conversations", json={}).json()["id"]
    run_turns(c, cid, 8)
    c.post("/v1/chat", json={"conversation_id": cid, "message": "question 9"})
    msgs = c.get(f"/v1/conversations/{cid}/messages").json()
    c.__exit__(None, None, None)

    assert len(msgs) == 18  # full history is still persisted
    prompt = pool.calls[-1]["prompt"]
    for i in range(1, 6):
        assert f"User: question {i}\n" not in prompt
        assert f"Assistant: answer {i}\n" not in prompt
    expected = [
        "User: question 6", "Assistant: answer 6",
        "User: question 7", "Assistant: answer 7",
        "User: question 8", "Assistant: answer 8",
        "User: question 9", "Assistant:",
    ]
    assert prompt.split("\n")[2:] == expected


def test_chat_prompt_bounded_by_default():
    pool = NumberedPool()
    c = chat_client(pool)
    cid = c.post("/v1/conversations", json={}).json()["id"]
    run_turns(c, cid, 50)
    c.__exit__(None, None, None)

    prompt = pool.calls[-1]["prompt"]
    history_lines = prompt.split("\n")[2:-2]
    assert len(history_lines) == 20  # default: last 10 turns
    assert history_lines[0] == "User: question 40"
    assert history_lines[-1] == "Assistant: answer 49"
    assert len(pool.calls[29]["prompt"]) == pytest.approx(len(prompt), abs=20)


def test_chat_auto_title_unaffected_by_history_limit(monkeypatch):
    monkeypatch.setenv("CHAT_HISTORY_TURNS", "1")
    pool = NumberedPool()
    c = chat_client(pool)
    cid = c.post("/v1/conversations", json={}).json()["id"]
    run_turns(c, cid, 3)
    convo = next(x for x in c.get("/v1/conversations").json() if x["id"] == cid)
    c.__exit__(None, None, None)

    assert pool.calls[0]["prompt"].split("\n")[2:] == ["User: question 1", "Assistant:"]
    assert convo["title"] == "question 1"  # titled on turn 1, not re-titled later


@pytest.mark.parametrize("raw,expected", [("", 10), ("4", 4), ("0", 1), ("-2", 1), ("ten", 10)])
def test_chat_history_turns_env(monkeypatch, raw, expected):
    monkeypatch.setenv("CHAT_HISTORY_TURNS", raw)
    assert chat_history_turns() == expected
