import httpx
import pytest
from fastapi.testclient import TestClient

from app import main


class FakePool:
    def __init__(self, status=200, body=None):
        self.status = status
        self.body = body or {"completion": "ok", "signature": "ab" * 32}
        self.calls: list[dict] = []

    async def forward(self, payload):
        self.calls.append(payload)
        return self.status, self.body

    async def aclose(self):
        pass


def client_with(pool: FakePool) -> TestClient:
    c = TestClient(main.app)
    c.__enter__()
    main.app.state.pool = pool
    return c


def test_generate_passes_through_body_and_payload():
    pool = FakePool()
    c = client_with(pool)
    r = c.post("/v1/generate", json={"prompt": "hi", "max_tokens": 8})
    c.__exit__(None, None, None)
    assert r.status_code == 200
    assert r.json() == pool.body
    assert pool.calls == [{"prompt": "hi", "max_tokens": 8}]


def test_generate_passes_through_error_status():
    pool = FakePool(status=503, body={"detail": "model overloaded"})
    c = client_with(pool)
    r = c.post("/v1/generate", json={"prompt": "hi"})
    c.__exit__(None, None, None)
    assert r.status_code == 503


def test_generate_validates_prompt():
    pool = FakePool()
    c = client_with(pool)
    r = c.post("/v1/generate", json={})
    c.__exit__(None, None, None)
    assert r.status_code == 422


@pytest.mark.parametrize("bad", [0, -1, 1025])
def test_generate_rejects_out_of_range_max_tokens(bad):
    pool = FakePool()
    c = client_with(pool)
    r = c.post("/v1/generate", json={"prompt": "hi", "max_tokens": bad})
    c.__exit__(None, None, None)
    assert r.status_code == 422
    assert pool.calls == []


def test_generate_accepts_max_tokens_at_limit():
    pool = FakePool()
    c = client_with(pool)
    r = c.post("/v1/generate", json={"prompt": "hi", "max_tokens": 1024})
    c.__exit__(None, None, None)
    assert r.status_code == 200
    assert pool.calls == [{"prompt": "hi", "max_tokens": 1024}]
