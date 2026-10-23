import pytest

from app.prompting import (
    DEFAULT_HISTORY_TURNS,
    SYSTEM_PREAMBLE,
    flatten_history,
    history_message_limit,
    token_estimate,
)


def test_flatten_empty_history():
    prompt = flatten_history([], "Hello there")
    assert prompt.startswith(SYSTEM_PREAMBLE)
    assert prompt.endswith("User: Hello there\nAssistant:")


def test_flatten_orders_turns():
    history = [
        {"role": "user", "content": "first question"},
        {"role": "assistant", "content": "first answer"},
    ]
    prompt = flatten_history(history, "second question")
    a = prompt.index("User: first question")
    b = prompt.index("Assistant: first answer")
    c = prompt.index("User: second question")
    assert a < b < c
    assert prompt.endswith("Assistant:")


def test_flatten_is_deterministic():
    history = [{"role": "user", "content": "x"}]
    assert flatten_history(history, "y") == flatten_history(history, "y")


def test_token_estimate():
    assert token_estimate("") == 1
    assert token_estimate("one two three four five six") == 8  # 6 words / 0.75


def test_history_message_limit_default(monkeypatch):
    monkeypatch.delenv("CHAT_HISTORY_TURNS", raising=False)
    assert history_message_limit() == 2 * DEFAULT_HISTORY_TURNS == 20


def test_history_message_limit_from_env(monkeypatch):
    monkeypatch.setenv("CHAT_HISTORY_TURNS", "3")
    assert history_message_limit() == 6


@pytest.mark.parametrize("raw", ["abc", "", "2.5", "0", "-4"])
def test_history_message_limit_invalid_falls_back(monkeypatch, raw):
    monkeypatch.setenv("CHAT_HISTORY_TURNS", raw)
    assert history_message_limit() == 2 * DEFAULT_HISTORY_TURNS
