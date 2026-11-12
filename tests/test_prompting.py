from app.prompting import SYSTEM_PREAMBLE, flatten_history, token_estimate, trim_history


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


def _msgs(n, words=1):
    return [{"role": "user" if i % 2 == 0 else "assistant",
             "content": " ".join([f"m{i}"] * words)} for i in range(n)]


def test_trim_history_keeps_most_recent_within_message_cap():
    kept = trim_history(_msgs(10), max_messages=4, token_budget=10_000)
    assert [m["content"] for m in kept] == ["m6", "m7", "m8", "m9"]


def test_trim_history_respects_token_budget():
    # each message is 3 words -> token_estimate == 4
    kept = trim_history(_msgs(10, words=3), max_messages=100, token_budget=10)
    assert [m["content"].split()[0] for m in kept] == ["m8", "m9"]


def test_trim_history_short_history_unchanged():
    history = _msgs(3)
    assert trim_history(history) == history
