import hashlib
import json

import pytest

from app import audit


def test_audit_records_are_scrubbed_and_hash_chained(tmp_path, monkeypatch):
    log = tmp_path / "audit.log"
    monkeypatch.setattr(audit, "AUDIT_LOG", str(log))
    monkeypatch.setattr(audit, "_prev_digest", None)

    audit.write_audit({"path": "/v1/generate", "client": "bob@example.com"})
    audit.write_audit({"path": "/v1/chat", "client": "123-45-6789"})

    lines = [json.loads(line) for line in log.read_text().splitlines()]
    assert "bob@example.com" not in lines[0]["entry"]
    assert "123-45-6789" not in lines[1]["entry"]
    first = hashlib.sha256(lines[0]["entry"].encode()).digest()
    assert lines[0]["integrity"] == first.hex()
    second = hashlib.sha256(first + lines[1]["entry"].encode()).digest()
    assert lines[1]["integrity"] == second.hex()


@pytest.mark.parametrize("path_len", [10, 10_000])
def test_chain_resumes_from_existing_log(tmp_path, monkeypatch, path_len):
    log = tmp_path / "audit.log"
    monkeypatch.setattr(audit, "AUDIT_LOG", str(log))
    monkeypatch.setattr(audit, "_prev_digest", None)
    audit.write_audit({"path": "/" + "a" * path_len})
    monkeypatch.setattr(audit, "_prev_digest", None)  # simulate restart
    audit.write_audit({"n": 2})

    first, second = [json.loads(line)["integrity"] for line in log.read_text().splitlines()]
    expected = hashlib.sha256(bytes.fromhex(first) + b'{"n": 2}').hexdigest()
    assert second == expected
