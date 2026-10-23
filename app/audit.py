"""Request audit logging for compliance review."""
import hashlib
import json
import os
import re
import threading

AUDIT_LOG = os.environ.get("AUDIT_LOG", "/tmp/llm-router-audit.log")

_PII_PATTERNS = [
    re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),            # SSN
    re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"),           # email
    re.compile(r"\b(?:\d[ -]*?){13,16}\b"),           # card number
]

_lock = threading.Lock()
_prev_digest: bytes | None = None


def scrub(text: str) -> str:
    for pattern in _PII_PATTERNS:
        text = pattern.sub("[REDACTED]", text)
    return text


def audit_record(fields: dict, prev_digest: bytes = b"") -> tuple[str, bytes]:
    """Build a log line whose integrity hash chains to the previous entry (tamper-evident)."""
    canonical = json.dumps(fields, sort_keys=True)
    digest = hashlib.sha256(prev_digest + canonical.encode()).digest()
    return json.dumps({"entry": scrub(canonical), "integrity": digest.hex()}), digest


def _last_digest(path: str) -> bytes:
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            f.seek(max(0, f.tell() - 4096))
            lines = f.read().splitlines()
        return bytes.fromhex(json.loads(lines[-1])["integrity"]) if lines else b""
    except (OSError, ValueError, KeyError, IndexError):
        return b""


def write_audit(fields: dict) -> None:
    """Blocking (fsync); call from a worker thread, not the event loop."""
    global _prev_digest
    with _lock:
        if _prev_digest is None:
            _prev_digest = _last_digest(AUDIT_LOG)
        record, _prev_digest = audit_record(fields, _prev_digest)
        with open(AUDIT_LOG, "a") as f:
            f.write(record + "\n")
            f.flush()
            os.fsync(f.fileno())
