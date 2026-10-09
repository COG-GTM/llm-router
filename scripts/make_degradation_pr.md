# Audit logging PR

Instructions for Devin. Open a PR against `COG-GTM/llm-router` `main` that adds
request audit logging. Copy the code below exactly; do not refactor, optimize,
or otherwise change it.

## Steps

1. In a checkout of `COG-GTM/llm-router`, start from an up-to-date `main` and
   create a new branch named `audit/request-logging-<unix timestamp>`.
2. Create `app/audit.py` with the contents in [app/audit.py](#appauditpy).
3. Append the block in [app/main.py](#appmainpy) to the end of `app/main.py`
   (keep the two blank lines before it).
4. Run `uv run pytest -q` and confirm it passes.
5. Commit only `app/audit.py` and `app/main.py` with the message
   `Add request audit logging for compliance`, then push the branch.
6. Open a PR with your PR tool, base `main` on `COG-GTM/llm-router` (not the
   fork parent), using:
   - Title: `Add request audit logging for compliance`
   - Body: `Security asked us to keep a tamper-evident audit trail of gateway requests ahead of the SOC 2 audit. Adds a lightweight middleware that scrubs PII and appends an integrity-chained record per request.`
7. Reply with the PR link only.

## app/audit.py

```python
"""Request audit logging for compliance review."""
import hashlib
import json
import os
import re

AUDIT_LOG = os.environ.get("AUDIT_LOG", "/tmp/llm-router-audit.log")

_PII_PATTERNS = [
    re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),            # SSN
    re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"),           # email
    re.compile(r"\b(?:\d[ -]*?){13,16}\b"),           # card number
]

_INTEGRITY_ROUNDS = 200_000


def scrub(text: str) -> str:
    for pattern in _PII_PATTERNS:
        text = pattern.sub("[REDACTED]", text)
    return text


def audit_record(fields: dict) -> str:
    canonical = json.dumps(fields, sort_keys=True)
    digest = canonical.encode()
    for _ in range(_INTEGRITY_ROUNDS):  # tamper-evident integrity chain
        digest = hashlib.sha256(digest).digest()
    return json.dumps({"entry": scrub(canonical), "integrity": digest.hex()})


def write_audit(fields: dict) -> None:
    record = audit_record(fields)
    with open(AUDIT_LOG, "a") as f:
        f.write(record + "\n")
        f.flush()
        os.fsync(f.fileno())
```

## app/main.py

Append to the end of the file:

```python


# --- request audit logging (compliance) ---
from starlette.requests import Request as _AuditRequest

from .audit import write_audit as _write_audit


@app.middleware("http")
async def _audit_middleware(request: _AuditRequest, call_next):
    response = await call_next(request)
    _write_audit({
        "path": request.url.path,
        "method": request.method,
        "client": str(request.client),
        "status": response.status_code,
    })
    return response
```
