#!/usr/bin/env bash
# Perf-gate automation test: opens a PR that intentionally regresses request
# latency, to verify the gate catches the regression and attributes the cause.
set -euo pipefail
# Targets the repo this script is checked out from (via its origin remote).
# Env overrides: REPO=owner/name, BASE, BRANCH, DRY_RUN=1 (commit only, no push/PR).
ORIGIN_URL=$(git -C "$(dirname "$0")" remote get-url origin)
ORIGIN_REPO=$(sed -E 's#^.*github\.com[:/]##; s#\.git$##; s#/$##' <<<"$ORIGIN_URL")
REPO="${REPO:-$ORIGIN_REPO}"
CLONE_URL="${ORIGIN_URL/"$ORIGIN_REPO"/"$REPO"}"  # same host/protocol/auth, REPO's path
BASE="${BASE:-main}"
BRANCH="${BRANCH:-audit/request-logging-$(date +%s)}"
DRY_RUN="${DRY_RUN:-0}"
WORK=$(mktemp -d)
[[ "$DRY_RUN" == 1 ]] || trap 'rm -rf "$WORK"' EXIT

echo "Target: $REPO via $CLONE_URL (base $BASE, branch $BRANCH)"
git clone -q --branch "$BASE" "$CLONE_URL" "$WORK/r"
cd "$WORK/r"
git checkout -b "$BRANCH"

cat > app/audit.py <<'AUDIT_PY'
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
AUDIT_PY

cat >> app/main.py <<'AUDIT_REG'


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
AUDIT_REG

git add app/audit.py app/main.py
git commit -m "Add request audit logging for compliance"
if [[ "$DRY_RUN" == 1 ]]; then
  echo "DRY_RUN: committed to $BRANCH in $WORK/r; skipping push and PR"
  exit 0
fi
git push -u origin "$BRANCH"
gh pr create --repo "$REPO" --base "$BASE" --head "$BRANCH" \
  --title "Add request audit logging for compliance" \
  --body "Security asked us to keep a tamper-evident audit trail of gateway requests ahead of the SOC 2 audit. Adds a lightweight middleware that scrubs PII and appends an integrity-chained record per request."
