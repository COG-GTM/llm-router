"""SQLite persistence for conversations and messages."""
import functools
import logging
import os
import sqlite3
import threading
import time
import uuid
from collections import deque

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    pinned INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS messages (
    id TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content TEXT NOT NULL,
    latency_ms INTEGER,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_conversation
    ON messages(conversation_id, created_at);
"""


def _row_to_dict(row: sqlite3.Row, drop: tuple[str, ...] = ()) -> dict:
    return {k: row[k] for k in row.keys() if k not in drop}


def _conversation_dict(row: sqlite3.Row) -> dict:
    conversation = _row_to_dict(row)
    conversation["pinned"] = bool(conversation["pinned"])
    return conversation


def new_message(conversation_id: str, role: str, content: str,
                latency_ms: int | None = None) -> dict:
    """Build a message (id and created_at assigned now) without persisting it."""
    return {
        "id": uuid.uuid4().hex,
        "conversation_id": conversation_id,
        "role": role,
        "content": content,
        "latency_ms": latency_ms,
        "created_at": time.time(),
    }


def _synchronized(method):
    """Run a Store method under the store lock, after applying deferred writes.

    The connection is shared across threads (`asyncio.to_thread`, background
    tasks), so every access is serialized; draining the deferred-write queue
    first means callers always read their own (deferred) writes.
    """
    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        with self._lock:
            self._drain_pending()
            return method(self, *args, **kwargs)
    return wrapper


class Store:
    def __init__(self, path: str):
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        self._lock = threading.RLock()
        self._pending: deque[tuple[list[dict], str | None]] = deque()
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.executescript(_SCHEMA)
        self._migrate_pinned_column()
        self._conn.commit()

    def _migrate_pinned_column(self) -> None:
        """Add the `pinned` column to a pre-existing database.

        `CREATE TABLE IF NOT EXISTS` in `_SCHEMA` is a no-op against a
        database file created before conversation pinning existed, so a
        legacy `conversations` table (with no `pinned` column) would
        otherwise stick around unchanged and every query that references
        `pinned` (e.g. `list_conversations`'s ORDER BY) would raise
        `sqlite3.OperationalError: no such column: pinned`. Backfill it here.
        """
        columns = {row["name"] for row in self._conn.execute("PRAGMA table_info(conversations)")}
        if "pinned" not in columns:
            self._conn.execute(
                "ALTER TABLE conversations ADD COLUMN pinned INTEGER NOT NULL DEFAULT 0"
            )

    # -- conversations -----------------------------------------------------

    @_synchronized
    def create_conversation(self, title: str = "New conversation") -> dict:
        now = time.time()
        conversation = {
            "id": uuid.uuid4().hex,
            "title": title,
            "created_at": now,
            "updated_at": now,
            "pinned": False,
        }
        self._conn.execute(
            "INSERT INTO conversations (id, title, created_at, updated_at, pinned)"
            " VALUES (?, ?, ?, ?, 0)",
            (conversation["id"], title, now, now),
        )
        self._conn.commit()
        return conversation

    @_synchronized
    def list_conversations(self, q: str | None = None) -> list[dict]:
        # Pinned conversations always float to the top; within each group,
        # most-recently-updated first.
        order_by = " ORDER BY pinned DESC, updated_at DESC"
        if q:
            # Escape SQL LIKE wildcards in the query itself, so searching for
            # e.g. a title containing a literal "%" or "_" does a plain
            # substring match instead of an unintended wildcard match.
            escaped = q.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
            rows = self._conn.execute(
                "SELECT * FROM conversations WHERE LOWER(title) LIKE LOWER(?) ESCAPE '\\'"
                + order_by,
                (f"%{escaped}%",),
            ).fetchall()
        else:
            rows = self._conn.execute("SELECT * FROM conversations" + order_by).fetchall()
        return [_conversation_dict(r) for r in rows]

    @_synchronized
    def get_conversation(self, conversation_id: str) -> dict | None:
        row = self._conn.execute(
            "SELECT * FROM conversations WHERE id = ?", (conversation_id,)
        ).fetchone()
        return _conversation_dict(row) if row else None

    @_synchronized
    def update_conversation(self, conversation_id: str, title: str | None = None,
                            pinned: bool | None = None) -> dict | None:
        """Partially update a conversation's title and/or pinned flag.

        Renaming bumps `updated_at` (as before). Pinning/unpinning does not:
        it's a user preference rather than conversation activity, so toggling
        it shouldn't reorder a conversation within its pinned/unpinned group
        on its own.
        """
        if title is None and pinned is None:
            return self.get_conversation(conversation_id)

        fields: list[str] = []
        params: list = []
        if title is not None:
            fields.append("title = ?")
            params.append(title)
            fields.append("updated_at = ?")
            params.append(time.time())
        if pinned is not None:
            fields.append("pinned = ?")
            params.append(1 if pinned else 0)
        params.append(conversation_id)

        cur = self._conn.execute(
            f"UPDATE conversations SET {', '.join(fields)} WHERE id = ?", params
        )
        self._conn.commit()
        if cur.rowcount == 0:
            return None
        return self.get_conversation(conversation_id)

    @_synchronized
    def delete_conversation(self, conversation_id: str) -> bool:
        cur = self._conn.execute(
            "DELETE FROM conversations WHERE id = ?", (conversation_id,)
        )
        self._conn.commit()
        return cur.rowcount > 0

    # -- messages ----------------------------------------------------------

    @_synchronized
    def add_message(self, conversation_id: str, role: str, content: str,
                    latency_ms: int | None = None) -> dict:
        message = new_message(conversation_id, role, content, latency_ms)
        self._insert_message(message)
        self._conn.commit()
        return message

    def _insert_message(self, message: dict) -> None:
        self._conn.execute(
            "INSERT INTO messages (id, conversation_id, role, content, latency_ms, created_at)"
            " VALUES (:id, :conversation_id, :role, :content, :latency_ms, :created_at)",
            message,
        )
        self._conn.execute(
            "UPDATE conversations SET updated_at = ? WHERE id = ?",
            (message["created_at"], message["conversation_id"]),
        )

    # -- deferred writes ---------------------------------------------------

    def defer_turn(self, messages: list[dict], title: str | None = None) -> None:
        """Queue a chat turn (its messages, plus an optional auto-title) for
        persistence by the next `flush()` or store call, whichever comes first."""
        with self._lock:
            self._pending.append((messages, title))

    @_synchronized
    def flush(self) -> None:
        """Apply all deferred writes (the work happens in `_synchronized`)."""

    def _drain_pending(self) -> None:
        """Write every queued turn in one transaction (a single commit/fsync)."""
        if not self._pending:
            return
        if not self._conn.in_transaction:
            # Explicit BEGIN: otherwise releasing the outermost savepoint
            # below would commit each turn on its own.
            self._conn.execute("BEGIN")
        while self._pending:
            messages, title = self._pending.popleft()
            # A savepoint per turn, so one bad turn (e.g. its conversation was
            # deleted while the turn was in flight) doesn't sink the batch.
            self._conn.execute("SAVEPOINT turn")
            try:
                if title is not None:
                    self._conn.execute(
                        "UPDATE conversations SET title = ? WHERE id = ?",
                        (title, messages[0]["conversation_id"]),
                    )
                for message in messages:
                    self._insert_message(message)
            except sqlite3.Error:
                self._conn.execute("ROLLBACK TO turn")
                logger.exception("dropping deferred chat turn for conversation %s",
                                 messages[0]["conversation_id"])
            finally:
                self._conn.execute("RELEASE turn")
        self._conn.commit()

    @_synchronized
    def list_messages(self, conversation_id: str, limit: int | None = None,
                      before: str | None = None) -> list[dict]:
        """List a conversation's messages in chronological order.

        With no arguments, returns the full history (unchanged behavior).
        `limit` caps how many are returned; `before` is a message id cursor —
        only messages that happened strictly before it are considered. The
        combination lets a caller page backwards through history: e.g.
        `limit=20` returns the most recent 20 turns, and passing the id of
        the oldest one back in as `before` fetches the 20 before that.
        """
        # Tie-break on the implicit sqlite rowid (insertion order), not the
        # message id: ids are random uuids, so using them to break ties
        # between same-timestamp rows would not reliably preserve the order
        # messages were actually written in.
        query = "SELECT *, rowid FROM messages WHERE conversation_id = ?"
        params: list = [conversation_id]
        if before:
            anchor = self._conn.execute(
                "SELECT created_at, rowid FROM messages WHERE id = ? AND conversation_id = ?",
                (before, conversation_id),
            ).fetchone()
            if anchor is not None:
                query += " AND (created_at, rowid) < (?, ?)"
                params += [anchor["created_at"], anchor["rowid"]]
        query += " ORDER BY created_at DESC, rowid DESC"
        if limit is not None:
            query += " LIMIT ?"
            params.append(limit)
        rows = self._conn.execute(query, params).fetchall()
        return [_row_to_dict(r, drop=("rowid",)) for r in reversed(rows)]

    def close(self) -> None:
        with self._lock:
            self._drain_pending()
            self._conn.close()
