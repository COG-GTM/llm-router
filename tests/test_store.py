import sqlite3
from concurrent.futures import ThreadPoolExecutor

from app.store import Store, new_message


def make_store(tmp_path):
    return Store(str(tmp_path / "test.db"))


def test_conversation_crud(tmp_path):
    s = make_store(tmp_path)
    c = s.create_conversation()
    assert c["title"] == "New conversation"
    assert c["pinned"] is False
    assert s.get_conversation(c["id"])["id"] == c["id"]

    renamed = s.update_conversation(c["id"], title="Renamed")
    assert renamed["title"] == "Renamed"
    assert s.update_conversation("nope", title="x") is None

    c2 = s.create_conversation(title="Second")
    ids = [row["id"] for row in s.list_conversations()]
    assert set(ids) == {c["id"], c2["id"]}

    assert s.delete_conversation(c["id"]) is True
    assert s.delete_conversation(c["id"]) is False
    assert s.get_conversation(c["id"]) is None
    s.close()


def test_messages_roundtrip_and_order(tmp_path):
    s = make_store(tmp_path)
    c = s.create_conversation()
    m1 = s.add_message(c["id"], "user", "hello")
    m2 = s.add_message(c["id"], "assistant", "hi there", latency_ms=1234)
    msgs = s.list_messages(c["id"])
    assert [m["id"] for m in msgs] == [m1["id"], m2["id"]]
    assert msgs[0]["latency_ms"] is None
    assert msgs[1]["latency_ms"] == 1234
    s.close()


def test_delete_cascades_messages(tmp_path):
    s = make_store(tmp_path)
    c = s.create_conversation()
    s.add_message(c["id"], "user", "hello")
    s.delete_conversation(c["id"])
    assert s.list_messages(c["id"]) == []
    s.close()


def test_creates_parent_directory(tmp_path):
    s = Store(str(tmp_path / "nested" / "dir" / "app.db"))
    assert s.create_conversation()["id"]
    s.close()


def test_list_conversations_search_is_case_insensitive_substring(tmp_path):
    s = make_store(tmp_path)
    match = s.create_conversation(title="Kubernetes migration plan")
    s.create_conversation(title="Weekend recipe ideas")

    assert [c["id"] for c in s.list_conversations(q="kubernetes")] == [match["id"]]
    assert [c["id"] for c in s.list_conversations(q="MIGRATION")] == [match["id"]]
    assert s.list_conversations(q="no such thing") == []
    assert len(s.list_conversations(q=None)) == 2
    assert len(s.list_conversations(q="")) == 2
    s.close()


def test_list_conversations_search_case_folding_is_ascii_only(tmp_path):
    # SQLite's LOWER()/LIKE case-folding only covers ASCII A-Z without the
    # ICU extension. This documents that real, current limitation (matching
    # accented letters case-insensitively is out of scope) rather than
    # leaving it as an untested assumption.
    s = make_store(tmp_path)
    s.create_conversation(title="café notes")
    assert [c["title"] for c in s.list_conversations(q="café")] == ["café notes"]
    assert s.list_conversations(q="CAFÉ") == []
    s.close()


def test_list_conversations_search_escapes_sql_wildcards(tmp_path):
    s = make_store(tmp_path)
    percent_title = s.create_conversation(title="50% done")
    s.create_conversation(title="anything at all")

    # A literal "%" in the query should match only titles containing that
    # literal character, not act as a SQL wildcard matching everything.
    assert [c["id"] for c in s.list_conversations(q="50%")] == [percent_title["id"]]
    assert [c["id"] for c in s.list_conversations(q="%")] == [percent_title["id"]]
    s.close()


def test_list_messages_limit_returns_most_recent(tmp_path):
    s = make_store(tmp_path)
    c = s.create_conversation()
    ids = [s.add_message(c["id"], "user", f"turn {n}")["id"] for n in range(4)]

    assert [m["id"] for m in s.list_messages(c["id"], limit=2)] == ids[-2:]
    assert [m["id"] for m in s.list_messages(c["id"])] == ids
    s.close()


def test_list_messages_before_cursor_pages_backwards(tmp_path):
    s = make_store(tmp_path)
    c = s.create_conversation()
    ids = [s.add_message(c["id"], "user", f"turn {n}")["id"] for n in range(5)]

    newest_two = s.list_messages(c["id"], limit=2)
    assert [m["id"] for m in newest_two] == ids[-2:]

    older_two = s.list_messages(c["id"], limit=2, before=newest_two[0]["id"])
    assert [m["id"] for m in older_two] == ids[-4:-2]

    # an unknown cursor is ignored rather than erroring
    assert [m["id"] for m in s.list_messages(c["id"], before="nope")] == ids
    s.close()


def test_pinned_conversations_sort_before_unpinned(tmp_path):
    s = make_store(tmp_path)
    old = s.create_conversation(title="Old, will be pinned")
    s.create_conversation(title="Newer, unpinned")

    # Before pinning, recency order puts "old" last.
    assert [c["id"] for c in s.list_conversations()][-1] == old["id"]

    pinned = s.update_conversation(old["id"], pinned=True)
    assert pinned["pinned"] is True

    # After pinning, it floats to the top despite being older.
    ordered = s.list_conversations()
    assert ordered[0]["id"] == old["id"]
    assert ordered[0]["pinned"] is True
    assert ordered[1]["pinned"] is False
    s.close()


def test_multiple_pinned_conversations_still_order_by_recency_within_the_group(tmp_path):
    s = make_store(tmp_path)
    first = s.create_conversation(title="First")
    second = s.create_conversation(title="Second")
    s.update_conversation(first["id"], pinned=True)
    s.update_conversation(second["id"], pinned=True)

    ordered = [c["id"] for c in s.list_conversations()]
    # Both pinned; `second` was created (and thus last updated) more
    # recently, so it stays on top of the pinned group.
    assert ordered == [second["id"], first["id"]]
    s.close()


def test_pinning_does_not_bump_updated_at(tmp_path):
    s = make_store(tmp_path)
    c = s.create_conversation()
    before = s.get_conversation(c["id"])["updated_at"]

    s.update_conversation(c["id"], pinned=True)

    after = s.get_conversation(c["id"])["updated_at"]
    assert after == before
    s.close()


def test_update_conversation_can_set_title_and_pinned_together(tmp_path):
    s = make_store(tmp_path)
    c = s.create_conversation()
    updated = s.update_conversation(c["id"], title="Renamed and pinned", pinned=True)
    assert updated["title"] == "Renamed and pinned"
    assert updated["pinned"] is True
    s.close()


def test_update_conversation_with_no_fields_is_a_no_op(tmp_path):
    s = make_store(tmp_path)
    c = s.create_conversation(title="Untouched")
    result = s.update_conversation(c["id"])
    assert result["title"] == "Untouched"
    assert result["pinned"] is False
    s.close()


def test_list_messages_before_cursor_from_another_conversation_is_ignored(tmp_path):
    s = make_store(tmp_path)
    c1 = s.create_conversation()
    c2 = s.create_conversation()
    other_id = s.add_message(c2["id"], "user", "from another conversation")["id"]
    mine = [s.add_message(c1["id"], "user", f"turn {n}")["id"] for n in range(3)]

    # a cursor id that exists, but belongs to a different conversation,
    # must not leak that conversation's position into this one's results
    assert [m["id"] for m in s.list_messages(c1["id"], before=other_id)] == mine
    s.close()


def test_store_migrates_a_pre_existing_database_missing_the_pinned_column(tmp_path):
    # Simulate a database file created before conversation pinning existed:
    # a `conversations` table with no `pinned` column at all. `Store.__init__`
    # must backfill the column rather than leave every `pinned`-referencing
    # query (e.g. list_conversations's ORDER BY) raising OperationalError.
    db_path = tmp_path / "legacy.db"
    legacy = sqlite3.connect(str(db_path))
    legacy.execute(
        """
        CREATE TABLE conversations (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL
        )
        """
    )
    legacy.execute(
        "INSERT INTO conversations (id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
        ("legacy-1", "Pre-existing chat", 1.0, 1.0),
    )
    legacy.commit()
    legacy.close()

    s = Store(str(db_path))
    conversations = s.list_conversations()
    assert len(conversations) == 1
    assert conversations[0]["id"] == "legacy-1"
    assert conversations[0]["pinned"] is False

    # Pinning round-trips normally once the column has been backfilled.
    updated = s.update_conversation("legacy-1", pinned=True)
    assert updated["pinned"] is True
    assert s.get_conversation("legacy-1")["pinned"] is True
    s.close()


def test_deferred_turn_applied_before_reads_and_on_close(tmp_path):
    s = make_store(tmp_path)
    c = s.create_conversation()
    user = new_message(c["id"], "user", "hi")
    assistant = new_message(c["id"], "assistant", "hello", latency_ms=7)
    s.defer_turn([user, assistant], title="hi")
    assert s.list_messages(c["id"]) == [user, assistant]
    assert s.get_conversation(c["id"])["title"] == "hi"
    assert s.get_conversation(c["id"])["updated_at"] == assistant["created_at"]

    s.defer_turn([new_message(c["id"], "user", "queued at shutdown")])
    s.close()
    s = make_store(tmp_path)
    assert [m["content"] for m in s.list_messages(c["id"])][-1] == "queued at shutdown"
    s.close()


def test_deferred_turn_for_deleted_conversation_is_dropped(tmp_path):
    s = make_store(tmp_path)
    c = s.create_conversation()
    s.defer_turn([new_message("gone", "user", "orphan")])
    s.defer_turn([new_message(c["id"], "user", "kept")])
    assert [m["content"] for m in s.list_messages(c["id"])] == ["kept"]
    s.close()


def test_store_is_safe_across_threads(tmp_path):
    s = make_store(tmp_path)
    c = s.create_conversation()

    def work(n):
        s.add_message(c["id"], "user", f"t{n}")
        s.defer_turn([new_message(c["id"], "assistant", f"d{n}")])
        return len(s.list_messages(c["id"]))

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(work, range(80)))
    assert len(s.list_messages(c["id"])) == 160
    s.close()
