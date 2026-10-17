from ron.contracts import MemoryItem
from ron.memory import InMemoryStore


def test_memory_recall_ranks_matching_items():
    store = InMemoryStore()
    store.remember(MemoryItem(key="name", content="اسم المستخدم زيريوس"))
    store.remember(MemoryItem(key="other", content="weather today"))
    hits = store.recall("ما اسم المستخدم؟", limit=2)
    assert hits[0].key == "name"


def test_arabic_tokens_are_retrievable():
    store = InMemoryStore()
    store.remember(MemoryItem(key="identity", content="اسمي زيريوس"))
    assert store.recall("زيريوس", limit=1)[0].content == "اسمي زيريوس"


def test_memory_rejects_empty_items():
    store = InMemoryStore()
    try:
        store.remember(MemoryItem(key="", content="value"))
    except ValueError:
        pass
    else:
        raise AssertionError("empty keys must be rejected")


def test_memory_limit_and_unknown_query():
    store = InMemoryStore()
    store.remember(MemoryItem(key="one", content="alpha beta"))
    assert store.recall("not-present", limit=1) == []
    assert store.recall("alpha", limit=1)[0].key == "one"



def test_sqlite_memory_survives_reopening_and_updates_existing_keys(tmp_path):
    from ron.memory import SQLiteMemoryStore

    database = tmp_path / "ron-memory.sqlite3"
    first = SQLiteMemoryStore(str(database))
    first.remember(MemoryItem(key="user.name", content="أركانوس", metadata={"kind": "name"}))
    first.remember(MemoryItem(key="user.name", content="أركانوس", metadata={"kind": "profile"}))

    reopened = SQLiteMemoryStore(str(database))
    hits = reopened.recall("أركانوس", limit=1)
    assert len(hits) == 1
    assert hits[0].key == "user.name"
    assert hits[0].metadata == {"kind": "profile"}


def test_sqlite_memory_rejects_empty_items_without_persisting(tmp_path):
    import pytest
    from ron.memory import SQLiteMemoryStore

    store = SQLiteMemoryStore(str(tmp_path / "memory.sqlite3"))
    with pytest.raises(ValueError):
        store.remember(MemoryItem(key="valid", content=""))
    assert store.recall("valid", limit=1) == []


def test_core_uses_sqlite_memory_only_when_explicitly_configured(tmp_path, monkeypatch):
    from ron.core import RonCore
    from ron.memory import InMemoryStore, SQLiteMemoryStore

    database = tmp_path / "ron-memory.sqlite3"
    monkeypatch.setenv("RON_MEMORY_DB_PATH", str(database))
    core = RonCore(provider=object())
    assert isinstance(core.memory, SQLiteMemoryStore)
    core.memory.remember(MemoryItem(key="user.name", content="أركانوس"))
    reopened = SQLiteMemoryStore(str(database))
    assert reopened.recall("أركانوس", limit=1)[0].content == "أركانوس"

    monkeypatch.delenv("RON_MEMORY_DB_PATH")
    isolated = RonCore(provider=object())
    assert isinstance(isolated.memory, InMemoryStore)
