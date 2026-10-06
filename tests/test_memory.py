from ron.contracts import MemoryItem
from ron.memory import InMemoryStore


def test_memory_recall_ranks_matching_items():
    store = InMemoryStore()
    store.remember(MemoryItem(key="name", content="اسم المستخدم زيريوس"))
    store.remember(MemoryItem(key="other", content="weather today"))
    hits = store.recall("ما اسم المستخدم؟", limit=2)
    assert hits[0].key == "name"


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
