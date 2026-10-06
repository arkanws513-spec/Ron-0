from ron.contracts import MemoryItem
from ron.memory import InMemoryStore

def test_memory_recall_ranks_matching_items():
    store = InMemoryStore()
    store.remember(MemoryItem(key="name", content="Ron is the assistant"))
    store.remember(MemoryItem(key="other", content="weather today"))
    hits = store.recall("assistant", limit=2)
    assert hits[0].key == "name"