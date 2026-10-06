from ron.contracts import MemoryItem, Message, ModelRequest

def test_model_request_is_provider_agnostic():
    request = ModelRequest(messages=(Message(role="user", content="Hello Ron"),))
    assert request.messages[0].role == "user"
    assert request.messages[0].content == "Hello Ron"

def test_memory_item_is_structured():
    item = MemoryItem(key="example", content="Ron-0")
    assert item.key == "example"
    assert item.content == "Ron-0"
