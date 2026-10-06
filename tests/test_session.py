import pytest
from ron.session import Conversation

def test_conversation_keeps_order():
    chat = Conversation("demo")
    chat.add("user", "مرحبا")
    chat.add("assistant", "أهلا")
    assert [m.role for m in chat.history()] == ["user", "assistant"]

def test_empty_messages_are_rejected():
    chat = Conversation("demo")
    with pytest.raises(ValueError):
        chat.add("user", "   ")
