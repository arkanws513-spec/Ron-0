from ron.contracts import MemoryItem
from ron.facts import answer_fact_question, extract_fact
from ron.memory import InMemoryStore


def test_extracts_name_without_special_command():
    fact = extract_fact("اريدك أن تعلم أن اسمي هو زيريوس")
    assert fact is not None
    assert fact.key == "user.name"
    assert fact.value == "زيريوس"


def test_extracts_preference():
    fact = extract_fact("أنا أحب البرمجة")
    assert fact is not None
    assert fact.key == "user.preference"
    assert fact.value == "البرمجة"


def test_answers_name_from_memory():
    store = InMemoryStore()
    store.remember(MemoryItem(key="user.name", content="زيريوس"))
    assert answer_fact_question("ما اسمي؟", store) == "اسمك زيريوس."


def test_arabic_variants_of_name_question_are_supported():
    store = InMemoryStore()
    store.remember(MemoryItem(key="user.name", content="زيريوس"))
    assert answer_fact_question("إيه اسمي؟", store) == "اسمك زيريوس."


def test_unknown_fact_question_is_safe():
    store = InMemoryStore()
    assert answer_fact_question("ما اسمي؟", store) == "لم تخبرني باسمك بعد."
