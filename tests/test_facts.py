from ron.contracts import MemoryItem
from ron.facts import answer_fact_question, extract_fact, extract_facts
from ron.memory import InMemoryStore


def test_extracts_name_without_special_command():
    fact = extract_fact("اريدك أن تعلم أن اسمي هو زيريوس")
    assert fact is not None
    assert fact.key == "user.name"
    assert fact.value == "زيريوس"


def test_extracts_natural_name():
    fact = extract_fact("انا زيريوس الاول")
    assert fact is not None
    assert fact.key == "user.name"
    assert fact.value == "زيريوس الاول"


def test_extracts_age():
    fact = extract_fact("انا عمري 34 عام")
    assert fact is not None
    assert fact.key == "user.age"
    assert fact.value == "34"


def test_extracts_combined_name_and_age():
    facts = extract_facts("انا زيريوس وعمري 34 عام")
    assert [(fact.key, fact.value) for fact in facts] == [
        ("user.name", "زيريوس"),
        ("user.age", "34"),
    ]


def test_extracts_name_and_age_correction():
    facts = extract_facts("اسمي زيريوس الاول فقط اما 34 فهذا عمري")
    assert [(fact.key, fact.value) for fact in facts] == [
        ("user.name", "زيريوس الاول"),
        ("user.age", "34"),
    ]


def test_extracts_preference():
    fact = extract_fact("أنا أحب البرمجة")
    assert fact is not None
    assert fact.key == "user.preference"
    assert fact.value == "البرمجة"


def test_answers_name_from_memory():
    store = InMemoryStore()
    store.remember(MemoryItem(key="user.name", content="زيريوس"))
    assert answer_fact_question("ما اسمي؟", store) == "اسمك زيريوس."


def test_answers_age_from_memory():
    store = InMemoryStore()
    store.remember(MemoryItem(key="user.age", content="34"))
    assert answer_fact_question("كم عمري؟", store) == "عمرك 34 سنة."


def test_answers_name_and_age_from_memory():
    store = InMemoryStore()
    store.remember(MemoryItem(key="user.name", content="زيريوس"))
    store.remember(MemoryItem(key="user.age", content="34"))
    assert answer_fact_question("ما اسمي وما عمري؟", store) == "اسمك زيريوس، وعمرك 34 سنة."


def test_arabic_variants_of_name_question_are_supported():
    store = InMemoryStore()
    store.remember(MemoryItem(key="user.name", content="زيريوس"))
    assert answer_fact_question("إيه اسمي؟", store) == "اسمك زيريوس."


def test_unknown_fact_question_is_safe():
    store = InMemoryStore()
    assert answer_fact_question("ما اسمي؟", store) == "لم تخبرني باسمك بعد."
