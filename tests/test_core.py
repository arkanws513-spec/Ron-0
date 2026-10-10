from ron.contracts import ModelRequest, ModelResponse
from ron.core import RonCore
from ron.self_improvement import Experience


class FakeProvider:
    def generate(self, request: ModelRequest) -> ModelResponse:
        return ModelResponse(
            content=request.messages[-1].content,
            model="fake",
            metadata=request.metadata,
        )


def test_core_injects_memory_and_records_turn():
    core = RonCore(provider=FakeProvider())
    first = core.respond("remember Ron")
    assert first.content == "remember Ron"
    second = core.respond("Ron")
    assert second.content == "Ron"
    assert len(core.memory.items) == 2


def test_core_learns_multiple_profile_facts_from_one_message():
    core = RonCore(provider=FakeProvider())
    response = core.respond("انا زيريوس وعمري 34 عام")
    assert response.content == "تم. سأحفظ أن اسمك زيريوس وأن عمرك 34 سنة."
    assert response.metadata["facts_written"] == 2

    name = core.respond("ما اسمي")
    age = core.respond("كم عمري")
    assert name.content == "اسمك زيريوس."
    assert age.content == "عمرك 34 سنة."


def test_core_learns_correction_without_mixing_name_and_age():
    core = RonCore(provider=FakeProvider())
    core.respond("اسمي زيريوس الاول فقط اما 34 فهذا عمري")
    assert core.respond("ما اسمي").content == "اسمك زيريوس الاول."
    assert core.respond("كم عمري").content == "عمرك 34 سنة."


def test_core_exposes_explicit_learning_and_active_skills():
    core = RonCore(provider=FakeProvider())
    core.learn_from_experience(
        Experience("task", "success", "prefer verified answers")
    )
    response = core.respond("hello")
    assert response.metadata["active_skills"] == []
    assert len(core.self_improvement.experiences) == 1


def test_core_passes_recent_dialogue_turns_to_the_provider():
    class CapturingProvider:
        def __init__(self):
            self.last_request = None

        def generate(self, request: ModelRequest) -> ModelResponse:
            self.last_request = request
            return ModelResponse(content="رد تجريبي", model="capture", metadata={})

    provider = CapturingProvider()
    core = RonCore(provider=provider)
    core.respond("سؤال أول")
    core.respond("متابعة")
    assert provider.last_request is not None
    assert any(m.role == "user" and m.content == "سؤال أول" for m in provider.last_request.messages)
    assert any(m.role == "assistant" and m.content == "رد تجريبي" for m in provider.last_request.messages)
    assert provider.last_request.messages[-1].content == "متابعة"



def test_meta_question_does_not_overwrite_profile_name():
    core = RonCore(provider=FakeProvider())
    core.respond("انا بسألك عن اسمي")
    assert core.respond("ما اسمي؟").content == "لم تخبرني باسمك بعد."

    correction = core.respond("اسمي اركانوس وليس اسمك بسألك عن اسمي")
    assert "اركانوس" in correction.content
    assert core.respond("ما اسمي؟").content == "اسمك اركانوس."

def test_why_followup_reuses_previous_topic_and_dialogue():
    class CapturingProvider:
        def __init__(self):
            self.requests = []

        def generate(self, request: ModelRequest) -> ModelResponse:
            self.requests.append(request)
            return ModelResponse(content="رد تجريبي", model="capture", metadata={})

    provider = CapturingProvider()
    core = RonCore(provider=provider)
    core.respond("ما عاصمة مصر؟")
    core.respond("ليه؟")
    request = provider.requests[-1]
    assert request.metadata["intent"] == "follow_up"
    assert request.metadata["topic"] == "ما عاصمة مصر؟"
    assert any(message.content == "ما عاصمة مصر؟" for message in request.messages)
    assert any(message.content == "رد تجريبي" for message in request.messages)
