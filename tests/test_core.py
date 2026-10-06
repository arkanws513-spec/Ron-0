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
