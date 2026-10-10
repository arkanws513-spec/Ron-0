from ron.core import RonCore
from ron.knowledge import CuratedKnowledgeBase, normalize_question


def test_curated_knowledge_retrieves_reviewed_exact_prompts():
    knowledge = CuratedKnowledgeBase()
    assert "القاهرة" in knowledge.answer("ما عاصمة مصر؟")
    assert "رون" in knowledge.answer("ما اسمك؟")
    assert "التدريب" in knowledge.answer("ما الفرق بين الذاكرة والتدريب؟")
    assert normalize_question("ما عاصمة مصر؟") == normalize_question("ما عاصمة مصر")


def test_curated_knowledge_does_not_guess_for_unseen_prompts():
    knowledge = CuratedKnowledgeBase()
    assert knowledge.answer("ما عاصمة دولة غير موجودة في البيانات؟") is None


def test_core_uses_curated_knowledge_before_weak_character_generation():
    class MustNotRunProvider:
        def generate(self, request):
            raise AssertionError("exact reviewed prompt should be answered by local retrieval")

    core = RonCore(provider=MustNotRunProvider())
    response = core.respond("ما عاصمة مصر؟")
    assert "القاهرة" in response.content
    assert response.model == "ron-curated-knowledge"
    assert response.metadata["local_retrieval"] is True
    assert response.metadata["external_model_used"] is False
