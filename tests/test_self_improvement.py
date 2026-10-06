from ron.self_improvement import (
    Evaluation,
    Experience,
    ImprovementCandidate,
    SelfImprovementEngine,
    Skill,
)


def candidate_for(experience: Experience) -> ImprovementCandidate:
    return ImprovementCandidate(
        skill=Skill(
            name="clear_answers",
            instruction=f"Use the verified lesson: {experience.lesson}",
            version=2,
        ),
        reason="Repeated successful experience",
    )


def test_learning_is_bounded():
    engine = SelfImprovementEngine(max_experiences=2)
    for index in range(3):
        engine.learn(Experience(str(index), "ok", "lesson"))
    assert [item.task for item in engine.experiences] == ["1", "2"]


def test_only_verified_improvement_is_promoted():
    engine = SelfImprovementEngine()
    experience = Experience("task", "good", "be concise")
    engine.learn(experience)
    candidate = engine.propose(experience, "clear_answers", candidate_for)

    rejected = engine.verify_and_promote(candidate, 0.8, 0.7, True)
    assert not rejected.improved
    assert engine.registry.get("clear_answers") is None

    accepted = engine.verify_and_promote(candidate, 0.8, 0.9, True)
    assert accepted.improved
    assert engine.registry.get("clear_answers") == candidate.skill
    assert engine.audit_log[-1].status == "promoted"


def test_failed_verification_does_not_change_active_skill():
    engine = SelfImprovementEngine()
    first = ImprovementCandidate(Skill("planner", "version one", version=1), "initial")
    second = ImprovementCandidate(Skill("planner", "version two", version=2), "experiment")
    engine.verify_and_promote(first, 0.0, 1.0, True)
    engine.verify_and_promote(second, 1.0, 2.0, False)

    assert engine.registry.get("planner") == first.skill
    assert engine.audit_log[-1].status == "rejected"


def test_rollback_restores_previous_skill():
    engine = SelfImprovementEngine()
    v1 = ImprovementCandidate(Skill("planner", "v1", 1), "first")
    v2 = ImprovementCandidate(Skill("planner", "v2", 2), "second")
    engine.verify_and_promote(v1, 0.0, 1.0, True)
    engine.verify_and_promote(v2, 1.0, 2.0, True)

    assert engine.rollback("planner") == v1.skill


def test_improve_runs_a_complete_verified_cycle():
    engine = SelfImprovementEngine()
    experience = Experience("task", "success", "use concise answers")
    result = engine.improve(
        experience,
        "clear_answers",
        candidate_for,
        lambda candidate: Evaluation(0.7, 0.9, True),
    )

    assert result.evaluation.improved
    assert engine.registry.get("clear_answers") == result.candidate.skill
    assert engine.experiences[-1] == experience
