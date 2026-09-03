import train_vcg_v11_conditioned_handling_seed1_convergence_continuation as continuation


def _record(stable):
    return {"stability_against_previous_round": {"stable": stable}}


def test_stopping_rule_requires_two_consecutive_stable_transitions():
    assert continuation.stopping_assessment([])["stop"] is False
    assert continuation.stopping_assessment([_record(True)])["stop"] is False
    passed = continuation.stopping_assessment([_record(True), _record(True)])
    assert passed["stop"] is True
    assert passed["passed"] is True
    assert passed["completed_additional_rounds"] == 2


def test_stopping_rule_continues_after_an_interruption_in_stability():
    records = [_record(True), _record(False), _record(True)]
    assert continuation.stable_streak(records) == 1
    assert continuation.stopping_assessment(records)["stop"] is False
    capped = continuation.stopping_assessment(records + [_record(False)])
    assert capped["stop"] is True
    assert capped["passed"] is False
    assert capped["decision"] == "do_not_evaluate_continuation_cap_reached"


def test_schedules_extend_parent_stream_without_reuse():
    first = continuation.continuation_schedule(0)
    second = continuation.continuation_schedule(1)
    assert len(first) == continuation.EPISODES_PER_ROUND
    assert first == continuation.continuation_schedule(0)
    assert first[0]["global_round_number"] == continuation.PARENT_GLOBAL_ROUNDS + 1
    assert second[0]["global_round_number"] == continuation.PARENT_GLOBAL_ROUNDS + 2
    first_seeds = {
        continuation.continuation_instance_seed(0, index)
        for index in range(continuation.EPISODES_PER_ROUND)
    }
    second_seeds = {
        continuation.continuation_instance_seed(1, index)
        for index in range(continuation.EPISODES_PER_ROUND)
    }
    assert first_seeds.isdisjoint(second_seeds)
