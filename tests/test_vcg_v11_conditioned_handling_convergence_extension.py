import train_vcg_v11_conditioned_handling_convergence_extension as extension


def _probe_rows(*, changed=(), rehandles=0):
    rows = []
    for index, value in enumerate(extension.original.PROBE_LAMBDAS):
        rows.append(
            {
                "behavior_lambda": value,
                "behavior_digest": f"digest-{index}-{'changed' if index in changed else 'base'}",
                "physical_rehandles": rehandles if index == 0 else 0,
                "return": 100.0 + index,
                "mean_absolute_error": 10.0,
                "strict_method_success": True,
            }
        )
    return rows


def _fit(mae=0.2, bias=0.01):
    return {"final_validation": {"mae": mae, "bias": bias}}


def test_extension_schedule_is_reproducible_stratified_and_disjoint():
    first = extension.extension_lambda_schedule(0)
    second = extension.extension_lambda_schedule(0)
    assert first == second
    assert len(first) == extension.EPISODES_PER_ROUND
    values = [row["behavior_lambda"] for row in first]
    assert all(0.0 < value < 0.2 for value in values)
    assert sorted(
        int(value / 0.2 * extension.EPISODES_PER_ROUND) for value in values
    ) == list(range(extension.EPISODES_PER_ROUND))
    assert first[0]["global_round_number"] == 5
    assert extension.extension_lambda_schedule(1)[0]["global_round_number"] == 6


def test_stability_accepts_unchanged_safe_calibrated_transition():
    previous = _probe_rows(rehandles=2)
    current = _probe_rows(rehandles=2)
    result = extension._stability(previous, current, _fit())
    assert result["stable"]
    assert result["changed_probe_behavior_count"] == 0
    assert all(result["criteria"].values())


def test_stability_rejects_two_policy_changes_even_with_same_rehandles():
    previous = _probe_rows(rehandles=2)
    current = _probe_rows(changed=(1, 3), rehandles=2)
    result = extension._stability(previous, current, _fit())
    assert not result["stable"]
    assert result["changed_probe_behavior_count"] == 2
    assert not result["criteria"]["changed_probe_behaviors_at_most_one"]


def test_stability_rejects_rehandle_or_calibration_drift():
    previous = _probe_rows(rehandles=2)
    current = _probe_rows(rehandles=3)
    result = extension._stability(previous, current, _fit(mae=0.4, bias=0.2))
    assert not result["stable"]
    assert not result["criteria"]["aggregate_probe_rehandles_unchanged"]
    assert not result["criteria"]["validation_mae_within_bound"]
    assert not result["criteria"]["absolute_validation_bias_within_bound"]
