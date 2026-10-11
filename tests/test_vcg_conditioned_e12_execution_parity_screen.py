from experiments.conditioned_vcg.E12_representation_ablation_94k import (
    screen_corrected_executor_parity as screen,
)


def test_parity_screen_contract_covers_all_specs_and_known_failures():
    contract = screen._contract(screen.DEFAULT_E12_OUTPUT)

    assert contract["expected_runs"] == 108
    assert contract["expected_historical_successes"] == 105
    assert contract["expected_historical_failures"] == 3
    assert len({case["case_id"] for case in contract["cases"]}) == 108
    failures = {
        screen._case_key(case)
        for case in contract["cases"]
        if not case["historical_strict_safe_complete"]
    }
    assert failures == screen.EXPECTED_HISTORICAL_FAILURES


def test_parity_screen_comparison_separates_completion_behavior_and_metrics():
    historical = {
        "strict_safe_complete": True,
        "behavior_digest": "old",
        **{field: 1 for field in screen.COMPARISON_FIELDS},
    }
    corrected = dict(historical, behavior_digest="new", steps=2)

    result = screen._comparison(historical, corrected)

    assert result["historical_success_retained"] is True
    assert result["behavior_digest_equal"] is False
    assert result["all_comparison_fields_equal"] is False
    assert result["changed_comparison_fields"] == ["steps"]
