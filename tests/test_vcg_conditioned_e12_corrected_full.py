from experiments.conditioned_vcg.E12_representation_ablation_94k import (
    confirm_corrected_executor_full as corrected,
)


def test_corrected_confirmation_is_full_separate_and_training_free():
    assert corrected.DEFAULT_OUTPUT != corrected.PARENT_E12
    assert corrected.EXPECTED_ROWS == 7_560
    assert len(corrected._specs()) == 36


def test_corrected_contract_binds_original_panel_and_executor_fix():
    contract = corrected.expected_contract()
    assert contract["original_E12_results_remain_immutable"] is True
    assert contract["training_runs"] == 0
    assert contract["parents"]["historical_e12_ledger_set"] == {
        "count": 7_560,
        "strict_safe_complete": 7_557,
        "sha256": contract["parents"]["historical_e12_ledger_set"]["sha256"],
    }
    assert contract["acceptance"]["all_three_historical_failures_repaired"] is True
    for path in (
        "example/Options/DirectDeliverOption.py",
        "example/Options/ReconfigureOption.py",
        "PSLAP/viability_candidates.py",
    ):
        assert path in contract["source_sha256"]
