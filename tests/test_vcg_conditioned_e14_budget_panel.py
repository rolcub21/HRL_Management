from experiments.conditioned_vcg.E14_certification_scalability_95k import (
    budget_panel,
)


def test_e14_budget_panel_is_the_frozen_main_grid_only():
    assert budget_panel.SCENARIOS == budget_panel.e13.MAIN_SCENARIOS
    assert len(budget_panel.SCENARIOS) == 9
    assert budget_panel.BUDGETS == (2, 4, 8, 16, 20_000)
    assert budget_panel.EXECUTED_BUDGETS == (2, 4, 8, 16)


def test_e14_contract_forbids_cross_budget_anchor_reuse():
    contract = budget_panel.expected_contract()
    assert contract["budget_arm_anchor_rule"] == (
        "fresh_per_episode_cache_and_anchor_acquired_under_arm_budget"
    )
    assert contract["cross_budget_positive_proof_reuse"] is False
    assert contract["reference_compute_rerun"] is False


def test_e14_budget_summary_counts_family_and_native_work_separately():
    summary = budget_panel._budget_summary(4, [])
    assert summary["family"]["attempts"] == 0
    assert summary["family"]["proofs"] == 0
    assert summary["native_searches"]["count"] == 0
    assert summary["strict_completion_rate"] is None
