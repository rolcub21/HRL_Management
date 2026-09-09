from experiments.conditioned_vcg.E14_certification_scalability_95k import (
    threshold_refinement as threshold,
)


def test_threshold_refinement_is_explicitly_post_hoc_and_bounded():
    contract = threshold.expected_contract()
    assert contract["post_hoc_declared_after_parent_E14_outcomes"] is True
    assert contract["refinement_budgets_max_nodes"] == [32, 64]
    assert contract["expected_new_rows"] == 18
    assert contract["training"] is False


def test_reference_trace_supports_conservative_128_node_cap():
    contract = threshold.expected_contract()
    assert contract["reference_trace_maximum_explored_nodes"] == 126
    assert contract["conservative_reference_equivalent_cap"] == 128


def test_matched_initial_summary_does_not_mix_trajectory_lengths():
    report = threshold.analyze(threshold.DEFAULT_OUTPUT, allow_partial=True)
    matched = {item["max_nodes"]: item for item in report["matched_initial_frontiers"]}
    assert matched[2]["matched_initial_states"] == 9
    assert matched[16]["current_state_safe"] == 4
    assert matched[20_000]["current_state_safe"] == 9
