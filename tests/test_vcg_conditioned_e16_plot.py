from experiments.conditioned_vcg.E16_prediction_ranking_audit import render_results as plot


def test_e16_figure_data_binds_complete_reports() -> None:
    e16b, e16c, e16d = plot._load_inputs()
    data = plot._figure_data(e16b, e16c, e16d)

    crossing = data["crossing_reproduction"]
    assert crossing["cases"] == 10
    assert crossing["action_type_only_exact"] == 2
    assert crossing["candidate_residual_only_exact"] == 8
    assert crossing["within_action_type_cases"] == 6
    assert crossing["within_action_type_action_type_only_exact"] == 0
    assert crossing["within_action_type_candidate_residual_only_exact"] == 6

    rollout = data["rollout_ablation"]
    assert rollout["matched_coordinates_per_arm"] == 90
    assert rollout["total_rows"] == 360
    assert rollout["strict_safe_complete_rows"] == 360
    assert set(rollout["points"]) == set(plot.ARMS)
    action_effect = rollout["points"]["action_type_only"]["paired_effects"]
    assert action_effect["rehandles_per_100"]["n"] == 90
    assert action_effect["mean_absolute_error"]["n"] == 90
