from experiments.conditioned_vcg.E13_operational_scalability_95k import program


def test_e13_panel_has_unique_frozen_coordinates():
    assert len(program.MAIN_SCENARIOS) == 9
    assert len(program.COMPANION_SCENARIOS) == 6
    assert len(program.SCENARIOS) == len(set(program.SCENARIOS)) == 15
    assert program.INSTANCE_SEEDS == (95_100, 95_101, 95_102)
    assert set(program.SCENARIOS) <= set(program.occupancy.SCENARIO_BY_ID)


def test_e13_parent_manifest_supplies_all_45_frozen_instances():
    inputs = program._parent_inputs()
    coordinates = {
        (item["scenario_id"], int(item["seed"]))
        for item in inputs["records"]
    }
    expected = {
        (scenario, seed)
        for scenario in program.SCENARIOS
        for seed in program.INSTANCE_SEEDS
    }
    assert coordinates == expected
    assert len(inputs["records"]) == 45


def test_e13_summary_never_promotes_partial_quality_to_main_metric():
    scenario = "size_5x5_occ_low"
    row = {
        "strict_safe_complete": True,
        "failure_class": "completed",
        "dense_return": 1.0,
        "mean_absolute_error": 2.0,
        "physical_rehandles_per_100_required_deliveries": 3.0,
        "steps_per_delivery": 4.0,
        "within_target_window_rate": 0.5,
        "frontiers": [],
        "decision_costs": [],
        "occupancy": {
            "initial_storage_occupancy_ratio": 0.25,
            "time_weighted_mean_storage_occupancy_ratio": 0.2,
            "peak_storage_occupancy_ratio": 0.25,
            "time_weighted_mean_admitted_workload": 1.0,
            "peak_concurrent_admitted_workload": 2,
            "peak_arrived_unadmitted": 1,
        },
        "episode_wall_seconds": 1.0,
        "process_peak_rss_after_kib": 1,
        "rss_after_bytes": 1,
    }
    summary = program._scenario_summary(scenario, [row])
    assert summary["complete_case_metrics"]["n"] == 1
    assert summary["all_required_rows_metrics"] is None
    assert not summary["complete_coordinate"]
