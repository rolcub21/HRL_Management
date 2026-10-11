from experiments.conditioned_vcg.E03_certification_ablation_90k import (
    scale_stress,
)


def test_scale_stress_coordinates_are_predeclared_and_saturating():
    assert len(scale_stress.SCENARIOS) == 2
    assert len(scale_stress.INSTANCE_SEEDS) == 10
    for scenario in scale_stress.SCENARIOS:
        assert scenario.total_jobs == scenario.storage_capacity
        assert 0.69 < (
            scenario.initial_occupied_slots / scenario.storage_capacity
        ) < 0.72
        assert scenario.make_base_env().start_state not in scenario.placement_cells


def test_scale_stress_instance_is_a_deterministic_synchronized_burst():
    scenario = scale_stress.SCENARIOS[0]
    first = scale_stress._instance(scenario, scale_stress.INSTANCE_SEEDS[0])
    second = scale_stress._instance(scenario, scale_stress.INSTANCE_SEEDS[0])
    assert first == second
    assert len(first.arrival_steps) == scenario.total_jobs
    assert set(first.arrival_steps) == {0}
    assert min(first.storage_steps_needed) >= 1


def test_paired_summary_keeps_certified_and_physical_directions_distinct():
    base = {
        "scenario_id": scale_stress.SCENARIO_IDS[0],
        "instance_seed": scale_stress.INSTANCE_SEEDS[0],
        "strict_completion": True,
        "dense_return": 1.0,
        "mean_absolute_error": 2.0,
        "steps": 3,
        "physical_rehandles_per_100_required_deliveries": 4.0,
    }
    rows = [
        {**base, "candidate_source": scale_stress.CERTIFIED},
        {
            **replace_dummy(base, strict_completion=False),
            "candidate_source": scale_stress.PHYSICAL,
        },
    ]
    summary = scale_stress._paired_summary(rows)
    assert summary["certified_only_complete"] == 1
    assert summary.get("physical_only_complete", 0) == 0


def replace_dummy(mapping, **updates):
    result = dict(mapping)
    result.update(updates)
    return result
