from experiments.conditioned_vcg.development.D10_scalability_support_screen import (
    occupancy_extension as extension,
)


def test_main_grid_uses_actual_storage_capacity_and_achievable_ratios():
    assert len(extension.SCENARIO_BY_ID) == len(extension.SCENARIOS) == 15
    expected = {
        5: (8, (2, 4, 6)),
        8: (35, (11, 18, 25)),
        10: (63, (19, 32, 44)),
    }
    for size, (capacity, occupied) in expected.items():
        rows = [
            scenario
            for scenario in extension.SCENARIOS
            if scenario.comparison == "size_by_occupancy" and scenario.rows == size
        ]
        assert len(rows) == 3
        assert all(scenario.storage_capacity == capacity for scenario in rows)
        assert tuple(scenario.initial_occupied_slots for scenario in rows) == occupied
        assert all(
            abs(scenario.initial_occupied_slots / capacity - target) <= 0.051
            for scenario, target in zip(rows, (0.30, 0.50, 0.70))
        )


def test_warm_start_realizes_declared_occupancy_and_tracks_every_step():
    for scenario in extension.SCENARIOS:
        instance = extension._instance(scenario, extension.INSTANCE_SEEDS[0])
        env = extension.OccupancyTrackingEnv(scenario)
        instance.validate_for(env)
        env.reset(instance=instance)
        initial = env.occupancy_summary()
        assert initial["usable_storage_capacity"] == scenario.storage_capacity
        assert initial["initial_occupied_storage_slots"] == scenario.initial_occupied_slots
        assert initial["peak_concurrent_admitted_workload"] == scenario.initial_occupied_slots
        env.step(env.ACTION_IDS["WAIT"])
        assert len(env.occupancy_trace) == 2
        assert env.occupancy_trace[-1]["time_step"] == 1


def test_matched_geometry_comparisons_share_workload_schedule():
    ids = (
        "size_5x5_occ_high",
        "fixed_workload_8x8_k6_n8",
        "fixed_workload_10x10_k6_n8",
    )
    instances = [
        extension._instance(extension.SCENARIO_BY_ID[scenario_id], 95_100)
        for scenario_id in ids
    ]
    assert len({instance.schedule_id for instance in instances}) == 1

    rectangles = [
        extension._instance(extension.SCENARIO_BY_ID[scenario_id], 95_100)
        for scenario_id in ("rectangle_6x10_medium", "rectangle_10x6_medium")
    ]
    assert len({instance.schedule_id for instance in rectangles}) == 1
    assert {
        extension.SCENARIO_BY_ID[scenario_id].storage_capacity
        for scenario_id in ("rectangle_6x10_medium", "rectangle_10x6_medium")
    } == {31}


def test_instances_are_deterministic_and_total_jobs_are_separate():
    for scenario in extension.SCENARIOS:
        left = extension._instance(scenario, 95_101)
        right = extension._instance(scenario, 95_101)
        assert left == right
        assert left.number_blocks == scenario.total_jobs
        assert scenario.public_dict()["turnover_jobs"] == (
            scenario.total_jobs - scenario.initial_occupied_slots
        )
