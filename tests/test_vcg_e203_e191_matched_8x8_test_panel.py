from __future__ import annotations

from experiments.conditioned_vcg.E191_fresh_long_stay_crowding_250k_v1 import (
    protocol as e191,
    scenario as e191_scenario,
)
from experiments.conditioned_vcg.E192_e191_matched_longstay_heuristics_251k_v1 import (
    run as e192,
)
from experiments.conditioned_vcg.E203_e191_matched_8x8_test_panel_262k_v1 import (
    plot,
    protocol,
    run,
    scenario,
)


def test_contract_locks_new_test_panel_before_evaluation():
    value = protocol.contract()
    assert value["scope"] == "sealed matched 8x8 paper test panel"
    assert value["test_instances"] == 30
    assert value["cases_per_workload"] == 10
    assert value["expected_episode_rows"] == 240
    assert value["model_seeds"] == [0, 1, 2]
    assert value["checkpoint_selection_uses_test_panel"] is False
    assert value["test_panel_design_locked"] is True
    assert value["test_panel_opened_at_evaluation"] is True
    assert value["certified_dynamic_included"] is False


def test_test_instances_are_deterministic_paired_and_disjoint():
    prior = set(e191.VALIDATION_SEEDS[scenario.SCENARIO])
    prior.update(e191_scenario.PILOT_SEEDS)
    prior.update(
        e191_scenario.TRAIN_BASE + index for index in range(e191.TOTAL_EPISODES)
    )
    assert not prior & set(scenario.TEST_SEEDS)
    streams = {}
    counts = {level: 0 for level in scenario.LEVELS}
    for seed in scenario.TEST_SEEDS:
        first = run.make_instance(seed)
        second = run.make_instance(seed)
        assert first == second
        level = scenario.level_for_seed(seed)
        counts[level] += 1
        assert seed % 3 == scenario.LEVELS.index(level)
        assert first.arrival_steps == tuple(
            scenario.GAPS[level] * index for index in range(scenario.TOTAL_JOBS)
        )
        replicate = scenario.replicate_for_seed(seed)
        if replicate in streams:
            assert streams[replicate] == first.storage_steps_needed
        else:
            streams[replicate] = first.storage_steps_needed
    assert counts == {level: 10 for level in scenario.LEVELS}


def test_self_check_runs_no_test_episode_and_authenticates_freezes(tmp_path):
    result = run.self_check(tmp_path / "e203")
    assert result["status"] == "self_check_passed"
    assert result["episode_runs"] == 0
    assert result["test_panel_opened"] is False
    assert result["duration_stream_pairing"]
    assert result["training_development_seed_disjoint"]
    assert {
        seed: value["selected_episode"]
        for seed, value in result["frozen_models"].items()
    } == {"0": 100, "1": 60, "2": 80}


def test_coordinate_plan_contains_every_declared_episode_once():
    coordinates = run._coordinates("run-all", None, None)
    assert len(coordinates) == protocol.EXPECTED_ROWS == 240
    assert len(set(coordinates)) == len(coordinates)
    assert sum(row[0] == "dynamic_pslap" for row in coordinates) == 30
    assert sum(row[0] == "ga" for row in coordinates) == 120
    assert sum(row[0] == "e191" for row in coordinates) == 90


def test_baseline_factory_uses_the_e191_8x8_environment():
    expected = e191.scenario_for(scenario.SCENARIO)
    environment = run._environment_factory(
        expected,
        arrival_rate=1.0 / scenario.GAPS[0.5],
        proc_mean=scenario.MEAN_DURATION,
        grid_rows=8,
        grid_cols=8,
        exit_width=3,
        number_blocks=scenario.TOTAL_JOBS,
    )
    assert isinstance(environment, e192.TracedLongStayEnv)
    assert len(environment.storage_positions) == scenario.CAPACITY == 35


def test_plot_workload_values_pool_only_the_three_frozen_seeds():
    report = {
        "baseline_by_workload": {
            method: {
                level: {"metrics": {"J": base + index}}
                for index, level in enumerate(plot.WORKLOADS)
            }
            for method, base in (("dynamic_pslap", 10.0), ("ga", 20.0))
        },
        "e191_selected_by_workload": {
            seed: {
                level: {"metrics": {"J": 30.0 + 10.0 * seed_index + level_index}}
                for level_index, level in enumerate(plot.WORKLOADS)
            }
            for seed_index, seed in enumerate(plot.MODEL_SEEDS)
        },
    }
    values = plot.workload_values(report, "J")
    assert values["dynamic_pslap"] == [10.0, 11.0, 12.0]
    assert values["ga"] == [20.0, 21.0, 22.0]
    assert values["vcg"] == [40.0, 41.0, 42.0]
    assert len(values["vcg_seeds"]) == 3
