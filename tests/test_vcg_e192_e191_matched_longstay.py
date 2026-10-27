from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.conditioned_vcg.E191_fresh_long_stay_crowding_250k_v1 import (
    protocol as e191,
    scenario as workload,
)
from experiments.conditioned_vcg.E192_e191_matched_longstay_heuristics_251k_v1 import (
    protocol as p,
    run,
)


@pytest.fixture(autouse=True)
def isolated_e191_summary(
    tmp_path_factory: pytest.TempPathFactory,
    monkeypatch: pytest.MonkeyPatch,
):
    """Keep E192 tests independent of ignored, machine-local E191 results."""

    instances = {
        seed: run.make_instance(seed)
        for seed in p.INSTANCE_SEEDS
    }
    summary = {
        "training_protocol": e191.PROTOCOL,
        "target_contract_sha256": e191.contract()["contract_sha256"],
        "scenario": e191.SCENARIO,
        "model_seed": p.MODEL_SEED,
        "validation_instance_seeds": list(p.INSTANCE_SEEDS),
        "validation_history": [{
            "checkpoint_episode": 20,
            "selection_score": [1.0, 1.0, -1.0, -1.0, 0.0, -20.0],
            "runs": [{
                "protocol": e191.PROTOCOL,
                "split": "development_validation",
                "scheduled_epsilon_after": 0.0,
                "episode_instance_id": instances[seed].instance_id,
                "instance_seed": seed,
                "strict_method_success": True,
                "steps": 1,
                "e187_metrics": {
                    "strict_complete": True,
                    "required_jobs": p.SCENARIO_SPEC.total_jobs,
                    "delivered_jobs": p.SCENARIO_SPEC.total_jobs,
                    "components": {},
                },
            } for seed in p.INSTANCE_SEEDS],
        }],
    }
    summary_path = (
        tmp_path_factory.mktemp("e191-summary") / "training-summary.json"
    )
    summary_path.write_text(
        json.dumps(summary, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(p, "E191_SUMMARY", summary_path)


def test_contract_matches_e191_longstay_problem_exactly():
    value = p.contract()
    source = e191.contract()

    assert p.INSTANCE_SEEDS == tuple(e191.VALIDATION_SEEDS[e191.SCENARIO])
    assert len(p.INSTANCE_SEEDS) == 9
    assert p.METHODS == ("dynamic_pslap", "ga")
    assert p.GA_REPLICATES == (0, 1, 2, 3)
    assert p.MAX_STEPS == 9000
    assert value["scenario"] == source["scenario"]
    assert value["scenario_public"] == source["scenario_public"]
    assert value["objective"] == source["objective"]
    assert value["finite_horizon"] == source["finite_horizon"]
    assert value["expected_episode_rows"] == 45
    assert value["certified_dynamic_included"] is False
    assert value["test_panel_opened"] is False


def test_prepare_freezes_balanced_paired_e191_instances(tmp_path: Path):
    output = tmp_path / "e192"
    first = run.prepare(output)
    second = run.prepare(output)

    assert first == second
    assert first["workload_instances"] == {"20%": 3, "50%": 3, "80%": 3}
    manifest = json.loads((output / "instances.json").read_text(encoding="utf-8"))
    assert tuple(row["seed"] for row in manifest["records"]) == p.INSTANCE_SEEDS
    assert len({row["instance_id"] for row in manifest["records"]}) == 9

    instances = {
        int(row["seed"]): run.frozen_instance(manifest, int(row["seed"]))
        for row in manifest["records"]
    }
    for replicate in range(workload.VALIDATION_CASES_PER_LEVEL):
        selected = [
            instances[workload.VALIDATION_BASE + level_index * 100 + replicate]
            for level_index in range(len(workload.LEVELS))
        ]
        assert len({item.storage_steps_needed for item in selected}) == 1
        assert len({item.arrival_steps for item in selected}) == 3


def test_coordinates_are_45_unique_resumable_rows(tmp_path: Path):
    paths = []
    for method in p.METHODS:
        for seed in p.INSTANCE_SEEDS:
            for replicate in run.replicates(method):
                paths.append(run.row_path(tmp_path, method, seed, replicate))
    assert len(paths) == len(set(paths)) == 45
    assert sum("dynamic_pslap" in str(path) for path in paths) == 9
    assert sum("/ga/" in str(path) for path in paths) == 36


def test_traced_environment_binds_instance_specific_rate_and_horizon():
    instance = run.make_instance(250_191_000)
    env = run.TracedLongStayEnv(p.SCENARIO_SPEC)
    assert env.arrival_rate != instance.arrival_rate
    env.reset(instance=instance)
    assert env.arrival_rate == instance.arrival_rate == pytest.approx(0.01)
    assert env.proc_mean == instance.proc_mean == 700
    assert env.e192_scenario.max_steps == 9000
    assert env.warm_start_audit["passed"] is True
    assert env.warm_start_audit["target_label"] == "20%"


def test_storage_clock_and_full_wait_discriminator():
    block = type("Block", (), {
        "arrival_step": 0,
        "stored_time_step": 30,
        "storage_steps_needed": 700,
        "delivered_time_step": 730,
        "delivered": True,
    })()
    scored = run.score_completed_blocks((block,), equipment_work=4, rehandles=1)
    assert scored["J"] == pytest.approx(25.4)
    assert scored["storage_clock_mae"] == 0.0
    assert scored["release_clock_mae"] == 30.0
    assert scored["components"]["admission_wait_accrual"] == 30.0


def test_validation_curve_uses_nine_greedy_matched_days():
    checkpoints = run.validation_checkpoints()
    assert checkpoints
    assert checkpoints[0]["checkpoint_episode"] == 20
    for checkpoint in checkpoints:
        points = checkpoint["points"]
        assert len(points) == 9
        assert {point["target_label"] for point in points} == {"20%", "50%", "80%"}
        assert all(point["strict_complete"] for point in points)


def test_self_check_is_non_episodic_and_idempotent(tmp_path: Path):
    first = run.self_check(tmp_path)
    second = run.self_check(tmp_path)
    assert first == second
    assert first["status"] == "self_check_passed"
    assert first["episode_runs"] == 0
    assert first["expected_rows"] == 45

