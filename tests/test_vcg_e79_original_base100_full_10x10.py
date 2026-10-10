"""No-episode contract and analysis checks for E79."""

from __future__ import annotations

import json
from unittest.mock import patch

from experiments.conditioned_vcg.E79_original_vcg_base100_full_10x10_144k_v1 import (
    protocol as p,
    run,
)
from experiments.conditioned_vcg.target_size_dense_stage_a_v4 import run as v4run
import train_vcg_dense_proper as dense


def test_e79_frozen_budget_reward_and_workload():
    assert p.SCENARIO == "empty_10x10_n64"
    assert p.MODEL_SEEDS == (0, 1, 2)
    assert p.EPISODES_BY_SCENARIO[p.SCENARIO] == 63
    assert p.SCENARIOS[p.SCENARIO].total_jobs == 64
    assert p.SPEC.to_dict()["dense_b"] == 100.0
    assert p.DELIVERY_BASE == 80.0 * (10.0 / 8.0)
    assert p.LATE_CHECKPOINTS == (36, 42, 48, 54, 60, 63)


def test_e79_prepare_self_check_and_partial_analysis(tmp_path):
    prepared = run.prepare(tmp_path / "e79")
    assert prepared["status"] == "prepared"
    assert run.self_check(tmp_path / "e79")["status"] == "passed"
    result = run.analyze(tmp_path / "e79")
    assert result["status"] == "partial"
    assert result["completed_training_episodes"] == {"0": 0, "1": 0, "2": 0}
    report = json.loads((tmp_path / "e79/e79-report.json").read_text())
    assert report["aggregate"] is None
    assert report["late_window_checkpoint_set"] == list(p.LATE_CHECKPOINTS)


def test_e79_late_window_refuses_missing_or_incomplete_checkpoints():
    records = [{"checkpoint_episode": step, "strict_complete_rows": 5,
                "validation_rows": 5} for step in p.LATE_CHECKPOINTS[:-1]]
    assert not run._late_window(records)["checkpoints_present"]
    records.append({"checkpoint_episode": 63, "strict_complete_rows": 4,
                    "validation_rows": 5})
    late = run._late_window(records)
    assert late["checkpoints_present"]
    assert not late["all_strict_complete"]
    assert "mae_mean" not in late


def test_e79_training_wiring_without_episode_execution(tmp_path):
    output = tmp_path / "e79"
    run.prepare(output)

    def fake_train(args):
        assert args.output == output.resolve()
        assert args.scenario == p.SCENARIO
        assert args.model_seed == 2
        assert args.device == "cuda"
        assert args.stop_after_episode == 2
        assert v4run.p is p
        assert dense.FROZEN_OBJECTIVE_SPEC is p.SPEC
        return {"status": "paused", "completed_training_episodes": 2}

    with patch.object(run.torch.cuda, "is_available", return_value=True), \
         patch.object(v4run, "train", side_effect=fake_train):
        result = run.train_seed(output, 2, through_episode=2)
    assert result["status"] == "paused"
    assert result["e79_contract_sha256"] == p.contract()["contract_sha256"]
