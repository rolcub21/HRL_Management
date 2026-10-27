from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from experiments.conditioned_vcg.E194_e191_120_development_freeze_253k_v1 import (
    run as e194,
)


def _record(episode: int, objective: float) -> dict:
    score = [1.0, 1.0, -objective / 36.0, -objective / 100.0, -50.0, -episode]
    workload = {
        level: {"means": {"J": objective}}
        for level in ("20%", "50%", "80%")
    }
    return {
        "checkpoint_episode": episode,
        "selection_score_fields": [
            "strict_success_rate",
            "completion_rate",
            "selected_objective_return_per_block",
            "negative_mean_absolute_error",
            "negative_relocations_per_100_deliveries",
            "negative_checkpoint_episode",
        ],
        "selection_score": score,
        "summary": {
            "mean_e187_J": objective,
            "mean_storage_clock_mae": objective / 100.0,
            "mean_admission_wait": objective / 200.0,
            "relocations_per_100_deliveries": 50.0,
            "mean_peak_stored": 14.0,
            "strict_method_success_rate": 1.0,
            "e191_by_workload": workload,
        },
    }


def _records(j100: float, j120: float) -> list[dict]:
    values = {20: 20000.0, 40: 19000.0, 60: 18000.0, 80: 17000.0,
              100: j100, 120: j120}
    return [_record(episode, values[episode]) for episode in range(20, 121, 20)]


def test_selects_episode_120_when_its_authoritative_score_is_better():
    selected = e194.select_record(_records(j100=4000.0, j120=3000.0))
    assert selected["checkpoint_episode"] == 120


@pytest.mark.parametrize("j120", [5000.0, 4000.0])
def test_worse_or_tied_episode_120_keeps_episode_100(j120: float):
    selected = e194.select_record(_records(j100=4000.0, j120=j120))
    assert selected["checkpoint_episode"] == 100


def test_selector_refuses_when_an_earlier_checkpoint_is_authoritative():
    records = _records(j100=4000.0, j120=3000.0)
    records[3] = _record(80, 2000.0)
    with pytest.raises(e194.FreezeError, match="earlier checkpoint outranks"):
        e194.select_record(records)


def _source_tree(tmp_path: Path, monkeypatch, *, j100=4000.0, j120=3000.0):
    contract_sha = "a" * 64
    monkeypatch.setattr(
        e194.e191_run,
        "authenticate",
        lambda _output: {"contract_sha256": contract_sha},
    )
    source = tmp_path / "source"
    seed_directory = source / e194.SCENARIO_RELATIVE
    seed_directory.mkdir(parents=True)
    records = _records(j100=j100, j120=j120)
    selected = e194.select_record(records)
    candidate = seed_directory / "best-candidate-slot-a.pth"
    torch.save(
        {
            "training_protocol": e194.e191_protocol.PROTOCOL,
            "method_version": e194.e191_protocol.METHOD_VERSION,
            "checkpoint_role": "best_candidate_unfinalized",
            "model_seed": 0,
            "completed_training_episodes": selected["checkpoint_episode"],
            "trainer_resumable": False,
            "protocol_training_complete": False,
            "selection_finalized_after_total_episodes": False,
            "deployment_checkpoint_eligible": False,
            "performance_claim_authorized": False,
            "test_panels_opened": False,
            "best_validation_record": selected,
            "e187_contract_sha256": contract_sha,
            "state": {"weight": torch.tensor([1.0, 2.0])},
        },
        candidate,
    )
    summary = {
        "completed_training_episodes": 120,
        "status": "paused",
        "test_panels_opened": False,
        "performance_claim_authorized": False,
        "validation_history": records,
        "best_validation_record": selected,
        "best_candidate_checkpoint_filename": candidate.name,
        "best_candidate_checkpoint_sha256": e194.sha256(candidate),
    }
    (seed_directory / "training-summary.json").write_text(
        json.dumps(summary), encoding="utf-8"
    )
    return source, candidate


def test_freeze_is_byte_identical_idempotent_and_does_not_mutate_source(
    tmp_path, monkeypatch
):
    source, candidate = _source_tree(tmp_path, monkeypatch)
    output = tmp_path / "frozen"
    before = (candidate.stat().st_mtime_ns, e194.sha256(candidate))

    first = e194.freeze(source, output)
    second = e194.freeze(source, output)

    frozen = output / "seed-0" / e194.FROZEN_NAME
    manifest = output / "seed-0" / e194.MANIFEST_NAME
    assert frozen.read_bytes() == candidate.read_bytes()
    assert first["selected_episode"] == 120
    assert first["frozen_checkpoint_sha256"] == before[1]
    assert first["manifest_core_sha256"] == second["manifest_core_sha256"]
    assert json.loads(manifest.read_text(encoding="utf-8"))["development_only"]
    assert (candidate.stat().st_mtime_ns, e194.sha256(candidate)) == before


def test_inspect_refuses_before_episode_120(tmp_path, monkeypatch):
    source, _candidate = _source_tree(tmp_path, monkeypatch)
    summary_path = source / e194.SOURCE_SUMMARY_RELATIVE
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["completed_training_episodes"] = 119
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    with pytest.raises(e194.FreezeError, match="exactly after episode 120"):
        e194.inspect(source)


def test_inspect_refuses_candidate_hash_mismatch(tmp_path, monkeypatch):
    source, _candidate = _source_tree(tmp_path, monkeypatch)
    summary_path = source / e194.SOURCE_SUMMARY_RELATIVE
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["best_candidate_checkpoint_sha256"] = "0" * 64
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    with pytest.raises(e194.FreezeError, match="SHA-256 differs"):
        e194.inspect(source)


def test_inspect_refuses_an_opened_test_panel(tmp_path, monkeypatch):
    source, _candidate = _source_tree(tmp_path, monkeypatch)
    summary_path = source / e194.SOURCE_SUMMARY_RELATIVE
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["test_panels_opened"] = True
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    with pytest.raises(e194.FreezeError, match="test panels were opened"):
        e194.inspect(source)
