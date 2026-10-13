"""Cheap contract and arithmetic checks; no 10x10 episode is run."""

from __future__ import annotations

import json
from types import SimpleNamespace

import run_vcg_objective_gamma_audit as audit
import train_vcg_dense_proper as dense
from experiments.conditioned_vcg.E85_direct_10x10_cost_audit_150k_v1 import run as e85


def _profile(seed: int, decisions: int, *, proof: float = 0.4) -> dict:
    return {
        "protocol": e85.PROTOCOL,
        "model_seed": 0,
        "instance_seed": seed,
        "checkpoint_sha256": "test-checkpoint",
        "captured_decisions": decisions,
        "prefix_wall_seconds_including_shadow": 1.1 * decisions,
        "frontiers": [{
            "frontier_seconds": 0.5,
            "exact_search_seconds": proof,
            "relocation_family_seconds": 0.0,
            "cache_misses_including_family_proofs": 2,
            "native_recovery_searches": 2,
            "relocation_family_proofs": 0,
            "cache_hits": 3,
            "legal_candidates": 5,
            "safe_candidates": 4,
            "unsafe_candidates": 1,
            "unknown_candidates": 0,
            "legal_shadow_seconds": 0.1,
        } for _ in range(decisions)],
        "selection": [{"seconds": 0.2} for _ in range(decisions)],
        "executions": [{"seconds": 0.3} for _ in range(decisions)],
        "td_sample": {"status": "sampled_in_memory", "learn_seconds": 0.1},
    }


def test_e79_runtime_args_reconstruct_checkpoint_config():
    values = dict(dense.FROZEN_AUDIT_DEFAULTS)
    values.update(grid_rows=10, grid_cols=10, number_blocks=64,
                  arrival_rate=0.1, proc_mean=80,
                  max_steps=e85.e79.SCENARIOS[e85.e79.SCENARIO].max_steps)
    args = audit.build_parser().parse_args([
        "--output-dir", "/tmp/e85-test", "--model-seed", "0", "--dense-b", "100",
    ])
    for name, value in values.items():
        setattr(args, name, value)
    config = audit._graph_config(args, 0.99)
    observed = e85._runtime_args(SimpleNamespace(config=config))
    assert observed.grid_rows == 10
    assert observed.number_blocks == 64
    assert audit._graph_config(observed, 0.99).to_dict() == config.to_dict()


def test_short_prefix_never_opens_training_gate(tmp_path):
    e85._write_new(tmp_path / "profile-seed-0-instance-150100-decisions-8.json",
                   _profile(150100, 8))
    result = e85.analyze(tmp_path)
    assert result["gate"] == "insufficient_coverage_for_cost_decision"
    report = json.loads((tmp_path / "e85-report-1-profiles.json").read_text())
    assert report["summary"]["unsafe_candidates"] == 8
    assert report["summary"]["cache_misses_including_family_proofs"] == 16


def test_two_expanded_prefixes_open_only_cost_signal(tmp_path):
    for seed in e85.DEFAULT_SEEDS:
        e85._write_new(tmp_path / f"profile-seed-0-instance-{seed}-decisions-20.json",
                       _profile(seed, 20))
    result = e85.analyze(tmp_path)
    assert result["gate"] == "cost_signal_only_prepare_matched_10x10_pilot"
    report = json.loads((tmp_path / "e85-report-2-profiles.json").read_text())
    assert report["summary"]["certification_work_fraction_of_training_proxy"] > 0.30
    assert report["training_runs"] == 0


def test_low_proof_share_refuses_proposal_training(tmp_path):
    for seed in e85.DEFAULT_SEEDS:
        e85._write_new(tmp_path / f"profile-seed-0-instance-{seed}-decisions-20.json",
                       _profile(seed, 20, proof=0.1))
    result = e85.analyze(tmp_path)
    assert result["gate"] == "proof_not_dominant_do_not_train_proposal_arm"


def test_expanded_prefix_supersedes_same_instance_smoke(tmp_path):
    e85._write_new(tmp_path / "profile-seed-0-instance-150100-decisions-8.json",
                   _profile(150100, 8))
    for seed in e85.DEFAULT_SEEDS:
        e85._write_new(tmp_path / f"profile-seed-0-instance-{seed}-decisions-20.json",
                       _profile(seed, 20))
    result = e85.analyze(tmp_path)
    assert result["gate"] == "cost_signal_only_prepare_matched_10x10_pilot"
    report = json.loads((tmp_path / "e85-report-3-profiles.json").read_text())
    assert report["summary"]["boundaries"] == 40
    assert report["excluded_shorter_profile_files"] == [
        "profile-seed-0-instance-150100-decisions-8.json"
    ]


def test_self_check_has_no_episode_or_training():
    result = e85.self_check()
    assert result["status"] == "passed"
    assert result["episode_runs"] == result["training_runs"] == 0
