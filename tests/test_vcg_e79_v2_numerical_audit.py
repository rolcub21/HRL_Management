"""No-episode tests for E79's versioned numerical accounting repair."""

from __future__ import annotations

import math
from unittest.mock import patch

import pytest
import run_vcg_objective_gamma_audit as audit
import torch
import train_vcg_dense_proper as dense
from vcg_objective_audit import DENSE_PIECEWISE
from viability_graph_episodic_audit import EPISODIC_VIABILITY_GRAPH_CHECKPOINT_FAMILY
from experiments.conditioned_vcg.E79_original_vcg_base100_full_10x10_144k_v1 import (
    protocol as old_p,
    run as old_run,
)
from experiments.conditioned_vcg.E79_original_vcg_base100_full_10x10_144k_v2 import (
    protocol as p,
    roundoff,
    run,
)


def test_v2_changes_only_contract_identity_and_numerical_audit():
    old = old_p.contract()
    new = p.contract()
    assert old["contract_sha256"] == (
        "f25a890d6b16f2735ac092bb9af3d43363fffcac826c1a7bdcce2cdfaa78f620"
    )
    for key in ("scenario", "objective", "gamma", "epsilon", "model_seeds",
                "episodes_per_seed", "validation_seeds", "late_window_checkpoints"):
        assert new[key] == old[key]
    assert new["protocol"] != old["protocol"]
    assert new["numerical_audit_repair"]


def test_roundoff_bound_is_episode_specific_and_rejects_material_error():
    records = [{"step_reward_by_objective": {"legacy": -150.0, "dense": 100.0}}
               for _ in range(6400)]
    bound = roundoff._bound_from_records(tuple(records))
    assert 1e-9 < bound < 1e-4
    proxy = roundoff._MathProxy()
    proxy.bound = bound
    assert proxy.isclose(-5000.0, -5000.0 + bound / 2,
                         rel_tol=0.0, abs_tol=1e-9)
    assert proxy.relaxed_comparisons == 1
    with pytest.raises(RuntimeError, match="exceeds the recorded"):
        proxy.isclose(-5000.0, -5000.0 + 10 * bound,
                      rel_tol=0.0, abs_tol=1e-9)
    assert proxy.isclose(1.0, 1.0 + 5e-13, rel_tol=0.0, abs_tol=1e-12)
    assert not proxy.isclose(1.0, 1.0 + 5e-8, rel_tol=0.0, abs_tol=1e-12)


def test_wrapper_scopes_math_patch_without_episode(tmp_path):
    output = tmp_path / "e79-v2"
    with patch.object(old_run, "p", p):
        assert old_run.prepare(output)["status"] == "prepared"
        assert old_run.self_check(output)["status"] == "passed"
    assert math.isclose(1.0, 1.0)

    # The patch is scoped to the new runner. It does not alter the original
    # shared audit module after its context exits.
    original_math = audit.math
    with roundoff.repaired_audit():
        assert audit.math is not original_math
    assert audit.math is original_math


@pytest.mark.parametrize("completed", (2, 6))
def test_migration_keeps_agent_state_history_and_old_output(tmp_path, completed):
    source = tmp_path / "v1"
    source.mkdir()
    old_contract = old_p.contract()
    audit._atomic_json_save(old_contract, source / "contract.json")
    source_dir = source / "training/operational" / p.SCENARIO / "seed-0"
    source_dir.mkdir(parents=True)
    old_resume = {
        "training_protocol": old_p.PROTOCOL,
        "method_version": old_p.METHOD_VERSION,
        "target_contract_sha256": old_contract["contract_sha256"],
        "model_seed": 0,
        "train_instance_seed_base": 12345,
        # Checkpoints retain this tuple while the JSON manifest loads it as a
        # list. Migration must compare their normalized representations.
        "validation_instance_seeds": (112100, 112101, 112102),
        "graph_config": {},
    }
    old_hash = audit._contract_hash(old_resume)
    agent_state = {"optimizer": {"one": 1}, "replay": [1, 2], "rng_state": 7}
    checkpoint = {
        "checkpoint_family": EPISODIC_VIABILITY_GRAPH_CHECKPOINT_FAMILY,
        "protocol": old_p.PROTOCOL,
        "training_protocol": old_p.PROTOCOL,
        "method_version": old_p.METHOD_VERSION,
        "trainer_schema_version": dense.TRAINER_SCHEMA_VERSION,
        "trainer_resumable": True,
        "checkpoint_role": "latest_resumable",
        "model_seed": 0,
        "completed_training_episodes": completed,
        "timing_objective": DENSE_PIECEWISE,
        "timing_objective_spec": p.SPEC.to_dict(),
        "gamma": 0.99,
        "exact_full": True,
        "exact_safe_mask_authoritative": True,
        "viability_critic_enabled": False,
        "baseline_teacher": False,
        "baseline_policy_query": False,
        "proper_training_confirmation_candidate": True,
        "development_mechanism_screen_only": False,
        "performance_claim_authorized": False,
        "test_panels_opened": False,
        "protocol_training_complete": False,
        "selection_finalized_after_total_episodes": False,
        "deployment_checkpoint_eligible": False,
        "resume_contract": old_resume,
        "resume_contract_sha256": old_hash,
        "config": {},
        "agent_state": agent_state,
        "global_rng_state": {"python": 123},
        "train_instance_seeds": tuple(range(12345, 12345 + completed)),
        "training_history": tuple({"episode": index + 1}
                                  for index in range(completed)),
        "validation_history": ({"checkpoint_episode": 6},) if completed == 6 else (),
        "best_validation_record": ({"checkpoint_episode": 6,
                                    "selection_score": [1.0],
                                    "selection_score_fields": ["test"],
                                    "deployment_eligible": False}
                                   if completed == 6 else None),
        "best_candidate_checkpoint_filename": None,
        "best_candidate_checkpoint_sha256": None,
        "td_backup_implementation": p.TD_BACKUP_IMPLEMENTATION,
        "training_budget_rule": "ceil(4000 / jobs_per_episode)",
    }
    if completed == 6:
        candidate_name = dense.CANDIDATE_FILENAMES[0]
        candidate = {
            "training_protocol": old_p.PROTOCOL,
            "method_version": old_p.METHOD_VERSION,
            "resume_contract": old_resume,
            "best_validation_record": checkpoint["best_validation_record"],
        }
        audit._atomic_torch_save(candidate, source_dir / candidate_name)
        checkpoint["best_candidate_checkpoint_filename"] = candidate_name
        checkpoint["best_candidate_checkpoint_sha256"] = audit._sha256(
            source_dir / candidate_name
        )
    audit._atomic_json_save({"contract": old_resume,
                             "resume_contract_sha256": old_hash},
                            source_dir / "training-contract.json")
    audit._atomic_json_save({"completed_training_episodes": completed,
                             "resume_contract_sha256": old_hash},
                            source_dir / "training-summary.json")
    audit._atomic_torch_save(checkpoint, source_dir / "latest.pth")
    original_sha = audit._sha256(source_dir / "latest.pth")

    result = run.migrate_seed(source, tmp_path / "v2", 0)
    assert result["status"] == "migrated"
    assert result["completed_training_episodes"] == completed
    assert audit._sha256(source_dir / "latest.pth") == original_sha
    new_path = tmp_path / "v2/training/operational" / p.SCENARIO / "seed-0/latest.pth"
    migrated = torch.load(new_path, map_location="cpu", weights_only=False)
    assert migrated["agent_state"] == agent_state
    assert migrated["training_history"] == checkpoint["training_history"]
    assert migrated["resume_contract"]["target_contract_sha256"] == p.contract()["contract_sha256"]
    if completed == 6:
        migrated_candidate_path = new_path.parent / candidate_name
        assert audit._sha256(migrated_candidate_path) == (
            migrated["best_candidate_checkpoint_sha256"]
        )
