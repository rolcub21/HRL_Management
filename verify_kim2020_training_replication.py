#!/usr/bin/env python3
"""Verify an independent Kim-v7 training run against the frozen seed-0 run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from PSLAP.kim2020_a3c_spatial import kim2020_deployment_digest


FIXED_CHECKPOINT_FIELDS = (
    "checkpoint_schema_version",
    "trainer_version",
    "adaptation_contract",
    "adaptation_status",
    "exact_paper_reproduction",
    "source_paper_doi",
    "model_selection_contract",
    "validation_contract",
    "return_contract",
    "reward_contract",
    "action_mapping",
    "selector_architecture",
    "selector_feature_version",
    "exact_recovery_fallback_contract",
    "exact_recovery_max_nodes",
    "deployment_modes_evaluated",
    "config",
    "trainer_config",
    "training_lambda",
    "training_mu",
    "geometry",
    "geometry_contract",
    "training_instance_seed_base",
    "validation_instance_seeds",
    "validation_policy_seed_base",
    "stochastic_rollouts",
    "max_steps",
    "deterministic_algorithms",
    "training_device",
    "cublas_workspace_config",
)


def _load_checkpoint(path):
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise AssertionError(f"checkpoint is not a dictionary: {path}")
    return payload


def _load_history(path):
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8") as handle:
        history = json.load(handle)
    if not isinstance(history, list) or not history:
        raise AssertionError(f"invalid or empty history: {path}")
    return history


def verify_replication(reference_dir, candidate_dir, expected_seed):
    reference_dir = Path(reference_dir)
    candidate_dir = Path(candidate_dir)
    expected_seed = int(expected_seed)
    reference_latest = _load_checkpoint(reference_dir / "latest.pth")
    reference_best = _load_checkpoint(reference_dir / "best.pth")
    candidate_latest = _load_checkpoint(candidate_dir / "latest.pth")
    candidate_best = _load_checkpoint(candidate_dir / "best.pth")
    reference_history = _load_history(reference_dir / "training-history.json")
    candidate_history = _load_history(candidate_dir / "training-history.json")

    mismatches = {}
    for field in FIXED_CHECKPOINT_FIELDS:
        if candidate_latest.get(field) != reference_latest.get(field):
            mismatches[field] = {
                "reference": reference_latest.get(field),
                "candidate": candidate_latest.get(field),
            }
        if candidate_best.get(field) != reference_best.get(field):
            mismatches[f"best.{field}"] = {
                "reference": reference_best.get(field),
                "candidate": candidate_best.get(field),
            }
    if mismatches:
        raise AssertionError(f"frozen training contract mismatch: {mismatches!r}")

    if int(reference_latest.get("training_seed", -1)) != 0:
        raise AssertionError("reference run is not training seed 0")
    for name, payload in (
        ("latest", candidate_latest),
        ("best", candidate_best),
    ):
        if int(payload.get("training_seed", -1)) != expected_seed:
            raise AssertionError(f"{name} training seed mismatch")
        if payload.get("checkpoint_kind") != name:
            raise AssertionError(f"{name} checkpoint-kind mismatch")
        if payload.get("python_hash_seed") != str(expected_seed):
            raise AssertionError(f"{name} PYTHONHASHSEED mismatch")
        if payload.get("resumed_from_checkpoint") is not None:
            raise AssertionError(f"{name} unexpectedly resumed")
        if int(payload.get("resume_start_episode", -1)) != 0:
            raise AssertionError(f"{name} has a nonzero resume start")

    completed = int(candidate_latest.get("completed_training_episodes", -1))
    if completed != 1000:
        raise AssertionError(f"latest checkpoint stopped at episode {completed}")
    trained = int(candidate_latest.get("training_episode_count", -1))
    skipped = int(candidate_latest.get("skipped_episode_count", -1))
    if trained + skipped != completed:
        raise AssertionError("trained plus skipped episode counts are inconsistent")

    reference_eval_episodes = [int(row["episode"]) for row in reference_history]
    candidate_eval_episodes = [int(row["episode"]) for row in candidate_history]
    if candidate_eval_episodes != reference_eval_episodes:
        raise AssertionError(
            "evaluation cadence mismatch: "
            f"{candidate_eval_episodes!r} != {reference_eval_episodes!r}"
        )
    if candidate_eval_episodes != list(range(100, 1001, 100)):
        raise AssertionError("reference evaluation cadence is not the frozen protocol")

    best_episode = int(candidate_best.get("completed_training_episodes", -1))
    if best_episode not in candidate_eval_episodes:
        raise AssertionError("best checkpoint was not selected at an evaluation epoch")
    selection = candidate_best.get("validation_evaluation", {}).get(
        "selection", {}
    )
    if selection.get("eligible") is not True:
        raise AssertionError("best checkpoint did not pass strict stochastic validation")
    if float(selection.get("primary_strict_method_success_rate", 0.0)) != 1.0:
        raise AssertionError("best checkpoint stochastic validation was not strict-success")

    reference_digest = kim2020_deployment_digest(reference_best)
    candidate_digest = kim2020_deployment_digest(candidate_best)
    if candidate_digest == reference_digest:
        raise AssertionError("independent seed produced the reference deployment digest")
    return {
        "verified": True,
        "training_seed": expected_seed,
        "completed_training_episodes": completed,
        "trained_episode_count": trained,
        "skipped_episode_count": skipped,
        "evaluation_episodes": candidate_eval_episodes,
        "selected_episode": best_episode,
        "deployment_digest": candidate_digest,
        "reference_deployment_digest": reference_digest,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--expected-seed", type=int, required=True)
    parser.add_argument("--digest-only", action="store_true")
    args = parser.parse_args(argv)
    audit = verify_replication(
        args.reference_dir, args.candidate_dir, args.expected_seed
    )
    if args.digest_only:
        print(audit["deployment_digest"])
    else:
        print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()

