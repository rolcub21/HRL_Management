#!/usr/bin/env python3
"""Dedicated proper training for the frozen VCG-Dense v1.1 controller.

This entry point promotes the winning development-screen condition into one
single-arm, resumable replication protocol.  It intentionally exposes no CLI
switch for the objective, gamma, episode budget, exploration schedule,
environment, model architecture, or validation panel.  Model seed, device,
output location, and pause/resume controls are the only runtime choices.

Frozen condition
----------------

* dense delivery reward ``40 - 1.5 |e| - 0.5 (|e| - 20)_+``;
* SMDP gamma ``0.99``;
* 500 unique training EpisodeInstances per model seed;
* epsilon ``0.90 -> 0.05`` over 3,000 macro decisions;
* greedy checkpoint selection every 25 episodes on seeds 79000--79019;
* canonical VCG-v1 graph, complete exact-SAFE frontier, and liveness guard;
* no viability critic, baseline teacher/query, fallback, or future schedule.

The 69000--69009 stress panel and 77000--77049 in-regime panel remain sealed.
This trainer never authorizes opening them; it only produces candidate
checkpoints for later three-training-seed confirmation.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from typing import Mapping, Optional, Sequence

import torch

import run_vcg_objective_gamma_audit as audit
from train_viability_graph_smdp import (
    CERTIFICATE_SCOPE,
    NO_FALLBACK_CONTRACT,
    SEALED_STRESS_V1_HOLDOUT_SEEDS,
    SMDP_RETURN_CONTRACT,
    epsilon_at,
    resolve_device,
    seed_everything,
)
from train_viability_graph_smdp_proper import SEALED_IN_REGIME_TEST_SEEDS
from vcg_objective_audit import DENSE_PIECEWISE, TimingObjectiveSpec
from viability_graph_episodic_audit import (
    EPISODIC_TERMINAL_BOUNDARY_CONTRACT,
    EPISODIC_VIABILITY_GRAPH_CHECKPOINT_FAMILY,
    EpisodicViabilityGraphHierarchyAgent,
)


TRAINING_PROTOCOL = "vcg_dense_piecewise_gamma0p99_proper_training_v1"
METHOD_VERSION = "vcg_dense_v1_1"
TRAINER_SCHEMA_VERSION = 1
CHECKPOINT_SELECTION_VERSION = (
    "strict_completion_dense_return_mae_relocation_earlier_lexicographic_v1"
)
FROZEN_RECIPE_SOURCE = "vcg_objective_gamma_factorial_development_screen_v1"

FROZEN_GAMMA = 0.99
FROZEN_TOTAL_EPISODES = 500
FROZEN_VALIDATION_SEEDS = tuple(range(79_000, 79_020))
FROZEN_TRAIN_SEED_ORIGIN = 50_000_000
FROZEN_TRAIN_SEED_STRIDE = 1_000_000
FROZEN_EVAL_EVERY = 25
FROZEN_CHECKPOINT_EVERY = 25
FROZEN_EPSILON_START = 0.90
FROZEN_EPSILON_END = 0.05
FROZEN_EPSILON_WARMUP_DECISIONS = 0
FROZEN_EPSILON_DECAY_DECISIONS = 3_000
FROZEN_MAX_STEPS = 2_000
ALLOWED_MODEL_SEEDS = (0, 1, 2)
CANDIDATE_FILENAMES = (
    "best-candidate-slot-a.pth",
    "best-candidate-slot-b.pth",
)

# These panels have already been used for calibration, legacy-v1 validation,
# the objective/discount mechanism screen, or the post-screen development
# comparison.  The proper-training protocol uses neither them nor either
# sealed test panel.
OPENED_DEVELOPMENT_SEEDS = frozenset(
    set(range(74_000, 74_003))
    | set(range(75_000, 75_020))
    | set(range(76_000, 76_010))
    | set(range(78_000, 78_020))
)

FROZEN_OBJECTIVE_SPEC = TimingObjectiveSpec.dense(
    dense_b=40.0,
    lambda_abs=1.5,
    lambda_outside=0.5,
    window=20.0,
)
PROTECTED_SEEDS = frozenset(
    SEALED_STRESS_V1_HOLDOUT_SEEDS | SEALED_IN_REGIME_TEST_SEEDS
)
RESERVED_NONTRAINING_SEEDS = frozenset(
    set(PROTECTED_SEEDS) | set(OPENED_DEVELOPMENT_SEEDS)
)

FROZEN_AUDIT_DEFAULTS = {
    "grid_rows": 5,
    "grid_cols": 5,
    "number_blocks": 8,
    "arrival_rate": 10.0,
    "proc_mean": 80,
    "legacy_base": 10.0,
    "legacy_max_bonus": 30.0,
    "learning_rate": 5.0e-5,
    "reward_scale": 0.01,
    "batch_size": 128,
    "replay_size": 20_000,
    "update_every": 1,
    "updates_per_macro": 1,
    "target_update_every": 200,
    "grad_clip": 5.0,
    "huber_delta": 1.0,
    "graph_hidden_dim": 64,
    "graph_embedding_dim": 64,
    "message_passing_steps": 3,
    "action_embedding_dim": 32,
    "head_hidden_dim": 128,
    "tau_accept": 0.1,
    "tau_recover": 0.1,
    "tau_defer": 0.1,
    "tau_mode": 1.0,
    "search_max_depth": None,
    "search_max_nodes": 20_000,
    "search_max_primitive_steps": None,
    "search_order": "goal_directed",
    "reserve_queue_cells": True,
    "max_defer_steps": 10,
    "max_consecutive_defers": 16,
    "max_nonprogress_recovery_decisions": 2,
    "force_recovery_witness_when_due": False,
}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train one frozen VCG-Dense v1.1 replication"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--model-seed", type=int, choices=ALLOWED_MODEL_SEEDS, required=True
    )
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda"), default="auto"
    )
    parser.add_argument("--resume-existing", action="store_true")
    parser.add_argument("--stop-after-episode", type=int)
    parser.add_argument("--log-every", type=int, default=5)
    return parser


def _frozen_audit_args(runtime) -> argparse.Namespace:
    """Construct the audit executor namespace from immutable defaults."""

    argv = [
        "--output-dir",
        str(runtime.output_dir),
        "--model-seed",
        str(runtime.model_seed),
        "--total-episodes",
        str(FROZEN_TOTAL_EPISODES),
        "--train-instance-seed-base",
        str(_derived_train_seed_base(runtime.model_seed)),
        "--validation-seeds",
        *(str(seed) for seed in FROZEN_VALIDATION_SEEDS),
        "--eval-every",
        str(FROZEN_EVAL_EVERY),
        "--checkpoint-every",
        str(FROZEN_CHECKPOINT_EVERY),
        "--log-every",
        str(runtime.log_every),
        "--epsilon-start",
        str(FROZEN_EPSILON_START),
        "--epsilon-end",
        str(FROZEN_EPSILON_END),
        "--epsilon-warmup-decisions",
        str(FROZEN_EPSILON_WARMUP_DECISIONS),
        "--epsilon-decay-decisions",
        str(FROZEN_EPSILON_DECAY_DECISIONS),
        "--max-steps",
        str(FROZEN_MAX_STEPS),
        "--dense-b",
        str(FROZEN_OBJECTIVE_SPEC.dense_b),
        "--lambda-abs",
        str(FROZEN_OBJECTIVE_SPEC.lambda_abs),
        "--lambda-outside",
        str(FROZEN_OBJECTIVE_SPEC.lambda_outside),
        "--target-window",
        str(FROZEN_OBJECTIVE_SPEC.window),
        "--device",
        str(runtime.device),
    ]
    if runtime.stop_after_episode is not None:
        argv.extend(
            ("--stop-after-episode", str(runtime.stop_after_episode))
        )
    return audit.build_parser().parse_args(argv)


def _derived_train_seed_base(model_seed: int) -> int:
    return FROZEN_TRAIN_SEED_ORIGIN + int(model_seed) * FROZEN_TRAIN_SEED_STRIDE


def _next_candidate_filename(current: Optional[str]) -> str:
    if current is None or current == CANDIDATE_FILENAMES[1]:
        return CANDIDATE_FILENAMES[0]
    if current == CANDIDATE_FILENAMES[0]:
        return CANDIDATE_FILENAMES[1]
    raise ValueError(f"invalid candidate checkpoint filename: {current!r}")


def _candidate_path(output_dir: Path, filename: Optional[str]) -> Path:
    if filename not in CANDIDATE_FILENAMES:
        raise ValueError(f"invalid candidate checkpoint filename: {filename!r}")
    return output_dir / str(filename)


def _is_sha256(value) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _best_record_identity(record: Mapping) -> tuple:
    """Finite identity for the selected checkpoint, excluding run NaNs."""

    return (
        int(record["checkpoint_episode"]),
        tuple(float(value) for value in record["selection_score"]),
        tuple(str(value) for value in record["selection_score_fields"]),
        bool(record["deployment_eligible"]),
    )


def _assert_frozen_recipe(args) -> int:
    """Reject any semantic drift from the selected development condition."""

    train_seed_base = audit._validate_args(args)
    expected_scalars = {
        "total_episodes": FROZEN_TOTAL_EPISODES,
        "eval_every": FROZEN_EVAL_EVERY,
        "checkpoint_every": FROZEN_CHECKPOINT_EVERY,
        "max_steps": FROZEN_MAX_STEPS,
        "epsilon_start": FROZEN_EPSILON_START,
        "epsilon_end": FROZEN_EPSILON_END,
        "epsilon_warmup_decisions": FROZEN_EPSILON_WARMUP_DECISIONS,
        "epsilon_decay_decisions": FROZEN_EPSILON_DECAY_DECISIONS,
        **FROZEN_AUDIT_DEFAULTS,
    }
    mismatches = {
        name: (getattr(args, name), expected)
        for name, expected in expected_scalars.items()
        if getattr(args, name) != expected
    }
    if tuple(args.validation_seeds) != FROZEN_VALIDATION_SEEDS:
        mismatches["validation_seeds"] = (
            tuple(args.validation_seeds),
            FROZEN_VALIDATION_SEEDS,
        )
    expected_base = _derived_train_seed_base(args.model_seed)
    if int(train_seed_base) != expected_base:
        mismatches["train_instance_seed_base"] = (
            int(train_seed_base),
            expected_base,
        )
    objective_spec = audit._objective_spec(args, DENSE_PIECEWISE)
    if objective_spec.to_dict() != FROZEN_OBJECTIVE_SPEC.to_dict():
        mismatches["objective_spec"] = (
            objective_spec.to_dict(),
            FROZEN_OBJECTIVE_SPEC.to_dict(),
        )
    graph_config = audit._graph_config(args, FROZEN_GAMMA)
    if float(graph_config.gamma) != FROZEN_GAMMA:
        mismatches["gamma"] = (graph_config.gamma, FROZEN_GAMMA)
    if mismatches:
        raise ValueError(f"VCG-Dense frozen recipe mismatch: {mismatches!r}")
    opened = set(
        range(train_seed_base, train_seed_base + FROZEN_TOTAL_EPISODES)
    ) | set(FROZEN_VALIDATION_SEEDS)
    opened &= RESERVED_NONTRAINING_SEEDS
    if opened:
        raise ValueError(
            "VCG-Dense trainer refuses sealed or previously opened "
            f"development seeds: {sorted(opened)}"
        )
    return int(train_seed_base)


def _resume_contract(args, train_seed_base: int) -> dict:
    graph_config = audit._graph_config(args, FROZEN_GAMMA)
    search_config = audit._search_config(args)
    liveness_rule = audit._liveness_rule(args)
    return {
        "training_protocol": TRAINING_PROTOCOL,
        "method_version": METHOD_VERSION,
        "trainer_schema_version": TRAINER_SCHEMA_VERSION,
        "frozen_recipe_source": FROZEN_RECIPE_SOURCE,
        "objective_spec": FROZEN_OBJECTIVE_SPEC.to_dict(),
        "gamma": FROZEN_GAMMA,
        "model_seed": int(args.model_seed),
        "total_episodes": FROZEN_TOTAL_EPISODES,
        "train_instance_seed_base": int(train_seed_base),
        "validation_instance_seeds": FROZEN_VALIDATION_SEEDS,
        "environment": {
            "grid_rows": args.grid_rows,
            "grid_cols": args.grid_cols,
            "number_blocks": args.number_blocks,
            "arrival_rate": args.arrival_rate,
            "proc_mean": args.proc_mean,
            "max_steps": FROZEN_MAX_STEPS,
        },
        "graph_config": graph_config.to_dict(),
        "search_config": asdict(search_config),
        "liveness_rule": asdict(liveness_rule),
        "updates_per_macro": int(args.updates_per_macro),
        "epsilon_schedule": {
            "start": FROZEN_EPSILON_START,
            "end": FROZEN_EPSILON_END,
            "warmup_decisions": FROZEN_EPSILON_WARMUP_DECISIONS,
            "decay_decisions": FROZEN_EPSILON_DECAY_DECISIONS,
        },
        "validation_every_episodes": FROZEN_EVAL_EVERY,
        "checkpoint_every_episodes": FROZEN_CHECKPOINT_EVERY,
        "checkpoint_selection_version": CHECKPOINT_SELECTION_VERSION,
        "exact_verifier_authoritative": True,
        "viability_critic_enabled": False,
        "baseline_teacher": False,
        "baseline_policy_query": False,
        "future_schedule_visible_to_policy": False,
        "terminal_boundary_contract": EPISODIC_TERMINAL_BOUNDARY_CONTRACT,
        "dual_rescore_fixed_realized_trajectory": True,
        "sealed_test_panels_opened": False,
    }


def _checkpoint(
    agent: EpisodicViabilityGraphHierarchyAgent,
    *,
    include_replay: bool,
    resumable: bool,
    checkpoint_role: str,
    completed_episodes: int,
    contract: dict,
    train_history: Sequence[dict],
    validation_history: Sequence[dict],
    best_record: Optional[dict],
    best_candidate_filename: Optional[str],
    best_candidate_sha256: Optional[str],
    best_checkpoint_sha256: Optional[str],
    protocol_training_complete: bool = False,
    selection_finalized_after_total_episodes: bool = False,
    deployment_checkpoint_eligible: bool = False,
) -> dict:
    complete = int(completed_episodes) == FROZEN_TOTAL_EPISODES
    finalized = bool(selection_finalized_after_total_episodes)
    if bool(protocol_training_complete) != finalized:
        raise ValueError(
            "training completion and checkpoint-selection finalization must "
            "change atomically"
        )
    if finalized and not complete:
        raise ValueError(
            "checkpoint selection cannot be finalized before the frozen "
            "training budget is consumed"
        )
    if deployment_checkpoint_eligible and not finalized:
        raise ValueError(
            "an unfinalized checkpoint cannot be deployment eligible"
        )
    if (
        deployment_checkpoint_eligible
        and checkpoint_role != "best_deployment_finalized"
    ):
        raise ValueError(
            "only the finalized selected-best artifact can be deployment "
            "eligible"
        )
    selected_episode = (
        None
        if best_record is None
        else int(best_record["checkpoint_episode"])
    )
    return agent.checkpoint(
        include_replay=include_replay,
        protocol=TRAINING_PROTOCOL,
        training_protocol=TRAINING_PROTOCOL,
        method_version=METHOD_VERSION,
        trainer_schema_version=TRAINER_SCHEMA_VERSION,
        trainer_resumable=bool(resumable),
        checkpoint_role=str(checkpoint_role),
        model_seed=int(contract["model_seed"]),
        completed_training_episodes=int(completed_episodes),
        total_training_episodes=FROZEN_TOTAL_EPISODES,
        next_training_episode=int(completed_episodes) + 1,
        next_train_instance_seed=(
            int(contract["train_instance_seed_base"]) + completed_episodes
        ),
        resume_contract=contract,
        resume_contract_sha256=audit._contract_hash(contract),
        timing_objective=DENSE_PIECEWISE,
        timing_objective_spec=FROZEN_OBJECTIVE_SPEC.to_dict(),
        train_instance_seeds=tuple(
            range(
                int(contract["train_instance_seed_base"]),
                int(contract["train_instance_seed_base"])
                + completed_episodes,
            )
        ),
        validation_instance_seeds=FROZEN_VALIDATION_SEEDS,
        environment=dict(contract["environment"]),
        viability_search=dict(contract["search_config"]),
        liveness_rule=dict(contract["liveness_rule"]),
        smdp_return_contract=SMDP_RETURN_CONTRACT,
        no_fallback_contract=NO_FALLBACK_CONTRACT,
        certificate_scope=CERTIFICATE_SCOPE,
        training_history=tuple(train_history),
        validation_history=tuple(validation_history),
        best_validation_record=best_record,
        best_candidate_checkpoint_filename=best_candidate_filename,
        best_candidate_checkpoint_sha256=best_candidate_sha256,
        best_checkpoint_sha256=best_checkpoint_sha256,
        global_rng_state=(
            audit._capture_global_rng_state() if resumable else None
        ),
        exact_full=True,
        viability_critic_enabled=False,
        future_schedule_visible_to_policy=False,
        proper_training_confirmation_candidate=True,
        protocol_training_complete=bool(protocol_training_complete),
        selection_finalized_after_total_episodes=finalized,
        selection_finalized_at_training_episode=(
            FROZEN_TOTAL_EPISODES if finalized else None
        ),
        selected_checkpoint_episode=selected_episode,
        deployment_checkpoint_eligible=bool(
            deployment_checkpoint_eligible
        ),
        development_mechanism_screen_only=False,
        performance_claim_authorized=False,
        sealed_in_regime_test_seeds=tuple(
            sorted(SEALED_IN_REGIME_TEST_SEEDS)
        ),
        sealed_stress_test_seeds=tuple(
            sorted(SEALED_STRESS_V1_HOLDOUT_SEEDS)
        ),
        test_panels_opened=False,
    )


def _finalize_best_candidate(
    candidate_path: Path,
    best_path: Path,
    *,
    candidate_sha256: str,
    contract: dict,
    train_history: Sequence[dict],
    validation_history: Sequence[dict],
    best_record: dict,
) -> str:
    """Finalize the selected weights only after the fixed budget completes."""

    if len(train_history) != FROZEN_TOTAL_EPISODES:
        raise ValueError("cannot finalize before all training episodes exist")
    if not candidate_path.is_file():
        raise ValueError("cannot finalize a missing candidate checkpoint")
    if candidate_path.name not in CANDIDATE_FILENAMES:
        raise ValueError("cannot finalize an unrecognized candidate slot")
    if audit._sha256(candidate_path) != candidate_sha256:
        raise ValueError("best-candidate.pth SHA-256 mismatch")
    payload = torch.load(
        candidate_path, map_location="cpu", weights_only=False
    )
    if not isinstance(payload, Mapping):
        raise ValueError("best-candidate.pth must be a mapping")
    expected = {
        "checkpoint_family": EPISODIC_VIABILITY_GRAPH_CHECKPOINT_FAMILY,
        "training_protocol": TRAINING_PROTOCOL,
        "method_version": METHOD_VERSION,
        "checkpoint_role": "best_candidate_unfinalized",
        "model_seed": contract["model_seed"],
        "timing_objective": DENSE_PIECEWISE,
        "timing_objective_spec": FROZEN_OBJECTIVE_SPEC.to_dict(),
        "gamma": FROZEN_GAMMA,
        "protocol_training_complete": False,
        "selection_finalized_after_total_episodes": False,
        "deployment_checkpoint_eligible": False,
        "best_candidate_checkpoint_filename": candidate_path.name,
    }
    mismatches = {
        name: (payload.get(name), value)
        for name, value in expected.items()
        if payload.get(name) != value
    }
    if payload.get("resume_contract") != contract:
        mismatches["resume_contract"] = (
            payload.get("resume_contract"),
            contract,
        )
    if payload.get("config") != contract["graph_config"]:
        mismatches["config"] = (
            payload.get("config"),
            contract["graph_config"],
        )
    candidate_record = payload.get("best_validation_record")
    if not isinstance(candidate_record, Mapping) or (
        _best_record_identity(candidate_record)
        != _best_record_identity(best_record)
    ):
        mismatches["best_validation_record"] = (
            (
                None
                if not isinstance(candidate_record, Mapping)
                else _best_record_identity(candidate_record)
            ),
            _best_record_identity(best_record),
        )
    if mismatches:
        raise ValueError(
            f"cannot finalize incompatible best candidate: {mismatches!r}"
        )

    # The parameter state remains byte-for-byte the selected candidate.  Only
    # lifecycle/provenance metadata is completed after the entire selection
    # horizon has been observed.
    payload.update(
        {
            "trainer_resumable": False,
            "checkpoint_role": "best_deployment_finalized",
            "completed_training_episodes": FROZEN_TOTAL_EPISODES,
            "total_training_episodes": FROZEN_TOTAL_EPISODES,
            "next_training_episode": FROZEN_TOTAL_EPISODES + 1,
            "next_train_instance_seed": (
                int(contract["train_instance_seed_base"])
                + FROZEN_TOTAL_EPISODES
            ),
            "train_instance_seeds": tuple(
                range(
                    int(contract["train_instance_seed_base"]),
                    int(contract["train_instance_seed_base"])
                    + FROZEN_TOTAL_EPISODES,
                )
            ),
            "training_history": tuple(train_history),
            "validation_history": tuple(validation_history),
            "best_validation_record": best_record,
            "best_candidate_checkpoint_sha256": candidate_sha256,
            "best_checkpoint_sha256": None,
            "global_rng_state": None,
            "protocol_training_complete": True,
            "selection_finalized_after_total_episodes": True,
            "selection_finalized_at_training_episode": (
                FROZEN_TOTAL_EPISODES
            ),
            "selected_checkpoint_episode": int(
                best_record["checkpoint_episode"]
            ),
            "deployment_checkpoint_eligible": bool(
                best_record["deployment_eligible"]
            ),
        }
    )
    audit._atomic_torch_save(payload, best_path)
    return audit._sha256(best_path)


def _validate_resume(payload: Mapping, contract: dict) -> None:
    completed = int(payload.get("completed_training_episodes", -1))
    complete = completed == FROZEN_TOTAL_EPISODES
    best_record = payload.get("best_validation_record")
    expected = {
        "checkpoint_family": EPISODIC_VIABILITY_GRAPH_CHECKPOINT_FAMILY,
        "training_protocol": TRAINING_PROTOCOL,
        "method_version": METHOD_VERSION,
        "trainer_schema_version": TRAINER_SCHEMA_VERSION,
        "trainer_resumable": True,
        "checkpoint_role": "latest_resumable",
        "model_seed": contract["model_seed"],
        "timing_objective": DENSE_PIECEWISE,
        "timing_objective_spec": FROZEN_OBJECTIVE_SPEC.to_dict(),
        "gamma": FROZEN_GAMMA,
        "exact_full": True,
        "exact_safe_mask_authoritative": True,
        "viability_critic_enabled": False,
        "baseline_teacher": False,
        "baseline_policy_query": False,
        "proper_training_confirmation_candidate": True,
        "development_mechanism_screen_only": False,
        "performance_claim_authorized": False,
        "test_panels_opened": False,
        "protocol_training_complete": complete,
        "selection_finalized_after_total_episodes": complete,
        # latest.pth contains the final training weights, which may differ
        # from the selected checkpoint.  It is resumable provenance, never a
        # deployment artifact.
        "deployment_checkpoint_eligible": False,
    }
    mismatches = {
        name: (payload.get(name), value)
        for name, value in expected.items()
        if payload.get(name) != value
    }
    if payload.get("resume_contract") != contract:
        mismatches["resume_contract"] = (
            payload.get("resume_contract"),
            contract,
        )
    if payload.get("config") != contract["graph_config"]:
        mismatches["config"] = (
            payload.get("config"),
            contract["graph_config"],
        )
    expected_hash = audit._contract_hash(contract)
    if payload.get("resume_contract_sha256") != expected_hash:
        mismatches["resume_contract_sha256"] = (
            payload.get("resume_contract_sha256"),
            expected_hash,
        )
    state = payload.get("agent_state", {})
    missing = tuple(
        name for name in ("optimizer", "replay", "rng_state") if name not in state
    )
    if missing:
        mismatches["resumable_agent_state"] = (missing, ())
    if payload.get("global_rng_state") is None:
        mismatches["global_rng_state"] = (None, "required")
    if not 0 <= completed <= FROZEN_TOTAL_EPISODES:
        mismatches["completed_training_episodes"] = (
            completed,
            f"integer in [0, {FROZEN_TOTAL_EPISODES}]",
        )
    expected_train_seeds = tuple(
        range(
            int(contract["train_instance_seed_base"]),
            int(contract["train_instance_seed_base"]) + completed,
        )
    )
    if tuple(payload.get("train_instance_seeds", ())) != expected_train_seeds:
        mismatches["train_instance_seeds"] = (
            tuple(payload.get("train_instance_seeds", ())),
            expected_train_seeds,
        )
    candidate_filename = payload.get("best_candidate_checkpoint_filename")
    candidate_sha256 = payload.get("best_candidate_checkpoint_sha256")
    if best_record is None:
        if candidate_filename is not None or candidate_sha256 is not None:
            mismatches["candidate_without_best_record"] = (
                (candidate_filename, candidate_sha256),
                (None, None),
            )
    elif (
        candidate_filename not in CANDIDATE_FILENAMES
        or not _is_sha256(candidate_sha256)
    ):
        mismatches["authenticated_candidate"] = (
            (candidate_filename, candidate_sha256),
            "recognized slot and nonempty SHA-256",
        )
    if mismatches:
        raise ValueError(
            f"incompatible VCG-Dense resumable checkpoint: {mismatches!r}"
        )


def _result_payload(
    args,
    *,
    completed_episodes: int,
    contract: dict,
    train_history: Sequence[dict],
    validation_history: Sequence[dict],
    best_record: Optional[dict],
    best_candidate_filename: Optional[str],
    best_candidate_sha256: Optional[str],
    best_checkpoint_sha256: Optional[str],
    agent: EpisodicViabilityGraphHierarchyAgent,
) -> dict:
    complete = int(completed_episodes) == FROZEN_TOTAL_EPISODES
    finalized = bool(complete and best_checkpoint_sha256)
    deployment_eligible = bool(
        finalized
        and best_record
        and best_record["deployment_eligible"] is True
    )
    return {
        "status": "complete" if complete else "paused",
        "training_protocol": TRAINING_PROTOCOL,
        "method_version": METHOD_VERSION,
        "trainer_schema_version": TRAINER_SCHEMA_VERSION,
        "proper_training_confirmation_candidate": True,
        "protocol_training_complete": complete,
        "selection_finalized_after_total_episodes": finalized,
        "selection_finalized_at_training_episode": (
            FROZEN_TOTAL_EPISODES if finalized else None
        ),
        "performance_claim_authorized": False,
        "formal_test_authorized": False,
        "test_panels_opened": False,
        "objective_spec": FROZEN_OBJECTIVE_SPEC.to_dict(),
        "gamma": FROZEN_GAMMA,
        "model_seed": int(args.model_seed),
        "completed_training_episodes": int(completed_episodes),
        "total_training_episodes": FROZEN_TOTAL_EPISODES,
        "train_instance_seed_base": int(
            contract["train_instance_seed_base"]
        ),
        "validation_instance_seeds": FROZEN_VALIDATION_SEEDS,
        "resume_contract": contract,
        "resume_contract_sha256": audit._contract_hash(contract),
        "training": {
            "runs": tuple(train_history),
            "summary": audit.summarize_objective_runs(train_history),
        },
        "validation_history": tuple(validation_history),
        "final_validation_record": (
            validation_history[-1] if validation_history else None
        ),
        "best_validation_record": best_record,
        "deployment_checkpoint_eligible": deployment_eligible,
        "best_candidate_checkpoint_filename": best_candidate_filename,
        "best_candidate_checkpoint_sha256": best_candidate_sha256,
        "best_checkpoint_sha256": best_checkpoint_sha256,
        "agent_audit": agent.audit(include_decisions=False),
        "latest_checkpoint": str((args.output_dir / "latest.pth").resolve()),
        "best_candidate_checkpoint": (
            str(
                _candidate_path(
                    args.output_dir, best_candidate_filename
                ).resolve()
            )
            if best_record is not None and best_candidate_filename is not None
            else None
        ),
        "best_checkpoint": (
            str((args.output_dir / "best.pth").resolve())
            if finalized
            else None
        ),
    }


def _run(args, *, device: torch.device, train_seed_base: int) -> dict:
    contract = _resume_contract(args, train_seed_base)
    contract_hash = audit._contract_hash(contract)
    manifest_path = args.output_dir / "training-contract.json"
    latest_path = args.output_dir / "latest.pth"
    best_path = args.output_dir / "best.pth"
    summary_path = args.output_dir / "training-summary.json"

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        if not args.resume_existing:
            raise FileExistsError(
                "fresh VCG-Dense training refuses a nonempty output "
                "directory; pass --resume-existing or choose a new directory"
            )
        if not manifest_path.is_file():
            raise ValueError(
                "VCG-Dense resume requires training-contract.json"
            )
        with manifest_path.open(encoding="utf-8") as handle:
            manifest = json.load(handle)
        if manifest.get("resume_contract_sha256") != contract_hash:
            raise ValueError("VCG-Dense resume contract hash mismatch")
        if manifest.get("contract") != audit._json_safe(contract):
            raise ValueError("VCG-Dense resume contract payload mismatch")
        if not latest_path.is_file():
            unexpected = sorted(
                path.name
                for path in args.output_dir.iterdir()
                if path.name != manifest_path.name
                and not path.name.startswith(f".{latest_path.name}.tmp-")
            )
            if unexpected:
                raise ValueError(
                    "VCG-Dense refuses episode-0 recovery because artifacts "
                    "show that a later checkpoint may have been lost: "
                    f"{unexpected!r}"
                )
            # In the only legitimate no-latest state, the authenticated
            # manifest was committed but the episode-0 checkpoint was not.
            # Reconstructing from the frozen model seed is exact and occurs
            # before any training episode can be committed.
            seed_everything(args.model_seed)
            recovery_agent = EpisodicViabilityGraphHierarchyAgent(
                config=audit._graph_config(args, FROZEN_GAMMA),
                seed=args.model_seed,
                device=device,
                epsilon=FROZEN_EPSILON_START,
            )
            initial_latest = _checkpoint(
                recovery_agent,
                include_replay=True,
                resumable=True,
                checkpoint_role="latest_resumable",
                completed_episodes=0,
                contract=contract,
                train_history=(),
                validation_history=(),
                best_record=None,
                best_candidate_filename=None,
                best_candidate_sha256=None,
                best_checkpoint_sha256=None,
            )
            audit._atomic_torch_save(initial_latest, latest_path)
            print(
                "VCG-Dense recovered the deterministic episode-0 "
                "checkpoint from its authenticated manifest",
                flush=True,
            )
        payload = torch.load(
            latest_path, map_location="cpu", weights_only=False
        )
        if not isinstance(payload, Mapping):
            raise ValueError("VCG-Dense latest checkpoint must be a mapping")
        _validate_resume(payload, contract)
        completed = int(payload["completed_training_episodes"])
        best_record = payload.get("best_validation_record")
        best_candidate_filename = payload.get(
            "best_candidate_checkpoint_filename"
        )
        best_candidate_sha256 = payload.get(
            "best_candidate_checkpoint_sha256"
        )
        best_checkpoint_sha256 = payload.get("best_checkpoint_sha256")
        if best_record is not None:
            candidate_path = _candidate_path(
                args.output_dir, best_candidate_filename
            )
            if not candidate_path.is_file() or not best_candidate_sha256:
                raise ValueError(
                    "VCG-Dense resume is missing authenticated "
                    "best-candidate.pth"
                )
            if audit._sha256(candidate_path) != best_candidate_sha256:
                raise ValueError(
                    "VCG-Dense best-candidate.pth SHA-256 mismatch"
                )
        if completed == FROZEN_TOTAL_EPISODES:
            if not best_path.is_file() or not best_checkpoint_sha256:
                raise ValueError(
                    "complete VCG-Dense run has no authenticated best.pth"
                )
            if audit._sha256(best_path) != best_checkpoint_sha256:
                raise ValueError("VCG-Dense best.pth SHA-256 mismatch")
            if not summary_path.is_file():
                raise ValueError("complete VCG-Dense run has no summary")
            with summary_path.open(encoding="utf-8") as handle:
                result = json.load(handle)
            expected_summary = {
                "status": "complete",
                "training_protocol": TRAINING_PROTOCOL,
                "method_version": METHOD_VERSION,
                "model_seed": int(contract["model_seed"]),
                "completed_training_episodes": FROZEN_TOTAL_EPISODES,
                "resume_contract_sha256": contract_hash,
                "protocol_training_complete": True,
                "selection_finalized_after_total_episodes": True,
                "objective_spec": FROZEN_OBJECTIVE_SPEC.to_dict(),
                "gamma": FROZEN_GAMMA,
                "best_candidate_checkpoint_filename": (
                    best_candidate_filename
                ),
                "best_candidate_checkpoint_sha256": (
                    best_candidate_sha256
                ),
                "best_checkpoint_sha256": best_checkpoint_sha256,
                "deployment_checkpoint_eligible": bool(
                    best_record
                    and best_record.get("deployment_eligible") is True
                ),
            }
            summary_mismatches = {
                name: (result.get(name), value)
                for name, value in expected_summary.items()
                if result.get(name) != value
            }
            if summary_mismatches:
                raise ValueError(
                    "complete VCG-Dense summary/latest mismatch: "
                    f"{summary_mismatches!r}"
                )
            return result
        agent = EpisodicViabilityGraphHierarchyAgent.from_checkpoint(
            dict(payload),
            device=device,
            resumable=True,
            seed=args.model_seed,
        )
        audit._optimizer_to(agent.optimizer, device)
        train_history = list(payload.get("training_history", ()))
        validation_history = list(payload.get("validation_history", ()))
        if len(train_history) != completed:
            raise ValueError("VCG-Dense resume history length mismatch")
        if best_checkpoint_sha256 is not None:
            raise ValueError(
                "unfinished latest.pth must not authenticate a finalized best"
            )
        audit._restore_global_rng_state(payload["global_rng_state"])
        print(f"VCG-Dense v1.1 resumed at episode {completed}", flush=True)
    else:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        audit._atomic_json_save(
            {
                "training_protocol": TRAINING_PROTOCOL,
                "method_version": METHOD_VERSION,
                "resume_contract_sha256": contract_hash,
                "contract": contract,
            },
            manifest_path,
        )
        seed_everything(args.model_seed)
        agent = EpisodicViabilityGraphHierarchyAgent(
            config=audit._graph_config(args, FROZEN_GAMMA),
            seed=args.model_seed,
            device=device,
            epsilon=FROZEN_EPSILON_START,
        )
        completed = 0
        train_history = []
        validation_history = []
        best_record = None
        best_candidate_filename = None
        best_candidate_sha256 = None
        best_checkpoint_sha256 = None
        initial_latest = _checkpoint(
            agent,
            include_replay=True,
            resumable=True,
            checkpoint_role="latest_resumable",
            completed_episodes=0,
            contract=contract,
            train_history=train_history,
            validation_history=validation_history,
            best_record=best_record,
            best_candidate_filename=best_candidate_filename,
            best_candidate_sha256=best_candidate_sha256,
            best_checkpoint_sha256=best_checkpoint_sha256,
        )
        audit._atomic_torch_save(initial_latest, latest_path)

    print(
        "VCG-Dense v1.1 | "
        f"device={device} | model_seed={args.model_seed} | "
        f"episodes={FROZEN_TOTAL_EPISODES} | gamma={FROZEN_GAMMA} | "
        "objective=40-1.5|e|-0.5(|e|-20)+",
        flush=True,
    )
    objective_spec = FROZEN_OBJECTIVE_SPEC
    search_config = audit._search_config(args)
    liveness_rule = audit._liveness_rule(args)
    env = audit._new_env(args, objective_spec)
    stop_at = FROZEN_TOTAL_EPISODES
    if args.stop_after_episode is not None:
        stop_at = min(stop_at, int(args.stop_after_episode))
    if completed > stop_at:
        raise ValueError("resume checkpoint is beyond --stop-after-episode")

    for episode_index in range(completed, stop_at):
        episode_number = episode_index + 1
        instance_seed = train_seed_base + episode_index
        env.current_episode = episode_number
        run = audit.run_objective_audit_episode(
            agent,
            env,
            objective_spec=objective_spec,
            instance_seed=instance_seed,
            training=True,
            max_steps=FROZEN_MAX_STEPS,
            search_config=search_config,
            liveness_rule=liveness_rule,
            epsilon_start=FROZEN_EPSILON_START,
            epsilon_end=FROZEN_EPSILON_END,
            epsilon_decay_decisions=FROZEN_EPSILON_DECAY_DECISIONS,
            updates_per_macro=args.updates_per_macro,
            epsilon_warmup_decisions=FROZEN_EPSILON_WARMUP_DECISIONS,
            run_protocol=TRAINING_PROTOCOL,
        )
        train_history.append(run)
        agent.decision_log.clear()
        completed = episode_number
        agent.set_epsilon(
            epsilon_at(
                agent.decision_count,
                start=FROZEN_EPSILON_START,
                end=FROZEN_EPSILON_END,
                warmup_decisions=FROZEN_EPSILON_WARMUP_DECISIONS,
                decay_decisions=FROZEN_EPSILON_DECAY_DECISIONS,
            )
        )

        if episode_number == 1 or episode_number % args.log_every == 0:
            recent = train_history[-min(10, len(train_history)) :]
            print(
                f"Ep {episode_number:4d} | "
                f"DenseR {audit._mean(item['dense_rescored_return'] for item in recent):8.2f} | "
                f"Strict {int(run['strict_method_success'])} | "
                f"Dec {run['macro_decisions']:3d} | "
                f"Replay {len(agent.replay):5d} | "
                f"Grad {agent.gradient_steps:6d} | "
                f"Eps {agent.epsilon:.3f}",
                flush=True,
            )

        validation_due = bool(
            episode_number % FROZEN_EVAL_EVERY == 0
            or episode_number == FROZEN_TOTAL_EPISODES
        )
        if validation_due:
            validation_runs, validation_summary = audit._run_validation(
                agent,
                args,
                objective_spec=objective_spec,
                device=device,
                search_config=search_config,
                liveness_rule=liveness_rule,
                run_protocol=TRAINING_PROTOCOL,
            )
            record = audit._validation_record(
                validation_runs,
                validation_summary,
                checkpoint_episode=episode_number,
                number_blocks=args.number_blocks,
            )
            validation_history.append(record)
            print(
                f"Validation Ep {episode_number:4d} | "
                f"DenseR {validation_summary['mean_dense_rescored_return']:.2f} | "
                f"MAE {validation_summary['mean_absolute_error']:.3f} | "
                f"First2 {validation_summary['first_two_mean_absolute_error']:.3f} | "
                f"Window {validation_summary['within_target_window_rate']:.3f} | "
                f"Strict {validation_summary['strict_method_success_rate']:.3f}",
                flush=True,
            )
            if best_record is None or audit._score_tuple(
                record
            ) > audit._score_tuple(best_record):
                best_record = record
                next_candidate_filename = _next_candidate_filename(
                    best_candidate_filename
                )
                next_candidate_path = _candidate_path(
                    args.output_dir, next_candidate_filename
                )
                best_payload = _checkpoint(
                    agent,
                    include_replay=False,
                    resumable=False,
                    checkpoint_role="best_candidate_unfinalized",
                    completed_episodes=completed,
                    contract=contract,
                    train_history=train_history,
                    validation_history=validation_history,
                    best_record=best_record,
                    best_candidate_filename=next_candidate_filename,
                    best_candidate_sha256=None,
                    best_checkpoint_sha256=None,
                )
                audit._atomic_torch_save(best_payload, next_candidate_path)
                best_candidate_filename = next_candidate_filename
                best_candidate_sha256 = audit._sha256(
                    next_candidate_path
                )
                best_checkpoint_sha256 = None

        checkpoint_due = bool(
            validation_due
            or episode_number % FROZEN_CHECKPOINT_EVERY == 0
            or episode_number == stop_at
        )
        if checkpoint_due:
            protocol_complete = completed == FROZEN_TOTAL_EPISODES
            if protocol_complete:
                if best_record is None or best_candidate_sha256 is None:
                    raise RuntimeError(
                        "proper training finished without a selected candidate"
                    )
                selected_candidate_path = _candidate_path(
                    args.output_dir, best_candidate_filename
                )
                best_checkpoint_sha256 = _finalize_best_candidate(
                    selected_candidate_path,
                    best_path,
                    candidate_sha256=best_candidate_sha256,
                    contract=contract,
                    train_history=train_history,
                    validation_history=validation_history,
                    best_record=best_record,
                )
            latest = _checkpoint(
                agent,
                include_replay=True,
                resumable=True,
                checkpoint_role="latest_resumable",
                completed_episodes=completed,
                contract=contract,
                train_history=train_history,
                validation_history=validation_history,
                best_record=best_record,
                best_candidate_filename=best_candidate_filename,
                best_candidate_sha256=best_candidate_sha256,
                best_checkpoint_sha256=best_checkpoint_sha256,
                protocol_training_complete=protocol_complete,
                selection_finalized_after_total_episodes=protocol_complete,
                deployment_checkpoint_eligible=False,
            )
            result = _result_payload(
                args,
                completed_episodes=completed,
                contract=contract,
                train_history=train_history,
                validation_history=validation_history,
                best_record=best_record,
                best_candidate_filename=best_candidate_filename,
                best_candidate_sha256=best_candidate_sha256,
                best_checkpoint_sha256=best_checkpoint_sha256,
                agent=agent,
            )
            if protocol_complete:
                # `latest.pth` is the completion commit marker.  A crash
                # before this final write leaves the previous resumable
                # checkpoint authoritative; orphan final artifacts are safe
                # to recompute on resume.
                audit._atomic_json_save(result, summary_path)
                audit._atomic_torch_save(latest, latest_path)
            else:
                audit._atomic_torch_save(latest, latest_path)
                audit._atomic_json_save(result, summary_path)

    if completed == 0:
        raise RuntimeError("VCG-Dense trainer completed no episode")
    return _result_payload(
        args,
        completed_episodes=completed,
        contract=contract,
        train_history=train_history,
        validation_history=validation_history,
        best_record=best_record,
        best_candidate_filename=best_candidate_filename,
        best_candidate_sha256=best_candidate_sha256,
        best_checkpoint_sha256=best_checkpoint_sha256,
        agent=agent,
    )


def main(argv: Optional[Sequence[str]] = None) -> dict:
    runtime = _build_parser().parse_args(argv)
    if runtime.log_every <= 0:
        raise ValueError("--log-every must be positive")
    if runtime.stop_after_episode is not None and not (
        1 <= runtime.stop_after_episode <= FROZEN_TOTAL_EPISODES
    ):
        raise ValueError(
            f"--stop-after-episode must be in [1, {FROZEN_TOTAL_EPISODES}]"
        )
    args = _frozen_audit_args(runtime)
    args.resume_existing = bool(runtime.resume_existing)
    train_seed_base = _assert_frozen_recipe(args)
    device = resolve_device(runtime.device)
    result = _run(args, device=device, train_seed_base=train_seed_base)
    summary_path = runtime.output_dir / "training-summary.json"
    audit._atomic_json_save(result, summary_path)
    print(
        json.dumps(
            {
                "status": result["status"],
                "method_version": METHOD_VERSION,
                "model_seed": result["model_seed"],
                "completed_training_episodes": result[
                    "completed_training_episodes"
                ],
                "deployment_checkpoint_eligible": result[
                    "deployment_checkpoint_eligible"
                ],
                "best_checkpoint": result["best_checkpoint"],
                "latest_checkpoint": result["latest_checkpoint"],
                "summary": str(summary_path.resolve()),
            },
            indent=2,
        ),
        flush=True,
    )
    return result


if __name__ == "__main__":
    main()
