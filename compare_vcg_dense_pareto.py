#!/usr/bin/env python3
"""Paired VCG-Dense timing/relocation Pareto development experiment.

The protocol evaluates the finalized selected checkpoint from model seeds
0/1/2 and, diagnostically, the episode-500 weights from model seeds 1/2.  The
episode-500 seed-0 artifact is authenticated but omitted because its deployment
policy is byte-identical to the selected seed-0 checkpoint.  Four online
baselines are executed once per immutable EpisodeInstance and reused in every
paired comparison.

This is deliberately a development experiment.  It refuses the declared
sealed panels and cannot authorize a final performance claim.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Mapping, Optional, Sequence
import uuid

import torch

import run_vcg_objective_gamma_audit as objective_audit
from benchmark_viability_critic_priority import (
    EXACT_FULL,
    _liveness_rule,
    _load_controller_checkpoint,
    _make_env,
    _search_config,
    _sha256_file,
    run_arm,
)
from compare_viability_graph_baselines import (
    DYNAMIC_METHOD,
    ENHANCED_GA_METHOD,
    METHOD_TO_SOURCE,
    NEAREST_METHOD,
    RESULT_FIELDS,
    ROLLING_GA_METHOD,
    _baseline_runtime_args,
    _checkpoint_readiness,
    _csv_value,
    _json_safe,
    _normalize_baseline,
    _normalize_vcg,
)
from contention_metrics import (
    CANONICAL_CONTENTION_FIELDS,
    validate_contention_metric_record,
)
from example.episode_instance import EpisodeInstance
from track_b_urgency_evaluate import evaluate_assignment_ablation_one, resolve_device
from train_vcg_dense_proper import (
    FROZEN_GAMMA,
    FROZEN_MAX_STEPS,
    FROZEN_OBJECTIVE_SPEC,
    FROZEN_TRAIN_SEED_ORIGIN,
    FROZEN_TRAIN_SEED_STRIDE,
    FROZEN_TOTAL_EPISODES,
    FROZEN_VALIDATION_SEEDS,
    METHOD_VERSION,
    TRAINING_PROTOCOL,
)
from train_viability_graph_smdp import SEALED_STRESS_V1_HOLDOUT_SEEDS
from train_viability_graph_smdp_proper import SEALED_IN_REGIME_TEST_SEEDS


PROTOCOL = "vcg_dense_v1_1_timing_relocation_pareto_development_v1"
PROTOCOL_SCHEMA_VERSION = 1
EVALUATION_SEEDS = tuple(range(80_000, 80_030))
MODEL_SEEDS = (0, 1, 2)
LATEST_DIAGNOSTIC_SEEDS = (1, 2)
BASELINE_METHODS = (
    NEAREST_METHOD,
    DYNAMIC_METHOD,
    ROLLING_GA_METHOD,
    ENHANCED_GA_METHOD,
)
MAX_STEPS = FROZEN_MAX_STEPS
MAX_DEFER_STEPS = 10
LOOKAHEAD_MARGIN_STEPS = 2.0
ROLLING_POPULATION = 16
ROLLING_GENERATIONS = 10
ROLLING_GA_EGRESS_WEIGHT = 4
GA_SEED_BASE = 310_000
MAE_NONINFERIORITY_MARGIN = 2.0
BOOTSTRAP_SAMPLES = 20_000
BOOTSTRAP_RNG_SEED = 20_260_808

SELECTED_BEST = "selected_best"
FINAL_DIAGNOSTIC = "episode500_final_diagnostic"
BASELINE_GROUP = "baseline"

RESULTS_FILENAME = "pareto-runs.csv"
REPORT_FILENAME = "pareto-report.json"
AUDIT_FILENAME = "pareto-audit.json"
INSTANCE_MANIFEST_FILENAME = "instance-manifest.json"
PROTOCOL_MANIFEST_FILENAME = "protocol-manifest.json"

IDENTITY_FIELDS = (
    "method_id",
    "policy_group",
    "system_family",
    "model_seed",
    "checkpoint_variant",
    "checkpoint_weight_episode",
    "checkpoint_path",
    "checkpoint_sha256",
    "deployment_policy_digest",
    "primary_analysis",
    "diagnostic_only",
    "deployment_checkpoint_eligible",
    "checkpoint_readiness_interpretation",
    "execution_reused",
    "derived_from_exact_policy_duplicate",
    "derived_from_method_id",
    "environment_legacy_return",
    "dense_objective_return",
    "primary_objective_return",
    "relocations_per_100_deliveries",
)
# Keep the frozen V1 development CSV schema byte-compatible with its existing
# 80000--80029 artifacts.  New prospective protocols define their own additive
# canonical contention columns instead of silently mutating this V1 schema.
RUN_FIELDS = IDENTITY_FIELDS + tuple(
    field
    for field in RESULT_FIELDS
    if field not in {"protocol", "method", *CANONICAL_CONTENTION_FIELDS}
)


@dataclass(frozen=True)
class PolicyArm:
    method_id: str
    policy_group: str
    model_seed: int
    checkpoint_variant: str
    checkpoint_weight_episode: int
    checkpoint_path: Path
    checkpoint_sha256: str
    deployment_policy_digest: str
    primary_analysis: bool
    diagnostic_only: bool
    readiness: dict
    payload: dict


@dataclass(frozen=True)
class TrainingBundle:
    training_dir: Path
    model_seed: int
    selected: PolicyArm
    authenticated_final: PolicyArm
    final_diagnostic: Optional[PolicyArm]
    checkpoint_records: tuple[dict, ...]
    shared_execution_contract: dict


def _canonical_json(value) -> str:
    return json.dumps(
        _json_safe(value),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _digest_json(value) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.tmp-{uuid.uuid4().hex}"
    try:
        temporary.write_text(text, encoding="utf-8")
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_json(path: Path, value) -> None:
    _atomic_text(
        path,
        json.dumps(
            _json_safe(value), indent=2, sort_keys=True, allow_nan=False
        )
        + "\n",
    )


def _load_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _policy_digest(payload: Mapping) -> str:
    """Canonical digest of exactly the state used by greedy deployment."""

    state = payload.get("agent_state", {}).get("Q_local")
    if not isinstance(state, Mapping) or not state:
        raise ValueError("checkpoint has no Q_local deployment state")
    digest = hashlib.sha256()
    digest.update(_canonical_json(payload.get("config")).encode("utf-8"))
    for name in sorted(state):
        tensor = state[name]
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(f"Q_local[{name!r}] is not a tensor")
        tensor = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(_canonical_json(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _compact_deployment_payload(payload: Mapping) -> dict:
    """Drop replay/optimizer state after authenticating a latest checkpoint."""

    compact = dict(payload)
    state = dict(compact["agent_state"])
    for name in ("optimizer", "replay", "rng_state", "loss_history"):
        state.pop(name, None)
    compact["agent_state"] = state
    return compact


def _shared_execution_contract(payload: Mapping) -> dict:
    return {
        "checkpoint_family": payload.get("checkpoint_family"),
        "controller_architecture": payload.get("controller_architecture"),
        "network_architecture": payload.get("network_architecture"),
        "candidate_interface": payload.get("candidate_interface"),
        "backup_version": payload.get("backup_version"),
        "config": payload.get("config"),
        "environment": payload.get("environment"),
        "viability_search": payload.get("viability_search"),
        "liveness_rule": payload.get("liveness_rule"),
        "timing_objective": payload.get("timing_objective"),
        "timing_objective_spec": payload.get("timing_objective_spec"),
        "gamma": payload.get("gamma"),
        "episode_horizon_steps": payload.get("episode_horizon_steps"),
        "exact_safe_mask_authoritative": payload.get(
            "exact_safe_mask_authoritative"
        ),
        "baseline_teacher": payload.get("baseline_teacher"),
        "baseline_policy_query": payload.get("baseline_policy_query"),
        "viability_critic_enabled": payload.get("viability_critic_enabled"),
    }


def _require_equal(name: str, observed, expected) -> None:
    if _json_safe(observed) != _json_safe(expected):
        raise ValueError(
            f"authenticated training bundle {name} mismatch: "
            f"observed={observed!r}, expected={expected!r}"
        )


def _validate_frozen_resume_contract(contract: Mapping, model_seed: int) -> None:
    expected = {
        "training_protocol": TRAINING_PROTOCOL,
        "method_version": METHOD_VERSION,
        "model_seed": model_seed,
        "total_episodes": FROZEN_TOTAL_EPISODES,
        "train_instance_seed_base": (
            FROZEN_TRAIN_SEED_ORIGIN
            + model_seed * FROZEN_TRAIN_SEED_STRIDE
        ),
        "validation_instance_seeds": FROZEN_VALIDATION_SEEDS,
        "objective_spec": FROZEN_OBJECTIVE_SPEC.to_dict(),
        "gamma": FROZEN_GAMMA,
        "environment": {
            "grid_rows": 5,
            "grid_cols": 5,
            "number_blocks": 8,
            "arrival_rate": 10.0,
            "proc_mean": 80,
            "max_steps": MAX_STEPS,
        },
        "graph_config": {
            "graph_hidden_dim": 64,
            "graph_embedding_dim": 64,
            "message_passing_steps": 3,
            "action_embedding_dim": 32,
            "head_hidden_dim": 128,
            "tau_accept": 0.1,
            "tau_recover": 0.1,
            "tau_defer": 0.1,
            "tau_mode": 1.0,
            "timing_scale": 100.0,
            "gamma": FROZEN_GAMMA,
            "reward_scale": 0.01,
            "learning_rate": 5.0e-5,
            "batch_size": 128,
            "replay_capacity": 20_000,
            "update_every": 1,
            "target_update_every": 200,
            "grad_clip": 5.0,
            "huber_delta": 1.0,
            "max_nonprogress_recovery_decisions": 2,
            "force_recovery_witness_when_due": False,
            "episode_horizon_steps": MAX_STEPS,
        },
        "search_config": {
            "max_depth": None,
            "max_nodes": 20_000,
            "max_primitive_steps": None,
            "reserve_queue_cells": True,
            "search_order": "goal_directed",
        },
        "liveness_rule": {
            "max_option_steps": 10,
            "max_consecutive_defer_decisions": 16,
            "version": "bounded_observable_event_or_positive_deadline_defer_v1",
        },
        "updates_per_macro": 1,
        "epsilon_schedule": {
            "start": 0.9,
            "end": 0.05,
            "warmup_decisions": 0,
            "decay_decisions": 3_000,
        },
        "validation_every_episodes": 25,
        "checkpoint_every_episodes": 25,
        "checkpoint_selection_version": (
            "strict_completion_dense_return_mae_relocation_earlier_lexicographic_v1"
        ),
        "exact_verifier_authoritative": True,
        "viability_critic_enabled": False,
        "baseline_teacher": False,
        "baseline_policy_query": False,
        "future_schedule_visible_to_policy": False,
        "sealed_test_panels_opened": False,
    }
    for name, value in expected.items():
        _require_equal(f"resume_contract.{name}", contract.get(name), value)
    graph = contract.get("graph_config")
    if not isinstance(graph, Mapping):
        raise ValueError("resume_contract.graph_config is missing")
    _require_equal("resume_contract.graph_config.gamma", graph.get("gamma"), FROZEN_GAMMA)
    _require_equal(
        "resume_contract.graph_config.episode_horizon_steps",
        graph.get("episode_horizon_steps"),
        MAX_STEPS,
    )
    if set(EVALUATION_SEEDS).intersection(
        range(
            int(expected["train_instance_seed_base"]),
            int(expected["train_instance_seed_base"]) + FROZEN_TOTAL_EPISODES,
        )
    ):
        raise RuntimeError("Pareto development seeds intersect training seeds")


def _checkpoint_record(
    arm: PolicyArm,
    *,
    executed: bool,
    omission_reason: Optional[str] = None,
    analysis_alias: bool = False,
) -> dict:
    return {
        "method_id": arm.method_id,
        "policy_group": arm.policy_group,
        "model_seed": arm.model_seed,
        "checkpoint_variant": arm.checkpoint_variant,
        "checkpoint_weight_episode": arm.checkpoint_weight_episode,
        "checkpoint_path": str(arm.checkpoint_path.resolve()),
        "checkpoint_sha256": arm.checkpoint_sha256,
        "deployment_policy_digest": arm.deployment_policy_digest,
        "primary_analysis": arm.primary_analysis,
        "diagnostic_only": arm.diagnostic_only,
        "checkpoint_readiness": arm.readiness,
        "executed": bool(executed),
        "analysis_alias": bool(analysis_alias),
        "omission_reason": omission_reason,
    }


def _authenticate_training_bundle(training_dir: Path) -> TrainingBundle:
    training_dir = training_dir.resolve()
    manifest = _load_json(training_dir / "training-contract.json")
    summary = _load_json(training_dir / "training-summary.json")
    best_path = training_dir / "best.pth"
    latest_path = training_dir / "latest.pth"
    best = _load_controller_checkpoint(best_path)
    latest = _load_controller_checkpoint(latest_path)

    contract = best.get("resume_contract")
    if not isinstance(contract, Mapping):
        raise ValueError(f"{best_path} has no authenticated resume contract")
    contract_hash = objective_audit._contract_hash(dict(contract))
    _require_equal(
        "manifest training protocol",
        manifest.get("training_protocol"),
        TRAINING_PROTOCOL,
    )
    _require_equal(
        "manifest method version", manifest.get("method_version"), METHOD_VERSION
    )
    _require_equal("manifest contract", manifest.get("contract"), contract)
    _require_equal(
        "manifest contract SHA-256",
        manifest.get("resume_contract_sha256"),
        contract_hash,
    )
    for label, payload in (("best", best), ("latest", latest)):
        _require_equal(f"{label} resume contract", payload.get("resume_contract"), contract)
        _require_equal(
            f"{label} resume contract SHA-256",
            payload.get("resume_contract_sha256"),
            contract_hash,
        )
    _require_equal("summary resume contract", summary.get("resume_contract"), contract)
    _require_equal(
        "summary resume contract SHA-256",
        summary.get("resume_contract_sha256"),
        contract_hash,
    )

    model_seed = int(best.get("model_seed", -1))
    if model_seed not in MODEL_SEEDS:
        raise ValueError(f"unexpected VCG-Dense model seed: {model_seed}")
    _validate_frozen_resume_contract(contract, model_seed)
    _require_equal(
        "resume_contract.graph_config",
        contract.get("graph_config"),
        best.get("config"),
    )
    expected_common = {
        "training_protocol": TRAINING_PROTOCOL,
        "method_version": METHOD_VERSION,
        "model_seed": model_seed,
        "completed_training_episodes": FROZEN_TOTAL_EPISODES,
        "protocol_training_complete": True,
        "selection_finalized_after_total_episodes": True,
        "proper_training_confirmation_candidate": True,
        "timing_objective": "dense_piecewise_v1",
        "gamma": FROZEN_GAMMA,
        "test_panels_opened": False,
        "performance_claim_authorized": False,
        "future_schedule_visible_to_policy": False,
        "exact_full": True,
    }
    for label, payload in (("best", best), ("latest", latest)):
        for name, expected in expected_common.items():
            _require_equal(f"{label}.{name}", payload.get(name), expected)
        _require_equal(
            f"{label}.timing_objective_spec",
            payload.get("timing_objective_spec"),
            FROZEN_OBJECTIVE_SPEC.to_dict(),
        )
        if payload.get("baseline_policy_query") is not False:
            raise ValueError(f"{label} checkpoint permits a baseline policy query")
        if payload.get("viability_critic_enabled") is not False:
            raise ValueError(f"{label} checkpoint enables the viability critic")
    _require_equal("latest model seed", latest.get("model_seed"), model_seed)

    for name, expected in {
        "status": "complete",
        "training_protocol": TRAINING_PROTOCOL,
        "method_version": METHOD_VERSION,
        "model_seed": model_seed,
        "completed_training_episodes": FROZEN_TOTAL_EPISODES,
        "protocol_training_complete": True,
        "selection_finalized_after_total_episodes": True,
        "deployment_checkpoint_eligible": True,
        "gamma": FROZEN_GAMMA,
        "test_panels_opened": False,
        "performance_claim_authorized": False,
    }.items():
        _require_equal(f"summary.{name}", summary.get(name), expected)
    _require_equal(
        "summary objective spec",
        summary.get("objective_spec"),
        FROZEN_OBJECTIVE_SPEC.to_dict(),
    )

    best_sha = _sha256_file(best_path)
    latest_sha = _sha256_file(latest_path)
    _require_equal(
        "summary best checkpoint SHA-256",
        summary.get("best_checkpoint_sha256"),
        best_sha,
    )
    _require_equal(
        "latest best checkpoint SHA-256",
        latest.get("best_checkpoint_sha256"),
        best_sha,
    )
    _require_equal(
        "best-candidate SHA-256",
        best.get("best_candidate_checkpoint_sha256"),
        summary.get("best_candidate_checkpoint_sha256"),
    )
    _require_equal(
        "latest best-candidate SHA-256",
        latest.get("best_candidate_checkpoint_sha256"),
        summary.get("best_candidate_checkpoint_sha256"),
    )

    best_record = best.get("best_validation_record")
    summary_record = summary.get("best_validation_record")
    if not isinstance(best_record, Mapping) or not isinstance(summary_record, Mapping):
        raise ValueError("completed bundle has no best validation record")
    _require_equal("best validation record", best_record, summary_record)
    selected_episode = int(best_record.get("checkpoint_episode", -1))
    _require_equal(
        "best selected checkpoint episode",
        best.get("selected_checkpoint_episode"),
        selected_episode,
    )
    _require_equal(
        "latest selected checkpoint episode",
        latest.get("selected_checkpoint_episode"),
        selected_episode,
    )

    best_readiness = _checkpoint_readiness(best)
    latest_readiness = _checkpoint_readiness(latest)
    if not best_readiness["deployment_checkpoint_eligible"]:
        raise ValueError("selected best checkpoint is not deployment eligible")
    if not best_readiness["finalized_best_deployment_artifact"]:
        raise ValueError("selected best checkpoint is not a finalized artifact")
    if latest_readiness["deployment_checkpoint_eligible"]:
        raise ValueError("latest checkpoint must remain deployment-ineligible")
    _require_equal("best checkpoint role", best.get("checkpoint_role"), "best_deployment_finalized")
    _require_equal("best trainer resumable", best.get("trainer_resumable"), False)
    _require_equal("latest checkpoint role", latest.get("checkpoint_role"), "latest_resumable")
    _require_equal("latest trainer resumable", latest.get("trainer_resumable"), True)
    _require_equal("latest deployment eligibility", latest.get("deployment_checkpoint_eligible"), False)

    shared = _shared_execution_contract(best)
    _require_equal("best/latest execution contract", _shared_execution_contract(latest), shared)
    selected_digest = _policy_digest(best)
    final_digest = _policy_digest(latest)
    selected = PolicyArm(
        method_id=f"vcg_dense_seed{model_seed}_selected_best",
        policy_group=SELECTED_BEST,
        model_seed=model_seed,
        checkpoint_variant="best",
        checkpoint_weight_episode=selected_episode,
        checkpoint_path=best_path,
        checkpoint_sha256=best_sha,
        deployment_policy_digest=selected_digest,
        primary_analysis=True,
        diagnostic_only=False,
        readiness=best_readiness,
        payload=_compact_deployment_payload(best),
    )
    final = PolicyArm(
        method_id=f"vcg_dense_seed{model_seed}_episode500_final",
        policy_group=FINAL_DIAGNOSTIC,
        model_seed=model_seed,
        checkpoint_variant="latest",
        checkpoint_weight_episode=FROZEN_TOTAL_EPISODES,
        checkpoint_path=latest_path,
        checkpoint_sha256=latest_sha,
        deployment_policy_digest=final_digest,
        primary_analysis=False,
        diagnostic_only=True,
        readiness=latest_readiness,
        payload=_compact_deployment_payload(latest),
    )

    records = [_checkpoint_record(selected, executed=True)]
    if model_seed == 0:
        if selected_episode != FROZEN_TOTAL_EPISODES:
            raise ValueError("seed-0 selected checkpoint is not episode 500")
        if selected_digest != final_digest:
            raise ValueError(
                "seed-0 best/latest were expected to be deployment-policy duplicates"
            )
        records.append(
            _checkpoint_record(
                final,
                executed=False,
                omission_reason=(
                    "execution_reused_from_exact_Q_local_duplicate_of_seed0_selected_best"
                ),
                analysis_alias=True,
            )
        )
        final_diagnostic = None
    else:
        if model_seed not in LATEST_DIAGNOSTIC_SEEDS:
            raise ValueError(f"latest diagnostic not declared for seed {model_seed}")
        records.append(_checkpoint_record(final, executed=True))
        final_diagnostic = final

    return TrainingBundle(
        training_dir=training_dir,
        model_seed=model_seed,
        selected=selected,
        authenticated_final=final,
        final_diagnostic=final_diagnostic,
        checkpoint_records=tuple(records),
        shared_execution_contract=shared,
    )


def _load_training_bundles(training_dirs: Sequence[Path]) -> tuple[TrainingBundle, ...]:
    if len(training_dirs) != len(MODEL_SEEDS):
        raise ValueError("exactly three --training-dirs are required")
    resolved = tuple(path.resolve() for path in training_dirs)
    if len(set(resolved)) != len(resolved):
        raise ValueError("--training-dirs must be distinct")
    bundles = tuple(_authenticate_training_bundle(path) for path in resolved)
    by_seed = {bundle.model_seed: bundle for bundle in bundles}
    if set(by_seed) != set(MODEL_SEEDS) or len(by_seed) != len(bundles):
        raise ValueError("training bundles must contain model seeds 0, 1, and 2 once")
    ordered = tuple(by_seed[seed] for seed in MODEL_SEEDS)
    shared = ordered[0].shared_execution_contract
    for bundle in ordered[1:]:
        _require_equal(
            f"seed-{bundle.model_seed} shared execution contract",
            bundle.shared_execution_contract,
            shared,
        )
    return ordered


def _instance_contract(payload: Mapping) -> dict:
    return {
        "protocol": PROTOCOL,
        "protocol_schema_version": PROTOCOL_SCHEMA_VERSION,
        "seeds": EVALUATION_SEEDS,
        "environment": payload["environment"],
        "shared_execution_contract_sha256": _digest_json(
            _shared_execution_contract(payload)
        ),
    }


def _load_or_create_instances(
    output_dir: Path, payload: Mapping, *, resume_existing: bool
) -> tuple[dict[int, EpisodeInstance], dict]:
    instances_dir = output_dir / "instances"
    manifest_path = output_dir / INSTANCE_MANIFEST_FILENAME
    environment = _make_env(payload)
    contract = _instance_contract(payload)
    existing_manifest = _load_json(manifest_path) if manifest_path.is_file() else None
    if existing_manifest is not None:
        _require_equal("instance manifest contract", existing_manifest.get("contract"), contract)
    elif resume_existing and (output_dir / "run-ledger").exists():
        raise ValueError("cannot resume a run ledger without an instance manifest")

    instances: dict[int, EpisodeInstance] = {}
    records = {}
    instances_dir.mkdir(parents=True, exist_ok=True)
    for seed in EVALUATION_SEEDS:
        expected_env = _make_env(payload)
        expected = expected_env.sample_episode_instance(seed)
        expected_text = expected.to_json() + "\n"
        path = instances_dir / f"seed-{seed}.json"
        if path.is_file():
            text = path.read_text(encoding="utf-8")
            if text != expected_text:
                raise ValueError(f"immutable EpisodeInstance mismatch: {path}")
        else:
            _atomic_text(path, expected_text)
        instance = EpisodeInstance.from_json(path.read_text(encoding="utf-8"))
        instance.validate_for(environment)
        if instance.seed is not None and int(instance.seed) != seed:
            raise ValueError(f"EpisodeInstance seed mismatch: {path}")
        instances[seed] = instance
        records[str(seed)] = {
            "path": str(path.resolve()),
            "sha256": _sha256_file(path),
            "instance_id": instance.instance_id,
            "schedule_id": instance.schedule_id,
        }
    manifest = {"contract": contract, "instances": records}
    if existing_manifest is not None:
        _require_equal("instance manifest", existing_manifest, manifest)
    else:
        _atomic_json(manifest_path, manifest)
    return instances, manifest


def _baseline_args(payload: Mapping, device: torch.device) -> SimpleNamespace:
    frozen = SimpleNamespace(
        max_steps=MAX_STEPS,
        max_defer_steps=MAX_DEFER_STEPS,
        lookahead_margin_steps=LOOKAHEAD_MARGIN_STEPS,
        target_window=FROZEN_OBJECTIVE_SPEC.window,
        ga_seed_base=GA_SEED_BASE,
        rolling_population=ROLLING_POPULATION,
        rolling_generations=ROLLING_GENERATIONS,
        rolling_ga_egress_weight=ROLLING_GA_EGRESS_WEIGHT,
    )
    return _baseline_runtime_args(frozen, payload, device)


def _identity_for_policy(arm: PolicyArm) -> dict:
    return {
        "method_id": arm.method_id,
        "policy_group": arm.policy_group,
        "system_family": "learned_exact_viability_constrained_graph_smdp",
        "model_seed": arm.model_seed,
        "checkpoint_variant": arm.checkpoint_variant,
        "checkpoint_weight_episode": arm.checkpoint_weight_episode,
        "checkpoint_path": str(arm.checkpoint_path.resolve()),
        "checkpoint_sha256": arm.checkpoint_sha256,
        "deployment_policy_digest": arm.deployment_policy_digest,
        "primary_analysis": arm.primary_analysis,
        "diagnostic_only": arm.diagnostic_only,
        "deployment_checkpoint_eligible": arm.readiness[
            "deployment_checkpoint_eligible"
        ],
        "checkpoint_readiness_interpretation": arm.readiness["interpretation"],
        "execution_reused": False,
        "derived_from_exact_policy_duplicate": False,
        "derived_from_method_id": None,
    }


def _identity_for_baseline(method: str) -> dict:
    return {
        "method_id": method,
        "policy_group": BASELINE_GROUP,
        "system_family": "deterministic_online_baseline",
        "model_seed": None,
        "checkpoint_variant": None,
        "checkpoint_weight_episode": None,
        "checkpoint_path": None,
        "checkpoint_sha256": None,
        "deployment_policy_digest": None,
        "primary_analysis": True,
        "diagnostic_only": False,
        "deployment_checkpoint_eligible": None,
        "checkpoint_readiness_interpretation": "not_applicable_deterministic_baseline",
        "execution_reused": False,
        "derived_from_exact_policy_duplicate": False,
        "derived_from_method_id": None,
    }


def _finish_row(normalized: dict, identity: Mapping) -> dict:
    delivery_count = int(normalized["delivery_count"])
    contention = validate_contention_metric_record(
        normalized, require_legacy_alias=True
    )
    physical_relocations = contention["physical_storage_relocations"]
    row = {
        **normalized,
        **identity,
        "protocol": PROTOCOL,
        "method": identity["method_id"],
        # Frozen V1 projection only: the already-opened Pareto artifact wrote
        # the generic physical event total into both legacy columns.  Preserve
        # that byte-level schema/value contract without using this historical
        # alias as a causal obstruction metric.  Prospective V2 consumes the
        # canonical normalized fields directly and keeps obstructive_moves
        # equal to target_bound_obstruction_clearances (zero for VCG).
        "obstructive_moves": physical_relocations,
        "environment_legacy_return": normalized["legacy_rescored_return"],
        "dense_objective_return": normalized["dense_rescored_return"],
        "primary_objective_return": normalized["dense_rescored_return"],
        "physical_storage_relocations_per_100_deliveries": (
            100.0 * physical_relocations / delivery_count
            if delivery_count
            else None
        ),
        # Frozen V1 compatibility alias.
        "relocations_per_100_deliveries": (
            100.0 * physical_relocations / delivery_count
            if delivery_count
            else None
        ),
    }
    return row


def _derive_exact_duplicate_rows(
    rows: Sequence[dict], bundle: TrainingBundle
) -> tuple[dict, ...]:
    """Alias seed-0 selected rollouts into the complete diagnostic grid.

    The alias is legitimate only because bundle authentication proved exact
    equality of the Q_local deployment-policy digests.  It is an analysis row,
    not another rollout or independent observation.
    """

    if bundle.model_seed != 0 or bundle.final_diagnostic is not None:
        raise ValueError("exact-duplicate alias is reserved for model seed 0")
    final = bundle.authenticated_final
    if final.deployment_policy_digest != bundle.selected.deployment_policy_digest:
        raise ValueError("cannot alias nonidentical deployment policies")
    selected_rows = tuple(
        row for row in rows if row["method_id"] == bundle.selected.method_id
    )
    if len(selected_rows) != len(EVALUATION_SEEDS):
        raise RuntimeError("seed-0 selected rollout grid is incomplete")
    identity = _identity_for_policy(final)
    identity.update(
        {
            "execution_reused": True,
            "derived_from_exact_policy_duplicate": True,
            "derived_from_method_id": bundle.selected.method_id,
        }
    )
    output = []
    for source in selected_rows:
        alias = {
            **source,
            **identity,
            "method": final.method_id,
        }
        # Method-level audit timings belong to the one physical execution and
        # are retained only in that execution's ledger, not copied as evidence.
        alias.pop("method_audit", None)
        output.append(alias)
    return tuple(output)


def _run_policy_once(
    arm: PolicyArm,
    instance: EpisodeInstance,
    seed: int,
    *,
    block_count: int,
    device: torch.device,
) -> dict:
    raw = run_arm(
        arm=EXACT_FULL,
        controller_payload=arm.payload,
        instance=instance,
        instance_seed=seed,
        search_config=_search_config(arm.payload),
        liveness_rule=_liveness_rule(arm.payload),
        prioritizer=None,
        max_steps=MAX_STEPS,
        device=device,
    )
    normalized = _normalize_vcg(
        raw, instance, block_count, FROZEN_OBJECTIVE_SPEC
    )
    return _finish_row(normalized, _identity_for_policy(arm))


def _run_baseline_once(
    method: str,
    instance: EpisodeInstance,
    seed: int,
    *,
    block_count: int,
    runtime_args: SimpleNamespace,
) -> dict:
    raw = evaluate_assignment_ablation_one(
        runtime_args,
        seed,
        None,
        assignment_source=METHOD_TO_SOURCE[method],
        episode_instance=instance,
    )
    normalized = _normalize_baseline(
        method, raw, instance, block_count, FROZEN_OBJECTIVE_SPEC
    )
    return _finish_row(normalized, _identity_for_baseline(method))


def _input_contract(
    *,
    method_id: str,
    instance_record: Mapping,
    checkpoint_sha256: Optional[str],
    policy_digest: Optional[str],
    shared_contract_sha256: str,
) -> dict:
    return {
        "protocol": PROTOCOL,
        "protocol_schema_version": PROTOCOL_SCHEMA_VERSION,
        "method_id": method_id,
        "instance_seed": int(Path(instance_record["path"]).stem.split("-")[-1]),
        "instance_id": instance_record["instance_id"],
        "schedule_id": instance_record["schedule_id"],
        "instance_sha256": instance_record["sha256"],
        "checkpoint_sha256": checkpoint_sha256,
        "deployment_policy_digest": policy_digest,
        "shared_execution_contract_sha256": shared_contract_sha256,
        "max_steps": MAX_STEPS,
        "lookahead_margin_steps": LOOKAHEAD_MARGIN_STEPS,
        "baseline_methods": BASELINE_METHODS,
        "rolling_ga": {
            "seed_base": GA_SEED_BASE,
            "population": ROLLING_POPULATION,
            "generations": ROLLING_GENERATIONS,
            "egress_weight": ROLLING_GA_EGRESS_WEIGHT,
        },
        "timing_objective_spec": FROZEN_OBJECTIVE_SPEC.to_dict(),
    }


def _ledger_path(output_dir: Path, method_id: str, seed: int) -> Path:
    return output_dir / "run-ledger" / method_id / f"seed-{seed}.json"


def _execution_plan(
    policy_arms: Sequence[PolicyArm],
) -> tuple[tuple[tuple[str, int], ...], tuple[tuple[str, int], ...]]:
    """Return the frozen crossed grid without instantiating any environment."""

    baseline = tuple(
        (method, seed)
        for method in BASELINE_METHODS
        for seed in EVALUATION_SEEDS
    )
    learned = tuple(
        (arm.method_id, seed)
        for arm in policy_arms
        for seed in EVALUATION_SEEDS
    )
    if len(baseline) != len(set(baseline)):
        raise RuntimeError("baseline execution plan contains a duplicate run")
    if len(learned) != len(set(learned)):
        raise RuntimeError("learned execution plan contains a duplicate run")
    return baseline, learned


def _load_or_execute_run(
    *,
    output_dir: Path,
    method_id: str,
    seed: int,
    input_contract: dict,
    execute,
) -> tuple[dict, dict]:
    path = _ledger_path(output_dir, method_id, seed)
    fingerprint = _digest_json(input_contract)
    if path.is_file():
        record = _load_json(path)
        _require_equal("ledger protocol", record.get("protocol"), PROTOCOL)
        _require_equal("ledger run key", record.get("run_key"), f"{method_id}:{seed}")
        _require_equal("ledger input contract", record.get("input_contract"), input_contract)
        _require_equal("ledger input fingerprint", record.get("input_fingerprint"), fingerprint)
        row = record.get("run")
        if not isinstance(row, dict):
            raise ValueError(f"ledger has no run row: {path}")
    else:
        row = execute()
        if row["method_id"] != method_id or int(row["instance_seed"]) != seed:
            raise RuntimeError("executor returned the wrong method/instance identity")
        record = {
            "protocol": PROTOCOL,
            "run_key": f"{method_id}:{seed}",
            "input_contract": input_contract,
            "input_fingerprint": fingerprint,
            "run": {key: value for key, value in row.items() if key != "method_audit"},
            "method_audit": row.get("method_audit"),
        }
        _atomic_json(path, record)
    if row["instance_id"] != input_contract["instance_id"]:
        raise RuntimeError("ledger run instance provenance mismatch")
    if row["schedule_id"] != input_contract["schedule_id"]:
        raise RuntimeError("ledger run schedule provenance mismatch")
    return row, record


def _ledger_manifest_entry(
    output_dir: Path, method_id: str, seed: int, record: Mapping
) -> dict:
    path = _ledger_path(output_dir, method_id, seed)
    return {
        "run_key": record["run_key"],
        "path": str(path.resolve()),
        "sha256": _sha256_file(path),
        "input_fingerprint": record["input_fingerprint"],
        "method_audit_present": record.get("method_audit") is not None,
    }


def _analysis_report(rows: Sequence[dict]) -> dict:
    """Isolated adapter to the concurrently maintained pure analysis module."""

    try:
        from vcg_dense_pareto_analysis import build_pareto_report
    except ModuleNotFoundError as error:  # pragma: no cover - integration guard
        raise RuntimeError(
            "vcg_dense_pareto_analysis.py is required to finalize this experiment"
        ) from error
    return build_pareto_report(
        rows,
        mae_noninferiority_margin=MAE_NONINFERIORITY_MARGIN,
        bootstrap_samples=BOOTSTRAP_SAMPLES,
        rng_seed=BOOTSTRAP_RNG_SEED,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the frozen VCG-Dense timing/relocation Pareto diagnostic"
    )
    parser.add_argument("--training-dirs", nargs=3, type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume-existing", action="store_true")
    return parser


def _validate_protocol() -> None:
    sealed = set(EVALUATION_SEEDS).intersection(
        SEALED_STRESS_V1_HOLDOUT_SEEDS | SEALED_IN_REGIME_TEST_SEEDS
    )
    if sealed:
        raise RuntimeError(f"Pareto development protocol intersects sealed seeds: {sorted(sealed)}")
    if set(EVALUATION_SEEDS).intersection(FROZEN_VALIDATION_SEEDS):
        raise RuntimeError("Pareto development protocol intersects validation seeds")
    if len(EVALUATION_SEEDS) != len(set(EVALUATION_SEEDS)):
        raise RuntimeError("Pareto development seeds are not unique")


def main(argv: Optional[Sequence[str]] = None) -> dict:
    args = _build_parser().parse_args(argv)
    _validate_protocol()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.resume_existing:
        raise FileExistsError(
            f"{output_dir} is nonempty; pass --resume-existing or use a new directory"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    bundles = _load_training_bundles(args.training_dirs)
    canonical_payload = bundles[0].selected.payload
    shared_contract = bundles[0].shared_execution_contract
    shared_contract_sha = _digest_json(shared_contract)
    policy_arms = tuple(bundle.selected for bundle in bundles) + tuple(
        bundle.final_diagnostic
        for bundle in bundles
        if bundle.final_diagnostic is not None
    )
    if len(policy_arms) != 5:
        raise RuntimeError("frozen Pareto protocol requires exactly five learned arms")
    if len({arm.method_id for arm in policy_arms}) != len(policy_arms):
        raise RuntimeError("learned method identifiers are not unique")

    checkpoint_records = tuple(
        record for bundle in bundles for record in bundle.checkpoint_records
    )
    instances, instance_manifest = _load_or_create_instances(
        output_dir,
        canonical_payload,
        resume_existing=args.resume_existing,
    )
    device = resolve_device(args.device)
    protocol_manifest = {
        "protocol": PROTOCOL,
        "protocol_schema_version": PROTOCOL_SCHEMA_VERSION,
        "scope": "development_only_sealed_panels_unopened",
        "performance_claim_authorized": False,
        "training_dirs": tuple(str(bundle.training_dir) for bundle in bundles),
        "checkpoint_records": checkpoint_records,
        "shared_execution_contract": shared_contract,
        "shared_execution_contract_sha256": shared_contract_sha,
        "evaluation_seeds": EVALUATION_SEEDS,
        "execution_device": str(device),
        "baseline_methods": BASELINE_METHODS,
        "fixed_configuration": {
            "max_steps": MAX_STEPS,
            "max_defer_steps": MAX_DEFER_STEPS,
            "lookahead_margin_steps": LOOKAHEAD_MARGIN_STEPS,
            "rolling_population": ROLLING_POPULATION,
            "rolling_generations": ROLLING_GENERATIONS,
            "rolling_ga_egress_weight": ROLLING_GA_EGRESS_WEIGHT,
            "ga_seed_base": GA_SEED_BASE,
            "mae_noninferiority_margin": MAE_NONINFERIORITY_MARGIN,
            "bootstrap_samples": BOOTSTRAP_SAMPLES,
            "bootstrap_rng_seed": BOOTSTRAP_RNG_SEED,
        },
        "instance_manifest_sha256": _digest_json(instance_manifest),
    }
    protocol_manifest_path = output_dir / PROTOCOL_MANIFEST_FILENAME
    if protocol_manifest_path.is_file():
        _require_equal(
            "protocol manifest",
            _load_json(protocol_manifest_path),
            protocol_manifest,
        )
    else:
        if args.resume_existing and (output_dir / "run-ledger").exists():
            raise ValueError("cannot resume a run ledger without a protocol manifest")
        _atomic_json(protocol_manifest_path, protocol_manifest)

    baseline_args = _baseline_args(canonical_payload, device)
    block_count = int(canonical_payload["environment"]["number_blocks"])
    instance_records = instance_manifest["instances"]
    rows = []
    ledger_audits = []
    baseline_plan, learned_plan = _execution_plan(policy_arms)
    arms_by_id = {arm.method_id: arm for arm in policy_arms}

    # Baselines are deliberately outside the training-seed loop and therefore
    # execute exactly once per paired EpisodeInstance.
    for method, seed in baseline_plan:
        instance = instances[seed]
        contract = _input_contract(
            method_id=method,
            instance_record=instance_records[str(seed)],
            checkpoint_sha256=None,
            policy_digest=None,
            shared_contract_sha256=shared_contract_sha,
        )
        row, ledger = _load_or_execute_run(
            output_dir=output_dir,
            method_id=method,
            seed=seed,
            input_contract=contract,
            execute=lambda method=method, instance=instance, seed=seed: _run_baseline_once(
                method,
                instance,
                seed,
                block_count=block_count,
                runtime_args=baseline_args,
            ),
        )
        rows.append(row)
        ledger_audits.append(
            _ledger_manifest_entry(output_dir, method, seed, ledger)
        )
        print(
            f"[{method}] seed={seed} DenseR={row['dense_rescored_return']:.2f} "
            f"MAE={row['mean_absolute_error']} reloc={row['relocations']}",
            flush=True,
        )

    for method_id, seed in learned_plan:
        arm = arms_by_id[method_id]
        instance = instances[seed]
        contract = _input_contract(
            method_id=arm.method_id,
            instance_record=instance_records[str(seed)],
            checkpoint_sha256=arm.checkpoint_sha256,
            policy_digest=arm.deployment_policy_digest,
            shared_contract_sha256=shared_contract_sha,
        )
        row, ledger = _load_or_execute_run(
            output_dir=output_dir,
            method_id=arm.method_id,
            seed=seed,
            input_contract=contract,
            execute=lambda arm=arm, instance=instance, seed=seed: _run_policy_once(
                arm,
                instance,
                seed,
                block_count=block_count,
                device=device,
            ),
        )
        rows.append(row)
        ledger_audits.append(
            _ledger_manifest_entry(output_dir, arm.method_id, seed, ledger)
        )
        print(
            f"[{arm.method_id}] seed={seed} DenseR={row['dense_rescored_return']:.2f} "
            f"MAE={row['mean_absolute_error']} reloc={row['relocations']}",
            flush=True,
        )

    expected_execution_count = len(EVALUATION_SEEDS) * (
        len(BASELINE_METHODS) + len(policy_arms)
    )
    observed = {(row["method_id"], row["instance_id"]) for row in rows}
    if (
        len(rows) != expected_execution_count
        or len(observed) != expected_execution_count
    ):
        raise RuntimeError("Pareto method/instance execution grid is incomplete")
    for row in rows:
        record = instance_records[str(row["instance_seed"])]
        if row["instance_id"] != record["instance_id"] or row["schedule_id"] != record["schedule_id"]:
            raise RuntimeError("paired EpisodeInstance provenance mismatch")

    derived_rows = _derive_exact_duplicate_rows(rows, bundles[0])
    analysis_rows = tuple(rows) + derived_rows
    expected_analysis_count = len(EVALUATION_SEEDS) * (
        len(BASELINE_METHODS) + 2 * len(MODEL_SEEDS)
    )
    analysis_grid = {
        (row["method_id"], row["instance_id"]) for row in analysis_rows
    }
    if (
        len(analysis_rows) != expected_analysis_count
        or len(analysis_grid) != expected_analysis_count
    ):
        raise RuntimeError("Pareto analysis grid is incomplete")

    analysis = _analysis_report(analysis_rows)
    report = {
        **analysis,
        "runner_protocol": PROTOCOL,
        "scope": "development_only_sealed_panels_unopened",
        "performance_claim_authorized": False,
        "checkpoint_manifest": checkpoint_records,
        "instance_manifest": instance_manifest,
        "shared_execution_contract_sha256": shared_contract_sha,
        "device": str(device),
        "execution_run_count": len(rows),
        "analysis_row_count": len(analysis_rows),
        "baseline_run_count": len(EVALUATION_SEEDS) * len(BASELINE_METHODS),
        "learned_execution_run_count": len(EVALUATION_SEEDS) * len(policy_arms),
        "learned_analysis_row_count": len(EVALUATION_SEEDS) * 2 * len(MODEL_SEEDS),
        "derived_exact_duplicate_row_count": len(derived_rows),
        "runs": tuple({key: value for key, value in row.items() if key != "method_audit"} for row in analysis_rows),
    }
    audit = {
        "protocol": PROTOCOL,
        "protocol_manifest": protocol_manifest,
        "instance_manifest": instance_manifest,
        "run_ledger_manifest": ledger_audits,
    }

    results_path = output_dir / RESULTS_FILENAME
    with_rows = []
    for row in analysis_rows:
        with_rows.append({name: _csv_value(row.get(name)) for name in RUN_FIELDS})
    temporary_results = results_path.parent / f".{results_path.name}.tmp-{uuid.uuid4().hex}"
    try:
        with temporary_results.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=RUN_FIELDS)
            writer.writeheader()
            writer.writerows(with_rows)
        temporary_results.replace(results_path)
    finally:
        if temporary_results.exists():
            temporary_results.unlink()
    _atomic_json(output_dir / REPORT_FILENAME, report)
    _atomic_json(output_dir / AUDIT_FILENAME, audit)
    print(f"Runs: {results_path}", flush=True)
    print(f"Report: {output_dir / REPORT_FILENAME}", flush=True)
    print(f"Audit: {output_dir / AUDIT_FILENAME}", flush=True)
    return report


if __name__ == "__main__":
    main()
