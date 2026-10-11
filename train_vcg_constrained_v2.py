#!/usr/bin/env python3
"""Isolated smoke trainer contract for constrained VCG-SMDP V2.

This file intentionally does not modify or wrap a V1/V1.2 trainer.  The V2
controller supplies a small runtime adapter (``build_smoke_runtime``) while
this module owns the experiment contract: the dense operational objective,
primitive-event physical-rehandle accounting, projected episode dual, strict
and budget-gated validation, checkpoint metadata, and development seed guard.

The CLI is deliberately short and development-only.  ``--contract-only`` is
usable before the V2 controller runtime lands; a normal run fails clearly if
the required adapter is absent rather than silently falling back to V1.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import importlib
import json
import math
from numbers import Integral, Real
import os
from pathlib import Path
from statistics import fmean
from typing import Mapping, Optional, Protocol, Sequence

import torch

from PSLAP.viability_candidates_hold_v2 import (
    CERTIFIED_HOLD_INTERFACE_V2,
    CERTIFIED_HOLD_RULE_V2,
    HIDDEN_SCHEDULE_CONTRACT_V2,
    ROBUST_STUTTER_CERTIFICATION_CONTRACT_V2,
)
from vcg_objective_audit import TimingObjectiveSpec


TRAINING_PROTOCOL = "vcg_constrained_vector_smdp_v2_development_smoke_v1"
METHOD_VERSION = "vcg_constrained_v2"
TRAINER_SCHEMA_VERSION = 1
CHECKPOINT_FAMILY = "vcg_constrained_vector_smdp_v2"
CHECKPOINT_SCHEMA_VERSION = 1
CONTROLLER_ARCHITECTURE = "exact_safe_shared_graph_vector_q_primal_dual_v2"
BACKUP_VERSION = (
    "elapsed_weighted_single_lagrangian_policy_vector_expected_double_q_two_discount_smdp_v2"
)
COST_DEFINITION = "primitive_completed_storage_to_storage_rehandle_count_v1"
BUDGET_DEFINITION = "expected_rehandles_per_100_required_deliveries_v1"
DUAL_UPDATE_VERSION = "projected_episode_batch_residual_ascent_v1"
CHECKPOINT_SELECTION_VERSION = (
    "strict_exact_safe_budget_feasible_dense_return_mae_rehandle_earlier_v1"
)

FRESH_TRAIN_SEED_BASE = 60_000_000
DEFAULT_VALIDATION_SEEDS = (84_000, 84_001, 84_002)
# The prospective 83xxx namespace is intentionally broader than the currently
# opened 83000--83059 panel.  A smoke must not consume a future 83xxx panel by
# accident merely because its exact bounds changed elsewhere.
PROTECTED_PROSPECTIVE_SEEDS = frozenset(range(83_000, 84_000))

FROZEN_OBJECTIVE_SPEC = TimingObjectiveSpec.dense(
    dense_b=40.0,
    lambda_abs=1.5,
    lambda_outside=0.5,
    window=20.0,
)


class ConstrainedV2TrainerError(ValueError):
    """Raised when a V2 trainer or artifact violates its contract."""


def _finite(value, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ConstrainedV2TrainerError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ConstrainedV2TrainerError(f"{name} must be finite")
    return result


def _nonnegative_int(value, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ConstrainedV2TrainerError(f"{name} must be an integer")
    result = int(value)
    if result < 0:
        raise ConstrainedV2TrainerError(f"{name} must be non-negative")
    return result


def _positive_int(value, *, name: str) -> int:
    result = _nonnegative_int(value, name=name)
    if result <= 0:
        raise ConstrainedV2TrainerError(f"{name} must be positive")
    return result


def primitive_physical_rehandle_cost(info: Mapping) -> int:
    """Return the actual storage-to-storage event cost for one primitive step.

    The current simulator emits ``False`` or the relocated block label in
    ``relocated_block``.  ``True`` is accepted as a lossier legacy event.  No
    action name, selected mode, or post-hoc motive is consulted.
    """

    if not isinstance(info, Mapping):
        raise ConstrainedV2TrainerError("primitive info must be a mapping")
    value = info.get("relocated_block", False)
    if value is False or value is None:
        return 0
    if value is True:
        return 1
    if isinstance(value, str) and value.strip():
        return 1
    raise ConstrainedV2TrainerError(
        "relocated_block must be False/None, True, or a non-empty block label"
    )


def physical_rehandle_cost(step_infos: Sequence[Mapping]) -> int:
    if isinstance(step_infos, (str, bytes)) or not isinstance(step_infos, Sequence):
        raise ConstrainedV2TrainerError("step_infos must be a sequence")
    return sum(primitive_physical_rehandle_cost(info) for info in step_infos)


@dataclass(frozen=True)
class DualConfig:
    budget_per_100_required_deliveries: float = 20.0
    learning_rate: float = 0.05
    lambda_initial: float = 0.0
    lambda_max: float = 20.0
    update_every_episodes: int = 1

    def __post_init__(self) -> None:
        budget = _finite(
            self.budget_per_100_required_deliveries,
            name="budget_per_100_required_deliveries",
        )
        learning_rate = _finite(self.learning_rate, name="dual learning_rate")
        initial = _finite(self.lambda_initial, name="lambda_initial")
        maximum = _finite(self.lambda_max, name="lambda_max")
        _positive_int(self.update_every_episodes, name="update_every_episodes")
        if budget < 0.0:
            raise ConstrainedV2TrainerError("budget must be non-negative")
        if learning_rate <= 0.0:
            raise ConstrainedV2TrainerError("dual learning rate must be positive")
        if maximum <= 0.0:
            raise ConstrainedV2TrainerError("lambda_max must be positive")
        if not 0.0 <= initial <= maximum:
            raise ConstrainedV2TrainerError(
                "lambda_initial must lie in [0, lambda_max]"
            )

    @property
    def budget_per_required_delivery(self) -> float:
        return float(self.budget_per_100_required_deliveries) / 100.0


def episode_budget_residual(
    physical_rehandles: int,
    required_deliveries: int,
    config: DualConfig,
) -> float:
    cost = _nonnegative_int(physical_rehandles, name="physical_rehandles")
    workload = _positive_int(required_deliveries, name="required_deliveries")
    return float(cost - config.budget_per_required_delivery * workload)


@dataclass
class ProjectedEpisodeDual:
    config: DualConfig
    lambda_value: Optional[float] = None
    update_count: int = 0
    episodes_consumed: int = 0
    last_mean_residual: Optional[float] = None
    saturation_count: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.config, DualConfig):
            raise ConstrainedV2TrainerError("dual config must be DualConfig")
        if self.lambda_value is None:
            self.lambda_value = float(self.config.lambda_initial)
        value = _finite(self.lambda_value, name="lambda_value")
        if not 0.0 <= value <= float(self.config.lambda_max):
            raise ConstrainedV2TrainerError(
                "lambda_value must lie in [0, lambda_max]"
            )
        self.lambda_value = value
        self.update_count = _nonnegative_int(self.update_count, name="update_count")
        self.episodes_consumed = _nonnegative_int(
            self.episodes_consumed, name="episodes_consumed"
        )
        self.saturation_count = _nonnegative_int(
            self.saturation_count, name="saturation_count"
        )
        if self.last_mean_residual is not None:
            self.last_mean_residual = _finite(
                self.last_mean_residual, name="last_mean_residual"
            )

    def update(self, episodes: Sequence[Mapping]) -> dict:
        if isinstance(episodes, (str, bytes)) or not isinstance(episodes, Sequence):
            raise ConstrainedV2TrainerError("dual episodes must be a sequence")
        if not episodes:
            raise ConstrainedV2TrainerError("dual update requires at least one episode")
        residuals = []
        for index, episode in enumerate(episodes):
            if not isinstance(episode, Mapping):
                raise ConstrainedV2TrainerError(
                    f"dual episode {index} must be a mapping"
                )
            residuals.append(
                episode_budget_residual(
                    episode["physical_rehandles"],
                    episode["required_deliveries"],
                    self.config,
                )
            )
        before = float(self.lambda_value)
        mean_residual = float(fmean(residuals))
        unprojected = before + float(self.config.learning_rate) * mean_residual
        after = min(max(unprojected, 0.0), float(self.config.lambda_max))
        saturated = bool(
            math.isclose(after, float(self.config.lambda_max), abs_tol=0.0)
            and mean_residual > 0.0
        )
        self.lambda_value = after
        self.update_count += 1
        self.episodes_consumed += len(episodes)
        self.last_mean_residual = mean_residual
        self.saturation_count += int(saturated)
        return {
            "protocol": DUAL_UPDATE_VERSION,
            "lambda_before": before,
            "lambda_after": after,
            "mean_budget_residual": mean_residual,
            "episode_residuals": tuple(residuals),
            "episode_count": len(episodes),
            "unprojected_lambda": unprojected,
            "projected": not math.isclose(after, unprojected, abs_tol=1.0e-15),
            "saturated_with_positive_residual": saturated,
        }

    def state_dict(self) -> dict:
        return {
            "protocol": DUAL_UPDATE_VERSION,
            "config": asdict(self.config),
            "lambda_value": float(self.lambda_value),
            "update_count": int(self.update_count),
            "episodes_consumed": int(self.episodes_consumed),
            "last_mean_residual": self.last_mean_residual,
            "saturation_count": int(self.saturation_count),
        }

    @classmethod
    def from_state_dict(cls, state: Mapping, config: DualConfig) -> "ProjectedEpisodeDual":
        if not isinstance(state, Mapping):
            raise ConstrainedV2TrainerError("dual state must be a mapping")
        if state.get("protocol") != DUAL_UPDATE_VERSION:
            raise ConstrainedV2TrainerError("dual protocol mismatch")
        if dict(state.get("config", {})) != asdict(config):
            raise ConstrainedV2TrainerError("dual configuration mismatch")
        return cls(
            config=config,
            lambda_value=state["lambda_value"],
            update_count=state["update_count"],
            episodes_consumed=state["episodes_consumed"],
            last_mean_residual=state.get("last_mean_residual"),
            saturation_count=state.get("saturation_count", 0),
        )


_RUN_ALIASES = {
    "physical_rehandles": ("physical_storage_relocations",),
    "required_deliveries": ("required_delivery_count",),
    "dense_return": ("dense_rescored_return",),
}


def _run_value(run: Mapping, name: str, index: int):
    names = (name, *_RUN_ALIASES.get(name, ()))
    present = [(key, run[key]) for key in names if key in run]
    if not present:
        raise ConstrainedV2TrainerError(
            f"validation run {index} is missing {name!r}"
        )
    value = present[0][1]
    for key, candidate in present[1:]:
        if candidate != value:
            raise ConstrainedV2TrainerError(
                f"validation run {index} has conflicting aliases for {name!r}"
            )
    return value


def normalize_v2_run(run: Mapping, index: int = 0) -> dict:
    if not isinstance(run, Mapping):
        raise ConstrainedV2TrainerError(f"validation run {index} must be a mapping")
    normalized = dict(run)
    strict = run.get("strict_method_success")
    if not isinstance(strict, bool):
        raise ConstrainedV2TrainerError(
            f"validation run {index} strict_method_success must be boolean"
        )
    normalized["strict_method_success"] = strict
    normalized["physical_rehandles"] = _nonnegative_int(
        _run_value(run, "physical_rehandles", index), name="physical_rehandles"
    )
    normalized["required_deliveries"] = _positive_int(
        _run_value(run, "required_deliveries", index), name="required_deliveries"
    )
    normalized["delivery_count"] = _nonnegative_int(
        run.get("delivery_count"), name="delivery_count"
    )
    normalized["dense_return"] = _finite(
        _run_value(run, "dense_return", index), name="dense_return"
    )
    raw_mae = run.get("mean_absolute_error")
    if raw_mae is None:
        if strict:
            raise ConstrainedV2TrainerError(
                "a strict successful run requires mean_absolute_error"
            )
        normalized["mean_absolute_error"] = None
    else:
        normalized["mean_absolute_error"] = _finite(
            raw_mae, name="mean_absolute_error"
        )
        if normalized["mean_absolute_error"] < 0.0:
            raise ConstrainedV2TrainerError(
                "mean_absolute_error must be non-negative"
            )
    if "method_failure_reason" not in run:
        raise ConstrainedV2TrainerError(
            f"validation run {index} is missing 'method_failure_reason'"
        )
    normalized["method_failure_reason"] = run["method_failure_reason"]
    for field in ("illegal_drops", "fallbacks", "witness_mismatches"):
        if field not in run:
            raise ConstrainedV2TrainerError(
                f"validation run {index} is missing {field!r}"
            )
        normalized[field] = _nonnegative_int(run[field], name=field)
    if "all_selected_candidates_exact_safe" not in run:
        raise ConstrainedV2TrainerError(
            f"validation run {index} is missing "
            "'all_selected_candidates_exact_safe'"
        )
    exact = run["all_selected_candidates_exact_safe"]
    if not isinstance(exact, bool):
        raise ConstrainedV2TrainerError(
            "all_selected_candidates_exact_safe must be boolean"
        )
    normalized["all_selected_candidates_exact_safe"] = exact
    return normalized


def summarize_v2_validation(
    runs: Sequence[Mapping],
    dual_config: DualConfig,
    *,
    checkpoint_episode: int,
) -> dict:
    if isinstance(runs, (str, bytes)) or not isinstance(runs, Sequence) or not runs:
        raise ConstrainedV2TrainerError("validation runs must be non-empty")
    normalized = [normalize_v2_run(run, index) for index, run in enumerate(runs)]
    total_cost = sum(run["physical_rehandles"] for run in normalized)
    total_required = sum(run["required_deliveries"] for run in normalized)
    budget_rate = 100.0 * total_cost / total_required
    residuals = [
        episode_budget_residual(
            run["physical_rehandles"], run["required_deliveries"], dual_config
        )
        for run in normalized
    ]
    safety_issues = []
    for index, run in enumerate(normalized):
        issues = []
        if not run["strict_method_success"]:
            issues.append("strict_method_failure")
        if run["delivery_count"] != run["required_deliveries"]:
            issues.append("incomplete_required_workload")
        if run["method_failure_reason"] is not None:
            issues.append("method_failure_reason_present")
        if not run["all_selected_candidates_exact_safe"]:
            issues.append("selected_candidate_not_exact_safe")
        for field in ("illegal_drops", "fallbacks", "witness_mismatches"):
            if run[field]:
                issues.append(f"nonzero_{field}")
        if issues:
            safety_issues.append({"run_index": index, "issues": issues})
    strict_gate = not safety_issues
    budget_gate = budget_rate <= float(
        dual_config.budget_per_100_required_deliveries
    ) + 1.0e-12
    mean_mae = (
        float(fmean(run["mean_absolute_error"] for run in normalized))
        if all(run["mean_absolute_error"] is not None for run in normalized)
        else None
    )
    return {
        "checkpoint_episode": _positive_int(
            checkpoint_episode, name="checkpoint_episode"
        ),
        "run_count": len(normalized),
        "strict_safety_completion_gate": strict_gate,
        "budget_gate": budget_gate,
        "development_candidate_eligible": bool(strict_gate and budget_gate),
        # This file is a development smoke.  Final deployment eligibility is
        # owned by a later frozen proper-training/holdout protocol.
        "deployment_checkpoint_eligible": False,
        "safety_issues": safety_issues,
        "mean_dense_return": float(fmean(run["dense_return"] for run in normalized)),
        "mean_absolute_error": mean_mae,
        "total_physical_rehandles": total_cost,
        "total_required_deliveries": total_required,
        "physical_rehandles_per_100_required_deliveries": budget_rate,
        "budget_per_100_required_deliveries": float(
            dual_config.budget_per_100_required_deliveries
        ),
        "mean_budget_residual": float(fmean(residuals)),
        "episode_budget_residuals": tuple(float(value) for value in residuals),
        "episode_budget_violation_rate": float(
            sum(value > 1.0e-12 for value in residuals) / len(residuals)
        ),
        "budget_semantics": BUDGET_DEFINITION,
        "complete_case_filtering_used": False,
    }


def validation_score(summary: Mapping) -> Optional[tuple[float, ...]]:
    if not bool(summary.get("development_candidate_eligible")):
        return None
    return (
        float(summary["mean_dense_return"]),
        -float(summary["mean_absolute_error"]),
        -float(summary["physical_rehandles_per_100_required_deliveries"]),
        -float(summary["checkpoint_episode"]),
    )


def select_better_validation(
    incumbent: Optional[Mapping], candidate: Mapping
) -> Optional[Mapping]:
    candidate_score = validation_score(candidate)
    if candidate_score is None:
        return incumbent
    if incumbent is None:
        return candidate
    incumbent_score = validation_score(incumbent)
    if incumbent_score is None or candidate_score > incumbent_score:
        return candidate
    return incumbent


def _json_safe(value):
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def contract_hash(contract: Mapping) -> str:
    encoded = json.dumps(
        _json_safe(contract), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_seed_namespace(
    train_seed_base: int,
    episodes: int,
    validation_seeds: Sequence[int],
) -> None:
    base = _nonnegative_int(train_seed_base, name="train_seed_base")
    count = _positive_int(episodes, name="episodes")
    if isinstance(validation_seeds, (str, bytes)) or not isinstance(
        validation_seeds, Sequence
    ):
        raise ConstrainedV2TrainerError("validation_seeds must be a sequence")
    validation = tuple(
        _nonnegative_int(seed, name="validation_seed") for seed in validation_seeds
    )
    if not validation or len(validation) != len(set(validation)):
        raise ConstrainedV2TrainerError(
            "validation_seeds must be non-empty and unique"
        )
    training = set(range(base, base + count))
    opened = (training | set(validation)) & set(PROTECTED_PROSPECTIVE_SEEDS)
    if opened:
        raise ConstrainedV2TrainerError(
            f"V2 smoke refuses protected prospective 83xxx seeds: {sorted(opened)}"
        )
    overlap = training & set(validation)
    if overlap:
        raise ConstrainedV2TrainerError(
            f"training and validation seeds overlap: {sorted(overlap)}"
        )


def build_training_contract(args: argparse.Namespace) -> dict:
    _validate_args(args)
    _validate_seed_namespace(args.train_seed_base, args.episodes, args.validation_seeds)
    dual_config = DualConfig(
        budget_per_100_required_deliveries=args.rehandle_budget_per_100,
        learning_rate=args.dual_lr,
        lambda_initial=args.lambda_initial,
        lambda_max=args.lambda_max,
        update_every_episodes=args.dual_update_episodes,
    )
    contract = {
        "training_protocol": TRAINING_PROTOCOL,
        "method_version": METHOD_VERSION,
        "trainer_schema_version": TRAINER_SCHEMA_VERSION,
        "checkpoint_family": CHECKPOINT_FAMILY,
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "backup_version": BACKUP_VERSION,
        "cost_definition": COST_DEFINITION,
        "budget_definition": BUDGET_DEFINITION,
        "dual_update_version": DUAL_UPDATE_VERSION,
        "checkpoint_selection_version": CHECKPOINT_SELECTION_VERSION,
        "development_only": True,
        "performance_claim_authorized": False,
        "prospective_83xxx_panel_opened": False,
        "baseline_policy_query": False,
        "baseline_viability_teacher": False,
        "exact_verifier_authoritative": True,
        "no_v1_fallback": True,
        "model_seed": int(args.model_seed),
        "train_seed_base": int(args.train_seed_base),
        "episodes": int(args.episodes),
        "validation_seeds": tuple(int(seed) for seed in args.validation_seeds),
        "validation_every": int(args.validation_every),
        "max_steps": int(args.max_steps),
        "environment": {
            "grid_rows": int(args.grid_rows),
            "grid_cols": int(args.grid_cols),
            "number_blocks": int(args.number_blocks),
            "arrival_rate": float(args.arrival_rate),
            "proc_mean": int(args.proc_mean),
        },
        "dense_objective_spec": FROZEN_OBJECTIVE_SPEC.to_dict(),
        "certified_hold": {
            "frontier_interface": CERTIFIED_HOLD_INTERFACE_V2,
            "rule": CERTIFIED_HOLD_RULE_V2,
            "robust_stutter_contract": ROBUST_STUTTER_CERTIFICATION_CONTRACT_V2,
            "hidden_schedule_contract": HIDDEN_SCHEDULE_CONTRACT_V2,
            "max_option_steps": int(args.max_hold_steps),
            "max_idle_steps": int(args.max_idle_steps),
            "base_v1_defer_frontier_allowed": False,
            "observed_successor_required": True,
        },
        "gamma_operational": float(args.gamma_operational),
        "gamma_rehandle": 1.0,
        "reward_scale": float(args.reward_scale),
        "dual": asdict(dual_config),
        "device": str(args.device),
    }
    contract["contract_sha256"] = contract_hash(contract)
    return contract


def build_checkpoint_metadata(
    *,
    contract: Mapping,
    dual: ProjectedEpisodeDual,
    completed_episodes: int,
    checkpoint_role: str,
    validation_summary: Optional[Mapping],
    core_contract: Mapping,
) -> dict:
    if contract.get("checkpoint_family") != CHECKPOINT_FAMILY:
        raise ConstrainedV2TrainerError("checkpoint contract family mismatch")
    if contract_hash({k: v for k, v in contract.items() if k != "contract_sha256"}) != contract.get(
        "contract_sha256"
    ):
        raise ConstrainedV2TrainerError("training contract hash mismatch")
    role = str(checkpoint_role)
    if role not in ("latest_development_state", "best_development_candidate"):
        raise ConstrainedV2TrainerError("unknown checkpoint role")
    eligible = bool(
        validation_summary is not None
        and validation_summary.get("development_candidate_eligible")
    )
    if role == "best_development_candidate" and not eligible:
        raise ConstrainedV2TrainerError(
            "best development candidate requires eligible validation"
        )
    return {
        "checkpoint_family": CHECKPOINT_FAMILY,
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "checkpoint_role": role,
        # This calibration is deliberately fresh-run-only.  Persisting a
        # diagnostic latest state must not be mislabeled as crash-resumable,
        # especially when a multi-episode dual batch is still pending.
        "resumable": False,
        "development_candidate_eligible": eligible,
        "deployment_checkpoint_eligible": False,
        "completed_episodes": _nonnegative_int(
            completed_episodes, name="completed_episodes"
        ),
        "training_contract_sha256": contract["contract_sha256"],
        "method_version": METHOD_VERSION,
        "backup_version": BACKUP_VERSION,
        "cost_definition": COST_DEFINITION,
        "budget_definition": BUDGET_DEFINITION,
        "dual_state": dual.state_dict(),
        "validation_summary": (
            None if validation_summary is None else dict(validation_summary)
        ),
        "core_contract": dict(core_contract),
        "lambda_frozen_at_evaluation": True,
        "exact_safe_frontier_authoritative": True,
        "scalarized_reward_stored_in_replay": False,
    }


class V2SmokeRuntime(Protocol):
    """Dependency-injection boundary implemented by the isolated V2 core."""

    @property
    def core_contract(self) -> Mapping: ...

    @property
    def dual_lambda(self) -> float: ...

    def set_dual_lambda(self, value: float) -> None: ...

    def run_episode(self, *, instance_seed: int, training: bool, max_steps: int) -> Mapping: ...

    def checkpoint_state(self, *, include_replay: bool) -> Mapping: ...


def _atomic_json(payload: Mapping, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(_json_safe(payload), handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_torch(payload: Mapping, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        torch.save(dict(payload), temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_default_runtime(args: argparse.Namespace, contract: Mapping) -> V2SmokeRuntime:
    try:
        core = importlib.import_module("viability_graph_constrained_v2")
    except ImportError as error:
        raise RuntimeError(
            "constrained V2 core is not installed; use --contract-only until "
            "viability_graph_constrained_v2.py provides build_smoke_runtime"
        ) from error
    factory = getattr(core, "build_smoke_runtime", None)
    if not callable(factory):
        raise RuntimeError(
            "viability_graph_constrained_v2.build_smoke_runtime(args, contract) "
            "is required by the isolated trainer"
        )
    runtime = factory(args=args, contract=dict(contract))
    _validate_runtime(runtime)
    return runtime


def _validate_runtime(runtime: V2SmokeRuntime) -> None:
    for name in ("set_dual_lambda", "run_episode", "checkpoint_state"):
        if not callable(getattr(runtime, name, None)):
            raise RuntimeError(f"V2 smoke runtime is missing callable {name!r}")
    if not isinstance(getattr(runtime, "core_contract", None), Mapping):
        raise RuntimeError("V2 smoke runtime core_contract must be a mapping")
    if runtime.core_contract.get("checkpoint_family") != CHECKPOINT_FAMILY:
        raise RuntimeError("V2 smoke runtime checkpoint family mismatch")
    if runtime.core_contract.get("controller") != CONTROLLER_ARCHITECTURE:
        raise RuntimeError("V2 smoke runtime controller architecture mismatch")
    hold_interface = runtime.core_contract.get("hold_frontier_interface")
    if hold_interface != CERTIFIED_HOLD_INTERFACE_V2:
        raise RuntimeError(
            "V2 smoke runtime must authenticate the certified Hold V2 frontier; "
            f"found {hold_interface!r}"
        )
    if runtime.core_contract.get("cost_definition") != COST_DEFINITION:
        raise RuntimeError(
            "V2 smoke runtime physical cost definition mismatch"
        )
    if runtime.core_contract.get("backup_version") != BACKUP_VERSION:
        raise RuntimeError("V2 smoke runtime shared vector backup mismatch")
    if runtime.core_contract.get("scalarized_reward_stored_in_replay") is not False:
        raise RuntimeError(
            "V2 smoke runtime must store separate operational and rehandle outcomes"
        )
    expected = {
        "training_policy": "induced_nested_regularized_lagrangian_sample",
        "deployment_policy": "nested_lagrangian_map",
        "mixed_discount_policy_merit": (
            "gamma_operational_pow_elapsed_times_qop_minus_lambda_qphys"
        ),
        "operational_gamma": 0.99,
        "rehandle_gamma": 1.0,
        "operational_reward_scale": 0.01,
        "dual_update_authority": "trainer_projected_episode_residual",
        "exact_safe_frontier_authoritative": True,
        "baseline_teacher": False,
        "baseline_policy_query": False,
    }
    mismatches = {
        name: (runtime.core_contract.get(name), value)
        for name, value in expected.items()
        if runtime.core_contract.get(name) != value
    }
    if mismatches:
        raise RuntimeError(f"V2 smoke runtime semantic contract mismatch: {mismatches!r}")
    value = getattr(runtime, "dual_lambda", None)
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(float(value)):
        raise RuntimeError("V2 smoke runtime must expose a finite dual_lambda")


def _assert_runtime_dual(runtime: V2SmokeRuntime, expected: float) -> None:
    observed = float(runtime.dual_lambda)
    if not math.isclose(observed, float(expected), rel_tol=0.0, abs_tol=0.0):
        raise RuntimeError(
            f"V2 runtime dual mismatch: observed {observed}, expected {expected}"
        )


def run_smoke_training(
    args: argparse.Namespace,
    *,
    runtime: Optional[V2SmokeRuntime] = None,
) -> dict:
    contract = build_training_contract(args)
    output_dir = Path(args.output_dir).resolve()
    contract_path = output_dir / "training-contract.json"
    if contract_path.exists() or (output_dir.exists() and any(output_dir.iterdir())):
        raise ConstrainedV2TrainerError(
            "fresh V2 smoke refuses a nonempty output directory"
        )
    if not args.contract_only:
        runtime = runtime or load_default_runtime(args, contract)
        _validate_runtime(runtime)
    _atomic_json(contract, contract_path)
    if args.contract_only:
        summary = {
            "status": "contract_only",
            "training_contract": str(contract_path),
            "training_contract_sha256": contract["contract_sha256"],
            "episodes_executed": 0,
            "prospective_83xxx_panel_opened": False,
        }
        _atomic_json(summary, output_dir / "training-summary.json")
        return summary

    assert runtime is not None  # established before any run artifact was written
    dual_config = DualConfig(**contract["dual"])
    dual = ProjectedEpisodeDual(dual_config)
    runtime.set_dual_lambda(float(dual.lambda_value))
    _assert_runtime_dual(runtime, float(dual.lambda_value))
    training_history = []
    validation_history = []
    pending_dual = []
    best_validation = None
    best_path = output_dir / "best-development-candidate.pth"
    latest_path = output_dir / "latest.pth"

    for episode_index in range(int(args.episodes)):
        episode_number = episode_index + 1
        run = normalize_v2_run(
            runtime.run_episode(
                instance_seed=int(args.train_seed_base) + episode_index,
                training=True,
                max_steps=int(args.max_steps),
            ),
            episode_index,
        )
        training_history.append(run)
        pending_dual.append(run)
        dual_updated = False
        if len(pending_dual) == dual_config.update_every_episodes:
            dual.update(pending_dual)
            pending_dual.clear()
            runtime.set_dual_lambda(float(dual.lambda_value))
            _assert_runtime_dual(runtime, float(dual.lambda_value))
            dual_updated = True

        print(
            "V2 "
            f"Ep {episode_number:4d} | "
            f"TrainR {run['dense_return']:8.2f} | "
            f"Strict {int(run['strict_method_success'])} | "
            f"Reh {run['physical_rehandles']:3d}/"
            f"{run['required_deliveries']:3d} | "
            f"Lambda {float(dual.lambda_value):7.4f} | "
            f"DualUpd {int(dual_updated)} | "
            f"Macros {int(run.get('macro_decisions', 0)):3d} | "
            f"Grad {int(run.get('loss_updates', 0)):3d}",
            flush=True,
        )

        validation_summary = None
        if episode_number % int(args.validation_every) == 0 or episode_number == int(
            args.episodes
        ):
            validation_runs = [
                runtime.run_episode(
                    instance_seed=int(seed),
                    training=False,
                    max_steps=int(args.max_steps),
                )
                for seed in args.validation_seeds
            ]
            validation_summary = summarize_v2_validation(
                validation_runs,
                dual_config,
                checkpoint_episode=episode_number,
            )
            validation_history.append(validation_summary)
            validation_mae = validation_summary["mean_absolute_error"]
            validation_mae_text = (
                "undefined"
                if validation_mae is None
                else f"{float(validation_mae):7.2f}"
            )
            print(
                "V2 "
                f"Validation Ep {episode_number:4d} | "
                f"R {validation_summary['mean_dense_return']:8.2f} | "
                f"MAE {validation_mae_text} | "
                f"Strict {int(validation_summary['strict_safety_completion_gate'])} | "
                "Reh/100 "
                f"{validation_summary['physical_rehandles_per_100_required_deliveries']:7.2f} | "
                f"Budget {int(validation_summary['budget_gate'])} | "
                f"Eligible {int(validation_summary['development_candidate_eligible'])}",
                flush=True,
            )
            selected = select_better_validation(best_validation, validation_summary)
            if selected is validation_summary:
                best_validation = validation_summary
                metadata = build_checkpoint_metadata(
                    contract=contract,
                    dual=dual,
                    completed_episodes=episode_number,
                    checkpoint_role="best_development_candidate",
                    validation_summary=validation_summary,
                    core_contract=runtime.core_contract,
                )
                _atomic_torch(
                    {
                        **metadata,
                        "agent_state": runtime.checkpoint_state(include_replay=False),
                    },
                    best_path,
                )

        latest_metadata = build_checkpoint_metadata(
            contract=contract,
            dual=dual,
            completed_episodes=episode_number,
            checkpoint_role="latest_development_state",
            validation_summary=validation_summary,
            core_contract=runtime.core_contract,
        )
        _atomic_torch(
            {
                **latest_metadata,
                "agent_state": runtime.checkpoint_state(include_replay=True),
                "training_history": training_history,
                "validation_history": validation_history,
            },
            latest_path,
        )

    # A final partial dual batch is not silently discarded.
    if pending_dual:
        dual.update(pending_dual)
        runtime.set_dual_lambda(float(dual.lambda_value))
        _assert_runtime_dual(runtime, float(dual.lambda_value))
        pending_dual.clear()
        latest_metadata = build_checkpoint_metadata(
            contract=contract,
            dual=dual,
            completed_episodes=int(args.episodes),
            checkpoint_role="latest_development_state",
            validation_summary=(validation_history[-1] if validation_history else None),
            core_contract=runtime.core_contract,
        )
        _atomic_torch(
            {
                **latest_metadata,
                "agent_state": runtime.checkpoint_state(include_replay=True),
                "training_history": training_history,
                "validation_history": validation_history,
            },
            latest_path,
        )

    summary = {
        "status": "complete",
        "method_version": METHOD_VERSION,
        "completed_training_episodes": len(training_history),
        "dual_state": dual.state_dict(),
        "development_candidate_eligible": best_validation is not None,
        "deployment_checkpoint_eligible": False,
        "best_validation": best_validation,
        "best_development_candidate": (
            str(best_path) if best_path.exists() else None
        ),
        "latest_checkpoint": str(latest_path),
        "training_contract": str(contract_path),
        "prospective_83xxx_panel_opened": False,
    }
    _atomic_json(summary, output_dir / "training-summary.json")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a short development-only constrained VCG V2 smoke"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--train-seed-base", type=int, default=FRESH_TRAIN_SEED_BASE)
    parser.add_argument(
        "--validation-seeds", type=int, nargs="+", default=DEFAULT_VALIDATION_SEEDS
    )
    parser.add_argument("--validation-every", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--grid-rows", type=int, default=5)
    parser.add_argument("--grid-cols", type=int, default=5)
    parser.add_argument("--number-blocks", type=int, default=2)
    parser.add_argument("--arrival-rate", type=float, default=10.0)
    parser.add_argument("--proc-mean", type=int, default=80)
    parser.add_argument("--gamma-operational", type=float, default=0.99)
    parser.add_argument("--reward-scale", type=float, default=0.01)
    parser.add_argument("--rehandle-budget-per-100", type=float, default=20.0)
    parser.add_argument("--dual-lr", type=float, default=0.05)
    parser.add_argument("--lambda-initial", type=float, default=0.0)
    parser.add_argument("--lambda-max", type=float, default=20.0)
    parser.add_argument("--dual-update-episodes", type=int, default=1)
    parser.add_argument("--max-hold-steps", type=int, default=10)
    parser.add_argument("--max-idle-steps", type=int, default=20)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--contract-only", action="store_true")
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    for name in (
        "episodes",
        "validation_every",
        "max_steps",
        "grid_rows",
        "grid_cols",
        "number_blocks",
        "proc_mean",
        "dual_update_episodes",
        "max_hold_steps",
        "max_idle_steps",
    ):
        _positive_int(getattr(args, name), name=name)
    gamma = _finite(args.gamma_operational, name="gamma_operational")
    if gamma != 0.99:
        raise ConstrainedV2TrainerError(
            "this frozen V2 family requires gamma_operational=0.99"
        )
    if _finite(args.reward_scale, name="reward_scale") != 0.01:
        raise ConstrainedV2TrainerError(
            "this frozen V2 family requires reward_scale=0.01"
        )
    _finite(args.arrival_rate, name="arrival_rate")
    if float(args.arrival_rate) <= 0.0:
        raise ConstrainedV2TrainerError("arrival_rate must be positive")


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    _validate_args(args)
    summary = run_smoke_training(args)
    print(json.dumps(_json_safe(summary), indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()


__all__ = [
    "BACKUP_VERSION",
    "BUDGET_DEFINITION",
    "CHECKPOINT_FAMILY",
    "COST_DEFINITION",
    "FROZEN_OBJECTIVE_SPEC",
    "ConstrainedV2TrainerError",
    "DualConfig",
    "ProjectedEpisodeDual",
    "V2SmokeRuntime",
    "build_checkpoint_metadata",
    "build_parser",
    "build_training_contract",
    "episode_budget_residual",
    "normalize_v2_run",
    "physical_rehandle_cost",
    "primitive_physical_rehandle_cost",
    "run_smoke_training",
    "select_better_validation",
    "summarize_v2_validation",
    "validation_score",
]
