#!/usr/bin/env python3
"""Fresh two-timescale development calibration for constrained VCG V2.1.

V2.1 is deliberately isolated from the V2 calibration and every V1 family.
The trainer owns the complete block clock, temperature schedule, projected
dual, validation gate, and artifact identities.  The controller runtime owns
the critics and environment interaction, but must authenticate and echo every
trainer-supplied policy state.  There is no compatibility fallback.

Frozen schedule
---------------

* episodes 1--20: critic warm-up, lambda fixed at zero;
* episodes 21--100: eight complete 10-episode primal--dual blocks;
* lambda is fixed within a block and updated only after a complete active
  block from the ten raw workload-normalized constraint residuals;
* behavior and Bellman continuation use the same block-constant temperature;
* validation is deterministic nested MAP and occurs only after a block.

Artifacts are development-only and intentionally non-resumable.  Pending
block residuals and the exact clock are nevertheless persisted so an audit can
prove that no partial or off-schedule dual update occurred.
"""

from __future__ import annotations

import argparse
from collections import Counter
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
from train_vcg_constrained_v2 import (
    BUDGET_DEFINITION,
    COST_DEFINITION,
    FROZEN_OBJECTIVE_SPEC,
    normalize_v2_run,
)


TRAINING_PROTOCOL = (
    "vcg_constrained_vector_smdp_v2_1_two_timescale_development_calibration_v1"
)
METHOD_VERSION = "vcg_constrained_v2_1"
TRAINER_SCHEMA_VERSION = 1
CHECKPOINT_FAMILY = "vcg_constrained_vector_smdp_v2_1"
CHECKPOINT_SCHEMA_VERSION = 1
CONTROLLER_ARCHITECTURE = (
    "exact_safe_shared_graph_vector_q_primal_dual_two_timescale_v2_1"
)
BACKUP_VERSION = (
    "elapsed_weighted_shared_annealed_lagrangian_policy_vector_expected_double_q_"
    "two_discount_smdp_v2_1"
)
POLICY_SCHEDULE_PROTOCOL = (
    "block_constant_shared_behavior_backup_temperature_to_map_v1"
)
DUAL_UPDATE_PROTOCOL = (
    "projected_complete_active_block_workload_residual_ascent_v2_1"
)
CHECKPOINT_SELECTION_VERSION = (
    "strict_integrity_aggregate_map_budget_dense_return_mae_rehandle_earlier_v2_1"
)
POLICY_DIAGNOSTIC_DISTRIBUTION = (
    "regularized_operator_distribution_not_executed_map_randomness"
)
FROZEN_VIABILITY_SEARCH_CONFIG = {
    "max_depth": None,
    "max_nodes": 100_000,
    "max_primitive_steps": None,
    "reserve_queue_cells": True,
    "search_order": "goal_directed",
}

TOTAL_EPISODES = 100
BLOCK_EPISODES = 10
WARMUP_EPISODES = 20
WARMUP_BLOCKS = WARMUP_EPISODES // BLOCK_EPISODES
TOTAL_BLOCKS = TOTAL_EPISODES // BLOCK_EPISODES
ANNEAL_FIRST_BLOCK = 3
ANNEAL_LAST_BLOCK = 8
WITHIN_TEMPERATURE_INITIAL = 0.10
WITHIN_TEMPERATURE_FLOOR = 0.01
GROUP_TEMPERATURE_INITIAL = 1.0
GROUP_TEMPERATURE_FLOOR = 0.05
TIMING_NONCOLLAPSE_MAE_MAXIMUM = float(FROZEN_OBJECTIVE_SPEC.window)

# Deliberately paired with the controlled V2 development workload.  Isolation
# comes from the V2.1 family/output identities, not from changing the task
# distribution during an algorithmic repair.
FRESH_TRAIN_SEED_BASE = 60_000_000
DEFAULT_VALIDATION_SEEDS = (84_000, 84_001, 84_002)
PROTECTED_PROSPECTIVE_SEEDS = frozenset(range(83_000, 84_000))


class ConstrainedV21TrainerError(ValueError):
    """Raised when the frozen V2.1 development protocol is violated."""


def _finite(value, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ConstrainedV21TrainerError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ConstrainedV21TrainerError(f"{name} must be finite")
    return result


def _nonnegative_int(value, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ConstrainedV21TrainerError(f"{name} must be an integer")
    result = int(value)
    if result < 0:
        raise ConstrainedV21TrainerError(f"{name} must be non-negative")
    return result


def _positive_int(value, *, name: str) -> int:
    result = _nonnegative_int(value, name=name)
    if result <= 0:
        raise ConstrainedV21TrainerError(f"{name} must be positive")
    return result


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


@dataclass(frozen=True)
class V21DualConfig:
    budget_per_100_required_deliveries: float = 20.0
    learning_rate: float = 0.01
    lambda_initial: float = 0.0
    lambda_max: float = 20.0
    warmup_episodes: int = WARMUP_EPISODES
    block_episodes: int = BLOCK_EPISODES

    def __post_init__(self) -> None:
        budget = _finite(
            self.budget_per_100_required_deliveries,
            name="budget_per_100_required_deliveries",
        )
        learning_rate = _finite(self.learning_rate, name="dual learning_rate")
        initial = _finite(self.lambda_initial, name="lambda_initial")
        maximum = _finite(self.lambda_max, name="lambda_max")
        if budget < 0.0:
            raise ConstrainedV21TrainerError("budget must be non-negative")
        if learning_rate != 0.01:
            raise ConstrainedV21TrainerError(
                "the frozen V2.1 calibration requires dual learning_rate=0.01"
            )
        if initial != 0.0:
            raise ConstrainedV21TrainerError(
                "the frozen V2.1 critic warm-up requires lambda_initial=0"
            )
        if maximum != 20.0:
            raise ConstrainedV21TrainerError(
                "the frozen V2.1 calibration requires lambda_max=20"
            )
        if self.warmup_episodes != WARMUP_EPISODES:
            raise ConstrainedV21TrainerError("V2.1 warm-up must be 20 episodes")
        if self.block_episodes != BLOCK_EPISODES:
            raise ConstrainedV21TrainerError("V2.1 dual blocks must be 10 episodes")

    @property
    def budget_per_required_delivery(self) -> float:
        return float(self.budget_per_100_required_deliveries) / 100.0


def episode_budget_residual(
    physical_rehandles: int,
    required_deliveries: int,
    config: V21DualConfig,
) -> float:
    if not isinstance(config, V21DualConfig):
        raise ConstrainedV21TrainerError("V2.1 residual requires V21DualConfig")
    cost = _nonnegative_int(physical_rehandles, name="physical_rehandles")
    workload = _positive_int(required_deliveries, name="required_deliveries")
    return float(cost - config.budget_per_required_delivery * workload)


@dataclass
class CompleteBlockProjectedDual:
    """Projected dual that accepts complete active blocks and nothing else."""

    config: V21DualConfig
    lambda_value: float = 0.0
    update_count: int = 0
    active_episodes_consumed: int = 0
    last_completed_block: int = WARMUP_BLOCKS
    last_mean_residual: Optional[float] = None
    saturation_count: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.config, V21DualConfig):
            raise ConstrainedV21TrainerError("dual config must be V21DualConfig")
        self.lambda_value = _finite(self.lambda_value, name="lambda_value")
        if not 0.0 <= self.lambda_value <= self.config.lambda_max:
            raise ConstrainedV21TrainerError(
                "lambda_value must lie in [0, lambda_max]"
            )
        self.update_count = _nonnegative_int(self.update_count, name="update_count")
        self.active_episodes_consumed = _nonnegative_int(
            self.active_episodes_consumed, name="active_episodes_consumed"
        )
        self.last_completed_block = _nonnegative_int(
            self.last_completed_block, name="last_completed_block"
        )
        self.saturation_count = _nonnegative_int(
            self.saturation_count, name="saturation_count"
        )
        if self.last_mean_residual is not None:
            self.last_mean_residual = _finite(
                self.last_mean_residual, name="last_mean_residual"
            )

    def update_complete_block(
        self, episodes: Sequence[Mapping], *, block_number: int
    ) -> dict:
        proposal = self.propose_complete_block(
            episodes, block_number=block_number
        )
        block = int(proposal["block_number"])
        if block >= TOTAL_BLOCKS:
            raise ConstrainedV21TrainerError(
                "the terminal block proposal is diagnostic and must not be applied"
            )
        self.lambda_value = float(proposal["lambda_after"])
        self.update_count += 1
        self.active_episodes_consumed += len(episodes)
        self.last_completed_block = block
        self.last_mean_residual = float(proposal["mean_budget_residual"])
        self.saturation_count += int(
            proposal["saturated_with_positive_residual"]
        )
        return proposal

    def propose_complete_block(
        self, episodes: Sequence[Mapping], *, block_number: int
    ) -> dict:
        """Compute a complete-block update without mutating the dual.

        This is used both to validate an update before applying blocks 3--9
        and to retain block 10's deliberately unapplied terminal proposal.
        """

        block = _positive_int(block_number, name="block_number")
        expected = self.last_completed_block + 1
        if block != expected:
            raise ConstrainedV21TrainerError(
                f"dual expected active block {expected}, received {block}"
            )
        if block <= WARMUP_BLOCKS:
            raise ConstrainedV21TrainerError("warm-up residuals must never update lambda")
        if isinstance(episodes, (str, bytes)) or not isinstance(episodes, Sequence):
            raise ConstrainedV21TrainerError("dual episodes must be a sequence")
        if len(episodes) != self.config.block_episodes:
            raise ConstrainedV21TrainerError(
                "dual update requires exactly one complete 10-episode block"
            )
        residuals = []
        for index, episode in enumerate(episodes):
            if not isinstance(episode, Mapping):
                raise ConstrainedV21TrainerError(
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
        unprojected = before + self.config.learning_rate * mean_residual
        after = min(max(unprojected, 0.0), self.config.lambda_max)
        saturated = bool(
            math.isclose(after, self.config.lambda_max, abs_tol=0.0)
            and mean_residual > 0.0
        )
        return {
            "protocol": DUAL_UPDATE_PROTOCOL,
            "block_number": block,
            "lambda_before": before,
            "lambda_after": after,
            "mean_budget_residual": mean_residual,
            "episode_residuals": tuple(float(value) for value in residuals),
            "episode_count": len(episodes),
            "unprojected_lambda": unprojected,
            "projected": not math.isclose(after, unprojected, abs_tol=1.0e-15),
            "saturated_with_positive_residual": saturated,
        }

    def state_dict(self) -> dict:
        return {
            "protocol": DUAL_UPDATE_PROTOCOL,
            "config": asdict(self.config),
            "lambda_value": float(self.lambda_value),
            "update_count": int(self.update_count),
            "active_episodes_consumed": int(self.active_episodes_consumed),
            "last_completed_block": int(self.last_completed_block),
            "last_mean_residual": self.last_mean_residual,
            "saturation_count": int(self.saturation_count),
        }


@dataclass(frozen=True)
class V21ScheduleState:
    phase: str
    episode_number: int
    block_number: int
    within_group_temperatures: tuple[float, float, float, float]
    group_temperature: float
    policy_mode: str
    critic_updates_enabled: bool
    dual_updates_enabled: bool

    def __post_init__(self) -> None:
        if self.phase not in (
            "critic_warmup",
            "primal_dual_anneal",
            "low_temperature_stabilization",
        ):
            raise ConstrainedV21TrainerError("invalid V2.1 schedule phase")
        episode = _positive_int(self.episode_number, name="episode_number")
        block = _positive_int(self.block_number, name="block_number")
        if not 1 <= episode <= TOTAL_EPISODES:
            raise ConstrainedV21TrainerError("episode_number lies outside V2.1 clock")
        if block != (episode - 1) // BLOCK_EPISODES + 1:
            raise ConstrainedV21TrainerError("block_number does not match episode clock")
        if self.policy_mode not in ("regularized_sample", "map"):
            raise ConstrainedV21TrainerError("invalid policy_mode")
        if len(self.within_group_temperatures) != 4:
            raise ConstrainedV21TrainerError(
                "within_group_temperatures must contain four values"
            )
        for value in (*self.within_group_temperatures, self.group_temperature):
            if _finite(value, name="temperature") <= 0.0:
                raise ConstrainedV21TrainerError("temperatures must be positive")
        if not isinstance(self.critic_updates_enabled, bool) or not isinstance(
            self.dual_updates_enabled, bool
        ):
            raise ConstrainedV21TrainerError("schedule update flags must be boolean")
        expected_phase = (
            "critic_warmup"
            if block <= WARMUP_BLOCKS
            else (
                "primal_dual_anneal"
                if block <= ANNEAL_LAST_BLOCK
                else "low_temperature_stabilization"
            )
        )
        if self.phase != expected_phase:
            raise ConstrainedV21TrainerError("phase disagrees with block clock")
        if self.policy_mode == "map":
            if self.critic_updates_enabled or self.dual_updates_enabled:
                raise ConstrainedV21TrainerError("validation cannot update critics or dual")
        else:
            if self.policy_mode != "regularized_sample":
                raise ConstrainedV21TrainerError("training must use regularized sampling")
            if not self.critic_updates_enabled:
                raise ConstrainedV21TrainerError("all training blocks update critics")
            expected_dual = block > WARMUP_BLOCKS
            if self.dual_updates_enabled != expected_dual:
                raise ConstrainedV21TrainerError("dual phase flag disagrees with block clock")

    def to_dict(self) -> dict:
        return {
            "protocol": POLICY_SCHEDULE_PROTOCOL,
            "phase": self.phase,
            "episode_number": int(self.episode_number),
            "block_number": int(self.block_number),
            "within_group_temperatures": tuple(
                float(value) for value in self.within_group_temperatures
            ),
            "group_temperature": float(self.group_temperature),
            "policy_mode": self.policy_mode,
            "critic_updates_enabled": self.critic_updates_enabled,
            "dual_updates_enabled": self.dual_updates_enabled,
            "same_temperatures_for_behavior_and_backup": True,
        }


def _geometric(start: float, finish: float, fraction: float) -> float:
    return float(start * (finish / start) ** fraction)


def temperatures_for_block(block_number: int) -> tuple[tuple[float, ...], float]:
    block = _positive_int(block_number, name="block_number")
    if not 1 <= block <= TOTAL_BLOCKS:
        raise ConstrainedV21TrainerError("block_number lies outside V2.1 clock")
    if block <= ANNEAL_FIRST_BLOCK:
        within = WITHIN_TEMPERATURE_INITIAL
        group = GROUP_TEMPERATURE_INITIAL
    elif block >= ANNEAL_LAST_BLOCK:
        within = WITHIN_TEMPERATURE_FLOOR
        group = GROUP_TEMPERATURE_FLOOR
    else:
        fraction = (block - ANNEAL_FIRST_BLOCK) / (
            ANNEAL_LAST_BLOCK - ANNEAL_FIRST_BLOCK
        )
        within = _geometric(
            WITHIN_TEMPERATURE_INITIAL, WITHIN_TEMPERATURE_FLOOR, fraction
        )
        group = _geometric(
            GROUP_TEMPERATURE_INITIAL, GROUP_TEMPERATURE_FLOOR, fraction
        )
    return (float(within),) * 4, float(group)


def schedule_for_episode(
    episode_number: int, *, validation: bool = False
) -> V21ScheduleState:
    episode = _positive_int(episode_number, name="episode_number")
    if not 1 <= episode <= TOTAL_EPISODES:
        raise ConstrainedV21TrainerError("episode_number lies outside V2.1 clock")
    block = (episode - 1) // BLOCK_EPISODES + 1
    within, group = temperatures_for_block(block)
    if validation:
        if episode % BLOCK_EPISODES:
            raise ConstrainedV21TrainerError("validation is allowed only after a block")
        return V21ScheduleState(
            phase=(
                "critic_warmup"
                if block <= WARMUP_BLOCKS
                else (
                    "primal_dual_anneal"
                    if block <= ANNEAL_LAST_BLOCK
                    else "low_temperature_stabilization"
                )
            ),
            episode_number=episode,
            block_number=block,
            within_group_temperatures=within,
            group_temperature=group,
            policy_mode="map",
            critic_updates_enabled=False,
            dual_updates_enabled=False,
        )
    return V21ScheduleState(
        phase=(
            "critic_warmup"
            if block <= WARMUP_BLOCKS
            else (
                "primal_dual_anneal"
                if block <= ANNEAL_LAST_BLOCK
                else "low_temperature_stabilization"
            )
        ),
        episode_number=episode,
        block_number=block,
        within_group_temperatures=within,
        group_temperature=group,
        policy_mode="regularized_sample",
        critic_updates_enabled=True,
        dual_updates_enabled=block > WARMUP_BLOCKS,
    )


def frozen_schedule_table() -> tuple[dict, ...]:
    rows = []
    for block in range(1, TOTAL_BLOCKS + 1):
        episode = block * BLOCK_EPISODES
        state = schedule_for_episode(episode)
        rows.append(
            {
                "block_number": block,
                "episode_first": (block - 1) * BLOCK_EPISODES + 1,
                "episode_last": episode,
                "phase": state.phase,
                "within_group_temperatures": state.within_group_temperatures,
                "group_temperature": state.group_temperature,
                "dual_update_after_complete_block": (
                    WARMUP_BLOCKS < block < TOTAL_BLOCKS
                ),
                "terminal_dual_proposal_only": block == TOTAL_BLOCKS,
                "development_candidate_temperature_eligible": (
                    block >= ANNEAL_LAST_BLOCK
                ),
            }
        )
    return tuple(rows)


def _validate_seed_namespace(
    train_seed_base: int, episodes: int, validation_seeds: Sequence[int]
) -> None:
    base = _nonnegative_int(train_seed_base, name="train_seed_base")
    count = _positive_int(episodes, name="episodes")
    if count != TOTAL_EPISODES:
        raise ConstrainedV21TrainerError("V2.1 calibration is frozen at 100 episodes")
    if isinstance(validation_seeds, (str, bytes)) or not isinstance(
        validation_seeds, Sequence
    ):
        raise ConstrainedV21TrainerError("validation_seeds must be a sequence")
    validation = tuple(
        _nonnegative_int(seed, name="validation_seed") for seed in validation_seeds
    )
    if not validation or len(validation) != len(set(validation)):
        raise ConstrainedV21TrainerError(
            "validation_seeds must be non-empty and unique"
        )
    training = set(range(base, base + count))
    protected = (training | set(validation)) & set(PROTECTED_PROSPECTIVE_SEEDS)
    if protected:
        raise ConstrainedV21TrainerError(
            f"V2.1 refuses protected prospective 83xxx seeds: {sorted(protected)}"
        )
    overlap = training & set(validation)
    if overlap:
        raise ConstrainedV21TrainerError(
            f"training and validation seeds overlap: {sorted(overlap)}"
        )


def frozen_agent_config(args: argparse.Namespace) -> dict:
    """Return every optimization/architecture field used by V2.1.

    The complete base dataclass is serialized rather than a hand-picked subset
    so changes to an inherited default cannot silently alter a later run.
    """

    from viability_graph_constrained_v2 import ConstrainedV2Config

    return ConstrainedV2Config(
        gamma_operational=0.99,
        gamma_rehandle=1.0,
        operational_reward_scale=0.01,
        episode_horizon_steps=int(args.max_steps),
        max_hold_steps=int(args.max_hold_steps),
        max_idle_steps=int(args.max_idle_steps),
        lambda_initial=float(args.lambda_initial),
        lambda_max=float(args.lambda_max),
        batch_size=16,
        replay_capacity=2_000,
        target_update_every=50,
    ).to_dict()


def build_training_contract(args: argparse.Namespace) -> dict:
    _validate_args(args)
    _validate_seed_namespace(args.train_seed_base, args.episodes, args.validation_seeds)
    dual = V21DualConfig(
        budget_per_100_required_deliveries=args.rehandle_budget_per_100,
        learning_rate=args.dual_lr,
        lambda_initial=args.lambda_initial,
        lambda_max=args.lambda_max,
    )
    contract = {
        "training_protocol": TRAINING_PROTOCOL,
        "method_version": METHOD_VERSION,
        "trainer_schema_version": TRAINER_SCHEMA_VERSION,
        "checkpoint_family": CHECKPOINT_FAMILY,
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "controller": CONTROLLER_ARCHITECTURE,
        "backup_version": BACKUP_VERSION,
        "policy_schedule_protocol": POLICY_SCHEDULE_PROTOCOL,
        "dual_update_protocol": DUAL_UPDATE_PROTOCOL,
        "checkpoint_selection_version": CHECKPOINT_SELECTION_VERSION,
        "cost_definition": COST_DEFINITION,
        "budget_definition": BUDGET_DEFINITION,
        "development_only": True,
        "resumable": False,
        "performance_claim_authorized": False,
        "prospective_83xxx_panel_opened": False,
        "baseline_policy_query": False,
        "baseline_viability_teacher": False,
        "exact_verifier_authoritative": True,
        "no_v1_or_v2_fallback": True,
        "model_seed": int(args.model_seed),
        "train_seed_base": int(args.train_seed_base),
        "episodes": int(args.episodes),
        "validation_seeds": tuple(int(seed) for seed in args.validation_seeds),
        "validation_every": BLOCK_EPISODES,
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
        "viability_search_config": dict(FROZEN_VIABILITY_SEARCH_CONFIG),
        "fresh_certificate_cache_per_episode": True,
        "gamma_operational": 0.99,
        "gamma_rehandle": 1.0,
        "reward_scale": 0.01,
        "agent_config": frozen_agent_config(args),
        "dual": asdict(dual),
        "critic_warmup": {
            "episodes": WARMUP_EPISODES,
            "lambda_frozen": 0.0,
            "critic_updates_enabled": True,
            "residuals_discarded_before_dual_phase": True,
        },
        "two_timescale": {
            "block_episodes": BLOCK_EPISODES,
            "dual_fixed_within_block": True,
            "complete_block_updates_only": True,
            "partial_final_update": False,
        },
        "temperature_schedule": {
            "protocol": POLICY_SCHEDULE_PROTOCOL,
            "same_temperatures_for_behavior_and_backup": True,
            "block_constant": True,
            "table": frozen_schedule_table(),
            "validation_policy": "nested_lagrangian_map",
        },
        "development_candidate_gate": {
            "eligible_blocks": (8, 9, 10),
            "floor_temperature_required": True,
            "strict_integrity_required": True,
            "aggregate_map_budget_required": True,
            "timing_noncollapse_metric": "mean_absolute_error",
            "timing_noncollapse_maximum": TIMING_NONCOLLAPSE_MAE_MAXIMUM,
            "timing_noncollapse_source": "pre_existing_dense_objective_service_window",
            "baseline_reference_used": False,
        },
        "policy_diagnostics": {
            "distribution": POLICY_DIAGNOSTIC_DISTRIBUTION,
            "validation_action_execution": "deterministic_nested_map",
            "nonzero_validation_entropy_implies_random_execution": False,
        },
        "device": str(args.device),
    }
    contract["contract_sha256"] = contract_hash(contract)
    return contract


def summarize_validation(
    runs: Sequence[Mapping],
    config: V21DualConfig,
    *,
    checkpoint_episode: int,
    validation_lambda: float,
    schedule_state: Mapping,
) -> dict:
    if isinstance(runs, (str, bytes)) or not isinstance(runs, Sequence) or not runs:
        raise ConstrainedV21TrainerError("validation runs must be non-empty")
    normalized = [normalize_v2_run(run, index) for index, run in enumerate(runs)]
    total_cost = sum(run["physical_rehandles"] for run in normalized)
    total_required = sum(run["required_deliveries"] for run in normalized)
    cost_rate = 100.0 * total_cost / total_required
    issues = []
    for index, run in enumerate(normalized):
        run_issues = []
        if not run["strict_method_success"]:
            run_issues.append("strict_method_failure")
        if run["delivery_count"] != run["required_deliveries"]:
            run_issues.append("incomplete_required_workload")
        if run["method_failure_reason"] is not None:
            run_issues.append("method_failure_reason_present")
        if not run["all_selected_candidates_exact_safe"]:
            run_issues.append("selected_candidate_not_exact_safe")
        for field in ("illegal_drops", "fallbacks", "witness_mismatches"):
            if run[field]:
                run_issues.append(f"nonzero_{field}")
        if run_issues:
            issues.append({"run_index": index, "issues": run_issues})
    strict_gate = not issues
    budget_gate = cost_rate <= config.budget_per_100_required_deliveries + 1.0e-12
    mean_mae = (
        float(fmean(run["mean_absolute_error"] for run in normalized))
        if all(run["mean_absolute_error"] is not None for run in normalized)
        else None
    )
    block_number = _positive_int(
        schedule_state.get("block_number"), name="validation block_number"
    )
    floor_temperature_gate = block_number >= ANNEAL_LAST_BLOCK
    timing_noncollapse_gate = bool(
        mean_mae is not None and mean_mae <= TIMING_NONCOLLAPSE_MAE_MAXIMUM
    )
    action_counts: Counter[str] = Counter()
    hold_outcome_counts: Counter[str] = Counter()
    total_macro_decisions = 0
    total_policy_diagnostics = 0
    diagnostic_weighted_sums = Counter()
    diagnostic_fields = (
        "mean_outer_policy_entropy",
        "mean_outer_map_probability",
        "mean_selected_within_policy_entropy",
        "mean_selected_within_map_probability",
    )
    for run in normalized:
        action_counts.update(run.get("selected_action_counts", {}))
        hold_outcome_counts.update(run.get("hold_outcome_counts", {}))
        total_macro_decisions += int(run.get("macro_decisions", 0))
        diagnostic_count = int(run.get("policy_diagnostic_decisions", 0))
        total_policy_diagnostics += diagnostic_count
        for field in diagnostic_fields:
            value = run.get(field)
            if value is not None:
                diagnostic_weighted_sums[field] += float(value) * diagnostic_count
    action_shares = {
        name: (count / total_macro_decisions if total_macro_decisions else 0.0)
        for name, count in sorted(action_counts.items())
    }
    aggregate_policy_diagnostics = {
        field: (
            None
            if total_policy_diagnostics == 0
            else float(diagnostic_weighted_sums[field] / total_policy_diagnostics)
        )
        for field in diagnostic_fields
    }
    return {
        "checkpoint_episode": _positive_int(
            checkpoint_episode, name="checkpoint_episode"
        ),
        "run_count": len(normalized),
        "validation_policy": "nested_lagrangian_map",
        "validation_lambda": _finite(validation_lambda, name="validation_lambda"),
        "schedule_state": dict(schedule_state),
        "strict_integrity_gate": strict_gate,
        "aggregate_map_budget_gate": budget_gate,
        "floor_temperature_gate": floor_temperature_gate,
        "timing_noncollapse_gate": timing_noncollapse_gate,
        "timing_noncollapse_mae_maximum": TIMING_NONCOLLAPSE_MAE_MAXIMUM,
        "reference_free_timing_gate": True,
        "development_candidate_eligible": bool(
            strict_gate
            and budget_gate
            and floor_temperature_gate
            and timing_noncollapse_gate
        ),
        "deployment_checkpoint_eligible": False,
        "safety_issues": issues,
        "mean_dense_return": float(fmean(run["dense_return"] for run in normalized)),
        "mean_absolute_error": mean_mae,
        "total_physical_rehandles": int(total_cost),
        "total_required_deliveries": int(total_required),
        "physical_rehandles_per_100_required_deliveries": float(cost_rate),
        "budget_per_100_required_deliveries": float(
            config.budget_per_100_required_deliveries
        ),
        "total_macro_decisions": int(total_macro_decisions),
        "selected_action_counts": dict(sorted(action_counts.items())),
        "selected_action_shares": action_shares,
        "hold_decisions": int(action_counts.get("defer", 0)),
        "hold_decision_share": float(action_shares.get("defer", 0.0)),
        "deliver_decisions": int(action_counts.get("deliver", 0)),
        "deliver_decision_share": float(action_shares.get("deliver", 0.0)),
        "hold_outcome_counts": dict(sorted(hold_outcome_counts.items())),
        "policy_diagnostic_decisions": int(total_policy_diagnostics),
        "policy_diagnostics": aggregate_policy_diagnostics,
        "complete_case_filtering_used": False,
    }


def validation_score(summary: Mapping) -> Optional[tuple[float, ...]]:
    if not bool(summary.get("development_candidate_eligible")):
        return None
    mae = summary.get("mean_absolute_error")
    if mae is None:
        return None
    return (
        float(summary["mean_dense_return"]),
        -float(mae),
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


class V21DevelopmentRuntime(Protocol):
    @property
    def core_contract(self) -> Mapping: ...

    @property
    def dual_lambda(self) -> float: ...

    @property
    def schedule_state(self) -> Mapping: ...

    def set_dual_lambda(self, value: float) -> None: ...

    def set_schedule_state(self, state: Mapping) -> None: ...

    def run_episode(self, *, instance_seed: int, training: bool, max_steps: int) -> Mapping: ...

    def checkpoint_state(self, *, include_replay: bool) -> Mapping: ...


def _validate_runtime(
    runtime: V21DevelopmentRuntime, contract: Optional[Mapping] = None
) -> None:
    for name in (
        "set_dual_lambda",
        "set_schedule_state",
        "run_episode",
        "checkpoint_state",
    ):
        if not callable(getattr(runtime, name, None)):
            raise RuntimeError(f"V2.1 runtime is missing callable {name!r}")
    core = getattr(runtime, "core_contract", None)
    if not isinstance(core, Mapping):
        raise RuntimeError("V2.1 runtime core_contract must be a mapping")
    expected = {
        "checkpoint_family": CHECKPOINT_FAMILY,
        "controller": CONTROLLER_ARCHITECTURE,
        "backup_version": BACKUP_VERSION,
        "policy_schedule_protocol": POLICY_SCHEDULE_PROTOCOL,
        "hold_frontier_interface": CERTIFIED_HOLD_INTERFACE_V2,
        "hold_rule": CERTIFIED_HOLD_RULE_V2,
        "robust_stutter_contract": ROBUST_STUTTER_CERTIFICATION_CONTRACT_V2,
        "hidden_schedule_contract": HIDDEN_SCHEDULE_CONTRACT_V2,
        "base_v1_defer_frontier_allowed": False,
        "cost_definition": COST_DEFINITION,
        "scalarized_reward_stored_in_replay": False,
        "same_temperatures_for_behavior_and_backup": True,
        "training_policy": "block_annealed_nested_regularized_lagrangian_sample",
        "deployment_policy": "nested_lagrangian_map",
        "mixed_discount_policy_merit": (
            "gamma_operational_pow_elapsed_times_qop_minus_lambda_qphys"
        ),
        "operational_gamma": 0.99,
        "rehandle_gamma": 1.0,
        "operational_reward_scale": 0.01,
        "dual_update_authority": "trainer_complete_active_block_residual",
        "exact_safe_frontier_authoritative": True,
        "baseline_teacher": False,
        "baseline_policy_query": False,
        "policy_diagnostic_distribution": POLICY_DIAGNOSTIC_DISTRIBUTION,
        "viability_search_config": dict(FROZEN_VIABILITY_SEARCH_CONFIG),
        "fresh_certificate_cache_per_episode": True,
    }
    mismatches = {
        key: (core.get(key), expected_value)
        for key, expected_value in expected.items()
        if core.get(key) != expected_value
    }
    if mismatches:
        raise RuntimeError(f"V2.1 runtime semantic contract mismatch: {mismatches!r}")
    agent_config = core.get("agent_config")
    if not isinstance(agent_config, Mapping):
        raise RuntimeError("V2.1 runtime must expose the complete agent_config")
    if contract is not None and dict(agent_config) != dict(contract["agent_config"]):
        raise RuntimeError("V2.1 runtime agent_config mismatch")
    if not isinstance(getattr(runtime, "schedule_state", None), Mapping):
        raise RuntimeError("V2.1 runtime schedule_state must be a mapping")
    _assert_runtime_dual(runtime, float(runtime.dual_lambda))


def load_default_runtime(
    args: argparse.Namespace, contract: Mapping
) -> V21DevelopmentRuntime:
    try:
        core = importlib.import_module("viability_graph_constrained_v2_1")
    except ImportError as error:
        raise RuntimeError(
            "isolated V2.1 core is unavailable; no V2/V1 fallback is permitted"
        ) from error
    factory = getattr(core, "build_v2_1_development_runtime", None)
    if not callable(factory):
        raise RuntimeError(
            "viability_graph_constrained_v2_1.build_v2_1_development_runtime "
            "is required; no compatibility fallback is permitted"
        )
    runtime = factory(args=args, contract=dict(contract))
    _validate_runtime(runtime, contract)
    return runtime


def _assert_runtime_dual(runtime: V21DevelopmentRuntime, expected: float) -> None:
    observed = getattr(runtime, "dual_lambda", None)
    if isinstance(observed, bool) or not isinstance(observed, Real):
        raise RuntimeError("V2.1 runtime dual_lambda must be numeric")
    if not math.isclose(float(observed), float(expected), rel_tol=0.0, abs_tol=0.0):
        raise RuntimeError(
            f"V2.1 runtime dual mismatch: observed {observed}, expected {expected}"
        )


def _install_schedule(
    runtime: V21DevelopmentRuntime, state: V21ScheduleState
) -> dict:
    expected = state.to_dict()
    runtime.set_schedule_state(dict(expected))
    observed = _json_safe(dict(runtime.schedule_state))
    if observed != _json_safe(expected):
        raise RuntimeError(
            f"V2.1 runtime schedule mismatch: observed {observed!r}, "
            f"expected {_json_safe(expected)!r}"
        )
    return expected


def _normalize_authenticated_run(
    run: Mapping,
    *,
    index: int,
    schedule: Mapping,
    dual_lambda: float,
) -> dict:
    normalized = normalize_v2_run(run, index)
    expected = {
        "method_version": METHOD_VERSION,
        "policy_schedule_protocol": POLICY_SCHEDULE_PROTOCOL,
        "policy_mode": schedule["policy_mode"],
        "within_group_temperatures": tuple(schedule["within_group_temperatures"]),
        "group_temperature": float(schedule["group_temperature"]),
    }
    for key, value in expected.items():
        observed = run.get(key)
        if key == "within_group_temperatures" and isinstance(observed, list):
            observed = tuple(observed)
        if observed != value:
            raise RuntimeError(
                f"V2.1 run did not authenticate {key}: {observed!r} != {value!r}"
            )
    observed_lambda = _finite(run.get("dual_lambda"), name="run dual_lambda")
    if not math.isclose(observed_lambda, float(dual_lambda), rel_tol=0.0, abs_tol=0.0):
        raise RuntimeError("V2.1 run used a different dual lambda than the trainer")
    normalized.update(expected)
    normalized["dual_lambda"] = observed_lambda
    normalized["schedule_state"] = dict(schedule)
    if schedule["policy_mode"] == "map":
        if run.get("evaluation_learning") is not False:
            raise RuntimeError("V2.1 MAP validation must echo evaluation_learning=False")
        if run.get("training_agent_unchanged") is not True:
            raise RuntimeError("V2.1 MAP validation changed the training agent")
        if run.get("policy_diagnostic_distribution") != POLICY_DIAGNOSTIC_DISTRIBUTION:
            raise RuntimeError("V2.1 MAP diagnostic distribution semantics mismatch")
        macro_decisions = _nonnegative_int(
            run.get("macro_decisions"), name="macro_decisions"
        )
        counts = run.get("selected_action_counts")
        if not isinstance(counts, Mapping):
            raise RuntimeError("V2.1 MAP validation is missing selected_action_counts")
        counts = {
            str(key): _nonnegative_int(value, name=f"selected_action_counts[{key!r}]")
            for key, value in counts.items()
        }
        if sum(counts.values()) != macro_decisions:
            raise RuntimeError("V2.1 MAP action counts do not match macro_decisions")
        hold_counts = run.get("hold_outcome_counts")
        if not isinstance(hold_counts, Mapping):
            raise RuntimeError("V2.1 MAP validation is missing hold_outcome_counts")
        normalized["selected_action_counts"] = counts
        normalized["hold_outcome_counts"] = {
            str(key): _nonnegative_int(value, name=f"hold_outcome_counts[{key!r}]")
            for key, value in hold_counts.items()
        }
        normalized["macro_decisions"] = macro_decisions
        diagnostic_count = _nonnegative_int(
            run.get("policy_diagnostic_decisions"),
            name="policy_diagnostic_decisions",
        )
        if diagnostic_count != macro_decisions:
            raise RuntimeError(
                "V2.1 MAP policy diagnostics do not match macro_decisions"
            )
        normalized["policy_diagnostic_decisions"] = diagnostic_count
        for field in (
            "mean_outer_policy_entropy",
            "mean_outer_map_probability",
            "mean_selected_within_policy_entropy",
            "mean_selected_within_map_probability",
        ):
            value = run.get(field)
            if diagnostic_count == 0 and value is None:
                normalized[field] = None
                continue
            value = _finite(value, name=field)
            if "entropy" in field and value < 0.0:
                raise RuntimeError(f"{field} must be non-negative")
            if "probability" in field and not 0.0 <= value <= 1.0:
                raise RuntimeError(f"{field} must lie in [0, 1]")
            normalized[field] = value
        normalized["evaluation_learning"] = False
        normalized["training_agent_unchanged"] = True
    return normalized


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


def _checkpoint_payload(
    *,
    contract: Mapping,
    runtime: V21DevelopmentRuntime,
    dual: CompleteBlockProjectedDual,
    completed_episodes: int,
    role: str,
    schedule_state: Mapping,
    pending_dual_runs: Sequence[Mapping],
    validation_summary: Optional[Mapping],
    include_replay: bool,
    terminal_dual_proposal: Optional[Mapping] = None,
    training_history: Optional[Sequence[Mapping]] = None,
    validation_history: Optional[Sequence[Mapping]] = None,
) -> dict:
    if role not in ("latest_development_state", "best_development_candidate"):
        raise ConstrainedV21TrainerError("invalid checkpoint role")
    if contract.get("checkpoint_family") != CHECKPOINT_FAMILY:
        raise ConstrainedV21TrainerError("checkpoint family mismatch")
    if contract_hash({k: v for k, v in contract.items() if k != "contract_sha256"}) != contract.get(
        "contract_sha256"
    ):
        raise ConstrainedV21TrainerError("training contract hash mismatch")
    eligible = bool(
        validation_summary is not None
        and validation_summary.get("development_candidate_eligible")
    )
    if role == "best_development_candidate" and not eligible:
        raise ConstrainedV21TrainerError(
            "best development candidate requires eligible MAP validation"
        )
    pending_residuals = tuple(
        episode_budget_residual(
            run["physical_rehandles"], run["required_deliveries"], dual.config
        )
        for run in pending_dual_runs
    )
    runtime_lambda = _finite(runtime.dual_lambda, name="runtime dual_lambda")
    validated_lambda = (
        None
        if validation_summary is None
        else _finite(
            validation_summary["validation_lambda"], name="validation_lambda"
        )
    )
    checkpoint_block = _positive_int(
        schedule_state.get("block_number"), name="checkpoint block_number"
    )
    runtime_lambda_applies_from_episode = (
        int(completed_episodes) + 1
        if (
            validated_lambda is not None
            and not math.isclose(
                runtime_lambda, validated_lambda, rel_tol=0.0, abs_tol=0.0
            )
        )
        else (checkpoint_block - 1) * BLOCK_EPISODES + 1
    )
    payload = {
        "checkpoint_family": CHECKPOINT_FAMILY,
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "checkpoint_role": role,
        "resumable": False,
        "development_candidate_eligible": eligible,
        "deployment_checkpoint_eligible": False,
        "completed_episodes": _nonnegative_int(
            completed_episodes, name="completed_episodes"
        ),
        "training_contract_sha256": contract["contract_sha256"],
        "method_version": METHOD_VERSION,
        "training_protocol": TRAINING_PROTOCOL,
        "backup_version": BACKUP_VERSION,
        "policy_schedule_protocol": POLICY_SCHEDULE_PROTOCOL,
        "dual_update_protocol": DUAL_UPDATE_PROTOCOL,
        "dual_state": dual.state_dict(),
        "runtime_lambda": runtime_lambda,
        "validated_block_lambda": validated_lambda,
        "validated_policy_lambda": validated_lambda,
        "next_block_lambda": runtime_lambda,
        "lambda_applies_from_episode": (
            runtime_lambda_applies_from_episode
            if runtime_lambda_applies_from_episode <= TOTAL_EPISODES
            else None
        ),
        "validation_authenticates_checkpoint_lambda": bool(
            validated_lambda is not None
            and math.isclose(
                runtime_lambda, validated_lambda, rel_tol=0.0, abs_tol=0.0
            )
        ),
        "schedule_state": dict(schedule_state),
        "pending_dual_batch": {
            "episode_count": len(pending_dual_runs),
            "episode_numbers": tuple(
                int(run["training_episode_number"]) for run in pending_dual_runs
            ),
            "residuals": pending_residuals,
            "lambda_frozen_for_batch": float(dual.lambda_value),
            "audit_only_nonresumable": True,
        },
        "terminal_dual_proposal": (
            None
            if terminal_dual_proposal is None
            else dict(terminal_dual_proposal)
        ),
        "validation_summary": (
            None if validation_summary is None else dict(validation_summary)
        ),
        "core_contract": dict(runtime.core_contract),
        "lambda_frozen_within_complete_block": True,
        "same_temperatures_for_behavior_and_backup": True,
        "validation_map_only": True,
        "exact_safe_frontier_authoritative": True,
        "scalarized_reward_stored_in_replay": False,
        "agent_state": runtime.checkpoint_state(include_replay=include_replay),
    }
    if training_history is not None:
        payload["training_history"] = list(training_history)
    if validation_history is not None:
        payload["validation_history"] = list(validation_history)
    return payload


def run_development_calibration(
    args: argparse.Namespace,
    *,
    runtime: Optional[V21DevelopmentRuntime] = None,
) -> dict:
    contract = build_training_contract(args)
    output_dir = Path(args.output_dir).resolve()
    contract_path = output_dir / "training-contract.json"
    if contract_path.exists() or (output_dir.exists() and any(output_dir.iterdir())):
        raise ConstrainedV21TrainerError(
            "fresh V2.1 calibration refuses a nonempty output directory"
        )
    if not args.contract_only:
        runtime = runtime or load_default_runtime(args, contract)
        _validate_runtime(runtime, contract)
    _atomic_json(contract, contract_path)
    if args.contract_only:
        summary = {
            "status": "contract_only",
            "method_version": METHOD_VERSION,
            "episodes_executed": 0,
            "training_contract": str(contract_path),
            "training_contract_sha256": contract["contract_sha256"],
            "prospective_83xxx_panel_opened": False,
            "deployment_checkpoint_eligible": False,
        }
        _atomic_json(summary, output_dir / "training-summary.json")
        return summary

    assert runtime is not None
    dual = CompleteBlockProjectedDual(V21DualConfig(**contract["dual"]))
    runtime.set_dual_lambda(dual.lambda_value)
    _assert_runtime_dual(runtime, dual.lambda_value)
    training_history: list[dict] = []
    validation_history: list[dict] = []
    dual_update_history: list[dict] = []
    pending_dual_runs: list[dict] = []
    terminal_dual_proposal = None
    best_validation = None
    latest_path = output_dir / "latest.pth"
    best_path = output_dir / "best-development-candidate.pth"

    for episode_number in range(1, TOTAL_EPISODES + 1):
        state = schedule_for_episode(episode_number)
        schedule = _install_schedule(runtime, state)
        block_lambda = float(dual.lambda_value)
        runtime.set_dual_lambda(block_lambda)
        _assert_runtime_dual(runtime, block_lambda)
        raw_run = runtime.run_episode(
            instance_seed=int(args.train_seed_base) + episode_number - 1,
            training=True,
            max_steps=int(args.max_steps),
        )
        run = _normalize_authenticated_run(
            raw_run,
            index=episode_number - 1,
            schedule=schedule,
            dual_lambda=block_lambda,
        )
        run["training_episode_number"] = episode_number
        run["block_number"] = state.block_number
        training_history.append(run)
        if state.dual_updates_enabled:
            pending_dual_runs.append(run)

        print(
            "V2.1 "
            f"Ep {episode_number:4d} | "
            f"Phase {state.phase:13s} | "
            f"TrainR {run['dense_return']:8.2f} | "
            f"Strict {int(run['strict_method_success'])} | "
            f"Reh {run['physical_rehandles']:3d}/{run['required_deliveries']:3d} | "
            f"Lambda {block_lambda:7.4f} | "
            f"TauM {state.group_temperature:6.4f} | "
            f"TauC {state.within_group_temperatures[0]:6.4f}",
            flush=True,
        )

        validation_summary = None
        checkpoint_schedule = schedule
        if episode_number % BLOCK_EPISODES == 0:
            validation_state = schedule_for_episode(episode_number, validation=True)
            checkpoint_schedule = _install_schedule(runtime, validation_state)
            # Authenticate the policy that generated this complete block.
            # The dual proposal is computed only after this MAP validation and
            # can therefore affect the next block, never the current candidate.
            validation_lambda = block_lambda
            runtime.set_dual_lambda(validation_lambda)
            _assert_runtime_dual(runtime, validation_lambda)
            validation_runs = [
                _normalize_authenticated_run(
                    runtime.run_episode(
                        instance_seed=int(seed),
                        training=False,
                        max_steps=int(args.max_steps),
                    ),
                    index=index,
                    schedule=checkpoint_schedule,
                    dual_lambda=validation_lambda,
                )
                for index, seed in enumerate(args.validation_seeds)
            ]
            validation_summary = summarize_validation(
                validation_runs,
                dual.config,
                checkpoint_episode=episode_number,
                validation_lambda=validation_lambda,
                schedule_state=checkpoint_schedule,
            )
            validation_summary["validated_before_dual_update"] = True
            validation_summary["validated_training_block_lambda"] = block_lambda
            validation_history.append(validation_summary)
            mae = validation_summary["mean_absolute_error"]
            mae_text = "undefined" if mae is None else f"{float(mae):7.2f}"
            diagnostics = validation_summary["policy_diagnostics"]
            outer_probability = diagnostics["mean_outer_map_probability"]
            within_probability = diagnostics[
                "mean_selected_within_map_probability"
            ]
            outer_text = (
                "undefined"
                if outer_probability is None
                else f"{float(outer_probability):5.3f}"
            )
            within_text = (
                "undefined"
                if within_probability is None
                else f"{float(within_probability):5.3f}"
            )
            print(
                "V2.1 "
                f"Validation Ep {episode_number:4d} | "
                f"R {validation_summary['mean_dense_return']:8.2f} | "
                f"MAE {mae_text} | "
                f"Strict {int(validation_summary['strict_integrity_gate'])} | "
                "Reh/100 "
                f"{validation_summary['physical_rehandles_per_100_required_deliveries']:7.2f} | "
                f"Budget {int(validation_summary['aggregate_map_budget_gate'])} | "
                f"Hold {100.0 * validation_summary['hold_decision_share']:5.1f}% | "
                f"Deliver {100.0 * validation_summary['deliver_decision_share']:5.1f}% | "
                f"OpP {outer_text} | InP {within_text} | "
                f"Eligible {int(validation_summary['development_candidate_eligible'])}",
                flush=True,
            )
            selected = select_better_validation(best_validation, validation_summary)
            if selected is validation_summary:
                best_validation = validation_summary
                _atomic_torch(
                    _checkpoint_payload(
                        contract=contract,
                        runtime=runtime,
                        dual=dual,
                        completed_episodes=episode_number,
                        role="best_development_candidate",
                        schedule_state=checkpoint_schedule,
                        pending_dual_runs=pending_dual_runs,
                        validation_summary=validation_summary,
                        include_replay=False,
                    ),
                    best_path,
                )

            if state.dual_updates_enabled:
                if state.block_number == TOTAL_BLOCKS:
                    proposal = dual.propose_complete_block(
                        pending_dual_runs, block_number=state.block_number
                    )
                    terminal_dual_proposal = {
                        **proposal,
                        "applied": False,
                        "reason_not_applied": (
                            "no_subsequent_primal_block_to_authenticate_proposal"
                        ),
                    }
                    dual_update_history.append(terminal_dual_proposal)
                else:
                    update = dual.update_complete_block(
                        pending_dual_runs, block_number=state.block_number
                    )
                    update = {**update, "applied": True}
                    dual_update_history.append(update)
                    runtime.set_dual_lambda(dual.lambda_value)
                    _assert_runtime_dual(runtime, dual.lambda_value)
                pending_dual_runs.clear()
            elif pending_dual_runs:
                raise RuntimeError("warm-up residuals leaked into the dual batch")

        _atomic_torch(
            _checkpoint_payload(
                contract=contract,
                runtime=runtime,
                dual=dual,
                completed_episodes=episode_number,
                role="latest_development_state",
                schedule_state=checkpoint_schedule,
                pending_dual_runs=pending_dual_runs,
                validation_summary=validation_summary,
                include_replay=True,
                terminal_dual_proposal=terminal_dual_proposal,
                training_history=training_history,
                validation_history=validation_history,
            ),
            latest_path,
        )

    if pending_dual_runs:
        raise RuntimeError("V2.1 ended with an incomplete dual block")
    summary = {
        "status": "complete",
        "method_version": METHOD_VERSION,
        "completed_training_episodes": len(training_history),
        "completed_blocks": TOTAL_BLOCKS,
        "critic_warmup_episodes": WARMUP_EPISODES,
        "active_primal_dual_episodes": TOTAL_EPISODES - WARMUP_EPISODES,
        "dual_state": dual.state_dict(),
        "dual_update_history": dual_update_history,
        "terminal_dual_proposal": terminal_dual_proposal,
        "development_candidate_eligible": best_validation is not None,
        "deployment_checkpoint_eligible": False,
        "best_validation": best_validation,
        "validation_history": validation_history,
        "best_development_candidate": str(best_path) if best_path.exists() else None,
        "latest_checkpoint": str(latest_path),
        "training_contract": str(contract_path),
        "prospective_83xxx_panel_opened": False,
    }
    _atomic_json(summary, output_dir / "training-summary.json")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the frozen 100-episode constrained VCG V2.1 calibration"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=TOTAL_EPISODES)
    parser.add_argument("--train-seed-base", type=int, default=FRESH_TRAIN_SEED_BASE)
    parser.add_argument(
        "--validation-seeds", type=int, nargs="+", default=DEFAULT_VALIDATION_SEEDS
    )
    parser.add_argument("--validation-every", type=int, default=BLOCK_EPISODES)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--grid-rows", type=int, default=5)
    parser.add_argument("--grid-cols", type=int, default=5)
    parser.add_argument("--number-blocks", type=int, default=2)
    parser.add_argument("--arrival-rate", type=float, default=10.0)
    parser.add_argument("--proc-mean", type=int, default=80)
    parser.add_argument("--gamma-operational", type=float, default=0.99)
    parser.add_argument("--reward-scale", type=float, default=0.01)
    parser.add_argument("--rehandle-budget-per-100", type=float, default=20.0)
    parser.add_argument("--dual-lr", type=float, default=0.01)
    parser.add_argument("--lambda-initial", type=float, default=0.0)
    parser.add_argument("--lambda-max", type=float, default=20.0)
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
        "max_hold_steps",
        "max_idle_steps",
    ):
        _positive_int(getattr(args, name), name=name)
    if args.episodes != TOTAL_EPISODES:
        raise ConstrainedV21TrainerError("V2.1 calibration requires 100 episodes")
    if args.validation_every != BLOCK_EPISODES:
        raise ConstrainedV21TrainerError("V2.1 validation must occur every 10 episodes")
    if _finite(args.gamma_operational, name="gamma_operational") != 0.99:
        raise ConstrainedV21TrainerError("V2.1 requires gamma_operational=0.99")
    if _finite(args.reward_scale, name="reward_scale") != 0.01:
        raise ConstrainedV21TrainerError("V2.1 requires reward_scale=0.01")
    if _finite(args.arrival_rate, name="arrival_rate") <= 0.0:
        raise ConstrainedV21TrainerError("arrival_rate must be positive")
    V21DualConfig(
        budget_per_100_required_deliveries=args.rehandle_budget_per_100,
        learning_rate=args.dual_lr,
        lambda_initial=args.lambda_initial,
        lambda_max=args.lambda_max,
    )


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    summary = run_development_calibration(args)
    print(json.dumps(_json_safe(summary), indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()


__all__ = [
    "BACKUP_VERSION",
    "CHECKPOINT_FAMILY",
    "CONTROLLER_ARCHITECTURE",
    "DUAL_UPDATE_PROTOCOL",
    "METHOD_VERSION",
    "POLICY_SCHEDULE_PROTOCOL",
    "TRAINING_PROTOCOL",
    "CompleteBlockProjectedDual",
    "ConstrainedV21TrainerError",
    "V21DevelopmentRuntime",
    "V21DualConfig",
    "V21ScheduleState",
    "build_parser",
    "build_training_contract",
    "episode_budget_residual",
    "frozen_schedule_table",
    "load_default_runtime",
    "run_development_calibration",
    "schedule_for_episode",
    "select_better_validation",
    "summarize_validation",
    "temperatures_for_block",
    "validation_score",
]
