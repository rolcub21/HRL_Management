#!/usr/bin/env python3
"""Prospective policy/operator-consistent development calibration for VCG V2.2.

V2.2 keeps V2.1's exact SAFE frontier, vector critic, dense operational
objective, raw physical-rehandle critic, two-timescale dual, and blockwise
temperature clock.  Its one algorithmic change is deliberately narrow: the
same induced nested stochastic Lagrangian policy is used for behavior,
Bellman continuation, validation, and deployment.  Deterministic MAP is not a
V2.2 policy and is never used to select a development checkpoint.

Validation estimates the expected constrained-policy cost on 12 fixed
EpisodeInstances, each crossed with four controller-only action RNG streams.
Every row uses a fresh, frozen, weight-only clone.  The rehandle gate is the
predeclared Bonferroni-adjusted, model-based Student-t bound over the 12
EpisodeInstance cluster means; it is a development/FWER screen rather than a
distribution-free guarantee.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
import importlib
import math
from numbers import Integral, Real
from pathlib import Path
from statistics import fmean, stdev
from typing import Mapping, Optional, Protocol, Sequence

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
import train_vcg_constrained_v2_1 as v21t


TRAINING_PROTOCOL = (
    "vcg_constrained_vector_smdp_v2_2_policy_operator_consistent_"
    "development_calibration_v1"
)
METHOD_VERSION = "vcg_constrained_v2_2"
TRAINER_SCHEMA_VERSION = 1
CHECKPOINT_FAMILY = "vcg_constrained_vector_smdp_v2_2"
CHECKPOINT_SCHEMA_VERSION = 1
CONTROLLER_ARCHITECTURE = (
    "exact_safe_shared_graph_vector_q_primal_dual_policy_consistent_v2_2"
)
BACKUP_VERSION = (
    "elapsed_weighted_shared_annealed_lagrangian_policy_vector_expected_double_q_"
    "two_discount_smdp_v2_2"
)
POLICY_SCHEDULE_PROTOCOL = (
    "block_constant_shared_behavior_backup_validation_deployment_stochastic_v1"
)
DUAL_UPDATE_PROTOCOL = (
    "projected_complete_active_block_workload_residual_ascent_v2_2"
)
CHECKPOINT_SELECTION_VERSION = (
    "strict_integrity_expected_cost_ucb_dense_return_mae_rehandle_earlier_v2_2"
)
POLICY_DIAGNOSTIC_DISTRIBUTION = (
    "executed_induced_nested_regularized_lagrangian_distribution_v2_2"
)
POLICY_REALIZATION = "induced_nested_regularized_lagrangian_sample"
VALIDATION_PROTOCOL = (
    "fixed_instance_crossed_fixed_action_rng_fresh_frozen_clone_expected_cost_v1"
)

FROZEN_VIABILITY_SEARCH_CONFIG = dict(v21t.FROZEN_VIABILITY_SEARCH_CONFIG)
TOTAL_EPISODES = 200
BLOCK_EPISODES = v21t.BLOCK_EPISODES
WARMUP_EPISODES = v21t.WARMUP_EPISODES
WARMUP_BLOCKS = v21t.WARMUP_BLOCKS
TOTAL_BLOCKS = TOTAL_EPISODES // BLOCK_EPISODES
ANNEAL_FIRST_BLOCK = v21t.ANNEAL_FIRST_BLOCK
ANNEAL_LAST_BLOCK = v21t.ANNEAL_LAST_BLOCK
WITHIN_TEMPERATURE_INITIAL = v21t.WITHIN_TEMPERATURE_INITIAL
WITHIN_TEMPERATURE_FLOOR = v21t.WITHIN_TEMPERATURE_FLOOR
GROUP_TEMPERATURE_INITIAL = v21t.GROUP_TEMPERATURE_INITIAL
GROUP_TEMPERATURE_FLOOR = v21t.GROUP_TEMPERATURE_FLOOR
TIMING_NONCOLLAPSE_MAE_MAXIMUM = float(FROZEN_OBJECTIVE_SPEC.window)

FRESH_TRAIN_SEED_BASE = 61_000_000
DEFAULT_MODEL_SEED = 10
DEFAULT_VALIDATION_SEEDS = tuple(range(85_000, 85_012))
VALIDATION_EVERY = 20
CANDIDATE_EPISODES = (80, 100, 120, 140, 160, 180, 200)
TRAINING_POLICY_RNG_BASE = 610_000_000
REPLAY_RNG_SEED = 610_100_010
VALIDATION_POLICY_RNG_BASE = 620_000_000
VALIDATION_POLICY_RNG_COUNT = 4
FINAL_PANEL_SEEDS = frozenset(range(86_000, 86_030))
FINAL_POLICY_RNG_BASE = 622_000_000
FINAL_POLICY_RNG_STOP = FINAL_POLICY_RNG_BASE + 30 * 4
# Bonferroni-adjusted one-sided .05/7 Student-t quantile, df=11.
VALIDATION_COST_T_95 = 2.906203359932373


class ConstrainedV22TrainerError(ValueError):
    """Raised when the frozen V2.2 protocol is violated."""


def _finite(value, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ConstrainedV22TrainerError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ConstrainedV22TrainerError(f"{name} must be finite")
    return result


def _nonnegative_int(value, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ConstrainedV22TrainerError(f"{name} must be an integer")
    result = int(value)
    if result < 0:
        raise ConstrainedV22TrainerError(f"{name} must be non-negative")
    return result


def _positive_int(value, *, name: str) -> int:
    result = _nonnegative_int(value, name=name)
    if result <= 0:
        raise ConstrainedV22TrainerError(f"{name} must be positive")
    return result


def _json_safe(value):
    return v21t._json_safe(value)


def contract_hash(contract: Mapping) -> str:
    return v21t.contract_hash(contract)


def training_policy_rng_seed(episode_number: int) -> int:
    episode = _positive_int(episode_number, name="training episode number")
    if episode > TOTAL_EPISODES:
        raise ConstrainedV22TrainerError("training episode is outside the frozen clock")
    return TRAINING_POLICY_RNG_BASE + episode - 1


def validation_policy_rng_seed(instance_index: int, rng_index: int) -> int:
    instance_index = _nonnegative_int(instance_index, name="validation instance index")
    rng_index = _nonnegative_int(rng_index, name="validation policy RNG index")
    if instance_index >= len(DEFAULT_VALIDATION_SEEDS) or rng_index >= VALIDATION_POLICY_RNG_COUNT:
        raise ConstrainedV22TrainerError("validation RNG coordinates are out of range")
    return VALIDATION_POLICY_RNG_BASE + VALIDATION_POLICY_RNG_COUNT * instance_index + rng_index


@dataclass(frozen=True)
class V22DualConfig:
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
        if budget < 0.0:
            raise ConstrainedV22TrainerError("budget must be non-negative")
        if _finite(self.learning_rate, name="dual learning_rate") != 0.01:
            raise ConstrainedV22TrainerError("V2.2 requires dual learning_rate=0.01")
        if _finite(self.lambda_initial, name="lambda_initial") != 0.0:
            raise ConstrainedV22TrainerError("V2.2 warm-up requires lambda_initial=0")
        if _finite(self.lambda_max, name="lambda_max") != 20.0:
            raise ConstrainedV22TrainerError("V2.2 requires lambda_max=20")
        if self.warmup_episodes != WARMUP_EPISODES:
            raise ConstrainedV22TrainerError("V2.2 warm-up must be 20 episodes")
        if self.block_episodes != BLOCK_EPISODES:
            raise ConstrainedV22TrainerError("V2.2 dual blocks must be 10 episodes")

    @property
    def budget_per_required_delivery(self) -> float:
        return float(self.budget_per_100_required_deliveries) / 100.0


def episode_budget_residual(
    physical_rehandles: int,
    required_deliveries: int,
    config: V22DualConfig,
) -> float:
    if not isinstance(config, V22DualConfig):
        raise ConstrainedV22TrainerError("V2.2 residual requires V22DualConfig")
    cost = _nonnegative_int(physical_rehandles, name="physical_rehandles")
    workload = _positive_int(required_deliveries, name="required_deliveries")
    return float(cost - config.budget_per_required_delivery * workload)


@dataclass
class CompleteBlockProjectedDual:
    config: V22DualConfig
    lambda_value: float = 0.0
    update_count: int = 0
    active_episodes_consumed: int = 0
    last_completed_block: int = WARMUP_BLOCKS
    last_mean_residual: Optional[float] = None
    saturation_count: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.config, V22DualConfig):
            raise ConstrainedV22TrainerError("dual config must be V22DualConfig")
        self.lambda_value = _finite(self.lambda_value, name="lambda_value")
        if not 0.0 <= self.lambda_value <= self.config.lambda_max:
            raise ConstrainedV22TrainerError("lambda_value is out of range")

    def propose_complete_block(
        self, episodes: Sequence[Mapping], *, block_number: int
    ) -> dict:
        block = _positive_int(block_number, name="block_number")
        if block != self.last_completed_block + 1:
            raise ConstrainedV22TrainerError("dual update is out of block order")
        if block <= WARMUP_BLOCKS:
            raise ConstrainedV22TrainerError("warm-up residuals cannot update lambda")
        if len(episodes) != self.config.block_episodes:
            raise ConstrainedV22TrainerError("dual requires one complete block")
        residuals = tuple(
            episode_budget_residual(
                run["physical_rehandles"], run["required_deliveries"], self.config
            )
            for run in episodes
        )
        before = float(self.lambda_value)
        mean_residual = float(fmean(residuals))
        unprojected = before + self.config.learning_rate * mean_residual
        after = min(max(unprojected, 0.0), self.config.lambda_max)
        saturated = bool(after == self.config.lambda_max and mean_residual > 0.0)
        return {
            "protocol": DUAL_UPDATE_PROTOCOL,
            "block_number": block,
            "lambda_before": before,
            "lambda_after": float(after),
            "mean_budget_residual": mean_residual,
            "episode_residuals": residuals,
            "episode_count": len(residuals),
            "unprojected_lambda": float(unprojected),
            "projected": not math.isclose(after, unprojected, abs_tol=1e-15),
            "saturated_with_positive_residual": saturated,
        }

    def update_complete_block(
        self, episodes: Sequence[Mapping], *, block_number: int
    ) -> dict:
        proposal = self.propose_complete_block(episodes, block_number=block_number)
        if int(proposal["block_number"]) >= TOTAL_BLOCKS:
            raise ConstrainedV22TrainerError("terminal proposal cannot be applied")
        self.lambda_value = float(proposal["lambda_after"])
        self.update_count += 1
        self.active_episodes_consumed += len(episodes)
        self.last_completed_block = int(proposal["block_number"])
        self.last_mean_residual = float(proposal["mean_budget_residual"])
        self.saturation_count += int(proposal["saturated_with_positive_residual"])
        return proposal

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
class V22ScheduleState:
    phase: str
    episode_number: int
    block_number: int
    within_group_temperatures: tuple[float, float, float, float]
    group_temperature: float
    execution_context: str
    critic_updates_enabled: bool
    dual_updates_enabled: bool

    def __post_init__(self) -> None:
        if self.execution_context not in ("training", "validation_deployment"):
            raise ConstrainedV22TrainerError("invalid V2.2 execution context")
        episode = _positive_int(self.episode_number, name="episode_number")
        block = _positive_int(self.block_number, name="block_number")
        if not 1 <= episode <= TOTAL_EPISODES:
            raise ConstrainedV22TrainerError("episode is outside V2.2 clock")
        if block != (episode - 1) // BLOCK_EPISODES + 1:
            raise ConstrainedV22TrainerError("block disagrees with episode clock")
        expected_phase = (
            "critic_warmup" if block <= WARMUP_BLOCKS else
            "primal_dual_anneal" if block <= ANNEAL_LAST_BLOCK else
            "low_temperature_stabilization"
        )
        if self.phase != expected_phase:
            raise ConstrainedV22TrainerError("phase disagrees with block clock")
        if len(self.within_group_temperatures) != 4:
            raise ConstrainedV22TrainerError("four within-group temperatures required")
        for value in (*self.within_group_temperatures, self.group_temperature):
            if _finite(value, name="temperature") <= 0.0:
                raise ConstrainedV22TrainerError("temperatures must be positive")
        if self.execution_context == "training":
            if not self.critic_updates_enabled:
                raise ConstrainedV22TrainerError("training must update the critic")
            if self.dual_updates_enabled != (block > WARMUP_BLOCKS):
                raise ConstrainedV22TrainerError("dual flag disagrees with block clock")
        elif self.critic_updates_enabled or self.dual_updates_enabled:
            raise ConstrainedV22TrainerError("validation/deployment cannot update")

    def to_dict(self) -> dict:
        return {
            "protocol": POLICY_SCHEDULE_PROTOCOL,
            "phase": self.phase,
            "episode_number": int(self.episode_number),
            "block_number": int(self.block_number),
            "within_group_temperatures": tuple(map(float, self.within_group_temperatures)),
            "group_temperature": float(self.group_temperature),
            "policy_mode": POLICY_REALIZATION,
            "execution_context": self.execution_context,
            "critic_updates_enabled": self.critic_updates_enabled,
            "dual_updates_enabled": self.dual_updates_enabled,
            "same_policy_for_behavior_backup_validation_deployment": True,
            "map_policy_available": False,
        }


def temperatures_for_block(block_number: int) -> tuple[tuple[float, ...], float]:
    block = _positive_int(block_number, name="block_number")
    if not 1 <= block <= TOTAL_BLOCKS:
        raise ConstrainedV22TrainerError("block is outside the V2.2 clock")
    if block <= ANNEAL_FIRST_BLOCK:
        within, group = WITHIN_TEMPERATURE_INITIAL, GROUP_TEMPERATURE_INITIAL
    elif block >= ANNEAL_LAST_BLOCK:
        within, group = WITHIN_TEMPERATURE_FLOOR, GROUP_TEMPERATURE_FLOOR
    else:
        fraction = (block - ANNEAL_FIRST_BLOCK) / (ANNEAL_LAST_BLOCK - ANNEAL_FIRST_BLOCK)
        within = WITHIN_TEMPERATURE_INITIAL * (
            WITHIN_TEMPERATURE_FLOOR / WITHIN_TEMPERATURE_INITIAL
        ) ** fraction
        group = GROUP_TEMPERATURE_INITIAL * (
            GROUP_TEMPERATURE_FLOOR / GROUP_TEMPERATURE_INITIAL
        ) ** fraction
    return (float(within),) * 4, float(group)


def schedule_for_episode(
    episode_number: int, *, validation: bool = False
) -> V22ScheduleState:
    episode = _positive_int(episode_number, name="episode_number")
    if not 1 <= episode <= TOTAL_EPISODES:
        raise ConstrainedV22TrainerError("episode is outside V2.2 clock")
    if validation and episode % VALIDATION_EVERY:
        raise ConstrainedV22TrainerError("validation is allowed only at the frozen 20-episode cadence")
    block = (episode - 1) // BLOCK_EPISODES + 1
    within, group = temperatures_for_block(block)
    phase = (
        "critic_warmup" if block <= WARMUP_BLOCKS else
        "primal_dual_anneal" if block <= ANNEAL_LAST_BLOCK else
        "low_temperature_stabilization"
    )
    return V22ScheduleState(
        phase=phase,
        episode_number=episode,
        block_number=block,
        within_group_temperatures=tuple(within),
        group_temperature=float(group),
        execution_context="validation_deployment" if validation else "training",
        critic_updates_enabled=not validation,
        dual_updates_enabled=bool(not validation and block > WARMUP_BLOCKS),
    )


def frozen_schedule_table() -> tuple[dict, ...]:
    rows = []
    for block in range(1, TOTAL_BLOCKS + 1):
        state = schedule_for_episode(block * BLOCK_EPISODES)
        rows.append({
            "block_number": block,
            "episode_first": (block - 1) * BLOCK_EPISODES + 1,
            "episode_last": block * BLOCK_EPISODES,
            "phase": state.phase,
            "within_group_temperatures": state.within_group_temperatures,
            "group_temperature": state.group_temperature,
            "dual_update_after_complete_block": WARMUP_BLOCKS < block < TOTAL_BLOCKS,
            "terminal_dual_proposal_only": block == TOTAL_BLOCKS,
            "development_candidate_eligible_look": block * BLOCK_EPISODES in CANDIDATE_EPISODES,
        })
    return tuple(rows)


def _validate_seed_namespace(
    train_seed_base: int, episodes: int, validation_seeds: Sequence[int]
) -> None:
    base = _nonnegative_int(train_seed_base, name="train_seed_base")
    if base != FRESH_TRAIN_SEED_BASE:
        raise ConstrainedV22TrainerError("V2.2 training seed base must be 61000000")
    if _positive_int(episodes, name="episodes") != TOTAL_EPISODES:
        raise ConstrainedV22TrainerError("V2.2 is frozen at 200 episodes")
    validation = tuple(_nonnegative_int(seed, name="validation_seed") for seed in validation_seeds)
    if validation != DEFAULT_VALIDATION_SEEDS:
        raise ConstrainedV22TrainerError("V2.2 requires the frozen 85000..85011 validation panel")
    training = set(range(base, base + episodes))
    if (training | set(validation)) & (set(v21t.PROTECTED_PROSPECTIVE_SEEDS) | set(FINAL_PANEL_SEEDS)):
        raise ConstrainedV22TrainerError("V2.2 refuses protected 83xxx/final 86xxx seeds")
    if training & set(validation):
        raise ConstrainedV22TrainerError("training and validation seeds overlap")


def frozen_agent_config(args: argparse.Namespace) -> dict:
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
    dual = V22DualConfig(
        budget_per_100_required_deliveries=args.rehandle_budget_per_100,
        learning_rate=args.dual_lr,
        lambda_initial=args.lambda_initial,
        lambda_max=args.lambda_max,
    )
    nuisance_seeds = tuple(
        tuple(validation_policy_rng_seed(i, j) for j in range(VALIDATION_POLICY_RNG_COUNT))
        for i in range(len(DEFAULT_VALIDATION_SEEDS))
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
        "validation_protocol": VALIDATION_PROTOCOL,
        "checkpoint_selection_version": CHECKPOINT_SELECTION_VERSION,
        "cost_definition": COST_DEFINITION,
        "budget_definition": BUDGET_DEFINITION,
        "development_only": True,
        "resumable": False,
        "performance_claim_authorized": False,
        "prospective_83xxx_panel_opened": False,
        "prior_83xxx_usage": "83000..83059_were_used_by_prior_work",
        "83xxx_globally_untouched_claimed": False,
        "final_86xxx_panel_opened": False,
        "baseline_policy_query": False,
        "baseline_viability_teacher": False,
        "exact_verifier_authoritative": True,
        "no_v1_v2_v2_1_fallback": True,
        "model_seed": int(args.model_seed),
        "train_seed_base": int(args.train_seed_base),
        "episodes": int(args.episodes),
        "validation_seeds": tuple(map(int, args.validation_seeds)),
        "validation_every": VALIDATION_EVERY,
        "training_policy_rng_formula": "610000000 + episode_number - 1",
        "replay_rng_seed": REPLAY_RNG_SEED,
        "validation_policy_rng_grid": nuisance_seeds,
        "validation_policy_rng_scope": "action_sampling_only",
        "validation_rows_per_checkpoint": len(args.validation_seeds) * VALIDATION_POLICY_RNG_COUNT,
        "max_steps": int(args.max_steps),
        "environment": {
            "grid_rows": int(args.grid_rows), "grid_cols": int(args.grid_cols),
            "number_blocks": int(args.number_blocks), "arrival_rate": float(args.arrival_rate),
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
            "episodes": WARMUP_EPISODES, "lambda_frozen": 0.0,
            "critic_updates_enabled": True, "residuals_discarded_before_dual_phase": True,
        },
        "two_timescale": {
            "block_episodes": BLOCK_EPISODES, "dual_fixed_within_block": True,
            "complete_block_updates_only": True, "partial_final_update": False,
        },
        "temperature_schedule": {
            "protocol": POLICY_SCHEDULE_PROTOCOL,
            "same_policy_for_behavior_backup_validation_deployment": True,
            "block_constant": True, "table": frozen_schedule_table(),
            "validation_policy": POLICY_REALIZATION,
            "deployment_policy": POLICY_REALIZATION,
            "map_policy_available": False,
        },
        "validation_design": {
            "protocol": VALIDATION_PROTOCOL,
            "fixed_episode_instances": True,
            "crossed_nuisance_rng_panel": True,
            "fresh_weight_only_clone_per_row": True,
            "evaluation_learning": False,
            "nuisance_seeds_per_instance": VALIDATION_POLICY_RNG_COUNT,
            "expected_cost_estimator": "per_instance_cluster_rate_averaged_over_four_action_rngs",
            "budget_gate": "bonferroni_one_sided_95_student_t_ucb_over_12_episode_instances",
            "one_sided_t_quantile": VALIDATION_COST_T_95,
            "statistical_unit": "EpisodeInstance_cluster",
            "candidate_looks": CANDIDATE_EPISODES,
            "final_panel_instance_seeds": tuple(sorted(FINAL_PANEL_SEEDS)),
            "final_panel_action_rng_range": (FINAL_POLICY_RNG_BASE, FINAL_POLICY_RNG_STOP - 1),
            "final_panel_opened": False,
        },
        "development_candidate_gate": {
            "eligible_episodes": CANDIDATE_EPISODES, "floor_temperature_required": True,
            "strict_integrity_required": True,
            "expected_cost_point_budget_required": True,
            "expected_cost_ucb_budget_required": True,
            "timing_noncollapse_metric": "mean_absolute_error",
            "timing_noncollapse_maximum": TIMING_NONCOLLAPSE_MAE_MAXIMUM,
            "no_positive_residual_dual_saturation_required": True,
            "baseline_reference_used": False,
        },
        "device": str(args.device),
    }
    contract["contract_sha256"] = contract_hash(contract)
    return contract


def _sample_ucb(values: Sequence[float]) -> tuple[float, float, float]:
    values = tuple(float(value) for value in values)
    if len(values) != len(DEFAULT_VALIDATION_SEEDS):
        raise ConstrainedV22TrainerError("UCB requires all 12 EpisodeInstance clusters")
    mean = float(fmean(values))
    standard_error = float(stdev(values) / math.sqrt(len(values)))
    return mean, standard_error, float(mean + VALIDATION_COST_T_95 * standard_error)


def summarize_validation(
    runs: Sequence[Mapping], config: V22DualConfig, *, checkpoint_episode: int,
    validation_lambda: float, schedule_state: Mapping,
    validation_seeds: Sequence[int] = DEFAULT_VALIDATION_SEEDS,
    positive_residual_saturation_count: int = 0,
) -> dict:
    expected_rows = len(validation_seeds) * VALIDATION_POLICY_RNG_COUNT
    if len(runs) != expected_rows:
        raise ConstrainedV22TrainerError(f"validation requires exactly {expected_rows} rows")
    normalized = [normalize_v2_run(run, i) for i, run in enumerate(runs)]
    by_key = {}
    identities_by_seed = {}
    issues = []
    action_counts: Counter[str] = Counter()
    hold_counts: Counter[str] = Counter()
    diagnostic_sums: Counter[str] = Counter()
    diagnostic_count = 0
    fields = (
        "mean_outer_policy_entropy", "mean_outer_map_probability",
        "mean_selected_within_policy_entropy", "mean_selected_within_map_probability",
    )
    for row_index, (raw, run) in enumerate(zip(runs, normalized)):
        instance_seed = _nonnegative_int(raw.get("instance_seed"), name="instance_seed")
        expected_instance_index = tuple(validation_seeds).index(instance_seed) if instance_seed in validation_seeds else None
        if raw.get("instance_index") != expected_instance_index:
            raise ConstrainedV22TrainerError("validation instance index/seed mismatch")
        rng_index = _nonnegative_int(raw.get("policy_rng_index"), name="policy_rng_index")
        rng_seed = _nonnegative_int(raw.get("policy_rng_seed"), name="policy_rng_seed")
        key = (instance_seed, rng_index)
        if instance_seed not in validation_seeds or key in by_key:
            raise ConstrainedV22TrainerError("validation row identity is invalid or duplicated")
        instance_index = tuple(validation_seeds).index(instance_seed)
        if rng_seed != validation_policy_rng_seed(instance_index, rng_index):
            raise ConstrainedV22TrainerError("validation action RNG is outside frozen panel")
        by_key[key] = run
        identity = (
            raw.get("episode_instance_id"), raw.get("schedule_id"),
            raw.get("episode_instance_sha256"),
        )
        if not all(isinstance(value, str) and value for value in identity[:2]):
            raise ConstrainedV22TrainerError("validation row is missing instance identity")
        if not isinstance(identity[2], str) or len(identity[2]) != 64 or any(
            char not in "0123456789abcdef" for char in identity[2]
        ):
            raise ConstrainedV22TrainerError("validation row has invalid canonical instance SHA")
        incumbent_identity = identities_by_seed.setdefault(instance_seed, identity)
        if incumbent_identity != identity:
            raise ConstrainedV22TrainerError("EpisodeInstance identity drifted within its action-RNG cluster")
        row_issues = []
        if not run["strict_method_success"]: row_issues.append("strict_method_failure")
        if run["delivery_count"] != run["required_deliveries"]: row_issues.append("incomplete")
        if run["method_failure_reason"] is not None: row_issues.append("method_failure")
        if not run["all_selected_candidates_exact_safe"]: row_issues.append("not_exact_safe")
        for field in ("illegal_drops", "fallbacks", "witness_mismatches"):
            if run[field]: row_issues.append(f"nonzero_{field}")
        if raw.get("evaluation_learning") is not False: row_issues.append("evaluation_learning")
        if raw.get("training_agent_unchanged") is not True: row_issues.append("training_agent_changed")
        if raw.get("validation_batch_state_unchanged") is not True:
            row_issues.append("validation_batch_state_changed_or_unauthenticated")
        if raw.get("fresh_evaluation_clone") is not True: row_issues.append("clone_not_fresh")
        if raw.get("stochastic_selection_only") is not True: row_issues.append("not_stochastic_policy")
        if row_issues: issues.append({"run_index": row_index, "issues": row_issues})
        action_counts.update(raw.get("selected_action_counts", {}))
        hold_counts.update(raw.get("hold_outcome_counts", {}))
        count = int(raw.get("policy_diagnostic_decisions", 0))
        diagnostic_count += count
        for field in fields:
            if raw.get(field) is not None:
                diagnostic_sums[field] += float(raw[field]) * count
    expected_keys = {(int(seed), i) for seed in validation_seeds for i in range(VALIDATION_POLICY_RNG_COUNT)}
    if set(by_key) != expected_keys:
        raise ConstrainedV22TrainerError("validation panel is incomplete")

    instance_rates = []
    row_returns = []
    row_maes = []
    complete_timing = True
    for seed in validation_seeds:
        cluster = [by_key[(int(seed), rng_index)] for rng_index in range(VALIDATION_POLICY_RNG_COUNT)]
        instance_rates.append(100.0 * sum(r["physical_rehandles"] for r in cluster) /
                              sum(r["required_deliveries"] for r in cluster))
        row_returns.extend(float(r["dense_return"]) for r in cluster)
        for row in cluster:
            if row["mean_absolute_error"] is None:
                complete_timing = False
            else:
                row_maes.append(float(row["mean_absolute_error"]))
    point_rate, cost_se, cost_ucb = _sample_ucb(instance_rates)
    block = _positive_int(schedule_state.get("block_number"), name="validation block")
    mean_mae = float(fmean(row_maes)) if complete_timing and len(row_maes) == expected_rows else None
    strict_gate = not issues
    budget_gate = cost_ucb <= config.budget_per_100_required_deliveries + 1e-12
    floor_gate = block >= ANNEAL_LAST_BLOCK
    timing_gate = bool(mean_mae is not None and mean_mae <= TIMING_NONCOLLAPSE_MAE_MAXIMUM)
    point_budget_gate = point_rate <= config.budget_per_100_required_deliveries + 1e-12
    candidate_look_gate = int(checkpoint_episode) in CANDIDATE_EPISODES
    positive_residual_saturation = bool(
        math.isclose(float(validation_lambda), config.lambda_max, rel_tol=0.0, abs_tol=0.0)
        and cost_ucb > config.budget_per_100_required_deliveries + 1e-12
    )
    saturation_count = _nonnegative_int(
        positive_residual_saturation_count,
        name="positive_residual_saturation_count",
    )
    total_macros = sum(int(run.get("macro_decisions", 0)) for run in runs)
    shares = {name: count / total_macros for name, count in sorted(action_counts.items())} if total_macros else {}
    return {
        "checkpoint_episode": int(checkpoint_episode),
        "run_count": len(runs),
        "unique_instance_count": len(validation_seeds),
        "nuisance_rng_count": VALIDATION_POLICY_RNG_COUNT,
        "validation_protocol": VALIDATION_PROTOCOL,
        "validation_policy": POLICY_REALIZATION,
        "map_validation_used": False,
        "validation_lambda": float(validation_lambda),
        "schedule_state": dict(schedule_state),
        "strict_integrity_gate": strict_gate,
        "expected_cost_point_budget_gate": point_budget_gate,
        "expected_cost_ucb_budget_gate": budget_gate,
        "floor_temperature_gate": floor_gate,
        "candidate_look_gate": candidate_look_gate,
        "positive_residual_dual_saturation": positive_residual_saturation,
        "positive_residual_dual_saturation_count_before_validation": saturation_count,
        "timing_noncollapse_gate": timing_gate,
        "timing_noncollapse_mae_maximum": TIMING_NONCOLLAPSE_MAE_MAXIMUM,
        "development_candidate_eligible": bool(
            strict_gate and point_budget_gate and budget_gate and floor_gate
            and timing_gate and candidate_look_gate and not positive_residual_saturation
        ),
        "deployment_checkpoint_eligible": False,
        "safety_issues": issues,
        "mean_dense_return": float(fmean(row_returns)),
        "mean_absolute_error": mean_mae,
        "expected_physical_rehandles_per_100_required_deliveries": point_rate,
        "physical_rehandle_rate_cluster_standard_error": cost_se,
        "physical_rehandle_rate_one_sided_95_ucb": cost_ucb,
        "budget_per_100_required_deliveries": float(config.budget_per_100_required_deliveries),
        "episode_instance_cluster_rehandle_rates": tuple(instance_rates),
        "selected_action_counts": dict(sorted(action_counts.items())),
        "selected_action_shares": shares,
        "hold_outcome_counts": dict(sorted(hold_counts.items())),
        "policy_diagnostic_decisions": diagnostic_count,
        "policy_diagnostics": {field: (None if not diagnostic_count else diagnostic_sums[field] / diagnostic_count) for field in fields},
        "complete_case_filtering_used": False,
    }


def validation_score(summary: Mapping) -> Optional[tuple[float, ...]]:
    if not summary.get("development_candidate_eligible"):
        return None
    return (
        float(summary["mean_dense_return"]), -float(summary["mean_absolute_error"]),
        -float(summary["physical_rehandle_rate_one_sided_95_ucb"]),
        -float(summary["expected_physical_rehandles_per_100_required_deliveries"]),
        -float(summary["checkpoint_episode"]),
    )


def select_better_validation(incumbent: Optional[Mapping], candidate: Mapping) -> Optional[Mapping]:
    score = validation_score(candidate)
    if score is None: return incumbent
    if incumbent is None or validation_score(incumbent) is None or score > validation_score(incumbent):
        return candidate
    return incumbent


class V22DevelopmentRuntime(Protocol):
    @property
    def core_contract(self) -> Mapping: ...
    @property
    def dual_lambda(self) -> float: ...
    @property
    def schedule_state(self) -> Mapping: ...
    def set_dual_lambda(self, value: float) -> None: ...
    def set_schedule_state(self, state: Mapping) -> None: ...
    def begin_validation_batch(self) -> Mapping: ...
    def end_validation_batch(self) -> Mapping: ...
    def run_episode(self, *, instance_seed: int, training: bool, max_steps: int,
                    policy_rng_index: Optional[int] = None,
                    policy_rng_seed: Optional[int] = None) -> Mapping: ...
    def checkpoint_state(self, *, include_replay: bool) -> Mapping: ...


def _validate_runtime(runtime: V22DevelopmentRuntime, contract: Mapping) -> None:
    for name in (
        "set_dual_lambda", "set_schedule_state", "begin_validation_batch",
        "end_validation_batch", "run_episode", "checkpoint_state",
    ):
        if not callable(getattr(runtime, name, None)):
            raise RuntimeError(f"V2.2 runtime is missing callable {name!r}")
    core = getattr(runtime, "core_contract", None)
    expected = {
        "checkpoint_family": CHECKPOINT_FAMILY,
        "controller": CONTROLLER_ARCHITECTURE,
        "backup_version": BACKUP_VERSION,
        "policy_schedule_protocol": POLICY_SCHEDULE_PROTOCOL,
        "training_policy": POLICY_REALIZATION,
        "backup_policy": POLICY_REALIZATION,
        "validation_policy": POLICY_REALIZATION,
        "deployment_policy": POLICY_REALIZATION,
        "map_policy_available": False,
        "same_policy_for_behavior_backup_validation_deployment": True,
        "single_shared_lagrangian_policy_for_both_target_heads": True,
        "entropy_or_kl_added_to_raw_component_targets": False,
        "validation_protocol": VALIDATION_PROTOCOL,
        "exact_safe_frontier_authoritative": True,
        "baseline_teacher": False,
        "baseline_policy_query": False,
    }
    if not isinstance(core, Mapping):
        raise RuntimeError("V2.2 runtime core_contract must be a mapping")
    mismatches = {key: (core.get(key), value) for key, value in expected.items() if core.get(key) != value}
    if mismatches: raise RuntimeError(f"V2.2 runtime contract mismatch: {mismatches!r}")
    if dict(core.get("agent_config", {})) != dict(contract["agent_config"]):
        raise RuntimeError("V2.2 runtime agent config mismatch")


def load_default_runtime(args: argparse.Namespace, contract: Mapping) -> V22DevelopmentRuntime:
    try:
        core = importlib.import_module("viability_graph_constrained_v2_2")
    except ImportError as error:
        raise RuntimeError("isolated V2.2 core unavailable; no fallback permitted") from error
    factory = getattr(core, "build_v2_2_development_runtime", None)
    if not callable(factory): raise RuntimeError("V2.2 runtime factory is unavailable")
    runtime = factory(args=args, contract=dict(contract))
    _validate_runtime(runtime, contract)
    return runtime


def _install_schedule(runtime: V22DevelopmentRuntime, state: V22ScheduleState) -> dict:
    expected = state.to_dict()
    runtime.set_schedule_state(expected)
    if _json_safe(dict(runtime.schedule_state)) != _json_safe(expected):
        raise RuntimeError("V2.2 runtime schedule authentication failed")
    return expected


def _normalize_authenticated_run(run: Mapping, *, index: int, schedule: Mapping,
                                 dual_lambda: float, training: bool) -> dict:
    normalized = normalize_v2_run(run, index)
    expected = {
        "method_version": METHOD_VERSION,
        "policy_schedule_protocol": POLICY_SCHEDULE_PROTOCOL,
        "policy_mode": POLICY_REALIZATION,
        "execution_context": schedule["execution_context"],
        "within_group_temperatures": tuple(schedule["within_group_temperatures"]),
        "group_temperature": float(schedule["group_temperature"]),
    }
    for key, value in expected.items():
        observed = run.get(key)
        if key == "within_group_temperatures" and isinstance(observed, list): observed = tuple(observed)
        if observed != value: raise RuntimeError(f"V2.2 run failed {key} authentication")
    if float(run.get("dual_lambda")) != float(dual_lambda):
        raise RuntimeError("V2.2 run used the wrong lambda")
    if run.get("map_selection_used") is not False:
        raise RuntimeError("V2.2 forbids MAP selection")
    if not training:
        required = {
            "evaluation_learning": False, "training_agent_unchanged": True,
            "fresh_evaluation_clone": True, "stochastic_selection_only": True,
            "option_evaluation_mode": True,
        }
        for key, value in required.items():
            if run.get(key) is not value: raise RuntimeError(f"V2.2 validation failed {key}")
        rng_index = _nonnegative_int(run.get("policy_rng_index"), name="policy_rng_index")
        instance_index = _nonnegative_int(run.get("instance_index"), name="instance_index")
        if run.get("policy_rng_seed") != validation_policy_rng_seed(instance_index, rng_index):
            raise RuntimeError("V2.2 validation nuisance RNG mismatch")
    else:
        episode_index = _nonnegative_int(run.get("policy_rng_index"), name="training policy_rng_index")
        if episode_index != index:
            raise RuntimeError("V2.2 training action RNG index does not match episode index")
        if run.get("policy_rng_seed") != training_policy_rng_seed(episode_index + 1):
            raise RuntimeError("V2.2 training action RNG was not reset from the frozen namespace")
        if run.get("replay_rng_seed") != REPLAY_RNG_SEED:
            raise RuntimeError("V2.2 replay RNG seed mismatch")
        if run.get("behavior_policy_rng_reset") is not True:
            raise RuntimeError("V2.2 training behavior RNG was not reset per episode")
    normalized.update(expected)
    normalized.update({key: run.get(key) for key in (
        "macro_decisions", "selected_action_counts", "hold_outcome_counts",
        "policy_diagnostic_decisions", "mean_outer_policy_entropy",
        "mean_outer_map_probability", "mean_selected_within_policy_entropy",
        "mean_selected_within_map_probability", "policy_rng_index", "policy_rng_seed",
        "evaluation_learning", "training_agent_unchanged", "fresh_evaluation_clone",
        "stochastic_selection_only", "option_evaluation_mode", "map_selection_used",
        "instance_index", "episode_instance_id", "schedule_id", "episode_instance_sha256",
        "replay_rng_seed", "behavior_policy_rng_reset",
        "validation_batch_state_unchanged",
    )})
    normalized["dual_lambda"] = float(dual_lambda)
    normalized["schedule_state"] = dict(schedule)
    return normalized


def _checkpoint_payload(*, contract: Mapping, runtime: V22DevelopmentRuntime,
                        dual: CompleteBlockProjectedDual, completed_episodes: int,
                        role: str, schedule_state: Mapping,
                        pending_dual_runs: Sequence[Mapping],
                        validation_summary: Optional[Mapping], include_replay: bool,
                        terminal_dual_proposal: Optional[Mapping] = None,
                        training_history: Optional[Sequence[Mapping]] = None,
                        validation_history: Optional[Sequence[Mapping]] = None) -> dict:
    if role not in ("latest_development_state", "best_development_candidate"):
        raise ConstrainedV22TrainerError("invalid checkpoint role")
    validated_lambda = None if validation_summary is None else float(validation_summary["validation_lambda"])
    lambda_authenticated = bool(
        validated_lambda is not None
        and math.isclose(float(runtime.dual_lambda), validated_lambda, rel_tol=0.0, abs_tol=0.0)
    )
    # ``latest.pth`` is always an audit-only mutable training-state artifact.
    # Eligibility belongs exclusively to the immutable best-candidate role.
    eligible = bool(
        role == "best_development_candidate"
        and validation_summary
        and validation_summary.get("development_candidate_eligible")
        and lambda_authenticated
    )
    if role == "best_development_candidate" and not eligible:
        raise ConstrainedV22TrainerError("best checkpoint requires an eligible stochastic validation")
    payload = {
        "checkpoint_family": CHECKPOINT_FAMILY,
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "checkpoint_role": role,
        "resumable": False,
        "development_candidate_eligible": eligible,
        "deployment_checkpoint_eligible": False,
        "completed_episodes": int(completed_episodes),
        "training_contract_sha256": contract["contract_sha256"],
        "method_version": METHOD_VERSION,
        "training_protocol": TRAINING_PROTOCOL,
        "backup_version": BACKUP_VERSION,
        "policy_schedule_protocol": POLICY_SCHEDULE_PROTOCOL,
        "dual_update_protocol": DUAL_UPDATE_PROTOCOL,
        "validation_protocol": VALIDATION_PROTOCOL,
        "dual_state": dual.state_dict(),
        "runtime_lambda": float(runtime.dual_lambda),
        "validated_policy_lambda": validated_lambda,
        "validation_authenticates_checkpoint_lambda": lambda_authenticated,
        "schedule_state": dict(schedule_state),
        "pending_dual_batch": {
            "episode_count": len(pending_dual_runs),
            "episode_numbers": tuple(int(run["training_episode_number"]) for run in pending_dual_runs),
            "residuals": tuple(episode_budget_residual(run["physical_rehandles"], run["required_deliveries"], dual.config) for run in pending_dual_runs),
            "audit_only_nonresumable": True,
        },
        "terminal_dual_proposal": None if terminal_dual_proposal is None else dict(terminal_dual_proposal),
        "validation_summary": None if validation_summary is None else dict(validation_summary),
        "core_contract": dict(runtime.core_contract),
        "same_policy_for_behavior_backup_validation_deployment": True,
        "validation_stochastic_only": True,
        "map_validation_used": False,
        "agent_state": runtime.checkpoint_state(include_replay=include_replay),
    }
    if training_history is not None: payload["training_history"] = list(training_history)
    if validation_history is not None: payload["validation_history"] = list(validation_history)
    return payload


def run_development_calibration(args: argparse.Namespace,
                                *, runtime: Optional[V22DevelopmentRuntime] = None) -> dict:
    contract = build_training_contract(args)
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ConstrainedV22TrainerError("V2.2 requires a fresh output directory")
    v21t._atomic_json(contract, output_dir / "training-contract.json")
    if args.contract_only:
        summary = {"status": "contract_only", "method_version": METHOD_VERSION,
                   "episodes_executed": 0, "training_contract": str(output_dir / "training-contract.json"),
                   "prospective_83xxx_panel_opened": False,
                   "prior_83xxx_usage": "83000..83059_were_used_by_prior_work",
                   "83xxx_globally_untouched_claimed": False,
                   "final_86xxx_panel_opened": False,
                   "deployment_checkpoint_eligible": False}
        v21t._atomic_json(summary, output_dir / "training-summary.json")
        return summary
    runtime = runtime or load_default_runtime(args, contract)
    _validate_runtime(runtime, contract)
    dual = CompleteBlockProjectedDual(V22DualConfig(**contract["dual"]))
    runtime.set_dual_lambda(dual.lambda_value)
    training_history, validation_history, dual_history, pending = [], [], [], []
    terminal_proposal = best_validation = None
    latest_path = output_dir / "latest.pth"
    best_path = output_dir / "best-development-candidate.pth"
    nuisance_grid = tuple(tuple(row) for row in contract["validation_policy_rng_grid"])
    validation_identity_manifest = None

    for episode_number in range(1, TOTAL_EPISODES + 1):
        state = schedule_for_episode(episode_number)
        schedule = _install_schedule(runtime, state)
        block_lambda = float(dual.lambda_value)
        runtime.set_dual_lambda(block_lambda)
        run = _normalize_authenticated_run(
            runtime.run_episode(
                instance_seed=args.train_seed_base + episode_number - 1,
                training=True,
                max_steps=args.max_steps,
                policy_rng_index=episode_number - 1,
                policy_rng_seed=training_policy_rng_seed(episode_number),
            ),
            index=episode_number - 1, schedule=schedule, dual_lambda=block_lambda,
            training=True,
        )
        run.update(training_episode_number=episode_number, block_number=state.block_number)
        training_history.append(run)
        if state.dual_updates_enabled: pending.append(run)
        print(f"V2.2 Ep {episode_number:4d} | Phase {state.phase:13s} | "
              f"TrainR {run['dense_return']:8.2f} | Strict {int(run['strict_method_success'])} | "
              f"Reh {run['physical_rehandles']:3d}/{run['required_deliveries']:3d} | "
              f"Lambda {block_lambda:7.4f} | TauM {state.group_temperature:6.4f} | "
              f"TauC {state.within_group_temperatures[0]:6.4f}", flush=True)

        validation_summary = None
        checkpoint_schedule = schedule
        if episode_number % VALIDATION_EVERY == 0:
            checkpoint_schedule = _install_schedule(runtime, schedule_for_episode(episode_number, validation=True))
            raw_rows = []
            row_index = 0
            runtime.begin_validation_batch()
            try:
                for instance_index, instance_seed in enumerate(args.validation_seeds):
                    for rng_index, rng_seed in enumerate(nuisance_grid[instance_index]):
                        raw = runtime.run_episode(
                            instance_seed=int(instance_seed), training=False,
                            max_steps=args.max_steps, policy_rng_index=rng_index,
                            policy_rng_seed=int(rng_seed),
                        )
                        if raw.get("instance_seed") != int(instance_seed):
                            raise RuntimeError("V2.2 runtime did not echo the requested validation seed")
                        raw = dict(raw)
                        raw["instance_index"] = instance_index
                        raw_rows.append(raw)
            finally:
                validation_batch_audit = runtime.end_validation_batch()
            if validation_batch_audit.get("training_agent_unchanged") is not True:
                raise RuntimeError("V2.2 stochastic validation mutated authenticated training state")
            if any(row.get("training_agent_unchanged") is not True for row in raw_rows):
                raise RuntimeError("V2.2 validation mutated and later restored training state within a batch")
            rows = []
            for row_index, raw in enumerate(raw_rows):
                raw["validation_batch_state_unchanged"] = True
                rows.append(_normalize_authenticated_run(
                    raw, index=row_index, schedule=checkpoint_schedule,
                    dual_lambda=block_lambda, training=False,
                ))
            identities = {}
            for row in rows:
                seed = int(row["instance_seed"])
                identity = {
                    "instance_seed": seed,
                    "episode_instance_id": row["episode_instance_id"],
                    "schedule_id": row["schedule_id"],
                    "episode_instance_sha256": row["episode_instance_sha256"],
                }
                incumbent = identities.setdefault(seed, identity)
                if incumbent != identity:
                    raise RuntimeError("V2.2 validation instance drifted across action RNG rows")
            ordered_identities = tuple(identities[int(seed)] for seed in args.validation_seeds)
            if validation_identity_manifest is None:
                validation_identity_manifest = {
                    "schema_version": 1,
                    "validation_protocol": VALIDATION_PROTOCOL,
                    "common_grid_every_checkpoint": True,
                    "instances": ordered_identities,
                    "prospective_83xxx_panel_opened": False,
                    "final_86xxx_panel_opened": False,
                }
                validation_identity_manifest["manifest_sha256"] = contract_hash(validation_identity_manifest)
                v21t._atomic_json(validation_identity_manifest, output_dir / "validation-instance-manifest.json")
            elif ordered_identities != tuple(validation_identity_manifest["instances"]):
                raise RuntimeError("V2.2 common validation EpisodeInstance grid drifted across looks")
            validation_summary = summarize_validation(
                rows, dual.config, checkpoint_episode=episode_number,
                validation_lambda=block_lambda, schedule_state=checkpoint_schedule,
                validation_seeds=args.validation_seeds,
                positive_residual_saturation_count=dual.saturation_count,
            )
            validation_summary["validated_before_dual_update"] = True
            validation_summary["validation_batch_audit"] = dict(validation_batch_audit)
            ledger_relative = f"validation-ledger/episode-{episode_number:04d}.json"
            ledger_payload = {
                "schema_version": 1,
                "validation_protocol": VALIDATION_PROTOCOL,
                "checkpoint_episode": episode_number,
                "validated_lambda": block_lambda,
                "schedule_state": dict(checkpoint_schedule),
                "row_count": len(rows),
                "complete_case_filtering_used": False,
                "validation_batch_audit": dict(validation_batch_audit),
                "rows": rows,
            }
            ledger_payload["ledger_sha256"] = contract_hash(ledger_payload)
            v21t._atomic_json(ledger_payload, output_dir / ledger_relative)
            validation_summary["validation_ledger"] = {
                "relative_path": ledger_relative,
                "ledger_sha256": ledger_payload["ledger_sha256"],
                "row_count": len(rows),
            }
            validation_history.append(validation_summary)
            mae_text = "NA" if validation_summary["mean_absolute_error"] is None else f"{validation_summary['mean_absolute_error']:7.2f}"
            print(f"V2.2 Validation Ep {episode_number:4d} | "
                  f"R {validation_summary['mean_dense_return']:8.2f} | "
                  f"MAE {mae_text} | "
                  f"Strict {int(validation_summary['strict_integrity_gate'])} | "
                  f"Reh/100 {validation_summary['expected_physical_rehandles_per_100_required_deliveries']:7.2f} | "
                  f"UCB {validation_summary['physical_rehandle_rate_one_sided_95_ucb']:7.2f} | "
                  f"Eligible {int(validation_summary['development_candidate_eligible'])}", flush=True)
            if select_better_validation(best_validation, validation_summary) is validation_summary:
                best_validation = validation_summary
                v21t._atomic_torch(_checkpoint_payload(
                    contract=contract, runtime=runtime, dual=dual,
                    completed_episodes=episode_number, role="best_development_candidate",
                    schedule_state=checkpoint_schedule, pending_dual_runs=pending,
                    validation_summary=validation_summary, include_replay=False,
                ), best_path)

        # Dual cadence remains every ten training episodes, independently of
        # the every-twenty-episode stochastic validation cadence.
        if episode_number % BLOCK_EPISODES == 0:
            if state.dual_updates_enabled:
                if state.block_number == TOTAL_BLOCKS:
                    proposal = dual.propose_complete_block(pending, block_number=state.block_number)
                    terminal_proposal = {**proposal, "applied": False,
                                         "reason_not_applied": "no_subsequent_primal_block"}
                    dual_history.append(terminal_proposal)
                else:
                    update = {**dual.update_complete_block(pending, block_number=state.block_number), "applied": True}
                    dual_history.append(update)
                    runtime.set_dual_lambda(dual.lambda_value)
                pending.clear()
            elif pending:
                raise RuntimeError("warm-up residuals leaked into dual batch")

        v21t._atomic_torch(_checkpoint_payload(
            contract=contract, runtime=runtime, dual=dual,
            completed_episodes=episode_number, role="latest_development_state",
            schedule_state=checkpoint_schedule, pending_dual_runs=pending,
            validation_summary=validation_summary, include_replay=True,
            terminal_dual_proposal=terminal_proposal,
            training_history=training_history, validation_history=validation_history,
        ), latest_path)

    if pending: raise RuntimeError("V2.2 ended with an incomplete dual block")
    summary = {
        "status": "complete", "method_version": METHOD_VERSION,
        "completed_training_episodes": len(training_history),
        "completed_blocks": TOTAL_BLOCKS, "dual_state": dual.state_dict(),
        "dual_update_history": dual_history, "terminal_dual_proposal": terminal_proposal,
        "development_candidate_eligible": best_validation is not None,
        "deployment_checkpoint_eligible": False, "best_validation": best_validation,
        "validation_history": validation_history,
        "best_development_candidate": str(best_path) if best_path.exists() else None,
        "latest_checkpoint": str(latest_path),
        "training_contract": str(output_dir / "training-contract.json"),
        "validation_instance_manifest": str(output_dir / "validation-instance-manifest.json"),
        "prospective_83xxx_panel_opened": False,
        "prior_83xxx_usage": "83000..83059_were_used_by_prior_work",
        "83xxx_globally_untouched_claimed": False,
        "final_86xxx_panel_opened": False,
    }
    v21t._atomic_json(summary, output_dir / "training-summary.json")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run frozen constrained VCG V2.2 calibration")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-seed", type=int, default=DEFAULT_MODEL_SEED)
    parser.add_argument("--episodes", type=int, default=TOTAL_EPISODES)
    parser.add_argument("--train-seed-base", type=int, default=FRESH_TRAIN_SEED_BASE)
    parser.add_argument("--validation-seeds", type=int, nargs="+", default=DEFAULT_VALIDATION_SEEDS)
    parser.add_argument("--validation-every", type=int, default=VALIDATION_EVERY)
    parser.add_argument("--max-steps", type=int, default=2_000)
    parser.add_argument("--grid-rows", type=int, default=5)
    parser.add_argument("--grid-cols", type=int, default=5)
    parser.add_argument("--number-blocks", type=int, default=8)
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
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--contract-only", action="store_true")
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    for name in ("episodes", "validation_every", "max_steps", "grid_rows", "grid_cols",
                 "number_blocks", "proc_mean", "max_hold_steps", "max_idle_steps"):
        _positive_int(getattr(args, name), name=name)
    if args.episodes != TOTAL_EPISODES or args.validation_every != VALIDATION_EVERY:
        raise ConstrainedV22TrainerError("V2.2 clock is frozen at 200 episodes / validation every 20")
    if args.model_seed != DEFAULT_MODEL_SEED:
        raise ConstrainedV22TrainerError("V2.2 requires model seed 10")
    if args.train_seed_base != FRESH_TRAIN_SEED_BASE:
        raise ConstrainedV22TrainerError("V2.2 requires train seed base 61000000")
    if (args.grid_rows, args.grid_cols, args.number_blocks, args.max_steps) != (5, 5, 8, 2_000):
        raise ConstrainedV22TrainerError("V2.2 requires the frozen 5x5/8-block/2000-step workload")
    if _finite(args.gamma_operational, name="gamma_operational") != 0.99:
        raise ConstrainedV22TrainerError("V2.2 requires gamma_operational=0.99")
    if _finite(args.reward_scale, name="reward_scale") != 0.01:
        raise ConstrainedV22TrainerError("V2.2 requires reward_scale=0.01")
    if _finite(args.arrival_rate, name="arrival_rate") <= 0:
        raise ConstrainedV22TrainerError("arrival_rate must be positive")
    if (float(args.arrival_rate), int(args.proc_mean), int(args.max_hold_steps), int(args.max_idle_steps)) != (10.0, 80, 10, 20):
        raise ConstrainedV22TrainerError("V2.2 arrival/proc/Hold constants are frozen")
    V22DualConfig(args.rehandle_budget_per_100, args.dual_lr,
                  args.lambda_initial, args.lambda_max)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    print(__import__("json").dumps(_json_safe(run_development_calibration(args)),
                                   indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()


__all__ = [
    "BACKUP_VERSION", "CHECKPOINT_FAMILY", "CHECKPOINT_SCHEMA_VERSION",
    "CONTROLLER_ARCHITECTURE", "DUAL_UPDATE_PROTOCOL", "METHOD_VERSION",
    "POLICY_DIAGNOSTIC_DISTRIBUTION", "POLICY_REALIZATION",
    "POLICY_SCHEDULE_PROTOCOL", "TRAINING_PROTOCOL", "VALIDATION_PROTOCOL",
    "VALIDATION_POLICY_RNG_COUNT", "VALIDATION_POLICY_RNG_BASE",
    "CompleteBlockProjectedDual", "ConstrainedV22TrainerError", "V22DualConfig",
    "V22ScheduleState", "build_parser", "build_training_contract",
    "episode_budget_residual", "frozen_schedule_table", "load_default_runtime",
    "run_development_calibration", "schedule_for_episode", "select_better_validation",
    "summarize_validation", "temperatures_for_block", "validation_policy_rng_seed",
    "validation_score",
]
