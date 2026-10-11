#!/usr/bin/env python3
"""Controlled gamma ablation for policy/operator-consistent VCG V2.3.

V2.3 is derived from V2.2 and changes exactly one algorithmic quantity:
``gamma_operational`` is 1.0 instead of 0.99.  The raw dense objective,
``gamma_rehandle=1``, rehandle budget, instances, RNG streams, network,
optimizer, policy, temperature clock, dual, Hold, validation, and gates are
otherwise frozen to V2.2.  The already-open V2.2 training and development
panels are deliberately reused for a controlled paired ablation; this is not
a fresh prospective experiment.

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
import hashlib
import importlib
import io
import json
import math
from numbers import Integral, Real
from pathlib import Path
from statistics import fmean, stdev
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
import train_vcg_constrained_v2_1 as v21t


TRAINING_PROTOCOL = (
    "vcg_constrained_vector_smdp_v2_3_gamma_operational_1_"
    "controlled_ablation_development_calibration_v1"
)
METHOD_VERSION = "vcg_constrained_v2_3_gamma1"
TRAINER_SCHEMA_VERSION = 1
CHECKPOINT_FAMILY = "vcg_constrained_vector_smdp_v2_3_gamma1"
CHECKPOINT_SCHEMA_VERSION = 1
CONTROLLER_ARCHITECTURE = (
    "exact_safe_shared_graph_vector_q_primal_dual_policy_consistent_gamma1_v2_3"
)
BACKUP_VERSION = (
    "elapsed_weighted_shared_annealed_lagrangian_policy_vector_expected_double_q_"
    "matched_undiscounted_operational_rehandle_smdp_v2_3"
)
POLICY_SCHEDULE_PROTOCOL = (
    "block_constant_shared_behavior_backup_validation_deployment_stochastic_v2_3_gamma1"
)
DUAL_UPDATE_PROTOCOL = (
    "projected_complete_active_block_workload_residual_ascent_v2_3"
)
CHECKPOINT_SELECTION_VERSION = (
    "strict_integrity_expected_cost_ucb_dense_return_mae_rehandle_earlier_v2_3"
)
POLICY_DIAGNOSTIC_DISTRIBUTION = (
    "executed_induced_nested_regularized_lagrangian_distribution_v2_3"
)
POLICY_REALIZATION = "induced_nested_regularized_lagrangian_sample"
VALIDATION_PROTOCOL = (
    "fixed_instance_crossed_fixed_action_rng_fresh_frozen_clone_expected_cost_v2_3_gamma1"
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
PARENT_V2_2_CONTRACT_SHA256 = (
    "d764582412a73d6bc5e4f8688c28a354e87bc24e7f162c974857f1149738e0c1"
)
PARENT_V2_2_VALIDATION_MANIFEST_SHA256 = (
    "cd63f0672e6557c0fa53c598e67c12609c8d6393fb8a45f8f46e2d116b108833"
)
PARENT_V2_2_VALIDATION_INSTANCE_IDENTITIES = (
    {"instance_seed": 85000, "episode_instance_id": "9000bfcb50159813", "schedule_id": "805cfca6acadc0c2", "episode_instance_sha256": "4871766be3e57fbb83563a9b95af4ffca48090d514562cf81055c722be6dc5da"},
    {"instance_seed": 85001, "episode_instance_id": "6c86a7aab6084717", "schedule_id": "e7cb521cc81a9ee6", "episode_instance_sha256": "61186678d9aea227878f69c66b68b3b0ed3b41ae8bb33e6964863be8b4d29249"},
    {"instance_seed": 85002, "episode_instance_id": "38d5df1e6714852f", "schedule_id": "3db6b818a9b9d4b8", "episode_instance_sha256": "4915112c3d75c3a5b934ff6cf74ddec9caf0b6cd61f5b81744f93a3b1a95aa20"},
    {"instance_seed": 85003, "episode_instance_id": "261b8cf77b50dc1f", "schedule_id": "f6b5b989a5286029", "episode_instance_sha256": "e82ae3ff0d85e103636febfdb525c551d49225146ce886eec60c842dff09ac59"},
    {"instance_seed": 85004, "episode_instance_id": "42df31b4886f378d", "schedule_id": "f2518a2afea45f71", "episode_instance_sha256": "fc4d582ca2416b59480ebdad5eba69307e6ee7062c1b2debb2f2e56f4e017b5d"},
    {"instance_seed": 85005, "episode_instance_id": "14e4495e9dce333c", "schedule_id": "ccd3db7b4c8b8ae4", "episode_instance_sha256": "c6c533d587688afbd9f195e8ec191ea5a196d5af4d59bf55159bd9dbf0b0b0b4"},
    {"instance_seed": 85006, "episode_instance_id": "f5c10703dca8e767", "schedule_id": "02b257dca174d068", "episode_instance_sha256": "9bae6708e0ed769c58085a917b63e2cba6fd4c34b4c3c11c3897873be673c299"},
    {"instance_seed": 85007, "episode_instance_id": "52ee13ad03fcd827", "schedule_id": "fe1412a302ee9a39", "episode_instance_sha256": "6d8b3248fb89a4b22cebbcd7f3605bee32579be9dde27c0d4edee837daf92190"},
    {"instance_seed": 85008, "episode_instance_id": "cb0029ffab665826", "schedule_id": "9709cb8cfc68ba9c", "episode_instance_sha256": "006434a1a53b61ec3c0a108faf3dd18850c570baba6823f85b751464c8136faf"},
    {"instance_seed": 85009, "episode_instance_id": "7dd23c647ecd3953", "schedule_id": "06fbd48f299c780b", "episode_instance_sha256": "be5f5d296e711da09330eff544799322bc229936cee18e43b44d706815efb959"},
    {"instance_seed": 85010, "episode_instance_id": "e117c463566e619a", "schedule_id": "f59b392e84de3853", "episode_instance_sha256": "a9fe72b40e05f1f4574b518ffd469af9c8c2025db0e80fa527c770a55c90a277"},
    {"instance_seed": 85011, "episode_instance_id": "2c30ecb9cb02e2b3", "schedule_id": "467096ab06aa3155", "episode_instance_sha256": "4c1ff6ee906e0d07a1906e875ecf81193c720b28df470dbdf440508c1dcc7f13"},
)


class ConstrainedV23TrainerError(ValueError):
    """Raised when the frozen V2.3 protocol is violated."""


def _finite(value, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ConstrainedV23TrainerError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ConstrainedV23TrainerError(f"{name} must be finite")
    return result


def _nonnegative_int(value, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ConstrainedV23TrainerError(f"{name} must be an integer")
    result = int(value)
    if result < 0:
        raise ConstrainedV23TrainerError(f"{name} must be non-negative")
    return result


def _positive_int(value, *, name: str) -> int:
    result = _nonnegative_int(value, name=name)
    if result <= 0:
        raise ConstrainedV23TrainerError(f"{name} must be positive")
    return result


def _json_safe(value):
    return v21t._json_safe(value)


def contract_hash(contract: Mapping) -> str:
    return v21t.contract_hash(contract)


def training_policy_rng_seed(episode_number: int) -> int:
    episode = _positive_int(episode_number, name="training episode number")
    if episode > TOTAL_EPISODES:
        raise ConstrainedV23TrainerError("training episode is outside the frozen clock")
    return TRAINING_POLICY_RNG_BASE + episode - 1


def validation_policy_rng_seed(instance_index: int, rng_index: int) -> int:
    instance_index = _nonnegative_int(instance_index, name="validation instance index")
    rng_index = _nonnegative_int(rng_index, name="validation policy RNG index")
    if instance_index >= len(DEFAULT_VALIDATION_SEEDS) or rng_index >= VALIDATION_POLICY_RNG_COUNT:
        raise ConstrainedV23TrainerError("validation RNG coordinates are out of range")
    return VALIDATION_POLICY_RNG_BASE + VALIDATION_POLICY_RNG_COUNT * instance_index + rng_index


@dataclass(frozen=True)
class V23DualConfig:
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
            raise ConstrainedV23TrainerError("budget must be non-negative")
        if _finite(self.learning_rate, name="dual learning_rate") != 0.01:
            raise ConstrainedV23TrainerError("V2.3 requires dual learning_rate=0.01")
        if _finite(self.lambda_initial, name="lambda_initial") != 0.0:
            raise ConstrainedV23TrainerError("V2.3 warm-up requires lambda_initial=0")
        if _finite(self.lambda_max, name="lambda_max") != 20.0:
            raise ConstrainedV23TrainerError("V2.3 requires lambda_max=20")
        if self.warmup_episodes != WARMUP_EPISODES:
            raise ConstrainedV23TrainerError("V2.3 warm-up must be 20 episodes")
        if self.block_episodes != BLOCK_EPISODES:
            raise ConstrainedV23TrainerError("V2.3 dual blocks must be 10 episodes")

    @property
    def budget_per_required_delivery(self) -> float:
        return float(self.budget_per_100_required_deliveries) / 100.0


def episode_budget_residual(
    physical_rehandles: int,
    required_deliveries: int,
    config: V23DualConfig,
) -> float:
    if not isinstance(config, V23DualConfig):
        raise ConstrainedV23TrainerError("V2.3 residual requires V23DualConfig")
    cost = _nonnegative_int(physical_rehandles, name="physical_rehandles")
    workload = _positive_int(required_deliveries, name="required_deliveries")
    return float(cost - config.budget_per_required_delivery * workload)


@dataclass
class CompleteBlockProjectedDual:
    config: V23DualConfig
    lambda_value: float = 0.0
    update_count: int = 0
    active_episodes_consumed: int = 0
    last_completed_block: int = WARMUP_BLOCKS
    last_mean_residual: Optional[float] = None
    saturation_count: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.config, V23DualConfig):
            raise ConstrainedV23TrainerError("dual config must be V23DualConfig")
        self.lambda_value = _finite(self.lambda_value, name="lambda_value")
        if not 0.0 <= self.lambda_value <= self.config.lambda_max:
            raise ConstrainedV23TrainerError("lambda_value is out of range")

    def propose_complete_block(
        self, episodes: Sequence[Mapping], *, block_number: int
    ) -> dict:
        block = _positive_int(block_number, name="block_number")
        if block != self.last_completed_block + 1:
            raise ConstrainedV23TrainerError("dual update is out of block order")
        if block <= WARMUP_BLOCKS:
            raise ConstrainedV23TrainerError("warm-up residuals cannot update lambda")
        if len(episodes) != self.config.block_episodes:
            raise ConstrainedV23TrainerError("dual requires one complete block")
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
            raise ConstrainedV23TrainerError("terminal proposal cannot be applied")
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
class V23ScheduleState:
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
            raise ConstrainedV23TrainerError("invalid V2.3 execution context")
        episode = _positive_int(self.episode_number, name="episode_number")
        block = _positive_int(self.block_number, name="block_number")
        if not 1 <= episode <= TOTAL_EPISODES:
            raise ConstrainedV23TrainerError("episode is outside V2.3 clock")
        if block != (episode - 1) // BLOCK_EPISODES + 1:
            raise ConstrainedV23TrainerError("block disagrees with episode clock")
        expected_phase = (
            "critic_warmup" if block <= WARMUP_BLOCKS else
            "primal_dual_anneal" if block <= ANNEAL_LAST_BLOCK else
            "low_temperature_stabilization"
        )
        if self.phase != expected_phase:
            raise ConstrainedV23TrainerError("phase disagrees with block clock")
        if len(self.within_group_temperatures) != 4:
            raise ConstrainedV23TrainerError("four within-group temperatures required")
        for value in (*self.within_group_temperatures, self.group_temperature):
            if _finite(value, name="temperature") <= 0.0:
                raise ConstrainedV23TrainerError("temperatures must be positive")
        if self.execution_context == "training":
            if not self.critic_updates_enabled:
                raise ConstrainedV23TrainerError("training must update the critic")
            if self.dual_updates_enabled != (block > WARMUP_BLOCKS):
                raise ConstrainedV23TrainerError("dual flag disagrees with block clock")
        elif self.critic_updates_enabled or self.dual_updates_enabled:
            raise ConstrainedV23TrainerError("validation/deployment cannot update")

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
        raise ConstrainedV23TrainerError("block is outside the V2.3 clock")
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
) -> V23ScheduleState:
    episode = _positive_int(episode_number, name="episode_number")
    if not 1 <= episode <= TOTAL_EPISODES:
        raise ConstrainedV23TrainerError("episode is outside V2.3 clock")
    if validation and episode % VALIDATION_EVERY:
        raise ConstrainedV23TrainerError("validation is allowed only at the frozen 20-episode cadence")
    block = (episode - 1) // BLOCK_EPISODES + 1
    within, group = temperatures_for_block(block)
    phase = (
        "critic_warmup" if block <= WARMUP_BLOCKS else
        "primal_dual_anneal" if block <= ANNEAL_LAST_BLOCK else
        "low_temperature_stabilization"
    )
    return V23ScheduleState(
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
        raise ConstrainedV23TrainerError("V2.3 training seed base must be 61000000")
    if _positive_int(episodes, name="episodes") != TOTAL_EPISODES:
        raise ConstrainedV23TrainerError("V2.3 is frozen at 200 episodes")
    validation = tuple(_nonnegative_int(seed, name="validation_seed") for seed in validation_seeds)
    if validation != DEFAULT_VALIDATION_SEEDS:
        raise ConstrainedV23TrainerError("V2.3 requires the frozen 85000..85011 validation panel")
    training = set(range(base, base + episodes))
    if (training | set(validation)) & (set(v21t.PROTECTED_PROSPECTIVE_SEEDS) | set(FINAL_PANEL_SEEDS)):
        raise ConstrainedV23TrainerError("V2.3 refuses protected 83xxx/final 86xxx seeds")
    if training & set(validation):
        raise ConstrainedV23TrainerError("training and validation seeds overlap")


def frozen_agent_config(args: argparse.Namespace) -> dict:
    from viability_graph_constrained_v2_3 import ConstrainedV23Config
    return ConstrainedV23Config(
        gamma_operational=1.0,
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
    dual = V23DualConfig(
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
        "experiment_role": "controlled_gamma_operational_ablation",
        "parent_method_version": "vcg_constrained_v2_2",
        "isolated_algorithmic_change": {
            "parameter": "gamma_operational",
            "parent_value": 0.99,
            "ablation_value": 1.0,
            "all_other_algorithmic_and_protocol_settings_frozen_to_parent": True,
        },
        "parent_v2_2_binding": {
            "training_contract_canonical_sha256": PARENT_V2_2_CONTRACT_SHA256,
            "validation_manifest_canonical_sha256": PARENT_V2_2_VALIDATION_MANIFEST_SHA256,
            "exact_validation_instance_identities_required": True,
            "validation_instance_identities": PARENT_V2_2_VALIDATION_INSTANCE_IDENTITIES,
        },
        "development_panel_reused_after_v2_2": True,
        "training_instances_reused_after_v2_2": True,
        "fresh_prospective_experiment": False,
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
        "no_v1_v2_v2_1_v2_2_fallback": True,
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
        "gamma_operational": 1.0,
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
        "candidate_look_diagnostics": {
            "save_every_candidate_look": True,
            "episodes": CANDIDATE_EPISODES,
            "immutable_unique_path_per_look": True,
            "model_only": True,
            "diagnostic_only": True,
            "candidate_eligibility_always_false": True,
            "eligible_best_checkpoint_is_separate": True,
        },
        "device": str(args.device),
    }
    contract["contract_sha256"] = contract_hash(contract)
    return contract


def _sample_ucb(values: Sequence[float]) -> tuple[float, float, float]:
    values = tuple(float(value) for value in values)
    if len(values) != len(DEFAULT_VALIDATION_SEEDS):
        raise ConstrainedV23TrainerError("UCB requires all 12 EpisodeInstance clusters")
    mean = float(fmean(values))
    standard_error = float(stdev(values) / math.sqrt(len(values)))
    return mean, standard_error, float(mean + VALIDATION_COST_T_95 * standard_error)


def summarize_validation(
    runs: Sequence[Mapping], config: V23DualConfig, *, checkpoint_episode: int,
    validation_lambda: float, schedule_state: Mapping,
    validation_seeds: Sequence[int] = DEFAULT_VALIDATION_SEEDS,
    positive_residual_saturation_count: int = 0,
) -> dict:
    expected_rows = len(validation_seeds) * VALIDATION_POLICY_RNG_COUNT
    if len(runs) != expected_rows:
        raise ConstrainedV23TrainerError(f"validation requires exactly {expected_rows} rows")
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
            raise ConstrainedV23TrainerError("validation instance index/seed mismatch")
        rng_index = _nonnegative_int(raw.get("policy_rng_index"), name="policy_rng_index")
        rng_seed = _nonnegative_int(raw.get("policy_rng_seed"), name="policy_rng_seed")
        key = (instance_seed, rng_index)
        if instance_seed not in validation_seeds or key in by_key:
            raise ConstrainedV23TrainerError("validation row identity is invalid or duplicated")
        instance_index = tuple(validation_seeds).index(instance_seed)
        if rng_seed != validation_policy_rng_seed(instance_index, rng_index):
            raise ConstrainedV23TrainerError("validation action RNG is outside frozen panel")
        by_key[key] = run
        identity = (
            raw.get("episode_instance_id"), raw.get("schedule_id"),
            raw.get("episode_instance_sha256"),
        )
        if not all(isinstance(value, str) and value for value in identity[:2]):
            raise ConstrainedV23TrainerError("validation row is missing instance identity")
        if not isinstance(identity[2], str) or len(identity[2]) != 64 or any(
            char not in "0123456789abcdef" for char in identity[2]
        ):
            raise ConstrainedV23TrainerError("validation row has invalid canonical instance SHA")
        incumbent_identity = identities_by_seed.setdefault(instance_seed, identity)
        if incumbent_identity != identity:
            raise ConstrainedV23TrainerError("EpisodeInstance identity drifted within its action-RNG cluster")
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
        raise ConstrainedV23TrainerError("validation panel is incomplete")

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


class V23DevelopmentRuntime(Protocol):
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


def _validate_runtime(runtime: V23DevelopmentRuntime, contract: Mapping) -> None:
    for name in (
        "set_dual_lambda", "set_schedule_state", "begin_validation_batch",
        "end_validation_batch", "run_episode", "checkpoint_state",
    ):
        if not callable(getattr(runtime, name, None)):
            raise RuntimeError(f"V2.3 runtime is missing callable {name!r}")
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
        raise RuntimeError("V2.3 runtime core_contract must be a mapping")
    mismatches = {key: (core.get(key), value) for key, value in expected.items() if core.get(key) != value}
    if mismatches: raise RuntimeError(f"V2.3 runtime contract mismatch: {mismatches!r}")
    if dict(core.get("agent_config", {})) != dict(contract["agent_config"]):
        raise RuntimeError("V2.3 runtime agent config mismatch")


def load_default_runtime(args: argparse.Namespace, contract: Mapping) -> V23DevelopmentRuntime:
    try:
        core = importlib.import_module("viability_graph_constrained_v2_3")
    except ImportError as error:
        raise RuntimeError("isolated V2.3 core unavailable; no fallback permitted") from error
    factory = getattr(core, "build_v2_3_development_runtime", None)
    if not callable(factory): raise RuntimeError("V2.3 runtime factory is unavailable")
    runtime = factory(args=args, contract=dict(contract))
    _validate_runtime(runtime, contract)
    return runtime


def _install_schedule(runtime: V23DevelopmentRuntime, state: V23ScheduleState) -> dict:
    expected = state.to_dict()
    runtime.set_schedule_state(expected)
    if _json_safe(dict(runtime.schedule_state)) != _json_safe(expected):
        raise RuntimeError("V2.3 runtime schedule authentication failed")
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
        if observed != value: raise RuntimeError(f"V2.3 run failed {key} authentication")
    if float(run.get("dual_lambda")) != float(dual_lambda):
        raise RuntimeError("V2.3 run used the wrong lambda")
    if run.get("map_selection_used") is not False:
        raise RuntimeError("V2.3 forbids MAP selection")
    if not training:
        required = {
            "evaluation_learning": False, "training_agent_unchanged": True,
            "fresh_evaluation_clone": True, "stochastic_selection_only": True,
            "option_evaluation_mode": True,
        }
        for key, value in required.items():
            if run.get(key) is not value: raise RuntimeError(f"V2.3 validation failed {key}")
        rng_index = _nonnegative_int(run.get("policy_rng_index"), name="policy_rng_index")
        instance_index = _nonnegative_int(run.get("instance_index"), name="instance_index")
        if run.get("policy_rng_seed") != validation_policy_rng_seed(instance_index, rng_index):
            raise RuntimeError("V2.3 validation nuisance RNG mismatch")
    else:
        episode_index = _nonnegative_int(run.get("policy_rng_index"), name="training policy_rng_index")
        if episode_index != index:
            raise RuntimeError("V2.3 training action RNG index does not match episode index")
        if run.get("policy_rng_seed") != training_policy_rng_seed(episode_index + 1):
            raise RuntimeError("V2.3 training action RNG was not reset from the frozen namespace")
        if run.get("replay_rng_seed") != REPLAY_RNG_SEED:
            raise RuntimeError("V2.3 replay RNG seed mismatch")
        if run.get("behavior_policy_rng_reset") is not True:
            raise RuntimeError("V2.3 training behavior RNG was not reset per episode")
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


_MODEL_ONLY_AGENT_STATE_KEYS = frozenset({
    "Q_local",
    "Q_target",
    "dual_lambda",
    "schedule_state",
    "replay_rng_seed",
})
_CANDIDATE_LOOK_MANIFEST_KEYS = frozenset({
    "schema_version", "checkpoint_family", "training_contract_sha256",
    "expected_candidate_look_episodes", "saved_candidate_look_episodes",
    "artifact_count", "complete", "model_only", "diagnostic_only",
    "final_86xxx_panel_opened", "artifacts", "manifest_sha256",
})
_CANDIDATE_LOOK_ARTIFACT_KEYS = frozenset({
    "checkpoint_episode", "relative_path", "sha256", "checkpoint_role",
    "model_only", "diagnostic_only", "development_candidate_eligible",
    "deployment_checkpoint_eligible",
    "validation_development_candidate_eligible",
})
_CANDIDATE_LOOK_OUTER_KEYS = frozenset({
    "checkpoint_family", "checkpoint_schema_version", "checkpoint_role",
    "resumable", "development_candidate_eligible",
    "deployment_checkpoint_eligible", "diagnostic_only", "model_only",
    "immutable_unique_candidate_look_artifact", "completed_episodes",
    "training_contract_sha256", "method_version", "training_protocol",
    "backup_version", "policy_schedule_protocol", "dual_update_protocol",
    "validation_protocol", "dual_state", "runtime_lambda",
    "validated_policy_lambda", "validation_authenticates_checkpoint_lambda",
    "schedule_state", "pending_dual_batch", "terminal_dual_proposal",
    "validation_summary", "core_contract",
    "same_policy_for_behavior_backup_validation_deployment",
    "validation_stochastic_only", "map_validation_used", "agent_state",
})
_MODEL_ONLY_NESTED_CHECKPOINT_KEYS = frozenset({
    "action_feature_names", "action_types", "agent_state", "backup_policy",
    "backup_version", "baseline_policy_query", "baseline_teacher",
    "behavior_rng_scope", "candidate_interface", "certification_contract",
    "checkpoint_family", "checkpoint_schema_version", "component_backup",
    "config", "control_groups", "controller_architecture", "cost_definition",
    "cost_head_nonnegative", "deployment_policy",
    "entropy_or_kl_added_to_raw_component_targets", "epsilon_exploration",
    "exact_safe_mask_authoritative", "exact_verifier_authority",
    "graph_node_feature_names", "group_mapping", "group_temperature",
    "hold_rule", "map_budget_guarantee", "map_policy_available",
    "method_version", "model_only", "network_architecture",
    "operational_gamma", "operational_reward_scale", "optimizer_included",
    "policy_diagnostic_distribution", "policy_diagnostic_version",
    "policy_realization", "policy_realization_decoupled_from_training_flag",
    "policy_schedule_protocol", "policy_version", "reference_distribution",
    "rehandle_gamma", "replay_included", "replay_rng_scope",
    "replay_rng_seed", "replay_version",
    "same_policy_for_behavior_backup_validation_deployment",
    "scalarized_reward_stored_in_replay", "schedule_state",
    "single_shared_lagrangian_policy_for_both_target_heads", "time_context",
    "training_policy", "training_rng_state_included",
    "unsafe_unknown_fail_closed", "validation_policy",
    "within_group_temperatures",
})
_BEST_NESTED_CHECKPOINT_KEYS = frozenset(
    _MODEL_ONLY_NESTED_CHECKPOINT_KEYS.difference({
        "model_only", "optimizer_included", "replay_included",
        "training_rng_state_included",
    })
)
_BEST_AGENT_STATE_KEYS = frozenset({
    "Q_local", "Q_target", "optimizer", "epsilon", "dual_lambda",
    "transition_count", "decision_count", "gradient_steps", "target_updates",
    "group_decisions", "action_decisions", "selection_sources",
    "exact_rejections_seen", "interface_rejections", "safe_candidates_scored",
    "operational_loss_history", "rehandle_loss_history", "total_loss_history",
    "rng_state", "recovery_witness_guard", "schedule_state",
    "policy_diagnostics", "behavior_rng_state", "behavior_rng_seed",
    "behavior_rng_reset_count", "replay_rng_state", "replay_rng_seed",
})
_VALIDATION_INSTANCE_MANIFEST_KEYS = frozenset({
    "schema_version", "validation_protocol", "common_grid_every_checkpoint",
    "parent_v2_2_validation_manifest_sha256",
    "exact_parent_v2_2_instance_identity_match", "instances",
    "prospective_83xxx_panel_opened", "final_86xxx_panel_opened",
    "manifest_sha256",
})
_VALIDATION_LEDGER_KEYS = frozenset({
    "schema_version", "validation_protocol", "checkpoint_episode",
    "validated_lambda", "schedule_state", "row_count",
    "complete_case_filtering_used", "validation_batch_audit", "rows",
    "ledger_sha256",
})
_VALIDATION_LEDGER_REFERENCE_KEYS = frozenset({
    "relative_path", "ledger_sha256", "row_count",
})
_VALIDATION_BATCH_AUDIT_KEYS = frozenset({
    "validation_batch_ended", "training_agent_unchanged",
    "full_training_state_digest_before", "full_training_state_digest_after",
})
_DUAL_STATE_KEYS = frozenset({
    "config", "lambda_value", "update_count", "last_mean_residual",
    "last_completed_block", "active_episodes_consumed", "protocol",
    "saturation_count",
})
_PENDING_DUAL_BATCH_KEYS = frozenset({
    "episode_count", "episode_numbers", "residuals",
    "audit_only_nonresumable",
})


def _require_exact_keys(value: Mapping, expected: frozenset[str], *, name: str) -> None:
    observed = set(value)
    if observed != expected:
        raise ConstrainedV23TrainerError(
            f"{name} schema mismatch: missing={sorted(expected - observed)!r}, "
            f"extra={sorted(observed - expected)!r}"
        )


def _model_only_runtime_checkpoint(runtime: V23DevelopmentRuntime) -> dict:
    checkpoint = dict(runtime.checkpoint_state(include_replay=False))
    state = checkpoint.get("agent_state")
    if not isinstance(state, Mapping):
        raise ConstrainedV23TrainerError(
            "model-only diagnostic requires an authenticated nested agent checkpoint"
        )
    missing = sorted(_MODEL_ONLY_AGENT_STATE_KEYS.difference(state))
    if missing:
        raise ConstrainedV23TrainerError(
            f"model-only diagnostic checkpoint is missing {missing!r}"
        )
    checkpoint["agent_state"] = {
        key: state[key] for key in sorted(_MODEL_ONLY_AGENT_STATE_KEYS)
    }
    checkpoint["model_only"] = True
    checkpoint["optimizer_included"] = False
    checkpoint["replay_included"] = False
    checkpoint["training_rng_state_included"] = False
    return checkpoint


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _authenticated_json_mapping(source, *, name: str) -> tuple[dict, Optional[Path]]:
    if isinstance(source, Mapping):
        return dict(source), None
    path = Path(source).resolve()
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ConstrainedV23TrainerError(f"cannot read {name}") from error
    if not isinstance(value, Mapping):
        raise ConstrainedV23TrainerError(f"{name} must contain a mapping")
    return dict(value), path


def _verify_self_hash(payload: Mapping, *, hash_field: str, name: str) -> str:
    received = payload.get(hash_field)
    if not isinstance(received, str) or len(received) != 64:
        raise ConstrainedV23TrainerError(f"{name} has an invalid {hash_field}")
    canonical = dict(payload)
    canonical.pop(hash_field, None)
    if contract_hash(canonical) != received:
        raise ConstrainedV23TrainerError(f"{name} self-hash mismatch")
    return received


def _independently_reconstructed_contract(*, device: str) -> dict:
    if device not in ("cpu", "cuda"):
        raise ConstrainedV23TrainerError(
            "V2.3 diagnostic contract device must be exactly cpu or cuda"
        )
    trusted_args = argparse.Namespace(
        output_dir=Path("__v2_3_contract_authentication_only__"),
        model_seed=DEFAULT_MODEL_SEED,
        episodes=TOTAL_EPISODES,
        train_seed_base=FRESH_TRAIN_SEED_BASE,
        validation_seeds=DEFAULT_VALIDATION_SEEDS,
        validation_every=VALIDATION_EVERY,
        max_steps=2_000,
        grid_rows=5,
        grid_cols=5,
        number_blocks=8,
        arrival_rate=10.0,
        proc_mean=80,
        gamma_operational=1.0,
        reward_scale=0.01,
        rehandle_budget_per_100=20.0,
        dual_lr=0.01,
        lambda_initial=0.0,
        lambda_max=20.0,
        max_hold_steps=10,
        max_idle_steps=20,
        device=device,
        contract_only=False,
    )
    return build_training_contract(trusted_args)


def _authenticate_frozen_v23_contract(contract) -> tuple[dict, Optional[Path], str]:
    contract_payload, resolved_contract_path = _authenticated_json_mapping(
        contract, name="V2.3 training contract"
    )
    contract_sha = _verify_self_hash(
        contract_payload,
        hash_field="contract_sha256",
        name="V2.3 training contract",
    )
    contract_expected = {
        "checkpoint_family": CHECKPOINT_FAMILY,
        "method_version": METHOD_VERSION,
        "training_protocol": TRAINING_PROTOCOL,
        "model_seed": DEFAULT_MODEL_SEED,
        "train_seed_base": FRESH_TRAIN_SEED_BASE,
        "episodes": TOTAL_EPISODES,
        "validation_seeds": _json_safe(DEFAULT_VALIDATION_SEEDS),
        "gamma_operational": 1.0,
        "gamma_rehandle": 1.0,
        "reward_scale": 0.01,
        "dense_objective_spec": _json_safe(FROZEN_OBJECTIVE_SPEC.to_dict()),
        "development_panel_reused_after_v2_2": True,
        "training_instances_reused_after_v2_2": True,
        "fresh_prospective_experiment": False,
        "development_only": True,
        "final_86xxx_panel_opened": False,
    }
    contract_mismatches = {
        key: (contract_payload.get(key), value)
        for key, value in contract_expected.items()
        if _json_safe(contract_payload.get(key)) != value
    }
    if contract_mismatches:
        raise ConstrainedV23TrainerError(
            f"V2.3 diagnostic contract mismatch: {contract_mismatches!r}"
        )
    expected_parent_binding = {
        "training_contract_canonical_sha256": PARENT_V2_2_CONTRACT_SHA256,
        "validation_manifest_canonical_sha256": PARENT_V2_2_VALIDATION_MANIFEST_SHA256,
        "exact_validation_instance_identities_required": True,
        "validation_instance_identities": PARENT_V2_2_VALIDATION_INSTANCE_IDENTITIES,
    }
    if _json_safe(contract_payload.get("parent_v2_2_binding")) != _json_safe(
        expected_parent_binding
    ):
        raise ConstrainedV23TrainerError("V2.3 parent binding mismatch")
    trusted_contract = _independently_reconstructed_contract(
        device=contract_payload.get("device")
    )
    if _json_safe(contract_payload) != _json_safe(trusted_contract):
        differences = sorted(
            key
            for key in set(contract_payload) | set(trusted_contract)
            if _json_safe(contract_payload.get(key))
            != _json_safe(trusted_contract.get(key))
        )
        raise ConstrainedV23TrainerError(
            f"V2.3 contract provenance mismatch: {differences!r}"
        )
    return contract_payload, resolved_contract_path, contract_sha


def load_candidate_look_diagnostic(
    checkpoint_path,
    *,
    manifest_path,
    contract,
    device="cpu",
) -> dict:
    """Fail-closed loader for one immutable V2.3 candidate-look diagnostic."""

    contract_payload, _, contract_sha = _authenticate_frozen_v23_contract(contract)

    manifest_payload, resolved_manifest_path = _authenticated_json_mapping(
        manifest_path, name="candidate-look manifest"
    )
    if resolved_manifest_path is None:
        raise ConstrainedV23TrainerError("candidate-look manifest must be loaded from a file")
    manifest_sha = _verify_self_hash(
        manifest_payload,
        hash_field="manifest_sha256",
        name="candidate-look manifest",
    )
    _require_exact_keys(
        manifest_payload,
        _CANDIDATE_LOOK_MANIFEST_KEYS,
        name="candidate-look manifest",
    )
    manifest_expected = {
        "schema_version": 1,
        "checkpoint_family": CHECKPOINT_FAMILY,
        "training_contract_sha256": contract_sha,
        "model_only": True,
        "diagnostic_only": True,
        "final_86xxx_panel_opened": False,
    }
    manifest_mismatches = {
        key: (manifest_payload.get(key), value)
        for key, value in manifest_expected.items()
        if manifest_payload.get(key) != value
    }
    if manifest_mismatches:
        raise ConstrainedV23TrainerError(
            f"candidate-look manifest contract mismatch: {manifest_mismatches!r}"
        )
    artifacts = manifest_payload.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise ConstrainedV23TrainerError("candidate-look manifest has no artifacts")
    if manifest_payload.get("artifact_count") != len(artifacts):
        raise ConstrainedV23TrainerError("candidate-look manifest count mismatch")

    root = resolved_manifest_path.parent.resolve()
    requested = Path(checkpoint_path).resolve()
    matching = []
    seen_paths = set()
    seen_episodes = set()
    for artifact in artifacts:
        if not isinstance(artifact, Mapping):
            raise ConstrainedV23TrainerError("candidate-look manifest artifact is invalid")
        _require_exact_keys(
            artifact,
            _CANDIDATE_LOOK_ARTIFACT_KEYS,
            name="candidate-look manifest artifact",
        )
        relative = artifact.get("relative_path")
        episode = artifact.get("checkpoint_episode")
        if (
            not isinstance(relative, str)
            or not relative
            or Path(relative).is_absolute()
            or isinstance(episode, bool)
            or not isinstance(episode, int)
            or episode not in CANDIDATE_EPISODES
        ):
            raise ConstrainedV23TrainerError("candidate-look manifest identity is invalid")
        canonical_relative = (
            f"candidate-look-checkpoints/episode-{episode:04d}-model-only.pth"
        )
        if relative != canonical_relative:
            raise ConstrainedV23TrainerError(
                "candidate-look manifest relative path is non-canonical"
            )
        resolved = (root / relative).resolve()
        if not resolved.is_relative_to(root):
            raise ConstrainedV23TrainerError("candidate-look path escapes manifest root")
        if relative in seen_paths or episode in seen_episodes:
            raise ConstrainedV23TrainerError("candidate-look manifest contains duplicates")
        seen_paths.add(relative)
        seen_episodes.add(episode)
        if resolved == requested:
            matching.append((dict(artifact), resolved))
    saved = manifest_payload.get("saved_candidate_look_episodes")
    expected_looks = manifest_payload.get("expected_candidate_look_episodes")
    artifact_episodes = tuple(
        int(artifact["checkpoint_episode"]) for artifact in artifacts
    )
    if (
        tuple(expected_looks) if isinstance(expected_looks, list) else None
    ) != CANDIDATE_EPISODES:
        raise ConstrainedV23TrainerError("candidate-look manifest expected lifecycle mismatch")
    if (tuple(saved) if isinstance(saved, list) else None) != artifact_episodes:
        raise ConstrainedV23TrainerError("candidate-look manifest saved lifecycle mismatch")
    if artifact_episodes != CANDIDATE_EPISODES[: len(artifact_episodes)]:
        raise ConstrainedV23TrainerError("candidate-look manifest episode order mismatch")
    expected_complete = len(artifact_episodes) == len(CANDIDATE_EPISODES)
    if manifest_payload.get("complete") is not expected_complete:
        raise ConstrainedV23TrainerError("candidate-look manifest completion mismatch")
    if len(matching) != 1:
        raise ConstrainedV23TrainerError(
            "checkpoint path is not exactly contained in the candidate-look manifest"
        )
    artifact, resolved_checkpoint = matching[0]
    artifact_expected = {
        "checkpoint_role": "candidate_look_diagnostic",
        "model_only": True,
        "diagnostic_only": True,
        "development_candidate_eligible": False,
        "deployment_checkpoint_eligible": False,
    }
    artifact_mismatches = {
        key: (artifact.get(key), value)
        for key, value in artifact_expected.items()
        if artifact.get(key) != value
    }
    if artifact_mismatches:
        raise ConstrainedV23TrainerError(
            f"candidate-look artifact flags mismatch: {artifact_mismatches!r}"
        )
    if not resolved_checkpoint.is_file():
        raise ConstrainedV23TrainerError("candidate-look checkpoint file is missing")
    if _file_sha256(resolved_checkpoint) != artifact.get("sha256"):
        raise ConstrainedV23TrainerError("candidate-look checkpoint file SHA mismatch")

    try:
        payload = torch.load(
            resolved_checkpoint,
            map_location=device,
            weights_only=True,
        )
    except Exception as error:
        raise ConstrainedV23TrainerError("cannot safely load candidate-look checkpoint") from error
    if not isinstance(payload, Mapping):
        raise ConstrainedV23TrainerError("candidate-look checkpoint is not a mapping")
    payload = dict(payload)
    _require_exact_keys(
        payload,
        _CANDIDATE_LOOK_OUTER_KEYS,
        name="candidate-look checkpoint envelope",
    )
    episode = int(artifact["checkpoint_episode"])
    outer_expected = {
        "checkpoint_family": CHECKPOINT_FAMILY,
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "method_version": METHOD_VERSION,
        "training_protocol": TRAINING_PROTOCOL,
        "backup_version": BACKUP_VERSION,
        "policy_schedule_protocol": POLICY_SCHEDULE_PROTOCOL,
        "dual_update_protocol": DUAL_UPDATE_PROTOCOL,
        "validation_protocol": VALIDATION_PROTOCOL,
        "checkpoint_role": "candidate_look_diagnostic",
        "training_contract_sha256": contract_sha,
        "completed_episodes": episode,
        "resumable": False,
        "development_candidate_eligible": False,
        "deployment_checkpoint_eligible": False,
        "diagnostic_only": True,
        "model_only": True,
        "immutable_unique_candidate_look_artifact": True,
        "validation_authenticates_checkpoint_lambda": True,
        "same_policy_for_behavior_backup_validation_deployment": True,
        "validation_stochastic_only": True,
        "map_validation_used": False,
        "final_86xxx_panel_opened": None,
    }
    # ``final_86xxx_panel_opened`` belongs to the authenticated contract and
    # manifest; the checkpoint envelope must not claim a contradictory value.
    if "final_86xxx_panel_opened" in payload:
        outer_expected["final_86xxx_panel_opened"] = False
    else:
        outer_expected.pop("final_86xxx_panel_opened")
    outer_mismatches = {
        key: (payload.get(key), value)
        for key, value in outer_expected.items()
        if payload.get(key) != value
    }
    if outer_mismatches:
        raise ConstrainedV23TrainerError(
            f"candidate-look checkpoint envelope mismatch: {outer_mismatches!r}"
        )
    validation = payload.get("validation_summary")
    nested = payload.get("agent_state")
    if not isinstance(validation, Mapping) or not isinstance(nested, Mapping):
        raise ConstrainedV23TrainerError(
            "candidate-look checkpoint lacks validation or nested model state"
        )
    _require_exact_keys(
        nested,
        _MODEL_ONLY_NESTED_CHECKPOINT_KEYS,
        name="nested candidate-look checkpoint envelope",
    )
    expected_schedule = schedule_for_episode(episode, validation=True).to_dict()
    schedules = (
        payload.get("schedule_state"),
        validation.get("schedule_state"),
        nested.get("schedule_state"),
        nested.get("agent_state", {}).get("schedule_state")
        if isinstance(nested.get("agent_state"), Mapping)
        else None,
    )
    if any(dict(value) != expected_schedule for value in schedules if isinstance(value, Mapping)) or any(
        not isinstance(value, Mapping) for value in schedules
    ):
        raise ConstrainedV23TrainerError("candidate-look schedule authentication failed")
    if (
        validation.get("checkpoint_episode") != episode
        or validation.get("candidate_look_gate") is not True
    ):
        raise ConstrainedV23TrainerError("candidate-look validation identity mismatch")
    if artifact.get("validation_development_candidate_eligible") is not bool(
        validation.get("development_candidate_eligible")
    ):
        raise ConstrainedV23TrainerError(
            "candidate-look artifact/validation eligibility mismatch"
        )
    lambda_values = (
        payload.get("runtime_lambda"),
        payload.get("validated_policy_lambda"),
        validation.get("validation_lambda"),
        payload.get("dual_state", {}).get("lambda_value")
        if isinstance(payload.get("dual_state"), Mapping)
        else None,
        nested.get("agent_state", {}).get("dual_lambda")
        if isinstance(nested.get("agent_state"), Mapping)
        else None,
    )
    if any(isinstance(value, bool) or not isinstance(value, Real) for value in lambda_values):
        raise ConstrainedV23TrainerError("candidate-look lambda authentication failed")
    if any(float(value) != float(lambda_values[0]) for value in lambda_values[1:]):
        raise ConstrainedV23TrainerError("candidate-look lambda authentication failed")
    if nested.get("model_only") is not True or nested.get("optimizer_included") is not False:
        raise ConstrainedV23TrainerError("nested candidate-look checkpoint is not model-only")
    if nested.get("replay_included") is not False or nested.get("training_rng_state_included") is not False:
        raise ConstrainedV23TrainerError("nested candidate-look checkpoint includes training state")
    inner_state = nested.get("agent_state")
    if not isinstance(inner_state, Mapping) or set(inner_state) != _MODEL_ONLY_AGENT_STATE_KEYS:
        raise ConstrainedV23TrainerError("nested candidate-look agent state keys are not model-only")
    if dict(nested.get("config", {})) != dict(contract_payload.get("agent_config", {})):
        raise ConstrainedV23TrainerError("nested candidate-look agent config mismatch")

    from viability_graph_constrained_v2_3 import ConstrainedV23HierarchyAgent

    try:
        agent = ConstrainedV23HierarchyAgent.from_checkpoint(
            nested,
            device=device,
            resumable=False,
            seed=DEFAULT_MODEL_SEED,
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ConstrainedV23TrainerError(
            "nested candidate-look V2.3 checkpoint authentication failed"
        ) from error
    return {
        "agent": agent,
        "checkpoint": payload,
        "artifact": artifact,
        "contract_sha256": contract_sha,
        "manifest_sha256": manifest_sha,
        "checkpoint_sha256": artifact["sha256"],
    }


def _require_sha256(value, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise ConstrainedV23TrainerError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _checkpoint_values_identical(left, right) -> bool:
    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        return (
            isinstance(left, torch.Tensor)
            and isinstance(right, torch.Tensor)
            and left.dtype == right.dtype
            and tuple(left.shape) == tuple(right.shape)
            and torch.equal(left, right)
        )
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        return (
            isinstance(left, Mapping)
            and isinstance(right, Mapping)
            and set(left) == set(right)
            and all(_checkpoint_values_identical(left[key], right[key]) for key in left)
        )
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        return (
            type(left) is type(right)
            and len(left) == len(right)
            and all(_checkpoint_values_identical(a, b) for a, b in zip(left, right))
        )
    return type(left) is type(right) and left == right


def _trusted_runtime_core_contract(contract: Mapping) -> dict:
    return {
        "controller": CONTROLLER_ARCHITECTURE,
        "checkpoint_family": CHECKPOINT_FAMILY,
        "backup_version": BACKUP_VERSION,
        "policy_schedule_protocol": POLICY_SCHEDULE_PROTOCOL,
        "validation_protocol": VALIDATION_PROTOCOL,
        "training_policy": POLICY_REALIZATION,
        "backup_policy": POLICY_REALIZATION,
        "validation_policy": POLICY_REALIZATION,
        "deployment_policy": POLICY_REALIZATION,
        "same_policy_for_behavior_backup_validation_deployment": True,
        "map_policy_available": False,
        "policy_realization_decoupled_from_training_flag": True,
        "single_shared_lagrangian_policy_for_both_target_heads": True,
        "entropy_or_kl_added_to_raw_component_targets": False,
        "behavior_rng_scope": "per_rollout_action_sampling_only",
        "replay_rng_scope": "persistent_replay_sampling_only",
        "replay_rng_seed": REPLAY_RNG_SEED,
        "hold_frontier_interface": CERTIFIED_HOLD_INTERFACE_V2,
        "hold_rule": CERTIFIED_HOLD_RULE_V2,
        "cost_definition": COST_DEFINITION,
        "scalarized_reward_stored_in_replay": False,
        "operational_gamma": 1.0,
        "rehandle_gamma": 1.0,
        "operational_reward_scale": 0.01,
        "dual_update_authority": "trainer_complete_active_block_residual",
        "exact_safe_frontier_authoritative": True,
        "baseline_teacher": False,
        "baseline_policy_query": False,
        "agent_config": dict(contract["agent_config"]),
        "viability_search_config": dict(FROZEN_VIABILITY_SEARCH_CONFIG),
        "fresh_certificate_cache_per_episode": True,
    }


def _authenticate_validation_instance_manifest(
    manifest_path, *, root: Path
) -> tuple[dict, str]:
    payload, resolved_path = _authenticated_json_mapping(
        manifest_path, name="V2.3 validation-instance manifest"
    )
    if resolved_path != (root / "validation-instance-manifest.json").resolve():
        raise ConstrainedV23TrainerError(
            "validation-instance manifest path is not canonical or colocated"
        )
    _require_exact_keys(
        payload,
        _VALIDATION_INSTANCE_MANIFEST_KEYS,
        name="validation-instance manifest",
    )
    manifest_sha = _verify_self_hash(
        payload,
        hash_field="manifest_sha256",
        name="validation-instance manifest",
    )
    expected = {
        "schema_version": 1,
        "validation_protocol": VALIDATION_PROTOCOL,
        "common_grid_every_checkpoint": True,
        "parent_v2_2_validation_manifest_sha256": PARENT_V2_2_VALIDATION_MANIFEST_SHA256,
        "exact_parent_v2_2_instance_identity_match": True,
        "instances": PARENT_V2_2_VALIDATION_INSTANCE_IDENTITIES,
        "prospective_83xxx_panel_opened": False,
        "final_86xxx_panel_opened": False,
    }
    expected["manifest_sha256"] = contract_hash(expected)
    if _json_safe(payload) != _json_safe(expected):
        raise ConstrainedV23TrainerError(
            "validation-instance manifest does not match the frozen V2.2-bound grid"
        )
    return payload, manifest_sha


def _authenticate_candidate_validation_ledger(
    *, root: Path, diagnostic: Mapping, contract: Mapping,
    validation_instance_manifest: Mapping,
) -> dict:
    checkpoint = diagnostic.get("checkpoint")
    if not isinstance(checkpoint, Mapping):
        raise ConstrainedV23TrainerError("candidate diagnostic checkpoint is unavailable")
    validation = checkpoint.get("validation_summary")
    if not isinstance(validation, Mapping):
        raise ConstrainedV23TrainerError("candidate diagnostic validation is unavailable")
    episode = _positive_int(
        checkpoint.get("completed_episodes"), name="candidate validation episode"
    )
    ledger_reference = validation.get("validation_ledger")
    if not isinstance(ledger_reference, Mapping):
        raise ConstrainedV23TrainerError("candidate validation lacks a ledger reference")
    _require_exact_keys(
        ledger_reference,
        _VALIDATION_LEDGER_REFERENCE_KEYS,
        name="candidate validation ledger reference",
    )
    canonical_relative = f"validation-ledger/episode-{episode:04d}.json"
    if ledger_reference.get("relative_path") != canonical_relative:
        raise ConstrainedV23TrainerError("candidate validation ledger path is non-canonical")
    ledger_path = (root / canonical_relative).resolve()
    if not ledger_path.is_relative_to(root) or not ledger_path.is_file():
        raise ConstrainedV23TrainerError("candidate validation ledger file is missing")
    ledger, resolved_ledger_path = _authenticated_json_mapping(
        ledger_path, name="candidate validation ledger"
    )
    if resolved_ledger_path != ledger_path:
        raise ConstrainedV23TrainerError("candidate validation ledger path drifted")
    _require_exact_keys(
        ledger, _VALIDATION_LEDGER_KEYS, name="candidate validation ledger"
    )
    ledger_sha = _verify_self_hash(
        ledger,
        hash_field="ledger_sha256",
        name="candidate validation ledger",
    )
    _require_sha256(
        ledger_reference.get("ledger_sha256"),
        name="candidate validation ledger reference SHA",
    )
    if ledger_sha != ledger_reference.get("ledger_sha256"):
        raise ConstrainedV23TrainerError("candidate validation ledger SHA binding failed")

    expected_schedule = schedule_for_episode(episode, validation=True).to_dict()
    expected_rows = len(DEFAULT_VALIDATION_SEEDS) * VALIDATION_POLICY_RNG_COUNT
    ledger_expected = {
        "schema_version": 1,
        "validation_protocol": VALIDATION_PROTOCOL,
        "checkpoint_episode": episode,
        "validated_lambda": validation.get("validation_lambda"),
        "schedule_state": expected_schedule,
        "row_count": expected_rows,
        "complete_case_filtering_used": False,
    }
    ledger_mismatches = {
        key: (ledger.get(key), value)
        for key, value in ledger_expected.items()
        if _json_safe(ledger.get(key)) != _json_safe(value)
    }
    if ledger_mismatches:
        raise ConstrainedV23TrainerError(
            f"candidate validation ledger metadata mismatch: {ledger_mismatches!r}"
        )
    if ledger_reference.get("row_count") != expected_rows:
        raise ConstrainedV23TrainerError("candidate validation ledger row-count binding failed")
    batch_audit = ledger.get("validation_batch_audit")
    if not isinstance(batch_audit, Mapping):
        raise ConstrainedV23TrainerError("candidate validation batch audit is missing")
    _require_exact_keys(
        batch_audit,
        _VALIDATION_BATCH_AUDIT_KEYS,
        name="candidate validation batch audit",
    )
    digest_before = batch_audit.get("full_training_state_digest_before")
    if (
        batch_audit.get("validation_batch_ended") is not True
        or batch_audit.get("training_agent_unchanged") is not True
        or not isinstance(digest_before, str)
        or not digest_before
        or batch_audit.get("full_training_state_digest_after") != digest_before
    ):
        raise ConstrainedV23TrainerError(
            "candidate validation batch mutation audit failed"
        )
    if _json_safe(validation.get("validation_batch_audit")) != _json_safe(batch_audit):
        raise ConstrainedV23TrainerError(
            "candidate validation summary/batch-audit binding failed"
        )

    rows = ledger.get("rows")
    if not isinstance(rows, list) or len(rows) != expected_rows:
        raise ConstrainedV23TrainerError("candidate validation ledger grid is incomplete")
    identities = {
        int(item["instance_seed"]): dict(item)
        for item in validation_instance_manifest["instances"]
    }
    expected_coordinates = tuple(
        (instance_index, int(instance_seed), rng_index)
        for instance_index, instance_seed in enumerate(DEFAULT_VALIDATION_SEEDS)
        for rng_index in range(VALIDATION_POLICY_RNG_COUNT)
    )
    normalized_rows = []
    for row_index, (row, coordinates) in enumerate(zip(rows, expected_coordinates)):
        if not isinstance(row, Mapping):
            raise ConstrainedV23TrainerError("candidate validation ledger row is invalid")
        instance_index, instance_seed, rng_index = coordinates
        row_expected = {
            "instance_index": instance_index,
            "instance_seed": instance_seed,
            "policy_rng_index": rng_index,
            "policy_rng_seed": validation_policy_rng_seed(instance_index, rng_index),
            "policy_rng_scope": "action_sampling_only",
            "method_version": METHOD_VERSION,
            "policy_schedule_protocol": POLICY_SCHEDULE_PROTOCOL,
            "policy_mode": POLICY_REALIZATION,
            "execution_context": expected_schedule["execution_context"],
            "dual_lambda": float(validation["validation_lambda"]),
            "replay_rng_seed": REPLAY_RNG_SEED,
            "evaluation_learning": False,
            "training_agent_unchanged": True,
            "validation_batch_state_unchanged": True,
            "fresh_evaluation_clone": True,
            "stochastic_selection_only": True,
            "option_evaluation_mode": True,
            "map_selection_used": False,
            "behavior_policy_rng_reset": True,
            "baseline_teacher": False,
            "baseline_policy_query": False,
        }
        row_mismatches = {
            key: (row.get(key), value)
            for key, value in row_expected.items()
            if row.get(key) != value
        }
        identity = identities[instance_seed]
        for key in ("episode_instance_id", "schedule_id", "episode_instance_sha256"):
            if row.get(key) != identity[key]:
                row_mismatches[key] = (row.get(key), identity[key])
        if _json_safe(row.get("schedule_state")) != _json_safe(expected_schedule):
            row_mismatches["schedule_state"] = (
                row.get("schedule_state"), expected_schedule
            )
        if _json_safe(row.get("within_group_temperatures")) != _json_safe(
            expected_schedule["within_group_temperatures"]
        ):
            row_mismatches["within_group_temperatures"] = (
                row.get("within_group_temperatures"),
                expected_schedule["within_group_temperatures"],
            )
        if row.get("group_temperature") != expected_schedule["group_temperature"]:
            row_mismatches["group_temperature"] = (
                row.get("group_temperature"), expected_schedule["group_temperature"]
            )
        if row_mismatches:
            raise ConstrainedV23TrainerError(
                f"candidate validation ledger row {row_index} mismatch: {row_mismatches!r}"
            )
        try:
            normalized = _normalize_authenticated_run(
                row,
                index=row_index,
                schedule=expected_schedule,
                dual_lambda=float(validation["validation_lambda"]),
                training=False,
            )
        except (KeyError, TypeError, ValueError, RuntimeError) as error:
            raise ConstrainedV23TrainerError(
                f"candidate validation ledger row {row_index} failed authentication"
            ) from error
        if _json_safe(normalized) != _json_safe(row):
            raise ConstrainedV23TrainerError(
                f"candidate validation ledger row {row_index} is not canonical"
            )
        normalized_rows.append(normalized)

    try:
        dual_config = V23DualConfig(**dict(contract["dual"]))
        recomputed = summarize_validation(
            normalized_rows,
            dual_config,
            checkpoint_episode=episode,
            validation_lambda=float(validation["validation_lambda"]),
            schedule_state=expected_schedule,
            validation_seeds=DEFAULT_VALIDATION_SEEDS,
            positive_residual_saturation_count=int(
                validation["positive_residual_dual_saturation_count_before_validation"]
            ),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ConstrainedV23TrainerError(
            "candidate validation summary cannot be recomputed"
        ) from error
    recomputed["validated_before_dual_update"] = True
    recomputed["validation_batch_audit"] = dict(batch_audit)
    recomputed["validation_ledger"] = dict(ledger_reference)
    if _json_safe(recomputed) != _json_safe(validation):
        raise ConstrainedV23TrainerError(
            "candidate validation summary does not match its authenticated ledger"
        )
    return {
        "ledger": ledger,
        "ledger_sha256": ledger_sha,
        "summary": recomputed,
    }


def _authenticate_candidate_dual_lifecycle(
    checkpoint: Mapping, *, contract: Mapping, episode: int
) -> None:
    dual_state = checkpoint.get("dual_state")
    pending = checkpoint.get("pending_dual_batch")
    validation = checkpoint.get("validation_summary")
    if not all(isinstance(value, Mapping) for value in (dual_state, pending, validation)):
        raise ConstrainedV23TrainerError("candidate dual lifecycle is incomplete")
    _require_exact_keys(dual_state, _DUAL_STATE_KEYS, name="candidate dual state")
    _require_exact_keys(
        pending, _PENDING_DUAL_BATCH_KEYS, name="candidate pending dual batch"
    )
    block_number = episode // BLOCK_EPISODES
    update_count = block_number - WARMUP_BLOCKS - 1
    dual_expected = {
        "config": contract["dual"],
        "update_count": update_count,
        "last_completed_block": block_number - 1,
        "active_episodes_consumed": update_count * BLOCK_EPISODES,
        "protocol": DUAL_UPDATE_PROTOCOL,
        "saturation_count": validation[
            "positive_residual_dual_saturation_count_before_validation"
        ],
    }
    dual_mismatches = {
        key: (dual_state.get(key), value)
        for key, value in dual_expected.items()
        if _json_safe(dual_state.get(key)) != _json_safe(value)
    }
    if dual_mismatches:
        raise ConstrainedV23TrainerError(
            f"candidate dual-state lifecycle mismatch: {dual_mismatches!r}"
        )
    expected_episode_numbers = tuple(range(episode - BLOCK_EPISODES + 1, episode + 1))
    residuals = pending.get("residuals")
    if (
        pending.get("episode_count") != BLOCK_EPISODES
        or tuple(pending.get("episode_numbers", ())) != expected_episode_numbers
        or not isinstance(residuals, (list, tuple))
        or len(residuals) != BLOCK_EPISODES
        or any(not math.isfinite(_finite(value, name="pending dual residual")) for value in residuals)
        or pending.get("audit_only_nonresumable") is not True
        or checkpoint.get("terminal_dual_proposal") is not None
    ):
        raise ConstrainedV23TrainerError("candidate pending dual lifecycle mismatch")


def load_best_development_candidate(
    checkpoint_path,
    *,
    expected_best_sha256,
    manifest_path,
    expected_manifest_sha256,
    validation_instance_manifest_path,
    contract,
    device="cpu",
) -> dict:
    """Authenticate and load the selected V2.3 development candidate.

    The two expected digests must come from an independent completed-run audit.
    A manifest self-hash is integrity metadata, not an authenticity root, so
    neither digest is inferred from the files being authenticated.
    """

    expected_best_sha = _require_sha256(
        expected_best_sha256, name="expected best-checkpoint SHA"
    )
    expected_manifest_sha = _require_sha256(
        expected_manifest_sha256, name="expected candidate-manifest SHA"
    )
    resolved_best_path = Path(checkpoint_path).resolve()
    root = resolved_best_path.parent.resolve()
    if resolved_best_path != (root / "best-development-candidate.pth").resolve():
        raise ConstrainedV23TrainerError("best-development checkpoint path is non-canonical")
    if not resolved_best_path.is_file():
        raise ConstrainedV23TrainerError("best-development checkpoint file is missing")
    try:
        best_bytes = resolved_best_path.read_bytes()
    except OSError as error:
        raise ConstrainedV23TrainerError("cannot read best-development checkpoint") from error
    observed_best_sha = hashlib.sha256(best_bytes).hexdigest()
    if observed_best_sha != expected_best_sha:
        raise ConstrainedV23TrainerError("best-development checkpoint raw file SHA mismatch")

    contract_payload, resolved_contract_path, contract_sha = (
        _authenticate_frozen_v23_contract(contract)
    )
    if resolved_contract_path != (root / "training-contract.json").resolve():
        raise ConstrainedV23TrainerError(
            "training contract path is not canonical or colocated"
        )
    manifest_payload, resolved_manifest_path = _authenticated_json_mapping(
        manifest_path, name="candidate-look manifest"
    )
    if resolved_manifest_path != (
        root / "candidate-look-checkpoint-manifest.json"
    ).resolve():
        raise ConstrainedV23TrainerError(
            "candidate-look manifest path is not canonical or colocated"
        )
    _require_exact_keys(
        manifest_payload,
        _CANDIDATE_LOOK_MANIFEST_KEYS,
        name="candidate-look manifest",
    )
    observed_manifest_sha = _verify_self_hash(
        manifest_payload,
        hash_field="manifest_sha256",
        name="candidate-look manifest",
    )
    if observed_manifest_sha != expected_manifest_sha:
        raise ConstrainedV23TrainerError(
            "candidate-look manifest does not match the externally audited SHA"
        )
    validation_instance_manifest, validation_instance_manifest_sha = (
        _authenticate_validation_instance_manifest(
            validation_instance_manifest_path, root=root
        )
    )

    artifacts = manifest_payload.get("artifacts")
    if not isinstance(artifacts, list) or len(artifacts) != len(CANDIDATE_EPISODES):
        raise ConstrainedV23TrainerError(
            "best selection requires the complete seven-look candidate manifest"
        )
    trusted_core = _trusted_runtime_core_contract(contract_payload)
    candidate_records = []
    selected_summary = None
    selected_record = None
    for artifact in artifacts:
        if not isinstance(artifact, Mapping):
            raise ConstrainedV23TrainerError("candidate manifest artifact is invalid")
        diagnostic_path = (root / str(artifact.get("relative_path", ""))).resolve()
        diagnostic = load_candidate_look_diagnostic(
            diagnostic_path,
            manifest_path=resolved_manifest_path,
            contract=resolved_contract_path,
            device=device,
        )
        if (
            diagnostic.get("manifest_sha256") != expected_manifest_sha
            or diagnostic.get("contract_sha256") != contract_sha
        ):
            raise ConstrainedV23TrainerError(
                "candidate diagnostic trust roots changed during authentication"
            )
        diagnostic_checkpoint = diagnostic["checkpoint"]
        episode = int(diagnostic_checkpoint["completed_episodes"])
        if _json_safe(diagnostic_checkpoint.get("core_contract")) != _json_safe(
            trusted_core
        ):
            raise ConstrainedV23TrainerError(
                "candidate diagnostic runtime core contract mismatch"
            )
        _authenticate_candidate_dual_lifecycle(
            diagnostic_checkpoint, contract=contract_payload, episode=episode
        )
        validation_record = _authenticate_candidate_validation_ledger(
            root=root,
            diagnostic=diagnostic,
            contract=contract_payload,
            validation_instance_manifest=validation_instance_manifest,
        )
        record = {
            **diagnostic,
            "validation_ledger": validation_record["ledger"],
            "validation_ledger_sha256": validation_record["ledger_sha256"],
            "validation_summary": validation_record["summary"],
        }
        candidate_records.append(record)
        incumbent = selected_summary
        selected_summary = select_better_validation(
            selected_summary, validation_record["summary"]
        )
        if selected_summary is not incumbent:
            selected_record = record
    if selected_summary is None or selected_record is None:
        raise ConstrainedV23TrainerError(
            "candidate manifest contains no eligible development checkpoint"
        )

    try:
        payload = torch.load(
            io.BytesIO(best_bytes), map_location=device, weights_only=True
        )
    except Exception as error:
        raise ConstrainedV23TrainerError(
            "cannot safely load best-development checkpoint"
        ) from error
    if not isinstance(payload, Mapping):
        raise ConstrainedV23TrainerError("best-development checkpoint is not a mapping")
    payload = dict(payload)
    _require_exact_keys(
        payload,
        _CANDIDATE_LOOK_OUTER_KEYS,
        name="best-development checkpoint envelope",
    )
    selected_episode = int(selected_summary["checkpoint_episode"])
    if payload.get("completed_episodes") != selected_episode:
        raise ConstrainedV23TrainerError(
            "best-development checkpoint fails the recomputed frozen selection"
        )
    outer_expected = {
        "checkpoint_family": CHECKPOINT_FAMILY,
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "checkpoint_role": "best_development_candidate",
        "resumable": False,
        "development_candidate_eligible": True,
        "deployment_checkpoint_eligible": False,
        "diagnostic_only": False,
        "model_only": False,
        "immutable_unique_candidate_look_artifact": False,
        "completed_episodes": selected_episode,
        "training_contract_sha256": contract_sha,
        "method_version": METHOD_VERSION,
        "training_protocol": TRAINING_PROTOCOL,
        "backup_version": BACKUP_VERSION,
        "policy_schedule_protocol": POLICY_SCHEDULE_PROTOCOL,
        "dual_update_protocol": DUAL_UPDATE_PROTOCOL,
        "validation_protocol": VALIDATION_PROTOCOL,
        "validation_authenticates_checkpoint_lambda": True,
        "same_policy_for_behavior_backup_validation_deployment": True,
        "validation_stochastic_only": True,
        "map_validation_used": False,
        "terminal_dual_proposal": None,
    }
    outer_mismatches = {
        key: (payload.get(key), value)
        for key, value in outer_expected.items()
        if payload.get(key) != value
    }
    if outer_mismatches:
        raise ConstrainedV23TrainerError(
            f"best-development checkpoint envelope mismatch: {outer_mismatches!r}"
        )
    if _json_safe(payload.get("core_contract")) != _json_safe(trusted_core):
        raise ConstrainedV23TrainerError("best-development runtime core contract mismatch")
    if _json_safe(payload.get("validation_summary")) != _json_safe(selected_summary):
        raise ConstrainedV23TrainerError(
            "best-development validation is not the authenticated selected validation"
        )
    expected_schedule = schedule_for_episode(selected_episode, validation=True).to_dict()
    nested = payload.get("agent_state")
    selected_nested = selected_record["checkpoint"].get("agent_state")
    if not isinstance(nested, Mapping) or not isinstance(selected_nested, Mapping):
        raise ConstrainedV23TrainerError("best-development nested checkpoint is missing")
    _require_exact_keys(
        nested,
        _BEST_NESTED_CHECKPOINT_KEYS,
        name="nested best-development checkpoint envelope",
    )
    schedules = (
        payload.get("schedule_state"),
        payload.get("validation_summary", {}).get("schedule_state"),
        nested.get("schedule_state"),
        nested.get("agent_state", {}).get("schedule_state")
        if isinstance(nested.get("agent_state"), Mapping)
        else None,
    )
    if any(
        not isinstance(value, Mapping)
        or _json_safe(value) != _json_safe(expected_schedule)
        for value in schedules
    ):
        raise ConstrainedV23TrainerError(
            "best-development schedule authentication failed"
        )
    lambda_values = (
        payload.get("runtime_lambda"),
        payload.get("validated_policy_lambda"),
        selected_summary.get("validation_lambda"),
        payload.get("dual_state", {}).get("lambda_value")
        if isinstance(payload.get("dual_state"), Mapping)
        else None,
        nested.get("agent_state", {}).get("dual_lambda")
        if isinstance(nested.get("agent_state"), Mapping)
        else None,
        selected_record["checkpoint"].get("runtime_lambda"),
    )
    if any(isinstance(value, bool) or not isinstance(value, Real) for value in lambda_values):
        raise ConstrainedV23TrainerError("best-development lambda authentication failed")
    if any(float(value) != float(lambda_values[0]) for value in lambda_values[1:]):
        raise ConstrainedV23TrainerError("best-development lambda authentication failed")
    _authenticate_candidate_dual_lifecycle(
        payload, contract=contract_payload, episode=selected_episode
    )
    for key in ("dual_state", "pending_dual_batch", "terminal_dual_proposal"):
        if not _checkpoint_values_identical(
            payload.get(key), selected_record["checkpoint"].get(key)
        ):
            raise ConstrainedV23TrainerError(
                f"best-development {key} does not match selected diagnostic"
            )
    if dict(nested.get("config", {})) != dict(contract_payload["agent_config"]):
        raise ConstrainedV23TrainerError("nested best-development agent config mismatch")
    inner_state = nested.get("agent_state")
    selected_inner_state = selected_nested.get("agent_state")
    if not isinstance(inner_state, Mapping) or set(inner_state) != _BEST_AGENT_STATE_KEYS:
        raise ConstrainedV23TrainerError(
            "nested best-development agent-state schema mismatch"
        )
    if not isinstance(selected_inner_state, Mapping):
        raise ConstrainedV23TrainerError("selected diagnostic model state is missing")
    for key in _BEST_NESTED_CHECKPOINT_KEYS.difference({"agent_state"}):
        if not _checkpoint_values_identical(nested[key], selected_nested.get(key)):
            raise ConstrainedV23TrainerError(
                f"best-development policy metadata differs from selected diagnostic: {key}"
            )
    for key in _MODEL_ONLY_AGENT_STATE_KEYS:
        if not _checkpoint_values_identical(inner_state[key], selected_inner_state.get(key)):
            raise ConstrainedV23TrainerError(
                f"best-development model differs from selected diagnostic: {key}"
            )

    from viability_graph_constrained_v2_3 import ConstrainedV23HierarchyAgent

    try:
        agent = ConstrainedV23HierarchyAgent.from_checkpoint(
            nested,
            device=device,
            resumable=False,
            seed=DEFAULT_MODEL_SEED,
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ConstrainedV23TrainerError(
            "nested best-development V2.3 checkpoint authentication failed"
        ) from error
    return {
        "agent": agent,
        "checkpoint": payload,
        "checkpoint_sha256": observed_best_sha,
        "contract_sha256": contract_sha,
        "manifest_sha256": observed_manifest_sha,
        "validation_instance_manifest_sha256": validation_instance_manifest_sha,
        "selected_episode": selected_episode,
        "selected_diagnostic": selected_record,
        "candidate_records": tuple(candidate_records),
        "validation_summary": selected_summary,
    }


def _checkpoint_payload(*, contract: Mapping, runtime: V23DevelopmentRuntime,
                        dual: CompleteBlockProjectedDual, completed_episodes: int,
                        role: str, schedule_state: Mapping,
                        pending_dual_runs: Sequence[Mapping],
                        validation_summary: Optional[Mapping], include_replay: bool,
                        terminal_dual_proposal: Optional[Mapping] = None,
                        training_history: Optional[Sequence[Mapping]] = None,
                        validation_history: Optional[Sequence[Mapping]] = None) -> dict:
    allowed_roles = (
        "latest_development_state",
        "best_development_candidate",
        "candidate_look_diagnostic",
    )
    if role not in allowed_roles:
        raise ConstrainedV23TrainerError("invalid checkpoint role")
    validated_lambda = None if validation_summary is None else float(validation_summary["validation_lambda"])
    lambda_authenticated = bool(
        validated_lambda is not None
        and math.isclose(float(runtime.dual_lambda), validated_lambda, rel_tol=0.0, abs_tol=0.0)
    )
    # ``latest.pth`` is always an audit-only mutable training-state artifact.
    # Eligibility belongs exclusively to the immutable best-candidate role.
    diagnostic_only = role == "candidate_look_diagnostic"
    eligible = bool(
        role == "best_development_candidate"
        and validation_summary
        and validation_summary.get("development_candidate_eligible")
        and lambda_authenticated
    )
    if role == "best_development_candidate" and not eligible:
        raise ConstrainedV23TrainerError("best checkpoint requires an eligible stochastic validation")
    if diagnostic_only:
        if include_replay:
            raise ConstrainedV23TrainerError("candidate-look diagnostics must be model-only")
        if (
            validation_summary is None
            or int(completed_episodes) not in CANDIDATE_EPISODES
            or int(validation_summary.get("checkpoint_episode", -1)) != int(completed_episodes)
            or not lambda_authenticated
        ):
            raise ConstrainedV23TrainerError(
                "candidate-look diagnostic requires an authenticated candidate validation"
            )
    runtime_checkpoint = (
        _model_only_runtime_checkpoint(runtime)
        if diagnostic_only
        else runtime.checkpoint_state(include_replay=include_replay)
    )
    payload = {
        "checkpoint_family": CHECKPOINT_FAMILY,
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "checkpoint_role": role,
        "resumable": False,
        "development_candidate_eligible": eligible,
        "deployment_checkpoint_eligible": False,
        "diagnostic_only": diagnostic_only,
        "model_only": diagnostic_only,
        "immutable_unique_candidate_look_artifact": diagnostic_only,
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
        "agent_state": runtime_checkpoint,
    }
    if training_history is not None: payload["training_history"] = list(training_history)
    if validation_history is not None: payload["validation_history"] = list(validation_history)
    return payload


def run_development_calibration(args: argparse.Namespace,
                                *, runtime: Optional[V23DevelopmentRuntime] = None) -> dict:
    contract = build_training_contract(args)
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ConstrainedV23TrainerError("V2.3 requires a fresh output directory")
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
    dual = CompleteBlockProjectedDual(V23DualConfig(**contract["dual"]))
    runtime.set_dual_lambda(dual.lambda_value)
    training_history, validation_history, dual_history, pending = [], [], [], []
    terminal_proposal = best_validation = None
    latest_path = output_dir / "latest.pth"
    best_path = output_dir / "best-development-candidate.pth"
    candidate_look_artifacts = []
    candidate_manifest_path = output_dir / "candidate-look-checkpoint-manifest.json"
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
        print(f"V2.3 Ep {episode_number:4d} | Phase {state.phase:13s} | "
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
                            raise RuntimeError("V2.3 runtime did not echo the requested validation seed")
                        raw = dict(raw)
                        raw["instance_index"] = instance_index
                        raw_rows.append(raw)
            finally:
                validation_batch_audit = runtime.end_validation_batch()
            if validation_batch_audit.get("training_agent_unchanged") is not True:
                raise RuntimeError("V2.3 stochastic validation mutated authenticated training state")
            if any(row.get("training_agent_unchanged") is not True for row in raw_rows):
                raise RuntimeError("V2.3 validation mutated and later restored training state within a batch")
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
                    raise RuntimeError("V2.3 validation instance drifted across action RNG rows")
            ordered_identities = tuple(identities[int(seed)] for seed in args.validation_seeds)
            if ordered_identities != PARENT_V2_2_VALIDATION_INSTANCE_IDENTITIES:
                raise RuntimeError(
                    "V2.3 validation grid does not exactly match the bound V2.2 manifest"
                )
            if validation_identity_manifest is None:
                validation_identity_manifest = {
                    "schema_version": 1,
                    "validation_protocol": VALIDATION_PROTOCOL,
                    "common_grid_every_checkpoint": True,
                    "parent_v2_2_validation_manifest_sha256": PARENT_V2_2_VALIDATION_MANIFEST_SHA256,
                    "exact_parent_v2_2_instance_identity_match": True,
                    "instances": ordered_identities,
                    "prospective_83xxx_panel_opened": False,
                    "final_86xxx_panel_opened": False,
                }
                validation_identity_manifest["manifest_sha256"] = contract_hash(validation_identity_manifest)
                v21t._atomic_json(validation_identity_manifest, output_dir / "validation-instance-manifest.json")
            elif ordered_identities != tuple(validation_identity_manifest["instances"]):
                raise RuntimeError("V2.3 common validation EpisodeInstance grid drifted across looks")
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
            print(f"V2.3 Validation Ep {episode_number:4d} | "
                  f"R {validation_summary['mean_dense_return']:8.2f} | "
                  f"MAE {mae_text} | "
                  f"Strict {int(validation_summary['strict_integrity_gate'])} | "
                  f"Reh/100 {validation_summary['expected_physical_rehandles_per_100_required_deliveries']:7.2f} | "
                  f"UCB {validation_summary['physical_rehandle_rate_one_sided_95_ucb']:7.2f} | "
                  f"Eligible {int(validation_summary['development_candidate_eligible'])}", flush=True)
            if episode_number in CANDIDATE_EPISODES:
                diagnostic_relative = (
                    f"candidate-look-checkpoints/episode-{episode_number:04d}-model-only.pth"
                )
                diagnostic_path = output_dir / diagnostic_relative
                if diagnostic_path.exists():
                    raise ConstrainedV23TrainerError(
                        "candidate-look diagnostic path is not immutable/unique"
                    )
                v21t._atomic_torch(_checkpoint_payload(
                    contract=contract, runtime=runtime, dual=dual,
                    completed_episodes=episode_number,
                    role="candidate_look_diagnostic",
                    schedule_state=checkpoint_schedule,
                    pending_dual_runs=pending,
                    validation_summary=validation_summary,
                    include_replay=False,
                ), diagnostic_path)
                candidate_look_artifacts.append({
                    "checkpoint_episode": episode_number,
                    "relative_path": diagnostic_relative,
                    "sha256": _file_sha256(diagnostic_path),
                    "checkpoint_role": "candidate_look_diagnostic",
                    "model_only": True,
                    "diagnostic_only": True,
                    "development_candidate_eligible": False,
                    "deployment_checkpoint_eligible": False,
                    "validation_development_candidate_eligible": bool(
                        validation_summary["development_candidate_eligible"]
                    ),
                })
                candidate_manifest = {
                    "schema_version": 1,
                    "checkpoint_family": CHECKPOINT_FAMILY,
                    "training_contract_sha256": contract["contract_sha256"],
                    "expected_candidate_look_episodes": CANDIDATE_EPISODES,
                    "saved_candidate_look_episodes": tuple(
                        item["checkpoint_episode"] for item in candidate_look_artifacts
                    ),
                    "artifact_count": len(candidate_look_artifacts),
                    "complete": len(candidate_look_artifacts) == len(CANDIDATE_EPISODES),
                    "model_only": True,
                    "diagnostic_only": True,
                    "final_86xxx_panel_opened": False,
                    "artifacts": tuple(candidate_look_artifacts),
                }
                candidate_manifest["manifest_sha256"] = contract_hash(candidate_manifest)
                v21t._atomic_json(candidate_manifest, candidate_manifest_path)
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

    if pending: raise RuntimeError("V2.3 ended with an incomplete dual block")
    if tuple(
        item["checkpoint_episode"] for item in candidate_look_artifacts
    ) != CANDIDATE_EPISODES:
        raise RuntimeError("V2.3 did not save every candidate-look diagnostic")
    summary = {
        "status": "complete", "method_version": METHOD_VERSION,
        "completed_training_episodes": len(training_history),
        "completed_blocks": TOTAL_BLOCKS, "dual_state": dual.state_dict(),
        "dual_update_history": dual_history, "terminal_dual_proposal": terminal_proposal,
        "development_candidate_eligible": best_validation is not None,
        "deployment_checkpoint_eligible": False, "best_validation": best_validation,
        "validation_history": validation_history,
        "candidate_look_diagnostic_checkpoints": candidate_look_artifacts,
        "candidate_look_checkpoint_manifest": str(candidate_manifest_path),
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
    parser = argparse.ArgumentParser(description="Run frozen constrained VCG V2.3 calibration")
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
    parser.add_argument("--gamma-operational", type=float, default=1.0)
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
        raise ConstrainedV23TrainerError("V2.3 clock is frozen at 200 episodes / validation every 20")
    if args.model_seed != DEFAULT_MODEL_SEED:
        raise ConstrainedV23TrainerError("V2.3 requires model seed 10")
    if args.train_seed_base != FRESH_TRAIN_SEED_BASE:
        raise ConstrainedV23TrainerError("V2.3 requires train seed base 61000000")
    if (args.grid_rows, args.grid_cols, args.number_blocks, args.max_steps) != (5, 5, 8, 2_000):
        raise ConstrainedV23TrainerError("V2.3 requires the frozen 5x5/8-block/2000-step workload")
    if _finite(args.gamma_operational, name="gamma_operational") != 1.0:
        raise ConstrainedV23TrainerError("V2.3 requires gamma_operational=1.0")
    if _finite(args.reward_scale, name="reward_scale") != 0.01:
        raise ConstrainedV23TrainerError("V2.3 requires reward_scale=0.01")
    if _finite(args.arrival_rate, name="arrival_rate") <= 0:
        raise ConstrainedV23TrainerError("arrival_rate must be positive")
    if (float(args.arrival_rate), int(args.proc_mean), int(args.max_hold_steps), int(args.max_idle_steps)) != (10.0, 80, 10, 20):
        raise ConstrainedV23TrainerError("V2.3 arrival/proc/Hold constants are frozen")
    V23DualConfig(args.rehandle_budget_per_100, args.dual_lr,
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
    "CompleteBlockProjectedDual", "ConstrainedV23TrainerError", "V23DualConfig",
    "V23ScheduleState", "build_parser", "build_training_contract",
    "episode_budget_residual", "frozen_schedule_table", "load_default_runtime",
    "load_best_development_candidate", "load_candidate_look_diagnostic",
    "run_development_calibration",
    "schedule_for_episode", "select_better_validation",
    "summarize_validation", "temperatures_for_block", "validation_policy_rng_seed",
    "validation_score",
]
