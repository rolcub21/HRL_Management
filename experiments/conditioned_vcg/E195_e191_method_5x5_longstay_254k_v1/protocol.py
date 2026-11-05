"""Frozen protocol for E191-method training on scale-matched 5x5."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from experiments.conditioned_vcg.E191_fresh_long_stay_crowding_250k_v1 import (
    protocol as e191,
)

from . import representation
from . import scenario as workload


ROOT = e191.ROOT
PROTOCOL = "vcg_conditioned_e195_e191_method_5x5_longstay_254k_v1"
METHOD_VERSION = "vcg_dense_v1_1_e195_e191_method_scale_matched_5x5"
SCENARIO = workload.SCENARIO
SCENARIOS = {SCENARIO: workload.SCENARIO_SPEC}
MODEL_SEEDS = (0, 1, 2)
TOTAL_EPISODES = e191.TOTAL_EPISODES
EPISODES_BY_SCENARIO = {SCENARIO: TOTAL_EPISODES}
VALIDATION_EVERY = e191.VALIDATION_EVERY
CHECKPOINT_EVERY = e191.CHECKPOINT_EVERY
VALIDATION_SEEDS = {SCENARIO: workload.VALIDATION_SEEDS}
TRAIN_BASE = {SCENARIO: workload.TRAIN_BASE}
TRAINING_JOB_BUDGET = TOTAL_EPISODES * workload.TOTAL_JOBS
MAX_STEPS = workload.MAX_STEPS
GAMMA = e191.GAMMA
EPSILON_START = e191.EPSILON_START
EPSILON_END = e191.EPSILON_END
EPSILON_WARMUP_DECISIONS = e191.EPSILON_WARMUP_DECISIONS
# Preserve E191's exploration exposure per training job.  A literal 18,500
# decision decay would not finish on the much shorter eight-job episodes.
EPSILON_DECAY_DECISIONS = round(
    e191.EPSILON_DECAY_DECISIONS * workload.TOTAL_JOBS
    / e191.workload.TOTAL_JOBS
)
REPORT_CHECKPOINTS = tuple(
    range(VALIDATION_EVERY, TOTAL_EPISODES + 1, VALIDATION_EVERY)
)
LATE_CHECKPOINTS = REPORT_CHECKPOINTS
TD_BACKUP_IMPLEMENTATION = e191.TD_BACKUP_IMPLEMENTATION
SPEC = e191.SPEC
TargetDenseEnv = workload.LongStay5x5Env
DEFAULT_OUTPUT = (
    ROOT / "results/vcg-conditioned-e195-e191-method-5x5-longstay-254k-v1"
)


def scenario_for(name: str):
    try:
        return SCENARIOS[name]
    except KeyError as error:
        raise ValueError(f"unknown E195 scenario: {name}") from error


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def digest(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


UNCHANGED_METHOD_FIELDS = e191.UNCHANGED_METHOD_FIELDS


def contract() -> dict:
    parent = e191.contract()
    directory = Path(__file__).resolve().parent
    sources = (
        directory / "protocol.py",
        directory / "scenario.py",
        directory / "representation.py",
        directory / "instrumentation.py",
        directory / "stage.py",
        directory / "run.py",
        directory / "run.sh",
        directory / "README.md",
        ROOT / "experiments/conditioned_vcg/E191_fresh_long_stay_crowding_250k_v1/representation.py",
        ROOT / "experiments/conditioned_vcg/E188_mask_only_progress_guard_247k_v1/guard.py",
        ROOT / "experiments/conditioned_vcg/E187_wait_corrected_e78_246k_v1/objective.py",
        ROOT / "experiments/conditioned_vcg/E100_accelerated_release_clock_163k_v1/fast_inputs.py",
        ROOT / "experiments/conditioned_vcg/target_size_dense_stage_a_v4/acceleration.py",
    )
    unchanged = {
        name: deepcopy(parent[name]) for name in UNCHANGED_METHOD_FIELDS
    }
    payload = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "method_version": METHOD_VERSION,
        "scope": (
            "fresh E191-method training on a scale-matched long-stay 5x5 "
            "workload; development only"
        ),
        "parent_method_protocol": e191.PROTOCOL,
        "parent_method_contract_sha256": parent["contract_sha256"],
        "unchanged_method_fields": list(UNCHANGED_METHOD_FIELDS),
        "unchanged_method_fields_sha256": digest(unchanged),
        **unchanged,
        "scenario": SCENARIO,
        "scenario_public": workload.SCENARIO_SPEC.public_dict(),
        "model_seeds": list(MODEL_SEEDS),
        "episodes_per_seed": TOTAL_EPISODES,
        "training_jobs_per_seed": TRAINING_JOB_BUDGET,
        "train_instance_seed_base": TRAIN_BASE[SCENARIO],
        "validation_every": VALIDATION_EVERY,
        "validation_seeds": list(VALIDATION_SEEDS[SCENARIO]),
        "validation_cases_per_level": workload.VALIDATION_CASES_PER_LEVEL,
        "checkpoint_every": CHECKPOINT_EVERY,
        "report_checkpoints": list(REPORT_CHECKPOINTS),
        "finite_horizon": {
            "steps": MAX_STEPS,
            "role": "error cap only; never an action rule",
            "derivation": (
                "gap100 last release 700 + two mean stays 320 + "
                "100 primitive steps per job 800 = 1820; rounded to 2000"
            ),
        },
        "epsilon": {
            "start": EPSILON_START,
            "end": EPSILON_END,
            "warmup_decisions": EPSILON_WARMUP_DECISIONS,
            "decay_decisions": EPSILON_DECAY_DECISIONS,
            "rule": (
                "E191 start/end and linear decision schedule; decay decisions "
                "scaled by 8/36 to preserve exploration exposure per training job"
            ),
            "e191_decay_decisions": e191.EPSILON_DECAY_DECISIONS,
            "job_budget_scale": (
                workload.TOTAL_JOBS / e191.workload.TOTAL_JOBS
            ),
        },
        "workload": {
            "empty_start": True,
            "jobs_per_episode": workload.TOTAL_JOBS,
            "capacity": workload.CAPACITY,
            "mean_storage_duration": workload.MEAN_DURATION,
            "duration_distribution": "max(1, Poisson(160))",
            "target_nominal_expected_occupancy": list(workload.LEVELS),
            "arrival_gap_steps": {
                f"{100 * level:.0f}%": workload.gap_for_level(level)
                for level in workload.LEVELS
            },
            "arrival_steps": "gap*j for j=0,...,7",
            "training_level_rule": (
                "deterministic balanced seed stratification over 20%,50%,80%"
            ),
            "occupancy_definition": (
                "arrival_rate * mean_storage_duration / 8; not initial fill"
            ),
            "occupancy_is_initial_yard_fill": False,
            "finite_episode_caveat": (
                "empty-start ramp and terminal drain lower whole-episode mean; "
                "report achieved mean, arrival-window mean, steady-window mean, and peak"
            ),
            "future_release_schedule_visible_to_policy": False,
            "unreleased_duration_visible_to_policy": False,
        },
        "scale_match_to_e191": {
            "e191_capacity": 35,
            "e195_capacity": workload.CAPACITY,
            "e191_mean_duration_per_cell": 20.0,
            "e195_mean_duration_per_cell": (
                workload.MEAN_DURATION / workload.CAPACITY
            ),
            "same_arrival_gaps": True,
            "same_nominal_occupancy_levels": True,
            "episode_count_matched": True,
            "job_or_decision_budget_matched": False,
            "jobs_per_episode_difference_reported": True,
            "epsilon_decay_job_budget_normalized": True,
        },
        "architecture": {
            "controller_architecture": representation.CONTROLLER_ARCHITECTURE,
            "network_architecture": representation.NETWORK_ARCHITECTURE,
            "queue_context_names": list(representation.QUEUE_CONTEXT_NAMES),
            "queue_context_dim": representation.QUEUE_CONTEXT_DIM,
            "exactly_identical_to_e191": True,
            "arrival_history_features_added": False,
        },
        "initialization": {
            "kind": "fresh_random_from_model_seed",
            "transferred_modules": [],
            "source_checkpoint": None,
            "optimizer": "fresh",
            "replay": "empty",
            "counters": "zero",
            "guard_memory": "empty",
            "certificate_cache": "per-episode empty",
        },
        "warm_start": {
            "enabled": False,
            "kind": "fresh_random_from_model_seed",
            "transferred_modules": [],
            "source_checkpoint": None,
            "optimizer": "fresh",
            "replay": "empty",
            "counters": "zero",
            "guard_memory": "empty",
        },
        "defer_contract": {
            "identical_to_e191": True,
            "max_option_steps": 10,
            "max_consecutive_defers": 16,
            "known_caveat": parent["defer_contract"]["known_caveat"],
        },
        "acceleration_assertion": {
            "identical_to_e191": True,
            "relocation_family_certification": True,
            "timing_invariant_certificate_cache": True,
            "path_cleanup": True,
            "device_graph_input_cache": True,
            "identity_deduplicated_scoring": True,
            "cached_batched_td_backup": True,
        },
        "checkpoint_selection": (
            "development panel average across all three workload levels; "
            "retain every checkpoint; sealed test panel remains closed"
        ),
        "historical_5x5_relation": {
            "old_checkpoint_used_for_initialization": False,
            "old_dense_objective_used_for_learning": False,
            "old_gamma_0p99_used": False,
            "separate_frozen_counterfactual_evaluation_required": True,
        },
        "test_panel_opened": False,
        "paper_evidence": False,
        "source_sha256": {
            str(path.relative_to(ROOT)): sha256(path) for path in sources
        },
    }
    return {**payload, "contract_sha256": digest(payload)}


def trainer_protocol():
    value = contract()
    return SimpleNamespace(
        PROTOCOL=PROTOCOL,
        METHOD_VERSION=METHOD_VERSION,
        SCENARIO=SCENARIO,
        SCENARIOS=SCENARIOS,
        MODEL_SEEDS=MODEL_SEEDS,
        EPISODES_BY_SCENARIO=EPISODES_BY_SCENARIO,
        TOTAL_EPISODES=TOTAL_EPISODES,
        VALIDATION_EVERY=VALIDATION_EVERY,
        CHECKPOINT_EVERY=CHECKPOINT_EVERY,
        VALIDATION_SEEDS=VALIDATION_SEEDS,
        TRAIN_BASE=TRAIN_BASE,
        TRAINING_JOB_BUDGET=TRAINING_JOB_BUDGET,
        TD_BACKUP_IMPLEMENTATION=TD_BACKUP_IMPLEMENTATION,
        LATE_CHECKPOINTS=LATE_CHECKPOINTS,
        DEFAULT_OUTPUT=DEFAULT_OUTPUT,
        TargetDenseEnv=TargetDenseEnv,
        scenario_for=scenario_for,
        contract=lambda: value,
    )


__all__ = [
    "CHECKPOINT_EVERY", "DEFAULT_OUTPUT", "EPISODES_BY_SCENARIO",
    "EPSILON_DECAY_DECISIONS", "EPSILON_END", "EPSILON_START",
    "EPSILON_WARMUP_DECISIONS", "GAMMA", "LATE_CHECKPOINTS", "MAX_STEPS",
    "METHOD_VERSION", "MODEL_SEEDS", "PROTOCOL", "REPORT_CHECKPOINTS",
    "SCENARIO", "SCENARIOS", "SPEC", "TD_BACKUP_IMPLEMENTATION",
    "TOTAL_EPISODES", "TRAIN_BASE", "TRAINING_JOB_BUDGET",
    "TargetDenseEnv", "VALIDATION_EVERY", "VALIDATION_SEEDS", "contract",
    "digest", "scenario_for", "sha256", "trainer_protocol",
]
