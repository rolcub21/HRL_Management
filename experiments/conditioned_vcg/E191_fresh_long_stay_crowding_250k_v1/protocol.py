"""Frozen protocol for fresh E189-architecture long-stay training."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from experiments.conditioned_vcg.E189_e188_roundoff_repair_248k_v1 import (
    protocol as e189,
)

from . import representation
from . import scenario as workload


ROOT = e189.ROOT
PROTOCOL = "vcg_conditioned_e191_fresh_long_stay_crowding_250k_v1"
METHOD_VERSION = "vcg_dense_v1_1_e191_e189_arch_fresh_longstay"
SCENARIO = workload.SCENARIO
SCENARIOS = {SCENARIO: workload.SCENARIO_SPEC}
MODEL_SEEDS = (0, 1, 2)
TOTAL_EPISODES = 300
EPISODES_BY_SCENARIO = {SCENARIO: TOTAL_EPISODES}
VALIDATION_EVERY = 20
CHECKPOINT_EVERY = 1
VALIDATION_SEEDS = {SCENARIO: workload.VALIDATION_SEEDS}
TRAIN_BASE = {SCENARIO: workload.TRAIN_BASE}
TRAINING_JOB_BUDGET = TOTAL_EPISODES * workload.TOTAL_JOBS
MAX_STEPS = workload.MAX_STEPS
GAMMA = e189.GAMMA
EPSILON_START = e189.EPSILON_START
EPSILON_END = e189.EPSILON_END
EPSILON_WARMUP_DECISIONS = e189.EPSILON_WARMUP_DECISIONS
EPSILON_DECAY_DECISIONS = e189.EPSILON_DECAY_DECISIONS
REPORT_CHECKPOINTS = tuple(
    range(VALIDATION_EVERY, TOTAL_EPISODES + 1, VALIDATION_EVERY)
)
LATE_CHECKPOINTS = REPORT_CHECKPOINTS
TD_BACKUP_IMPLEMENTATION = e189.TD_BACKUP_IMPLEMENTATION
SPEC = e189.SPEC
TargetDenseEnv = workload.LongStayCrowdingEnv
DEFAULT_OUTPUT = ROOT / "results/vcg-conditioned-e191-fresh-long-stay-crowding-250k-v1"


def scenario_for(name: str):
    try:
        return SCENARIOS[name]
    except KeyError as error:
        raise ValueError(f"unknown E191 scenario: {name}") from error


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def digest(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


UNCHANGED_METHOD_FIELDS = (
    "objective", "audit_scaffold_objective", "gamma", "frontier",
    "acceleration",
)


def contract() -> dict:
    parent = e189.contract()
    directory = Path(__file__).resolve().parent
    sources = (
        directory / "protocol.py",
        directory / "scenario.py",
        directory / "representation.py",
        directory / "instrumentation.py",
        directory / "roundoff.py",
        directory / "stage.py",
        directory / "run.py",
        directory / "run.sh",
        directory / "README.md",
        ROOT / "experiments/conditioned_vcg/E189_e188_roundoff_repair_248k_v1/roundoff.py",
        ROOT / "experiments/conditioned_vcg/E188_mask_only_progress_guard_247k_v1/guard.py",
        ROOT / "experiments/conditioned_vcg/E188_mask_only_progress_guard_247k_v1/representation.py",
        ROOT / "experiments/conditioned_vcg/E187_wait_corrected_e78_246k_v1/objective.py",
        ROOT / "experiments/conditioned_vcg/E187_wait_corrected_e78_246k_v1/representation.py",
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
            "fresh training with exact E189 architecture/method on a "
            "serviceable long-stay workload family; development only"
        ),
        "parent_method_protocol": e189.PROTOCOL,
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
                "gap100 last release 3500 + two mean stays 1400 + "
                "100 primitive steps per job 3600 = 8500; rounded to 9000"
            ),
        },
        "epsilon": {
            "start": EPSILON_START,
            "end": EPSILON_END,
            "warmup_decisions": EPSILON_WARMUP_DECISIONS,
            "decay_decisions": EPSILON_DECAY_DECISIONS,
            "rule": "unchanged E189 decision-budget schedule",
        },
        "workload": {
            "empty_start": True,
            "jobs_per_episode": workload.TOTAL_JOBS,
            "capacity": workload.CAPACITY,
            "mean_storage_duration": workload.MEAN_DURATION,
            "duration_distribution": "max(1, Poisson(700))",
            "target_nominal_expected_occupancy": list(workload.LEVELS),
            "arrival_gap_steps": {
                f"{100 * level:.0f}%": workload.gap_for_level(level)
                for level in workload.LEVELS
            },
            "arrival_steps": "gap*j for j=0,...,35",
            "training_level_rule": (
                "deterministic balanced seed stratification over 20%,50%,80%"
            ),
            "occupancy_definition": (
                "arrival_rate * mean_storage_duration / 35; not initial fill"
            ),
            "occupancy_is_initial_yard_fill": False,
            "finite_episode_caveat": (
                "empty-start ramp and terminal drain lower whole-episode mean; "
                "report achieved mean, arrival-window mean, steady-window mean, and peak"
            ),
            "future_release_schedule_visible_to_policy": False,
            "unreleased_duration_visible_to_policy": False,
        },
        "architecture": {
            "controller_architecture": representation.CONTROLLER_ARCHITECTURE,
            "network_architecture": representation.NETWORK_ARCHITECTURE,
            "queue_context_names": list(representation.QUEUE_CONTEXT_NAMES),
            "queue_context_dim": representation.QUEUE_CONTEXT_DIM,
            "exactly_identical_to_e189": True,
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
        # Compatibility name consumed by the inherited E187 trainer.  It is
        # intentionally the same fresh-initialization declaration, not a
        # checkpoint source.
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
        # The old Defer contract is kept deliberately so this run changes the
        # scenario and initialization, not action semantics.  Telemetry makes
        # any binding of this limit visible rather than silently repairing it.
        "defer_contract": {
            "unchanged_from_e189": True,
            "max_option_steps": 10,
            "max_consecutive_defers": 16,
            "known_caveat": (
                "a >160-step interval without an observed arrival/window event "
                "can remove Defer; inspect guard and early-delivery telemetry"
            ),
        },
        "acceleration_assertion": {
            "unchanged_from_e189": True,
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
