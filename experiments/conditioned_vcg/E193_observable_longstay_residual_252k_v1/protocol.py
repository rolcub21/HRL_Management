"""Authenticated protocol for E193's second long-stay learning model."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from experiments.conditioned_vcg.E191_fresh_long_stay_crowding_250k_v1 import (
    protocol as e191,
    scenario as workload,
)

from . import defer, representation, residual


ROOT = e191.ROOT
PROTOCOL = "vcg_conditioned_e193_observable_longstay_residual_252k_v1"
METHOD_VERSION = "vcg_dense_v1_1_e193_observable_event_residual_longstay"
SCENARIO = workload.SCENARIO
SCENARIOS = {SCENARIO: workload.SCENARIO_SPEC}
MODEL_SEEDS = e191.MODEL_SEEDS
TOTAL_EPISODES = e191.TOTAL_EPISODES
EPISODES_BY_SCENARIO = {SCENARIO: TOTAL_EPISODES}
VALIDATION_EVERY = e191.VALIDATION_EVERY
CHECKPOINT_EVERY = e191.CHECKPOINT_EVERY
VALIDATION_SEEDS = e191.VALIDATION_SEEDS
TRAIN_BASE = e191.TRAIN_BASE
TRAINING_JOB_BUDGET = e191.TRAINING_JOB_BUDGET
MAX_STEPS = e191.MAX_STEPS
GAMMA = e191.GAMMA
EPSILON_START = e191.EPSILON_START
EPSILON_END = e191.EPSILON_END
EPSILON_WARMUP_DECISIONS = e191.EPSILON_WARMUP_DECISIONS
EPSILON_DECAY_DECISIONS = e191.EPSILON_DECAY_DECISIONS
REPORT_CHECKPOINTS = e191.REPORT_CHECKPOINTS
LATE_CHECKPOINTS = e191.LATE_CHECKPOINTS
TD_BACKUP_IMPLEMENTATION = (
    "e193_observable_nominal_residual_identity_dedup_"
    "frozen_target_cache_v1"
)
SPEC = e191.SPEC
TargetDenseEnv = workload.LongStayCrowdingEnv
DEFAULT_OUTPUT = (
    ROOT / "results/vcg-conditioned-e193-observable-longstay-residual-252k-v1"
)


def scenario_for(name: str):
    try:
        return SCENARIOS[name]
    except KeyError as error:
        raise ValueError(f"unknown E193 scenario: {name}") from error


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


def contract() -> dict:
    parent = e191.contract()
    directory = Path(__file__).resolve().parent
    sources = (
        directory / "protocol.py",
        directory / "representation.py",
        directory / "residual.py",
        directory / "defer.py",
        directory / "instrumentation.py",
        directory / "stage.py",
        directory / "run.py",
        directory / "run.sh",
        directory / "README.md",
        ROOT / "tests/test_vcg_e193_observable_longstay_residual.py",
        ROOT / "experiments/conditioned_vcg/E180_wait_cost_unmasked_pilot_239k_v1/objective.py",
        ROOT / "experiments/conditioned_vcg/E187_wait_corrected_e78_246k_v1/objective.py",
        ROOT / "experiments/conditioned_vcg/E188_mask_only_progress_guard_247k_v1/guard.py",
        ROOT / "experiments/conditioned_vcg/E191_fresh_long_stay_crowding_250k_v1/scenario.py",
        ROOT / "experiments/conditioned_vcg/target_size_dense_stage_a_v4/acceleration.py",
    )
    payload = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "method_version": METHOD_VERSION,
        "scope": (
            "fresh second-model training on the exact E191 workload with "
            "observable cadence, long-stay scaling, event Defer, and nominal "
            "immediate-cost residual learning; development only"
        ),
        "parent_protocol": e191.PROTOCOL,
        "parent_contract_sha256": parent["contract_sha256"],
        "controlled_constants": {
            "objective": deepcopy(parent["objective"]),
            "audit_scaffold_objective": deepcopy(
                parent["audit_scaffold_objective"]
            ),
            "gamma": parent["gamma"],
            "scenario_public": deepcopy(parent["scenario_public"]),
            "workload": deepcopy(parent["workload"]),
            "frontier_physical_certification": (
                parent["frontier"]["certification"]
            ),
            "progress_guard": parent["frontier"]["progress_guard"],
            "model_seeds": list(MODEL_SEEDS),
            "episodes_per_seed": TOTAL_EPISODES,
            "validation_every": VALIDATION_EVERY,
            "validation_seeds": list(VALIDATION_SEEDS[SCENARIO]),
            "checkpoint_every": CHECKPOINT_EVERY,
            "epsilon": deepcopy(parent["epsilon"]),
            "finite_horizon": deepcopy(parent["finite_horizon"]),
        },
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
        "finite_horizon": deepcopy(parent["finite_horizon"]),
        "epsilon": deepcopy(parent["epsilon"]),
        "objective": deepcopy(parent["objective"]),
        "audit_scaffold_objective": deepcopy(
            parent["audit_scaffold_objective"]
        ),
        "gamma": GAMMA,
        "workload": deepcopy(parent["workload"]),
        "initialization": deepcopy(parent["initialization"]),
        "warm_start": deepcopy(parent["warm_start"]),
        "representation": {
            "controller_architecture": residual.CONTROLLER_ARCHITECTURE,
            "network_architecture": residual.NETWORK_ARCHITECTURE,
            "base_representation": representation.NETWORK_ARCHITECTURE,
            "queue_context_names": list(representation.QUEUE_CONTEXT_NAMES),
            "queue_context_dim": representation.QUEUE_CONTEXT_DIM,
            "observed_arrival_history": True,
            "future_schedule_visible_to_policy": False,
            "arrival_window_steps": representation.ARRIVAL_WINDOW_STEPS,
            "queue_age_scale_steps": representation.DURATION_SCALE_STEPS,
            "arrival_age_scale_steps": (
                representation.SLOWEST_ARRIVAL_GAP_STEPS
            ),
            "graph_timing_scale": representation.TIMING_SCALE,
        },
        "defer_contract": {
            "version": defer.DEFER_VERSION,
            "option_version": defer.OPTION_VERSION,
            "events": ["observed_release", "stored_delivery_window_open"],
            "storage_clock_window": 20,
            "inbound_defer_allowed": True,
            "numeric_consecutive_defer_cap": None,
            "future_release_timestamp_read": False,
            "episode_horizon_role": (
                "nonbinding option error boundary while an unseen release exists"
            ),
        },
        "value_learning": {
            "contract": residual.RESIDUAL_VALUE_CONTRACT,
            "network_output": "scalar residual reward-to-go",
            "selection_and_target_value": "known_scaled_nominal_return + residual",
            "known_components": list(
                parent["objective"]["component_names"]
            ),
            "known_information": (
                "currently observable stored/queued jobs and nominal bound path"
            ),
            "hidden_arrivals_during_macro": "learned residual",
            "no_double_count_algebra": (
                "prediction=K+f; target=scaled_observed_macro_return+next_total"
            ),
        },
        "frontier": {
            **deepcopy(parent["frontier"]),
            "defer_semantics": defer.DEFER_VERSION,
            "physical_certification_unchanged": True,
        },
        "acceleration": deepcopy(parent["acceleration"]),
        "acceleration_assertion": {
            **deepcopy(parent["acceleration_assertion"]),
            "cost_aware_deduplicated_scoring": True,
            "cost_aware_frozen_target_cache": True,
        },
        "checkpoint_selection": parent["checkpoint_selection"],
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
