"""Locked protocol for the final Stage-A method on scale-matched 10x10."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from experiments.conditioned_vcg.E191_fresh_long_stay_crowding_250k_v1 import (
    protocol as e191,
)
from experiments.conditioned_vcg.E209_e191_method_10x10_longstay_268k_v1 import (
    representation,
    scenario as workload,
)
from projects.temporal_consequence_realization.method_conformance import (
    CHECKLIST,
    validate_protocol,
)


ROOT = e191.ROOT
PROTOCOL = "vcg_conditioned_e244_e215_scale_matched_10x10_stage_a_303k_v1"
METHOD_VERSION = "vcg_dense_v1_1_e244_e215_scale_matched_10x10_stage_a"
METHOD_REFERENCE = "projects/temporal_consequence_realization/METHOD_REFERENCE.md"
METHOD_REVISION = "TCR-METHOD-2026-10-31.10"
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
GAMMA = 1.0
EPSILON_START = 0.90
EPSILON_END = 0.05
EPSILON_WARMUP_DECISIONS = 0
E215_DECAY_DECISIONS = 57_300
E215_JOBS_PER_EPISODE = 36
EPSILON_DECAY_DECISIONS = round(
    E215_DECAY_DECISIONS * workload.TOTAL_JOBS / E215_JOBS_PER_EPISODE
)
EPSILON_TARGET_FLOOR_EPISODE = 220
REPORT_CHECKPOINTS = tuple(
    range(VALIDATION_EVERY, TOTAL_EPISODES + 1, VALIDATION_EVERY)
)
LATE_CHECKPOINTS = REPORT_CHECKPOINTS
TD_BACKUP_IMPLEMENTATION = e191.TD_BACKUP_IMPLEMENTATION
SPEC = e191.SPEC
TargetDenseEnv = workload.LongStay10x10Env
DEFAULT_OUTPUT = (
    ROOT
    / "results/vcg-conditioned-e244-e215-scale-matched-10x10-stage-a-303k-v1"
)


OBJECTIVE = {
    "component_names": [
        "storage_tardiness_accrual", "storage_earliness_at_delivery",
        "admission_wait_accrual", "equipment_work", "physical_rehandles",
    ],
    "e78_dense_reward_used_for_learning": False,
    "formula": "2*T_storage + E_storage + 0.5*W_release_to_admission + 0.1*L + 10*R",
    "gamma": 1.0,
    "name": "J_storage_clock_plus_full_admission_wait",
    "primitive_accrual": True,
    "storage_due_rule": "stored_time_step + storage_steps_needed",
    "storage_tolerance_steps": 20,
    "wait_grace_steps": 0,
    "wait_rule": "stored_time_step - arrival_step",
    "weights": [2.0, 1.0, 0.5, 0.1, 10.0],
}
AUDIT_SCAFFOLD_OBJECTIVE = {
    "contract": "dual_rescorable_timing_objective_components_v1",
    "dense_b": 80.0,
    "dense_matches_legacy_inside_window": False,
    "exact_legacy": True,
    "lambda_abs": 1.5,
    "lambda_outside": 0.5,
    "legacy_base": 10.0,
    "legacy_max_bonus": 30.0,
    "legacy_window": 20.0,
    "objective": "dense_piecewise_v1",
    "schema_version": 1,
    "window": 20.0,
}
FRONTIER = {
    "accept_first_mask": False,
    "certification": "unchanged E78 exact safe frontier",
    "critic_retains_progress_choice": True,
    "full_witness_commitment": False,
    "inbound_defer_mask": False,
    "intake_restoration_preference": False,
    "one_witness_step_last_resort": True,
    "physical_actions": "unchanged E78",
    "progress_guard": (
        "at the E187 nonprogress threshold remove Reconfigure only; the critic "
        "chooses among certified Accept, Deliver, and Defer; use one retained "
        "witness step only when none remains"
    ),
    "reconfigure_mask_at_threshold": True,
}
ACCELERATION = {
    "certificate": "D12 relocation family plus E14 timing-invariant cache/path cleanup",
    "decision_semantics_changed": False,
    "graph_inputs": "E100 exact device tensor cache",
    "queue_scoring": "E107 identity-deduplicated batched scoring",
    "td_backup": "e187_queue_context_exact_device_graph_inputs_frozen_target_cache_v1",
}
UNCHANGED_METHOD_FIELDS = (
    "objective", "audit_scaffold_objective", "gamma", "frontier", "acceleration",
)


def scenario_for(name: str):
    try:
        return SCENARIOS[name]
    except KeyError as error:
        raise ValueError(f"unknown E244 scenario: {name}") from error


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


def method_declaration() -> dict:
    audit = {}
    for identifier, _description in CHECKLIST:
        if identifier == "reproducibility.method_revision":
            status = "pass"
            evidence = f"The protocol pins {METHOD_REVISION}."
        elif identifier == "reproducibility.content_hashes":
            status = "pass"
            evidence = "The contract hashes the runner and every inherited behavior source."
        elif identifier == "reproducibility.oracle_declarations":
            status = "pass"
            evidence = "No oracle input is used; the historical VCG scope is explicit."
        elif identifier == "reproducibility.failed_gates_retained":
            status = "pass"
            evidence = "Every checkpoint and fixed development validation is retained."
        elif identifier == "reproducibility.preprotected_integration_gate":
            status = "not_applicable"
            evidence = (
                "E244 trains only on declared development instances and does not "
                "open a fresh or protected evaluation split."
            )
        else:
            status = "not_applicable"
            evidence = (
                "E244 is a historical conditioned-VCG scale experiment and makes "
                "no canonical law-condition representation claim."
            )
        audit[identifier] = {"status": status, "evidence": evidence}
    declaration = {
        "method_reference": METHOD_REFERENCE,
        "method_revision": METHOD_REVISION,
        "claim_level": "expressibility",
        "declared_deviations": [
            "E244 evaluates the paper's conditioned-VCG controller rather than "
            "the canonical law-condition consequence-realization representation."
        ],
        "oracle_inputs": [],
        "nonconforming_shortcuts": [],
        "conformance_audit": audit,
    }
    validate_protocol(declaration, source=PROTOCOL)
    return declaration


def contract() -> dict:
    directory = Path(__file__).resolve().parent
    inherited = ROOT / "experiments/conditioned_vcg"
    sources = (
        directory / "protocol.py",
        directory / "run.py",
        directory / "run.sh",
        directory / "README.md",
        inherited / "E209_e191_method_10x10_longstay_268k_v1/scenario.py",
        inherited / "E209_e191_method_10x10_longstay_268k_v1/representation.py",
        inherited / "E209_e191_method_10x10_longstay_268k_v1/instrumentation.py",
        inherited / "E209_e191_method_10x10_longstay_268k_v1/stage.py",
        inherited / "E209_e191_method_10x10_longstay_268k_v1/run.py",
        inherited / "E188_mask_only_progress_guard_247k_v1/guard.py",
        inherited / "E187_wait_corrected_e78_246k_v1/objective.py",
        inherited / "E187_wait_corrected_e78_246k_v1/run.py",
        inherited / "E100_accelerated_release_clock_163k_v1/fast_inputs.py",
        inherited / "target_size_dense_stage_a_v4/acceleration.py",
        ROOT / "train_vcg_dense_proper.py",
    )
    unchanged = {
        "objective": deepcopy(OBJECTIVE),
        "audit_scaffold_objective": deepcopy(AUDIT_SCAFFOLD_OBJECTIVE),
        "gamma": GAMMA,
        "frontier": deepcopy(FRONTIER),
        "acceleration": deepcopy(ACCELERATION),
    }
    payload = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        **method_declaration(),
        "method_version": METHOD_VERSION,
        "scope": (
            "fresh final E215/E218 Stage-A training on the scale-matched "
            "long-stay 10x10 workload; development only"
        ),
        "final_stage_a_reference": {
            "seed0_protocol": "vcg_conditioned_e215_e191_long_exploration_drift_pilot_274k_v1",
            "seed12_protocol": "vcg_conditioned_e218_e215_long_exploration_replication_277k_v1",
            "reference_contract_sha256": "02bf7aedca3d1df430a0af26548be986f26215f5527b13f39d7ba678642cfb42",
        },
        "physical_scale_reference": {
            "protocol": "vcg_conditioned_e209_e191_method_10x10_longstay_268k_v1",
            "reference_contract_sha256": "e6db5f570945e6a7572c0c30dbb413c9a9624bbc385bf68beccc26d8c956534d",
        },
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
                "gap100 last release 6300 + two mean stays 2520 + "
                "100 primitive steps per job 6400 = 15220; rounded to 16000"
            ),
        },
        "epsilon": {
            "start": EPSILON_START,
            "end": EPSILON_END,
            "warmup_decisions": EPSILON_WARMUP_DECISIONS,
            "decay_decisions": EPSILON_DECAY_DECISIONS,
            "target_floor_episode_approximate": EPSILON_TARGET_FLOOR_EPISODE,
            "rule": (
                "E215/E218 linear decision schedule, normalized from 36 to 64 "
                "jobs: round(57300*64/36)=101867"
            ),
            "e215_decay_decisions": E215_DECAY_DECISIONS,
            "job_budget_scale": workload.TOTAL_JOBS / E215_JOBS_PER_EPISODE,
        },
        "workload": {
            "empty_start": True,
            "jobs_per_episode": workload.TOTAL_JOBS,
            "capacity": workload.CAPACITY,
            "mean_storage_duration": workload.MEAN_DURATION,
            "duration_distribution": "max(1, Poisson(1260))",
            "target_nominal_expected_occupancy": list(workload.LEVELS),
            "arrival_gap_steps": {
                f"{100 * level:.0f}%": workload.gap_for_level(level)
                for level in workload.LEVELS
            },
            "arrival_steps": "gap*j for j=0,...,63",
            "training_level_rule": "balanced deterministic 20%, 50%, 80% stratification",
            "occupancy_definition": "arrival_rate * mean_storage_duration / 63",
            "occupancy_is_initial_yard_fill": False,
            "future_release_schedule_visible_to_policy": False,
            "unreleased_duration_visible_to_policy": False,
        },
        "scale_match": {
            "reference_capacity": 35,
            "capacity": workload.CAPACITY,
            "mean_duration_per_cell": workload.MEAN_DURATION / workload.CAPACITY,
            "same_arrival_gaps": True,
            "same_nominal_occupancy_levels": True,
            "same_episodes": True,
            "epsilon_decision_budget_normalized_by_jobs": True,
        },
        "architecture": {
            "controller_architecture": representation.CONTROLLER_ARCHITECTURE,
            "network_architecture": representation.NETWORK_ARCHITECTURE,
            "queue_context_names": list(representation.QUEUE_CONTEXT_NAMES),
            "queue_context_dim": representation.QUEUE_CONTEXT_DIM,
            "identical_to_e215_e218": True,
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
            "identical_to_e215_e218": True,
            "max_option_steps": 10,
            "max_consecutive_defers": 16,
            "known_caveat": (
                "a >160-step interval without an observed arrival/window event "
                "can remove Defer; inspect guard and early-delivery telemetry"
            ),
        },
        "guard": {
            "identical_to_e215_e218": True,
            "kind": "mask-only progress guard with one witness step only as last resort",
        },
        "acceleration_assertion": {
            "identical_to_e215_e218": True,
            "decision_semantics_changed": False,
            "relocation_family_certification": True,
            "timing_invariant_certificate_cache": True,
            "path_cleanup": True,
            "device_graph_input_cache": True,
            "identity_deduplicated_scoring": True,
            "cached_batched_td_backup": True,
        },
        "launch_stages": {
            "smoke_through_episode": 3,
            "first_validation_episode": 20,
            "full_training_episode": 300,
        },
        "checkpoint_selection": (
            "development-panel mean across 20%, 50%, and 80%; retain every "
            "checkpoint; sealed test panel remains closed"
        ),
        "test_panel_opened": False,
        "paper_evidence": False,
        "source_sha256": {
            str(path.relative_to(ROOT)): sha256(path) for path in sources
        },
    }
    validate_protocol(payload, source=PROTOCOL)
    return {**payload, "contract_sha256": digest(payload)}


def trainer_protocol() -> SimpleNamespace:
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
    "EPSILON_TARGET_FLOOR_EPISODE", "EPSILON_WARMUP_DECISIONS", "GAMMA",
    "LATE_CHECKPOINTS", "MAX_STEPS", "METHOD_REFERENCE", "METHOD_REVISION",
    "METHOD_VERSION", "MODEL_SEEDS", "PROTOCOL", "REPORT_CHECKPOINTS",
    "SCENARIO", "SCENARIOS", "SPEC", "TD_BACKUP_IMPLEMENTATION",
    "TOTAL_EPISODES", "TRAIN_BASE", "TRAINING_JOB_BUDGET", "TargetDenseEnv",
    "UNCHANGED_METHOD_FIELDS", "VALIDATION_EVERY", "VALIDATION_SEEDS",
    "contract", "digest", "method_declaration", "scenario_for", "sha256",
    "trainer_protocol",
]
