"""Frozen protocol for the wait-corrected E78 continuation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from experiments.conditioned_vcg.E78_original_vcg_base80_full_8x8_143k_v1 import (
    protocol as e78,
)
from .objective import (
    COMPONENT_NAMES,
    OBJECTIVE_WEIGHTS,
    TOLERANCE,
    WaitCorrectedE78Env,
)
from .representation import (
    QUEUE_CONTEXT_NAMES,
    source_manifest,
)


ROOT = e78.ROOT
PROTOCOL = "vcg_conditioned_e187_wait_corrected_e78_246k_v1"
METHOD_VERSION = "vcg_dense_v1_1_e187_wait_corrected_queue_observed"
SCENARIO = e78.SCENARIO
SCENARIOS = e78.SCENARIOS
MODEL_SEEDS = (0, 1, 2)
TOTAL_EPISODES = 300
EPISODES_BY_SCENARIO = {SCENARIO: TOTAL_EPISODES}
VALIDATION_EVERY = 20
CHECKPOINT_EVERY = 1
# Fresh development instances.  The inherited E78 panel selected the source
# checkpoints, so reusing it here would make the E187 learning curve
# source-selection-contaminated.  These seeds are disjoint from E78 training,
# validation, pilot, and sealed-test identities.
VALIDATION_SEEDS = {
    SCENARIO: tuple(range(246_187_000, 246_187_005))
}
TRAIN_BASE = {SCENARIO: 187_800_000}
TRAINING_JOB_BUDGET = TOTAL_EPISODES * SCENARIOS[SCENARIO].total_jobs
MAX_STEPS = SCENARIOS[SCENARIO].max_steps
GAMMA = 1.0
EPSILON_START = 0.90
EPSILON_END = 0.05
EPSILON_WARMUP_DECISIONS = 0
EPSILON_DECAY_DECISIONS = 18_500
REPORT_CHECKPOINTS = tuple(range(VALIDATION_EVERY, TOTAL_EPISODES + 1, VALIDATION_EVERY))
LATE_CHECKPOINTS = REPORT_CHECKPOINTS
TD_BACKUP_IMPLEMENTATION = (
    "e187_queue_context_exact_device_graph_inputs_frozen_target_cache_v1"
)
SPEC = e78.SPEC  # Audit scaffold only; E187 replaces every primitive reward.
TargetDenseEnv = WaitCorrectedE78Env
DEFAULT_OUTPUT = ROOT / "results/vcg-conditioned-e187-wait-corrected-e78-246k-v1"
scenario_for = e78.scenario_for


def _assert_seed_splits() -> None:
    development = set(VALIDATION_SEEDS[SCENARIO])
    reserved = set(e78.VALIDATION_SEEDS[SCENARIO])
    reserved.add(112_820)  # Historical target-size pilot.
    reserved.update(range(112_850, 112_853))  # Sealed target-size test.
    reserved.update(range(77_000, 77_050))  # Global sealed panel.
    reserved.update(range(69_000, 69_010))  # Global stress panel.
    for base in (112_800_000, 113_800_000, 114_800_000, 187_800_000,
                 188_800_000, 189_800_000):
        reserved.update(range(base, base + TOTAL_EPISODES))
    if development & reserved or len(development) != 5:
        raise RuntimeError("E187 development identities overlap a reserved split")


_assert_seed_splits()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def digest(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def contract() -> dict:
    directory = Path(__file__).resolve().parent
    sources = (
        directory / "protocol.py",
        directory / "objective.py",
        directory / "representation.py",
        directory / "instrumentation.py",
        directory / "run.py",
        directory / "run.sh",
        ROOT / "experiments/conditioned_vcg/E186_e78_long_exploration_245k_v1/protocol.py",
        ROOT / "experiments/conditioned_vcg/E186_e78_long_exploration_245k_v1/run.py",
        ROOT / "experiments/conditioned_vcg/E78_original_vcg_base80_full_8x8_143k_v1/protocol.py",
        ROOT / "experiments/conditioned_vcg/E70_original_vcg_reward_base_135k_v1/protocol.py",
        ROOT / "experiments/conditioned_vcg/E107_queue_context_stage_a_8x8_171k_v1/representation.py",
        ROOT / "experiments/conditioned_vcg/E100_accelerated_release_clock_163k_v1/fast_inputs.py",
        ROOT / "experiments/conditioned_vcg/target_size_dense_stage_a_v4/protocol.py",
        ROOT / "experiments/conditioned_vcg/target_size_dense_stage_a_v4/run.py",
        ROOT / "experiments/conditioned_vcg/target_size_dense_stage_a_v4/acceleration.py",
        ROOT / "experiments/conditioned_vcg/target_size_dense_stage_a_v3/acceleration.py",
        ROOT / "experiments/conditioned_vcg/target_size_dense_stage_a_v2/protocol.py",
        ROOT / "experiments/conditioned_vcg/target_size_dense_stage_a_v2/run.py",
        ROOT / "experiments/conditioned_vcg/target_size_vcg_training/protocol.py",
        ROOT / "experiments/conditioned_vcg/E28_empty_start_scale_baselines_100k/run.py",
        ROOT / "experiments/conditioned_vcg/development/D10_scalability_support_screen/occupancy_extension.py",
        ROOT / "experiments/conditioned_vcg/E14_certification_scalability_95k/reuse.py",
        ROOT / "experiments/conditioned_vcg/E13_operational_scalability_95k/program.py",
        ROOT / "example/episode_instance.py",
        ROOT / "example/small_rooms_env.py",
        ROOT / "example/Options/ExplicitAcceptOption.py",
        ROOT / "example/Options/DirectDeliverOption.py",
        ROOT / "example/Options/ReconfigureOption.py",
        ROOT / "example/Options/ExplicitDeferOption.py",
        ROOT / "train_vcg_dense_proper.py",
        ROOT / "train_viability_graph_smdp.py",
        ROOT / "train_viability_graph_smdp_proper.py",
        ROOT / "vcg_objective_audit.py",
        ROOT / "run_vcg_objective_gamma_audit.py",
        ROOT / "viability_graph_hierarchy.py",
        ROOT / "viability_graph_episodic_audit.py",
        ROOT / "PSLAP/dynamic_yard.py",
        ROOT / "PSLAP/neutral_protocol.py",
        ROOT / "PSLAP/viability.py",
        ROOT / "PSLAP/viability_candidates.py",
        ROOT / "PSLAP/viability_filter.py",
        ROOT / "PSLAP/viability_prioritizer.py",
        ROOT / "PSLAP/yard_graph.py",
        ROOT / "PSLAP/relocation_family_certification.py",
    )
    payload = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "method_version": METHOD_VERSION,
        "scope": "development-only wait-corrected E78 continuation",
        "scenario": SCENARIO,
        "scenario_public": SCENARIOS[SCENARIO].public_dict(),
        "model_seeds": list(MODEL_SEEDS),
        "episodes_per_seed": TOTAL_EPISODES,
        "training_jobs_per_seed": TRAINING_JOB_BUDGET,
        "train_instance_seed_base": TRAIN_BASE[SCENARIO],
        "validation_every": VALIDATION_EVERY,
        "validation_seeds": list(VALIDATION_SEEDS[SCENARIO]),
        "validation_panel_role": (
            "fresh E187-only development panel; disjoint from warm-source "
            "selection and the sealed test panel"
        ),
        "checkpoint_every": CHECKPOINT_EVERY,
        "report_checkpoints": list(REPORT_CHECKPOINTS),
        "objective": {
            "name": "J_storage_clock_plus_full_admission_wait",
            "formula": "2*T_storage + E_storage + 0.5*W_release_to_admission + 0.1*L + 10*R",
            "component_names": list(COMPONENT_NAMES),
            "weights": list(OBJECTIVE_WEIGHTS),
            "storage_tolerance_steps": TOLERANCE,
            "storage_due_rule": "stored_time_step + storage_steps_needed",
            "wait_rule": "stored_time_step - arrival_step",
            "wait_grace_steps": 0,
            "primitive_accrual": True,
            "gamma": GAMMA,
            "e78_dense_reward_used_for_learning": False,
        },
        "audit_scaffold_objective": SPEC.to_dict(),
        "gamma": GAMMA,
        "finite_horizon": {
            "steps": MAX_STEPS,
            "role": "error cap only; any hit is an explicit failed episode",
        },
        "epsilon": {
            "start": EPSILON_START,
            "end": EPSILON_END,
            "warmup_decisions": EPSILON_WARMUP_DECISIONS,
            "decay_decisions": EPSILON_DECAY_DECISIONS,
            "target_floor_episode_approximate": 220,
            "derivation": "E186 pooled E78 rate: 84.333 decisions/episode",
        },
        "workload": "E78 empty 8x8, 36 jobs, releases 10*j, Poisson(80) durations",
        "architecture": (
            "E78 graph architecture plus bounded public queue context; scalar Q"
        ),
        "critic_information": {
            "queue_context": list(QUEUE_CONTEXT_NAMES),
            "future_arrival_schedule_visible": False,
            "unreleased_duration_visible": False,
            "context_stored_in_replay": True,
        },
        "warm_start": {
            "sources": source_manifest(),
            "transferred_modules": ["graph_encoder"],
            "trainable_after_transfer": True,
            "reset_modules": ["action_encoder", "context_encoder", "q_head"],
            "reset_state": ["optimizer", "replay", "counters", "guard_memory"],
        },
        "frontier": {
            "physical_actions": "unchanged E78",
            "certification": "unchanged E78 exact safe frontier",
            "progress_guard": "unchanged E78 manuscript guard",
            "accept_first_mask": False,
            "inbound_defer_mask": False,
            "intake_restoration_preference": False,
        },
        "acceleration": {
            "certificate": "D12 relocation family plus E14 timing-invariant cache/path cleanup",
            "graph_inputs": "E100 exact device tensor cache",
            "queue_scoring": "E107 identity-deduplicated batched scoring",
            "td_backup": TD_BACKUP_IMPLEMENTATION,
            "decision_semantics_changed": False,
        },
        "checkpoint_selection": (
            "full curve; trainer best is descriptive development evidence only"
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
    "CHECKPOINT_EVERY",
    "DEFAULT_OUTPUT",
    "EPISODES_BY_SCENARIO",
    "EPSILON_DECAY_DECISIONS",
    "EPSILON_END",
    "EPSILON_START",
    "EPSILON_WARMUP_DECISIONS",
    "GAMMA",
    "MAX_STEPS",
    "METHOD_VERSION",
    "MODEL_SEEDS",
    "PROTOCOL",
    "REPORT_CHECKPOINTS",
    "SCENARIO",
    "SCENARIOS",
    "SPEC",
    "TD_BACKUP_IMPLEMENTATION",
    "TOTAL_EPISODES",
    "TRAIN_BASE",
    "TRAINING_JOB_BUDGET",
    "VALIDATION_EVERY",
    "VALIDATION_SEEDS",
    "TargetDenseEnv",
    "contract",
    "scenario_for",
    "trainer_protocol",
]
