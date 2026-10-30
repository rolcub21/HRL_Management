"""Frozen protocol for the seed-1/2 E215 training replication."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

from experiments.conditioned_vcg.E191_fresh_long_stay_crowding_250k_v1 import (
    protocol as e191,
)


ROOT = e191.ROOT
PROTOCOL = "vcg_conditioned_e218_e215_long_exploration_replication_277k_v1"
METHOD_VERSION = "vcg_dense_v1_1_e218_e215_long_exploration_replication"
SCENARIO = e191.SCENARIO
SCENARIOS = e191.SCENARIOS
MODEL_SEEDS = (1, 2)
TOTAL_EPISODES = 300
EPISODES_BY_SCENARIO = {SCENARIO: TOTAL_EPISODES}
VALIDATION_EVERY = 20
CHECKPOINT_EVERY = 1
VALIDATION_SEEDS = e191.VALIDATION_SEEDS
TRAIN_BASE = e191.TRAIN_BASE
TRAINING_JOB_BUDGET = TOTAL_EPISODES * e191.workload.TOTAL_JOBS
MAX_STEPS = e191.MAX_STEPS
GAMMA = e191.GAMMA
EPSILON_START = 0.90
EPSILON_END = 0.05
EPSILON_WARMUP_DECISIONS = 0
EPSILON_DECAY_DECISIONS = 57_300
EPSILON_TARGET_FLOOR_EPISODE = 220
REPORT_CHECKPOINTS = tuple(range(VALIDATION_EVERY, TOTAL_EPISODES + 1,
                                 VALIDATION_EVERY))
LATE_CHECKPOINTS = REPORT_CHECKPOINTS
TD_BACKUP_IMPLEMENTATION = e191.TD_BACKUP_IMPLEMENTATION
SPEC = e191.SPEC
TargetDenseEnv = e191.TargetDenseEnv
scenario_for = e191.scenario_for
digest = e191.digest
sha256 = e191.sha256
workload = e191.workload
DEFAULT_OUTPUT = (
    ROOT
    / "results/vcg-conditioned-e218-e215-long-exploration-replication-277k-v1"
)

METHOD_REFERENCE = "projects/temporal_consequence_realization/METHOD_REFERENCE.md"
METHOD_REVISION = "TCR-METHOD-2026-10-30.9"
CLAIM_LEVEL = "expressibility"
DECLARED_DEVIATIONS = [
    "E218 replicates the historical conditioned-VCG E215 optimization "
    "diagnostic and makes no canonical law-condition method claim."
]
ORACLE_INPUTS: list[str] = []

# Snapshot the checklist for this revision.  An in-flight multi-day run must
# not acquire new declaration fields merely because the living reference later
# advances.  New outputs still pass the current validator before episode 1.
CHECKLIST_SNAPSHOT = (
    "representation.shared_law",
    "representation.permutation_invariant_read_set",
    "representation.carrier_frame_responsibility",
    "representation.state_memory_roles",
    "representation.no_persistent_bypass",
    "representation.responsibility_audits",
    "representation.persistent_conditions",
    "representation.anonymous_conditions",
    "representation.continuous_attribute_default",
    "representation.full_field_support",
    "representation.no_uncharged_addresses",
    "learning.one_full_likelihood",
    "learning.joint_or_declared_alternation",
    "learning.belief_samples",
    "learning.capacity_by_total_cost",
    "learning.evidence_monotonicity",
    "learning.labels_grading_only",
    "learning.importance_correction",
    "learning.restoration_audit",
    "control.persistent_imagined_conditions",
    "control.update_after_consequence",
    "control.evidence_weighted_laws",
    "control.heldout_calibration",
    "control.belief_scramble_preoutcome",
    "control.absorbing_goals_separate_risk",
    "control.law_belief_values",
    "reproducibility.method_revision",
    "reproducibility.content_hashes",
    "reproducibility.oracle_declarations",
    "reproducibility.failed_gates_retained",
)


def _conformance_audit() -> dict[str, dict[str, str]]:
    audit = {}
    for identifier in CHECKLIST_SNAPSHOT:
        if identifier == "reproducibility.method_revision":
            audit[identifier] = {
                "status": "pass",
                "evidence": f"The protocol pins {METHOD_REVISION}.",
            }
        elif identifier == "reproducibility.content_hashes":
            audit[identifier] = {
                "status": "pass",
                "evidence": (
                    "The contract hashes E218 and all inherited controller, "
                    "scenario, acceleration, and trainer sources."
                ),
            }
        elif identifier == "reproducibility.oracle_declarations":
            audit[identifier] = {
                "status": "pass",
                "evidence": "No oracle inputs are used; the historical scope is explicit.",
            }
        elif identifier == "reproducibility.failed_gates_retained":
            audit[identifier] = {
                "status": "pass",
                "evidence": (
                    "Per-episode checkpoints and fixed-panel validations are retained."
                ),
            }
        else:
            audit[identifier] = {
                "status": "not_applicable",
                "evidence": (
                    "E218 is a conditioned-VCG optimization replication and "
                    "makes no canonical law-condition method claim."
                ),
            }
    return audit


def contract() -> dict:
    parent = e191.contract()
    directory = Path(__file__).resolve().parent
    sources = (
        directory / "protocol.py",
        directory / "run.py",
        directory / "run.sh",
        directory / "README.md",
        ROOT
        / "experiments/conditioned_vcg/E191_fresh_long_stay_crowding_250k_v1/protocol.py",
        ROOT
        / "experiments/conditioned_vcg/E191_fresh_long_stay_crowding_250k_v1/run.py",
        ROOT
        / "experiments/conditioned_vcg/E191_fresh_long_stay_crowding_250k_v1/stage.py",
        ROOT / "train_vcg_dense_proper.py",
    )
    payload = deepcopy(parent)
    payload.pop("contract_sha256", None)
    payload.update({
        "schema_version": 1,
        "protocol": PROTOCOL,
        "method_version": METHOD_VERSION,
        "scope": (
            "fresh matched seed-1/2 replication of E215 long-exploration "
            "training; fixed development panel only"
        ),
        "parent_training_protocol": e191.PROTOCOL,
        "parent_training_contract_sha256": parent["contract_sha256"],
        "matched_diagnostic_protocol": (
            "vcg_conditioned_e215_e191_long_exploration_drift_pilot_274k_v1"
        ),
        "model_seeds": list(MODEL_SEEDS),
        "episodes_per_seed": TOTAL_EPISODES,
        "training_jobs_per_seed": TRAINING_JOB_BUDGET,
        "validation_every": VALIDATION_EVERY,
        "checkpoint_every": CHECKPOINT_EVERY,
        "report_checkpoints": list(REPORT_CHECKPOINTS),
        "epsilon": {
            "start": EPSILON_START,
            "end": EPSILON_END,
            "warmup_decisions": EPSILON_WARMUP_DECISIONS,
            "decay_decisions": EPSILON_DECAY_DECISIONS,
            "target_floor_episode_approximate": EPSILON_TARGET_FLOOR_EPISODE,
            "matched_to_e215": True,
        },
        "replication": {
            "seeds": list(MODEL_SEEDS),
            "fresh_network_per_seed": True,
            "same_training_instances_as_e215_seed0": True,
            "same_validation_panel_as_e215_seed0": True,
            "architecture_unchanged": True,
            "objective_unchanged": True,
            "certification_unchanged": True,
            "progress_mask_unchanged": True,
            "optimizer_unchanged": True,
            "replay_capacity": 20_000,
            "acceleration_unchanged": True,
            "test_panel_opened": False,
        },
        "method_reference": METHOD_REFERENCE,
        "method_revision": METHOD_REVISION,
        "claim_level": CLAIM_LEVEL,
        "declared_deviations": list(DECLARED_DEVIATIONS),
        "oracle_inputs": list(ORACLE_INPUTS),
        "nonconforming_shortcuts": [],
        "conformance_audit": _conformance_audit(),
        "test_panel_opened": False,
        "paper_evidence": False,
        "source_sha256": {
            str(path.relative_to(ROOT)): sha256(path) for path in sources
        },
    })
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
    "LATE_CHECKPOINTS", "MAX_STEPS", "METHOD_VERSION", "MODEL_SEEDS",
    "PROTOCOL", "REPORT_CHECKPOINTS", "SCENARIO", "SCENARIOS", "SPEC",
    "TD_BACKUP_IMPLEMENTATION", "TOTAL_EPISODES", "TRAIN_BASE",
    "TRAINING_JOB_BUDGET", "TargetDenseEnv", "VALIDATION_EVERY",
    "VALIDATION_SEEDS", "contract", "digest", "scenario_for", "sha256",
    "trainer_protocol",
]
