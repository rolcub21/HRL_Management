"""Frozen protocol for the one-seed E191 long-exploration diagnostic."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

from projects.temporal_consequence_realization.method_conformance import CHECKLIST
from experiments.conditioned_vcg.E191_fresh_long_stay_crowding_250k_v1 import (
    protocol as e191,
)


ROOT = e191.ROOT
PROTOCOL = "vcg_conditioned_e215_e191_long_exploration_drift_pilot_274k_v1"
METHOD_VERSION = "vcg_dense_v1_1_e215_e191_long_exploration_drift_pilot"
SCENARIO = e191.SCENARIO
SCENARIOS = e191.SCENARIOS
MODEL_SEEDS = (0,)
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

# E191 seed 0 executed 31,252 macro decisions in 120 episodes, or
# 260.433 decisions/episode.  57,300 decisions therefore puts the epsilon
# floor at episode 220.02 at that observed rate.  The old 18,500-decision
# schedule reached the floor at episode 70 and is the only training change.
E191_SEED0_EPISODES_OBSERVED = 120
E191_SEED0_DECISIONS_OBSERVED = 31_252
EPSILON_DECAY_DECISIONS = 57_300
EPSILON_TARGET_FLOOR_EPISODE = 220

REPORT_CHECKPOINTS = tuple(range(VALIDATION_EVERY, TOTAL_EPISODES + 1,
                                 VALIDATION_EVERY))
LATE_CHECKPOINTS = REPORT_CHECKPOINTS
TD_BACKUP_IMPLEMENTATION = e191.TD_BACKUP_IMPLEMENTATION
SPEC = e191.SPEC
TargetDenseEnv = e191.TargetDenseEnv
DEFAULT_OUTPUT = (
    ROOT
    / "results/vcg-conditioned-e215-e191-long-exploration-drift-pilot-274k-v1"
)
ORIGINAL_E191_OUTPUT = e191.DEFAULT_OUTPUT
scenario_for = e191.scenario_for
digest = e191.digest
sha256 = e191.sha256
e189 = e191.e189
workload = e191.workload
UNCHANGED_METHOD_FIELDS = e191.UNCHANGED_METHOD_FIELDS

METHOD_REFERENCE = (
    "projects/temporal_consequence_realization/METHOD_REFERENCE.md"
)
METHOD_REVISION = "TCR-METHOD-2026-10-29.8"
CLAIM_LEVEL = "expressibility"
DECLARED_DEVIATIONS = [
    "E215 is a historical conditioned-VCG optimization diagnostic, not an "
    "implementation or evidence claim for the canonical law-condition method; "
    "the validator's minimum claim-level label is used only for run provenance."
]
ORACLE_INPUTS: list[str] = []


def _conformance_audit() -> dict[str, dict[str, str]]:
    audit = {}
    for identifier, _description in CHECKLIST:
        if identifier == "reproducibility.method_revision":
            audit[identifier] = {
                "status": "pass",
                "evidence": f"The protocol pins {METHOD_REVISION}.",
            }
        elif identifier == "reproducibility.content_hashes":
            audit[identifier] = {
                "status": "pass",
                "evidence": (
                    "The contract hashes E215 and inherited E191 sources; the "
                    "trainer retains checkpoint and contract hashes."
                ),
            }
        elif identifier == "reproducibility.oracle_declarations":
            audit[identifier] = {
                "status": "pass",
                "evidence": (
                    "No oracle inputs are used and the historical-method "
                    "deviation is explicit."
                ),
            }
        elif identifier == "reproducibility.failed_gates_retained":
            audit[identifier] = {
                "status": "pass",
                "evidence": (
                    "All per-episode checkpoints, training diagnostics, fixed-"
                    "panel validations, and failed comparisons are retained."
                ),
            }
        else:
            audit[identifier] = {
                "status": "not_applicable",
                "evidence": (
                    "E215 diagnoses epsilon/replay optimization in the historical "
                    "conditioned-VCG controller and makes no canonical law-"
                    "condition claim."
                ),
            }
    return audit


def observed_decisions_per_episode() -> float:
    return E191_SEED0_DECISIONS_OBSERVED / E191_SEED0_EPISODES_OBSERVED


def implied_floor_episode() -> float:
    return EPSILON_DECAY_DECISIONS / observed_decisions_per_episode()


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
        ROOT
        / "projects/temporal_consequence_realization/method_conformance.py",
    )
    payload = deepcopy(parent)
    payload.pop("contract_sha256", None)
    payload.update({
        "schema_version": 1,
        "protocol": PROTOCOL,
        "method_version": METHOD_VERSION,
        "scope": (
            "one-seed fresh E191 late-drift diagnostic; only epsilon decay "
            "changes; development panel only"
        ),
        "parent_protocol": e191.PROTOCOL,
        "parent_contract_sha256": parent["contract_sha256"],
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
            "implied_floor_episode_at_e191_seed0_rate": implied_floor_episode(),
            "derivation": {
                "source_protocol": e191.PROTOCOL,
                "source_model_seed": 0,
                "observed_episodes": E191_SEED0_EPISODES_OBSERVED,
                "observed_macro_decisions": E191_SEED0_DECISIONS_OBSERVED,
                "mean_macro_decisions_per_episode": observed_decisions_per_episode(),
            },
            "only_training_change_from_e191": True,
        },
        "drift_diagnostic": {
            "primary_matched_seed": 0,
            "primary_common_checkpoints": [20, 40, 60, 80, 100, 120],
            "post_new_floor_checkpoints": [240, 260],
            "original_e191_epsilon_decay_decisions": e191.EPSILON_DECAY_DECISIONS,
            "replay_capacity_unchanged": 20_000,
            "optimizer_unchanged": True,
            "network_unchanged": True,
            "objective_unchanged": True,
            "certification_unchanged": True,
            "acceleration_unchanged": True,
            "test_panel_opened": False,
        },
        "checkpoint_selection": (
            "diagnostic curve on the unchanged fixed nine-case development "
            "panel; do not select or open the sealed test panel"
        ),
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
    "VALIDATION_SEEDS", "contract", "digest", "implied_floor_episode",
    "observed_decisions_per_episode", "scenario_for", "sha256",
    "trainer_protocol",
]
