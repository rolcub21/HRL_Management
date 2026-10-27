"""E188 protocol with only a bounded floating-point audit repair."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

from experiments.conditioned_vcg.E188_mask_only_progress_guard_247k_v1 import (
    protocol as v1,
)


ROOT = v1.ROOT
PROTOCOL = "vcg_conditioned_e189_e188_roundoff_repair_248k_v1"
METHOD_VERSION = "vcg_dense_v1_1_e189_e188_roundoff_repair"
SCENARIO = v1.SCENARIO
SCENARIOS = v1.SCENARIOS
MODEL_SEEDS = v1.MODEL_SEEDS
TOTAL_EPISODES = v1.TOTAL_EPISODES
EPISODES_BY_SCENARIO = dict(v1.EPISODES_BY_SCENARIO)
VALIDATION_EVERY = v1.VALIDATION_EVERY
CHECKPOINT_EVERY = v1.CHECKPOINT_EVERY
VALIDATION_SEEDS = deepcopy(v1.VALIDATION_SEEDS)
TRAIN_BASE = dict(v1.TRAIN_BASE)
TRAINING_JOB_BUDGET = v1.TRAINING_JOB_BUDGET
MAX_STEPS = v1.MAX_STEPS
GAMMA = v1.GAMMA
EPSILON_START = v1.EPSILON_START
EPSILON_END = v1.EPSILON_END
EPSILON_WARMUP_DECISIONS = v1.EPSILON_WARMUP_DECISIONS
EPSILON_DECAY_DECISIONS = v1.EPSILON_DECAY_DECISIONS
REPORT_CHECKPOINTS = v1.REPORT_CHECKPOINTS
LATE_CHECKPOINTS = v1.LATE_CHECKPOINTS
TD_BACKUP_IMPLEMENTATION = v1.TD_BACKUP_IMPLEMENTATION
SPEC = v1.SPEC
TargetDenseEnv = v1.TargetDenseEnv
scenario_for = v1.scenario_for
DEFAULT_OUTPUT = ROOT / "results/vcg-conditioned-e189-e188-roundoff-repair-248k-v1"
SOURCE_OUTPUT = v1.DEFAULT_OUTPUT
PARENT_OUTPUT = v1.PARENT_OUTPUT

# E188's runner expects these public protocol helpers.
e187 = v1.e187
MATCHED_PARENT_FIELDS = v1.MATCHED_PARENT_FIELDS
sha256 = v1.sha256
digest = v1.digest


def contract() -> dict:
    parent = v1.contract()
    directory = Path(__file__).resolve().parent
    sources = deepcopy(parent["source_sha256"])
    for name in ("protocol.py", "roundoff.py", "run.py", "run.sh"):
        path = directory / name
        sources[str(path.relative_to(ROOT))] = sha256(path)
    payload = {
        key: deepcopy(value)
        for key, value in parent.items()
        if key != "contract_sha256"
    }
    payload.update({
        "protocol": PROTOCOL,
        "method_version": METHOD_VERSION,
        "scope": (
            "E188 paired guard arm with numerical-only return-accounting "
            "repair; development only"
        ),
        "numerical_parent_protocol": v1.PROTOCOL,
        "numerical_parent_contract_sha256": parent["contract_sha256"],
        "numerical_audit_repair": {
            "behavioral_change": False,
            "learning_change": False,
            "reward_change": False,
            "step_level_checks_unchanged": True,
            "scope": "end-of-episode primitive-versus-macro return comparisons",
            "bound": (
                "episode-specific sequential floating-point summation bound "
                "derived from recorded primitive rewards"
            ),
            "provenance": (
                "validated E79-v2 long-episode roundoff repair"
            ),
        },
        "source_sha256": sources,
    })
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
    "METHOD_VERSION", "MODEL_SEEDS", "PARENT_OUTPUT", "PROTOCOL",
    "REPORT_CHECKPOINTS", "SCENARIO", "SCENARIOS", "SOURCE_OUTPUT", "SPEC",
    "TD_BACKUP_IMPLEMENTATION", "TOTAL_EPISODES", "TRAIN_BASE",
    "TRAINING_JOB_BUDGET", "VALIDATION_EVERY", "VALIDATION_SEEDS",
    "TargetDenseEnv", "contract", "digest", "e187", "scenario_for",
    "sha256", "trainer_protocol", "v1",
]
