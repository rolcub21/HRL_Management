"""Frozen contract for the E105 wait-aware Stage-A pilot."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from experiments.conditioned_vcg.E78_original_vcg_base80_full_8x8_143k_v1 import (
    protocol as e78,
)
from .rewards import WAIT_CAP, WAIT_PENALTY_PER_JOB_STEP, WaitAwareDenseEnv


ROOT = e78.ROOT
PROTOCOL = "vcg_conditioned_e105_wait_aware_stage_a_8x8_169k_v1"
METHOD_VERSION = "vcg_dense_v1_1_e105_wait_aware_stage_a"
SCENARIO = e78.SCENARIO
MODEL_SEEDS = (0,)
PILOT_MAX_EPISODE = 36
DEFAULT_OUTPUT = ROOT / "results/vcg-conditioned-e105-wait-aware-stage-a-8x8-169k-v1"


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
        directory / "rewards.py",
        directory / "instrumentation.py",
        directory / "run.py",
        directory / "run.sh",
        ROOT / "experiments/conditioned_vcg/E78_original_vcg_base80_full_8x8_143k_v1/protocol.py",
        ROOT / "experiments/conditioned_vcg/E100_accelerated_release_clock_163k_v1/fast_inputs.py",
        ROOT / "experiments/conditioned_vcg/target_size_dense_stage_a_v4/run.py",
        ROOT / "experiments/conditioned_vcg/target_size_dense_stage_a_v4/acceleration.py",
        ROOT / "experiments/conditioned_vcg/target_size_dense_stage_a_v2/run.py",
        ROOT / "train_vcg_dense_proper.py",
        ROOT / "vcg_objective_audit.py",
        ROOT / "run_vcg_objective_gamma_audit.py",
    )
    payload = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "method_version": METHOD_VERSION,
        "scope": "development-only one-seed 8x8 Stage-A mechanism pilot",
        "control": e78.PROTOCOL,
        "architecture": "original VCG-Dense v1.1 Stage A; lambda 0",
        "scenario": SCENARIO,
        "scenario_public": e78.SCENARIOS[SCENARIO].public_dict(),
        "model_seeds": list(MODEL_SEEDS),
        "pilot_max_episode": PILOT_MAX_EPISODE,
        "validation_every": e78.VALIDATION_EVERY,
        "validation_seeds": list(e78.VALIDATION_SEEDS[SCENARIO]),
        "objective": {
            "delivery_component": e78.SPEC.to_dict(),
            "outside_wait_component": (
                f"-{WAIT_PENALTY_PER_JOB_STEP} per primitive step for every "
                f"released, unstored job with age > {WAIT_CAP} after that step"
            ),
            "completed_episode_wait_term": (
                f"-{WAIT_PENALTY_PER_JOB_STEP} * sum_i max(0, "
                f"stored_i-arrival_i-{WAIT_CAP})"
            ),
            "policy_invariance_claim": False,
        },
        "gamma": 0.99,
        "reward_scale": 0.01,
        "epsilon": {"start": 0.90, "end": 0.05, "decisions": 3000},
        "single_change_from_e78": "accrued outside-wait-cap penalty",
        "everything_else": "E78 recipe with E100 exact graph-input cache",
        "reported_clocks": ["release", "storage"],
        "mechanism_thresholds_at_episode_36": {
            "strict_completion_rate": 1.0,
            "mean_wait_before_storage_max": 256.0,
            "storage_clock_mae_max": 20.0,
            "within_20_rate_min": 0.70,
            "mean_steps_max": 1200.0,
        },
        "threshold_origin": (
            "development screen: >=25% wait reduction from E104 lambda-0 "
            "341.2-step reference while retaining its storage-clock regime"
        ),
        "checkpoint_policy": "report every checkpoint; do not select best checkpoint",
        "stage_b_authorized": False,
        "independent_test_opened": False,
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
        SCENARIOS=e78.SCENARIOS,
        MODEL_SEEDS=MODEL_SEEDS,
        EPISODES_BY_SCENARIO=e78.EPISODES_BY_SCENARIO,
        TOTAL_EPISODES=0,
        VALIDATION_EVERY=e78.VALIDATION_EVERY,
        CHECKPOINT_EVERY=e78.CHECKPOINT_EVERY,
        VALIDATION_SEEDS=e78.VALIDATION_SEEDS,
        TRAIN_BASE=e78.TRAIN_BASE,
        TRAINING_JOB_BUDGET=e78.TRAINING_JOB_BUDGET,
        TD_BACKUP_IMPLEMENTATION=e78.TD_BACKUP_IMPLEMENTATION,
        LATE_CHECKPOINTS=e78.LATE_CHECKPOINTS,
        DEFAULT_OUTPUT=DEFAULT_OUTPUT,
        TargetDenseEnv=WaitAwareDenseEnv,
        scenario_for=e78.scenario_for,
        contract=lambda: value,
    )
