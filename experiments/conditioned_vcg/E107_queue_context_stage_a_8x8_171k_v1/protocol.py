"""Frozen protocol for queue-observed, long-horizon Stage-A training."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from experiments.conditioned_vcg.E105_wait_aware_stage_a_8x8_169k_v1 import (
    protocol as e105,
)
from .representation import QUEUE_CONTEXT_NAMES


ROOT = e105.ROOT
PROTOCOL = "vcg_conditioned_e107_queue_context_stage_a_8x8_171k_v1"
METHOD_VERSION = "vcg_dense_v1_1_e107_queue_observed_stage_a"
SCENARIO = e105.SCENARIO
MODEL_SEEDS = (0,)
MAX_EPISODE = 224
DECISION_CHECKPOINTS = (36, 60, 112, 160, 224)
EPSILON_START = 0.90
EPSILON_END = 0.05
EPSILON_WARMUP_DECISIONS = 0
EPSILON_DECAY_DECISIONS = 18_000
TRAIN_BASE = {SCENARIO: 171_000_000}
TRAINING_JOB_BUDGET = MAX_EPISODE * e105.e78.SCENARIOS[SCENARIO].total_jobs
TD_BACKUP_IMPLEMENTATION = (
    "e107_queue_context_exact_device_graph_inputs_frozen_target_cache_v1"
)
DEFAULT_OUTPUT = (
    ROOT / "results/vcg-conditioned-e107-queue-context-stage-a-8x8-171k-v1"
)
HORIZON_RULE = (
    "fresh development training through at most 224 episodes; inspect the full "
    "trajectory at 36,60,112,160,224; episode 112 is not a stopping rule"
)


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
    scenario = e105.e78.SCENARIOS[SCENARIO]
    sources = (
        directory / "protocol.py",
        directory / "representation.py",
        directory / "instrumentation.py",
        directory / "run.py",
        directory / "run.sh",
        ROOT / "experiments/conditioned_vcg/E105_wait_aware_stage_a_8x8_169k_v1/rewards.py",
        ROOT / "experiments/conditioned_vcg/E105_wait_aware_stage_a_8x8_169k_v1/instrumentation.py",
        ROOT / "experiments/conditioned_vcg/E100_accelerated_release_clock_163k_v1/fast_inputs.py",
        ROOT / "experiments/conditioned_vcg/target_size_dense_stage_a_v4/acceleration.py",
        ROOT / "experiments/conditioned_vcg/target_size_dense_stage_a_v3/acceleration.py",
        ROOT / "experiments/conditioned_vcg/target_size_dense_stage_a_v2/run.py",
        ROOT / "train_vcg_dense_proper.py",
        ROOT / "run_vcg_objective_gamma_audit.py",
        ROOT / "viability_graph_hierarchy.py",
        ROOT / "viability_graph_episodic_audit.py",
    )
    payload = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "method_version": METHOD_VERSION,
        "scope": "development-only one-seed queue-observed Stage-A training",
        "scenario": SCENARIO,
        "scenario_public": scenario.public_dict(),
        "model_seeds": list(MODEL_SEEDS),
        "fresh_initialization": True,
        "source_checkpoint": None,
        "maximum_episode": MAX_EPISODE,
        "decision_checkpoints": list(DECISION_CHECKPOINTS),
        "training_job_budget": TRAINING_JOB_BUDGET,
        "train_instance_seed_base": TRAIN_BASE[SCENARIO],
        "validation_every": e105.e78.VALIDATION_EVERY,
        "validation_seeds": list(e105.e78.VALIDATION_SEEDS[SCENARIO]),
        "validation_panel_role": "reused development panel; no test claim",
        "objective": {
            "delivery_component": e105.e78.SPEC.to_dict(),
            "outside_wait_component": (
                "-0.5 per primitive step for every released, unstored job "
                "whose outside age is at least 80"
            ),
        },
        "critic_information": {
            "native_graph_and_action_inputs_retained": True,
            "public_queue_context": list(QUEUE_CONTEXT_NAMES),
            "context_bounds": "every public context value lies in [0,1]",
            "current_boundary_context_only": True,
            "context_stored_in_replay": True,
            "future_arrival_schedule_visible": False,
            "unreleased_duration_visible": False,
        },
        "epsilon": {
            "start": EPSILON_START,
            "end": EPSILON_END,
            "warmup_decisions": EPSILON_WARMUP_DECISIONS,
            "decay_decisions": EPSILON_DECAY_DECISIONS,
            "post_floor_exploration_continues": True,
        },
        "epsilon_change_from_e105": "3000 -> 18000 decision linear decay",
        "horizon_change_from_e105": "36 episode pilot -> 224 episode maximum",
        "horizon_rule": HORIZON_RULE,
        "gamma": 0.99,
        "reward_scale": 0.01,
        "certification_and_guard": "unchanged exact VCG-Dense v1.1",
        "td_backup_implementation": TD_BACKUP_IMPLEMENTATION,
        "checkpoint_policy": (
            "report full trajectory; do not treat a single development-best "
            "checkpoint as paper evidence"
        ),
        "joint_mechanism_thresholds": {
            "strict_completion_rate": 1.0,
            "mean_wait_before_storage_max": 256.0,
            "storage_clock_mae_max": 20.0,
            "within_20_rate_min": 0.70,
            "mean_steps_max": 1200.0,
        },
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
        SCENARIOS=e105.e78.SCENARIOS,
        MODEL_SEEDS=MODEL_SEEDS,
        EPISODES_BY_SCENARIO={SCENARIO: MAX_EPISODE},
        TOTAL_EPISODES=MAX_EPISODE,
        VALIDATION_EVERY=e105.e78.VALIDATION_EVERY,
        CHECKPOINT_EVERY=e105.e78.CHECKPOINT_EVERY,
        VALIDATION_SEEDS=e105.e78.VALIDATION_SEEDS,
        TRAIN_BASE=TRAIN_BASE,
        TRAINING_JOB_BUDGET=TRAINING_JOB_BUDGET,
        TD_BACKUP_IMPLEMENTATION=TD_BACKUP_IMPLEMENTATION,
        LATE_CHECKPOINTS=DECISION_CHECKPOINTS,
        DEFAULT_OUTPUT=DEFAULT_OUTPUT,
        TargetDenseEnv=e105.WaitAwareDenseEnv,
        scenario_for=e105.e78.scenario_for,
        contract=lambda: value,
    )
