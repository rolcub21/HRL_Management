"""Frozen work-matched identities for target-size Stage A v4."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from experiments.conditioned_vcg.target_size_dense_stage_a_v2 import protocol as v2


PROTOCOL = "vcg_target_size_dense_stage_a_work_matched_cached_development_v4"
METHOD_VERSION = "vcg_dense_v1_1_target_size_work_matched_cached_v4"
TD_BACKUP_IMPLEMENTATION = "identity_state_dedup_plus_frozen_target_score_cache_v1"
SCENARIOS = v2.SCENARIOS
MODEL_SEEDS = v2.MODEL_SEEDS
TRAINING_JOB_BUDGET = 4_000
EPISODES_BY_SCENARIO = {
    "empty_8x8_n36": 112,
    "empty_10x10_n64": 63,
}
# Patched to the selected scenario by run.py before entering the v2 wrapper.
TOTAL_EPISODES = 0
VALIDATION_EVERY = 6
CHECKPOINT_EVERY = 1
VALIDATION_SEEDS = v2.VALIDATION_SEEDS
TRAIN_BASE = v2.TRAIN_BASE
ROOT = v2.ROOT
DEFAULT_OUTPUT = ROOT / "results/vcg-target-size-dense-stage-a-development-v4"
scenario_for = v2.scenario_for
instance_for = v2.instance_for
TargetDenseEnv = v2.TargetDenseEnv


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _digest(payload: dict) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def contract_payload() -> dict:
    directory = Path(__file__).resolve().parent
    v2_directory = directory.parent / "target_size_dense_stage_a_v2"
    return {
        "protocol": PROTOCOL,
        "method_version": METHOD_VERSION,
        "scope": "target-size Stage A development; work-matched training exposure; no test panel opened",
        "scenarios": {name: value.public_dict() for name, value in SCENARIOS.items()},
        "model_seeds": list(MODEL_SEEDS),
        "historical_5x5_training_jobs": 500 * 8,
        "training_job_budget": TRAINING_JOB_BUDGET,
        "episodes_by_scenario": EPISODES_BY_SCENARIO,
        "realized_training_jobs": {
            name: EPISODES_BY_SCENARIO[name] * scenario.total_jobs
            for name, scenario in SCENARIOS.items()
        },
        "budget_rule": "ceil(4000 / jobs_per_episode)",
        "validation_every": VALIDATION_EVERY,
        "validation_seeds": {name: list(values) for name, values in VALIDATION_SEEDS.items()},
        "checkpoint_every": CHECKPOINT_EVERY,
        "objective": "40-1.5|e|-0.5(|e|-20)+",
        "gamma": 0.99,
        "epsilon": {"start": 0.90, "end": 0.05, "decisions": 3000},
        "agent_family": "episodic_viability_graph_hierarchy",
        "instance_rule": "E28._instance: empty start, arrivals at 10*j, Poisson(80) durations",
        "certification_acceleration": "D12 relocation family + E14 timing cache/path cleanup",
        "td_backup_implementation": TD_BACKUP_IMPLEMENTATION,
        "selection_scoring_implementation": "native",
        "chosen_action_gradient_path": "native",
        "target_score_cache": {
            "scope": "Q_target only while frozen",
            "invalidation": "before first use after every target synchronization",
            "maximum_entries": 1_000_000,
            "checkpointed": False,
        },
        "source_sha256": {
            str(path): _sha256(path)
            for path in (
                directory / "protocol.py",
                directory / "acceleration.py",
                directory / "run.py",
                directory / "run.sh",
                v2_directory / "protocol.py",
                v2_directory / "run.py",
                ROOT / "train_vcg_dense_proper.py",
                ROOT / "run_vcg_objective_gamma_audit.py",
                ROOT / "viability_graph_hierarchy.py",
                ROOT / "viability_graph_episodic_audit.py",
                ROOT / "PSLAP/yard_graph.py",
            )
        },
    }


def contract() -> dict:
    payload = contract_payload()
    return {**payload, "contract_sha256": _digest(payload)}
