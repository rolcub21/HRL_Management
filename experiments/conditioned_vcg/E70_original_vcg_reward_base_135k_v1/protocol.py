"""Frozen identity for a base-80 arm of the original target-size VCG."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from experiments.conditioned_vcg.target_size_dense_stage_a_v4 import protocol as v4
from vcg_objective_audit import TimingObjectiveSpec

ROOT = v4.ROOT
PROTOCOL = "vcg_conditioned_e70_original_vcg_reward_base_135k_v1"
METHOD_VERSION = "vcg_dense_v1_1_target_size_base80_e70"
SCENARIO = "empty_8x8_n36"
SCENARIOS = {SCENARIO: v4.SCENARIOS[SCENARIO]}
MODEL_SEEDS = (0,)
EPISODES_BY_SCENARIO = {SCENARIO: 112}
TOTAL_EPISODES = 0  # Patched to 112 by the Stage-A-v4 wrapper.
VALIDATION_EVERY = 6
CHECKPOINT_EVERY = 1
VALIDATION_SEEDS = {SCENARIO: v4.VALIDATION_SEEDS[SCENARIO]}
TRAIN_BASE = {SCENARIO: v4.TRAIN_BASE[SCENARIO]}
DEFAULT_OUTPUT = ROOT / "results/vcg-conditioned-e70-original-vcg-reward-base-135k-v1"
BASE40_SUMMARY = (v4.DEFAULT_OUTPUT / "training/operational" / SCENARIO
                  / "seed-0/training-summary.json")
SPEC = TimingObjectiveSpec.dense(dense_b=80.0, lambda_abs=1.5,
                                 lambda_outside=0.5, window=20.0)
TD_BACKUP_IMPLEMENTATION = v4.TD_BACKUP_IMPLEMENTATION
TRAINING_JOB_BUDGET = v4.TRAINING_JOB_BUDGET
scenario_for = v4.scenario_for
TargetDenseEnv = v4.TargetDenseEnv


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _digest(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def _row(record: dict) -> dict:
    return {
        "checkpoint_episode": int(record["checkpoint_episode"]),
        "runs": [{
            "instance_seed": int(run["instance_seed"]),
            "episode_instance_id": run["episode_instance_id"],
            "strict_method_success": bool(run["strict_method_success"]),
            "delivery_count": int(run["delivery_count"]),
            "dense_rescored_return": float(run["dense_rescored_return"]),
            "mean_absolute_error": run["mean_absolute_error"],
            "within_target_window_rate": run["within_target_window_rate"],
            "steps": int(run["steps"]),
            "relocations": int(run["audit"]["relocations"]),
        } for run in record["runs"]],
    }


def base40_snapshot() -> list[dict]:
    source = json.loads(BASE40_SUMMARY.read_text(encoding="utf-8"))
    if (source.get("model_seed") != 0
            or source.get("scenario") != SCENARIO
            or source.get("completed_training_episodes", 0) < 30
            or source.get("objective_spec", {}).get("dense_b") != 40.0
            or source.get("training_protocol") != v4.PROTOCOL):
        raise RuntimeError("E70 historical base-40 control is not authenticated")
    rows = [_row(record) for record in source["validation_history"]
            if record["checkpoint_episode"] <= 30]
    if [row["checkpoint_episode"] for row in rows] != [6, 12, 18, 24, 30]:
        raise RuntimeError("E70 historical checkpoint coverage changed")
    seeds = list(VALIDATION_SEEDS[SCENARIO])
    if any([run["instance_seed"] for run in row["runs"]] != seeds for row in rows):
        raise RuntimeError("E70 historical validation seeds changed")
    return rows


def contract() -> dict:
    directory = Path(__file__).resolve().parent
    sources = (directory / "protocol.py", directory / "run.py", directory / "run.sh",
               ROOT / "experiments/conditioned_vcg/target_size_dense_stage_a_v4/protocol.py",
               ROOT / "experiments/conditioned_vcg/target_size_dense_stage_a_v4/run.py",
               ROOT / "experiments/conditioned_vcg/target_size_dense_stage_a_v4/acceleration.py",
               ROOT / "experiments/conditioned_vcg/target_size_dense_stage_a_v2/run.py",
               ROOT / "train_vcg_dense_proper.py", ROOT / "vcg_objective_audit.py")
    snapshot = base40_snapshot()
    payload = {
        "protocol": PROTOCOL, "method_version": METHOD_VERSION,
        "scope": "development-only original-architecture reward-base ablation",
        "scenarios": {SCENARIO: SCENARIOS[SCENARIO].public_dict()},
        "model_seeds": [0], "episodes_by_scenario": EPISODES_BY_SCENARIO,
        "training_job_budget": TRAINING_JOB_BUDGET,
        "validation_every": VALIDATION_EVERY,
        "validation_seeds": {SCENARIO: list(VALIDATION_SEEDS[SCENARIO])},
        "checkpoint_every": CHECKPOINT_EVERY,
        "objective": SPEC.to_dict(), "gamma": 0.99,
        "epsilon": {"start": 0.90, "end": 0.05, "decisions": 3000},
        "agent_family": "episodic_viability_graph_hierarchy",
        "instance_rule": "E28._instance: empty start, arrivals at 10*j, Poisson(80) durations",
        "certification_acceleration": "D12 relocation family + E14 timing cache/path cleanup",
        "td_backup_implementation": TD_BACKUP_IMPLEMENTATION,
        "base40_control": {
            "path": str(BASE40_SUMMARY.relative_to(ROOT)),
            "first_30_episode_snapshot_sha256": _digest({"rows": snapshot}),
            "rows": snapshot,
        },
        "source_sha256": {str(path.relative_to(ROOT)): _sha256(path) for path in sources},
        "paper_evidence": False,
    }
    return {**payload, "contract_sha256": _digest(payload)}
