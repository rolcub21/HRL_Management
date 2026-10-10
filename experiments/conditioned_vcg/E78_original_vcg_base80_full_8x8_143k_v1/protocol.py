"""Frozen original-VCG base-80 8x8 replication and stability protocol.

E70's already-running seed 0 remains under its original authenticated contract.
E78 adds fresh model seeds 1 and 2 with the same training recipe and evaluates
all three by terminal and prespecified late-window checkpoints.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from experiments.conditioned_vcg.E70_original_vcg_reward_base_135k_v1 import (
    protocol as e70,
)

ROOT = e70.ROOT
PROTOCOL = "vcg_conditioned_e78_original_base80_full_8x8_143k_v1"
METHOD_VERSION = "vcg_dense_v1_1_target_size_base80_e78"
SCENARIO = e70.SCENARIO
SCENARIOS = e70.SCENARIOS
MODEL_SEEDS = (1, 2)  # Seed 0 resumes under E70's frozen protocol.
AGGREGATE_SEEDS = (0, 1, 2)
EPISODES_BY_SCENARIO = {SCENARIO: 112}
TOTAL_EPISODES = 0  # Stage-A-v4 wrapper patches this to 112.
VALIDATION_EVERY = e70.VALIDATION_EVERY
CHECKPOINT_EVERY = e70.CHECKPOINT_EVERY
VALIDATION_SEEDS = e70.VALIDATION_SEEDS
TRAIN_BASE = e70.TRAIN_BASE
TRAINING_JOB_BUDGET = e70.TRAINING_JOB_BUDGET
TD_BACKUP_IMPLEMENTATION = e70.TD_BACKUP_IMPLEMENTATION
SPEC = e70.SPEC
DEFAULT_OUTPUT = ROOT / "results/vcg-conditioned-e78-original-base80-full-8x8-143k-v1"
E70_OUTPUT = e70.DEFAULT_OUTPUT
EARLY_CHECKPOINTS = (6, 12, 18, 24, 30)
LATE_CHECKPOINTS = (60, 66, 72, 78, 84, 90, 96, 102, 108, 112)
scenario_for = e70.scenario_for
TargetDenseEnv = e70.TargetDenseEnv


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def digest(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _seed0_prefix() -> dict:
    directory = (E70_OUTPUT / "training/operational" / SCENARIO / "seed-0")
    summary_path = directory / "training-summary.json"
    training_contract_path = directory / "training-contract.json"
    e70_contract_path = E70_OUTPUT / "contract.json"
    for path in (summary_path, training_contract_path, e70_contract_path):
        if not path.is_file():
            raise RuntimeError(f"E78 needs existing E70 seed-0 evidence: {path}")
    e70_contract = json.loads(e70_contract_path.read_text(encoding="utf-8"))
    if e70_contract != e70.contract():
        raise RuntimeError("E70 frozen contract or source changed")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if (summary.get("model_seed") != 0
            or summary.get("scenario") != SCENARIO
            or summary.get("training_protocol") != e70.PROTOCOL
            or summary.get("target_contract_sha256")
            != e70_contract["contract_sha256"]
            or summary.get("objective_spec") != SPEC.to_dict()
            or int(summary.get("completed_training_episodes", 0)) < 30):
        raise RuntimeError("E70 seed-0 prefix does not match base-80 training")
    rows = [e70._row(record) for record in summary["validation_history"]
            if int(record["checkpoint_episode"]) <= 30]
    if tuple(row["checkpoint_episode"] for row in rows) != EARLY_CHECKPOINTS:
        raise RuntimeError("E70 seed-0 first-30 validation coverage changed")
    return {
        "summary_path": str(summary_path.relative_to(ROOT)),
        "training_contract_sha256": sha256(training_contract_path),
        "e70_contract_sha256": e70_contract["contract_sha256"],
        "first_30_validation_sha256": digest(rows),
        "completed_at_preparation_at_least": 30,
    }


def contract() -> dict:
    directory = Path(__file__).resolve().parent
    sources = (
        directory / "protocol.py", directory / "run.py", directory / "run.sh",
        ROOT / "experiments/conditioned_vcg/E70_original_vcg_reward_base_135k_v1/protocol.py",
        ROOT / "experiments/conditioned_vcg/E70_original_vcg_reward_base_135k_v1/run.py",
        ROOT / "experiments/conditioned_vcg/target_size_dense_stage_a_v4/protocol.py",
        ROOT / "experiments/conditioned_vcg/target_size_dense_stage_a_v4/run.py",
        ROOT / "experiments/conditioned_vcg/target_size_dense_stage_a_v4/acceleration.py",
        ROOT / "experiments/conditioned_vcg/target_size_dense_stage_a_v2/run.py",
        ROOT / "train_vcg_dense_proper.py", ROOT / "vcg_objective_audit.py",
        ROOT / "run_vcg_objective_gamma_audit.py",
    )
    payload = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "method_version": METHOD_VERSION,
        "scope": "full_8x8_stage_a_three_seed_development_replication",
        "architecture": "original VCG-Dense v1.1 Stage A; no Stage B",
        "scenario": SCENARIO,
        "scenario_public": SCENARIOS[SCENARIO].public_dict(),
        "aggregate_model_seeds": list(AGGREGATE_SEEDS),
        "e78_new_model_seeds": list(MODEL_SEEDS),
        "seed0_source": _seed0_prefix(),
        "episodes_per_seed": EPISODES_BY_SCENARIO[SCENARIO],
        "training_job_budget_per_seed": TRAINING_JOB_BUDGET,
        "realized_training_jobs_per_seed": (
            EPISODES_BY_SCENARIO[SCENARIO] * SCENARIOS[SCENARIO].total_jobs
        ),
        "validation_every": VALIDATION_EVERY,
        "validation_seeds": list(VALIDATION_SEEDS[SCENARIO]),
        "checkpoint_every": CHECKPOINT_EVERY,
        "late_window_checkpoints": list(LATE_CHECKPOINTS),
        "primary_checkpoint_policy": (
            "report terminal episode 112 and every late-window validation; "
            "do not select the single best development checkpoint"
        ),
        "objective": SPEC.to_dict(),
        "gamma": 0.99,
        "epsilon": {"start": 0.90, "end": 0.05, "decisions": 3000},
        "instance_rule": "E28._instance: empty start, arrivals at 10*j, Poisson(80) durations",
        "certification_acceleration": "D12 relocation family + E14 timing cache/path cleanup",
        "td_backup_implementation": TD_BACKUP_IMPLEMENTATION,
        "new_seed_training_device": "cuda",
        "stage_b_authorized": False,
        "test_panel_opened": False,
        "paper_evidence": False,
        "source_sha256": {str(path.relative_to(ROOT)): sha256(path)
                          for path in sources},
    }
    return {**payload, "contract_sha256": digest(payload)}
