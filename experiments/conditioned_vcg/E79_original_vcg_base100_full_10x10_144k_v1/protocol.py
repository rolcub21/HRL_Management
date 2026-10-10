"""Frozen three-seed 10x10 reward-scaled original-VCG Stage-A protocol."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from experiments.conditioned_vcg.target_size_dense_stage_a_v4 import protocol as v4
from vcg_objective_audit import TimingObjectiveSpec


ROOT = v4.ROOT
PROTOCOL = "vcg_conditioned_e79_original_base100_full_10x10_144k_v1"
METHOD_VERSION = "vcg_dense_v1_1_target_size_base100_e79"
SCENARIO = "empty_10x10_n64"
SCENARIOS = {SCENARIO: v4.SCENARIOS[SCENARIO]}
MODEL_SEEDS = (0, 1, 2)
TRAINING_JOB_BUDGET = v4.TRAINING_JOB_BUDGET
EPISODES_BY_SCENARIO = {SCENARIO: 63}  # ceil(4000 / 64) = 63.
TOTAL_EPISODES = 0  # The Stage-A-v4 wrapper patches this to 63.
VALIDATION_EVERY = v4.VALIDATION_EVERY
CHECKPOINT_EVERY = v4.CHECKPOINT_EVERY
VALIDATION_SEEDS = {SCENARIO: v4.VALIDATION_SEEDS[SCENARIO]}
TRAIN_BASE = {SCENARIO: v4.TRAIN_BASE[SCENARIO]}
TD_BACKUP_IMPLEMENTATION = v4.TD_BACKUP_IMPLEMENTATION
DELIVERY_BASE_8X8 = 80.0
DELIVERY_BASE = DELIVERY_BASE_8X8 * (10.0 / 8.0)
SPEC = TimingObjectiveSpec.dense(
    dense_b=DELIVERY_BASE, lambda_abs=1.5, lambda_outside=0.5, window=20.0
)
LATE_CHECKPOINTS = (36, 42, 48, 54, 60, 63)
DEFAULT_OUTPUT = ROOT / "results/vcg-conditioned-e79-original-base100-full-10x10-144k-v1"
scenario_for = v4.scenario_for
TargetDenseEnv = v4.TargetDenseEnv


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def digest(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def contract() -> dict:
    directory = Path(__file__).resolve().parent
    sources = (
        directory / "protocol.py", directory / "run.py", directory / "run.sh",
        ROOT / "experiments/conditioned_vcg/target_size_dense_stage_a_v4/protocol.py",
        ROOT / "experiments/conditioned_vcg/target_size_dense_stage_a_v4/run.py",
        ROOT / "experiments/conditioned_vcg/target_size_dense_stage_a_v4/acceleration.py",
        ROOT / "experiments/conditioned_vcg/target_size_dense_stage_a_v3/acceleration.py",
        ROOT / "experiments/conditioned_vcg/target_size_dense_stage_a_v2/protocol.py",
        ROOT / "experiments/conditioned_vcg/target_size_dense_stage_a_v2/run.py",
        ROOT / "experiments/conditioned_vcg/target_size_vcg_training/protocol.py",
        ROOT / "experiments/conditioned_vcg/E28_empty_start_scale_baselines_100k/run.py",
        ROOT / "experiments/conditioned_vcg/E14_certification_scalability_95k/reuse.py",
        ROOT / "train_vcg_dense_proper.py", ROOT / "vcg_objective_audit.py",
        ROOT / "run_vcg_objective_gamma_audit.py",
        ROOT / "viability_graph_hierarchy.py",
        ROOT / "viability_graph_episodic_audit.py",
        ROOT / "PSLAP/yard_graph.py",
    )
    payload = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "method_version": METHOD_VERSION,
        "scope": "full_10x10_stage_a_three_seed_development_replication",
        "architecture": "original VCG-Dense v1.1 Stage A; no Stage B",
        "scenario": SCENARIO,
        "scenario_public": SCENARIOS[SCENARIO].public_dict(),
        "model_seeds": list(MODEL_SEEDS),
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
            "report terminal episode 63 and every late-window validation; "
            "do not select the single best development checkpoint"
        ),
        "objective": SPEC.to_dict(),
        "reward_shape_rule": "8x8 delivery base 80 times yard-width ratio 10/8",
        "gamma": 0.99,
        "epsilon": {"start": 0.90, "end": 0.05, "decisions": 3000},
        "instance_rule": "E28._instance: empty start, arrivals at 10*j, Poisson(80) durations",
        "certification_acceleration": "D12 relocation family + E14 timing cache/path cleanup",
        "td_backup_implementation": TD_BACKUP_IMPLEMENTATION,
        "training_device": "cuda",
        "stage_b_authorized": False,
        "test_panel_opened": False,
        "paper_evidence": False,
        "source_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in sources},
    }
    return {**payload, "contract_sha256": digest(payload)}
