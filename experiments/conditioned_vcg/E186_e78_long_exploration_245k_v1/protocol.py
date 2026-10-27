"""Frozen protocol for a fresh, long-exploration replication of E78."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from experiments.conditioned_vcg.E70_original_vcg_reward_base_135k_v1 import (
    protocol as e70,
)


ROOT = e70.ROOT
PROTOCOL = "vcg_conditioned_e186_e78_long_exploration_245k_v1"
METHOD_VERSION = "vcg_dense_v1_1_base80_e186_long_exploration"
SCENARIO = e70.SCENARIO
SCENARIOS = e70.SCENARIOS
MODEL_SEEDS = (0, 1, 2)
TOTAL_EPISODES = 300
EPISODES_BY_SCENARIO = {SCENARIO: TOTAL_EPISODES}
# Evaluation does not update the model. Every 20 episodes keeps the long run
# practical and gives exact reads at the requested 200/220 landmarks.
VALIDATION_EVERY = 20
CHECKPOINT_EVERY = 1
VALIDATION_SEEDS = e70.VALIDATION_SEEDS
TRAIN_BASE = e70.TRAIN_BASE
TRAINING_JOB_BUDGET = TOTAL_EPISODES * SCENARIOS[SCENARIO].total_jobs
TD_BACKUP_IMPLEMENTATION = e70.TD_BACKUP_IMPLEMENTATION
SPEC = e70.SPEC
DEFAULT_OUTPUT = ROOT / "results/vcg-conditioned-e186-e78-long-exploration-245k-v1"
EPSILON_START = 0.90
EPSILON_END = 0.05
EPSILON_WARMUP_DECISIONS = 0
# E78 used 28,336 decisions in 336 seed-episodes: 84.333/episode.
# 18,500 / 84.333 = episode 219.37 at the pooled E78 rate.
EPSILON_DECAY_DECISIONS = 18_500
EPSILON_TARGET_FLOOR_EPISODE = 220
REPORT_CHECKPOINTS = tuple(range(180, TOTAL_EPISODES + 1, VALIDATION_EVERY))
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


def e78_decision_evidence() -> dict:
    paths = (
        ROOT / "results/vcg-conditioned-e70-original-vcg-reward-base-135k-v1/training/operational/empty_8x8_n36/seed-0/training-summary.json",
        ROOT / "results/vcg-conditioned-e78-original-base80-full-8x8-143k-v1/training/operational/empty_8x8_n36/seed-1/training-summary.json",
        ROOT / "results/vcg-conditioned-e78-original-base80-full-8x8-143k-v1/training/operational/empty_8x8_n36/seed-2/training-summary.json",
    )
    rows = []
    for seed, path in enumerate(paths):
        if not path.is_file():
            raise RuntimeError(f"E186 needs E78 decision evidence: {path}")
        value = json.loads(path.read_text(encoding="utf-8"))
        runs = value.get("training", {}).get("runs", ())
        if (value.get("model_seed") != seed
                or value.get("completed_training_episodes") != 112
                or len(runs) != 112):
            raise RuntimeError(f"E186 E78 seed-{seed} evidence has changed identity")
        decisions = sum(int(item["macro_decisions"]) for item in runs)
        rows.append({
            "model_seed": seed,
            "episodes": 112,
            "macro_decisions": decisions,
            "summary_path": str(path.relative_to(ROOT)),
            "summary_sha256": sha256(path),
        })
    total_episodes = sum(item["episodes"] for item in rows)
    total_decisions = sum(item["macro_decisions"] for item in rows)
    mean_decisions = total_decisions / total_episodes
    return {
        "seeds": rows,
        "total_seed_episodes": total_episodes,
        "total_macro_decisions": total_decisions,
        "mean_decisions_per_episode": mean_decisions,
        "target_floor_episode": EPSILON_TARGET_FLOOR_EPISODE,
        "declared_decay_decisions": EPSILON_DECAY_DECISIONS,
        "implied_floor_episode_at_e78_mean": EPSILON_DECAY_DECISIONS / mean_decisions,
    }


def contract() -> dict:
    directory = Path(__file__).resolve().parent
    sources = (
        directory / "protocol.py",
        directory / "run.py",
        directory / "run.sh",
        ROOT / "experiments/conditioned_vcg/target_size_dense_stage_a_v4/acceleration.py",
        ROOT / "experiments/conditioned_vcg/target_size_dense_stage_a_v2/run.py",
        ROOT / "experiments/conditioned_vcg/E70_original_vcg_reward_base_135k_v1/protocol.py",
        ROOT / "train_vcg_dense_proper.py",
        ROOT / "run_vcg_objective_gamma_audit.py",
        ROOT / "viability_graph_hierarchy.py",
        ROOT / "viability_graph_episodic_audit.py",
        ROOT / "PSLAP/yard_graph.py",
    )
    payload = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "method_version": METHOD_VERSION,
        "scope": "fresh three-seed 300-episode E78 replication with longer exploration",
        "architecture": "original VCG-Dense v1.1 Stage A; no Stage B",
        "fresh_start": True,
        "warm_start_checkpoint": None,
        "scenario": SCENARIO,
        "scenario_public": SCENARIOS[SCENARIO].public_dict(),
        "model_seeds": list(MODEL_SEEDS),
        "episodes_per_seed": TOTAL_EPISODES,
        "training_jobs_per_seed": TRAINING_JOB_BUDGET,
        "validation_every": VALIDATION_EVERY,
        "validation_seeds": list(VALIDATION_SEEDS[SCENARIO]),
        "validation_change_from_e78": "20 instead of 6 episodes; evaluation-only runtime reduction",
        "checkpoint_every": CHECKPOINT_EVERY,
        "report_checkpoints": list(REPORT_CHECKPOINTS),
        "objective": SPEC.to_dict(),
        "gamma": 0.99,
        "epsilon": {
            "start": EPSILON_START,
            "end": EPSILON_END,
            "warmup_decisions": EPSILON_WARMUP_DECISIONS,
            "decay_decisions": EPSILON_DECAY_DECISIONS,
            "target_floor_episode_approximate": EPSILON_TARGET_FLOOR_EPISODE,
            "derivation": e78_decision_evidence(),
        },
        "instance_rule": "E28._instance: empty start, arrivals at 10*j, Poisson(80) durations",
        "certification_acceleration": "D12 relocation family + E14 timing cache/path cleanup",
        "td_backup_implementation": TD_BACKUP_IMPLEMENTATION,
        "training_budget_rule": "fixed 300 episodes per seed (10,800 jobs)",
        "checkpoint_policy": "resumable checkpoint every completed episode",
        "checkpoint_selection": "report terminal and every prespecified late checkpoint; best-development is descriptive only",
        "test_panel_opened": False,
        "paper_evidence": False,
        "source_sha256": {
            str(path.relative_to(ROOT)): sha256(path) for path in sources
        },
    }
    return {**payload, "contract_sha256": digest(payload)}
