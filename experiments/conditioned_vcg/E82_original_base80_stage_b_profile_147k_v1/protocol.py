"""Frozen identities and paired coordinates for the E80 Stage-B profile."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from experiments.conditioned_vcg.E28_empty_start_scale_baselines_100k import run as e28
from experiments.conditioned_vcg.E80_original_vcg_base80_stage_b_8x8_145k_v1 import (
    protocol as e80,
)

ROOT = e80.ROOT
PROTOCOL = "vcg_conditioned_e82_original_base80_stage_b_profile_147k_v1"
DEFAULT_OUTPUT = ROOT / "results/vcg-conditioned-e82-original-base80-stage-b-profile-147k-v1"
SCENARIO_ID = "empty_8x8_n36"
SCENARIO = e28.SCENARIO_BY_ID[SCENARIO_ID]
MODEL_SEEDS = (0, 1, 2)
LAMBDAS = (0.0, 0.05, 0.10, 0.20)
PANEL_SEEDS = {"pilot": (147_990,), "confirmation": (148_000, 148_001, 148_002)}


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def digest(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def terminal_identity(seed: int, parent: dict) -> dict:
    path = e80.DEFAULT_OUTPUT / "training/handling" / SCENARIO_ID / f"seed-{seed}" / "terminal.pth"
    if not path.is_file():
        raise RuntimeError(f"E82 needs completed E80 seed {seed}: {path}")
    return {
        "model_seed": seed,
        "checkpoint": str(path.relative_to(ROOT)),
        "checkpoint_sha256": sha256(path),
        "base_checkpoint_sha256": parent["stage_a"][seed]["terminal_checkpoint_sha256"],
        "checkpoint_rule": "terminal_round_8_only",
    }


def contract() -> dict:
    parent = e80.contract()
    report_path = e80.DEFAULT_OUTPUT / "e80-report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("status") != "complete" or len(report.get("model_seeds", ())) != 3:
        raise RuntimeError("E82 requires a complete three-seed E80 report")
    terminals = [terminal_identity(seed, parent) for seed in MODEL_SEEDS]
    for terminal, row in zip(terminals, report["model_seeds"]):
        if (row.get("model_seed") != terminal["model_seed"]
                or row.get("terminal_checkpoint_sha256") != terminal["checkpoint_sha256"]):
            raise RuntimeError("E82 E80 report and terminal checkpoints disagree")
    directory = Path(__file__).resolve().parent
    sources = (
        directory / "protocol.py", directory / "run.py", directory / "run.sh",
        ROOT / "benchmark_viability_critic_priority.py",
        ROOT / "experiments/conditioned_vcg/E28_empty_start_scale_baselines_100k/run.py",
        ROOT / "experiments/conditioned_vcg/E28_empty_start_scale_baselines_100k/adapter.py",
        ROOT / "experiments/conditioned_vcg/E14_certification_scalability_95k/reuse.py",
        ROOT / "PSLAP/relocation_family_certification.py",
        ROOT / "vcg_v11_conditioned_handling.py",
        ROOT / "viability_graph_episodic_audit.py",
    )
    payload = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "scope": "development_pilot_then_prespecified_fresh_paired_three_seed_panel",
        "scenario": SCENARIO.public_dict(),
        "model_seeds": list(MODEL_SEEDS),
        "lambdas": list(LAMBDAS),
        "pilot_instance_seeds": list(PANEL_SEEDS["pilot"]),
        "confirmation_instance_seeds": list(PANEL_SEEDS["confirmation"]),
        "pilot_model_seeds": [0],
        "e80_contract_sha256": parent["contract_sha256"],
        "e80_report_sha256": sha256(report_path),
        "terminal_heads": terminals,
        "stage_a": parent["stage_a"],
        "checkpoint_rule": "E78_E70_episode_112_Stage_A_plus_E80_round_8_Stage_B",
        "lambda_zero_rule": "exact_direct_Stage_A_delegation",
        "instance_rule": "E28._instance; empty start; arrivals at 10*j; Poisson(80) durations",
        "runtime": "shared exact SAFE frontier, D12 relocation family, E14 timing cache/path cleanup, same progress guard",
        "primary_outcomes": [
            "strict_completion", "arrival_to_storage_wait", "release_referenced_mae",
            "storage_referenced_mae", "steps_per_delivery", "rehandles_per_100",
        ],
        "failure_rule": "incomplete rows remain in completion denominator; quality conditional on strict completion",
        "reward_comparison": "none; frozen policies are compared on physical outcomes",
        "checkpoint_or_lambda_selection_after_outcomes": False,
        "training_runs": 0,
        "source_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in sources},
    }
    return {**payload, "contract_sha256": digest(payload)}
