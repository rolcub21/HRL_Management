"""Frozen Stage-B exposure and source identities for the E78 8x8 cohort."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from experiments.conditioned_vcg.E78_original_vcg_base80_full_8x8_143k_v1 import (
    protocol as e78,
)

ROOT = e78.ROOT
PROTOCOL = "vcg_conditioned_e80_original_base80_stage_b_8x8_145k_v1"
SCENARIO = e78.SCENARIO
MODEL_SEEDS = (0, 1, 2)
ROUNDS = 8
EPISODES_PER_ROUND = 14
VALIDATION_EPISODES_PER_ROUND = 3
LATE_UPDATE_FRACTION = 0.25
LAMBDA_MAX = 0.20
COLLECTION_BASE = 145_800_000
DEFAULT_OUTPUT = ROOT / "results/vcg-conditioned-e80-original-base80-stage-b-8x8-145k-v1"


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


def source_dir(seed: int) -> Path:
    if seed not in MODEL_SEEDS:
        raise ValueError("model seed must be 0, 1, or 2")
    root = e78.E70_OUTPUT if seed == 0 else e78.DEFAULT_OUTPUT
    return root / "training/operational" / SCENARIO / f"seed-{seed}"


def collection_seed(seed: int, round_index: int, offset: int) -> int:
    if seed not in MODEL_SEEDS or not 0 <= round_index < ROUNDS or not 0 <= offset < EPISODES_PER_ROUND:
        raise ValueError("invalid E80 collection coordinate")
    return COLLECTION_BASE + seed * 100_000 + round_index * 1000 + offset


def source_identity(seed: int) -> dict:
    directory = source_dir(seed)
    summary_path = directory / "training-summary.json"
    contract_path = directory / "training-contract.json"
    checkpoint_path = directory / "latest.pth"
    for path in (summary_path, contract_path, checkpoint_path):
        if not path.is_file():
            raise RuntimeError(f"E80 Stage-A source is missing: {path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    expected_protocol = (e78.e70.PROTOCOL if seed == 0 else e78.PROTOCOL)
    expected_contract_sha = (e78.e70.contract()["contract_sha256"] if seed == 0
                             else e78.contract()["contract_sha256"])
    if (summary.get("status") != "complete"
            or summary.get("completed_training_episodes") != 112
            or summary.get("model_seed") != seed
            or summary.get("scenario") != SCENARIO
            or summary.get("training_protocol") != expected_protocol
            or summary.get("target_contract_sha256") != expected_contract_sha
            or summary.get("objective_spec") != e78.SPEC.to_dict()
            or summary.get("gamma") != 0.99
            or summary.get("test_panel_opened", False)):
        raise RuntimeError(f"E80 seed {seed} is not the completed base-80 Stage A")
    return {
        "model_seed": seed,
        "source_protocol": expected_protocol,
        "source_contract_sha256": expected_contract_sha,
        "source_training_contract_sha256": sha256(contract_path),
        "source_summary_sha256": sha256(summary_path),
        "terminal_checkpoint": str(checkpoint_path.relative_to(ROOT)),
        "terminal_checkpoint_sha256": sha256(checkpoint_path),
        "checkpoint_policy": "episode_112_latest_weights_not_best_development",
    }


def contract() -> dict:
    directory = Path(__file__).resolve().parent
    sources = (
        directory / "protocol.py", directory / "run.py", directory / "run.sh",
        ROOT / "vcg_v11_conditioned_handling.py",
        ROOT / "train_vcg_preference_conditioned.py",
        ROOT / "viability_graph_episodic_audit.py",
        ROOT / "experiments/conditioned_vcg/E14_certification_scalability_95k/reuse.py",
        ROOT / "PSLAP/relocation_family_certification.py",
    )
    payload = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "scope": "three_seed_development_stage_b_no_test_panel",
        "scenario": SCENARIO,
        "objective": e78.SPEC.to_dict(),
        "gamma_op": 0.99,
        "model_seeds": list(MODEL_SEEDS),
        "stage_a": [source_identity(seed) for seed in MODEL_SEEDS],
        "stage_a_cohort_contract_sha256": e78.contract()["contract_sha256"],
        "stage_a_report_sha256": sha256(e78.DEFAULT_OUTPUT / "e78-report.json"),
        "rounds": ROUNDS,
        "episodes_per_round": EPISODES_PER_ROUND,
        "episodes_per_seed": ROUNDS * EPISODES_PER_ROUND,
        "validation_episodes_per_round": VALIDATION_EPISODES_PER_ROUND,
        "lambda_schedule": "stratified_positive_uniform_0_to_0.20_shuffled_per_round",
        "lambda_zero": "exact_delegation_to_frozen_stage_a",
        "collection_base": COLLECTION_BASE,
        "handling_head": {
            "initialization": "random_no_source_cost_head",
            "epochs_per_round": 8,
            "batch_size": 256,
            "learning_rate": 0.0005,
            "update_fraction_rounds_1_to_4": 1.0,
            "update_fraction_rounds_5_to_8": LATE_UPDATE_FRACTION,
            "future_label": "complete_episode_mc_future_physical_rehandles",
        },
        "frozen_stage_a": True,
        "primary_checkpoint_policy": "terminal_round_8_only_no_best_round_selection",
        "test_panel_opened": False,
        "paper_evidence": False,
        "source_sha256": {str(path.relative_to(ROOT)): sha256(path)
                          for path in sources},
    }
    return {**payload, "contract_sha256": digest(payload)}
