"""Frozen 8x8 test coordinates and source identities for E91.

10x10 remains a separate, fail-closed cohort: E79 Stage A alone does not
provide the trained Stage B heads required for the lambda sweep.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from experiments.conditioned_vcg.E82_original_base80_stage_b_profile_147k_v1 import (
    protocol as e82,
)
from experiments.conditioned_vcg.E28_empty_start_scale_baselines_100k import run as e28
from experiments.conditioned_vcg.target_size_vcg_training import protocol as target


ROOT = e82.ROOT
PROTOCOL = "vcg_conditioned_e91_matched_target_size_benchmark_156k_v1"
DEFAULT_OUTPUT = ROOT / "results/vcg-conditioned-e91-matched-target-size-156k-v1"
SCENARIO_ID = "empty_8x8_n36"
SCENARIO = e28.SCENARIO_BY_ID[SCENARIO_ID]
MODEL_SEEDS = e82.MODEL_SEEDS
LAMBDAS = e82.LAMBDAS
METHODS = ("vcg", "dynamic_pslap", "ga")
GA_REPLICATES = tuple(range(4))  # E25's frozen stochastic replicate count.
PANEL_SEEDS = {
    "pilot": (156_990,),
    "test": tuple(range(156_000, 156_030)),
}


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


def check_seed_separation() -> None:
    pilot = set(PANEL_SEEDS["pilot"])
    test = set(PANEL_SEEDS["test"])
    assert len(test) == 30 and not (pilot & test)
    known = set(target.VALIDATION_SEEDS[SCENARIO_ID])
    known.update(target.PILOT_SEEDS[SCENARIO_ID])
    known.update(target.TEST_SEEDS[SCENARIO_ID])
    known.update(e82.PANEL_SEEDS["pilot"])
    known.update(e82.PANEL_SEEDS["confirmation"])
    for model_seed in MODEL_SEEDS:
        # The older target-size helper permits only seeds 0/1, whereas the
        # E78 cohort explicitly added seed 2 under the same stride.
        base = target.TRAIN_BASE[SCENARIO_ID] + model_seed * 1_000_000
        known.update(range(base, base + 500))
        for round_index in range(e82.e80.ROUNDS):
            for offset in range(e82.e80.EPISODES_PER_ROUND):
                known.add(e82.e80.collection_seed(model_seed, round_index, offset))
    assert not (pilot & known) and not (test & known)


def contract() -> dict:
    check_seed_separation()
    parent = e82.contract()
    saved_parent = e82.DEFAULT_OUTPUT / "contract.json"
    if not saved_parent.is_file() or json.loads(saved_parent.read_text()) != parent:
        raise RuntimeError("E91 requires the authenticated, completed E82 source")
    directory = Path(__file__).resolve().parent
    sources = (
        directory / "protocol.py", directory / "run.py", directory / "run.sh",
        ROOT / "experiments/conditioned_vcg/E28_empty_start_scale_baselines_100k/run.py",
        ROOT / "experiments/conditioned_vcg/E28_empty_start_scale_baselines_100k/adapter.py",
        ROOT / "experiments/conditioned_vcg/E25_scale_baseline_transfer_95k/run.py",
        ROOT / "experiments/conditioned_vcg/E25_v4_reconfiguration_baselines_95k/run.py",
        ROOT / "benchmark_viability_critic_priority.py",
    )
    payload = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "scope": "fresh_matched_8x8_test_no_training_or_checkpoint_selection",
        "scenario": SCENARIO.public_dict(),
        "instance_rule": "E28._instance; empty start; 10-step arrivals; Poisson(80) durations",
        "pilot_seeds": list(PANEL_SEEDS["pilot"]),
        "test_seeds": list(PANEL_SEEDS["test"]),
        "model_seeds": list(MODEL_SEEDS),
        "lambdas": list(LAMBDAS),
        "methods": list(METHODS),
        "ga_replicates": list(GA_REPLICATES),
        "a3c_disposition": "excluded: no target-size-trained 8x8 A3C cohort",
        "vcg_source_contract_sha256": parent["contract_sha256"],
        "vcg_checkpoint_rule": "E78/E70 episode-112 Stage A and E80 round-8 Stage B",
        "baseline_rule": "E28 empty-start GA and Dynamic PSLAP with E25-v4 common scheduler",
        "baseline_wall_cap_seconds": e28.WALL_SECONDS,
        "aggregation": "replicates/model seeds first; 30 equal-weight paired instances second",
        "quality_failure_rule": "incomplete runs count against strict completion; no primary quality inference unless all arms fully complete",
        "inference": "paired t intervals; two-sided p; Holm across 4 lambdas x 2 baselines x 6 quality metrics",
        "primary_clock": "release_referenced_mae; storage_referenced_mae reported separately",
        "ten_by_ten": "not runnable until three matched 10x10 Stage A and Stage B checkpoints exist",
        "source_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in sources},
    }
    return {**payload, "contract_sha256": digest(payload)}
