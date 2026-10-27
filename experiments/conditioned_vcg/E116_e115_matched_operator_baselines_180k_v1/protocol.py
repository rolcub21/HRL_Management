"""Frozen protocol for the E115 matched Dynamic-PSLAP/GA comparison."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from experiments.conditioned_vcg.E115_release_et_cost_vcg_8x8_179k_v1 import (
    protocol as e115,
)
from experiments.conditioned_vcg.E91_matched_target_size_benchmark_156k_v1 import (
    protocol as e91,
)


ROOT = e115.ROOT
PROTOCOL = "vcg_conditioned_e116_e115_matched_operator_baselines_180k_v1"
DEFAULT_OUTPUT = (
    ROOT / "results/vcg-conditioned-e116-e115-matched-operator-baselines-180k-v1"
)
E115_OUTPUT = e115.DEFAULT_OUTPUT
SCENARIO_ID = e115.SCENARIO
INSTANCE_SEEDS = tuple(e115.e107.e105.e78.VALIDATION_SEEDS[SCENARIO_ID])
METHODS = ("dynamic_pslap", "ga")
GA_REPLICATES = e91.GA_REPLICATES


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def digest(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def contract() -> dict:
    expected_parent = e115.contract()
    parent_path = E115_OUTPUT / "contract.json"
    if not parent_path.is_file():
        raise RuntimeError("E116 requires the prepared E115 repair2 contract")
    observed_parent = json.loads(parent_path.read_text(encoding="utf-8"))
    if observed_parent != expected_parent:
        raise RuntimeError("E116 E115 source contract is missing or changed")
    e115_scenario = e115.e107.e105.e78.SCENARIOS[SCENARIO_ID]
    if e115_scenario.public_dict() != e91.SCENARIO.public_dict():
        raise RuntimeError("E116 baseline and E115 yard definitions differ")

    directory = Path(__file__).resolve().parent
    sources = (
        directory / "protocol.py",
        directory / "run.py",
        directory / "run.sh",
        ROOT / "experiments/conditioned_vcg/E91_matched_target_size_benchmark_156k_v1/run.py",
        ROOT / "experiments/conditioned_vcg/E28_empty_start_scale_baselines_100k/run.py",
        ROOT / "experiments/conditioned_vcg/E28_empty_start_scale_baselines_100k/adapter.py",
        ROOT / "experiments/conditioned_vcg/E25_v4_reconfiguration_baselines_95k/controller.py",
    )
    payload = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "scope": "development-only matched E115 baseline comparison",
        "scenario": e115_scenario.public_dict(),
        "instance_seeds": list(INSTANCE_SEEDS),
        "instance_rule": (
            "exact E115 validation EpisodeInstances: empty start, releases 30*j, "
            "frozen Poisson(80) durations"
        ),
        "methods": list(METHODS),
        "ga_replicates": list(GA_REPLICATES),
        "dynamic_replicates": [0],
        "e115_contract_sha256": expected_parent["contract_sha256"],
        "e115_validation_rule": (
            "report every available validation checkpoint; do not select a best "
            "checkpoint from the reused development panel"
        ),
        "baseline_rule": (
            "E28 empty-start Dynamic PSLAP and GA with the frozen E25-v4 common "
            "retrieval/reconfiguration/wait scheduler"
        ),
        "ga_aggregation": (
            "average four stochastic replicates within each instance before "
            "averaging the five paired instances"
        ),
        "objective": expected_parent["objective"],
        "completion_rule": (
            "strict completion is reported for every run; J is not treated as "
            "comparable for an incomplete trajectory"
        ),
        "inference": (
            "development-only descriptive paired differences and nominal t "
            "intervals over five instances; no superiority or paper claim"
        ),
        "training_runs": 0,
        "independent_test_opened": False,
        "paper_evidence": False,
        "source_sha256": {
            str(path.relative_to(ROOT)): sha256(path) for path in sources
        },
    }
    return {**payload, "contract_sha256": digest(payload)}


__all__ = [
    "DEFAULT_OUTPUT",
    "E115_OUTPUT",
    "GA_REPLICATES",
    "INSTANCE_SEEDS",
    "METHODS",
    "PROTOCOL",
    "ROOT",
    "SCENARIO_ID",
    "contract",
    "digest",
    "sha256",
]

