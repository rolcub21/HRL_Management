"""Frozen identities for the E189 matched heuristic development panel.

The 25 heuristic trajectories are independent of a learned checkpoint.  They
are therefore executed once on E189's five development instances and reused
when later E189 validation checkpoints become available.  The contract binds
the problem definition and evaluator sources, but deliberately does not hash
E189's growing training summary.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

from experiments.conditioned_vcg.E189_e188_roundoff_repair_248k_v1 import (
    protocol as e189,
)


ROOT = e189.ROOT
PROTOCOL = "vcg_conditioned_e190_e189_matched_heuristic_curve_249k_v1"
SCENARIO = e189.SCENARIO
SCENARIOS = e189.SCENARIOS
INSTANCE_SEEDS = tuple(e189.VALIDATION_SEEDS[SCENARIO])
METHODS = ("dynamic_pslap", "ga")
DYNAMIC_REPLICATES = (0,)
GA_REPLICATES = tuple(range(4))
MAX_STEPS = e189.MAX_STEPS
SOURCE_OUTPUT = e189.DEFAULT_OUTPUT
MODEL_SEED = 0
E189_SUMMARY = (
    SOURCE_OUTPUT / "training/operational" / SCENARIO
    / f"seed-{MODEL_SEED}/training-summary.json"
)
DEFAULT_OUTPUT = (
    ROOT
    / "results/vcg-conditioned-e190-e189-matched-heuristic-curve-249k-v1"
)
EXPECTED_ROWS = len(INSTANCE_SEEDS) * (
    len(DYNAMIC_REPLICATES) + len(GA_REPLICATES)
)
REPORT_NAME = "e190-report.json"
TABLE_NAME = "e190-table.md"


def sha256(path: Path) -> str:
    digest_value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest_value.update(chunk)
    return digest_value.hexdigest()


def digest(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def replicates(method: str) -> tuple[int, ...]:
    if method == "dynamic_pslap":
        return DYNAMIC_REPLICATES
    if method == "ga":
        return GA_REPLICATES
    raise ValueError(f"unknown E190 method: {method}")


def contract() -> dict:
    parent = e189.contract()
    expected_objective = {
        "name": "J_storage_clock_plus_full_admission_wait",
        "formula": (
            "2*T_storage + E_storage + 0.5*W_release_to_admission "
            "+ 0.1*L + 10*R"
        ),
        "component_names": [
            "storage_tardiness_accrual",
            "storage_earliness_at_delivery",
            "admission_wait_accrual",
            "equipment_work",
            "physical_rehandles",
        ],
        "weights": [2.0, 1.0, 0.5, 0.1, 10.0],
        "storage_tolerance_steps": 20,
        "storage_due_rule": "stored_time_step + storage_steps_needed",
        "wait_rule": "stored_time_step - arrival_step",
        "wait_grace_steps": 0,
        "primitive_accrual": True,
        "gamma": 1.0,
        "e78_dense_reward_used_for_learning": False,
    }
    if parent.get("objective") != expected_objective:
        raise RuntimeError("E190 source objective no longer matches corrected J")
    if tuple(parent.get("validation_seeds", ())) != INSTANCE_SEEDS:
        raise RuntimeError("E190 source validation panel changed")
    if int(parent["finite_horizon"]["steps"]) != MAX_STEPS:
        raise RuntimeError("E190 source finite horizon changed")
    if parent.get("scenario") != SCENARIO:
        raise RuntimeError("E190 source scenario changed")

    directory = Path(__file__).resolve().parent
    sources = (
        directory / "protocol.py",
        directory / "run.py",
        directory / "run.sh",
        ROOT
        / "experiments/conditioned_vcg/E116_e115_matched_operator_baselines_180k_v1/run.py",
        ROOT
        / "experiments/conditioned_vcg/E91_matched_target_size_benchmark_156k_v1/run.py",
        ROOT
        / "experiments/conditioned_vcg/E28_empty_start_scale_baselines_100k/run.py",
        ROOT
        / "experiments/conditioned_vcg/E28_empty_start_scale_baselines_100k/adapter.py",
        ROOT
        / "experiments/conditioned_vcg/E25_v4_reconfiguration_baselines_95k/controller.py",
        ROOT
        / "experiments/conditioned_vcg/E187_wait_corrected_e78_246k_v1/objective.py",
    )
    payload = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "scope": (
            "development-only matched Dynamic PSLAP and rolling-GA reference "
            "curve for every E189 validation checkpoint"
        ),
        "source_protocol": e189.PROTOCOL,
        "source_contract_sha256": parent["contract_sha256"],
        "scenario": SCENARIO,
        "scenario_public": deepcopy(parent["scenario_public"]),
        "instance_seeds": list(INSTANCE_SEEDS),
        "instance_rule": (
            "exact E189 development EpisodeInstances: empty 8x8 start, 36 "
            "jobs, releases at 10*j, seeded Poisson(80) storage durations"
        ),
        "methods": list(METHODS),
        "dynamic_replicates": list(DYNAMIC_REPLICATES),
        "ga_replicates": list(GA_REPLICATES),
        "expected_episode_rows": EXPECTED_ROWS,
        "baseline_rule": (
            "E28 Dynamic PSLAP and rolling GA with the E25-v4 common strict "
            "retrieval/reconfiguration/defer scheduler, executed directly in "
            "the E189 wait-corrected objective environment"
        ),
        "objective": deepcopy(parent["objective"]),
        "finite_horizon": deepcopy(parent["finite_horizon"]),
        "completion_rule": (
            "report strict completion first; J and quality contrasts are not "
            "ranked for incomplete trajectories"
        ),
        "checkpoint_rule": (
            "run the fixed heuristic panel once; rescan every complete E189 "
            "greedy development validation checkpoint without selecting one"
        ),
        "aggregation": (
            "average four GA replicates within each instance, then average "
            "and pair across the five instances; paired sample size is five"
        ),
        "inference": (
            "development-only descriptive paired differences and nominal "
            "95% t intervals; no confirmatory or sealed-test claim"
        ),
        "certified_dynamic_included": False,
        "certified_dynamic_disposition": (
            "separate slower arm; it does not block this plain heuristic curve"
        ),
        "training_runs": 0,
        "test_panel_opened": False,
        "independent_test_opened": False,
        "paper_evidence": False,
        "source_sha256": {
            str(path.relative_to(ROOT)): sha256(path) for path in sources
        },
    }
    return {**payload, "contract_sha256": digest(payload)}


__all__ = [
    "DEFAULT_OUTPUT",
    "DYNAMIC_REPLICATES",
    "E189_SUMMARY",
    "EXPECTED_ROWS",
    "GA_REPLICATES",
    "INSTANCE_SEEDS",
    "MAX_STEPS",
    "METHODS",
    "MODEL_SEED",
    "PROTOCOL",
    "ROOT",
    "REPORT_NAME",
    "SCENARIO",
    "SCENARIOS",
    "SOURCE_OUTPUT",
    "TABLE_NAME",
    "contract",
    "digest",
    "e189",
    "replicates",
    "sha256",
]

