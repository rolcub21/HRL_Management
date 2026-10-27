"""Frozen identities for the E191 matched long-stay heuristic panel."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

from experiments.conditioned_vcg.E191_fresh_long_stay_crowding_250k_v1 import (
    protocol as e191,
    scenario as workload,
)


ROOT = e191.ROOT
PROTOCOL = "vcg_conditioned_e192_e191_matched_longstay_heuristics_251k_v1"
SCENARIO = e191.SCENARIO
SCENARIO_SPEC = e191.SCENARIOS[SCENARIO]
INSTANCE_SEEDS = tuple(e191.VALIDATION_SEEDS[SCENARIO])
METHODS = ("dynamic_pslap", "ga")
DYNAMIC_REPLICATES = (0,)
GA_REPLICATES = tuple(range(4))
MAX_STEPS = e191.MAX_STEPS
SOURCE_OUTPUT = e191.DEFAULT_OUTPUT
MODEL_SEED = 0
E191_SUMMARY = (
    SOURCE_OUTPUT / "training/operational" / SCENARIO
    / f"seed-{MODEL_SEED}/training-summary.json"
)
DEFAULT_OUTPUT = (
    ROOT
    / "results/vcg-conditioned-e192-e191-matched-longstay-heuristics-251k-v1"
)
EXPECTED_ROWS = len(INSTANCE_SEEDS) * (
    len(DYNAMIC_REPLICATES) + len(GA_REPLICATES)
)
REPORT_NAME = "e192-report.json"
TABLE_NAME = "e192-table.md"


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
    raise ValueError(f"unknown E192 method: {method}")


def contract() -> dict:
    parent = e191.contract()
    if tuple(parent.get("validation_seeds", ())) != INSTANCE_SEEDS:
        raise RuntimeError("E192 source validation panel changed")
    if int(parent["finite_horizon"]["steps"]) != MAX_STEPS:
        raise RuntimeError("E192 source horizon changed")
    if parent.get("scenario") != SCENARIO:
        raise RuntimeError("E192 source scenario changed")

    directory = Path(__file__).resolve().parent
    sources = (
        directory / "protocol.py",
        directory / "run.py",
        directory / "run.sh",
        ROOT
        / "experiments/conditioned_vcg/E191_fresh_long_stay_crowding_250k_v1/scenario.py",
        ROOT
        / "experiments/conditioned_vcg/E187_wait_corrected_e78_246k_v1/objective.py",
        ROOT
        / "experiments/conditioned_vcg/E91_matched_target_size_benchmark_156k_v1/run.py",
        ROOT
        / "experiments/conditioned_vcg/E28_empty_start_scale_baselines_100k/run.py",
        ROOT
        / "experiments/conditioned_vcg/E28_empty_start_scale_baselines_100k/adapter.py",
        ROOT
        / "experiments/conditioned_vcg/E25_v4_reconfiguration_baselines_95k/controller.py",
    )
    payload = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "scope": (
            "development-only matched Dynamic PSLAP and rolling-GA reference "
            "curve for E191's long-stay workload family"
        ),
        "source_protocol": e191.PROTOCOL,
        "source_contract_sha256": parent["contract_sha256"],
        "scenario": SCENARIO,
        "scenario_public": deepcopy(parent["scenario_public"]),
        "instance_seeds": list(INSTANCE_SEEDS),
        "workload_by_seed": {
            str(seed): workload.workload_metadata(
                e191.TargetDenseEnv(e191.SCENARIO, e191.SPEC)
                .sample_episode_instance(seed)
            )["target_label"]
            for seed in INSTANCE_SEEDS
        },
        "instance_rule": (
            "exact E191 development EpisodeInstances: empty 8x8 start, 36 "
            "jobs, paired Poisson(700) durations, release gaps 100/40/25 "
            "for nominal 20%/50%/80% expected occupancy"
        ),
        "methods": list(METHODS),
        "dynamic_replicates": list(DYNAMIC_REPLICATES),
        "ga_replicates": list(GA_REPLICATES),
        "expected_episode_rows": EXPECTED_ROWS,
        "baseline_rule": (
            "E28 Dynamic PSLAP and rolling GA with the E25-v4 common physical "
            "retrieval/reconfiguration/defer scheduler, executed directly in "
            "the E191 long-stay wait-corrected objective environment"
        ),
        "objective": deepcopy(parent["objective"]),
        "finite_horizon": deepcopy(parent["finite_horizon"]),
        "runtime_override": (
            "the inherited baseline parser's historical arrival-rate and "
            "duration defaults are ignored; the authenticated E191 "
            "EpisodeInstance and 9000-step scenario are authoritative"
        ),
        "completion_rule": (
            "report strict completion first; J and quality contrasts are not "
            "ranked for incomplete trajectories"
        ),
        "checkpoint_rule": (
            "run the fixed heuristic panel once; rescan every complete E191 "
            "greedy development checkpoint without selecting one"
        ),
        "aggregation": (
            "average four GA replicates within each instance, then average "
            "and pair across nine instances; also report three instances per workload"
        ),
        "inference": (
            "development-only descriptive paired differences and nominal "
            "95% t intervals; no confirmatory or sealed-test claim"
        ),
        "reported_operational_metrics": [
            "achieved mean/steady/peak storage occupancy",
            "external queue",
            "transporter utilization",
            "J and every objective component",
        ],
        "certified_dynamic_included": False,
        "certified_dynamic_disposition": (
            "separate slower arm; it does not block this plain heuristic panel"
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
    "DEFAULT_OUTPUT", "DYNAMIC_REPLICATES", "E191_SUMMARY", "EXPECTED_ROWS",
    "GA_REPLICATES", "INSTANCE_SEEDS", "MAX_STEPS", "METHODS", "MODEL_SEED",
    "PROTOCOL", "REPORT_NAME", "ROOT", "SCENARIO", "SCENARIO_SPEC",
    "SOURCE_OUTPUT", "TABLE_NAME", "contract", "digest", "e191", "replicates",
    "sha256", "workload",
]
