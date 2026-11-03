"""Locked identities for the E203 matched 8x8 test panel."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

from experiments.conditioned_vcg.E191_fresh_long_stay_crowding_250k_v1 import (
    protocol as e191,
    scenario as e191_scenario,
)
from experiments.conditioned_vcg.E202_e191_multiseed_development_freeze_261k_v1 import (
    run as e202,
)
from experiments.conditioned_vcg.E192_e191_matched_longstay_heuristics_251k_v1 import (
    protocol as e192,
)

from . import scenario


ROOT = e191.ROOT
PROTOCOL = "vcg_conditioned_e203_e191_matched_8x8_test_panel_262k_v1"
MODEL_SEEDS = e191.MODEL_SEEDS
METHODS = ("dynamic_pslap", "ga", "e191")
DYNAMIC_REPLICATES = (0,)
GA_REPLICATES = tuple(range(4))
VCG_REPLICATES = (0,)
SOURCE_OUTPUT = e191.DEFAULT_OUTPUT
FREEZE_OUTPUT = e202.DEFAULT_OUTPUT
DEFAULT_OUTPUT = (
    ROOT / "results/vcg-conditioned-e203-e191-matched-8x8-test-panel-262k-v1"
)
REPORT_NAME = "e203-report.json"
TABLE_NAME = "e203-table.md"
BASELINE_ROWS = len(scenario.TEST_SEEDS) * (
    len(DYNAMIC_REPLICATES) + len(GA_REPLICATES)
)
VCG_ROWS = len(scenario.TEST_SEEDS) * len(MODEL_SEEDS)
EXPECTED_ROWS = BASELINE_ROWS + VCG_ROWS


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def replicates(method: str) -> tuple[int, ...]:
    if method == "dynamic_pslap":
        return DYNAMIC_REPLICATES
    if method == "ga":
        return GA_REPLICATES
    if method == "e191":
        return VCG_REPLICATES
    raise ValueError(f"unknown E203 method: {method}")


def freeze_manifest() -> dict:
    path = FREEZE_OUTPUT / "freeze.json"
    if not path.is_file():
        raise RuntimeError("E203 requires the completed E202 freeze")
    value = json.loads(path.read_text(encoding="utf-8"))
    parent = e191.contract()
    if (
        value.get("status") != "complete"
        or value.get("source_contract_sha256") != parent["contract_sha256"]
        or tuple(value.get("model_seeds", ())) != MODEL_SEEDS
        or value.get("test_panels_opened") is not False
    ):
        raise RuntimeError("E203 E202 freeze identity differs")
    return value


def contract() -> dict:
    parent = e191.contract()
    freeze = freeze_manifest()
    directory = Path(__file__).resolve().parent
    sources = (
        directory / "protocol.py",
        directory / "scenario.py",
        directory / "run.py",
        directory / "plot.py",
        directory / "run.sh",
        directory / "README.md",
        ROOT
        / "experiments/conditioned_vcg/E192_e191_matched_longstay_heuristics_251k_v1/run.py",
        ROOT
        / "experiments/conditioned_vcg/E202_e191_multiseed_development_freeze_261k_v1/run.py",
        ROOT
        / "experiments/conditioned_vcg/E191_fresh_long_stay_crowding_250k_v1/stage.py",
    )
    payload = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "scope": "sealed matched 8x8 paper test panel",
        "source_protocol": e191.PROTOCOL,
        "source_contract_sha256": parent["contract_sha256"],
        "freeze_protocol": e202.PROTOCOL,
        "freeze_manifest_sha256": sha256(FREEZE_OUTPUT / "freeze.json"),
        "matched_baseline_contract_sha256": e192.contract()["contract_sha256"],
        "frozen_checkpoint_sha256": deepcopy(
            freeze["frozen_checkpoint_sha256"]
        ),
        "selected_episodes": deepcopy(freeze["selected_episodes"]),
        "scenario": scenario.SCENARIO,
        "scenario_public": deepcopy(e191_scenario.SCENARIO_SPEC.public_dict()),
        "test_seeds": list(scenario.TEST_SEEDS),
        "test_seed_rule": (
            "seed=262203000+workload_index+3*replicate; ten replicates at "
            "each workload; duration_seed=262203500+replicate"
        ),
        "test_instances": len(scenario.TEST_SEEDS),
        "cases_per_workload": scenario.CASES_PER_LEVEL,
        "workload_levels": list(scenario.LEVELS),
        "duration_streams_paired_across_workloads": True,
        "model_seeds": list(MODEL_SEEDS),
        "methods": list(METHODS),
        "dynamic_replicates": list(DYNAMIC_REPLICATES),
        "ga_replicates": list(GA_REPLICATES),
        "vcg_replicates": list(VCG_REPLICATES),
        "expected_episode_rows": EXPECTED_ROWS,
        "baseline_rule": (
            "E192-matched Dynamic PSLAP and rolling GA with the E25-v4 common "
            "strict scheduler, executed in the E191 long-stay objective environment"
        ),
        "vcg_rule": (
            "three immutable E202 selected E191 checkpoints, epsilon zero, no "
            "learning, unchanged E191 certification authoritative"
        ),
        "objective": deepcopy(parent["objective"]),
        "finite_horizon": deepcopy(parent["finite_horizon"]),
        "completion_rule": (
            "report strict completion before cost; exclude incomplete runs from "
            "cost rankings"
        ),
        "ga_aggregation": (
            "average four nuisance replicates within each test instance"
        ),
        "vcg_aggregation": (
            "report each model seed and the across-seed mean on the same thirty "
            "test instances"
        ),
        "certified_dynamic_included": False,
        "checkpoint_selection_uses_test_panel": False,
        "training_runs": 0,
        "development_only": False,
        "test_panel_design_locked": True,
        "test_panel_opened_at_evaluation": True,
        "paper_evidence_when_complete": True,
        "source_sha256": {
            str(path.relative_to(ROOT)): sha256(path) for path in sources
        },
    }
    return {**payload, "contract_sha256": digest(payload)}


__all__ = [
    "BASELINE_ROWS",
    "DEFAULT_OUTPUT",
    "DYNAMIC_REPLICATES",
    "EXPECTED_ROWS",
    "FREEZE_OUTPUT",
    "GA_REPLICATES",
    "METHODS",
    "MODEL_SEEDS",
    "PROTOCOL",
    "REPORT_NAME",
    "ROOT",
    "SOURCE_OUTPUT",
    "TABLE_NAME",
    "VCG_REPLICATES",
    "VCG_ROWS",
    "contract",
    "digest",
    "freeze_manifest",
    "replicates",
    "sha256",
]

