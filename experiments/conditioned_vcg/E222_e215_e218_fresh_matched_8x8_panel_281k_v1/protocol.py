"""Locked identities for the E222 fresh matched 8x8 panel."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re

from experiments.conditioned_vcg.E191_fresh_long_stay_crowding_250k_v1 import (
    protocol as e191,
    scenario as e191_scenario,
)
from experiments.conditioned_vcg.E192_e191_matched_longstay_heuristics_251k_v1 import (
    protocol as e192,
)
from projects.temporal_consequence_realization.method_conformance import CHECKLIST

from . import scenario


ROOT = e191.ROOT
PROTOCOL = "vcg_conditioned_e222_e215_e218_fresh_matched_8x8_panel_281k_v1"
GATE_PROTOCOL = PROTOCOL + "_development_gate"
MODEL_SEEDS = (0, 1, 2)
METHODS = ("dynamic_pslap", "ga", "e191")
DYNAMIC_REPLICATES = (0,)
GA_REPLICATES = tuple(range(4))
VCG_REPLICATES = (0,)
E215_OUTPUT = ROOT / "results/vcg-conditioned-e215-e191-long-exploration-drift-pilot-274k-v1"
E218_OUTPUT = ROOT / "results/vcg-conditioned-e218-e215-long-exploration-replication-277k-v1"
DEFAULT_OUTPUT = ROOT / "results/vcg-conditioned-e222-e215-e218-fresh-matched-8x8-panel-281k-v1"
DEFAULT_GATE_OUTPUT = ROOT / "results/vcg-conditioned-e222-e215-e218-fresh-matched-8x8-panel-281k-v1-gate"
REPORT_NAME = "e222-report.json"
TABLE_NAME = "e222-table.md"
PLOT_NAME = "e222-test-panel.png"
KPI_PLOT_NAME = "e222-kpi-dashboard.png"
GATE_NAME = "gate.json"
GATE_SEED = 250_191_200  # Existing heavy development instance, never test data.
BASELINE_ROWS = len(scenario.TEST_SEEDS) * (len(DYNAMIC_REPLICATES) + len(GA_REPLICATES))
VCG_ROWS = len(scenario.TEST_SEEDS) * len(MODEL_SEEDS)
EXPECTED_ROWS = BASELINE_ROWS + VCG_ROWS
METHOD_REFERENCE = "projects/temporal_consequence_realization/METHOD_REFERENCE.md"
CLAIM_LEVEL = "expressibility"
DECLARED_DEVIATIONS = [
    "E222 evaluates the historical conditioned-VCG controller and makes no "
    "canonical law-condition method claim."
]
ORACLE_INPUTS: list[str] = []


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")).hexdigest()


def method_revision() -> str:
    text = (ROOT / METHOD_REFERENCE).read_text(encoding="utf-8")
    match = re.search(r"^\*\*Revision:\*\* `([^`]+)`", text, re.MULTILINE)
    if match is None:
        raise RuntimeError("E222 cannot parse the canonical method revision")
    return match.group(1)


def replicates(method: str) -> tuple[int, ...]:
    if method == "dynamic_pslap":
        return DYNAMIC_REPLICATES
    if method == "ga":
        return GA_REPLICATES
    if method == "e191":
        return VCG_REPLICATES
    raise ValueError(f"unknown E222 method: {method}")


def _source_paths(seed: int) -> tuple[Path, Path, Path]:
    if seed == 0:
        root = E215_OUTPUT
    elif seed in (1, 2):
        root = E218_OUTPUT
    else:
        raise ValueError(f"unknown E222 model seed: {seed}")
    directory = root / "training/operational" / e191.SCENARIO / f"seed-{seed}"
    return root / "contract.json", directory / "training-summary.json", directory / "best.pth"


def checkpoint_identity(seed: int) -> dict:
    contract_path, summary_path, checkpoint_path = _source_paths(seed)
    for path in (contract_path, summary_path, checkpoint_path):
        if not path.is_file():
            raise RuntimeError(f"E222 requires finalized source artifact: {path}")
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    selected = summary.get("best_validation_record", {})
    expected_episode = {0: 300, 1: 280, 2: 220}[seed]
    if (
        summary.get("status") != "complete"
        or int(summary.get("completed_training_episodes", -1)) != 300
        or int(selected.get("checkpoint_episode", -1)) != expected_episode
        or summary.get("best_checkpoint_sha256") != sha256(checkpoint_path)
    ):
        raise RuntimeError(f"E222 source seed {seed} is not its finalized selection")
    return {
        "model_seed": seed,
        "source_protocol": contract["protocol"],
        "source_contract_sha256": contract["contract_sha256"],
        "source_summary": str(summary_path.resolve()),
        "source_summary_sha256": sha256(summary_path),
        "source_checkpoint": str(checkpoint_path.resolve()),
        "source_checkpoint_sha256": sha256(checkpoint_path),
        "selected_episode": expected_episode,
        "selection_score": list(selected["selection_score"]),
        "selection_score_fields": list(selected["selection_score_fields"]),
        "test_panels_opened_before_selection": False,
    }


def checkpoint_identities() -> dict:
    return {str(seed): checkpoint_identity(seed) for seed in MODEL_SEEDS}


def gate_contract() -> dict:
    """Hash the exact live routes before any E222 test instance is created."""

    directory = Path(__file__).resolve().parent
    sources = (
        directory / "protocol.py",
        directory / "scenario.py",
        directory / "run.py",
        ROOT / "experiments/conditioned_vcg/E203_e191_matched_8x8_test_panel_262k_v1/run.py",
        ROOT / "experiments/conditioned_vcg/E192_e191_matched_longstay_heuristics_251k_v1/run.py",
        ROOT / "experiments/conditioned_vcg/E191_fresh_long_stay_crowding_250k_v1/instrumentation.py",
        ROOT / "experiments/conditioned_vcg/E191_fresh_long_stay_crowding_250k_v1/representation.py",
        ROOT / "experiments/conditioned_vcg/E191_fresh_long_stay_crowding_250k_v1/stage.py",
        ROOT / "experiments/conditioned_vcg/E189_e188_roundoff_repair_248k_v1/roundoff.py",
        ROOT / "experiments/conditioned_vcg/E107_queue_context_stage_a_8x8_171k_v1/representation.py",
        ROOT / "experiments/conditioned_vcg/E100_accelerated_release_clock_163k_v1/fast_inputs.py",
        ROOT / "experiments/conditioned_vcg/E91_matched_target_size_benchmark_156k_v1/run.py",
    )
    identities = checkpoint_identities()
    payload = {
        "schema_version": 1,
        "protocol": GATE_PROTOCOL,
        "scope": "development-only live integration gate for E222",
        "development_seed": GATE_SEED,
        "routes": ["e191", "dynamic_pslap", "ga"],
        "checkpoint_sha256": {
            key: value["source_checkpoint_sha256"]
            for key, value in identities.items()
        },
        "source_sha256": {
            str(path.relative_to(ROOT)): sha256(path) for path in sources
        },
        "test_panel_opened": False,
    }
    return {**payload, "contract_sha256": digest(payload)}


def gate_manifest() -> dict:
    path = DEFAULT_GATE_OUTPUT / GATE_NAME
    if not path.is_file():
        raise RuntimeError(
            "E222 requires the development-only live integration gate; run `gate` first"
        )
    value = json.loads(path.read_text(encoding="utf-8"))
    if (
        value.get("status") != "pass"
        or value.get("protocol") != GATE_PROTOCOL
        or int(value.get("development_seed", -1)) != GATE_SEED
        or value.get("test_panel_opened") is not False
        or value.get("gate_contract_sha256")
        != gate_contract()["contract_sha256"]
        or value.get("checkpoint_sha256")
        != {key: item["source_checkpoint_sha256"] for key, item in checkpoint_identities().items()}
    ):
        raise RuntimeError("E222 development gate identity or verdict differs")
    core = dict(value)
    recorded = core.pop("manifest_sha256", None)
    if recorded != digest(core):
        raise RuntimeError("E222 development gate manifest hash is invalid")
    return value


def _conformance_audit(gate: dict) -> dict[str, dict[str, str]]:
    values = {}
    for identifier, _ in CHECKLIST:
        if identifier == "reproducibility.method_revision":
            values[identifier] = {"status": "pass", "evidence": f"E222 pins {method_revision()}."}
        elif identifier == "reproducibility.content_hashes":
            values[identifier] = {"status": "pass", "evidence": "The split, implementations, gate, and all three finalized checkpoints are content-hashed."}
        elif identifier == "reproducibility.oracle_declarations":
            values[identifier] = {"status": "pass", "evidence": "No oracle inputs are used and the historical scope is explicit."}
        elif identifier == "reproducibility.failed_gates_retained":
            values[identifier] = {"status": "pass", "evidence": "Gate and per-episode artifacts are retained, including failed outcomes."}
        elif identifier == "reproducibility.preprotected_integration_gate":
            values[identifier] = {"status": "pass", "evidence": f"Development-only live gate {gate['manifest_sha256']} passed VCG, Dynamic, and GA before panel preparation."}
        else:
            values[identifier] = {"status": "not_applicable", "evidence": "E222 is a historical conditioned-VCG optimization comparison, not a canonical law-condition method claim."}
    return values


def contract() -> dict:
    parent = e191.contract()
    gate = gate_manifest()
    directory = Path(__file__).resolve().parent
    sources = (
        directory / "protocol.py", directory / "scenario.py", directory / "run.py",
        directory / "plot.py", directory / "run.sh", directory / "README.md",
        ROOT / METHOD_REFERENCE,
        ROOT / "projects/temporal_consequence_realization/method_conformance.py",
        ROOT / "experiments/conditioned_vcg/E203_e191_matched_8x8_test_panel_262k_v1/run.py",
        ROOT / "experiments/conditioned_vcg/E203_e191_matched_8x8_test_panel_262k_v1/plot.py",
        ROOT / "experiments/conditioned_vcg/E192_e191_matched_longstay_heuristics_251k_v1/run.py",
        ROOT / "experiments/conditioned_vcg/E191_fresh_long_stay_crowding_250k_v1/instrumentation.py",
        ROOT / "experiments/conditioned_vcg/E191_fresh_long_stay_crowding_250k_v1/representation.py",
        ROOT / "experiments/conditioned_vcg/E191_fresh_long_stay_crowding_250k_v1/stage.py",
        ROOT / "experiments/conditioned_vcg/E189_e188_roundoff_repair_248k_v1/roundoff.py",
        ROOT / "experiments/conditioned_vcg/E107_queue_context_stage_a_8x8_171k_v1/representation.py",
        ROOT / "experiments/conditioned_vcg/E100_accelerated_release_clock_163k_v1/fast_inputs.py",
        ROOT / "experiments/conditioned_vcg/E91_matched_target_size_benchmark_156k_v1/run.py",
    )
    identities = checkpoint_identities()
    payload = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "method_version": "vcg_dense_v1_1_finalized_long_exploration_multiseed",
        "scope": "fresh matched 8x8 paper test panel for finalized E215/E218 selections",
        "source_training_protocols": sorted({item["source_protocol"] for item in identities.values()}),
        "source_checkpoint_identities": deepcopy(identities),
        "matched_baseline_contract_sha256": e192.contract()["contract_sha256"],
        "development_gate": {
            "manifest": str((DEFAULT_GATE_OUTPUT / GATE_NAME).resolve()),
            "manifest_sha256": gate["manifest_sha256"],
            "development_seed": GATE_SEED,
        },
        "scenario": scenario.SCENARIO,
        "scenario_public": deepcopy(e191_scenario.SCENARIO_SPEC.public_dict()),
        "test_seeds": list(scenario.TEST_SEEDS),
        "test_seed_rule": "seed=281222000+workload_index+3*replicate; ten paired-duration replicates per workload",
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
        "baseline_rule": "E192-matched Dynamic PSLAP and rolling GA, same executor/actions/information/objective as the VCG evaluation",
        "vcg_rule": "three finalized development-selected checkpoints, epsilon zero, no learning, unchanged certification and acceleration",
        "objective": deepcopy(parent["objective"]),
        "finite_horizon": deepcopy(parent["finite_horizon"]),
        "completion_rule": "report strict completion before cost; incomplete runs are ineligible for cost rankings",
        "ga_aggregation": "average four nuisance replicates within each test instance",
        "vcg_aggregation": "report every seed and the across-seed mean on the same thirty instances",
        "certified_dynamic_included": False,
        "checkpoint_selection_uses_test_panel": False,
        "training_runs": 0,
        "development_only": False,
        "test_panel_design_locked": True,
        "paper_evidence_when_complete": True,
        "method_reference": METHOD_REFERENCE,
        "method_revision": method_revision(),
        "claim_level": CLAIM_LEVEL,
        "declared_deviations": list(DECLARED_DEVIATIONS),
        "oracle_inputs": list(ORACLE_INPUTS),
        "nonconforming_shortcuts": [],
        "conformance_audit": _conformance_audit(gate),
        "source_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in sources},
    }
    return {**payload, "contract_sha256": digest(payload)}


__all__ = [
    "BASELINE_ROWS", "DEFAULT_GATE_OUTPUT", "DEFAULT_OUTPUT", "DYNAMIC_REPLICATES",
    "EXPECTED_ROWS", "GA_REPLICATES", "GATE_NAME", "GATE_PROTOCOL", "GATE_SEED",
    "KPI_PLOT_NAME", "METHODS", "MODEL_SEEDS", "PLOT_NAME", "PROTOCOL",
    "REPORT_NAME", "ROOT", "TABLE_NAME", "VCG_REPLICATES", "VCG_ROWS",
    "checkpoint_identities", "checkpoint_identity", "contract", "digest",
    "gate_contract", "gate_manifest", "replicates", "sha256",
]
