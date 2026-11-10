"""Locked protocol for the corrected-controller prefilled certification curve."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

import torch

from experiments.conditioned_vcg.E195_e191_method_5x5_longstay_254k_v1 import protocol as e195
from experiments.conditioned_vcg.E243_e195_expanded_prefilled_curve_302k_v1 import protocol as original
from experiments.conditioned_vcg.E243_e195_expanded_prefilled_curve_302k_v1 import scenario
from experiments.conditioned_vcg.E255_e195_temperature_scaled_5x5_314k_v1 import protocol as e255
from projects.temporal_consequence_realization.method_conformance import CHECKLIST, validate_protocol


ROOT = e195.ROOT
PROTOCOL = "vcg_conditioned_e263_e255_prefilled_certification_curve_5x5_322k_v1"
GATE_PROTOCOL = PROTOCOL + "_development_gate"
METHOD_REFERENCE = "projects/temporal_consequence_realization/METHOD_REFERENCE.md"
MODEL_SEED = 0
SELECTED_EPISODE = 280
SOURCE_TRAINING = e255.DEFAULT_OUTPUT / f"training/operational/{e195.SCENARIO}/seed-0"
SOURCE_MANIFEST = SOURCE_TRAINING / "training-summary.json"
SOURCE_CHECKPOINT = SOURCE_TRAINING / "best.pth"
ARMS = tuple(original.ARMS)
CERTIFIED_ARMS = tuple(original.CERTIFIED_ARMS)
UNFILTERED_ARMS = tuple(original.UNFILTERED_ARMS)
DYNAMIC_ARMS = tuple(original.DYNAMIC_ARMS)
DEFAULT_OUTPUT = ROOT / "results/vcg-conditioned-e263-e255-prefilled-certification-curve-5x5-322k-v1"
DEFAULT_GATE_OUTPUT = Path(str(DEFAULT_OUTPUT) + "-gate")
GATE_NAME = "gate.json"
REPORT_NAME = "REPORT.txt"
REPORT_JSON = "e263-report.json"
TABLE_NAME = "e263-table.md"
EXPECTED_ROWS = len(ARMS) * len(scenario.SEEDS)


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


def _read(path: Path) -> dict:
    if not Path(path).is_file():
        raise RuntimeError(f"missing E263 source artifact: {path}")
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"invalid E263 source artifact: {path}")
    return value


def method_revision() -> str:
    match = re.search(
        r"^\*\*Revision:\*\* `([^`]+)`",
        (ROOT / METHOD_REFERENCE).read_text(encoding="utf-8"), re.MULTILINE,
    )
    if match is None:
        raise RuntimeError("E263 cannot parse the current method revision")
    return match.group(1)


def checkpoint_identity() -> dict:
    summary = _read(SOURCE_MANIFEST)
    best = summary.get("best_validation_record", {})
    expected = {
        "completed_training_episodes": 300,
        "selection_finalized_after_total_episodes": True,
    }
    mismatch = {key: (summary.get(key), value) for key, value in expected.items() if summary.get(key) != value}
    if mismatch or int(best.get("checkpoint_episode", -1)) != SELECTED_EPISODE:
        raise RuntimeError(f"E263 corrected seed-0 selection differs: {mismatch}")
    if Path(summary.get("best_checkpoint", "")).resolve() != SOURCE_CHECKPOINT.resolve():
        raise RuntimeError("E263 corrected seed-0 selected path differs")
    checkpoint_sha = sha256(SOURCE_CHECKPOINT)
    if checkpoint_sha != summary.get("best_checkpoint_sha256"):
        raise RuntimeError("E263 corrected seed-0 checkpoint hash differs")
    payload = torch.load(SOURCE_CHECKPOINT, map_location="cpu", weights_only=False, mmap=True)
    payload_expected = {
        "training_protocol": e255.PROTOCOL,
        "model_seed": MODEL_SEED,
        "selected_checkpoint_episode": SELECTED_EPISODE,
        "completed_training_episodes": 300,
    }
    payload_mismatch = {key: (payload.get(key), value) for key, value in payload_expected.items() if payload.get(key) != value}
    if payload_mismatch:
        raise RuntimeError(f"E263 corrected seed-0 metadata differs: {payload_mismatch}")
    return {
        "status": "ready",
        "source_protocol": e255.PROTOCOL,
        "source_checkpoint": str(SOURCE_CHECKPOINT.resolve()),
        "source_checkpoint_sha256": checkpoint_sha,
        "source_manifest": str(SOURCE_MANIFEST.resolve()),
        "source_manifest_sha256": sha256(SOURCE_MANIFEST),
        "selected_episode": SELECTED_EPISODE,
        "model_seed": MODEL_SEED,
        "development_only": True,
        "test_panels_opened": True,
    }


def dynamic_ranker_sha256() -> str:
    return original.dynamic_ranker_sha256()


def _source_paths() -> tuple[Path, ...]:
    directory = Path(__file__).resolve().parent
    return (
        directory / "__init__.py", directory / "protocol.py", directory / "run.py",
        directory / "run.sh", directory / "README.md", ROOT / METHOD_REFERENCE,
        ROOT / "projects/temporal_consequence_realization/method_conformance.py",
        ROOT / "experiments/conditioned_vcg/E243_e195_expanded_prefilled_curve_302k_v1/run.py",
        ROOT / "experiments/conditioned_vcg/E243_e195_expanded_prefilled_curve_302k_v1/scenario.py",
        ROOT / "experiments/conditioned_vcg/E243_e195_expanded_prefilled_curve_302k_v1/tracker.py",
        ROOT / "experiments/conditioned_vcg/E198_e195_shadow_certification_ablation_257k_v1/frontier.py",
        ROOT / "experiments/conditioned_vcg/E195_e191_method_5x5_longstay_254k_v1/instrumentation.py",
        ROOT / "experiments/conditioned_vcg/E195_e191_method_5x5_longstay_254k_v1/representation.py",
    )


def gate_contract() -> dict:
    checkpoint = checkpoint_identity()
    payload = {
        "schema_version": 1,
        "protocol": GATE_PROTOCOL,
        "scope": "development live integration gate for four matched E263 routes",
        "development_seed": scenario.SEEDS[0],
        "routes": list(ARMS),
        "checkpoint_sha256": checkpoint["source_checkpoint_sha256"],
        "dynamic_ranker_sha256": dynamic_ranker_sha256(),
        "source_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in _source_paths()},
        "test_panel_opened": False,
    }
    return {**payload, "contract_sha256": digest(payload)}


def gate_manifest() -> dict:
    path = DEFAULT_GATE_OUTPUT / GATE_NAME
    value = _read(path)
    expected = gate_contract()
    if (
        value.get("status") != "pass"
        or value.get("gate_contract_sha256") != expected["contract_sha256"]
        or value.get("test_panel_opened") is not False
    ):
        raise RuntimeError("E263 development gate identity or verdict differs")
    core = dict(value)
    recorded = core.pop("manifest_sha256", None)
    if recorded != digest(core):
        raise RuntimeError("E263 gate manifest hash is invalid")
    return value


def method_declaration(gate: dict) -> dict:
    audit = {
        identifier: {"status": "not_applicable", "evidence": "E263 is a frozen certification mechanism ablation."}
        for identifier, _description in CHECKLIST
    }
    evidence = {
        "reproducibility.method_revision": f"E263 records {method_revision()}.",
        "reproducibility.content_hashes": "The corrected checkpoint, state generator, shared ranker, verifier, runner, and gate are hashed.",
        "reproducibility.oracle_declarations": "Exact recoverability filters certified arms and is read-only telemetry in unfiltered arms.",
        "reproducibility.failed_gates_retained": "Initial-state witness failures, route failures, and unsafe events are retained.",
        "reproducibility.preprotected_integration_gate": f"Development gate {gate['manifest_sha256']} exercised all four routes.",
    }
    for identifier, text in evidence.items():
        audit[identifier] = {"status": "pass", "evidence": text}
    declaration = {
        "method_reference": METHOD_REFERENCE,
        "method_revision": method_revision(),
        "claim_level": "expressibility",
        "declared_deviations": [
            "Prefilled states are selected for an initial exact witness; this is a mechanism stress test, not an IID panel.",
            "E263 changes only E243's frozen VCG checkpoint to corrected E255 seed 0 episode 280.",
        ],
        "oracle_inputs": ["exact recoverability labels used only for filtering or read-only telemetry"],
        "nonconforming_shortcuts": [],
        "conformance_audit": audit,
    }
    validate_protocol(declaration, source=PROTOCOL)
    return declaration


def contract() -> dict:
    gate = gate_manifest()
    checkpoint = checkpoint_identity()
    payload = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        **method_declaration(gate),
        "scope": "corrected seed-0 frozen paired 5x5 prefilled certification curve; 25 witnessed-safe days per fill",
        "training_runs": 0,
        "model_seed": MODEL_SEED,
        "checkpoint": checkpoint,
        "initial_fills": list(scenario.INITIAL_FILLS),
        "initial_fill_fractions": [fill / scenario.CAPACITY for fill in scenario.INITIAL_FILLS],
        "cases_per_fill": scenario.CASES_PER_FILL,
        "instance_seeds": list(scenario.SEEDS),
        "initial_state_requirement": "every materialized start is exact SAFE with a concrete completion witness",
        "arms": {
            "vcg_certified": "corrected frozen critic over SAFE-only frontier",
            "vcg_physical_unfiltered": "same corrected critic over physical frontier",
            "dynamic_certified": "shared fixed Dynamic ranker over SAFE-only frontier",
            "dynamic_physical_unfiltered": "identical shared fixed Dynamic ranker over physical frontier",
        },
        "dynamic_pair_only_difference": "candidate certification filter",
        "dynamic_ranker_sha256": dynamic_ranker_sha256(),
        "guard_contract": "unchanged progress guard in all four arms",
        "expected_rows": EXPECTED_ROWS,
        "failure_curve": "strict failure rate by initial occupied-cell count",
        "cost_analysis": "J/components only among strict completers",
        "development_gate": {"manifest": str((DEFAULT_GATE_OUTPUT / GATE_NAME).resolve()), "manifest_sha256": gate["manifest_sha256"]},
        "source_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in _source_paths()},
        "test_panel_opened": False,
        "paper_evidence": False,
    }
    validate_protocol(payload, source=PROTOCOL)
    return {**payload, "contract_sha256": digest(payload)}


__all__ = [name for name in globals() if name.isupper()] + [
    "checkpoint_identity", "contract", "digest", "dynamic_ranker_sha256",
    "gate_contract", "gate_manifest", "method_declaration", "method_revision", "sha256",
]
