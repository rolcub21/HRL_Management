"""Authenticated protocol for the matched-ranker 5x5 prefilled curve."""

from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
import re

from experiments.conditioned_vcg.E27_certified_dynamic_sustained_99k.selector import (
    CertifiedDynamicAgent,
)
from experiments.conditioned_vcg.E195_e191_method_5x5_longstay_254k_v1 import (
    protocol as e195,
)
from projects.temporal_consequence_realization.method_conformance import (
    CHECKLIST,
    validate_protocol,
)

from . import scenario
from .tracker import MatchedDynamicE195Agent


ROOT = e195.ROOT
PROTOCOL = "vcg_conditioned_e243_e195_expanded_prefilled_curve_302k_v1"
GATE_PROTOCOL = PROTOCOL + "_development_gate"
METHOD_REFERENCE = "projects/temporal_consequence_realization/METHOD_REFERENCE.md"
MODEL_SEED = 0
SOURCE_BUNDLE = (
    ROOT
    / "results/vcg-conditioned-e243-e195-expanded-prefilled-curve-302k-v1-source"
    / "seed-0"
)
SOURCE_MANIFEST = SOURCE_BUNDLE / "freeze.json"
SOURCE_CHECKPOINT = SOURCE_BUNDLE / "selected-development-checkpoint.pth"
EXPECTED_SOURCE_CONTRACT_SHA256 = (
    "8720c8154aed2618ed50f5b07453d1b14e99430692e1f5ede7f80f0bc94e8f75"
)
EXPECTED_CHECKPOINT_SHA256 = (
    "03acf192e1162b8cc879df2831790e8d43f163aa8f3390ce3cb92bec5f1dfec4"
)
ARMS = (
    "vcg_certified",
    "vcg_physical_unfiltered",
    "dynamic_certified",
    "dynamic_physical_unfiltered",
)
CERTIFIED_ARMS = ("vcg_certified", "dynamic_certified")
UNFILTERED_ARMS = (
    "vcg_physical_unfiltered", "dynamic_physical_unfiltered",
)
DYNAMIC_ARMS = ("dynamic_certified", "dynamic_physical_unfiltered")
DEFAULT_OUTPUT = (
    ROOT / "results/vcg-conditioned-e243-e195-prefilled-certification-curve-302k-v1"
)
DEFAULT_GATE_OUTPUT = Path(str(DEFAULT_OUTPUT) + "-gate")
GATE_NAME = "gate.json"
REPORT_NAME = "REPORT.txt"
REPORT_JSON = "e243-report.json"
TABLE_NAME = "e243-table.md"
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
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"invalid E243 artifact: {path}")
    return value


def method_revision() -> str:
    match = re.search(
        r"^\*\*Revision:\*\* `([^`]+)`",
        (ROOT / METHOD_REFERENCE).read_text(encoding="utf-8"), re.MULTILINE,
    )
    if match is None:
        raise RuntimeError("E243 cannot parse the current method revision")
    return match.group(1)


def checkpoint_identity() -> dict:
    if not SOURCE_MANIFEST.is_file() or not SOURCE_CHECKPOINT.is_file():
        raise RuntimeError(
            "E243 source bundle is missing; extract the authenticated release "
            "asset at the repository root"
        )
    manifest = _read(SOURCE_MANIFEST)
    core = dict(manifest)
    recorded = core.pop("manifest_core_sha256", None)
    if recorded != digest(core):
        raise RuntimeError("E243 E196 freeze manifest hash is invalid")
    expected = {
        "status": "complete",
        "model_seed": MODEL_SEED,
        "source_protocol": e195.PROTOCOL,
        "source_contract_sha256": EXPECTED_SOURCE_CONTRACT_SHA256,
        "frozen_checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        "byte_identical_to_source_checkpoint": True,
        "test_panels_opened": False,
    }
    mismatches = {
        key: (manifest.get(key), value)
        for key, value in expected.items() if manifest.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"E243 source identity differs: {mismatches!r}")
    observed = sha256(SOURCE_CHECKPOINT)
    if observed != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError("E243 frozen E195 checkpoint SHA-256 differs")
    return {
        "status": "ready",
        "source_protocol": manifest["source_protocol"],
        "source_contract_sha256": manifest["source_contract_sha256"],
        "source_checkpoint": str(SOURCE_CHECKPOINT.resolve()),
        "source_checkpoint_sha256": observed,
        "source_manifest": str(SOURCE_MANIFEST.resolve()),
        "source_manifest_sha256": sha256(SOURCE_MANIFEST),
        "selected_episode": int(manifest["selected_episode"]),
        "model_seed": MODEL_SEED,
        "development_only": True,
        "test_panels_opened": False,
    }


def dynamic_ranker_sha256() -> str:
    source = "\n".join((
        inspect.getsource(CertifiedDynamicAgent._choice),
        inspect.getsource(CertifiedDynamicAgent.select),
        inspect.getsource(MatchedDynamicE195Agent.select),
    ))
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def _source_paths() -> tuple[Path, ...]:
    directory = Path(__file__).resolve().parent
    return (
        directory / "protocol.py", directory / "scenario.py",
        directory / "tracker.py", directory / "run.py",
        directory / "run.sh", directory / "README.md",
        ROOT / METHOD_REFERENCE,
        ROOT / "projects/temporal_consequence_realization/method_conformance.py",
        ROOT / "experiments/conditioned_vcg/E27_certified_dynamic_sustained_99k/selector.py",
        ROOT / "experiments/conditioned_vcg/E198_e195_shadow_certification_ablation_257k_v1/frontier.py",
        ROOT / "experiments/conditioned_vcg/E195_e191_method_5x5_longstay_254k_v1/instrumentation.py",
        ROOT / "experiments/conditioned_vcg/E195_e191_method_5x5_longstay_254k_v1/representation.py",
    )


def gate_contract() -> dict:
    checkpoint = checkpoint_identity()
    payload = {
        "schema_version": 1,
        "protocol": GATE_PROTOCOL,
        "scope": "development live integration gate for E243 four matched routes",
        "development_seed": scenario.SEEDS[0],
        "routes": list(ARMS),
        "checkpoint_sha256": checkpoint["source_checkpoint_sha256"],
        "dynamic_ranker_sha256": dynamic_ranker_sha256(),
        "source_sha256": {
            str(path.relative_to(ROOT)): sha256(path) for path in _source_paths()
        },
        "test_panel_opened": False,
    }
    return {**payload, "contract_sha256": digest(payload)}


def gate_manifest() -> dict:
    path = DEFAULT_GATE_OUTPUT / GATE_NAME
    if not path.is_file():
        raise RuntimeError("E243 requires its development integration gate first")
    value = _read(path)
    expected = gate_contract()
    if (
        value.get("status") != "pass"
        or value.get("gate_contract_sha256") != expected["contract_sha256"]
        or value.get("test_panel_opened") is not False
    ):
        raise RuntimeError("E243 development gate identity or verdict differs")
    core = dict(value)
    recorded = core.pop("manifest_sha256", None)
    if recorded != digest(core):
        raise RuntimeError("E243 gate manifest hash is invalid")
    return value


def method_declaration(gate: dict) -> dict:
    audit = {
        identifier: {
            "status": "not_applicable",
            "evidence": (
                "E243 is a historical conditioned-VCG certification mechanism "
                "ablation, not a canonical law-condition representation claim."
            ),
        }
        for identifier, _description in CHECKLIST
    }
    evidence = {
        "reproducibility.method_revision": f"E243 records {method_revision()}.",
        "reproducibility.content_hashes": (
            "The checkpoint, initial-state generator, shared ranker, verifier, "
            "runner, protocol, and gate are content-addressed."
        ),
        "reproducibility.oracle_declarations": (
            "Exact recoverability is authoritative only in certified arms and "
            "read-only telemetry in unfiltered arms."
        ),
        "reproducibility.failed_gates_retained": (
            "Initial-state witness failures, route failures, and unsafe events "
            "are retained rather than repaired after evaluation."
        ),
        "reproducibility.preprotected_integration_gate": (
            f"Development gate {gate['manifest_sha256']} exercised all four "
            "routes before the fresh stress panel."
        ),
    }
    for identifier, text in evidence.items():
        audit[identifier] = {"status": "pass", "evidence": text}
    declaration = {
        "method_reference": METHOD_REFERENCE,
        "method_revision": method_revision(),
        "claim_level": "expressibility",
        "declared_deviations": [
            "E243 evaluates historical conditioned VCG and a fixed Dynamic-style "
            "ranker, not the canonical law-condition representation.",
            "Prefilled states are selected only for an initial exact completion "
            "witness; this is a mechanism stress test, not an IID performance panel.",
        ],
        "oracle_inputs": [
            "exact recoverability labels used only for filtering or read-only telemetry"
        ],
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
        "scope": (
            "expanded frozen paired 5x5 prefilled certification failure curve; "
            "25 witnessed-safe days per fill level"
        ),
        "training_runs": 0,
        "model_seed": MODEL_SEED,
        "checkpoint": checkpoint,
        "initial_fills": list(scenario.INITIAL_FILLS),
        "initial_fill_fractions": [
            fill / scenario.CAPACITY for fill in scenario.INITIAL_FILLS
        ],
        "cases_per_fill": scenario.CASES_PER_FILL,
        "instance_seeds": list(scenario.SEEDS),
        "initial_state_requirement": (
            "every materialized start is exact SAFE with a concrete completion witness"
        ),
        "arms": {
            "vcg_certified": "frozen critic over SAFE-only frontier",
            "vcg_physical_unfiltered": "same critic over physical frontier",
            "dynamic_certified": "shared fixed Dynamic ranker over SAFE-only frontier",
            "dynamic_physical_unfiltered": (
                "identical shared fixed Dynamic ranker over physical frontier"
            ),
        },
        "dynamic_pair_only_difference": "candidate certification filter",
        "dynamic_ranker_class": (
            "E243_e195_expanded_prefilled_curve_302k_v1."
            "tracker.MatchedDynamicE195Agent"
        ),
        "dynamic_ranker_sha256": dynamic_ranker_sha256(),
        "guard_contract": "unchanged E195 progress guard in all four arms",
        "expected_rows": EXPECTED_ROWS,
        "failure_curve": "strict failure rate by initial occupied-cell count",
        "cost_analysis": "J/components only among strict completers",
        "development_gate": {
            "manifest": str((DEFAULT_GATE_OUTPUT / GATE_NAME).resolve()),
            "manifest_sha256": gate["manifest_sha256"],
        },
        "source_sha256": {
            str(path.relative_to(ROOT)): sha256(path) for path in _source_paths()
        },
        "test_panel_opened": False,
        "paper_evidence": False,
    }
    validate_protocol(payload, source=PROTOCOL)
    return {**payload, "contract_sha256": digest(payload)}


__all__ = [name for name in globals() if name.isupper()] + [
    "checkpoint_identity", "contract", "digest", "dynamic_ranker_sha256",
    "gate_contract", "gate_manifest", "method_declaration", "method_revision",
    "sha256",
]
