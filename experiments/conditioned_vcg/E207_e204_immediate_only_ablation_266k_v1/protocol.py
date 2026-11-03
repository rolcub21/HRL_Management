"""Locked protocol for E207's paired immediate-only ablation."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

from experiments.conditioned_vcg.E205_e204_paired_tradeoff_development_264k_v1 import (
    protocol as e205,
)
from projects.temporal_consequence_realization.method_conformance import (
    CHECKLIST,
    validate_protocol,
)


ROOT = e205.ROOT
PROTOCOL = "vcg_conditioned_e207_e204_immediate_only_ablation_266k_v1"
METHOD_REFERENCE = "projects/temporal_consequence_realization/METHOD_REFERENCE.md"
METHOD_REVISION = "TCR-METHOD-2026-10-29.7"
MODEL_SEED = e205.MODEL_SEED
SCENARIO = e205.SCENARIO
LAMBDAS = (20.0, 40.0)
REFERENCE_LAMBDA = 10.0
INSTANCE_SEEDS = e205.INSTANCE_SEEDS
SOURCE_OUTPUT = e205.DEFAULT_OUTPUT
DEFAULT_OUTPUT = ROOT / "results/vcg-conditioned-e207-e204-immediate-only-ablation-266k-v1"
REPORT_NAME = "REPORT.txt"
REPORT_JSON = "e207-report.json"
CSV_NAME = "e207-paired-comparison.csv"
PLOT_NAME = "e207-future-head-ablation.png"


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def digest(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def _read(path: Path) -> dict:
    if not Path(path).is_file():
        raise RuntimeError(f"missing E207 source artifact: {path}")
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"invalid E207 source artifact: {path}")
    return value


def method_declaration() -> dict:
    audit = {
        identifier: {
            "status": "not_applicable",
            "evidence": (
                "E207 is a controller-head ablation of frozen VCG trajectories; "
                "it does not train or evaluate a canonical law-condition model."
            ),
        }
        for identifier, _description in CHECKLIST
    }
    for identifier, evidence in {
        "reproducibility.method_revision": "The E207 contract pins the current method revision.",
        "reproducibility.content_hashes": "E207 hashes its sources, E205 contract and instance manifest, and E204 terminal.",
        "reproducibility.oracle_declarations": "No oracle inputs are used; the diagnostic deviation is explicit.",
        "reproducibility.failed_gates_retained": "All paired outcomes and dominance failures are retained.",
    }.items():
        audit[identifier] = {"status": "pass", "evidence": evidence}
    declaration = {
        "method_reference": METHOD_REFERENCE,
        "method_revision": METHOD_REVISION,
        "claim_level": "expressibility",
        "declared_deviations": [
            "E207 diagnoses a VCG deployment head and makes no canonical law-condition claim."
        ],
        "oracle_inputs": [],
        "nonconforming_shortcuts": [],
        "conformance_audit": audit,
    }
    validate_protocol(declaration, source=PROTOCOL)
    return declaration


def source_identity() -> dict:
    contract_path = SOURCE_OUTPUT / "contract.json"
    instances_path = SOURCE_OUTPUT / "instances.json"
    contract = _read(contract_path)
    if contract != e205.contract():
        raise RuntimeError("E207 E205 contract differs")
    if not instances_path.is_file():
        raise RuntimeError("E207 E205 instance manifest is missing")
    stage_b = e205.stage_b_identity()
    return {
        "protocol": e205.PROTOCOL,
        "contract_sha256": contract["contract_sha256"],
        "contract_file_sha256": sha256(contract_path),
        "instances_sha256": sha256(instances_path),
        "stage_b_terminal_sha256": stage_b["terminal_checkpoint_sha256"],
        "base_checkpoint_sha256": stage_b["base_checkpoint_sha256"],
        "base_policy_digest": stage_b["base_policy_digest"],
    }


def contract() -> dict:
    directory = Path(__file__).resolve().parent
    sources = tuple(directory / name for name in (
        "immediate.py", "protocol.py", "run.py", "plot.py", "run.sh", "README.md"
    ))
    payload = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        **deepcopy(method_declaration()),
        "scope": "small paired development head ablation; not paper evidence",
        "source": source_identity(),
        "scenario": SCENARIO,
        "model_seed": MODEL_SEED,
        "instance_seeds": list(INSTANCE_SEEDS),
        "lambda_values": list(LAMBDAS),
        "common_reference_lambda": REFERENCE_LAMBDA,
        "deployment_merit": "Q_E191 - 0.01*(lambda-10)*I_reconfigure",
        "learned_future_count": "removed; exact zero",
        "comparison_arm": "E205 learned I_reconfigure + N_future_hat on identical instances",
        "primary_question": (
            "Does the learned future-rehandle term improve the paired timing-versus-rehandle "
            "frontier beyond an immediate-only penalty?"
        ),
        "analysis": [
            "paired differences at equal lambda",
            "J_lambda comparison",
            "Pareto dominance in timing_cost_2T_plus_E versus physical_rehandles",
            "non-rehandle cost per rehandle avoided relative to common lambda=10",
        ],
        "episode_runs": len(INSTANCE_SEEDS) * len(LAMBDAS),
        "training_runs": 0,
        "test_panel_opened": False,
        "paper_evidence": False,
        "source_sha256": {
            str(path.relative_to(ROOT)): sha256(path) for path in sources
        },
    }
    return {**payload, "contract_sha256": digest(payload)}


__all__ = [
    "CSV_NAME", "DEFAULT_OUTPUT", "INSTANCE_SEEDS", "LAMBDAS", "MODEL_SEED",
    "PLOT_NAME", "PROTOCOL", "REFERENCE_LAMBDA", "REPORT_JSON", "REPORT_NAME",
    "SCENARIO", "SOURCE_OUTPUT", "contract", "digest", "method_declaration",
    "sha256", "source_identity",
]
