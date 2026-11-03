"""Locked protocol for the frozen E204 heavy-day mechanism audit."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

from experiments.conditioned_vcg.E191_fresh_long_stay_crowding_250k_v1 import (
    protocol as e191,
)
from experiments.conditioned_vcg.E204_e191_centered_rehandle_stage_b_263k_v1 import (
    protocol as e204,
)
from experiments.conditioned_vcg.E205_e204_paired_tradeoff_development_264k_v1 import (
    protocol as e205,
)
from experiments.conditioned_vcg.E207_e204_immediate_only_ablation_266k_v1 import (
    protocol as e207,
)
from projects.temporal_consequence_realization.method_conformance import (
    CHECKLIST,
    validate_protocol,
)


ROOT = e191.ROOT
PROTOCOL = "vcg_conditioned_e212_e204_heavy_band_mechanism_audit_271k_v1"
METHOD_REFERENCE = "projects/temporal_consequence_realization/METHOD_REFERENCE.md"
METHOD_REVISION = "TCR-METHOD-2026-10-29.8"
MODEL_SEED = 0
SCENARIO = e205.SCENARIO
HEAVY_INSTANCE_SEED = 264_205_202
TRACE_LAMBDA = 20.0
SWEEP_LAMBDAS = (15.0, 20.0, 25.0, 30.0, 40.0)
NEW_SWEEP_LAMBDAS = (15.0, 25.0, 30.0)
SOURCE_E205 = e205.DEFAULT_OUTPUT
SOURCE_E207 = e207.DEFAULT_OUTPUT
DEFAULT_OUTPUT = ROOT / "results/vcg-conditioned-e212-e204-heavy-band-mechanism-audit-271k-v1"


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()).hexdigest()


def _read(path: Path) -> dict:
    if not Path(path).is_file():
        raise RuntimeError(f"missing E212 source artifact: {path}")
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"invalid E212 source artifact: {path}")
    return value


def method_declaration() -> dict:
    audit = {
        identifier: {
            "status": "not_applicable",
            "evidence": (
                "E212 is a frozen VCG controller diagnostic and does not train "
                "or evaluate the canonical law-condition representation."
            ),
        }
        for identifier, _description in CHECKLIST
    }
    for identifier, evidence in {
        "reproducibility.method_revision": "E212 pins the current method revision.",
        "reproducibility.content_hashes": "Code, source contracts, rows, instance, and checkpoint are hashed.",
        "reproducibility.oracle_declarations": "No oracle input is used; only frozen controller quantities and realized outcomes are inspected.",
        "reproducibility.failed_gates_retained": "The collapsed lambda-20 row is retained and is the declared diagnostic target.",
    }.items():
        audit[identifier] = {"status": "pass", "evidence": evidence}
    value = {
        "method_reference": METHOD_REFERENCE,
        "method_revision": METHOD_REVISION,
        "claim_level": "expressibility",
        "declared_deviations": [
            "E212 diagnoses an E204 Stage-B controller failure; it is not a canonical representation experiment."
        ],
        "oracle_inputs": [],
        "nonconforming_shortcuts": [],
        "conformance_audit": audit,
    }
    validate_protocol(value, source=PROTOCOL)
    return value


def contract() -> dict:
    directory = Path(__file__).resolve().parent
    e205_contract = _read(SOURCE_E205 / "contract.json")
    e207_contract = _read(SOURCE_E207 / "contract.json")
    instance_path = SOURCE_E205 / "instances.json"
    e205_20 = SOURCE_E205 / "rows/rho-80/lambda-20.json"
    e205_40 = SOURCE_E205 / "rows/rho-80/lambda-40.json"
    e207_20 = SOURCE_E207 / "rows/rho-80/lambda-20.json"
    stage_b = e205.stage_b_identity()
    sources = tuple(directory / name for name in (
        "protocol.py", "run.py", "run.sh", "README.md"
    )) + (
        ROOT / "experiments/conditioned_vcg/E204_e191_centered_rehandle_stage_b_263k_v1/centered.py",
        ROOT / "experiments/conditioned_vcg/E204_e191_centered_rehandle_stage_b_263k_v1/run.py",
        ROOT / "experiments/conditioned_vcg/E205_e204_paired_tradeoff_development_264k_v1/run.py",
        ROOT / "experiments/conditioned_vcg/E207_e204_immediate_only_ablation_266k_v1/immediate.py",
        ROOT / METHOD_REFERENCE,
        ROOT / "projects/temporal_consequence_realization/method_conformance.py",
    )
    payload = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        **deepcopy(method_declaration()),
        "scope": "frozen development-only first-divergence replay and local lambda sweep",
        "scenario": SCENARIO,
        "model_seed": MODEL_SEED,
        "heavy_instance_seed": HEAVY_INSTANCE_SEED,
        "trace_lambda": TRACE_LAMBDA,
        "sweep_lambdas": list(SWEEP_LAMBDAS),
        "new_sweep_lambdas": list(NEW_SWEEP_LAMBDAS),
        "questions": [
            "Did the first learned-versus-immediate divergence select a reconfigure because predicted downstream savings exceeded its immediate cost?",
            "Did the predicted saving materialize in the realized continuation?",
            "Is the lambda-20 collapse isolated or part of a neighboring failure band?",
        ],
        "training_runs": 0,
        "source_e205_contract_sha256": e205_contract["contract_sha256"],
        "source_e207_contract_sha256": e207_contract["contract_sha256"],
        "source_stage_b": stage_b,
        "source_artifacts": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in (instance_path, e205_20, e205_40, e207_20, Path(stage_b["terminal_checkpoint"]))
        },
        "source_sha256": {
            str(path.relative_to(ROOT)): sha256(path) for path in sources
        },
        "test_panel_opened": False,
        "paper_evidence": False,
    }
    validate_protocol(payload, source=PROTOCOL)
    return {**payload, "contract_sha256": digest(payload)}


__all__ = [name for name in globals() if name.isupper()] + [
    "contract", "digest", "method_declaration", "sha256",
]
