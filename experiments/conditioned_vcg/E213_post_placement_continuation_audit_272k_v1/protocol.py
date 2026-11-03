"""Locked protocol for the E212 post-placement continuation audit."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

from experiments.conditioned_vcg.E191_fresh_long_stay_crowding_250k_v1 import (
    protocol as e191,
)
from experiments.conditioned_vcg.E205_e204_paired_tradeoff_development_264k_v1 import (
    protocol as e205,
)
from experiments.conditioned_vcg.E207_e204_immediate_only_ablation_266k_v1 import (
    protocol as e207,
)
from experiments.conditioned_vcg.E212_e204_heavy_band_mechanism_audit_271k_v1 import (
    protocol as e212,
)
from projects.temporal_consequence_realization.method_conformance import (
    CHECKLIST,
    validate_protocol,
)


ROOT = e191.ROOT
PROTOCOL = "vcg_conditioned_e213_post_placement_continuation_audit_272k_v1"
METHOD_REFERENCE = "projects/temporal_consequence_realization/METHOD_REFERENCE.md"
METHOD_REVISION = "TCR-METHOD-2026-10-29.8"
MODEL_SEED = 0
SCENARIO = e205.SCENARIO
HEAVY_INSTANCE_SEED = e212.HEAVY_INSTANCE_SEED
DIVERGENCE_INDEX = 6
FORCED_PLACEMENT_KEY = "accept:B3:6:6"
IMMEDIATE_LAMBDA = 20.0
PLAIN_E191_LAMBDA = 10.0
SOURCE_E205 = e205.DEFAULT_OUTPUT
SOURCE_E207 = e207.DEFAULT_OUTPUT
SOURCE_E212 = e212.DEFAULT_OUTPUT
DEFAULT_OUTPUT = ROOT / "results/vcg-conditioned-e213-post-placement-continuation-audit-272k-v1"


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
        raise RuntimeError(f"missing E213 source artifact: {path}")
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"invalid E213 source artifact: {path}")
    return value


def method_declaration() -> dict:
    audit = {
        identifier: {
            "status": "not_applicable",
            "evidence": (
                "E213 is a frozen VCG controller continuation diagnostic and "
                "does not train or evaluate the canonical law-condition representation."
            ),
        }
        for identifier, _description in CHECKLIST
    }
    for identifier, evidence in {
        "reproducibility.method_revision": "E213 pins the current method revision.",
        "reproducibility.content_hashes": "Code, frozen source rows, instance, placement decision, and checkpoints are hashed.",
        "reproducibility.oracle_declarations": "The E212-observed placement is deliberately forced as a diagnostic intervention and declared as a deviation, not an oracle method input.",
        "reproducibility.failed_gates_retained": "The lambda-20 collapse and both post-placement continuations are retained regardless of outcome.",
    }.items():
        audit[identifier] = {"status": "pass", "evidence": evidence}
    value = {
        "method_reference": METHOD_REFERENCE,
        "method_revision": METHOD_REVISION,
        "claim_level": "expressibility",
        "declared_deviations": [
            "E213 forces E212's observed seven-decision prefix through accept:B3:6:6, then switches controllers; it is a causal development diagnostic, not a deployable policy evaluation."
        ],
        "oracle_inputs": [],
        "nonconforming_shortcuts": [],
        "conformance_audit": audit,
    }
    validate_protocol(value, source=PROTOCOL)
    return value


def contract() -> dict:
    directory = Path(__file__).resolve().parent
    e212_contract = _read(SOURCE_E212 / "contract.json")
    divergence_path = SOURCE_E212 / "first-divergence.json"
    e205_10 = SOURCE_E205 / "rows/rho-80/lambda-10.json"
    e205_20 = SOURCE_E205 / "rows/rho-80/lambda-20.json"
    e207_20 = SOURCE_E207 / "rows/rho-80/lambda-20.json"
    stage_b = e205.stage_b_identity()
    sources = tuple(directory / name for name in (
        "protocol.py", "run.py", "run.sh", "README.md"
    )) + (
        ROOT / "experiments/conditioned_vcg/E204_e191_centered_rehandle_stage_b_263k_v1/centered.py",
        ROOT / "experiments/conditioned_vcg/E204_e191_centered_rehandle_stage_b_263k_v1/run.py",
        ROOT / "experiments/conditioned_vcg/E205_e204_paired_tradeoff_development_264k_v1/run.py",
        ROOT / "experiments/conditioned_vcg/E207_e204_immediate_only_ablation_266k_v1/immediate.py",
        ROOT / "experiments/conditioned_vcg/E212_e204_heavy_band_mechanism_audit_271k_v1/run.py",
        ROOT / METHOD_REFERENCE,
        ROOT / "projects/temporal_consequence_realization/method_conformance.py",
    )
    payload = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        **deepcopy(method_declaration()),
        "scope": "frozen development-only post-placement continuation intervention",
        "scenario": SCENARIO,
        "model_seed": MODEL_SEED,
        "heavy_instance_seed": HEAVY_INSTANCE_SEED,
        "divergence_index": DIVERGENCE_INDEX,
        "forced_placement_key": FORCED_PLACEMENT_KEY,
        "continuations": {
            "immediate_only_lambda20": IMMEDIATE_LAMBDA,
            "plain_e191_lambda10": PLAIN_E191_LAMBDA,
        },
        "questions": [
            "Does the immediate-only lambda-20 controller recover after the learned controller's B3 placement at (6,6)?",
            "Does plain E191 recover after the identical forced prefix and placement?",
            "Is the 309-rehandle collapse attributable mainly to the placement, to subsequent learned-head choices, or to their interaction?",
        ],
        "training_runs": 0,
        "expected_complete_episode_runs": 2,
        "source_e212_contract_sha256": e212_contract["contract_sha256"],
        "source_stage_b": stage_b,
        "source_artifacts": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in (
                divergence_path,
                e205_10,
                e205_20,
                e207_20,
                Path(stage_b["terminal_checkpoint"]),
            )
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
