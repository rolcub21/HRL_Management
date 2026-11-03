"""Canonical project-local freeze of the E214 Stage-B procedure."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

from projects.temporal_consequence_realization.method_conformance import (
    CHECKLIST,
    validate_protocol,
)
from experiments.conditioned_vcg.E214_on_policy_trace_rehandle_stage_b_8x8_273k_v1 import (
    protocol as e214,
)


ROOT = e214.ROOT
PROTOCOL = "vcg_conditioned_e216_e214_stage_b_procedure_freeze_275k_v1"
METHOD_REFERENCE = "projects/temporal_consequence_realization/METHOD_REFERENCE.md"
METHOD_REVISION = "TCR-METHOD-2026-10-29.8"
DEFAULT_OUTPUT = (
    ROOT / "results/vcg-conditioned-e216-e214-stage-b-procedure-freeze-275k-v1"
)
E214_OUTPUT = e214.DEFAULT_OUTPUT
E214_REPORT = E214_OUTPUT / "e214-report.json"
E214_TEXT_REPORT = E214_OUTPUT / "REPORT.txt"
E214_CONTRACT = E214_OUTPUT / "contract.json"
E214_SUMMARY = E214_OUTPUT / "training/seed-0/training-summary.json"
E214_ENDPOINT = E214_OUTPUT / "training/seed-0/checkpoints/round-4.pth"

E214_REPORT_SHA256 = "0ca79b81eea2da9c70484de2070ffa1569ba4ca3090659f2ef796dcca44dc600"
E214_TEXT_REPORT_SHA256 = "42cfe4f5573fff69736396bc3ffc7ee01eb44def503e136aebc678205a0691ce"
E214_CONTRACT_SHA256 = "83d5d549c40e7311e776cdf774bd141a2730c25c21e3ea2897f2e24012e4f688"
E214_SUMMARY_SHA256 = "e5ee36630b39c13deb7cc8510087c6db375c438f36ed0397e9dce66ab5d5a884"
E214_ENDPOINT_SHA256 = "d3045fcdd18d3be21c0df0a3817809b629c2b0cab5cbf991dbc48bc3e5099f8b"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()).hexdigest()


STAGE_B_PROCEDURE = {
    "name": "e214_round_frozen_on_policy_trace_future_rehandles_v1",
    "role": (
        "preference-conditioned correction of a frozen Stage-A operational "
        "critic using predicted future physical rehandles"
    ),
    "frozen_stage_a": True,
    "stage_a_reference_rehandle_weight": 10.0,
    "predicted_quantity": (
        "N_alpha(s,c)=E[sum_{k>0} I(reconfigure_k) | "
        "s,c,then follow pi_alpha]"
    ),
    "deployment_merit": (
        "Q_stage_a - 0.01*(alpha-10)*(I_reconfigure+N_alpha)"
    ),
    "preference_semantics": (
        "alpha is a preference coordinate producing operating points; it is "
        "not claimed to equal an exact objective price"
    ),
    "supported_alpha_range": [10.0, 40.0],
    "extrapolation_above_40": (
        "unsupported; alpha 80/160 requires a separately versioned range "
        "extension and new on-policy collection"
    ),
    "rounds": 4,
    "episodes_per_round": 9,
    "episodes_per_round_design": (
        "one episode for each of three workloads crossed with three continuous "
        "alpha bands"
    ),
    "alpha_bands": [[10.0, 20.0], [20.0, 30.0], [30.0, 40.0]],
    "alpha_sampling": "continuous uniform draw within band; fixed for episode",
    "behavior_head": (
        "freeze the round-start head for collection and for every bootstrap "
        "term in that round"
    ),
    "target": (
        "G_t=I(c_next)+(1-eta)N_round_start(s_next,c_next,alpha)+eta*G_next"
    ),
    "trace_eta": 0.99,
    "data_retention": (
        "fit only the current round's on-policy episodes; discard them after fit"
    ),
    "head_initialization": "fresh near-zero head; immediate-only initial policy",
    "fit": {
        "epochs_per_round": 40,
        "batch_size": 128,
        "learning_rate": 2.0e-4,
        "loss": "smooth_l1 on trace return",
        "gradient_clip": 5.0,
    },
    "primary_calibration": (
        "held-out Monte-Carlo future-rehandle MAE against an exact zero-head baseline"
    ),
    "required_comparator": (
        "same frozen Stage-A critic with alpha applied only to the immediate "
        "rehandle indicator"
    ),
    "required_evaluation": (
        "paired instances by workload and alpha; report rehandles, timing, "
        "admission wait, work, occupancy, J10, and rescored J_alpha"
    ),
    "required_safety": (
        "retain Stage-A certification and the same progress action-space mask"
    ),
}


EVIDENCE_BOUNDARY = {
    "supported_on": "E214 seed-0 8x8 development panel only",
    "supported_findings": [
        "the trace head beats the zero predictor on held-out MC count error",
        "the 309-rehandle collapse is removed",
        "at alpha=40 the learned future term beats immediate-only on all three days",
        "all learned and causal episodes complete strictly",
    ],
    "required_to_expand": [
        "replicate with independently trained Stage-A seeds",
        "retain the immediate-only paired comparator",
        "report the heavy alpha=20 7.4% J regression",
    ],
    "not_supported": [
        "alpha is an exact rehandle price",
        "rehandles are monotone in alpha on every instance",
        "the old decision-7 placement is intrinsically bad",
        "alpha above 40 is valid extrapolation",
        "sealed-test or multiseed superiority",
    ],
}


def method_declaration() -> dict:
    audit = {
        identifier: {
            "status": "not_applicable",
            "evidence": (
                "E216 freezes a historical VCG Stage-B controller procedure "
                "and makes no canonical law-condition representation claim."
            ),
        }
        for identifier, _description in CHECKLIST
    }
    for identifier, evidence in {
        "reproducibility.method_revision": (
            "The E216 freeze pins the current method-reference revision."
        ),
        "reproducibility.content_hashes": (
            "The complete E214 sources, contract, report, summary, and endpoint "
            "are content-addressed."
        ),
        "reproducibility.oracle_declarations": (
            "E214's forced-prefix causal diagnostic remains declared and is not "
            "part of deployment."
        ),
        "reproducibility.failed_gates_retained": (
            "The heavy alpha=20 regression and failed exact-price interpretation "
            "are explicit frozen limitations."
        ),
    }.items():
        audit[identifier] = {"status": "pass", "evidence": evidence}
    value = {
        "method_reference": METHOD_REFERENCE,
        "method_revision": METHOD_REVISION,
        "claim_level": "expressibility",
        "declared_deviations": [
            "E216 is a project-local VCG Stage-B procedure freeze, not evidence "
            "for the canonical law-condition representation."
        ],
        "oracle_inputs": [],
        "nonconforming_shortcuts": [],
        "conformance_audit": audit,
    }
    validate_protocol(value, source=PROTOCOL)
    return value


def _source_paths() -> tuple[Path, ...]:
    directory = Path(__file__).resolve().parent
    e214_directory = Path(e214.__file__).resolve().parent
    return (
        directory / "protocol.py",
        directory / "run.py",
        directory / "run.sh",
        directory / "PROCEDURE.md",
        e214_directory / "protocol.py",
        e214_directory / "returns.py",
        e214_directory / "run.py",
        e214_directory / "plot.py",
        e214_directory / "run.sh",
        e214_directory / "README.md",
        ROOT / METHOD_REFERENCE,
        ROOT / "projects/temporal_consequence_realization/method_conformance.py",
    )


def contract() -> dict:
    payload = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        **deepcopy(method_declaration()),
        "scope": "immutable project-local procedure freeze; no episode runs",
        "source_stage_b_protocol": e214.PROTOCOL,
        "source_stage_b_contract_sha256": e214.contract()["contract_sha256"],
        "stage_b_procedure": deepcopy(STAGE_B_PROCEDURE),
        "evidence_boundary": deepcopy(EVIDENCE_BOUNDARY),
        "frozen_artifacts": {
            str(E214_CONTRACT.relative_to(ROOT)): E214_CONTRACT_SHA256,
            str(E214_REPORT.relative_to(ROOT)): E214_REPORT_SHA256,
            str(E214_TEXT_REPORT.relative_to(ROOT)): E214_TEXT_REPORT_SHA256,
            str(E214_SUMMARY.relative_to(ROOT)): E214_SUMMARY_SHA256,
            str(E214_ENDPOINT.relative_to(ROOT)): E214_ENDPOINT_SHA256,
        },
        "future_protocol_rule": (
            "future Stage-B experiments import STAGE_B_PROCEDURE and declare "
            "every changed field; no silent E214 source edits"
        ),
        "training_runs": 0,
        "episode_runs": 0,
        "test_panel_opened": False,
        "paper_evidence": False,
        "source_sha256": {
            str(path.relative_to(ROOT)): sha256(path) for path in _source_paths()
        },
    }
    validate_protocol(payload, source=PROTOCOL)
    return {**payload, "contract_sha256": digest(payload)}


__all__ = [
    "DEFAULT_OUTPUT", "E214_ENDPOINT", "E214_ENDPOINT_SHA256",
    "E214_REPORT", "E214_REPORT_SHA256", "EVIDENCE_BOUNDARY",
    "METHOD_REFERENCE", "METHOD_REVISION", "PROTOCOL", "STAGE_B_PROCEDURE",
    "contract", "digest", "method_declaration", "sha256",
]
