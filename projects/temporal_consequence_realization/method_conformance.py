#!/usr/bin/env python3
"""Validate and render canonical method declarations for experiment protocols."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_REFERENCE = "projects/temporal_consequence_realization/METHOD_REFERENCE.md"
CLAIM_LEVELS = (
    "expressibility",
    "law-learning",
    "condition-learning",
    "belief-control",
    "full",
)
STATUSES = {"pass", "fail", "not_applicable"}

CHECKLIST = (
    ("representation.shared_law", "One shared law is reused across carriers and worlds."),
    ("representation.permutation_invariant_read_set", "The shared law reads each involved condition set permutation-invariantly; arbitrary enumeration order cannot change a consequence."),
    ("representation.carrier_frame_responsibility", "Mutually exclusive carrier effects are computed in each carrier's canonical frame and combined by permutation-equivariant responsibility; joint or sequential effects declare their composition rule."),
    ("representation.state_memory_roles", "State estimator, changing dynamic memory, persistent field, and shared law have explicitly declared predictive responsibilities."),
    ("representation.no_persistent_bypass", "Persistent condition knowledge acquired from history cannot bypass the designated field through an independent learned estimator or memory."),
    ("representation.responsibility_audits", "Applicable history-leakage, field-intervention, and field-ablation audits verify responsibility routing; non-applicability is justified."),
    ("representation.persistent_conditions", "Conditions are persistent carrier-level facts, not transition outcomes."),
    ("representation.anonymous_conditions", "Categories are anonymous and attributes/poses are charged."),
    ("representation.continuous_attribute_default", "Condition-family selection offers a charged continuous attribute with dimension zero allowed whenever ordered instance variation is plausible."),
    ("representation.full_field_support", "The field covers reachable unseen carriers, initialized from the prior."),
    ("representation.no_uncharged_addresses", "Absolute addresses are absent from the shared law unless selected and charged."),
    ("learning.one_full_likelihood", "Full consequences come from one conditional likelihood."),
    ("learning.joint_or_declared_alternation", "Law, conditions, and prior are fitted jointly or by a declared alternating procedure; departures from the preferred pooled-law/per-carrier-inference schedule are justified."),
    ("learning.belief_samples", "Belief samples, not only point estimates, enter training when uncertainty is claimed."),
    ("learning.capacity_by_total_cost", "Capacity and vocabulary choices use total cost and held-out likelihood."),
    ("learning.evidence_monotonicity", "Matched held-out predictive loss is audited across increasing evidence, with a paired interval or declared tolerance for nonmonotonicity."),
    ("learning.labels_grading_only", "Labels are grading-only."),
    ("learning.importance_correction", "Any biased sampler has exact importance correction."),
    ("learning.restoration_audit", "Any learned part that becomes unused is restored to a plausible label-free setting and rescored under the same final total description length; a cheaper restoration is reported as an optimization failure."),
    ("control.persistent_imagined_conditions", "Conditions persist across the imagined plan unless written."),
    ("control.update_after_consequence", "Beliefs update after every real consequence."),
    ("control.evidence_weighted_laws", "Law alternatives are evidence-weighted."),
    ("control.heldout_calibration", "Calibration is measured on held-out data."),
    ("control.belief_scramble_preoutcome", "Belief-scramble pairings, when used, are selected from beliefs before query outcomes are scored and actually exchange distinct beliefs."),
    ("control.absorbing_goals_separate_risk", "Goals are absorbing and utility and risk quantities are not conflated."),
    ("control.law_belief_values", "Values and plans are derived from the current law and belief."),
    ("reproducibility.method_revision", "The protocol records the method revision."),
    ("reproducibility.content_hashes", "Data, split, implementation, protocol, and checkpoint hashes are stored."),
    ("reproducibility.oracle_declarations", "Diagnostic oracle inputs and deviations are explicit."),
    ("reproducibility.failed_gates_retained", "Optimization diagnostics and failed gates are retained."),
)


class ConformanceError(ValueError):
    """Raised when a protocol cannot support its declared method claim."""


def _load(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except json.JSONDecodeError as error:
        raise ConformanceError(f"{path}: protocol must be valid JSON: {error}") from error
    if not isinstance(value, dict):
        raise ConformanceError(f"{path}: protocol root must be an object")
    return value


def _as_list(protocol: Mapping[str, Any], key: str, errors: list[str]) -> list[Any]:
    value = protocol.get(key)
    if not isinstance(value, list):
        errors.append(f"{key} must be a list")
        return []
    return value


def validate_protocol(protocol: Mapping[str, Any], *, source: str = "protocol") -> dict[str, Any]:
    """Return a normalized declaration or raise before a run can start."""
    errors: list[str] = []
    for key in ("method_reference", "method_revision", "claim_level",
                "declared_deviations", "oracle_inputs", "conformance_audit"):
        if key not in protocol:
            errors.append(f"missing required declaration: {key}")

    reference = protocol.get("method_reference")
    if reference != DEFAULT_REFERENCE:
        errors.append(f"method_reference must be {DEFAULT_REFERENCE!r}")
    revision = protocol.get("method_revision")
    if not isinstance(revision, str) or not revision.strip():
        errors.append("method_revision must be a nonempty string")
    else:
        reference_path = PROJECT_ROOT.parents[1] / DEFAULT_REFERENCE
        if not reference_path.exists():
            errors.append(f"canonical reference is missing: {reference_path}")
        else:
            match = re.search(r"^\*\*Revision:\*\* `([^`]+)`", reference_path.read_text(), re.MULTILINE)
            if match is None:
                errors.append("canonical reference has no parseable current revision")
            elif revision != match.group(1):
                errors.append(
                    f"method_revision {revision!r} is stale; current revision is {match.group(1)!r}")

    claim = protocol.get("claim_level")
    if claim not in CLAIM_LEVELS:
        errors.append(f"claim_level must be one of {', '.join(CLAIM_LEVELS)}")
    deviations = _as_list(protocol, "declared_deviations", errors)
    oracle_inputs = _as_list(protocol, "oracle_inputs", errors)
    shortcuts = protocol.get("nonconforming_shortcuts", [])
    if not isinstance(shortcuts, list):
        errors.append("nonconforming_shortcuts must be a list when present")
        shortcuts = []

    audit = protocol.get("conformance_audit")
    if not isinstance(audit, dict):
        errors.append("conformance_audit must be an object")
        audit = {}
    expected = {identifier for identifier, _ in CHECKLIST}
    extra = sorted(set(audit) - expected)
    if extra:
        errors.append("unknown conformance_audit items: " + ", ".join(extra))
    normalized_audit: dict[str, dict[str, str]] = {}
    for identifier, _ in CHECKLIST:
        entry = audit.get(identifier)
        if not isinstance(entry, dict):
            errors.append(f"conformance_audit.{identifier} must be an object")
            continue
        status = entry.get("status")
        evidence = entry.get("evidence")
        if status not in STATUSES:
            errors.append(f"conformance_audit.{identifier}.status must be pass, fail, or not_applicable")
        if not isinstance(evidence, str) or not evidence.strip():
            errors.append(f"conformance_audit.{identifier}.evidence must be a nonempty line")
        if status in STATUSES and isinstance(evidence, str) and evidence.strip():
            normalized_audit[identifier] = {"status": status, "evidence": evidence.strip()}

    failed = [key for key, entry in normalized_audit.items() if entry["status"] == "fail"]
    not_applicable = [key for key, entry in normalized_audit.items()
                      if entry["status"] == "not_applicable"]
    if claim == "full" and (deviations or oracle_inputs or shortcuts or failed or not_applicable):
        errors.append("claim_level full requires every checklist item to pass and forbids deviations, oracle inputs, and shortcuts")
    if (shortcuts or failed) and not deviations:
        errors.append("shortcuts or failed checklist items require an explicit declared_deviations entry")
    if errors:
        raise ConformanceError(source + ":\n- " + "\n- ".join(errors))
    return {
        "method_reference": reference,
        "method_revision": revision,
        "claim_level": claim,
        "declared_deviations": deviations,
        "oracle_inputs": oracle_inputs,
        "nonconforming_shortcuts": shortcuts,
        "conformance_audit": normalized_audit,
    }


def render_validated_conformance_audit(declaration: Mapping[str, Any]) -> str:
    """Render a declaration that was validated when its run started.

    This deliberately does not compare the recorded revision with the living
    reference. Expensive runs pin the revision that was current at launch;
    advancing the reference while such a run is active must not prevent its
    historical report from being finalized. New runs must still enter through
    :func:`validate_protocol` before doing work.
    """
    required = (
        "method_reference", "method_revision", "claim_level",
        "declared_deviations", "oracle_inputs", "conformance_audit",
    )
    missing = [key for key in required if key not in declaration]
    if missing:
        raise ConformanceError(
            "validated declaration is incomplete: " + ", ".join(missing))
    audit = declaration["conformance_audit"]
    if not isinstance(audit, Mapping) or not audit:
        raise ConformanceError("validated declaration has no conformance audit")
    descriptions = dict(CHECKLIST)
    ordered = [identifier for identifier, _ in CHECKLIST if identifier in audit]
    ordered.extend(identifier for identifier in audit if identifier not in descriptions)
    lines = [
        "CONFORMANCE AUDIT",
        "=" * 72,
        f"Method revision: {declaration['method_revision']}",
        f"Claim level: {declaration['claim_level']}",
        "",
    ]
    for identifier in ordered:
        entry = audit[identifier]
        if not isinstance(entry, Mapping):
            raise ConformanceError(f"recorded audit item {identifier!r} is not an object")
        status = entry.get("status")
        evidence = entry.get("evidence")
        if status not in STATUSES or not isinstance(evidence, str) or not evidence.strip():
            raise ConformanceError(f"recorded audit item {identifier!r} is malformed")
        description = descriptions.get(
            identifier, "Requirement recorded by the revision-pinned protocol.")
        lines.append(f"[{entry['status'].upper()}] {identifier}")
        lines.append(f"  Requirement: {description}")
        lines.append(f"  Evidence: {entry['evidence']}")
    lines.extend(("", "Declared deviations:"))
    lines.extend(f"- {item}" for item in declaration["declared_deviations"] or ["None"])
    lines.append("Oracle inputs:")
    lines.extend(f"- {item}" for item in declaration["oracle_inputs"] or ["None"])
    lines.append("Nonconforming shortcuts:")
    lines.extend(f"- {item}" for item in declaration["nonconforming_shortcuts"] or ["None"])
    return "\n".join(lines) + "\n"


def render_conformance_audit(protocol: Mapping[str, Any]) -> str:
    declaration = validate_protocol(protocol)
    return render_validated_conformance_audit(declaration)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("check", "render"))
    parser.add_argument("--protocol", required=True, type=Path)
    args = parser.parse_args()
    try:
        protocol = _load(args.protocol)
        if args.command == "render":
            print(render_conformance_audit(protocol), end="")
        else:
            declaration = validate_protocol(protocol, source=str(args.protocol))
            print("CONFORMANCE_OK", declaration["method_revision"],
                  declaration["claim_level"], args.protocol)
    except ConformanceError as error:
        parser.exit(2, f"CONFORMANCE_ERROR\n{error}\n")


if __name__ == "__main__":
    main()
