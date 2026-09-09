#!/usr/bin/env python3
"""Repair contention-event semantics on the frozen 80000--80029 panel.

This is a read-only, development-only reanalysis.  It never creates an
EpisodeInstance and never executes a controller.  Instead, it authenticates
the original Pareto ledger and its deterministic-baseline expansion, then
reconstructs each VCG macro transition from the recorded exact frontiers.

The old ``relocations`` and ``obstructive_moves`` fields are retained in the
output only as explicitly labelled legacy aliases.  They must not be read as
a shared causal obstruction count:

* VCG records physical Reconfigure macros, which may be proactive; and
* the deterministic baselines record retrieval-executor obstruction
  clearances via ``retrieve_relocations``.

No sealed or prospective panel is accepted by this tool.
"""

from __future__ import annotations

import argparse
import csv
from hashlib import sha256
import json
import math
from pathlib import Path
from statistics import fmean
from typing import Mapping, Optional, Sequence
import uuid

from contention_metrics import (
    CONTENTION_METRIC_SCHEMA_VERSION,
    contention_metric_record,
)


PROTOCOL = "vcg_contention_metric_reanalysis_development_v2"
PROTOCOL_SCHEMA_VERSION = 2
SOURCE_PROTOCOL = "vcg_dense_v1_1_timing_relocation_pareto_development_v1"
EXPANDED_PROTOCOL = "vcg_dense_expanded_online_baselines_development_v1"
EVALUATION_SEEDS = tuple(range(80_000, 80_030))
SCOPE = "development_only_frozen_80000_80029_no_rerollout"

SOURCE_REPORT = "pareto-report.json"
SOURCE_RUNS = "pareto-runs.csv"
SOURCE_AUDIT = "pareto-audit.json"
SOURCE_INSTANCE_MANIFEST = "instance-manifest.json"
SOURCE_PROTOCOL_MANIFEST = "protocol-manifest.json"
EXPANDED_REPORT = "expanded-baseline-report.json"
EXPANDED_AUDIT = "expanded-baseline-audit.json"
EXPANDED_PROTOCOL_MANIFEST = "expanded-baseline-protocol-manifest.json"

RUNS_FILENAME = "contention-metrics-runs.csv"
REPORT_FILENAME = "contention-metrics-report.json"
AUDIT_FILENAME = "contention-metrics-audit.json"

VCG_SELECTED_GROUP = "vcg_dense_v1_1_selected_3seed"
VCG_FINAL_GROUP = "vcg_dense_v1_1_episode500_diagnostic_3seed"

EXPECTED_METHOD_IDS = (
    "duration_aware_nearest_free",
    "duration_aware_dynamic_pslap",
    "duration_aware_pslap_ga_2009_rolling",
    "duration_aware_enhanced_complete_rolling_ga",
    "duration_aware_pslap_ga_duration_aware_rolling",
    "duration_aware_pslap_ga_operational_rolling",
    "vcg_dense_seed0_selected_best",
    "vcg_dense_seed1_selected_best",
    "vcg_dense_seed2_selected_best",
    "vcg_dense_seed0_episode500_final",
    "vcg_dense_seed1_episode500_final",
    "vcg_dense_seed2_episode500_final",
)

DIRECT_SELF = "directly_deliverable_self_reconfigurations"
OTHER_WITH_DIRECT = (
    "other_reconfigurations_while_any_direct_delivery_available"
)
NO_DIRECT = "no_direct_delivery_recovery_reconfigurations"
VCG_CATEGORIES = (DIRECT_SELF, OTHER_WITH_DIRECT, NO_DIRECT)

RUN_FIELDS = (
    "protocol",
    "scope",
    "source_protocol",
    "source_row_origin",
    "source_ledger_run_key",
    "source_ledger_sha256",
    "method_id",
    "method_group",
    "system_family",
    "policy_group",
    "model_seed",
    "instance_seed",
    "instance_id",
    "schedule_id",
    "strict_method_success",
    "completion_rate",
    "delivery_count",
    "method_failure_reason",
    "contention_metric_schema_version",
    "contention_metric_schema",
    "macro_decisions",
    "exact_frontier_count",
    "frontiers_with_any_direct_delivery",
    "frontiers_with_no_direct_delivery",
    "frontiers_with_no_direct_delivery_rate",
    "reconstructed_semantic_transitions",
    "equivalent_destination_alias_transitions",
    "physical_storage_relocations",
    "target_bound_obstruction_clearances",
    "standalone_reconfigurations",
    "standalone_with_direct_delivery_available",
    "standalone_without_direct_delivery_available",
    "total_physical_reconfigurations",
    DIRECT_SELF,
    OTHER_WITH_DIRECT,
    NO_DIRECT,
    "retrieval_executor_obstruction_clearances",
    "physical_storage_relocations_per_100_deliveries",
    "target_bound_obstruction_clearances_per_100_deliveries",
    "standalone_reconfigurations_per_100_deliveries",
    "standalone_with_direct_delivery_available_per_100_deliveries",
    "standalone_without_direct_delivery_available_per_100_deliveries",
    "total_physical_reconfigurations_per_100_deliveries",
    "directly_deliverable_self_reconfigurations_per_100_deliveries",
    "other_reconfigurations_while_any_direct_delivery_available_per_100_deliveries",
    "no_direct_delivery_recovery_reconfigurations_per_100_deliveries",
    "retrieval_executor_obstruction_clearances_per_100_deliveries",
    "relocations",
    "obstructive_moves",
    "relocations_per_100_deliveries",
    "legacy_alias_only",
    "legacy_obstructive_moves_semantics",
    "category_sum_matches_total",
    "reconstructed_total_matches_legacy_relocations",
    "retrieval_clearances_match_source_counter",
    "transition_count_matches_macro_decisions",
    "delivery_count_matches_reconstruction",
    "invariants_passed",
)


class ContentionAuditError(ValueError):
    """Raised when provenance or reconstruction fails closed."""


def _json_safe(value):
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _canonical_json(value) -> str:
    return json.dumps(
        _json_safe(value),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ContentionAuditError(f"{path} must contain a JSON object")
    return value


def _require_equal(label: str, observed, expected) -> None:
    if _json_safe(observed) != _json_safe(expected):
        raise ContentionAuditError(
            f"{label} mismatch: observed={observed!r}, expected={expected!r}"
        )


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.tmp-{uuid.uuid4().hex}"
    try:
        temporary.write_text(value, encoding="utf-8")
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_json(path: Path, value) -> None:
    _atomic_text(
        path,
        json.dumps(_json_safe(value), indent=2, sort_keys=True, allow_nan=False)
        + "\n",
    )


def _csv_value(value):
    if value is None:
        return ""
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (dict, list, tuple)):
        return _canonical_json(value)
    return value


def _write_csv(path: Path, rows: Sequence[Mapping]) -> None:
    lines = []
    from io import StringIO

    buffer = StringIO()
    writer = csv.DictWriter(buffer, fieldnames=RUN_FIELDS, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({field: _csv_value(row.get(field)) for field in RUN_FIELDS})
    lines.append(buffer.getvalue())
    _atomic_text(path, "".join(lines))


def _authenticate_ledger_manifest(
    entries: Sequence[Mapping],
    *,
    allowed_root: Path,
) -> tuple[dict[str, dict], tuple[dict, ...]]:
    records: dict[str, dict] = {}
    authenticated = []
    for entry in entries:
        key = str(entry.get("run_key"))
        if not key or key in records:
            raise ContentionAuditError(f"invalid or duplicate ledger key: {key}")
        path = Path(str(entry.get("path"))).resolve()
        if allowed_root != path and allowed_root not in path.parents:
            raise ContentionAuditError(f"ledger escapes authenticated root: {path}")
        observed_hash = _sha256_file(path)
        _require_equal("ledger sha256", observed_hash, entry.get("sha256"))
        payload = _load_json(path)
        _require_equal("ledger run key", payload.get("run_key"), key)
        if not isinstance(payload.get("run"), Mapping):
            raise ContentionAuditError(f"ledger has no run object: {path}")
        if not isinstance(payload.get("method_audit"), Mapping):
            raise ContentionAuditError(f"ledger has no method audit: {path}")
        records[key] = payload
        authenticated.append(
            {
                "run_key": key,
                "path": str(path),
                "sha256": observed_hash,
                "input_fingerprint": entry.get("input_fingerprint"),
            }
        )
    return records, tuple(authenticated)


def authenticate_inputs(
    source_dir: Path,
    expanded_dir: Path,
) -> tuple[list[dict], dict[str, dict], dict]:
    """Authenticate both frozen-panel artifact chains and return analysis rows."""

    source_dir = source_dir.resolve()
    expanded_dir = expanded_dir.resolve()
    source_paths = {
        SOURCE_REPORT: source_dir / SOURCE_REPORT,
        SOURCE_RUNS: source_dir / SOURCE_RUNS,
        SOURCE_AUDIT: source_dir / SOURCE_AUDIT,
        SOURCE_INSTANCE_MANIFEST: source_dir / SOURCE_INSTANCE_MANIFEST,
        SOURCE_PROTOCOL_MANIFEST: source_dir / SOURCE_PROTOCOL_MANIFEST,
    }
    expanded_paths = {
        EXPANDED_REPORT: expanded_dir / EXPANDED_REPORT,
        EXPANDED_AUDIT: expanded_dir / EXPANDED_AUDIT,
        EXPANDED_PROTOCOL_MANIFEST: expanded_dir / EXPANDED_PROTOCOL_MANIFEST,
    }
    source_hashes = {name: _sha256_file(path) for name, path in source_paths.items()}
    expanded_hashes = {
        name: _sha256_file(path) for name, path in expanded_paths.items()
    }

    source_report = _load_json(source_paths[SOURCE_REPORT])
    source_audit = _load_json(source_paths[SOURCE_AUDIT])
    instance_manifest = _load_json(source_paths[SOURCE_INSTANCE_MANIFEST])
    source_protocol_manifest = _load_json(source_paths[SOURCE_PROTOCOL_MANIFEST])
    expanded_report = _load_json(expanded_paths[EXPANDED_REPORT])
    expanded_audit = _load_json(expanded_paths[EXPANDED_AUDIT])
    expanded_protocol_manifest = _load_json(
        expanded_paths[EXPANDED_PROTOCOL_MANIFEST]
    )

    _require_equal(
        "source report protocol",
        source_report.get("runner_protocol"),
        SOURCE_PROTOCOL,
    )
    _require_equal("source audit protocol", source_audit.get("protocol"), SOURCE_PROTOCOL)
    _require_equal(
        "source protocol manifest protocol",
        source_protocol_manifest.get("protocol"),
        SOURCE_PROTOCOL,
    )
    _require_equal(
        "expanded report protocol", expanded_report.get("protocol"), EXPANDED_PROTOCOL
    )
    _require_equal(
        "expanded audit protocol", expanded_audit.get("protocol"), EXPANDED_PROTOCOL
    )
    _require_equal(
        "expanded protocol manifest protocol",
        expanded_protocol_manifest.get("protocol"),
        EXPANDED_PROTOCOL,
    )
    for artifact in (source_report, expanded_report, expanded_protocol_manifest):
        _require_equal(
            "development-only source scope",
            artifact.get("scope"),
            "development_only_sealed_panels_unopened",
        )
        if bool(artifact.get("performance_claim_authorized")):
            raise ContentionAuditError("source unexpectedly authorizes a performance claim")

    expected_seeds = list(EVALUATION_SEEDS)
    _require_equal(
        "source evaluation seeds",
        source_protocol_manifest.get("evaluation_seeds"),
        expected_seeds,
    )
    _require_equal(
        "expanded evaluation seeds",
        expanded_protocol_manifest.get("evaluation_seeds"),
        expected_seeds,
    )
    _require_equal(
        "instance-manifest seeds",
        instance_manifest.get("contract", {}).get("seeds"),
        expected_seeds,
    )
    _require_equal(
        "source audit instance manifest",
        source_audit.get("instance_manifest"),
        instance_manifest,
    )
    _require_equal(
        "expanded audit instance manifest",
        expanded_audit.get("source_instance_manifest"),
        instance_manifest,
    )
    _require_equal(
        "expanded source audit hash",
        expanded_audit.get("source_audit_sha256"),
        source_hashes[SOURCE_AUDIT],
    )
    _require_equal(
        "expanded source artifact hashes",
        expanded_protocol_manifest.get("source_artifact_sha256"),
        {
            "instance-manifest.json": source_hashes[SOURCE_INSTANCE_MANIFEST],
            "pareto-audit.json": source_hashes[SOURCE_AUDIT],
            "pareto-report.json": source_hashes[SOURCE_REPORT],
            "pareto-runs.csv": source_hashes[SOURCE_RUNS],
            "protocol-manifest.json": source_hashes[SOURCE_PROTOCOL_MANIFEST],
        },
    )

    # Verify every immutable EpisodeInstance byte hash without loading the
    # simulator.  This proves that no schedule was regenerated for reanalysis.
    authenticated_instances = []
    for seed in EVALUATION_SEEDS:
        record = instance_manifest.get("instances", {}).get(str(seed))
        if not isinstance(record, Mapping):
            raise ContentionAuditError(f"missing instance-manifest seed {seed}")
        path = Path(str(record.get("path"))).resolve()
        expected = (source_dir / "instances" / f"seed-{seed}.json").resolve()
        _require_equal("instance path", path, expected)
        observed_hash = _sha256_file(path)
        _require_equal("instance sha256", observed_hash, record.get("sha256"))
        authenticated_instances.append(
            {
                "seed": seed,
                "instance_id": record.get("instance_id"),
                "schedule_id": record.get("schedule_id"),
                "path": str(path),
                "sha256": observed_hash,
            }
        )

    original_ledgers, original_manifest = _authenticate_ledger_manifest(
        source_audit.get("run_ledger_manifest", []), allowed_root=source_dir
    )
    expanded_ledgers, expanded_manifest = _authenticate_ledger_manifest(
        expanded_audit.get("new_run_ledger_manifest", []),
        allowed_root=expanded_dir,
    )
    if len(original_ledgers) != 270:
        raise ContentionAuditError("source must contain 270 authenticated physical ledgers")
    if len(expanded_ledgers) != 60:
        raise ContentionAuditError("expansion must contain 60 authenticated physical ledgers")
    ledgers = {**original_ledgers, **expanded_ledgers}
    if len(ledgers) != len(original_ledgers) + len(expanded_ledgers):
        raise ContentionAuditError("ledger keys collide across source artifact chains")

    rows = expanded_report.get("runs")
    if not isinstance(rows, list) or len(rows) != 360:
        raise ContentionAuditError("expanded report must expose exactly 360 analysis rows")
    observed_grid = set()
    method_counts: dict[str, int] = {}
    for row in rows:
        seed = int(row.get("instance_seed"))
        if seed not in EVALUATION_SEEDS:
            raise ContentionAuditError(f"analysis row escapes frozen panel: {seed}")
        method = str(row.get("method_id"))
        method_counts[method] = method_counts.get(method, 0) + 1
        key = (method, seed)
        if key in observed_grid:
            raise ContentionAuditError(f"duplicate method/seed analysis row: {key}")
        observed_grid.add(key)
        manifest_record = instance_manifest["instances"][str(seed)]
        _require_equal("row instance id", row.get("instance_id"), manifest_record["instance_id"])
        _require_equal("row schedule id", row.get("schedule_id"), manifest_record["schedule_id"])
    _require_equal(
        "expanded method grid",
        method_counts,
        {method: len(EVALUATION_SEEDS) for method in EXPECTED_METHOD_IDS},
    )

    provenance = {
        "source_artifact_sha256": source_hashes,
        "expanded_artifact_sha256": expanded_hashes,
        "authenticated_instances": authenticated_instances,
        "authenticated_original_run_ledgers": original_manifest,
        "authenticated_expanded_run_ledgers": expanded_manifest,
        "source_row_count": len(source_report.get("runs", [])),
        "expanded_analysis_row_count": len(rows),
    }
    return [dict(row) for row in rows], ledgers, provenance


def _successor_matches(frontier: Mapping, next_frontier: Mapping) -> list[dict]:
    expected = _canonical_json(next_frontier.get("current_certificate"))
    return [
        dict(candidate)
        for candidate in frontier.get("candidates", [])
        if _canonical_json(candidate.get("certificate")) == expected
    ]


def reconstruct_vcg_contention(
    run: Mapping,
    method_audit: Mapping,
) -> dict:
    """Reconstruct the VCG semantic transition sequence and causal classes.

    Successor certificates identify the selected action/target semantic.  A
    direct delivery can have multiple equivalent exit destinations; those
    aliases are counted but are never reconfiguration ambiguities.  The final
    transition is identified by successful terminal completion and the sole
    directly deliverable target at the final frontier.
    """

    frontiers = method_audit.get("frontiers")
    if not isinstance(frontiers, list) or not frontiers:
        raise ContentionAuditError("VCG ledger has no exact frontier sequence")
    if not bool(method_audit.get("complete_frontier_exactly_verified")):
        raise ContentionAuditError("VCG frontier sequence is not exactly verified")
    if not bool(method_audit.get("exact_verifier_authoritative")):
        raise ContentionAuditError("VCG exact verifier is not authoritative")

    transitions = []
    equivalent_destination_aliases = 0
    categories = {name: 0 for name in VCG_CATEGORIES}
    reconfiguration_events = []
    frontiers_with_any_direct = sum(
        any(
            candidate.get("action_type") == "deliver"
            for candidate in frontier.get("candidates", [])
        )
        for frontier in frontiers
    )
    frontiers_with_no_direct = len(frontiers) - frontiers_with_any_direct

    for index, (frontier, next_frontier) in enumerate(zip(frontiers, frontiers[1:])):
        if int(next_frontier["decision_epoch"]) <= int(frontier["decision_epoch"]):
            raise ContentionAuditError("VCG decision epochs are not strictly increasing")
        matches = _successor_matches(frontier, next_frontier)
        if not matches:
            raise ContentionAuditError(
                f"frontier {index} has no candidate matching its recorded successor"
            )
        semantics = {
            (str(candidate.get("action_type")), candidate.get("target_label"))
            for candidate in matches
        }
        if len(semantics) != 1:
            raise ContentionAuditError(
                f"frontier {index} successor has ambiguous action semantics: {semantics}"
            )
        action_type, target_label = next(iter(semantics))
        if len(matches) > 1:
            if action_type != "deliver":
                raise ContentionAuditError(
                    "only direct-delivery exit aliases may share a successor certificate"
                )
            equivalent_destination_aliases += 1
        chosen = sorted(matches, key=lambda item: str(item.get("key")))[0]
        transitions.append(
            {
                "frontier_index": index,
                "decision_epoch": int(frontier["decision_epoch"]),
                "action_type": action_type,
                "target_label": target_label,
                "selected_key": (
                    chosen.get("key") if len(matches) == 1 else None
                ),
                "equivalent_candidate_keys": [item.get("key") for item in matches],
                "inference": "successor_certificate_identity",
            }
        )
        if action_type == "reconfigure":
            if len(matches) != 1:
                raise ContentionAuditError("reconfiguration transition is not lossless")
            direct = [
                candidate
                for candidate in frontier.get("candidates", [])
                if candidate.get("action_type") == "deliver"
            ]
            if any(item.get("target_label") == target_label for item in direct):
                category = DIRECT_SELF
            elif direct:
                category = OTHER_WITH_DIRECT
            else:
                category = NO_DIRECT
            categories[category] += 1
            reconfiguration_events.append(
                {
                    "frontier_index": index,
                    "decision_epoch": int(frontier["decision_epoch"]),
                    "selected_key": chosen.get("key"),
                    "target_label": target_label,
                    "source": chosen.get("source"),
                    "destination": chosen.get("destination"),
                    "classification": category,
                    "direct_delivery_candidate_count": len(direct),
                    "directly_deliverable_labels": sorted(
                        {
                            str(item.get("target_label"))
                            for item in direct
                            if item.get("target_label") is not None
                        }
                    ),
                }
            )

    # A successful, complete VCG episode ends immediately after its last
    # delivery, so no successor frontier is recorded for that macro.  Require
    # that exactly one target is directly deliverable at the final frontier.
    if float(run.get("strict_method_success", 0.0)) != 1.0 or float(
        run.get("completion_rate", 0.0)
    ) != 1.0:
        raise ContentionAuditError(
            "terminal-transition reconstruction requires a strict, complete VCG run"
        )
    final_frontier = frontiers[-1]
    final_deliveries = [
        candidate
        for candidate in final_frontier.get("candidates", [])
        if candidate.get("action_type") == "deliver"
    ]
    final_targets = {
        candidate.get("target_label") for candidate in final_deliveries
    }
    if not final_deliveries or len(final_targets) != 1:
        raise ContentionAuditError(
            "terminal frontier does not identify exactly one deliverable target"
        )
    final_target = next(iter(final_targets))
    transitions.append(
        {
            "frontier_index": len(frontiers) - 1,
            "decision_epoch": int(final_frontier["decision_epoch"]),
            "action_type": "deliver",
            "target_label": final_target,
            "selected_key": None,
            "equivalent_candidate_keys": [
                item.get("key") for item in final_deliveries
            ],
            "inference": "strict_terminal_completion_constraint",
        }
    )
    if len(final_deliveries) > 1:
        equivalent_destination_aliases += 1

    action_counts: dict[str, int] = {}
    for transition in transitions:
        action = str(transition["action_type"])
        action_counts[action] = action_counts.get(action, 0) + 1
    stored_total = int(run.get("relocations", -1))
    category_total = sum(categories.values())
    invariants = {
        "category_sum_matches_total": category_total == action_counts.get("reconfigure", 0),
        "reconstructed_total_matches_legacy_relocations": category_total == stored_total,
        "transition_count_matches_macro_decisions": len(transitions)
        == int(method_audit.get("macro_decisions", -1)),
        "delivery_count_matches_reconstruction": action_counts.get("deliver", 0)
        == int(run.get("delivery_count", -1)),
    }
    if not all(invariants.values()):
        raise ContentionAuditError(f"VCG reconstruction invariant failed: {invariants}")
    return {
        "macro_decisions": int(method_audit["macro_decisions"]),
        "exact_frontier_count": len(frontiers),
        "frontiers_with_any_direct_delivery": frontiers_with_any_direct,
        "frontiers_with_no_direct_delivery": frontiers_with_no_direct,
        "reconstructed_semantic_transitions": len(transitions),
        "equivalent_destination_alias_transitions": equivalent_destination_aliases,
        "action_counts": action_counts,
        "total_physical_reconfigurations": category_total,
        **categories,
        "reconfiguration_events": reconfiguration_events,
        "transitions": transitions,
        **invariants,
        "invariants_passed": True,
    }


def preserve_baseline_contention(run: Mapping, method_audit: Mapping) -> dict:
    """Preserve, but do not causally reinterpret, the baseline counter."""

    clearances = int(method_audit.get("retrieve_relocations", -1))
    legacy_relocations = int(run.get("relocations", -2))
    legacy_obstructive = int(run.get("obstructive_moves", -3))
    matches = clearances == legacy_relocations == legacy_obstructive
    if clearances < 0 or not matches:
        raise ContentionAuditError(
            "baseline retrieval-clearance counter does not match stored aliases"
        )
    return {
        # Under the shared simulator contract, every retrieve_relocations
        # increment is one executed storage-to-storage block move.  That makes
        # the physical event total comparable, while its baseline-specific
        # obstruction-clearance cause must remain separately labelled.
        "total_physical_reconfigurations": clearances,
        "retrieval_executor_obstruction_clearances": clearances,
        "retrieval_clearances_match_source_counter": matches,
        "invariants_passed": matches,
    }


def _method_group(row: Mapping) -> str:
    if row.get("policy_group") == "selected_best":
        return VCG_SELECTED_GROUP
    if row.get("policy_group") == "episode500_final_diagnostic":
        return VCG_FINAL_GROUP
    return str(row.get("method_id"))


def _per_100(value: Optional[int], deliveries: int) -> Optional[float]:
    if value is None or deliveries <= 0:
        return None
    return 100.0 * float(value) / float(deliveries)


def _physical_ledger_key(row: Mapping, ledgers: Mapping[str, Mapping]) -> str:
    method = str(row["method_id"])
    seed = int(row["instance_seed"])
    key = f"{method}:{seed}"
    if key in ledgers:
        return key
    derived = row.get("derived_from_method_id")
    if derived:
        derived_key = f"{derived}:{seed}"
        if derived_key in ledgers:
            return derived_key
    raise ContentionAuditError(f"analysis row has no authenticated physical ledger: {key}")


def repair_row(row: Mapping, ledgers: Mapping[str, Mapping]) -> tuple[dict, dict]:
    key = _physical_ledger_key(row, ledgers)
    ledger = ledgers[key]
    source_run = ledger["run"]
    method_audit = ledger["method_audit"]
    for field in (
        "instance_id",
        "schedule_id",
        "strict_method_success",
        "completion_rate",
        "delivery_count",
        "relocations",
        "obstructive_moves",
        "relocations_per_100_deliveries",
    ):
        _require_equal(f"analysis/ledger {field}", row.get(field), source_run.get(field))
    deliveries = int(row.get("delivery_count", 0))
    is_vcg = "frontiers" in method_audit

    if is_vcg:
        detail = reconstruct_vcg_contention(source_run, method_audit)
        clearances = None
        schema = "vcg_physical_reconfiguration_causal_decomposition_v2"
        legacy_semantics = (
            "legacy_alias_of_total_physical_reconfigurations;"
            "not_a_causal_obstruction_count"
        )
        retrieval_match = None
    else:
        detail = preserve_baseline_contention(source_run, method_audit)
        clearances = detail["retrieval_executor_obstruction_clearances"]
        schema = "baseline_retrieval_executor_obstruction_clearances_v1"
        legacy_semantics = (
            "legacy_alias_of_retrieval_executor_obstruction_clearances;"
            "proactive_categories_not_observed"
        )
        retrieval_match = detail["retrieval_clearances_match_source_counter"]

    total = detail.get("total_physical_reconfigurations")
    if is_vcg:
        canonical_contention = contention_metric_record(
            physical_storage_relocations=total,
            target_bound_obstruction_clearances=0,
            standalone_reconfigurations=total,
            standalone_with_direct_delivery_available=(
                int(detail[DIRECT_SELF]) + int(detail[OTHER_WITH_DIRECT])
            ),
            standalone_without_direct_delivery_available=int(
                detail[NO_DIRECT]
            ),
            directly_deliverable_self_reconfigurations=int(
                detail[DIRECT_SELF]
            ),
        )
    else:
        canonical_contention = contention_metric_record(
            physical_storage_relocations=total,
            target_bound_obstruction_clearances=clearances,
            standalone_reconfigurations=0,
            standalone_with_direct_delivery_available=0,
            standalone_without_direct_delivery_available=0,
            directly_deliverable_self_reconfigurations=0,
        )
    repaired = {
        "protocol": PROTOCOL,
        "scope": SCOPE,
        "source_protocol": row.get("expanded_source_protocol")
        or row.get("protocol"),
        "source_row_origin": row.get("expanded_row_origin"),
        "source_ledger_run_key": key,
        "source_ledger_sha256": None,
        "method_id": row.get("method_id"),
        "method_group": _method_group(row),
        "system_family": row.get("system_family"),
        "policy_group": row.get("policy_group"),
        "model_seed": row.get("model_seed"),
        "instance_seed": int(row["instance_seed"]),
        "instance_id": row.get("instance_id"),
        "schedule_id": row.get("schedule_id"),
        "strict_method_success": row.get("strict_method_success"),
        "completion_rate": row.get("completion_rate"),
        "delivery_count": deliveries,
        "method_failure_reason": row.get("method_failure_reason"),
        **canonical_contention,
        "contention_metric_schema": schema,
        "macro_decisions": detail.get("macro_decisions"),
        "exact_frontier_count": detail.get("exact_frontier_count"),
        "frontiers_with_any_direct_delivery": detail.get(
            "frontiers_with_any_direct_delivery"
        ),
        "frontiers_with_no_direct_delivery": detail.get(
            "frontiers_with_no_direct_delivery"
        ),
        "frontiers_with_no_direct_delivery_rate": (
            None
            if detail.get("exact_frontier_count") in (None, 0)
            else float(detail["frontiers_with_no_direct_delivery"])
            / float(detail["exact_frontier_count"])
        ),
        "reconstructed_semantic_transitions": detail.get(
            "reconstructed_semantic_transitions"
        ),
        "equivalent_destination_alias_transitions": detail.get(
            "equivalent_destination_alias_transitions"
        ),
        "total_physical_reconfigurations": total,
        **{
            name: (
                canonical_contention[name]
                if name == DIRECT_SELF
                else detail.get(name)
            )
            for name in VCG_CATEGORIES
        },
        "retrieval_executor_obstruction_clearances": clearances,
        "physical_storage_relocations_per_100_deliveries": _per_100(
            canonical_contention["physical_storage_relocations"], deliveries
        ),
        "target_bound_obstruction_clearances_per_100_deliveries": _per_100(
            canonical_contention["target_bound_obstruction_clearances"],
            deliveries,
        ),
        "standalone_reconfigurations_per_100_deliveries": _per_100(
            canonical_contention["standalone_reconfigurations"], deliveries
        ),
        "standalone_with_direct_delivery_available_per_100_deliveries": _per_100(
            canonical_contention[
                "standalone_with_direct_delivery_available"
            ],
            deliveries,
        ),
        "standalone_without_direct_delivery_available_per_100_deliveries": _per_100(
            canonical_contention[
                "standalone_without_direct_delivery_available"
            ],
            deliveries,
        ),
        "total_physical_reconfigurations_per_100_deliveries": _per_100(
            total, deliveries
        ),
        "directly_deliverable_self_reconfigurations_per_100_deliveries": _per_100(
            detail.get(DIRECT_SELF), deliveries
        ),
        "other_reconfigurations_while_any_direct_delivery_available_per_100_deliveries": _per_100(
            detail.get(OTHER_WITH_DIRECT), deliveries
        ),
        "no_direct_delivery_recovery_reconfigurations_per_100_deliveries": _per_100(
            detail.get(NO_DIRECT), deliveries
        ),
        "retrieval_executor_obstruction_clearances_per_100_deliveries": _per_100(
            clearances, deliveries
        ),
        # Retained verbatim for backwards compatibility, but never promoted as
        # a repaired cross-family causal metric.
        "relocations": int(row.get("relocations", 0)),
        "obstructive_moves": int(row.get("obstructive_moves", 0)),
        "relocations_per_100_deliveries": row.get(
            "relocations_per_100_deliveries"
        ),
        "legacy_alias_only": True,
        "legacy_obstructive_moves_semantics": legacy_semantics,
        "category_sum_matches_total": detail.get("category_sum_matches_total"),
        "reconstructed_total_matches_legacy_relocations": detail.get(
            "reconstructed_total_matches_legacy_relocations"
        ),
        "retrieval_clearances_match_source_counter": retrieval_match,
        "transition_count_matches_macro_decisions": detail.get(
            "transition_count_matches_macro_decisions"
        ),
        "delivery_count_matches_reconstruction": detail.get(
            "delivery_count_matches_reconstruction"
        ),
        "invariants_passed": bool(detail["invariants_passed"]),
    }
    audit = {
        "method_id": repaired["method_id"],
        "method_group": repaired["method_group"],
        "instance_seed": repaired["instance_seed"],
        "instance_id": repaired["instance_id"],
        "source_ledger_run_key": key,
        "contention_metric_schema": schema,
        "canonical_contention_metrics": canonical_contention,
        "legacy_aliases": {
            "relocations": repaired["relocations"],
            "obstructive_moves": repaired["obstructive_moves"],
            "relocations_per_100_deliveries": repaired[
                "relocations_per_100_deliveries"
            ],
            "semantics": legacy_semantics,
        },
        "reconstruction": detail,
    }
    return repaired, audit


def _sum_optional(rows: Sequence[Mapping], field: str) -> Optional[int]:
    values = [row.get(field) for row in rows]
    if all(value is None for value in values):
        return None
    if any(value is None for value in values):
        raise ContentionAuditError(f"mixed metric schema within group for {field}")
    return sum(int(value) for value in values)


def build_report(rows: Sequence[Mapping]) -> dict:
    groups: dict[str, list[Mapping]] = {}
    for row in rows:
        groups.setdefault(str(row["method_group"]), []).append(row)
    summaries = []
    for group, selected in sorted(groups.items()):
        deliveries = sum(int(row["delivery_count"]) for row in selected)
        total = _sum_optional(selected, "total_physical_reconfigurations")
        clearances = _sum_optional(
            selected, "retrieval_executor_obstruction_clearances"
        )
        frontier_count = _sum_optional(selected, "exact_frontier_count")
        exposed_frontiers = _sum_optional(
            selected, "frontiers_with_no_direct_delivery"
        )
        category_totals = {
            name: _sum_optional(selected, name) for name in VCG_CATEGORIES
        }
        canonical_totals = {
            name: _sum_optional(selected, name)
            for name in (
                "physical_storage_relocations",
                "target_bound_obstruction_clearances",
                "standalone_reconfigurations",
                "standalone_with_direct_delivery_available",
                "standalone_without_direct_delivery_available",
                "directly_deliverable_self_reconfigurations",
            )
        }
        legacy_rates = []
        for row in selected:
            value = row.get("relocations_per_100_deliveries")
            try:
                number = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(number):
                legacy_rates.append(number)
        strict_full = all(
            float(row["strict_method_success"]) == 1.0
            and float(row["completion_rate"]) == 1.0
            for row in selected
        )
        conditional_rates = {
            **{
                f"{name}_per_100_completed_deliveries": _per_100(
                    value, deliveries
                )
                for name, value in canonical_totals.items()
            },
            "total_physical_reconfigurations_per_100_completed_deliveries": _per_100(
                total, deliveries
            ),
            "retrieval_executor_obstruction_clearances_per_100_completed_deliveries": _per_100(
                clearances, deliveries
            ),
            **{
                f"{name}_per_100_completed_deliveries": _per_100(
                    value, deliveries
                )
                for name, value in category_totals.items()
            },
        }
        summaries.append(
            {
                "method_group": group,
                "contention_metric_schema_version": (
                    CONTENTION_METRIC_SCHEMA_VERSION
                ),
                "contention_metric_schema": selected[0][
                    "contention_metric_schema"
                ],
                "analysis_rows": len(selected),
                "model_seeds": sorted(
                    {
                        int(row["model_seed"])
                        for row in selected
                        if row.get("model_seed") not in (None, "")
                    }
                ),
                "strict_full_panel": strict_full,
                "numeric_comparison_eligible": strict_full,
                "failed_or_incomplete_rows": [
                    {
                        "method_id": row.get("method_id"),
                        "instance_seed": int(row["instance_seed"]),
                        "strict_method_success": row["strict_method_success"],
                        "completion_rate": row["completion_rate"],
                        "delivery_count": int(row["delivery_count"]),
                        "method_failure_reason": row.get(
                            "method_failure_reason"
                        ),
                    }
                    for row in selected
                    if float(row["strict_method_success"]) != 1.0
                    or float(row["completion_rate"]) != 1.0
                ],
                "total_deliveries": deliveries,
                "exact_frontier_count": frontier_count,
                "frontiers_with_no_direct_delivery": exposed_frontiers,
                "frontiers_with_no_direct_delivery_rate": (
                    None
                    if frontier_count in (None, 0)
                    else float(exposed_frontiers) / float(frontier_count)
                ),
                "total_physical_reconfigurations": total,
                **canonical_totals,
                **category_totals,
                "retrieval_executor_obstruction_clearances": clearances,
                "directly_deliverable_self_share_of_physical_reconfigurations": (
                    None
                    if total in (None, 0) or category_totals[DIRECT_SELF] is None
                    else float(category_totals[DIRECT_SELF]) / float(total)
                ),
                "other_with_direct_share_of_physical_reconfigurations": (
                    None
                    if total in (None, 0)
                    or category_totals[OTHER_WITH_DIRECT] is None
                    else float(category_totals[OTHER_WITH_DIRECT]) / float(total)
                ),
                "no_direct_recovery_share_of_physical_reconfigurations": (
                    None
                    if total in (None, 0) or category_totals[NO_DIRECT] is None
                    else float(category_totals[NO_DIRECT]) / float(total)
                ),
                "no_direct_frontier_reconfiguration_rate": (
                    None
                    if exposed_frontiers in (None, 0)
                    or category_totals[NO_DIRECT] is None
                    else float(category_totals[NO_DIRECT])
                    / float(exposed_frontiers)
                ),
                **{
                    f"{name}_per_100_deliveries": (
                        _per_100(value, deliveries) if strict_full else None
                    )
                    for name, value in canonical_totals.items()
                },
                "total_physical_reconfigurations_per_100_deliveries": _per_100(
                    total, deliveries
                ) if strict_full else None,
                **{
                    f"{name}_per_100_deliveries": (
                        _per_100(value, deliveries) if strict_full else None
                    )
                    for name, value in category_totals.items()
                },
                "retrieval_executor_obstruction_clearances_per_100_deliveries": (
                    _per_100(clearances, deliveries) if strict_full else None
                ),
                "conditional_on_completed_deliveries_diagnostic": (
                    None if strict_full else conditional_rates
                ),
                "legacy_mean_relocations_per_100_deliveries": (
                    float(fmean(legacy_rates)) if legacy_rates else None
                ),
                "invariants_passed": all(
                    bool(row["invariants_passed"]) for row in selected
                ),
            }
        )
    return {
        "protocol": PROTOCOL,
        "protocol_schema_version": PROTOCOL_SCHEMA_VERSION,
        "scope": SCOPE,
        "performance_claim_authorized": False,
        "rollouts_executed": 0,
        "new_episode_instances_generated": 0,
        "sealed_or_prospective_panels_opened": False,
        "canonical_contention_metric_schema_version": (
            CONTENTION_METRIC_SCHEMA_VERSION
        ),
        "metric_definitions": {
            "physical_storage_relocations": (
                "all executed storage-to-storage block moves; the canonical "
                "cross-family physical contention burden"
            ),
            "target_bound_obstruction_clearances": (
                "physical relocations performed inside a retrieval executor to "
                "clear the path for its bound delivery target"
            ),
            "standalone_reconfigurations": (
                "explicit standalone Reconfigure macros; partitioned by whether "
                "any exact-safe direct delivery was available"
            ),
            "standalone_with_direct_delivery_available": (
                "standalone reconfigurations at frontiers with at least one "
                "exact-safe direct-delivery candidate"
            ),
            "standalone_without_direct_delivery_available": (
                "standalone reconfigurations at frontiers with zero exact-safe "
                "direct-delivery candidates"
            ),
            "total_physical_reconfigurations": (
                "executed storage-to-storage block moves: VCG Reconfigure macros "
                "reconstructed from exact consecutive frontiers, and baseline "
                "retrieve relocations authenticated against their event aliases"
            ),
            DIRECT_SELF: (
                "VCG reconfigures block b while an exact-safe direct delivery "
                "candidate for that same block b exists at the decision frontier; "
                "strong evidence of proactive reconfiguration"
            ),
            OTHER_WITH_DIRECT: (
                "VCG reconfigures a block while at least one exact-safe direct "
                "delivery exists for another block; discretionary but causally ambiguous"
            ),
            NO_DIRECT: (
                "VCG reconfigures when no exact-safe direct-delivery candidate is "
                "available; recovery-context reconfiguration, not automatically "
                "equated to a PSLAP obstruction clearance"
            ),
            "retrieval_executor_obstruction_clearances": (
                "baseline scheduler retrieve_relocations counter, preserved under "
                "its native executor semantics"
            ),
            "frontiers_with_no_direct_delivery": (
                "VCG exact decision frontiers containing zero exact-safe direct-"
                "delivery candidates; denominator is exact_frontier_count"
            ),
            "legacy_aliases": (
                "relocations, obstructive_moves, and relocations_per_100_deliveries "
                "are retained verbatim from the historical ledger only for "
                "backward compatibility; historical VCG obstructive_moves is not "
                "the canonical target-bound count"
            ),
        },
        "cross_family_interpretation": {
            "common_total_physical_reconfiguration_count_available": True,
            "common_total_definition": (
                "executed storage-to-storage block moves; reconstructed VCG "
                "Reconfigure macros and authenticated baseline retrieve relocations"
            ),
            "common_causal_obstruction_count_available": False,
            "reason": (
                "VCG frontiers support proactive/recovery context classification, "
                "whereas baseline ledgers expose only retrieval-executor clearances"
            ),
            "permitted_descriptive_comparison": (
                "Compare total physical storage-to-storage reconfigurations, then "
                "report VCG causal-context classes and baseline retrieval "
                "clearances under their distinct labels"
            ),
            "forbidden_interpretation": (
                "Do not call all VCG Reconfigure macros obstructive moves, and do "
                "not rename the two native counters as one causal N_obs metric"
            ),
        },
        "whole_method_safety_rule": {
            "numeric_comparison_requires": (
                "strict success and full completion on every one of the 30 "
                "paired development instances"
            ),
            "failure_handling": (
                "retain every row and native event count, but suppress primary "
                "per-100 comparison rates for the whole method"
            ),
            "partial_delivery_rates": (
                "reported only under conditional_on_completed_deliveries_diagnostic"
            ),
        },
        "method_summaries": summaries,
        "all_row_invariants_passed": all(
            bool(row["invariants_passed"]) for row in rows
        ),
        "analysis_row_count": len(rows),
    }


def run_reanalysis(
    *,
    source_dir: Path,
    expanded_dir: Path,
    output_dir: Path,
) -> dict:
    rows, ledgers, provenance = authenticate_inputs(source_dir, expanded_dir)
    repaired_rows = []
    run_audits = []
    ledger_hash_by_key = {
        item["run_key"]: item["sha256"]
        for collection in (
            provenance["authenticated_original_run_ledgers"],
            provenance["authenticated_expanded_run_ledgers"],
        )
        for item in collection
    }
    for source in rows:
        repaired, audit = repair_row(source, ledgers)
        repaired["source_ledger_sha256"] = ledger_hash_by_key[
            repaired["source_ledger_run_key"]
        ]
        audit["source_ledger_sha256"] = repaired["source_ledger_sha256"]
        repaired_rows.append(repaired)
        run_audits.append(audit)
    repaired_rows.sort(
        key=lambda row: (
            str(row["method_group"]),
            -1 if row.get("model_seed") is None else int(row["model_seed"]),
            int(row["instance_seed"]),
        )
    )
    run_audits.sort(
        key=lambda row: (
            str(row["method_group"]),
            int(row["instance_seed"]),
            str(row["method_id"]),
        )
    )
    report = build_report(repaired_rows)
    if not report["all_row_invariants_passed"]:
        raise ContentionAuditError("one or more repaired rows failed invariants")
    audit = {
        "protocol": PROTOCOL,
        "protocol_schema_version": PROTOCOL_SCHEMA_VERSION,
        "scope": SCOPE,
        "performance_claim_authorized": False,
        "rollouts_executed": 0,
        "new_episode_instances_generated": 0,
        "sealed_or_prospective_panels_opened": False,
        "source_provenance": provenance,
        "run_reconstruction_audits": run_audits,
        "global_invariants": {
            "analysis_rows_match_expanded_source": len(repaired_rows) == 360,
            "every_source_ledger_hash_authenticated": all(
                bool(row["source_ledger_sha256"]) for row in repaired_rows
            ),
            "every_row_invariant_passed": all(
                bool(row["invariants_passed"]) for row in repaired_rows
            ),
            "no_rollouts_executed": True,
            "no_instances_generated": True,
        },
    }
    if not all(audit["global_invariants"].values()):
        raise ContentionAuditError("global contention-reanalysis invariant failed")

    output_dir = output_dir.resolve()
    _write_csv(output_dir / RUNS_FILENAME, repaired_rows)
    _atomic_json(output_dir / REPORT_FILENAME, report)
    _atomic_json(output_dir / AUDIT_FILENAME, audit)
    return {
        "runs": str(output_dir / RUNS_FILENAME),
        "report": str(output_dir / REPORT_FILENAME),
        "audit": str(output_dir / AUDIT_FILENAME),
        "summary": report,
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("results/vcg-dense-v1-1-pareto-development-30seed"),
    )
    parser.add_argument(
        "--expanded-dir",
        type=Path,
        default=Path("results/vcg-dense-expanded-online-baselines-development-30seed"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/vcg-contention-metric-reanalysis-development-30seed"),
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    result = run_reanalysis(
        source_dir=args.source_dir,
        expanded_dir=args.expanded_dir,
        output_dir=args.output_dir,
    )
    print(json.dumps(result["summary"]["method_summaries"], indent=2))
    print(f"Runs: {result['runs']}")
    print(f"Report: {result['report']}")
    print(f"Audit: {result['audit']}")


if __name__ == "__main__":
    main()
