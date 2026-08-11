#!/usr/bin/env python3
"""Extend the frozen 80000--80029 VCG development panel with GA baselines.

This runner never samples a new EpisodeInstance and never re-executes a VCG
policy.  It authenticates the original Pareto development artifacts, reuses
their 300 analysis rows, and executes only the two online deterministic GA
assignment sources that were absent from that experiment:

* duration-aware rolling GA; and
* operational rolling GA.

The full-schedule offline GA is deliberately kept outside the online ranking.
It is recorded as an unexecuted, information-advantaged reference because the
source-neutral Track-B controller interface does not expose it and because its
future-schedule access makes a pooled online rank scientifically misleading.

This remains a development-only diagnostic.  It cannot authorize a final
performance claim and it refuses any source panel other than the frozen
80000--80029 development panel.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
from itertools import combinations
import json
import math
from pathlib import Path
from statistics import fmean
from types import SimpleNamespace
from typing import Callable, Mapping, Optional, Sequence
import uuid

from compare_vcg_dense_pareto import (
    BASELINE_METHODS as SOURCE_BASELINE_METHODS,
    EVALUATION_SEEDS,
    FROZEN_OBJECTIVE_SPEC,
    GA_SEED_BASE,
    LOOKAHEAD_MARGIN_STEPS,
    MAX_DEFER_STEPS,
    MAX_STEPS,
    PROTOCOL as SOURCE_PROTOCOL,
    ROLLING_GA_EGRESS_WEIGHT,
    ROLLING_GENERATIONS,
    ROLLING_POPULATION,
    RUN_FIELDS as SOURCE_RUN_FIELDS,
    _baseline_args,
    _finish_row,
    _identity_for_baseline,
)
from compare_viability_graph_baselines import _normalize_baseline
from example.episode_instance import EpisodeInstance
from PSLAP.track_a import (
    TRACK_A_GA_OFFLINE,
    TRACK_A_GA_ROLLING_DURATION_AWARE,
    TRACK_A_GA_ROLLING_OPERATIONAL,
    TRACK_A_KIM2020_A3C_SPATIAL,
    TRACK_A_REG_SELECTOR_V5,
)
from track_b_urgency_evaluate import (
    evaluate_assignment_ablation_one,
    resolve_device,
)
from train_viability_graph_smdp import SEALED_STRESS_V1_HOLDOUT_SEEDS
from train_viability_graph_smdp_proper import SEALED_IN_REGIME_TEST_SEEDS


PROTOCOL = "vcg_dense_expanded_online_baselines_development_v1"
PROTOCOL_SCHEMA_VERSION = 1
EXPECTED_DELIVERY_COUNT = 8

DURATION_AWARE_GA_METHOD = (
    "duration_aware_pslap_ga_duration_aware_rolling"
)
OPERATIONAL_GA_METHOD = "duration_aware_pslap_ga_operational_rolling"
OFFLINE_GA_REFERENCE = "pslap_ga_2009_offline_full_schedule_reference"

NEW_ONLINE_METHOD_TO_SOURCE = {
    DURATION_AWARE_GA_METHOD: TRACK_A_GA_ROLLING_DURATION_AWARE,
    OPERATIONAL_GA_METHOD: TRACK_A_GA_ROLLING_OPERATIONAL,
}
NEW_ONLINE_METHODS = tuple(NEW_ONLINE_METHOD_TO_SOURCE)

VCG_SELECTED_GROUP = "vcg_dense_v1_1_selected_3seed"
VCG_FINAL_DIAGNOSTIC_GROUP = "vcg_dense_v1_1_episode500_diagnostic_3seed"

SOURCE_EXPECTED_METHOD_COUNTS = {
    **{method: len(EVALUATION_SEEDS) for method in SOURCE_BASELINE_METHODS},
    "vcg_dense_seed0_selected_best": len(EVALUATION_SEEDS),
    "vcg_dense_seed1_selected_best": len(EVALUATION_SEEDS),
    "vcg_dense_seed2_selected_best": len(EVALUATION_SEEDS),
    "vcg_dense_seed0_episode500_final": len(EVALUATION_SEEDS),
    "vcg_dense_seed1_episode500_final": len(EVALUATION_SEEDS),
    "vcg_dense_seed2_episode500_final": len(EVALUATION_SEEDS),
}

SOURCE_RESULTS_FILENAME = "pareto-runs.csv"
SOURCE_REPORT_FILENAME = "pareto-report.json"
SOURCE_AUDIT_FILENAME = "pareto-audit.json"
SOURCE_INSTANCE_MANIFEST_FILENAME = "instance-manifest.json"
SOURCE_PROTOCOL_MANIFEST_FILENAME = "protocol-manifest.json"

RESULTS_FILENAME = "expanded-baseline-runs.csv"
REPORT_FILENAME = "expanded-baseline-report.json"
AUDIT_FILENAME = "expanded-baseline-audit.json"
PROTOCOL_MANIFEST_FILENAME = "expanded-baseline-protocol-manifest.json"

EXPANSION_FIELDS = (
    "expanded_row_origin",
    "expanded_source_protocol",
    "expanded_source_artifact_sha256",
)
RUN_FIELDS = tuple(SOURCE_RUN_FIELDS) + EXPANSION_FIELDS

MEAN_FIELDS = (
    "dense_objective_return",
    "mean_absolute_error",
    "mean_tardiness",
    "mean_earliness",
    "within_target_window_rate",
    "relocations_per_100_deliveries",
    "steps",
    "planning_seconds",
)

SAFETY_FINITE_FIELDS = (
    "strict_method_success",
    "completion_rate",
    "dense_objective_return",
    "legacy_rescored_return",
    "mean_absolute_error",
    "first_two_mean_absolute_error",
    "positions_three_plus_mean_absolute_error",
    "mean_tardiness",
    "mean_earliness",
    "within_target_window_rate",
    "relocations_per_100_deliveries",
    "steps",
)
SAFETY_NONNEGATIVE_FIELDS = (
    "mean_absolute_error",
    "first_two_mean_absolute_error",
    "positions_three_plus_mean_absolute_error",
    "mean_tardiness",
    "mean_earliness",
    "relocations_per_100_deliveries",
)

OUTCOME_EQUIVALENCE_FIELDS = (
    "return",
    "environment_legacy_return",
    "legacy_rescored_return",
    "dense_rescored_return",
    "dense_objective_return",
    "primary_objective_return",
    "success",
    "strict_method_success",
    "completion_rate",
    "steps",
    "delivery_count",
    "delivery_deviations",
    "first_two_mean_absolute_error",
    "first_two_within_target_window_rate",
    "positions_three_plus_mean_absolute_error",
    "positions_three_plus_within_target_window_rate",
    "mean_signed_deviation",
    "mean_absolute_error",
    "mean_tardiness",
    "mean_earliness",
    "within_target_window_rate",
    "p90_tardiness",
    "p90_absolute_error",
    "relocations",
    "relocations_per_100_deliveries",
    "obstructive_moves",
    "illegal_drops",
    "invalid_assignments",
    "infeasible_epochs",
    "fallbacks",
    "method_failure_reason",
)

AGGREGATE_EQUIVALENCE_FIELDS = (
    "strict_success_rate",
    "completion_rate",
    "mean_dense_objective_return",
    "mean_mean_absolute_error",
    "mean_mean_tardiness",
    "mean_mean_earliness",
    "mean_within_target_window_rate",
    "mean_relocations_per_100_deliveries",
    "mean_steps",
)


@dataclass(frozen=True)
class SourcePanel:
    directory: Path
    runs: tuple[dict, ...]
    instances: Mapping[int, EpisodeInstance]
    instance_manifest: dict
    protocol_manifest: dict
    audit: dict
    artifact_sha256: Mapping[str, str]


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


def _digest_json(value) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def _load_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _require_equal(label: str, observed, expected) -> None:
    if _json_safe(observed) != _json_safe(expected):
        raise ValueError(
            f"{label} mismatch: observed={observed!r}, expected={expected!r}"
        )


def _validate_source_rows(
    rows: Sequence[Mapping], instance_manifest: Mapping
) -> None:
    expected_total = sum(SOURCE_EXPECTED_METHOD_COUNTS.values())
    if len(rows) != expected_total:
        raise ValueError(
            f"source analysis row count mismatch: {len(rows)} != {expected_total}"
        )
    records = instance_manifest.get("instances")
    if not isinstance(records, Mapping):
        raise ValueError("source instance manifest has no instances mapping")

    method_counts: dict[str, int] = {}
    run_keys = set()
    for row in rows:
        method = str(row.get("method_id"))
        method_counts[method] = method_counts.get(method, 0) + 1
        seed = int(row["instance_seed"])
        record = records.get(str(seed))
        if record is None:
            raise ValueError(f"source row uses undeclared instance seed {seed}")
        _require_equal("source row protocol", row.get("protocol"), SOURCE_PROTOCOL)
        _require_equal(
            "source row instance id", row.get("instance_id"), record["instance_id"]
        )
        _require_equal(
            "source row schedule id", row.get("schedule_id"), record["schedule_id"]
        )
        key = (method, row.get("instance_id"))
        if key in run_keys:
            raise ValueError(f"duplicate source method/instance row: {key}")
        run_keys.add(key)

    _require_equal(
        "source method grid", method_counts, SOURCE_EXPECTED_METHOD_COUNTS
    )


def _authenticate_source_panel(source_dir: Path) -> SourcePanel:
    source_dir = source_dir.resolve()
    required = {
        SOURCE_RESULTS_FILENAME: source_dir / SOURCE_RESULTS_FILENAME,
        SOURCE_REPORT_FILENAME: source_dir / SOURCE_REPORT_FILENAME,
        SOURCE_AUDIT_FILENAME: source_dir / SOURCE_AUDIT_FILENAME,
        SOURCE_INSTANCE_MANIFEST_FILENAME: (
            source_dir / SOURCE_INSTANCE_MANIFEST_FILENAME
        ),
        SOURCE_PROTOCOL_MANIFEST_FILENAME: (
            source_dir / SOURCE_PROTOCOL_MANIFEST_FILENAME
        ),
    }
    for path in required.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    artifact_sha256 = {
        name: _sha256_file(path) for name, path in required.items()
    }

    report = _load_json(required[SOURCE_REPORT_FILENAME])
    audit = _load_json(required[SOURCE_AUDIT_FILENAME])
    instance_manifest = _load_json(
        required[SOURCE_INSTANCE_MANIFEST_FILENAME]
    )
    protocol_manifest = _load_json(
        required[SOURCE_PROTOCOL_MANIFEST_FILENAME]
    )
    _require_equal("source report protocol", report.get("runner_protocol"), SOURCE_PROTOCOL)
    _require_equal("source audit protocol", audit.get("protocol"), SOURCE_PROTOCOL)
    _require_equal("source protocol manifest", protocol_manifest.get("protocol"), SOURCE_PROTOCOL)
    _require_equal(
        "source scope",
        report.get("scope"),
        "development_only_sealed_panels_unopened",
    )
    if bool(report.get("performance_claim_authorized")):
        raise ValueError("source panel unexpectedly authorizes a performance claim")
    _require_equal(
        "source evaluation seeds",
        protocol_manifest.get("evaluation_seeds"),
        list(EVALUATION_SEEDS),
    )
    _require_equal(
        "source baseline methods",
        protocol_manifest.get("baseline_methods"),
        list(SOURCE_BASELINE_METHODS),
    )
    _require_equal(
        "audit instance manifest", audit.get("instance_manifest"), instance_manifest
    )
    _require_equal(
        "audit protocol manifest", audit.get("protocol_manifest"), protocol_manifest
    )
    _require_equal(
        "report instance manifest", report.get("instance_manifest"), instance_manifest
    )

    contract = instance_manifest.get("contract", {})
    _require_equal("instance contract protocol", contract.get("protocol"), SOURCE_PROTOCOL)
    _require_equal(
        "instance contract seeds", contract.get("seeds"), list(EVALUATION_SEEDS)
    )
    instances: dict[int, EpisodeInstance] = {}
    for seed in EVALUATION_SEEDS:
        record = instance_manifest["instances"].get(str(seed))
        if not isinstance(record, Mapping):
            raise ValueError(f"missing source instance record for seed {seed}")
        path = Path(record["path"]).resolve()
        expected_path = (source_dir / "instances" / f"seed-{seed}.json").resolve()
        _require_equal("source instance path", path, expected_path)
        _require_equal("source instance hash", _sha256_file(path), record["sha256"])
        instance = EpisodeInstance.from_json(path.read_text(encoding="utf-8"))
        _require_equal("EpisodeInstance seed", instance.seed, seed)
        _require_equal("EpisodeInstance id", instance.instance_id, record["instance_id"])
        _require_equal("EpisodeInstance schedule", instance.schedule_id, record["schedule_id"])
        instances[seed] = instance

    ledger_manifest = audit.get("run_ledger_manifest")
    if not isinstance(ledger_manifest, list) or len(ledger_manifest) != 270:
        raise ValueError("source run-ledger manifest must contain 270 physical runs")
    ledger_keys = set()
    for entry in ledger_manifest:
        key = str(entry["run_key"])
        if key in ledger_keys:
            raise ValueError(f"duplicate source ledger key: {key}")
        ledger_keys.add(key)
        path = Path(entry["path"]).resolve()
        if source_dir not in path.parents:
            raise ValueError(f"source ledger escapes source directory: {path}")
        _require_equal("source ledger hash", _sha256_file(path), entry["sha256"])

    runs = report.get("runs")
    if not isinstance(runs, list):
        raise ValueError("source report has no runs list")
    _validate_source_rows(runs, instance_manifest)
    return SourcePanel(
        directory=source_dir,
        runs=tuple(dict(row) for row in runs),
        instances=instances,
        instance_manifest=instance_manifest,
        protocol_manifest=protocol_manifest,
        audit=audit,
        artifact_sha256=artifact_sha256,
    )


def _expanded_source_rows(panel: SourcePanel) -> tuple[dict, ...]:
    report_hash = panel.artifact_sha256[SOURCE_REPORT_FILENAME]
    output = []
    for source in panel.runs:
        row = dict(source)
        row.update(
            {
                "protocol": PROTOCOL,
                "expanded_row_origin": "authenticated_source_row_reuse",
                "expanded_source_protocol": SOURCE_PROTOCOL,
                "expanded_source_artifact_sha256": report_hash,
            }
        )
        output.append(row)
    return tuple(output)


def _baseline_runtime_args(panel: SourcePanel, device) -> SimpleNamespace:
    shared = panel.protocol_manifest.get("shared_execution_contract")
    if not isinstance(shared, Mapping) or not isinstance(
        shared.get("environment"), Mapping
    ):
        raise ValueError("source protocol has no shared environment contract")
    return _baseline_args({"environment": shared["environment"]}, device)


def _run_new_online_baseline(
    method: str,
    instance: EpisodeInstance,
    seed: int,
    *,
    block_count: int,
    runtime_args: SimpleNamespace,
) -> dict:
    raw = evaluate_assignment_ablation_one(
        runtime_args,
        seed,
        None,
        assignment_source=NEW_ONLINE_METHOD_TO_SOURCE[method],
        episode_instance=instance,
    )
    normalized = _normalize_baseline(
        method, raw, instance, block_count, FROZEN_OBJECTIVE_SPEC
    )
    row = _finish_row(normalized, _identity_for_baseline(method))
    row.update(
        {
            "protocol": PROTOCOL,
            "expanded_row_origin": "new_online_baseline_execution",
            "expanded_source_protocol": SOURCE_PROTOCOL,
            "expanded_source_artifact_sha256": None,
        }
    )
    return row


def _new_input_contract(
    method: str,
    seed: int,
    panel: SourcePanel,
) -> dict:
    record = panel.instance_manifest["instances"][str(seed)]
    return {
        "protocol": PROTOCOL,
        "protocol_schema_version": PROTOCOL_SCHEMA_VERSION,
        "scope": "development_only_sealed_panels_unopened",
        "method_id": method,
        "assignment_source": NEW_ONLINE_METHOD_TO_SOURCE[method],
        "instance_seed": seed,
        "instance_id": record["instance_id"],
        "schedule_id": record["schedule_id"],
        "instance_sha256": record["sha256"],
        "source_protocol": SOURCE_PROTOCOL,
        "source_protocol_manifest_sha256": panel.artifact_sha256[
            SOURCE_PROTOCOL_MANIFEST_FILENAME
        ],
        "source_instance_manifest_sha256": panel.artifact_sha256[
            SOURCE_INSTANCE_MANIFEST_FILENAME
        ],
        "max_steps": MAX_STEPS,
        "max_defer_steps": MAX_DEFER_STEPS,
        "lookahead_margin_steps": LOOKAHEAD_MARGIN_STEPS,
        "rolling_population": ROLLING_POPULATION,
        "rolling_generations": ROLLING_GENERATIONS,
        "rolling_ga_egress_weight": ROLLING_GA_EGRESS_WEIGHT,
        "ga_seed": GA_SEED_BASE + seed,
        "timing_objective_spec": FROZEN_OBJECTIVE_SPEC.to_dict(),
    }


def _ledger_path(output_dir: Path, method: str, seed: int) -> Path:
    return output_dir / "run-ledger" / method / f"seed-{seed}.json"


def _load_or_execute_new_run(
    *,
    output_dir: Path,
    method: str,
    seed: int,
    input_contract: Mapping,
    execute: Callable[[], dict],
) -> tuple[dict, dict]:
    path = _ledger_path(output_dir, method, seed)
    fingerprint = _digest_json(input_contract)
    if path.is_file():
        record = _load_json(path)
        _require_equal("expanded ledger protocol", record.get("protocol"), PROTOCOL)
        _require_equal("expanded ledger key", record.get("run_key"), f"{method}:{seed}")
        _require_equal(
            "expanded ledger input contract",
            record.get("input_contract"),
            input_contract,
        )
        _require_equal(
            "expanded ledger fingerprint",
            record.get("input_fingerprint"),
            fingerprint,
        )
        row = record.get("run")
        if not isinstance(row, dict):
            raise ValueError(f"expanded ledger has no run row: {path}")
    else:
        row = execute()
        _require_equal("executor method", row.get("method_id"), method)
        _require_equal("executor seed", int(row.get("instance_seed")), seed)
        record = {
            "protocol": PROTOCOL,
            "run_key": f"{method}:{seed}",
            "input_contract": dict(input_contract),
            "input_fingerprint": fingerprint,
            "run": {key: value for key, value in row.items() if key != "method_audit"},
            "method_audit": row.get("method_audit"),
        }
        _atomic_json(path, record)
    _require_equal("ledger instance", row.get("instance_id"), input_contract["instance_id"])
    _require_equal("ledger schedule", row.get("schedule_id"), input_contract["schedule_id"])
    return dict(row), record


def _method_group(row: Mapping) -> str:
    policy_group = row.get("policy_group")
    if policy_group == "selected_best":
        return VCG_SELECTED_GROUP
    if policy_group == "episode500_final_diagnostic":
        return VCG_FINAL_DIAGNOSTIC_GROUP
    return str(row["method_id"])


def _group_category(group: str) -> str:
    if group == VCG_SELECTED_GROUP:
        return "learned_online_primary"
    if group == VCG_FINAL_DIAGNOSTIC_GROUP:
        return "learned_online_diagnostic"
    if group in SOURCE_BASELINE_METHODS or group in NEW_ONLINE_METHODS:
        return "deterministic_online_primary"
    if group == OFFLINE_GA_REFERENCE:
        return "offline_full_schedule_reference"
    raise ValueError(f"unknown expanded method group: {group}")


def _finite_mean(rows: Sequence[Mapping], field: str) -> Optional[float]:
    values = []
    for row in rows:
        value = row.get(field)
        if value is None:
            continue
        number = float(value)
        if math.isfinite(number):
            values.append(number)
    return float(fmean(values)) if values else None


def _row_safety_issues(row: Mapping) -> list[str]:
    issues = []
    for field in SAFETY_FINITE_FIELDS:
        value = row.get(field)
        try:
            finite = value is not None and math.isfinite(float(value))
        except (TypeError, ValueError):
            finite = False
        if not finite:
            issues.append(f"missing_or_nonfinite:{field}")
    try:
        if float(row.get("strict_method_success")) != 1.0:
            issues.append("strict_method_failure")
    except (TypeError, ValueError):
        pass
    try:
        if float(row.get("completion_rate")) != 1.0:
            issues.append("incomplete_episode")
    except (TypeError, ValueError):
        pass
    try:
        delivery_count = int(row.get("delivery_count", 0))
        if delivery_count <= 0:
            issues.append("no_completed_deliveries")
        if delivery_count != EXPECTED_DELIVERY_COUNT:
            issues.append(
                f"unexpected_delivery_count:{delivery_count}:"
                f"expected_{EXPECTED_DELIVERY_COUNT}"
            )
    except (TypeError, ValueError):
        issues.append("invalid_delivery_count")
    try:
        if int(row.get("steps", 0)) <= 0:
            issues.append("nonpositive_steps")
    except (TypeError, ValueError):
        issues.append("invalid_steps")
    for field in SAFETY_NONNEGATIVE_FIELDS:
        value = row.get(field)
        try:
            if value is not None and math.isfinite(float(value)) and float(value) < 0.0:
                issues.append(f"negative:{field}")
        except (TypeError, ValueError):
            pass
    window = row.get("within_target_window_rate")
    try:
        if window is not None and math.isfinite(float(window)) and not 0.0 <= float(window) <= 1.0:
            issues.append("outside_unit_interval:within_target_window_rate")
    except (TypeError, ValueError):
        pass
    if row.get("method_failure_reason") is not None:
        issues.append("method_failure_reason_present")
    return issues


def _group_grid_issues(group: str, rows: Sequence[Mapping]) -> list[str]:
    issues = []
    expected_instances = set(EVALUATION_SEEDS)
    if group in (VCG_SELECTED_GROUP, VCG_FINAL_DIAGNOSTIC_GROUP):
        if len(rows) != 3 * len(EVALUATION_SEEDS):
            issues.append("learned_row_count_not_3x30")
        observed_seeds = {
            int(row["model_seed"])
            for row in rows
            if row.get("model_seed") is not None
        }
        if observed_seeds != {0, 1, 2}:
            issues.append("learned_model_seed_grid_not_0_1_2")
        for model_seed in (0, 1, 2):
            selected = [
                row for row in rows if row.get("model_seed") == model_seed
            ]
            if {int(row["instance_seed"]) for row in selected} != expected_instances:
                issues.append(f"learned_instance_grid_incomplete:seed{model_seed}")
            if len(selected) != len(EVALUATION_SEEDS):
                issues.append(f"learned_row_count_not_30:seed{model_seed}")
    else:
        if len(rows) != len(EVALUATION_SEEDS):
            issues.append("deterministic_row_count_not_30")
        if {int(row["instance_seed"]) for row in rows} != expected_instances:
            issues.append("deterministic_instance_grid_incomplete")
    identities = [
        (row.get("model_seed"), int(row["instance_seed"])) for row in rows
    ]
    if len(identities) != len(set(identities)):
        issues.append("duplicate_model_seed_instance_row")
    return issues


def _summarize_group(group: str, rows: Sequence[Mapping]) -> dict:
    selected = [row for row in rows if _method_group(row) == group]
    if not selected:
        raise ValueError(f"cannot summarize empty method group {group}")
    observed_instances = {int(row["instance_seed"]) for row in selected}
    grid_issues = _group_grid_issues(group, selected)
    row_issues = []
    for row in selected:
        issues = _row_safety_issues(row)
        if issues:
            row_issues.append(
                {
                    "instance_seed": int(row["instance_seed"]),
                    "model_seed": row.get("model_seed"),
                    "issues": issues,
                }
            )
    failure_rows = [
        {
            "instance_seed": int(row["instance_seed"]),
            "model_seed": row.get("model_seed"),
            "reason": row.get("method_failure_reason"),
        }
        for row in selected
        if float(row["strict_method_success"]) != 1.0
        or float(row["completion_rate"]) != 1.0
        or row.get("method_failure_reason") is not None
    ]
    summary = {
        "method_group": group,
        "category": _group_category(group),
        "row_count": len(selected),
        "instance_count": len(observed_instances),
        "model_seeds": sorted(
            {
                int(row["model_seed"])
                for row in selected
                if row.get("model_seed") is not None
            }
        ),
        "strict_success_rate": _finite_mean(selected, "strict_method_success"),
        "completion_rate": _finite_mean(selected, "completion_rate"),
        "full_instance_grid": not grid_issues,
        "grid_issues": grid_issues,
        "row_safety_issues": row_issues,
        "whole_method_numeric_eligible": bool(not grid_issues and not row_issues),
        "method_failures": failure_rows,
    }
    for field in MEAN_FIELDS:
        summary[f"mean_{field}"] = (
            _finite_mean(selected, field)
            if summary["whole_method_numeric_eligible"]
            else None
        )
    return summary


def _rank_summaries(summaries: Sequence[Mapping]) -> dict:
    online = [
        dict(item)
        for item in summaries
        if item["category"] in (
            "learned_online_primary",
            "deterministic_online_primary",
        )
        and item["whole_method_numeric_eligible"]
    ]
    specs = {
        "dense_objective_return_higher_is_better": (
            "mean_dense_objective_return",
            True,
        ),
        "mean_absolute_error_lower_is_better": (
            "mean_mean_absolute_error",
            False,
        ),
        "mean_tardiness_lower_is_better": ("mean_mean_tardiness", False),
        "within_target_window_rate_higher_is_better": (
            "mean_within_target_window_rate",
            True,
        ),
        "relocations_per_100_lower_is_better": (
            "mean_relocations_per_100_deliveries",
            False,
        ),
        "steps_lower_is_better": ("mean_steps", False),
    }
    output = {}
    for label, (field, descending) in specs.items():
        ordered = sorted(
            online,
            key=lambda item: (
                -float(item[field]) if descending else float(item[field]),
                item["method_group"],
            ),
        )
        ranked = []
        previous_value = None
        previous_rank = None
        for index, item in enumerate(ordered):
            value = float(item[field])
            if (
                previous_value is not None
                and math.isclose(value, previous_value, rel_tol=0.0, abs_tol=1e-12)
            ):
                rank = previous_rank
            else:
                rank = index + 1
            ranked.append(
                {
                    "rank": rank,
                    "method_group": item["method_group"],
                    "value": item[field],
                }
            )
            previous_value = value
            previous_rank = rank
        output[label] = ranked
    return output


def _numeric_ranking_invariants(
    rows: Sequence[Mapping], summaries: Sequence[Mapping]
) -> dict:
    eligible_groups = {
        item["method_group"]
        for item in summaries
        if item["category"] in (
            "learned_online_primary",
            "deterministic_online_primary",
        )
        and item["whole_method_numeric_eligible"]
    }
    violations = []
    by_instance: dict[int, dict[str, set[int]]] = {
        seed: {} for seed in EVALUATION_SEEDS
    }
    for row in rows:
        group = _method_group(row)
        if group not in eligible_groups:
            continue
        seed = int(row["instance_seed"])
        count = int(row["delivery_count"])
        by_instance[seed].setdefault(group, set()).add(count)
    for seed, groups in by_instance.items():
        if set(groups) != eligible_groups:
            violations.append(f"eligible_group_missing:instance_{seed}")
            continue
        observed = set()
        for group, counts in groups.items():
            observed.update(counts)
            if counts != {EXPECTED_DELIVERY_COUNT}:
                violations.append(
                    f"unexpected_delivery_count:instance_{seed}:{group}:"
                    f"{sorted(counts)}"
                )
        if len(observed) != 1:
            violations.append(
                f"cross_method_delivery_count_mismatch:instance_{seed}:"
                f"{sorted(observed)}"
            )
    if violations:
        raise RuntimeError(
            "numeric ranking delivery-count invariant failed: "
            + "; ".join(violations)
        )
    return {
        "passed": True,
        "expected_delivery_count_per_episode": EXPECTED_DELIVERY_COUNT,
        "eligible_online_groups": sorted(eligible_groups),
        "paired_instance_count": len(EVALUATION_SEEDS),
        "cross_method_per_instance_delivery_count_equal": True,
    }


def _outcome_equivalence(
    rows: Sequence[Mapping], summaries: Sequence[Mapping]
) -> dict:
    deterministic_groups = [
        item["method_group"]
        for item in summaries
        if item["category"] == "deterministic_online_primary"
    ]
    summaries_by_group = {
        item["method_group"]: item for item in summaries
    }
    rows_by_group = {
        group: {
            int(row["instance_seed"]): row
            for row in rows
            if _method_group(row) == group
        }
        for group in deterministic_groups
    }
    pairs = []
    for left, right in combinations(deterministic_groups, 2):
        left_rows = rows_by_group[left]
        right_rows = rows_by_group[right]
        common = sorted(set(left_rows).intersection(right_rows))
        differences = []
        identical = 0
        for seed in common:
            fields = [
                field
                for field in OUTCOME_EQUIVALENCE_FIELDS
                if _json_safe(left_rows[seed].get(field))
                != _json_safe(right_rows[seed].get(field))
            ]
            if fields:
                differences.append(
                    {"instance_seed": seed, "differing_fields": fields}
                )
            else:
                identical += 1
        left_summary = summaries_by_group[left]
        right_summary = summaries_by_group[right]
        aggregate_eligible = bool(
            left_summary["whole_method_numeric_eligible"]
            and right_summary["whole_method_numeric_eligible"]
        )
        aggregate_differences = []
        if aggregate_eligible:
            aggregate_differences = [
                field
                for field in AGGREGATE_EQUIVALENCE_FIELDS
                if _json_safe(left_summary.get(field))
                != _json_safe(right_summary.get(field))
            ]
        full_grid = set(common) == set(EVALUATION_SEEDS)
        row_identical = bool(full_grid and not differences)
        aggregate_equal = (
            bool(not aggregate_differences) if aggregate_eligible else None
        )
        pairs.append(
            {
                "left": left,
                "right": right,
                "aggregate_comparison_eligible": aggregate_eligible,
                "reported_aggregates_exactly_equal": aggregate_equal,
                "differing_aggregate_fields": aggregate_differences,
                "common_instance_count": len(common),
                "outcome_identical_instance_count": identical,
                "outcome_identical_on_all_30_instances": row_identical,
                "exact_aggregate_and_row_equivalent": bool(
                    aggregate_equal is True and row_identical
                ),
                "per_instance_differences": differences,
            }
        )
    return {
        "outcome_fields": OUTCOME_EQUIVALENCE_FIELDS,
        "aggregate_fields": AGGREGATE_EQUIVALENCE_FIELDS,
        "exact_equivalent_pairs": [
            {"left": item["left"], "right": item["right"]}
            for item in pairs
            if item["exact_aggregate_and_row_equivalent"]
        ],
        "pairs": pairs,
        "interpretation": (
            "Exact equivalence means identical recorded outcomes on this "
            "30-instance panel; it does not prove identical actions or policies "
            "outside the observed panel. Runtime fields are intentionally excluded."
        ),
    }


def _pareto_front(summaries: Sequence[Mapping]) -> list[str]:
    eligible = [
        item
        for item in summaries
        if item["category"] in (
            "learned_online_primary",
            "deterministic_online_primary",
        )
        and item["whole_method_numeric_eligible"]
    ]
    result = []
    for candidate in eligible:
        c_mae = float(candidate["mean_mean_absolute_error"])
        c_reloc = float(candidate["mean_relocations_per_100_deliveries"])
        dominated = False
        for other in eligible:
            if other is candidate:
                continue
            o_mae = float(other["mean_mean_absolute_error"])
            o_reloc = float(other["mean_relocations_per_100_deliveries"])
            if (
                o_mae <= c_mae
                and o_reloc <= c_reloc
                and (o_mae < c_mae or o_reloc < c_reloc)
            ):
                dominated = True
                break
        if not dominated:
            result.append(str(candidate["method_group"]))
    return sorted(result)


def build_expanded_report(rows: Sequence[Mapping]) -> dict:
    groups = (
        VCG_SELECTED_GROUP,
        VCG_FINAL_DIAGNOSTIC_GROUP,
        *SOURCE_BASELINE_METHODS,
        *NEW_ONLINE_METHODS,
    )
    summaries = [_summarize_group(group, rows) for group in groups]
    ranking_invariants = _numeric_ranking_invariants(rows, summaries)
    return {
        "protocol": PROTOCOL,
        "scope": "development_only_sealed_panels_unopened",
        "performance_claim_authorized": False,
        "comparison_level": "complete_system_fixed_source_neutral_duration_aware_scheduler",
        "pairing_key": "EpisodeInstance.instance_id_and_schedule_id",
        "online_primary_groups": [
            item["method_group"]
            for item in summaries
            if item["category"] in (
                "learned_online_primary",
                "deterministic_online_primary",
            )
        ],
        "diagnostic_groups": [VCG_FINAL_DIAGNOSTIC_GROUP],
        "offline_reference": {
            "method_group": OFFLINE_GA_REFERENCE,
            "assignment_source": TRACK_A_GA_OFFLINE,
            "information_regime": "offline_full_schedule",
            "executed": False,
            "included_in_online_ranking": False,
            "reason": (
                "full-future schedule access is information-advantaged and the "
                "neutral Track-B assignment interface does not expose this source"
            ),
        },
        "pending_matched_online_baselines": [
            {
                "method_group": TRACK_A_REG_SELECTOR_V5,
                "status": "pending_matched_5x5_8block_contention_checkpoint",
                "included_in_current_ranking": False,
            },
            {
                "method_group": f"{TRACK_A_KIM2020_A3C_SPATIAL}__stochastic",
                "status": "pending_matched_5x5_8block_contention_checkpoint_and_policy_seed_protocol",
                "included_in_current_ranking": False,
            },
            {
                "method_group": f"{TRACK_A_KIM2020_A3C_SPATIAL}__map",
                "status": "pending_matched_5x5_8block_contention_checkpoint_ablation",
                "included_in_current_ranking": False,
            },
        ],
        "method_summaries": summaries,
        "safety_ineligible_methods": [
            item["method_group"]
            for item in summaries
            if item["category"] != "learned_online_diagnostic"
            and not item["whole_method_numeric_eligible"]
        ],
        "rankings": _rank_summaries(summaries),
        "mae_relocation_pareto_front": _pareto_front(summaries),
        "numeric_ranking_invariants": ranking_invariants,
        "deterministic_outcome_equivalence": _outcome_equivalence(
            rows, summaries
        ),
        "interpretation_limits": [
            "development panel only; no confirmatory performance claim",
            "metric rankings are separate because no scalar multi-objective rank was prespecified",
            "a method with any strict/full failure remains in the safety ledger but is excluded whole from numeric ranking",
            "VCG selected results average equally over three model seeds and thirty paired instances; deterministic methods execute once per instance",
            "offline full-schedule GA is not pooled with online-arrived-only methods",
        ],
    }


def _write_csv(path: Path, rows: Sequence[Mapping]) -> None:
    temporary = path.parent / f".{path.name}.tmp-{uuid.uuid4().hex}"
    try:
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=RUN_FIELDS)
            writer.writeheader()
            for row in rows:
                encoded = {}
                for field in RUN_FIELDS:
                    value = row.get(field)
                    if isinstance(value, (list, tuple, dict)):
                        value = json.dumps(
                            _json_safe(value), separators=(",", ":")
                        )
                    encoded[field] = value
                writer.writerow(encoded)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _validate_protocol() -> None:
    if tuple(EVALUATION_SEEDS) != tuple(range(80_000, 80_030)):
        raise RuntimeError("expanded protocol requires the frozen 80000--80029 panel")
    sealed = set(EVALUATION_SEEDS).intersection(
        SEALED_STRESS_V1_HOLDOUT_SEEDS | SEALED_IN_REGIME_TEST_SEEDS
    )
    if sealed:
        raise RuntimeError(f"expanded development panel intersects sealed seeds: {sorted(sealed)}")
    if set(NEW_ONLINE_METHODS).intersection(SOURCE_EXPECTED_METHOD_COUNTS):
        raise RuntimeError("new online method identifiers collide with source rows")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extend the frozen VCG development panel with two online GA variants"
    )
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cpu")
    parser.add_argument("--resume-existing", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> dict:
    args = _build_parser().parse_args(argv)
    _validate_protocol()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.resume_existing:
        raise FileExistsError(
            f"{output_dir} is nonempty; pass --resume-existing or use a new directory"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    panel = _authenticate_source_panel(args.source_dir)
    device = resolve_device(args.device)
    protocol_manifest = {
        "protocol": PROTOCOL,
        "protocol_schema_version": PROTOCOL_SCHEMA_VERSION,
        "scope": "development_only_sealed_panels_unopened",
        "performance_claim_authorized": False,
        "source_directory": str(panel.directory),
        "source_protocol": SOURCE_PROTOCOL,
        "source_artifact_sha256": panel.artifact_sha256,
        "evaluation_seeds": EVALUATION_SEEDS,
        "source_rows_reused": len(panel.runs),
        "source_policies_reexecuted": False,
        "new_online_methods": NEW_ONLINE_METHODS,
        "new_online_assignment_sources": NEW_ONLINE_METHOD_TO_SOURCE,
        "offline_reference": {
            "method": OFFLINE_GA_REFERENCE,
            "assignment_source": TRACK_A_GA_OFFLINE,
            "executed": False,
            "included_in_online_ranking": False,
        },
        "execution_device": str(device),
        "fixed_configuration": {
            "max_steps": MAX_STEPS,
            "max_defer_steps": MAX_DEFER_STEPS,
            "lookahead_margin_steps": LOOKAHEAD_MARGIN_STEPS,
            "rolling_population": ROLLING_POPULATION,
            "rolling_generations": ROLLING_GENERATIONS,
            "rolling_ga_egress_weight": ROLLING_GA_EGRESS_WEIGHT,
            "ga_seed_base": GA_SEED_BASE,
            "timing_objective_spec": FROZEN_OBJECTIVE_SPEC.to_dict(),
        },
    }
    manifest_path = output_dir / PROTOCOL_MANIFEST_FILENAME
    if manifest_path.is_file():
        _require_equal("expanded protocol manifest", _load_json(manifest_path), protocol_manifest)
    else:
        if args.resume_existing and (output_dir / "run-ledger").exists():
            raise ValueError("cannot resume expanded ledgers without a protocol manifest")
        _atomic_json(manifest_path, protocol_manifest)

    runtime_args = _baseline_runtime_args(panel, device)
    block_count = int(
        panel.protocol_manifest["shared_execution_contract"]["environment"][
            "number_blocks"
        ]
    )
    rows = list(_expanded_source_rows(panel))
    ledger_manifest = []
    for method in NEW_ONLINE_METHODS:
        for seed in EVALUATION_SEEDS:
            instance = panel.instances[seed]
            contract = _new_input_contract(method, seed, panel)
            row, ledger = _load_or_execute_new_run(
                output_dir=output_dir,
                method=method,
                seed=seed,
                input_contract=contract,
                execute=lambda method=method, instance=instance, seed=seed: (
                    _run_new_online_baseline(
                        method,
                        instance,
                        seed,
                        block_count=block_count,
                        runtime_args=runtime_args,
                    )
                ),
            )
            rows.append(row)
            path = _ledger_path(output_dir, method, seed)
            ledger_manifest.append(
                {
                    "run_key": ledger["run_key"],
                    "path": str(path.resolve()),
                    "sha256": _sha256_file(path),
                    "input_fingerprint": ledger["input_fingerprint"],
                    "method_audit_present": ledger.get("method_audit") is not None,
                }
            )
            print(
                f"[{method}] seed={seed} "
                f"DenseR={row['dense_objective_return']:.2f} "
                f"strict={int(float(row['strict_method_success']))} "
                f"MAE={row['mean_absolute_error']} "
                f"reloc/100={row['relocations_per_100_deliveries']} "
                f"failure={row['method_failure_reason']}",
                flush=True,
            )

    expected_rows = len(panel.runs) + len(NEW_ONLINE_METHODS) * len(EVALUATION_SEEDS)
    grid = {(row["method_id"], row["instance_id"]) for row in rows}
    if len(rows) != expected_rows or len(grid) != expected_rows:
        raise RuntimeError("expanded method/instance grid is incomplete")

    report = {
        **build_expanded_report(rows),
        "source_artifact_sha256": panel.artifact_sha256,
        "source_row_count": len(panel.runs),
        "new_execution_row_count": len(NEW_ONLINE_METHODS) * len(EVALUATION_SEEDS),
        "analysis_row_count": len(rows),
        "runs": rows,
    }
    audit = {
        "protocol": PROTOCOL,
        "protocol_manifest": protocol_manifest,
        "source_audit_sha256": panel.artifact_sha256[SOURCE_AUDIT_FILENAME],
        "source_instance_manifest": panel.instance_manifest,
        "new_run_ledger_manifest": ledger_manifest,
    }
    _write_csv(output_dir / RESULTS_FILENAME, rows)
    _atomic_json(output_dir / REPORT_FILENAME, report)
    _atomic_json(output_dir / AUDIT_FILENAME, audit)
    print(f"Runs: {output_dir / RESULTS_FILENAME}", flush=True)
    print(f"Report: {output_dir / REPORT_FILENAME}", flush=True)
    print(f"Audit: {output_dir / AUDIT_FILENAME}", flush=True)
    return report


if __name__ == "__main__":
    main()
