#!/usr/bin/env python3
"""E19: post-hoc service tails and operational-cost sensitivity.

No policy is executed or trained. The primary analysis reuses the closed E1
90k ledgers. E11 and E13 contribute episode-level portability and scale
summaries only because their saved rows do not retain job-level deviations.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
from statistics import fmean, stdev
import sys
from typing import Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize


PROTOCOL = "vcg_conditioned_e19_service_tails_cost_sensitivity_v1"
SCHEMA_VERSION = 1
CONTRACT_NAME = "e19-contract.json"
REPORT_NAME = "e19-report.json"
SERVICE_TABLE = "e19-service-tail-table.md"
COST_TABLE = "e19-cost-sensitivity-table.md"
PORTABILITY_TABLE = "e19-portability-scale-table.md"
SERVICE_FIGURE = "e19-service-tails"
COST_FIGURE = "e19-cost-sensitivity"

DEFAULT_OUTPUT = PROJECT_ROOT / "results/vcg-conditioned-e19-service-cost"
E1_OUTPUT = PROJECT_ROOT / "results/vcg-conditioned-final-comparison-90k-cpu-v3"
E11_OUTPUT = PROJECT_ROOT / "results/vcg-conditioned-e11-distribution-shift-93k"
E13_OUTPUT = PROJECT_ROOT / "results/vcg-conditioned-e13-scalability-95k-v2"
E4_OUTPUT = PROJECT_ROOT / "results/vcg-conditioned-e04-ranking-ablation-92k"
E5C_OUTPUT = PROJECT_ROOT / "results/vcg-conditioned-e05c-future-consequence-ablation-92k"

TARDINESS_WEIGHTS = (1.0, 2.0, 4.0)
REHANDLE_EQUIVALENTS = (0.0, 10.0, 25.0, 50.0)
VCG_LAMBDAS = (0.0, 0.025, 0.0375, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2)
GA_KEY = "duration_aware_pslap_ga_2009_rolling_capacity_aware_partial"
V23_KEY = "vcg_2_3"
SELECTED_TAIL_KEYS = (
    "vcg_conditioned:lambda=0",
    "vcg_conditioned:lambda=0.05",
    "vcg_conditioned:lambda=0.1",
    "vcg_conditioned:lambda=0.2",
    V23_KEY,
    GA_KEY,
)


class E19Error(RuntimeError):
    pass


def _canonical(value: Mapping) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _digest(value: Mapping, field: str | None = None) -> str:
    payload = dict(value)
    if field is not None:
        payload.pop(field, None)
    return hashlib.sha256(_canonical(payload)).hexdigest()


def _with_hash(value: Mapping, field: str) -> dict:
    result = dict(value)
    result[field] = _digest(result)
    return result


def _sha(path: Path) -> str:
    path = Path(path).resolve()
    if not path.is_file() or path.is_symlink():
        raise E19Error(f"missing regular artifact: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_json(path: Path, label: str) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise E19Error(f"invalid {label}: {path}") from error
    if not isinstance(value, dict):
        raise E19Error(f"{label} must contain a JSON object")
    return value


def _verify_hash(value: Mapping, field: str, label: str) -> None:
    if value.get(field) != _digest(value, field):
        raise E19Error(f"{label} self-hash mismatch")


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(value, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(path: Path, value: Mapping) -> None:
    _atomic_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def _quantile(values: Sequence[float], probability: float) -> float | None:
    """Hyndman-Fan type-7 quantile (linear interpolation)."""

    ordered = sorted(float(value) for value in values)
    if not ordered:
        return None
    probability = float(probability)
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must lie in [0,1]")
    position = (len(ordered) - 1) * probability
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _tail_record(values: Sequence[float]) -> dict:
    values = tuple(float(value) for value in values)
    return {
        "n": len(values),
        "p50": _quantile(values, 0.50),
        "p90": _quantile(values, 0.90),
        "p95": _quantile(values, 0.95),
        "p99": _quantile(values, 0.99),
        "maximum": max(values) if values else None,
    }


def _ledger_index(root: Path) -> tuple[list[Path], str]:
    paths = sorted((root / "run-ledger").rglob("*.json"))
    records = []
    for path in paths:
        ledger = _load_json(path, f"{root.name} ledger")
        _verify_hash(ledger, "ledger_sha256", f"{root.name} ledger")
        records.append(
            {"path": str(path.relative_to(PROJECT_ROOT)), "sha256": _sha(path)}
        )
    digest = hashlib.sha256(
        json.dumps(records, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return paths, digest


def _parent_report(path: Path, *, expected_status: str | None = "complete") -> dict:
    report = _load_json(path, path.name)
    _verify_hash(report, "report_sha256", path.name)
    if expected_status is not None and report.get("status") != expected_status:
        raise E19Error(f"parent report is not complete: {path}")
    return report


def _artifact_inventory() -> dict:
    reports = {
        "E1": (E1_OUTPUT / "final90-report.json", "complete"),
        "E4": (E4_OUTPUT / "e04-report.json", "complete"),
        "E5c": (E5C_OUTPUT / "e05c-report.json", "complete"),
        "E11": (E11_OUTPUT / "e11-report.json", "complete"),
        "E13": (E13_OUTPUT / "e13-report.json", None),
    }
    loaded = {
        name: _parent_report(path, expected_status=status)
        for name, (path, status) in reports.items()
    }
    if (
        int(loaded["E1"].get("observed_total_rows", -1)) != 1860
        or int(loaded["E11"].get("rows", -1)) != 2730
        or int(loaded["E13"].get("observed_rows", -1)) != 45
        or bool(loaded["E13"].get("partial"))
    ):
        raise E19Error("parent result grids are incomplete")
    e1_paths, e1_digest = _ledger_index(E1_OUTPUT)
    e11_paths, e11_digest = _ledger_index(E11_OUTPUT)
    e13_paths, e13_digest = _ledger_index(E13_OUTPUT)
    if (len(e1_paths), len(e11_paths), len(e13_paths)) != (1860, 2730, 45):
        raise E19Error("parent ledger counts changed")
    return {
        "reports": {
            name: {
                "path": str(path.relative_to(PROJECT_ROOT)),
                "sha256": _sha(path),
                "report_sha256": loaded[name]["report_sha256"],
            }
            for name, (path, _status) in reports.items()
        },
        "ledger_sets": {
            "E1": {"count": len(e1_paths), "sha256": e1_digest},
            "E11": {"count": len(e11_paths), "sha256": e11_digest},
            "E13": {"count": len(e13_paths), "sha256": e13_digest},
        },
    }


def _contract() -> dict:
    return _with_hash(
        {
            "schema_version": SCHEMA_VERSION,
            "protocol": PROTOCOL,
            "scientific_question": (
                "do_mean_timing_handling_gains_translate_into_service_tail_"
                "reliability_and_plausible_operational_cost_sensitivity"
            ),
            "analysis_only": True,
            "new_training_runs": 0,
            "new_evaluation_rollouts": 0,
            "primary_panel": "closed_E1_90k",
            "secondary_panels": [
                "closed_E11_93k_episode_level_portability",
                "closed_E13_95k_episode_level_scale_summary",
            ],
            "service_tail_rule": (
                "job_level_quantiles_only_for_E1_method_points_with_all_rows_"
                "strict_safe_complete"
            ),
            "quantile_definition": "Hyndman_Fan_type_7_linear_interpolation",
            "tail_probabilities": [0.5, 0.9, 0.95, 0.99],
            "cost_definition": (
                "mean_earliness+w_tardy*mean_tardiness+"
                "k_rehandle*(physical_rehandles/required_deliveries)"
            ),
            "cost_units": "equivalent_delivery_timing_steps_per_required_delivery",
            "tardiness_weights": list(TARDINESS_WEIGHTS),
            "rehandle_equivalent_steps": list(REHANDLE_EQUIVALENTS),
            "steps_excluded_from_cost": (
                "primitive_steps_overlap_with_delivery_timing_and_handling_and_"
                "are_reported_separately"
            ),
            "not_claimed": [
                "monetary_savings",
                "energy_savings",
                "independent_job_level_replications",
                "hard_budget_guarantee_from_lambda",
            ],
            "checkpoint_or_preference_selection": False,
            "source_sha256": {
                str(Path(__file__).resolve().relative_to(PROJECT_ROOT)): _sha(
                    Path(__file__).resolve()
                )
            },
            "parents": _artifact_inventory(),
        },
        "contract_sha256",
    )


def prepare(output: Path) -> dict:
    output = output.resolve()
    expected = _contract()
    path = output / CONTRACT_NAME
    if path.is_file():
        observed = _load_json(path, "E19 contract")
        _verify_hash(observed, "contract_sha256", "E19 contract")
        if observed != expected:
            raise E19Error("E19 contract, sources, or parent artifacts changed")
    else:
        if output.exists() and any(output.iterdir()):
            raise E19Error("nonempty E19 output has no contract")
        output.mkdir(parents=True, exist_ok=True)
        _atomic_json(path, expected)
    return {
        "status": "prepared",
        "training_runs": 0,
        "evaluation_rollouts": 0,
        "e1_rows": expected["parents"]["ledger_sets"]["E1"]["count"],
        "e11_rows": expected["parents"]["ledger_sets"]["E11"]["count"],
        "e13_rows": expected["parents"]["ledger_sets"]["E13"]["count"],
        "contract": str(path),
    }


def authenticate(output: Path) -> dict:
    observed = _load_json(output / CONTRACT_NAME, "E19 contract")
    _verify_hash(observed, "contract_sha256", "E19 contract")
    if observed != _contract():
        raise E19Error("E19 contract, sources, or parent artifacts changed")
    return observed


def _e1_key(row: Mapping) -> tuple[str, str]:
    method = row.get("method") or row.get("method_id")
    if method == "vcg_conditioned":
        token = f"{float(row['preference_lambda']):g}"
        return f"vcg_conditioned:lambda={token}", f"VCG (lambda={token})"
    labels = {
        V23_KEY: "Historical VCG 2.3",
        "duration_aware_dynamic_pslap": "Dynamic PSLAP",
        GA_KEY: "Capacity-aware GA",
        "kim2020_a3c_spatial_adapted__stochastic": "Kim2020 adaptation",
    }
    if method not in labels:
        raise E19Error(f"unknown E1 method: {method}")
    return str(method), labels[str(method)]


def _e1_rows() -> list[dict]:
    paths, _digest_value = _ledger_index(E1_OUTPUT)
    rows = []
    for path in paths:
        source = _load_json(path, "E1 ledger").get("run")
        if not isinstance(source, dict):
            raise E19Error("E1 ledger lacks a run row")
        key, label = _e1_key(source)
        deviations = [float(value) for value in source.get("delivery_deviations", ())]
        if not all(math.isfinite(value) for value in deviations):
            raise E19Error("E1 contains a nonfinite delivery deviation")
        required = int(source.get("required_deliveries", 8))
        strict = bool(source.get("strict_safe_complete"))
        if strict and len(deviations) != required:
            raise E19Error("strict E1 row has an incomplete delivery trace")
        rows.append(
            {
                "method_key": key,
                "display_name": label,
                "strict_safe_complete": strict,
                "instance_seed": int(source["instance_seed"]),
                "model_seed": source.get("model_seed"),
                "rng_index": source.get("rng_index", source.get("rollout_index")),
                "required_deliveries": required,
                "delivery_deviations": deviations,
                "delivery_count": len(deviations),
                "mean_earliness": source.get("mean_earliness"),
                "mean_tardiness": source.get("mean_tardiness"),
                "mean_absolute_error": source.get("mean_absolute_error"),
                "within_target_window_rate": source.get(
                    "within_target_window_rate"
                ),
                "physical_rehandles": source.get(
                    "physical_storage_relocations"
                ),
                "steps": source.get("steps"),
                "failure_reason": source.get("method_failure_reason"),
            }
        )
    if len(rows) != 1860:
        raise E19Error("E1 row count changed")
    return rows


def _service_summary(rows: Sequence[Mapping]) -> dict:
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["method_key"]].append(row)
    result = {}
    for key, group in sorted(grouped.items()):
        strict = [row for row in group if row["strict_safe_complete"]]
        eligible = len(strict) == len(group)
        all_recorded = [
            value for row in group for value in row["delivery_deviations"]
        ]
        errors = [value for row in strict for value in row["delivery_deviations"]]
        failed_reasons = defaultdict(int)
        for row in group:
            if not row["strict_safe_complete"]:
                failed_reasons[str(row["failure_reason"])] += 1
        record = {
            "display_name": group[0]["display_name"],
            "rows": len(group),
            "unique_instances": len({row["instance_seed"] for row in group}),
            "strict_safe_complete": len(strict),
            "strict_completion_rate": len(strict) / len(group),
            "offered_jobs": sum(row["required_deliveries"] for row in group),
            "recorded_delivered_jobs": len(all_recorded),
            "recorded_unmet_jobs": sum(
                row["required_deliveries"] - row["delivery_count"] for row in group
            ),
            "failed_row_reasons": dict(sorted(failed_reasons.items())),
            "service_tail_eligible": eligible,
            "tail_suppression_reason": (
                None
                if eligible
                else "one_or_more_rows_failed_the_strict_completion_gate"
            ),
        }
        if eligible:
            absolute = [abs(value) for value in errors]
            tardiness = [max(value, 0.0) for value in errors]
            positive_tardiness = [value for value in errors if value > 0.0]
            earliness = [max(-value, 0.0) for value in errors]
            record["episode_metrics"] = {
                "mean_absolute_error": fmean(
                    float(row["mean_absolute_error"]) for row in strict
                ),
                "mean_earliness": fmean(
                    float(row["mean_earliness"]) for row in strict
                ),
                "mean_tardiness": fmean(
                    float(row["mean_tardiness"]) for row in strict
                ),
                "within_20_fraction": fmean(
                    float(row["within_target_window_rate"]) for row in strict
                ),
                "physical_rehandles_per_100_required_deliveries": 100.0
                * sum(float(row["physical_rehandles"]) for row in strict)
                / sum(int(row["required_deliveries"]) for row in strict),
                "steps_per_required_delivery": sum(
                    float(row["steps"]) for row in strict
                )
                / sum(int(row["required_deliveries"]) for row in strict),
            }
            record["service"] = {
                "jobs": len(errors),
                "early_job_fraction": sum(value < 0.0 for value in errors)
                / len(errors),
                "on_time_job_fraction": sum(value == 0.0 for value in errors)
                / len(errors),
                "late_job_fraction": sum(value > 0.0 for value in errors)
                / len(errors),
                "within_20_fraction": sum(abs(value) <= 20.0 for value in errors)
                / len(errors),
                "absolute_error": _tail_record(absolute),
                "tardiness_all_jobs": _tail_record(tardiness),
                "tardiness_late_jobs_only": _tail_record(positive_tardiness),
                "earliness_all_jobs": _tail_record(earliness),
                "worst_signed_deviation": {
                    "earliest": min(errors),
                    "latest": max(errors),
                },
            }
        else:
            record["episode_metrics"] = None
            record["service"] = None
        result[key] = record
    return result


def _cost_loss(row: Mapping, tardiness_weight: float, rehandle_equivalent: float):
    if not row["strict_safe_complete"]:
        return None
    return (
        float(row["mean_earliness"])
        + float(tardiness_weight) * float(row["mean_tardiness"])
        + float(rehandle_equivalent)
        * float(row["physical_rehandles"])
        / int(row["required_deliveries"])
    )


def _cost_sensitivity(rows: Sequence[Mapping], service: Mapping) -> dict:
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["method_key"]].append(row)
    scenarios = []
    for tardiness_weight in TARDINESS_WEIGHTS:
        for rehandle_equivalent in REHANDLE_EQUIVALENTS:
            methods = {}
            for key, group in sorted(grouped.items()):
                eligible = service[key]["service_tail_eligible"]
                if not eligible:
                    methods[key] = {
                        "display_name": group[0]["display_name"],
                        "eligible": False,
                        "mean_loss": None,
                        "reason": "strict_completion_gate_failed",
                    }
                    continue
                values = [
                    _cost_loss(row, tardiness_weight, rehandle_equivalent)
                    for row in group
                ]
                methods[key] = {
                    "display_name": group[0]["display_name"],
                    "eligible": True,
                    "mean_loss": fmean(values),
                    "sd_across_rows": stdev(values) if len(values) > 1 else None,
                    "rows": len(values),
                }
            eligible = [
                (key, item["mean_loss"])
                for key, item in methods.items()
                if item["eligible"]
            ]
            ranking = [key for key, _ in sorted(eligible, key=lambda item: item[1])]
            scenarios.append(
                {
                    "scenario_id": (
                        f"tardy-{tardiness_weight:g}_rehandle-{rehandle_equivalent:g}"
                    ),
                    "tardiness_weight": tardiness_weight,
                    "rehandle_equivalent_timing_steps": rehandle_equivalent,
                    "methods": methods,
                    "ranking": ranking,
                }
            )
    return {
        "definition": (
            "mean_earliness+w_tardy*mean_tardiness+"
            "k_rehandle*(physical_rehandles/required_deliveries)"
        ),
        "units": "equivalent_delivery_timing_steps_per_required_delivery",
        "scenario_count": len(scenarios),
        "scenarios": scenarios,
        "interpretation": (
            "hypothetical_exchange_rate_sensitivity_not_monetary_or_energy_cost"
        ),
        "reporting_rule": (
            "all_predeclared_operating_points_reported;ranking_is_descriptive_"
            "and_does_not_select_a_new_deployment_preference"
        ),
    }


def _e11_summary() -> list[dict]:
    paths, _digest_value = _ledger_index(E11_OUTPUT)
    groups = defaultdict(list)
    for path in paths:
        row = _load_json(path, "E11 ledger")["run"]
        method = str(row["method"])
        if method == "conditioned_vcg":
            key = f"vcg_conditioned:lambda={float(row['lambda']):g}"
        elif method == "frozen_qop_safe":
            key = "vcg_conditioned:lambda=0"
        else:
            continue
        groups[(str(row["regime_id"]), key)].append(row)
    result = []
    for (regime, key), rows in sorted(groups.items()):
        strict = [row for row in rows if row["strict_safe_complete"]]
        complete = len(strict) == len(rows)
        result.append(
            {
                "regime_id": regime,
                "method_key": key,
                "rows": len(rows),
                "strict_safe_complete": len(strict),
                "metrics_suppressed": not complete,
                "mean_absolute_error": (
                    fmean(float(row["mean_absolute_error"]) for row in strict)
                    if complete
                    else None
                ),
                "mean_earliness": (
                    fmean(float(row["mean_earliness"]) for row in strict)
                    if complete
                    else None
                ),
                "mean_tardiness": (
                    fmean(float(row["mean_tardiness"]) for row in strict)
                    if complete
                    else None
                ),
                "within_20_fraction": (
                    fmean(float(row["within_target_window_rate"]) for row in strict)
                    if complete
                    else None
                ),
                "rehandles_per_100": (
                    fmean(
                        float(row["physical_rehandles_per_100_required_deliveries"])
                        for row in strict
                    )
                    if complete
                    else None
                ),
            }
        )
    if len(result) != 28:
        raise E19Error(f"E11 VCG portability grid changed: {len(result)}/28")
    return result


def _e13_summary() -> list[dict]:
    paths, _digest_value = _ledger_index(E13_OUTPUT)
    groups = defaultdict(list)
    for path in paths:
        row = _load_json(path, "E13 ledger")["row"]
        groups[str(row["scenario_id"])].append(row)
    result = []
    for scenario, rows in sorted(groups.items()):
        strict = [row for row in rows if row["strict_safe_complete"]]
        complete = len(strict) == len(rows)
        metrics = {}
        for name in (
            "mean_absolute_error",
            "mean_earliness",
            "mean_tardiness",
            "within_target_window_rate",
            "physical_rehandles_per_100_required_deliveries",
            "steps_per_delivery",
        ):
            metrics[name] = (
                fmean(float(row[name]) for row in strict) if complete else None
            )
        result.append(
            {
                "scenario_id": scenario,
                "comparison": rows[0]["scenario"]["comparison"],
                "rows": len(rows),
                "strict_safe_complete": len(strict),
                "required_deliveries": rows[0]["required_deliveries"],
                "initial_storage_occupancy_ratio": fmean(
                    float(row["occupancy"]["initial_storage_occupancy_ratio"])
                    for row in rows
                ),
                "time_weighted_mean_storage_occupancy_ratio": fmean(
                    float(
                        row["occupancy"][
                            "time_weighted_mean_storage_occupancy_ratio"
                        ]
                    )
                    for row in rows
                ),
                "peak_storage_occupancy_ratio": fmean(
                    float(row["occupancy"]["peak_storage_occupancy_ratio"])
                    for row in rows
                ),
                **metrics,
            }
        )
    if len(result) != 15:
        raise E19Error("E13 scenario grid changed")
    return result


def _data_availability() -> dict:
    return {
        "cost_components": {
            "early_delivery": "available",
            "late_delivery": "available",
            "physical_relocation_count": "available",
            "primitive_steps": "available_but_excluded_from_combined_cost",
            "travel_distance": "not_separately_logged",
            "loading_unloading_time": "not_separately_logged",
            "transporter_occupancy": "not_separately_logged",
            "energy": "not_measured",
            "money": "not_measured",
        },
        "E1": {
            "rows": 1860,
            "job_level_signed_delivery_deviations": True,
            "episode_earliness_tardiness": True,
            "rehandles_and_steps": True,
            "use": "primary_service_tails_and_cost_sensitivity",
        },
        "E4_E5c": {
            "job_level_signed_delivery_deviations": False,
            "episode_earliness_tardiness": True,
            "rehandles_and_steps": True,
            "use": "no_reconstructed_job_tails;E1_supersedes_final_method_points",
        },
        "E11": {
            "rows": 2730,
            "job_level_signed_delivery_deviations": False,
            "episode_earliness_tardiness": True,
            "rehandles_and_steps": True,
            "use": "same_lambda_regime_portability_only",
        },
        "E13": {
            "rows": 45,
            "job_level_signed_delivery_deviations": False,
            "episode_earliness_tardiness": True,
            "rehandles_and_steps": True,
            "use": "scale_dependent_episode_summary_only",
        },
    }


def _service_table(service: Mapping) -> str:
    lines = [
        "# E19 service-tail summary on the closed E1 panel",
        "",
        "| Method | Strict complete | Jobs | Within ±20 | Absolute p50 | p90 | p95 | p99 | Worst | Late-job p95 | Worst late |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in service.values():
        if item["service"] is None:
            tail = ["—"] * 8
        else:
            data = item["service"]
            absolute = data["absolute_error"]
            late = data["tardiness_late_jobs_only"]
            tail = [
                f"{100.0 * data['within_20_fraction']:.1f}%",
                f"{absolute['p50']:.2f}",
                f"{absolute['p90']:.2f}",
                f"{absolute['p95']:.2f}",
                f"{absolute['p99']:.2f}",
                f"{absolute['maximum']:.2f}",
                f"{late['p95']:.2f}" if late["p95"] is not None else "—",
                f"{late['maximum']:.2f}" if late["maximum"] is not None else "—",
            ]
        lines.append(
            f"| {item['display_name']} | {item['strict_safe_complete']}/{item['rows']} | "
            f"{item['recorded_delivered_jobs']}/{item['offered_jobs']} | "
            + " | ".join(tail)
            + " |"
        )
    lines.extend(
        [
            "",
            "Quantiles use pooled job outcomes descriptively; rows/model seeds remain the replication structure. Dynamic PSLAP and Kim2020 tails are suppressed because their method-level strict-completion gates failed.",
        ]
    )
    return "\n".join(lines) + "\n"


def _cost_table(cost: Mapping) -> str:
    vcg_keys = [f"vcg_conditioned:lambda={value:g}" for value in VCG_LAMBDAS]
    vcg_headers = [f"VCG {value:g}" for value in VCG_LAMBDAS]
    lines = [
        "# E19 operational-cost sensitivity",
        "",
        "| Tardiness weight | Rehandle equivalent | "
        + " | ".join(vcg_headers)
        + " | GA | Historical VCG 2.3 |",
        "|---:|---:|" + "---:|" * (len(vcg_keys) + 2),
    ]
    for scenario in cost["scenarios"]:
        methods = scenario["methods"]
        vcg_values = " | ".join(
            f"{methods[key]['mean_loss']:.2f}" for key in vcg_keys
        )
        lines.append(
            f"| {scenario['tardiness_weight']:.0f} | "
            f"{scenario['rehandle_equivalent_timing_steps']:.0f} | "
            f"{vcg_values} | {methods[GA_KEY]['mean_loss']:.2f} | "
            f"{methods[V23_KEY]['mean_loss']:.2f} |"
        )
    lines.extend(
        [
            "",
            "Loss is mean earliness + tardiness_weight × mean tardiness + rehandle_equivalent × rehandles per delivery. Values are equivalent timing-step units, not money or energy. Primitive steps are excluded to avoid double counting and remain separate.",
            "",
            "All ten predeclared frozen VCG operating points are shown. This is a sensitivity surface, not a new test-panel preference selection.",
        ]
    )
    return "\n".join(lines) + "\n"


def _portability_table(e11: Sequence[Mapping], e13: Sequence[Mapping]) -> str:
    lines = [
        "# E19 preference portability and scale service summary",
        "",
        "## Same frozen preferences across E11 regimes",
        "",
        "| Regime | VCG point | Strict complete | MAE | Early | Tardy | Within ±20 | Rehandles/100 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in e11:
        lines.append(
            f"| {row['regime_id']} | {row['method_key'].replace('vcg_conditioned:', '')} | "
            f"{row['strict_safe_complete']}/{row['rows']} | "
            f"{row['mean_absolute_error']:.2f} | {row['mean_earliness']:.2f} | "
            f"{row['mean_tardiness']:.2f} | {100.0 * row['within_20_fraction']:.1f}% | "
            f"{row['rehandles_per_100']:.2f} |"
        )
    lines.extend(
        [
            "",
            "## E13 episode-level scale summary",
            "",
            "| Scenario | Strict complete | Jobs | Initial/mean/peak occupancy | MAE | Early | Tardy | Within ±20 | Rehandles/100 | Steps/delivery |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in e13:
        lines.append(
            f"| {row['scenario_id']} | {row['strict_safe_complete']}/{row['rows']} | "
            f"{row['required_deliveries']} | "
            f"{row['initial_storage_occupancy_ratio']:.2f}/"
            f"{row['time_weighted_mean_storage_occupancy_ratio']:.2f}/"
            f"{row['peak_storage_occupancy_ratio']:.2f} | "
            f"{row['mean_absolute_error']:.2f} | {row['mean_earliness']:.2f} | "
            f"{row['mean_tardiness']:.2f} | "
            f"{100.0 * row['within_target_window_rate']:.1f}% | "
            f"{row['physical_rehandles_per_100_required_deliveries']:.2f} | "
            f"{row['steps_per_delivery']:.2f} |"
        )
    lines.extend(
        [
            "",
            "E11 and E13 did not retain job-level deviations, so these are episode summaries rather than reconstructed tail distributions.",
        ]
    )
    return "\n".join(lines) + "\n"


def _survival(values: Sequence[float]) -> tuple[list[float], list[float]]:
    values = sorted(float(value) for value in values)
    xs = sorted(set([0.0, *values]))
    ys = [sum(value > threshold for value in values) / len(values) for threshold in xs]
    return xs, ys


def _plot_service(output: Path, rows: Sequence[Mapping], service: Mapping) -> None:
    by_key = defaultdict(list)
    for row in rows:
        by_key[row["method_key"]].extend(row["delivery_deviations"])
    colors = {
        "vcg_conditioned:lambda=0": "#087dbb",
        "vcg_conditioned:lambda=0.05": "#e99b00",
        "vcg_conditioned:lambda=0.1": "#0aa67d",
        "vcg_conditioned:lambda=0.2": "#cc6f9d",
        V23_KEY: "#62b5e5",
        GA_KEY: "#d55e00",
    }
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.4), sharey=True)
    for key in SELECTED_TAIL_KEYS:
        errors = by_key[key]
        for axis, values in zip(
            axes,
            ([abs(value) for value in errors], [max(value, 0.0) for value in errors]),
        ):
            xs, ys = _survival(values)
            axis.step(
                xs,
                ys,
                where="post",
                linewidth=2.0,
                color=colors[key],
                label=service[key]["display_name"],
            )
    axes[0].set_title("Absolute delivery-error tail")
    axes[1].set_title("Tardiness tail (zero for non-late jobs)")
    for axis in axes:
        axis.set_xlabel("Error relative to target [time steps]")
        axis.set_yscale("log")
        axis.set_ylim(1.0e-3, 1.05)
        axis.grid(True, alpha=0.25)
    axes[0].set_ylabel("Observed exceedance probability")
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.075),
        ncol=3,
        frameon=False,
    )
    fig.suptitle("Service tails on the closed common 90k panel")
    fig.text(
        0.5,
        0.022,
        "Descriptive pooled-job tails; Dynamic PSLAP and Kim2020 suppressed after strict-gate failures",
        ha="center",
        fontsize=9,
        color="#59636e",
    )
    fig.tight_layout(rect=(0, 0.22, 1, 0.94))
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"{SERVICE_FIGURE}.{suffix}", dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_cost(output: Path, cost: Mapping) -> None:
    scenarios = {
        (row["tardiness_weight"], row["rehandle_equivalent_timing_steps"]): row
        for row in cost["scenarios"]
    }
    cmap = plt.get_cmap("viridis")
    norm = Normalize(vmin=0.0, vmax=0.2)
    fig, axes = plt.subplots(1, 3, figsize=(14.4, 4.7))
    for axis, tardiness_weight in zip(axes, TARDINESS_WEIGHTS):
        for value in VCG_LAMBDAS:
            key = f"vcg_conditioned:lambda={value:g}"
            ys = [
                scenarios[(tardiness_weight, equivalent)]["methods"][key][
                    "mean_loss"
                ]
                for equivalent in REHANDLE_EQUIVALENTS
            ]
            axis.plot(
                REHANDLE_EQUIVALENTS,
                ys,
                marker="o",
                markersize=3.5,
                linewidth=1.3,
                alpha=0.85,
                color=cmap(norm(value)),
            )
        for key, label, color, marker in (
            (GA_KEY, "Capacity-aware GA", "#d55e00", "s"),
            (V23_KEY, "Historical VCG 2.3", "#62b5e5", "^"),
        ):
            ys = [
                scenarios[(tardiness_weight, equivalent)]["methods"][key][
                    "mean_loss"
                ]
                for equivalent in REHANDLE_EQUIVALENTS
            ]
            axis.plot(
                REHANDLE_EQUIVALENTS,
                ys,
                linestyle="--",
                marker=marker,
                linewidth=2.0,
                color=color,
                label=label,
            )
        axis.set_title(f"Tardiness weight = {tardiness_weight:g}")
        axis.set_xlabel("One rehandle ≡ timing steps")
        axis.grid(True, alpha=0.25)
    axes[0].set_ylabel("Hypothetical operational loss per delivery")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="lower left", bbox_to_anchor=(0.08, 0.01), frameon=False
    )
    colorbar = fig.colorbar(
        ScalarMappable(norm=norm, cmap=cmap),
        ax=axes,
        orientation="horizontal",
        fraction=0.06,
        pad=0.18,
        aspect=35,
    )
    colorbar.set_label("VCG deployment lambda (all ten frozen operating points)")
    fig.suptitle("Operational-cost sensitivity without monetary assumptions")
    fig.text(
        0.99,
        0.01,
        "Steps excluded to avoid double counting",
        ha="right",
        fontsize=9,
        color="#59636e",
    )
    fig.subplots_adjust(
        left=0.07, right=0.99, top=0.86, bottom=0.28, wspace=0.24
    )
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"{COST_FIGURE}.{suffix}", dpi=220, bbox_inches="tight")
    plt.close(fig)


def analyze(output: Path) -> dict:
    output = output.resolve()
    contract = authenticate(output)
    rows = _e1_rows()
    service = _service_summary(rows)
    cost = _cost_sensitivity(rows, service)
    e11 = _e11_summary()
    e13 = _e13_summary()
    report = _with_hash(
        {
            "schema_version": SCHEMA_VERSION,
            "protocol": PROTOCOL,
            "status": "complete",
            "paper_evidence": True,
            "contract_sha256": contract["contract_sha256"],
            "analysis_only": True,
            "training_runs": 0,
            "evaluation_rollouts": 0,
            "data_availability": _data_availability(),
            "statistical_scope": {
                "episode_rows_retain_model_seed_and_policy_rng_replication": True,
                "job_tail_counts_are_descriptive_not_independent_replications": True,
                "strict_gate_applied_at_whole_method_point": True,
                "cost_scenarios_are_post_hoc_sensitivity_not_model_selection": True,
            },
            "e1_service_tails": service,
            "e1_cost_sensitivity": cost,
            "e11_preference_portability": e11,
            "e13_scale_service_summary": e13,
            "claim_boundary": (
                "service_reliability_and_hypothetical_exchange_rate_"
                "interpretation_of_existing_simulator_outputs_only"
            ),
        },
        "report_sha256",
    )
    _atomic_json(output / REPORT_NAME, report)
    _atomic_text(output / SERVICE_TABLE, _service_table(service))
    _atomic_text(output / COST_TABLE, _cost_table(cost))
    _atomic_text(output / PORTABILITY_TABLE, _portability_table(e11, e13))
    _plot_service(output, rows, service)
    _plot_cost(output, cost)
    return {
        "status": "complete",
        "training_runs": 0,
        "evaluation_rollouts": 0,
        "e1_rows": len(rows),
        "service_method_points": len(service),
        "cost_scenarios": cost["scenario_count"],
        "e11_portability_coordinates": len(e11),
        "e13_scale_coordinates": len(e13),
        "report": str(output / REPORT_NAME),
        "service_figure": str(output / f"{SERVICE_FIGURE}.pdf"),
        "cost_figure": str(output / f"{COST_FIGURE}.pdf"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "analyze", "run"))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare(args.output)
    elif args.command == "analyze":
        result = analyze(args.output)
    else:
        prepare(args.output)
        result = analyze(args.output)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
