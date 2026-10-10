#!/usr/bin/env python3
"""E25-v2: the full frozen VCG preference family on E13's scale grid."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import platform
from statistics import fmean, stdev
import tempfile
from typing import Any, Mapping, Optional, Sequence
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[3]

import numpy as np
import torch

from experiments.conditioned_vcg.E13_operational_scalability_95k import (
    program as e13,
)
from experiments.conditioned_vcg.E25_scale_baseline_transfer_95k import (
    run as e25,
)
from experiments.conditioned_vcg.development.D10_scalability_support_screen import (
    occupancy_extension as occupancy,
)
import run_vcg_conditioned_final_comparison_90k as e1


PROTOCOL = "vcg_conditioned_e25_v2_full_preference_family_scale_95k_v2"
SCHEMA_VERSION = 1
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "results/vcg-conditioned-e25-v2-full-vcg-family-scale-95k-v2"
)
E13_OUTPUT = e13.DEFAULT_OUTPUT
E25_OUTPUT = e25.DEFAULT_OUTPUT
CONTRACT_NAME = "e25-v2-contract.json"
REPORT_NAME = "e25-v2-report.json"
TABLE_NAME = "e25-v2-full-family-table.md"
FIGURE_NAME = "e25-v2-full-family-scale-comparison"

SCENARIOS = tuple(e25.SCENARIOS)
INSTANCE_SEEDS = tuple(e25.INSTANCE_SEEDS)
LAMBDAS = tuple(float(value) for value in e1.LAMBDA_GRID)
HISTORICAL_LAMBDA = float(e13.PREFERENCE_LAMBDA)
NEW_LAMBDAS = LAMBDAS
EXPECTED_NEW_VCG_ROWS = len(LAMBDAS) * len(SCENARIOS) * len(INSTANCE_SEEDS)
EXPECTED_HISTORICAL_VCG_ROWS = len(SCENARIOS) * len(INSTANCE_SEEDS)
EXPECTED_REUSED_BASELINE_ROWS = e25.EXPECTED_NEW_ROWS
METHOD_VCG = e1.CONDITIONED_METHOD
BASELINE_METHODS = tuple(e25.NEW_METHODS)
METHOD_LABELS = {
    METHOD_VCG: "VCG",
    **{method: e25.METHOD_LABELS[method] for method in BASELINE_METHODS},
}
QUALITY_METRICS = tuple(e25.QUALITY_METRICS)


class E25V2Error(RuntimeError):
    pass


def _json_form(value: Any) -> Any:
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def _canonical(value: Mapping) -> bytes:
    return json.dumps(
        _json_form(dict(value)),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _digest(value: Mapping, field: Optional[str] = None) -> str:
    payload = dict(value)
    if field is not None:
        payload.pop(field, None)
    return hashlib.sha256(_canonical(payload)).hexdigest()


def _hashed(value: Mapping, field: str) -> dict:
    result = dict(value)
    result[field] = _digest(result)
    return result


def _sha(path: Path) -> str:
    path = path.resolve()
    if not path.is_file() or path.is_symlink():
        raise E25V2Error(f"missing regular file: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path, label: str) -> dict:
    path = path.resolve()
    if not path.is_file() or path.is_symlink():
        raise E25V2Error(f"missing {label}: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise E25V2Error(f"{label} is not a JSON object")
    return value


def _self_hashed(path: Path, field: str, label: str) -> dict:
    value = _load(path, label)
    if value.get(field) != _digest(value, field):
        raise E25V2Error(f"{label} self-hash mismatch")
    return value


def _atomic_json(path: Path, value: Mapping) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    os.close(descriptor)
    temporary = Path(name)
    try:
        temporary.write_text(
            json.dumps(_json_form(value), indent=2, sort_keys=True, allow_nan=False)
            + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _runtime() -> dict:
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "torch_intraop_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
        "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
        "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
        "device": "cpu",
    }


def _source_hashes() -> dict:
    hashes = e13._source_hashes()
    paths = (
        Path(__file__).resolve(),
        Path(e25.__file__).resolve(),
    )
    hashes.update(
        {str(path.relative_to(PROJECT_ROOT)): _sha(path) for path in paths}
    )
    return hashes


def _parents() -> tuple[dict, dict, dict, dict, dict]:
    e25_contract, e13_contract, e13_manifest, e13_report = e25.authenticate(
        E25_OUTPUT
    )
    e25_report = _self_hashed(E25_OUTPUT / e25.REPORT_NAME, "report_sha256", "E25 report")
    if (
        e25_report.get("contract_sha256") != e25_contract["contract_sha256"]
        or e25_report.get("status") != "complete"
        or int(e25_report.get("observed_new_rows", -1))
        != EXPECTED_REUSED_BASELINE_ROWS
        or e25_report.get("paper_evidence") is not True
    ):
        raise E25V2Error("completed E25 parent is not authenticated")
    return e25_contract, e25_report, e13_contract, e13_manifest, e13_report


def _contract_payload(
    e25_contract: Mapping,
    e25_report: Mapping,
    e13_contract: Mapping,
    e13_manifest: Mapping,
    e13_report: Mapping,
) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "post_hoc": True,
        "question": (
            "does_any_member_of_the_prespecified_VCG_preference_family_preserve_"
            "operational_quality_across_the_E13_scale_by_occupancy_grid"
        ),
        "model_seed": e13.MODEL_SEED,
        "preference_lambdas": list(LAMBDAS),
        "new_preference_lambdas": list(NEW_LAMBDAS),
        "historical_comparison_lambda": HISTORICAL_LAMBDA,
        "scenarios": list(SCENARIOS),
        "instance_seeds": list(INSTANCE_SEEDS),
        "expected_new_vcg_rows": EXPECTED_NEW_VCG_ROWS,
        "expected_historical_vcg_rows": EXPECTED_HISTORICAL_VCG_ROWS,
        "expected_reused_baseline_rows": EXPECTED_REUSED_BASELINE_ROWS,
        "weights_and_normalization_frozen": True,
        "training": False,
        "certification_stack": "E14_cleanup_plus_D12_relocation_family",
        "execution_parent_authentication": (
            "frozen_E13_contract_manifest_and_instances_with_current_executor_"
            "source_bound_by_E25_v2"
        ),
        "vcg_comparison_executor": (
            "all_ten_preferences_use_the_same_current_corrected_candidate_executor"
        ),
        "search_max_nodes": e13.SEARCH_MAX_NODES,
        "wall_limit_seconds_per_episode": e13.WALL_LIMIT_SECONDS,
        "aggregation": (
            "average nuisance/model replications within EpisodeInstance; "
            "then give the three EpisodeInstances equal weight"
        ),
        "quality_scope": "strict-complete rows only; completion always separate",
        "composition_warning": (
            "each main-grid workload contains two arriving jobs; all remaining "
            "jobs are initially stored"
        ),
        "parents": {
            "e25_contract_sha256": e25_contract["contract_sha256"],
            "e25_report_sha256": e25_report["report_sha256"],
            "e13_contract_sha256": e13_contract["contract_sha256"],
            "e13_manifest_sha256": e13_manifest["manifest_sha256"],
            "e13_report_sha256": e13_report["report_sha256"],
        },
        "parent_e13_manifest_sha256": e13_manifest["manifest_sha256"],
        "reused_controls_are_independent_replications": False,
        "source_sha256": _source_hashes(),
    }


def prepare(output: Path) -> dict:
    parents = _parents()
    expected = _hashed(_contract_payload(*parents), "contract_sha256")
    path = output.resolve() / CONTRACT_NAME
    if path.is_file():
        observed = _self_hashed(path, "contract_sha256", "E25-v2 contract")
        if observed != expected:
            raise E25V2Error("existing E25-v2 contract differs from current sources")
    else:
        _atomic_json(path, expected)
    return {
        "status": "prepared",
        "output": str(output.resolve()),
        "preference_lambdas": list(LAMBDAS),
        "new_vcg_rows": EXPECTED_NEW_VCG_ROWS,
        "historical_vcg_rows_for_parity_only": EXPECTED_HISTORICAL_VCG_ROWS,
        "reused_baseline_rows": EXPECTED_REUSED_BASELINE_ROWS,
        "contract_sha256": expected["contract_sha256"],
        "training_runs": 0,
    }


def authenticate(output: Path) -> tuple[dict, tuple[dict, dict, dict, dict, dict]]:
    parents = _parents()
    expected = _hashed(_contract_payload(*parents), "contract_sha256")
    observed = _self_hashed(
        output.resolve() / CONTRACT_NAME, "contract_sha256", "E25-v2 contract"
    )
    if observed != expected:
        raise E25V2Error("E25-v2 contract, parents, or sources changed")
    return observed, parents


def _lambda_token(value: float) -> str:
    return f"{float(value):.4f}".replace(".", "p")


def _ledger_path(output: Path, value: float, scenario_id: str, seed: int) -> Path:
    return (
        output.resolve()
        / "run-ledger"
        / f"lambda-{_lambda_token(value)}"
        / scenario_id
        / f"seed-{int(seed)}.json"
    )


def _authenticate_new_ledger(
    output: Path,
    contract: Mapping,
    value: float,
    scenario_id: str,
    seed: int,
) -> dict:
    ledger = _self_hashed(
        _ledger_path(output, value, scenario_id, seed),
        "ledger_sha256",
        "E25-v2 ledger",
    )
    row = ledger.get("row", {})
    if (
        ledger.get("contract_sha256") != contract["contract_sha256"]
        or row.get("scenario_id") != scenario_id
        or int(row.get("instance_seed", -1)) != int(seed)
        or not np.isclose(float(row.get("preference_lambda", -1.0)), float(value))
        or int(row.get("model_seed", -1)) != e13.MODEL_SEED
    ):
        raise E25V2Error("E25-v2 ledger coordinate mismatch")
    return ledger


def run_one(output: Path, value: float, scenario_id: str, seed: int) -> dict:
    value = float(value)
    if value not in NEW_LAMBDAS:
        raise E25V2Error("run-one accepts only predeclared preference values")
    if scenario_id not in SCENARIOS or int(seed) not in INSTANCE_SEEDS:
        raise E25V2Error("unsupported E25-v2 coordinate")
    contract, parents_bundle = authenticate(output)
    path = _ledger_path(output, value, scenario_id, seed)
    if path.is_file():
        return dict(
            _authenticate_new_ledger(
                output, contract, value, scenario_id, seed
            )["row"]
        )
    ledger = _execute_e13_coordinate(
        contract,
        parents_bundle,
        value=value,
        scenario_id=scenario_id,
        seed=int(seed),
        ledger_path=path,
    )
    return dict(
        _authenticate_new_ledger(output, contract, value, scenario_id, seed)["row"]
    )


def _execute_e13_coordinate(
    contract: Mapping,
    parents_bundle: tuple[dict, dict, dict, dict, dict],
    *,
    value: float,
    scenario_id: str,
    seed: int,
    ledger_path: Path,
) -> dict:
    """Run current E13 code against its authenticated frozen data parent."""

    _e25_contract, _e25_report, e13_contract, e13_manifest, _e13_report = (
        parents_bundle
    )
    inputs = e13._parent_inputs()
    expected_terminal = e13_contract.get("conditioned_terminal_sha256")
    observed_terminal = inputs["conditioned"]["terminal_sha256"][e13.MODEL_SEED]
    if observed_terminal != expected_terminal:
        raise E25V2Error("frozen E13 checkpoint identity changed")
    frozen_parent = (e13_contract, e13_manifest, inputs)
    with (
        patch.object(e13, "PREFERENCE_LAMBDA", float(value)),
        patch.object(e13, "authenticate", return_value=frozen_parent),
    ):
        return e13.run_coordinate(
            E13_OUTPUT,
            scenario_id,
            int(seed),
            contract_override=contract,
            ledger_path_override=ledger_path,
        )


def _coordinates(values: Sequence[float]) -> tuple[tuple[float, str, int], ...]:
    return tuple(
        (float(value), scenario_id, int(seed))
        for value in values
        for scenario_id in SCENARIOS
        for seed in INSTANCE_SEEDS
    )


def run_values(output: Path, values: Sequence[float], *, scope: str) -> dict:
    coordinates = _coordinates(values)
    strict = 0
    for index, (value, scenario_id, seed) in enumerate(coordinates, start=1):
        row = run_one(output, value, scenario_id, seed)
        strict += int(row["strict_safe_complete"])
        print(
            f"E25-v2 {index}/{len(coordinates)} | lambda={value:.4g} | "
            f"{scenario_id} seed={seed} strict={int(row['strict_safe_complete'])} | "
            f"wall={row['episode_wall_seconds']:.2f}s",
            flush=True,
        )
    return {
        "status": "complete",
        "scope": scope,
        "preference_lambdas": [float(value) for value in values],
        "rows": len(coordinates),
        "strict_safe_complete_rows": strict,
        "training_runs": 0,
    }


def run_pilot(output: Path) -> dict:
    coordinates = (
        (NEW_LAMBDAS[0], "size_5x5_occ_low", INSTANCE_SEEDS[0]),
        (HISTORICAL_LAMBDA, "size_5x5_occ_low", INSTANCE_SEEDS[0]),
        (NEW_LAMBDAS[-1], "size_5x5_occ_low", INSTANCE_SEEDS[0]),
    )
    rows = []
    for value, scenario_id, seed in coordinates:
        rows.append(run_one(output, value, scenario_id, seed))
    contract, parents = authenticate(output)
    historical_parity = _historical_parity(output, contract, parents[2])
    return {
        "status": "complete",
        "scope": "three-row endpoint-and-historical-parity pilot; not paper evidence by itself",
        "coordinates": [
            {
                "preference_lambda": value,
                "scenario_id": scenario_id,
                "instance_seed": seed,
                "strict_safe_complete": bool(row["strict_safe_complete"]),
            }
            for (value, scenario_id, seed), row in zip(coordinates, rows)
        ],
        "paper_evidence": False,
        "historical_lambda_010_parity": historical_parity,
        "training_runs": 0,
    }


def _normalize_vcg(source: Mapping) -> dict:
    scenario_id = str(source["scenario_id"])
    scenario = occupancy.SCENARIO_BY_ID[scenario_id]
    return {
        "method": METHOD_VCG,
        "method_label": METHOD_LABELS[METHOD_VCG],
        "preference_lambda": float(source["preference_lambda"]),
        "scenario_id": scenario_id,
        "scenario": scenario.public_dict(),
        "instance_seed": int(source["instance_seed"]),
        "replicate": 0,
        "model_seed": int(source["model_seed"]),
        "strict_safe_complete": bool(source["strict_safe_complete"]),
        "failure_class": source["failure_class"],
        "mean_absolute_error": source.get("mean_absolute_error"),
        "within_target_window_rate": source.get("within_target_window_rate"),
        "physical_rehandles_per_100_required_deliveries": source.get(
            "physical_rehandles_per_100_required_deliveries"
        ),
        "steps_per_delivery": source.get("steps_per_delivery"),
        "initially_stored_jobs": scenario.initial_occupied_slots,
        "arriving_jobs": scenario.total_jobs - scenario.initial_occupied_slots,
        "reused_control": False,
    }


def _new_vcg_rows(output: Path, contract: Mapping) -> list[dict]:
    rows = []
    for value, scenario_id, seed in _coordinates(LAMBDAS):
        path = _ledger_path(output, value, scenario_id, seed)
        if path.is_file():
            source = _authenticate_new_ledger(
                output, contract, value, scenario_id, seed
            )["row"]
            rows.append(_normalize_vcg(source))
    return rows


def _historical_parity(
    output: Path, contract: Mapping, e13_contract: Mapping
) -> dict:
    fields = (
        "strict_safe_complete",
        "failure_class",
        "mean_absolute_error",
        "within_target_window_rate",
        "physical_rehandles_per_100_required_deliveries",
        "steps_per_delivery",
    )
    comparisons = []
    for scenario_id in SCENARIOS:
        for seed in INSTANCE_SEEDS:
            path = _ledger_path(output, HISTORICAL_LAMBDA, scenario_id, seed)
            if not path.is_file():
                continue
            current = _authenticate_new_ledger(
                output, contract, HISTORICAL_LAMBDA, scenario_id, seed
            )["row"]
            historical = e13._authenticate_ledger(
                E13_OUTPUT, e13_contract, scenario_id, seed
            )["row"]
            comparisons.append(
                {
                    "scenario_id": scenario_id,
                    "instance_seed": seed,
                    "behavior_digest_equal": (
                        current.get("behavior_digest")
                        == historical.get("behavior_digest")
                    ),
                    "outcome_fields_equal": all(
                        current.get(field) == historical.get(field)
                        for field in fields
                    ),
                    "changed_outcome_fields": [
                        field
                        for field in fields
                        if current.get(field) != historical.get(field)
                    ],
                }
            )
    return {
        "matched_coordinates": len(comparisons),
        "expected_coordinates": EXPECTED_HISTORICAL_VCG_ROWS,
        "behavior_digest_equal": sum(
            item["behavior_digest_equal"] for item in comparisons
        ),
        "outcome_fields_equal": sum(
            item["outcome_fields_equal"] for item in comparisons
        ),
        "comparisons": comparisons,
    }


def _baseline_rows(e25_contract: Mapping) -> list[dict]:
    rows = e25._new_rows(E25_OUTPUT)
    if len(rows) != EXPECTED_REUSED_BASELINE_ROWS:
        raise E25V2Error("completed E25 baseline parent is incomplete")
    if any(row.get("contract_sha256") == e25_contract.get("contract_sha256") for row in rows):
        raise E25V2Error("unexpected contract field inside normalized E25 row")
    return rows


def _arm_key(row: Mapping) -> tuple[str, Optional[float]]:
    value = row.get("preference_lambda") if row["method"] == METHOD_VCG else None
    return row["method"], None if value is None else round(float(value), 8)


def _instance_means(rows: Sequence[Mapping]) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        groups[(*_arm_key(row), row["scenario_id"], int(row["instance_seed"]))].append(row)
    output = []
    for (method, value, scenario_id, seed), selected in sorted(
        groups.items(), key=lambda item: str(item[0])
    ):
        strict = [row for row in selected if row["strict_safe_complete"]]
        record = {
            "method": method,
            "preference_lambda": value,
            "scenario_id": scenario_id,
            "instance_seed": seed,
            "replicate_rows": len(selected),
            "strict_rows": len(strict),
            "strict_completion_rate": len(strict) / len(selected),
        }
        for metric in QUALITY_METRICS:
            values = [
                float(row[metric])
                for row in strict
                if row.get(metric) is not None
            ]
            record[metric] = fmean(values) if values else None
        output.append(record)
    return output


def _label(method: str, value: Optional[float]) -> str:
    if method != METHOD_VCG:
        return METHOD_LABELS[method]
    return rf"VCG, $\lambda={float(value):.4g}$"


def _summary(rows: Sequence[Mapping]) -> list[dict]:
    groups = defaultdict(list)
    for row in _instance_means(rows):
        groups[(row["method"], row["preference_lambda"], row["scenario_id"])].append(row)
    output = []
    for (method, value, scenario_id), selected in sorted(
        groups.items(), key=lambda item: str(item[0])
    ):
        scenario = occupancy.SCENARIO_BY_ID[scenario_id]
        record = {
            "method": method,
            "method_label": _label(method, value),
            "preference_lambda": value,
            "scenario_id": scenario_id,
            "yard_size": scenario.rows,
            "occupancy_level": scenario.occupancy_level,
            "initial_occupancy_ratio": (
                scenario.initial_occupied_slots / scenario.storage_capacity
            ),
            "initially_stored_jobs": scenario.initial_occupied_slots,
            "arriving_jobs": scenario.total_jobs - scenario.initial_occupied_slots,
            "episode_instances": len(selected),
            "strict_completion_rate": fmean(
                item["strict_completion_rate"] for item in selected
            ),
            "strict_completion_min": min(
                item["strict_completion_rate"] for item in selected
            ),
            "strict_completion_max": max(
                item["strict_completion_rate"] for item in selected
            ),
        }
        for metric in QUALITY_METRICS:
            values = [
                float(item[metric])
                for item in selected
                if item.get(metric) is not None
            ]
            record[metric] = fmean(values) if values else None
            record[f"{metric}_instance_n"] = len(values)
            record[f"{metric}_instance_min"] = min(values) if values else None
            record[f"{metric}_instance_max"] = max(values) if values else None
            record[f"{metric}_instance_sd"] = (
                stdev(values) if len(values) > 1 else None
            )
        output.append(record)
    return output


def _render(summary: Sequence[Mapping], output: Path) -> dict:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    from matplotlib.lines import Line2D

    baseline_colors = {
        e25.METHOD_GA: "#D55E00",
        e25.METHOD_DYNAMIC: "#009E73",
        e25.METHOD_A3C: "#CC79A7",
    }
    baseline_markers = {
        e25.METHOD_GA: "s",
        e25.METHOD_DYNAMIC: "^",
        e25.METHOD_A3C: "D",
    }
    metrics = (
        ("strict_completion_rate", "Strict completion (%) ↑", 100.0),
        ("mean_absolute_error", "Placement MAE (steps) ↓", 1.0),
        (
            "physical_rehandles_per_100_required_deliveries",
            "Rehandles per 100 ↓",
            1.0,
        ),
        ("steps_per_delivery", "Steps per delivery ↓", 1.0),
    )
    levels = ("low", "medium", "high")
    sizes = (5, 8, 10)
    norm = Normalize(vmin=min(LAMBDAS), vmax=max(LAMBDAS))
    cmap = plt.get_cmap("Blues")
    lookup = {
        (
            item["method"],
            item.get("preference_lambda"),
            item["occupancy_level"],
            item["yard_size"],
        ): item
        for item in summary
    }
    fig, axes = plt.subplots(4, 3, figsize=(8.4, 8.7), sharex=True, sharey="row")
    for column, level in enumerate(levels):
        axes[0, column].set_title(f"{level.capitalize()} initial occupancy", fontsize=10)
        for row_index, (metric, ylabel, scale) in enumerate(metrics):
            axis = axes[row_index, column]
            for value in LAMBDAS:
                selected = [
                    lookup.get((METHOD_VCG, round(value, 8), level, size))
                    for size in sizes
                ]
                numeric = np.asarray(
                    [
                        np.nan
                        if item is None or item.get(metric) is None
                        else scale * float(item[metric])
                        for item in selected
                    ],
                    dtype=float,
                )
                emphasized = value in (LAMBDAS[0], HISTORICAL_LAMBDA, LAMBDAS[-1])
                axis.plot(
                    sizes,
                    numeric,
                    color=cmap(0.25 + 0.70 * norm(value)),
                    linewidth=1.45 if emphasized else 0.85,
                    marker="o",
                    markersize=3.4 if emphasized else 2.2,
                    alpha=0.95 if emphasized else 0.68,
                    zorder=2,
                )
            for method in BASELINE_METHODS:
                selected = [
                    lookup.get((method, None, level, size)) for size in sizes
                ]
                values = [
                    np.nan
                    if item is None or item.get(metric) is None
                    else scale * float(item[metric])
                    for item in selected
                ]
                lows = []
                highs = []
                for item, value in zip(selected, values):
                    if item is None or np.isnan(value):
                        lows.append(np.nan)
                        highs.append(np.nan)
                    elif metric == "strict_completion_rate":
                        lows.append(100.0 * float(item["strict_completion_min"]))
                        highs.append(100.0 * float(item["strict_completion_max"]))
                    else:
                        lows.append(float(item[f"{metric}_instance_min"]))
                        highs.append(float(item[f"{metric}_instance_max"]))
                axis.plot(
                    sizes,
                    values,
                    color=baseline_colors[method],
                    marker=baseline_markers[method],
                    linewidth=1.8,
                    markersize=4.5,
                    zorder=4,
                )
                axis.fill_between(
                    sizes,
                    lows,
                    highs,
                    color=baseline_colors[method],
                    alpha=0.08,
                    linewidth=0,
                )
            axis.grid(axis="y", color="#D9D9D9", linewidth=0.7, alpha=0.7)
            axis.spines[["top", "right"]].set_visible(False)
            axis.set_xticks(sizes, ["5×5", "8×8", "10×10"])
            if column == 0:
                axis.set_ylabel(ylabel, fontsize=9)
            if row_index == len(metrics) - 1:
                axis.set_xlabel("Yard size", fontsize=9)
            if metric == "strict_completion_rate":
                axis.set_ylim(-3, 105)
    handles = [
        Line2D([0], [0], color="#4A90C2", marker="o", label="VCG preference family"),
        *[
            Line2D(
                [0],
                [0],
                color=baseline_colors[method],
                marker=baseline_markers[method],
                label=METHOD_LABELS[method],
            )
            for method in BASELINE_METHODS
        ],
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=8.3)
    cax = fig.add_axes([0.91, 0.16, 0.012, 0.68])
    colorbar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax)
    colorbar.set_label(r"VCG preference $\lambda$", fontsize=8.5)
    colorbar.ax.tick_params(labelsize=7.5)
    fig.subplots_adjust(
        left=0.105,
        right=0.885,
        top=0.96,
        bottom=0.09,
        hspace=0.22,
        wspace=0.12,
    )
    pdf = output / f"{FIGURE_NAME}.pdf"
    png = output / f"{FIGURE_NAME}.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=220, bbox_inches="tight")
    plt.close(fig)
    data = _hashed(
        {
            "schema_version": SCHEMA_VERSION,
            "protocol": PROTOCOL,
            "summary": list(summary),
            "visual_encoding": {
                "columns": list(levels),
                "rows": [item[0] for item in metrics],
                "x": list(sizes),
                "vcg_line_color": "preference_lambda",
                "baseline_bands": (
                    "minimum_to_maximum_across_three_EpisodeInstance_means"
                ),
            },
        },
        "data_sha256",
    )
    data_path = output / f"{FIGURE_NAME}-data.json"
    _atomic_json(data_path, data)
    return {"pdf": str(pdf), "png": str(png), "data": str(data_path)}


def analyze(output: Path, *, partial: bool = False) -> dict:
    output = output.resolve()
    contract, parents = authenticate(output)
    e25_contract, _e25_report, e13_contract, _e13_manifest, _e13_report = parents
    new_vcg = _new_vcg_rows(output, contract)
    if len(new_vcg) > EXPECTED_NEW_VCG_ROWS:
        raise E25V2Error("E25-v2 contains more rows than the frozen grid")
    if not partial and len(new_vcg) != EXPECTED_NEW_VCG_ROWS:
        raise E25V2Error(
            f"incomplete E25-v2 grid: {len(new_vcg)}/{EXPECTED_NEW_VCG_ROWS}"
        )
    baselines = _baseline_rows(e25_contract)
    rows = [*new_vcg, *baselines]
    historical_parity = _historical_parity(output, contract, e13_contract)
    summary = _summary(rows)
    expected_summaries = (len(LAMBDAS) + len(BASELINE_METHODS)) * len(SCENARIOS)
    if not partial and len(summary) != expected_summaries:
        raise E25V2Error(
            f"complete E25-v2 analysis has {len(summary)}/{expected_summaries} summaries"
        )
    figures = _render(summary, output)
    report = _hashed(
        {
            "schema_version": SCHEMA_VERSION,
            "protocol": PROTOCOL,
            "contract_sha256": contract["contract_sha256"],
            "status": "partial" if partial else "complete",
            "paper_evidence": not partial,
            "post_hoc": True,
            "preference_lambdas": list(LAMBDAS),
            "model_seed": e13.MODEL_SEED,
            "expected_new_vcg_rows": EXPECTED_NEW_VCG_ROWS,
            "observed_new_vcg_rows": len(new_vcg),
            "historical_vcg_rows_for_parity_only": EXPECTED_HISTORICAL_VCG_ROWS,
            "reused_baseline_rows": len(baselines),
            "reused_controls_are_independent_replications": False,
            "historical_lambda_010_parity": historical_parity,
            "summaries": summary,
            "vcg_failure_counts_by_lambda": {
                str(value): dict(
                    Counter(
                        row["failure_class"]
                        for row in new_vcg
                        if np.isclose(float(row["preference_lambda"]), value)
                    )
                )
                for value in LAMBDAS
            },
            "quality_scope": "strict-complete runs only",
            "statistical_unit": (
                "EpisodeInstance after within-instance nuisance averaging"
            ),
            "figures": figures,
            "runtime": _runtime(),
            "training_runs": 0,
        },
        "report_sha256",
    )
    _atomic_json(output / REPORT_NAME, report)
    lines = [
        "# E25-v2 full VCG preference-family scale comparison",
        "",
        "| Method | λ | Yard | Occupancy | Initial + arriving | Strict | MAE | Within ±20 | Rehandles/100 | Steps/delivery |",
        "|---|---:|---:|---|---:|---:|---:|---:|---:|---:|",
    ]

    def fmt(item: Mapping, name: str, multiplier: float = 1.0) -> str:
        value = item.get(name)
        return "—" if value is None else f"{multiplier * float(value):.2f}"

    for item in summary:
        value = item.get("preference_lambda")
        lines.append(
            f"| {METHOD_LABELS[item['method']]} | "
            f"{'—' if value is None else f'{float(value):.4g}'} | "
            f"{item['yard_size']}×{item['yard_size']} | "
            f"{item['occupancy_level']} ({item['initial_occupancy_ratio']:.1%}) | "
            f"{item['initially_stored_jobs']} + {item['arriving_jobs']} | "
            f"{fmt(item, 'strict_completion_rate', 100.0)}% | "
            f"{fmt(item, 'mean_absolute_error')} | "
            f"{fmt(item, 'within_target_window_rate', 100.0)}% | "
            f"{fmt(item, 'physical_rehandles_per_100_required_deliveries')} | "
            f"{fmt(item, 'steps_per_delivery')} |"
        )
    lines.extend(
        [
            "",
            "VCG uses the frozen seed-0 model and the same corrected executor at all ten predeclared E1 preferences. The historical E13 lambda=.10 rows are parity diagnostics only; all baseline rows are authenticated reused controls, not independent replications. Quality is conditional on strict completion. Each workload contains exactly two arriving jobs; all remaining jobs are initially stored.",
            "",
        ]
    )
    (output / TABLE_NAME).write_text("\n".join(lines), encoding="utf-8")
    return {
        "status": report["status"],
        "new_vcg_rows": len(new_vcg),
        "historical_vcg_parity_coordinates": historical_parity["matched_coordinates"],
        "reused_baseline_rows": len(baselines),
        "report": str((output / REPORT_NAME).resolve()),
        "table": str((output / TABLE_NAME).resolve()),
        "figures": figures,
        "training_runs": 0,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=(
            "prepare",
            "run-pilot",
            "run-one",
            "run-lambda",
            "run-all",
            "analyze",
            "analyze-partial",
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--preference-lambda", type=float)
    parser.add_argument("--scenario", choices=SCENARIOS)
    parser.add_argument("--seed", type=int, choices=INSTANCE_SEEDS)
    return parser


def _validated_new_lambda(value: Optional[float]) -> float:
    if value is None:
        raise E25V2Error("a --preference-lambda is required")
    matches = [item for item in NEW_LAMBDAS if np.isclose(item, float(value))]
    if len(matches) != 1:
        raise E25V2Error(f"unsupported preference lambda: {value}")
    return matches[0]


def main(argv: Optional[Sequence[str]] = None) -> dict:
    args = build_parser().parse_args(argv)
    output = args.output_dir.resolve()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    if args.command == "prepare":
        result = prepare(output)
    elif args.command == "run-pilot":
        result = run_pilot(output)
    elif args.command == "run-one":
        if args.scenario is None or args.seed is None:
            raise E25V2Error("run-one requires --scenario and --seed")
        value = _validated_new_lambda(args.preference_lambda)
        row = run_one(output, value, args.scenario, args.seed)
        result = {"status": "complete", "row": row, "training_runs": 0}
    elif args.command == "run-lambda":
        value = _validated_new_lambda(args.preference_lambda)
        result = run_values(output, (value,), scope="complete_lambda_slice")
    elif args.command == "run-all":
        result = run_values(output, NEW_LAMBDAS, scope="complete_new_vcg_grid")
    else:
        result = analyze(output, partial=args.command == "analyze-partial")
    print(json.dumps(_json_form(result), indent=2, sort_keys=True), flush=True)
    return result


if __name__ == "__main__":
    main()
