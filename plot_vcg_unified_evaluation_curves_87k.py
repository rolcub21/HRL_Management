#!/usr/bin/env python3
"""Plot cumulative frozen-policy performance on the common 87k panel."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
from statistics import fmean
from typing import Any, Callable, Mapping, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import run_kim2020_final86_comparison as kim_final
import run_vcg_final86_four_method as final86
import vcg_v2_3_kim2020_supplement_evaluation as kim85


PROJECT_ROOT = Path(__file__).resolve().parent
UNIFIED_ROOT = PROJECT_ROOT / "results/vcg-unified-frozen-lambda-confirmation-87k"
BASELINE_ROOT = PROJECT_ROOT / "results/vcg-unified-matched-baselines-87k"
OUTPUT_ROOT = PROJECT_ROOT / "results/vcg-unified-evaluation-curves-87k"
INSTANCE_SEEDS = tuple(range(87_000, 87_030))

METRICS = (
    ("dense_return", "Cumulative dense return ↑"),
    ("mean_absolute_error", "Cumulative MAE ↓"),
    (
        "physical_rehandles_per_100_required_deliveries",
        "Cumulative physical rehandles / 100 ↓",
    ),
)

METHOD_SPECS = (
    ("lambda0", "VCG  λ = 0", "#0072B2", "o"),
    ("lambda005", "VCG + handling  λ = 0.05", "#009E73", "D"),
    (final86.V11_METHOD, "VCG 1.1 (legacy)", "#E69F00", "s"),
    (final86.DYNAMIC_METHOD, "Dynamic PSLAP", "#CC79A7", "^"),
    (final86.GA_METHOD, "Capacity-aware GA", "#D55E00", "v"),
    (kim85.KIM_STOCHASTIC, "Kim2020 adaptation", "#6F4E7C", "P"),
)


class CurvePlotError(RuntimeError):
    pass


def _metric(row: Mapping[str, Any], name: str) -> float:
    if name == "dense_return":
        value = row.get("dense_return", row.get("dense_objective_return"))
    elif name == "physical_rehandles_per_100_required_deliveries":
        value = row.get(name)
        if value is None:
            physical = row.get(
                "physical_rehandles", row.get("physical_storage_relocations")
            )
            required = row.get("required_deliveries")
            if physical is not None and required:
                value = 100.0 * float(physical) / float(required)
    else:
        value = row.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CurvePlotError(f"row metric is missing or nonnumeric: {name}")
    value = float(value)
    if not math.isfinite(value):
        raise CurvePlotError(f"row metric is nonfinite: {name}")
    return value


def _load_unified(lambda_text: str) -> list[dict]:
    rows = []
    for model_seed in (15, 16, 17):
        path = (
            UNIFIED_ROOT
            / "validation-ledger"
            / f"seed-{model_seed}"
            / f"lambda-{lambda_text}.json"
        )
        ledger = final86._load_json(path, name="unified 87k validation ledger")
        final86._self_hash(
            ledger, "ledger_sha256", name="unified 87k validation ledger"
        )
        observed = ledger.get("rows")
        if (
            ledger.get("strict_integrity_gate") is not True
            or ledger.get("training_or_learning") is not False
            or not isinstance(observed, list)
            or len(observed) != 120
        ):
            raise CurvePlotError(f"unified ledger grid is incomplete: {path}")
        for source_row in observed:
            row = dict(source_row)
            row["model_seed"] = model_seed
            row["strict_safe_complete"] = bool(
                row.get("strict_method_success") is True
                and row.get("completion_rate") == 1.0
                and row.get("delivery_count") == row.get("required_deliveries") == 8
                and row.get("all_selected_candidates_exact_safe") is True
                and row.get("illegal_drops") == 0
                and row.get("fallbacks") == 0
                and row.get("witness_mismatches") == 0
                and row.get("method_failure_reason") is None
                and row.get("evaluation_learning") is False
                and row.get("training_agent_unchanged") is True
                and row.get("validation_batch_state_unchanged") is True
            )
            if row["strict_safe_complete"] is not True:
                raise CurvePlotError(
                    f"unified strict ledger contains an unsafe row: {path}"
                )
            rows.append(row)
    return rows


def _baseline_ledger_path(
    method: str, seed: int, model_seed: Optional[int], rng_index: Optional[int]
) -> Path:
    model = "none" if model_seed is None else str(model_seed)
    rng = "none" if rng_index is None else str(rng_index)
    return (
        BASELINE_ROOT
        / "run-ledger"
        / method
        / f"instance-{seed}-model-{model}-rng-{rng}.json"
    )


def _load_baseline_method(method: str) -> Optional[list[dict]]:
    grid: list[tuple[int, Optional[int], Optional[int]]] = []
    if method == final86.V11_METHOD:
        grid = [
            (seed, model_seed, None)
            for model_seed in final86.V11_MODEL_SEEDS
            for seed in INSTANCE_SEEDS
        ]
    elif method == final86.DYNAMIC_METHOD:
        grid = [(seed, None, None) for seed in INSTANCE_SEEDS]
    elif method == final86.GA_METHOD:
        grid = [
            (seed, None, rng_index)
            for seed in INSTANCE_SEEDS
            for rng_index in final86.RNG_INDICES
        ]
    else:
        raise CurvePlotError(f"unknown baseline method: {method}")
    paths = [
        _baseline_ledger_path(method, seed, model_seed, rng_index)
        for seed, model_seed, rng_index in grid
    ]
    present = [path.is_file() for path in paths]
    if not any(present):
        return None
    if not all(present):
        raise CurvePlotError(
            f"{method} ledger grid is partial ({sum(present)}/{len(paths)}); resume it"
        )
    rows = []
    for path, (seed, model_seed, rng_index) in zip(paths, grid):
        ledger = final86._load_json(path, name=f"{method} 87k ledger")
        final86._self_hash(ledger, "ledger_sha256", name=f"{method} 87k ledger")
        row = ledger.get("run")
        if not isinstance(row, Mapping):
            raise CurvePlotError(f"{method} ledger lacks a run row")
        if (
            row.get("instance_seed") != seed
            or row.get("model_seed") != model_seed
            or row.get("rng_index") != rng_index
        ):
            raise CurvePlotError(f"{method} ledger identity mismatch: {path}")
        rows.append(dict(row))
    return rows


def _load_kim() -> Optional[list[dict]]:
    grid = [
        (model_seed, seed, rollout)
        for model_seed in kim_final.MODEL_SEEDS
        for seed in INSTANCE_SEEDS
        for rollout in kim_final.ROLLOUTS
    ]
    paths = [
        BASELINE_ROOT
        / "run-ledger"
        / f"seed-{model_seed}"
        / f"instance-{seed}-roll-{rollout}.json"
        for model_seed, seed, rollout in grid
    ]
    present = [path.is_file() for path in paths]
    if not any(present):
        return None
    if not all(present):
        raise CurvePlotError(
            f"Kim ledger grid is partial ({sum(present)}/{len(paths)}); resume it"
        )
    rows = []
    for path, (model_seed, seed, rollout) in zip(paths, grid):
        ledger = final86._load_json(path, name="Kim 87k ledger")
        final86._self_hash(ledger, "ledger_sha256", name="Kim 87k ledger")
        row = ledger.get("run")
        if not isinstance(row, Mapping):
            raise CurvePlotError("Kim ledger lacks a run row")
        if (
            row.get("model_seed") != model_seed
            or row.get("instance_seed") != seed
            or row.get("rollout_index") != rollout
        ):
            raise CurvePlotError(f"Kim ledger identity mismatch: {path}")
        kim85._verify_self_hash(row, "row_sha256", label="Kim 87k row")
        rows.append(dict(row))
    return rows


def _is_safe(row: Mapping[str, Any]) -> bool:
    return row.get("strict_safe_complete") is True


def _reduce_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    replicate_key: Callable[[Mapping[str, Any]], Any],
    expected_replicates: Sequence[Any],
    nuisance_count: int,
) -> dict:
    grouped: dict[tuple[int, Any], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        seed = row.get("instance_seed")
        if type(seed) is not int or seed not in INSTANCE_SEEDS:
            raise CurvePlotError("curve row has an invalid instance seed")
        grouped[(seed, replicate_key(row))].append(row)

    expected_keys = {
        (seed, replicate) for seed in INSTANCE_SEEDS for replicate in expected_replicates
    }
    if set(grouped) != expected_keys:
        raise CurvePlotError("curve row grid is incomplete or contains duplicates")
    if any(len(group) != nuisance_count for group in grouped.values()):
        raise CurvePlotError("curve nuisance-replication count mismatch")

    failed_instances = []
    replicate_points: dict[Any, dict[str, list[float]]] = {
        replicate: {name: [] for name, _ in METRICS}
        for replicate in expected_replicates
    }
    for seed in INSTANCE_SEEDS:
        instance_safe = all(
            all(_is_safe(row) for row in grouped[(seed, replicate)])
            for replicate in expected_replicates
        )
        if not instance_safe:
            failed_instances.append(seed)
        for replicate in expected_replicates:
            group = grouped[(seed, replicate)]
            for name, _ in METRICS:
                if all(_is_safe(row) for row in group):
                    replicate_points[replicate][name].append(
                        fmean(_metric(row, name) for row in group)
                    )
                else:
                    replicate_points[replicate][name].append(float("nan"))

    first_failure_index = (
        min(INSTANCE_SEEDS.index(seed) for seed in failed_instances)
        if failed_instances
        else len(INSTANCE_SEEDS)
    )
    usable = first_failure_index
    curves = {}
    for name, _ in METRICS:
        replicate_curves = []
        for replicate in expected_replicates:
            values = np.asarray(replicate_points[replicate][name][:usable], dtype=float)
            if values.size and not np.isfinite(values).all():
                raise CurvePlotError("safe curve prefix contains a nonfinite metric")
            cumulative = np.cumsum(values) / np.arange(1, values.size + 1)
            replicate_curves.append(cumulative)
        matrix = (
            np.vstack(replicate_curves)
            if replicate_curves and usable
            else np.empty((len(expected_replicates), 0))
        )
        curves[name] = {
            "mean": matrix.mean(axis=0).tolist() if usable else [],
            "minimum": matrix.min(axis=0).tolist() if usable else [],
            "maximum": matrix.max(axis=0).tolist() if usable else [],
        }
    return {
        "curves": curves,
        "replicate_count": len(expected_replicates),
        "nuisance_count_within_replicate_instance": nuisance_count,
        "failed_instances": failed_instances,
        "first_failure_k": first_failure_index + 1 if failed_instances else None,
        "numeric_prefix_length": usable,
    }


def build_curve_data() -> dict:
    data = {
        "lambda0": _reduce_rows(
            _load_unified("0.00"),
            replicate_key=lambda row: row["model_seed"],
            expected_replicates=(15, 16, 17),
            nuisance_count=4,
        ),
        "lambda005": _reduce_rows(
            _load_unified("0.05"),
            replicate_key=lambda row: row["model_seed"],
            expected_replicates=(15, 16, 17),
            nuisance_count=4,
        ),
    }

    for method, replicate_key, expected_replicates, nuisance_count in (
        (
            final86.V11_METHOD,
            lambda row: row["model_seed"],
            final86.V11_MODEL_SEEDS,
            1,
        ),
        (
            final86.DYNAMIC_METHOD,
            lambda row: "deterministic",
            ("deterministic",),
            1,
        ),
        (
            final86.GA_METHOD,
            lambda row: row["rng_index"],
            final86.RNG_INDICES,
            1,
        ),
    ):
        rows = _load_baseline_method(method)
        if rows is not None:
            data[method] = _reduce_rows(
                rows,
                replicate_key=replicate_key,
                expected_replicates=expected_replicates,
                nuisance_count=nuisance_count,
            )

    kim_rows = _load_kim()
    if kim_rows is not None:
        data[kim85.KIM_STOCHASTIC] = _reduce_rows(
            kim_rows,
            replicate_key=lambda row: row["model_seed"],
            expected_replicates=kim_final.MODEL_SEEDS,
            nuisance_count=5,
        )

    # The completed unified report is an independent endpoint check.
    report = final86._load_json(
        UNIFIED_ROOT / "confirmation-report.json", name="unified confirmation report"
    )
    final86._self_hash(report, "report_sha256", name="unified confirmation report")
    for key, report_key in (("lambda0", "lambda0"), ("lambda005", "lambda005")):
        if data[key]["numeric_prefix_length"] != 30:
            raise CurvePlotError("completed unified curve unexpectedly stops early")
        endpoint = report["aggregate"][report_key]
        checks = {
            "dense_return": endpoint["dense_return"],
            "mean_absolute_error": endpoint["mean_absolute_error"],
            "physical_rehandles_per_100_required_deliveries": endpoint[
                "physical_rehandles_per_100"
            ],
        }
        for metric, expected in checks.items():
            observed = data[key]["curves"][metric]["mean"][-1]
            if not math.isclose(observed, float(expected), rel_tol=0.0, abs_tol=1e-10):
                raise CurvePlotError(f"unified k=30 endpoint mismatch: {key}/{metric}")
    return data


def render(output_root: Path = OUTPUT_ROOT) -> dict:
    data = build_curve_data()
    output = Path(output_root).resolve()
    output.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 11,
            "legend.fontsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(15.6, 4.7), sharex=True)
    failure_markers = []
    for key, label, color, marker in METHOD_SPECS:
        if key not in data:
            continue
        record = data[key]
        n = record["numeric_prefix_length"]
        display_label = label
        if record["first_failure_k"] is not None:
            display_label += f" (stops at k={record['first_failure_k']})"
        for axis, (metric, ylabel) in zip(axes, METRICS):
            curve = record["curves"][metric]
            x = np.arange(1, n + 1)
            mean = np.asarray(curve["mean"], dtype=float)
            low = np.asarray(curve["minimum"], dtype=float)
            high = np.asarray(curve["maximum"], dtype=float)
            if n:
                axis.plot(
                    x,
                    mean,
                    color=color,
                    linewidth=2.35 if key in ("lambda0", "lambda005") else 1.85,
                    marker=marker,
                    markersize=4.5,
                    markevery=[index for index in (0, 4, 9, 14, 19, 24, 29) if index < n],
                    label=display_label if axis is axes[0] else None,
                    zorder=4,
                )
                if record["replicate_count"] > 1:
                    axis.fill_between(x, low, high, color=color, alpha=0.10, linewidth=0)
                if record["first_failure_k"] is not None:
                    axis.scatter(
                        [record["first_failure_k"]],
                        [mean[-1]],
                        marker="x",
                        s=70,
                        linewidth=2.2,
                        color=color,
                        zorder=6,
                    )
            elif record["first_failure_k"] is not None:
                failure_markers.append((axis, record["first_failure_k"], color))
            axis.set_ylabel(ylabel)

    for axis in axes:
        axis.set_xlim(1, 30)
        axis.set_xticks((1, 5, 10, 15, 20, 25, 30))
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.8, alpha=0.8)
        axis.set_xlabel("Cumulative evaluation EpisodeInstances  k")
    for axis, x, color in failure_markers:
        y0, y1 = axis.get_ylim()
        axis.scatter([x], [(y0 + y1) / 2], marker="x", s=70, linewidth=2.2, color=color)

    fig.suptitle(
        "Frozen-policy performance as the common 87k evaluation panel accumulates",
        y=0.99,
        fontsize=13,
    )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.025),
        ncol=3,
        frameon=False,
    )
    fig.tight_layout(rect=(0, 0.11, 1, 0.95))

    png = output / "vcg-unified-matched-evaluation-curves-87k.png"
    pdf = output / "vcg-unified-matched-evaluation-curves-87k.pdf"
    fig.savefig(png, dpi=240, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "figure_role": "cumulative_frozen_policy_evaluation_curve",
        "not_a_training_learning_curve": True,
        "panel_order": list(INSTANCE_SEEDS),
        "metrics": [name for name, _ in METRICS],
        "band": "descriptive min-max across model seeds or optimizer realizations",
        "complete_case_filtering_used": False,
        "methods": {
            key: {
                "numeric_prefix_length": value["numeric_prefix_length"],
                "first_failure_k": value["first_failure_k"],
                "failed_instances": value["failed_instances"],
                "endpoint": {
                    metric: (
                        value["curves"][metric]["mean"][-1]
                        if value["curves"][metric]["mean"]
                        else None
                    )
                    for metric, _ in METRICS
                },
            }
            for key, value in data.items()
        },
        "png": str(png),
        "pdf": str(pdf),
    }
    summary["summary_sha256"] = final86._digest(summary)
    final86._atomic_json(output / "curve-summary.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    print(json.dumps(render(args.output_root), indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
