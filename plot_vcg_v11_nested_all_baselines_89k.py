#!/usr/bin/env python3
"""Plot nested VCG endpoints and all comparators on the common 89k panel."""

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
import run_vcg_v11_nested_all_baselines_89k as runner
import run_vcg_v11_nested_lambda_frontier_confirmation_89k as confirmation
import vcg_v2_3_kim2020_supplement_evaluation as kim85


PROJECT_ROOT = Path(__file__).resolve().parent
CONFIRMATION_ROOT = PROJECT_ROOT / "results/vcg-v1-1-nested-lambda-frontier-confirmation-89k"
BASELINE_ROOT = PROJECT_ROOT / "results/vcg-v1-1-nested-lambda-matched-baselines-89k"
INSTANCE_SEEDS = runner.INSTANCE_SEEDS

METRICS = (
    ("dense_return", "Cumulative dense return ↑"),
    ("mean_absolute_error", "Cumulative MAE ↓"),
    (
        "physical_rehandles_per_100_required_deliveries",
        "Cumulative physical rehandles / 100 ↓",
    ),
)
METHOD_SPECS = (
    ("lambda0", "VCG  (λ = 0)", "#0072B2", "o"),
    ("lambda02", "VCG + handling  (λ = 0.2)", "#009E73", "D"),
    (final86.V23_METHOD, "Historical VCG 2.3", "#56B4E9", "X"),
    (final86.DYNAMIC_METHOD, "Dynamic PSLAP", "#CC79A7", "^"),
    (final86.GA_METHOD, "Capacity-aware GA", "#D55E00", "v"),
    (kim85.KIM_STOCHASTIC, "Kim2020 adaptation", "#6F4E7C", "P"),
)


class Plot89Error(RuntimeError):
    pass


def _metric(row: Mapping[str, Any], name: str) -> float:
    if name == "dense_return":
        value = row.get("dense_return", row.get("dense_objective_return"))
    elif name == "physical_rehandles_per_100_required_deliveries":
        value = row.get(name, row.get("physical_rehandles_per_100"))
        if value is None:
            count = row.get(
                "physical_rehandles", row.get("physical_storage_relocations")
            )
            required = row.get("required_deliveries", 8)
            if count is not None:
                value = 100.0 * float(count) / float(required)
    else:
        value = row.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise Plot89Error(f"missing metric: {name}")
    result = float(value)
    if not math.isfinite(result):
        raise Plot89Error(f"nonfinite metric: {name}")
    return result


def _load_confirmation() -> tuple[dict, dict, dict[int, dict]]:
    final86._configure_runtime()
    contract = confirmation.authenticate_contract(PROJECT_ROOT, CONFIRMATION_ROOT)
    manifest = confirmation.authenticate_instance_manifest(PROJECT_ROOT, CONFIRMATION_ROOT)
    report = confirmation._load_json(
        CONFIRMATION_ROOT / confirmation.REPORT_NAME, label="89k confirmation report"
    )
    confirmation._verify_hash(report, "report_sha256", label="89k confirmation report")
    if (
        report.get("status") != "passed"
        or report.get("row_count") != confirmation.EXPECTED_ROWS
        or report.get("strict_safe_row_count") != confirmation.EXPECTED_ROWS
        or report.get("contract_sha256") != contract["contract_sha256"]
        or report.get("manifest_sha256") != manifest["manifest_sha256"]
    ):
        raise Plot89Error("nested VCG 89k confirmation is not eligible")
    records = {int(record["instance_seed"]): record for record in manifest["instances"]}
    return contract, manifest, records


def _load_nested(value: float) -> list[dict]:
    contract, manifest, records = _load_confirmation()
    rows = []
    for model_seed in confirmation.MODEL_SEEDS:
        for instance_seed in INSTANCE_SEEDS:
            path = confirmation._ledger_path(
                CONFIRMATION_ROOT, value, model_seed, instance_seed
            )
            ledger = confirmation._load_json(path, label="nested 89k ledger")
            row = confirmation._validate_ledger(
                ledger,
                contract=contract,
                manifest=manifest,
                value=value,
                model_seed=model_seed,
                record=records[instance_seed],
            )
            if (
                row.get("model_seed") != model_seed
                or row.get("instance_seed") != instance_seed
                or float(row.get("handling_lambda")) != value
            ):
                raise Plot89Error("nested ledger identity mismatch")
            rows.append(row)
    return rows


def _baseline_path(
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


def _load_baseline(method: str) -> list[dict]:
    if method == final86.V23_METHOD:
        grid = [
            (seed, model, rng)
            for model in final86.V23_MODEL_SEEDS
            for seed in INSTANCE_SEEDS
            for rng in final86.RNG_INDICES
        ]
    elif method == final86.DYNAMIC_METHOD:
        grid = [(seed, None, None) for seed in INSTANCE_SEEDS]
    elif method == final86.GA_METHOD:
        grid = [
            (seed, None, rng)
            for seed in INSTANCE_SEEDS
            for rng in final86.RNG_INDICES
        ]
    else:
        raise Plot89Error(f"unknown baseline: {method}")
    rows = []
    for seed, model, rng in grid:
        path = _baseline_path(method, seed, model, rng)
        if not path.is_file():
            raise Plot89Error(f"missing baseline ledger: {path}")
        ledger = final86._load_json(path, name="89k baseline ledger")
        final86._self_hash(ledger, "ledger_sha256", name="89k baseline ledger")
        row = dict(ledger.get("run", {}))
        if (
            row.get("instance_seed") != seed
            or row.get("model_seed") != model
            or row.get("rng_index") != rng
        ):
            raise Plot89Error(f"baseline ledger identity mismatch: {path}")
        rows.append(row)
    return rows


def _load_kim() -> list[dict]:
    rows = []
    for model_seed in kim_final.MODEL_SEEDS:
        for seed in INSTANCE_SEEDS:
            for rollout in kim_final.ROLLOUTS:
                path = (
                    BASELINE_ROOT
                    / "run-ledger"
                    / f"seed-{model_seed}"
                    / f"instance-{seed}-roll-{rollout}.json"
                )
                if not path.is_file():
                    raise Plot89Error(f"missing Kim ledger: {path}")
                ledger = final86._load_json(path, name="Kim 89k ledger")
                final86._self_hash(ledger, "ledger_sha256", name="Kim 89k ledger")
                row = dict(ledger.get("run", {}))
                kim85._verify_self_hash(row, "row_sha256", label="Kim 89k row")
                if (
                    row.get("model_seed") != model_seed
                    or row.get("instance_seed") != seed
                    or row.get("rollout_index") != rollout
                ):
                    raise Plot89Error("Kim ledger identity mismatch")
                rows.append(row)
    return rows


def _reduce(
    rows: Sequence[Mapping[str, Any]],
    *,
    replicate_key: Callable[[Mapping[str, Any]], Any],
    expected_replicates: Sequence[Any],
    nuisance_count: int,
) -> dict:
    grouped = defaultdict(list)
    for row in rows:
        seed = row.get("instance_seed")
        if type(seed) is not int or seed not in INSTANCE_SEEDS:
            raise Plot89Error("invalid instance seed")
        grouped[(seed, replicate_key(row))].append(row)
    expected = {
        (seed, replicate)
        for seed in INSTANCE_SEEDS
        for replicate in expected_replicates
    }
    if set(grouped) != expected or any(
        len(grouped[key]) != nuisance_count for key in expected
    ):
        raise Plot89Error("method row grid is incomplete or duplicated")
    failed = []
    points = {
        replicate: {metric: [] for metric, _ in METRICS}
        for replicate in expected_replicates
    }
    for seed in INSTANCE_SEEDS:
        seed_safe = all(
            all(row.get("strict_safe_complete") is True for row in grouped[(seed, rep)])
            for rep in expected_replicates
        )
        if not seed_safe:
            failed.append(seed)
        for rep in expected_replicates:
            group = grouped[(seed, rep)]
            group_safe = all(row.get("strict_safe_complete") is True for row in group)
            for metric, _ in METRICS:
                points[rep][metric].append(
                    fmean([_metric(row, metric) for row in group])
                    if group_safe
                    else float("nan")
                )
    usable = (
        min(INSTANCE_SEEDS.index(seed) for seed in failed)
        if failed
        else len(INSTANCE_SEEDS)
    )
    curves = {}
    for metric, _ in METRICS:
        paths = []
        for rep in expected_replicates:
            values = np.asarray(points[rep][metric][:usable], dtype=float)
            if values.size and not np.isfinite(values).all():
                raise Plot89Error("safe prefix contains nonfinite values")
            paths.append(np.cumsum(values) / np.arange(1, values.size + 1))
        matrix = np.vstack(paths) if usable else np.empty((len(paths), 0))
        curves[metric] = {
            "mean": matrix.mean(axis=0).tolist() if usable else [],
            "minimum": matrix.min(axis=0).tolist() if usable else [],
            "maximum": matrix.max(axis=0).tolist() if usable else [],
        }
    return {
        "curves": curves,
        "replicate_count": len(expected_replicates),
        "failed_instances": failed,
        "first_failure_k": INSTANCE_SEEDS.index(failed[0]) + 1 if failed else None,
        "numeric_prefix_length": usable,
        "whole_method_eligible": not failed,
    }


def build_curve_data() -> dict:
    data = {
        "lambda0": _reduce(
            _load_nested(0.0),
            replicate_key=lambda row: row["model_seed"],
            expected_replicates=confirmation.MODEL_SEEDS,
            nuisance_count=1,
        ),
        "lambda02": _reduce(
            _load_nested(0.2),
            replicate_key=lambda row: row["model_seed"],
            expected_replicates=confirmation.MODEL_SEEDS,
            nuisance_count=1,
        ),
        final86.V23_METHOD: _reduce(
            _load_baseline(final86.V23_METHOD),
            replicate_key=lambda row: row["model_seed"],
            expected_replicates=final86.V23_MODEL_SEEDS,
            nuisance_count=4,
        ),
        final86.DYNAMIC_METHOD: _reduce(
            _load_baseline(final86.DYNAMIC_METHOD),
            replicate_key=lambda row: "deterministic",
            expected_replicates=("deterministic",),
            nuisance_count=1,
        ),
        final86.GA_METHOD: _reduce(
            _load_baseline(final86.GA_METHOD),
            replicate_key=lambda row: row["rng_index"],
            expected_replicates=final86.RNG_INDICES,
            nuisance_count=1,
        ),
        kim85.KIM_STOCHASTIC: _reduce(
            _load_kim(),
            replicate_key=lambda row: row["model_seed"],
            expected_replicates=kim_final.MODEL_SEEDS,
            nuisance_count=5,
        ),
    }
    report = confirmation._load_json(
        CONFIRMATION_ROOT / confirmation.REPORT_NAME, label="89k confirmation report"
    )
    report_names = {
        "dense_return": "mean_dense_return",
        "mean_absolute_error": "mean_absolute_error",
        "physical_rehandles_per_100_required_deliveries": "physical_rehandles_per_100",
    }
    for arm, value in (("lambda0", 0.0), ("lambda02", 0.2)):
        for metric, _ in METRICS:
            observed = data[arm]["curves"][metric]["mean"][-1]
            expected = float(
                report["aggregate_metrics"][confirmation._lambda_key(value)][
                    report_names[metric]
                ]
            )
            if not math.isclose(observed, expected, rel_tol=0.0, abs_tol=1e-10):
                raise Plot89Error(f"nested endpoint mismatch: {arm}/{metric}")
    return data


def _save(figure, output: Path, stem: str) -> list[str]:
    paths = []
    for suffix in ("png", "pdf"):
        path = output / f"{stem}.{suffix}"
        figure.savefig(path, dpi=240 if suffix == "png" else None, bbox_inches="tight")
        paths.append(str(path))
    plt.close(figure)
    return paths


def _curves_figure(data: Mapping):
    figure, axes = plt.subplots(1, 3, figsize=(15.7, 4.7), sharex=True)
    for method, label, color, marker in METHOD_SPECS:
        record = data[method]
        n = record["numeric_prefix_length"]
        shown_label = label
        if record["first_failure_k"] is not None:
            shown_label += f" (stops at k={record['first_failure_k']})"
        for axis, (metric, ylabel) in zip(axes, METRICS):
            curve = record["curves"][metric]
            x = np.arange(1, n + 1)
            mean = np.asarray(curve["mean"], dtype=float)
            if n:
                axis.plot(
                    x,
                    mean,
                    color=color,
                    linewidth=2.4 if method in ("lambda0", "lambda02") else 1.8,
                    marker=marker,
                    markersize=4.3,
                    markevery=[i for i in (0, 4, 9, 14, 19, 24, 29) if i < n],
                    label=shown_label if axis is axes[0] else None,
                    zorder=3,
                )
                if record["replicate_count"] > 1:
                    axis.fill_between(
                        x,
                        curve["minimum"],
                        curve["maximum"],
                        color=color,
                        alpha=0.09,
                        linewidth=0,
                    )
                if record["first_failure_k"] is not None:
                    axis.scatter(
                        [record["first_failure_k"]],
                        [mean[-1]],
                        marker="x",
                        s=65,
                        linewidth=2,
                        color=color,
                        zorder=5,
                    )
            axis.set_ylabel(ylabel)
    for axis in axes:
        axis.set_xlim(1, 30)
        axis.set_xticks((1, 5, 10, 15, 20, 25, 30))
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.8, alpha=0.8)
        axis.set_xlabel("Cumulative common EpisodeInstances  k")
    figure.suptitle("Frozen-policy comparison on the common unseen 89k panel", y=0.99)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.025),
        ncol=3,
        frameon=False,
    )
    figure.tight_layout(rect=(0, 0.11, 1, 0.95))
    return figure


def _tradeoff_figure(data: Mapping):
    figure, axis = plt.subplots(figsize=(7.2, 5.0), constrained_layout=True)
    nested_xy = []
    for method, label, color, marker in METHOD_SPECS:
        record = data[method]
        if not record["whole_method_eligible"]:
            continue
        x = record["curves"]["physical_rehandles_per_100_required_deliveries"]["mean"][-1]
        y = record["curves"]["mean_absolute_error"]["mean"][-1]
        axis.scatter(
            x,
            y,
            s=74 if method in ("lambda0", "lambda02") else 58,
            color=color,
            marker=marker,
            edgecolor="white",
            linewidth=0.8,
            label=label,
            zorder=3,
        )
        if method in ("lambda0", "lambda02"):
            nested_xy.append((x, y))
    if len(nested_xy) == 2:
        axis.plot(
            [point[0] for point in nested_xy],
            [point[1] for point in nested_xy],
            color="#6B7280",
            linestyle="--",
            linewidth=1.2,
            zorder=1,
        )
    axis.set_xlabel("Physical rehandles / 100 required deliveries ↓")
    axis.set_ylabel("Mean absolute timing error ↓")
    axis.set_title("Matched operating-point comparison on unseen 89k")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False, ncol=2)
    return figure


def render(output_root: Path = BASELINE_ROOT) -> dict:
    data = build_curve_data()
    output = Path(output_root).resolve()
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 11,
            "legend.fontsize": 8.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    files = []
    files.extend(
        _save(_curves_figure(data), output, "vcg-nested-all-matched-curves-89k")
    )
    files.extend(
        _save(_tradeoff_figure(data), output, "vcg-nested-all-matched-tradeoff-89k")
    )
    summary = {
        "protocol": runner.PROTOCOL,
        "figure_role": "post_confirmation_matched_comparison",
        "not_a_training_learning_curve": True,
        "panel_order": list(INSTANCE_SEEDS),
        "complete_case_filtering_used": False,
        "methods": {
            method: {
                "whole_method_eligible": record["whole_method_eligible"],
                "numeric_prefix_length": record["numeric_prefix_length"],
                "first_failure_k": record["first_failure_k"],
                "failed_instances": record["failed_instances"],
                "endpoint": {
                    metric: (
                        record["curves"][metric]["mean"][-1]
                        if record["whole_method_eligible"]
                        else None
                    )
                    for metric, _ in METRICS
                },
            }
            for method, record in data.items()
        },
        "figures": files,
    }
    summary["summary_sha256"] = final86._digest(summary)
    final86._atomic_json(output / "matched-curve-summary.json", summary)
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=BASELINE_ROOT)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _parser().parse_args(argv)
    print(json.dumps(render(args.output_root), indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
