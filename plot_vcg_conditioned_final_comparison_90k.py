#!/usr/bin/env python3
"""Render the E1 benchmark figure, table, and supplementary diagnostics."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
import math
from pathlib import Path
from statistics import fmean, stdev
from typing import Mapping, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

import run_vcg_conditioned_final_comparison_90k as final90


COLORS = {
    "lambda0": "#0072B2",
    "lambda02": "#009E73",
    final90.V23_METHOD: "#56B4E9",
    final90.DYNAMIC_METHOD: "#CC79A7",
    final90.GA_METHOD: "#D55E00",
    final90.KIM_METHOD: "#6F4E7C",
}
MARKERS = {
    "lambda0": "o",
    "lambda02": "D",
    final90.V23_METHOD: "X",
    final90.DYNAMIC_METHOD: "^",
    final90.GA_METHOD: "v",
    final90.KIM_METHOD: "P",
}
LABELS = {
    "lambda0": "VCG (λ = 0)",
    "lambda02": "VCG + handling (λ = 0.2)",
    final90.V23_METHOD: "Historical VCG 2.3",
    final90.DYNAMIC_METHOD: "Dynamic PSLAP",
    final90.GA_METHOD: "Capacity-aware GA",
    final90.KIM_METHOD: "Kim2020 adaptation",
}
CURVE_METRICS = (
    ("dense_return", "Cumulative dense return ↑"),
    ("mean_absolute_error", "Cumulative MAE ↓"),
    (
        "physical_rehandles_per_100_required_deliveries",
        "Cumulative physical rehandles / 100 ↓",
    ),
)
SELECTED_LAMBDA_LABELS = frozenset((0.0, 0.05, 0.1, 0.2))
SEED_STYLES = {
    0: ("#7C8DA5", "-"),
    1: ("#9A88AD", "--"),
    2: ("#739B91", ":"),
}
TABLE_METRICS = (
    ("mean_absolute_error", "MAE ↓"),
    ("physical_rehandles_per_100_required_deliveries", "Rehandles/100 ↓"),
    ("within_target_window_rate", "Within ±20 ↑"),
    ("steps", "Steps ↓"),
    ("dense_return", "Return ↑"),
)


class Plot90Error(RuntimeError):
    pass


def _metric(row: Mapping, name: str) -> float:
    key = "dense_objective_return" if (
        name == "dense_return" and "dense_return" not in row
    ) else name
    value = row.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise Plot90Error(f"missing metric {name}")
    result = float(value)
    if not math.isfinite(result):
        raise Plot90Error(f"nonfinite metric {name}")
    return result


def _load_frozen_e1_artifacts(output_dir: Path) -> tuple[dict, list[dict], dict]:
    """Authenticate serialized E1 artifacts without reopening training inputs."""

    contract = final90._load_json(
        output_dir / final90.CONTRACT_NAME,
        label="frozen E1 contract",
    )
    final90._verify_hash(
        contract,
        "contract_sha256",
        label="frozen E1 contract",
    )
    manifest = final90._load_json(
        output_dir / final90.INSTANCE_MANIFEST_NAME,
        label="frozen E1 manifest",
    )
    final90._verify_hash(
        manifest,
        "manifest_sha256",
        label="frozen E1 manifest",
    )
    report = final90._load_json(
        output_dir / final90.REPORT_NAME,
        label="frozen E1 report",
    )
    final90._verify_hash(report, "report_sha256", label="frozen E1 report")
    if (
        contract.get("protocol") != final90.PROTOCOL
        or manifest.get("contract_sha256") != contract["contract_sha256"]
        or report.get("contract_sha256") != contract["contract_sha256"]
        or report.get("manifest_sha256") != manifest["manifest_sha256"]
        or report.get("status") != "complete"
        or report.get("observed_total_rows") != final90.EXPECTED_ROWS
        or report.get("all_conditioned_lambdas_eligible") is not True
    ):
        raise Plot90Error("frozen E1 contract, manifest, and report do not agree")

    records = manifest.get("instances", [])
    if (
        len(records) != len(final90.INSTANCE_SEEDS)
        or tuple(int(item["seed"]) for item in records)
        != final90.INSTANCE_SEEDS
    ):
        raise Plot90Error("frozen E1 manifest has the wrong instance grid")
    for record in records:
        path = output_dir / record["relative_path"]
        if final90._sha256(path) != record["raw_sha256"]:
            raise Plot90Error(f"frozen E1 EpisodeInstance changed: {path}")

    rows = []
    for value in final90.LAMBDA_GRID:
        for model_seed in final90.MODEL_SEEDS:
            for record in records:
                instance_seed = int(record["seed"])
                path = final90._conditioned_ledger_path(
                    output_dir,
                    value,
                    model_seed,
                    instance_seed,
                )
                ledger = final90._load_json(path, label="frozen E1 ledger")
                final90._verify_hash(
                    ledger,
                    "ledger_sha256",
                    label="frozen E1 ledger",
                )
                if (
                    ledger.get("contract_sha256") != contract["contract_sha256"]
                    or ledger.get("manifest_sha256")
                    != manifest["manifest_sha256"]
                    or ledger.get("method") != final90.CONDITIONED_METHOD
                    or int(ledger.get("model_seed", -1)) != model_seed
                    or float(ledger.get("preference_lambda", -1.0))
                    != float(value)
                    or int(ledger.get("instance_seed", -1)) != instance_seed
                ):
                    raise Plot90Error(f"frozen E1 ledger binding changed: {path}")
                row = dict(ledger["run"])
                final90._validate_conditioned_row(
                    row,
                    identity=final90._identity(record),
                    value=value,
                    seed=model_seed,
                )
                rows.append(row)
    if (
        len(rows) != final90.CONDITIONED_ROWS
        or not all(row["strict_safe_complete"] is True for row in rows)
    ):
        raise Plot90Error("frozen E1 conditioned ledger panel is incomplete")
    provenance = {
        "contract_sha256": contract["contract_sha256"],
        "manifest_sha256": manifest["manifest_sha256"],
        "report_sha256": report["report_sha256"],
        "conditioned_rows": len(rows),
    }
    return report, rows, provenance


def _seed_operating_paths(rows: Sequence[Mapping], report: Mapping) -> dict:
    grouped = defaultdict(list)
    for row in rows:
        grouped[
            (int(row["model_seed"]), float(row["preference_lambda"]))
        ].append(row)

    conditioned_records = {
        float(record["preference_lambda"]): record
        for record in report["methods"]
        if record["method"] == final90.CONDITIONED_METHOD
    }
    if set(conditioned_records) != set(final90.LAMBDA_GRID):
        raise Plot90Error("frozen E1 report has the wrong lambda grid")

    paths = {}
    for seed in final90.MODEL_SEEDS:
        points = []
        for value in final90.LAMBDA_GRID:
            selected = grouped[(seed, float(value))]
            if (
                len(selected) != len(final90.INSTANCE_SEEDS)
                or not all(row["strict_safe_complete"] is True for row in selected)
            ):
                raise Plot90Error(
                    f"model seed {seed}, lambda {value:g} is incomplete"
                )
            points.append(
                {
                    "preference_lambda": float(value),
                    "physical_rehandles_per_100_required_deliveries": fmean(
                        _metric(
                            row,
                            "physical_rehandles_per_100_required_deliveries",
                        )
                        for row in selected
                    ),
                    "mean_absolute_error": fmean(
                        _metric(row, "mean_absolute_error") for row in selected
                    ),
                }
            )
        paths[seed] = points

    for index, value in enumerate(final90.LAMBDA_GRID):
        record = conditioned_records[float(value)]
        for metric in (
            "physical_rehandles_per_100_required_deliveries",
            "mean_absolute_error",
        ):
            seed_mean = fmean(paths[seed][index][metric] for seed in paths)
            if not math.isclose(
                seed_mean,
                float(record["metrics"][metric]),
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise Plot90Error(
                    f"seed paths do not reproduce aggregate {metric} at lambda {value:g}"
                )
    return {
        "model_seeds": list(final90.MODEL_SEEDS),
        "lambdas": list(final90.LAMBDA_GRID),
        "paths": paths,
    }


def _prefix_curve(rows, *, expected_per_instance: int):
    grouped = defaultdict(list)
    for row in rows:
        grouped[int(row["instance_seed"])].append(row)
    points = []
    failed_at = None
    for position, seed in enumerate(final90.INSTANCE_SEEDS, start=1):
        group = grouped.get(seed, [])
        if (
            len(group) != expected_per_instance
            or not all(row.get("strict_safe_complete") is True for row in group)
        ):
            failed_at = position
            break
        points.append(
            {
                metric: fmean(_metric(row, metric) for row in group)
                for metric, _label in CURVE_METRICS
            }
        )
    curves = {}
    for metric, _label in CURVE_METRICS:
        values = [point[metric] for point in points]
        mean = []
        lower = []
        upper = []
        for count in range(1, len(values) + 1):
            prefix = values[:count]
            center = fmean(prefix)
            half = (
                0.0
                if count == 1
                else 1.96 * stdev(prefix) / math.sqrt(count)
            )
            mean.append(center)
            lower.append(center - half)
            upper.append(center + half)
        curves[metric] = {"mean": mean, "lower": lower, "upper": upper}
    return {"curves": curves, "failed_at": failed_at, "safe_prefix": len(points)}


def _load_curves(project_root: Path, output_dir: Path):
    manifest = final90.authenticate_manifest(project_root, output_dir)
    conditioned = final90._load_conditioned_rows(output_dir, manifest)
    methods = {
        "lambda0": _prefix_curve(
            [row for row in conditioned if row["preference_lambda"] == 0.0],
            expected_per_instance=len(final90.MODEL_SEEDS),
        ),
        "lambda02": _prefix_curve(
            [row for row in conditioned if row["preference_lambda"] == 0.2],
            expected_per_instance=len(final90.MODEL_SEEDS),
        ),
        final90.V23_METHOD: _prefix_curve(
            final90._load_baseline_rows(output_dir, final90.V23_METHOD),
            expected_per_instance=(
                len(final90.final86.V23_MODEL_SEEDS)
                * len(final90.final86.RNG_INDICES)
            ),
        ),
        final90.DYNAMIC_METHOD: _prefix_curve(
            final90._load_baseline_rows(output_dir, final90.DYNAMIC_METHOD),
            expected_per_instance=1,
        ),
        final90.GA_METHOD: _prefix_curve(
            final90._load_baseline_rows(output_dir, final90.GA_METHOD),
            expected_per_instance=len(final90.final86.RNG_INDICES),
        ),
        final90.KIM_METHOD: _prefix_curve(
            final90._load_kim_rows(project_root, output_dir),
            expected_per_instance=(
                len(final90.kim_final.MODEL_SEEDS)
                * len(final90.kim_final.ROLLOUTS)
            ),
        ),
    }
    return methods


def render_cumulative(project_root: Path, output_dir: Path) -> list[str]:
    methods = _load_curves(project_root, output_dir)
    figure, axes = plt.subplots(1, 3, figsize=(15.5, 5.0))
    for axis, (metric, ylabel) in zip(axes, CURVE_METRICS):
        for key, payload in methods.items():
            curve = payload["curves"][metric]
            count = len(curve["mean"])
            if not count:
                continue
            x = np.arange(1, count + 1)
            axis.plot(
                x,
                curve["mean"],
                color=COLORS[key],
                marker=MARKERS[key],
                markevery=max(1, count // 6),
                linewidth=2.0,
                markersize=5,
                label=LABELS[key],
            )
            axis.fill_between(
                x,
                curve["lower"],
                curve["upper"],
                color=COLORS[key],
                alpha=0.11,
                linewidth=0,
            )
            if payload["failed_at"] is not None:
                axis.scatter(
                    [payload["failed_at"]],
                    [curve["mean"][-1]],
                    marker="x",
                    s=65,
                    linewidths=2,
                    color=COLORS[key],
                    zorder=5,
                )
        axis.set_xlabel("Cumulative common EpisodeInstances $k$")
        axis.set_ylabel(ylabel)
        axis.set_xlim(1, len(final90.INSTANCE_SEEDS))
        axis.grid(axis="y", alpha=0.25)
        axis.spines[["top", "right"]].set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, -0.01),
    )
    figure.suptitle("Frozen-policy comparison on the prospective common 90k panel")
    figure.tight_layout(rect=(0, 0.11, 1, 0.94))
    paths = []
    for suffix in ("pdf", "png"):
        path = output_dir / f"e1-supplement-cumulative.{suffix}"
        figure.savefig(path, dpi=220, bbox_inches="tight")
        paths.append(str(path.resolve()))
    plt.close(figure)
    return paths


def render_frontier(
    output_dir: Path,
    report: Mapping,
    *,
    seed_paths: Optional[Mapping] = None,
) -> list[str]:
    eligible = [
        record for record in report["methods"] if record["whole_method_eligible"]
    ]
    conditioned = sorted(
        (
            record
            for record in eligible
            if record["method"] == final90.CONDITIONED_METHOD
        ),
        key=lambda record: float(record["preference_lambda"]),
    )
    figure, axis = plt.subplots(
        figsize=(8.7, 5.9) if seed_paths is not None else (7.4, 5.35)
    )
    legend_handles = []
    if conditioned:
        x = [
            record["metrics"]["physical_rehandles_per_100_required_deliveries"]
            for record in conditioned
        ]
        y = [record["metrics"]["mean_absolute_error"] for record in conditioned]
        lambdas = [float(record["preference_lambda"]) for record in conditioned]
        normalizer = Normalize(vmin=min(lambdas), vmax=max(lambdas))
        if seed_paths is not None:
            for seed in seed_paths["model_seeds"]:
                path = seed_paths["paths"][seed]
                seed_x = [
                    point[
                        "physical_rehandles_per_100_required_deliveries"
                    ]
                    for point in path
                ]
                seed_y = [point["mean_absolute_error"] for point in path]
                color, linestyle = SEED_STYLES[int(seed)]
                axis.plot(
                    seed_x,
                    seed_y,
                    color=color,
                    linestyle=linestyle,
                    linewidth=1.35,
                    alpha=0.72,
                    zorder=1,
                )
                axis.scatter(
                    seed_x,
                    seed_y,
                    c=lambdas,
                    cmap="viridis",
                    norm=normalizer,
                    edgecolor="none",
                    s=28,
                    alpha=0.42,
                    zorder=2,
                )
            axis.plot(
                x,
                y,
                color="#303844",
                linewidth=2.8,
                alpha=0.95,
                zorder=3,
            )
            legend_handles.append(
                Line2D(
                    [0],
                    [0],
                    color="#303844",
                    marker="o",
                    markerfacecolor="white",
                    markeredgecolor="#303844",
                    linewidth=2.8,
                    label="VCG mean (3 model seeds)",
                )
            )
            legend_handles.extend(
                Line2D(
                    [0],
                    [0],
                    color=SEED_STYLES[int(seed)][0],
                    linestyle=SEED_STYLES[int(seed)][1],
                    linewidth=1.6,
                    label=f"VCG model seed {seed}",
                )
                for seed in seed_paths["model_seeds"]
            )
        else:
            axis.plot(
                x,
                y,
                color="#5B6573",
                linewidth=1.8,
                alpha=0.8,
                zorder=1,
            )
        points = axis.scatter(
            x,
            y,
            c=lambdas,
            cmap="viridis",
            norm=normalizer,
            edgecolor="white",
            linewidth=0.75,
            s=68,
            zorder=4,
            label=(
                "Preference-conditioned VCG"
                if seed_paths is None
                else None
            ),
        )
        offsets = {
            0.0: (-8, 9),
            0.05: (7, -14),
            0.1: (7, -14),
            0.2: (7, 7),
        }
        for record, x_value, y_value in zip(conditioned, x, y):
            value = float(record["preference_lambda"])
            if value not in SELECTED_LAMBDA_LABELS:
                continue
            axis.annotate(
                f"λ={value:g}",
                (x_value, y_value),
                xytext=offsets[value],
                textcoords="offset points",
                fontsize=8,
            )
        colorbar = figure.colorbar(points, ax=axis, pad=0.025, fraction=0.055)
        colorbar.set_label("Handling preference λ")
    for record in eligible:
        if record["method"] == final90.CONDITIONED_METHOD:
            continue
        key = record["method"]
        baseline = axis.scatter(
            [record["metrics"]["physical_rehandles_per_100_required_deliveries"]],
            [record["metrics"]["mean_absolute_error"]],
            color=COLORS[key],
            marker=MARKERS[key],
            s=75,
            label=LABELS[key],
            zorder=4,
        )
        if seed_paths is not None:
            legend_handles.append(baseline)
    axis.set_xlabel("Physical rehandles / 100 required deliveries ↓")
    axis.set_ylabel("Mean absolute delivery error ↓")
    axis.grid(alpha=0.22, linewidth=0.8)
    axis.spines[["top", "right"]].set_visible(False)
    if seed_paths is not None:
        axis.set_title(
            "Timing–handling operating family across frozen model seeds",
            fontsize=12,
            pad=11,
        )
        axis.legend(
            handles=legend_handles,
            frameon=False,
            fontsize=8.4,
            loc="upper right",
        )
        figure.text(
            0.5,
            0.02,
            (
                "Light paths: model-seed means over the same 30 EpisodeInstances; "
                "bold path: three-seed mean. Dynamic PSLAP and Kim2020 are "
                "suppressed because strict completion failed."
            ),
            ha="center",
            fontsize=8,
            color="#4c566a",
        )
        figure.tight_layout(rect=(0, 0.065, 1, 1))
    else:
        axis.legend(frameon=False, fontsize=9, loc="upper right")
        figure.tight_layout()
    paths = []
    for suffix in ("pdf", "png"):
        path = output_dir / f"e1-operating-points.{suffix}"
        figure.savefig(path, dpi=220, bbox_inches="tight")
        paths.append(str(path.resolve()))
    plt.close(figure)
    return paths


def _table_rows(report: Mapping) -> list[dict[str, object]]:
    rows = []
    for record in report["methods"]:
        metrics = record["metrics"]
        row: dict[str, object] = {
            "method": record["display_name"],
            "strict_complete_runs": (
                f"{record['strict_safe_complete_rows']}/{record['expected_rows']}"
            ),
            "whole_method_eligible": bool(record["whole_method_eligible"]),
        }
        for metric, _label in TABLE_METRICS:
            row[metric] = None if metrics is None else float(metrics[metric])
        rows.append(row)
    return rows


def _display_metric(value: object, metric: str) -> str:
    if value is None:
        return "—"
    numeric = float(value)
    if metric == "within_target_window_rate":
        return f"{100.0 * numeric:.1f}%"
    return f"{numeric:.2f}"


def render_benchmark_table(output_dir: Path, report: Mapping) -> list[str]:
    rows = _table_rows(report)
    fields = (
        "method",
        "strict_complete_runs",
        *(metric for metric, _label in TABLE_METRICS),
    )

    csv_path = output_dir / "e1-benchmark-table.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    markdown_path = output_dir / "e1-benchmark-table.md"
    headers = ("Method", "Strict complete", *(label for _metric, label in TABLE_METRICS))
    markdown_lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    for row in rows:
        values = [str(row["method"]), str(row["strict_complete_runs"])]
        values.extend(
            _display_metric(row[metric], metric) for metric, _label in TABLE_METRICS
        )
        markdown_lines.append("| " + " | ".join(values) + " |")
    markdown_path.write_text("\n".join(markdown_lines) + "\n", encoding="utf-8")

    return [str(csv_path.resolve()), str(markdown_path.resolve())]


def render_seed_path_frontier(output_dir: Path) -> dict:
    report, rows, provenance = _load_frozen_e1_artifacts(output_dir)
    seed_paths = _seed_operating_paths(rows, report)
    figures = render_frontier(output_dir, report, seed_paths=seed_paths)
    result = {
        "schema_version": 1,
        "status": "complete",
        **provenance,
        "model_seeds": seed_paths["model_seeds"],
        "lambdas": seed_paths["lambdas"],
        "matched_instances_per_seed_lambda": len(final90.INSTANCE_SEEDS),
        "seed_paths_reproduce_report_aggregate": True,
        "incomplete_baselines_suppressed": [
            record["method"]
            for record in report["methods"]
            if not record["whole_method_eligible"]
        ],
        "figure_pdf": figures[0],
        "figure_png": figures[1],
        "seed_operating_points": [
            {
                "model_seed": int(seed),
                "points": seed_paths["paths"][seed],
            }
            for seed in seed_paths["model_seeds"]
        ],
    }
    metadata = output_dir / "e1-operating-points-data.json"
    metadata.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    result["metadata"] = str(metadata.resolve())
    return result


def render(project_root: Path, output_dir: Path) -> dict:
    report = final90.analyze(project_root, output_dir)
    return {
        "status": "complete",
        "main_operating_points": render_frontier(output_dir, report),
        "main_benchmark_table": render_benchmark_table(output_dir, report),
        "supplementary_cumulative": render_cumulative(project_root, output_dir),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root", type=Path, default=Path(__file__).resolve().parent
    )
    parser.add_argument("--output-dir", type=Path, default=final90.DEFAULT_OUTPUT)
    parser.add_argument(
        "--operating-points-only",
        action="store_true",
        help=(
            "render the E1/E2 operating-point figure with frozen model-seed "
            "paths without reopening historical training inputs"
        ),
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    if args.operating_points_only:
        result = render_seed_path_frontier(args.output_dir.resolve())
    else:
        result = render(args.project_root.resolve(), args.output_dir.resolve())
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
