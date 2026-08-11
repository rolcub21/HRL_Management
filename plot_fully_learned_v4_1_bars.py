#!/usr/bin/env python3
"""Plot the matched standard Track-B holdout as method-level bars.

The holdout seeds are exchangeable paired instances, not an ordered trajectory.
Consequently, this figure summarizes each method with a mean bar, a bootstrap
confidence interval, and the underlying episode observations instead of
connecting seeds with curves.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


METHODS = (
    "Fully learned v4.1",
    "Duration-aware + REG",
    "Duration-aware + dynamic PSLAP",
    "Duration-aware + rolling GA",
    "Duration-aware + time-aware GA",
    "Duration-aware + nearest-free",
)
BASELINE_LABELS = {
    "reg_selector_v5": METHODS[1],
    "dynamic_pslap": METHODS[2],
    "nearest_free": METHODS[5],
}
GA_REFERENCE_LABEL = METHODS[3]
GA_IMPROVED_LABEL = METHODS[4]
TICK_LABELS = (
    "Fully learned\nv4.1",
    "Duration-aware\n+ REG",
    "Duration-aware\n+ dynamic PSLAP",
    "Duration-aware\n+ rolling GA",
    "Duration-aware\n+ time-aware GA",
    "Duration-aware\n+ nearest-free",
)
COLORS = (
    "#0072B2",
    "#D55E00",
    "#009E73",
    "#CC79A7",
    "#E69F00",
    "#7A7A7A",
)
METRICS = (
    ("return", "Episode return", "A. Return", 1.0, "{:.1f}"),
    (
        "mean_absolute_error",
        "Mean absolute timing error",
        "B. Absolute timing error",
        1.0,
        "{:.3f}",
    ),
    (
        "mean_tardiness",
        "Mean tardiness",
        "C. Tardiness",
        1.0,
        "{:.3f}",
    ),
    (
        "strict_method_success",
        "Strict success (%)",
        "D. Strict completion",
        100.0,
        "{:.0f}%",
    ),
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--learned-holdout",
        type=Path,
        default=Path(
            "results/fully-learned-v4-1-holdout-48000-48009.json"
        ),
    )
    parser.add_argument(
        "--baseline-results",
        type=Path,
        default=Path(
            "results/track-b-hierarchy-confirmation-48000-48009/"
            "hierarchy-full.json"
        ),
    )
    parser.add_argument(
        "--bootstrap-samples", type=int, default=100_000
    )
    parser.add_argument(
        "--ga-reference-results",
        type=Path,
        default=Path(
            "results/track-b-duration-aware-rolling-ga-reference-48000-48009/"
            "assignment-source-audit.json"
        ),
    )
    parser.add_argument(
        "--ga-improved-results",
        type=Path,
        default=Path(
            "results/track-b-duration-aware-rolling-ga-improved-48000-48009/"
            "assignment-source-audit.json"
        ),
    )
    parser.add_argument("--bootstrap-seed", type=int, default=20_260_806)
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("results/fully-learned-v4-1-comparison-bars"),
    )
    args = parser.parse_args()
    if args.bootstrap_samples <= 0:
        parser.error("bootstrap-samples must be positive")
    return args


def load_json(path: Path):
    with path.open() as stream:
        return json.load(stream)


def load_paired_holdout(
    learned_path: Path,
    baseline_path: Path,
    ga_reference_path: Path,
    ga_improved_path: Path,
):
    learned_payload = load_json(learned_path)
    baseline_payload = load_json(baseline_path)
    learned_runs = sorted(
        learned_payload["runs"], key=lambda run: int(run["eval_seed"])
    )
    if len(learned_runs) != 10:
        raise ValueError(
            f"expected 10 learned holdout runs, found {len(learned_runs)}"
        )
    learned_by_instance = {run["instance_id"]: run for run in learned_runs}
    if len(learned_by_instance) != len(learned_runs):
        raise ValueError("learned holdout contains duplicate instance IDs")

    selected = {}
    for run in baseline_payload["runs"]:
        if (
            run["geometry_condition"] == "ordinary"
            and run["scheduler_variant"] == "duration_aware"
            and run["assignment_commitment"] == "decision_epoch_reserved"
            and run["assignment_source"] in BASELINE_LABELS
        ):
            selected.setdefault(run["assignment_source"], {})[
                run["instance_id"]
            ] = run

    expected_instances = set(learned_by_instance)
    ordered_ids = [run["instance_id"] for run in learned_runs]
    paired = {METHODS[0]: learned_runs}
    for source, label in BASELINE_LABELS.items():
        found = set(selected.get(source, {}))
        if found != expected_instances:
            raise ValueError(
                f"paired baseline mismatch for {source}: "
                f"missing={sorted(expected_instances - found)}, "
                f"extra={sorted(found - expected_instances)}"
            )
        paired[label] = [selected[source][key] for key in ordered_ids]

    extra_specs = (
        (
            ga_reference_path,
            "pslap_ga_2009_rolling",
            GA_REFERENCE_LABEL,
        ),
        (
            ga_improved_path,
            "pslap_ga_duration_aware_rolling",
            GA_IMPROVED_LABEL,
        ),
    )
    for path, source, label in extra_specs:
        payload = load_json(path)
        rows = {
            run["instance_id"]: run
            for run in payload["runs"]
            if run["assignment_source"] == source
        }
        found = set(rows)
        if found != expected_instances:
            raise ValueError(
                f"paired additional baseline mismatch for {source}: "
                f"missing={sorted(expected_instances - found)}, "
                f"extra={sorted(found - expected_instances)}"
            )
        paired[label] = [rows[key] for key in ordered_ids]

    reference = paired[METHODS[0]]
    for method in METHODS:
        if len(paired[method]) != len(reference):
            raise ValueError(f"run-count mismatch for {method}")
        for learned, run in zip(reference, paired[method]):
            identity = (
                int(run["eval_seed"]),
                run["instance_id"],
                run["schedule_id"],
            )
            expected = (
                int(learned["eval_seed"]),
                learned["instance_id"],
                learned["schedule_id"],
            )
            if identity != expected:
                raise ValueError(
                    f"paired identity mismatch for {method}: "
                    f"expected={expected}, found={identity}"
                )
    return paired


def values(paired, method, metric_name, scale=1.0):
    return np.asarray(
        [float(run[metric_name]) * scale for run in paired[method]],
        dtype=np.float64,
    )


def bootstrap_mean_interval(data, *, samples, rng):
    indices = rng.integers(0, len(data), size=(samples, len(data)))
    means = data[indices].mean(axis=1)
    return tuple(float(value) for value in np.quantile(means, (0.025, 0.975)))


def paired_contrasts(paired, *, samples, seed):
    """Report paired effects; positive always favors the left-hand method."""

    specifications = (
        ("fully_learned_minus_rolling_ga", METHODS[0], GA_REFERENCE_LABEL),
        ("fully_learned_minus_time_aware_ga", METHODS[0], GA_IMPROVED_LABEL),
        ("time_aware_ga_minus_rolling_ga", GA_IMPROVED_LABEL, GA_REFERENCE_LABEL),
    )
    metric_specs = (
        ("return_advantage", "return", 1.0),
        ("absolute_error_reduction", "mean_absolute_error", -1.0),
        ("tardiness_reduction", "mean_tardiness", -1.0),
    )
    rng = np.random.default_rng(seed)
    output = []
    for name, left, right in specifications:
        metrics = {}
        for metric, field, orientation in metric_specs:
            left_values = values(paired, left, field)
            right_values = values(paired, right, field)
            deltas = orientation * (left_values - right_values)
            low, high = bootstrap_mean_interval(
                deltas, samples=samples, rng=rng
            )
            metrics[metric] = {
                "mean": float(deltas.mean()),
                "bootstrap_95_ci": [low, high],
                "wins": int(np.sum(deltas > 0.0)),
                "ties": int(np.sum(deltas == 0.0)),
                "n": int(len(deltas)),
                "per_instance": deltas.tolist(),
            }
        output.append(
            {
                "contrast": name,
                "left": left,
                "right": right,
                "orientation": "positive favors left",
                "metrics": metrics,
            }
        )
    return output


def summarize(paired, *, samples, seed):
    rng = np.random.default_rng(seed)
    rows = []
    for method in METHODS:
        row = {"method": method, "n": len(paired[method])}
        for metric_name, _, _, scale, _ in METRICS:
            data = values(paired, method, metric_name, scale)
            lower, upper = bootstrap_mean_interval(
                data, samples=samples, rng=rng
            )
            row[f"{metric_name}_mean"] = float(data.mean())
            row[f"{metric_name}_ci_lower"] = lower
            row[f"{metric_name}_ci_upper"] = upper
            row[f"{metric_name}_observations"] = data.tolist()
        rows.append(row)
    return rows


def plot_panel(ax, paired, summary, metric_spec):
    metric_name, ylabel, title, scale, label_format = metric_spec
    x = np.arange(len(METHODS))
    means = np.asarray(
        [row[f"{metric_name}_mean"] for row in summary], dtype=float
    )
    lower = np.asarray(
        [row[f"{metric_name}_ci_lower"] for row in summary], dtype=float
    )
    upper = np.asarray(
        [row[f"{metric_name}_ci_upper"] for row in summary], dtype=float
    )
    error = np.vstack((means - lower, upper - means))
    bars = ax.bar(
        x,
        means,
        width=0.68,
        color=COLORS,
        edgecolor="black",
        linewidth=0.7,
        yerr=error,
        capsize=4,
        error_kw={"elinewidth": 1.1, "capthick": 1.1},
        zorder=2,
    )

    jitter = np.linspace(-0.17, 0.17, len(paired[METHODS[0]]))
    for index, method in enumerate(METHODS):
        data = values(paired, method, metric_name, scale)
        ax.scatter(
            np.full(len(data), index) + jitter,
            data,
            s=16,
            facecolor="white",
            edgecolor="black",
            linewidth=0.55,
            alpha=0.85,
            zorder=3,
        )

    maximum = max(float(upper.max()), float(means.max()), 1e-9)
    if metric_name == "strict_method_success":
        ax.set_ylim(0.0, 112.0)
        annotation_offset = 1.8
    else:
        ax.set_ylim(0.0, maximum * 1.20)
        annotation_offset = maximum * 0.025
    for bar, mean, high in zip(bars, means, upper):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            high + annotation_offset,
            label_format.format(mean),
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="bold",
        )

    ax.set_title(title, loc="left", fontweight="bold")
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(TICK_LABELS, fontsize=8)
    ax.grid(axis="y", linestyle="--", alpha=0.35, zorder=0)
    ax.grid(axis="x", visible=False)


def write_csv(path: Path, summary):
    metric_fields = []
    for metric_name, _, _, _, _ in METRICS:
        metric_fields.extend(
            (
                f"{metric_name}_mean",
                f"{metric_name}_ci_lower",
                f"{metric_name}_ci_upper",
            )
        )
    fields = ("method", "n", *metric_fields)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in summary:
            writer.writerow({key: row[key] for key in fields})


def main():
    args = parse_args()
    paired = load_paired_holdout(
        args.learned_holdout,
        args.baseline_results,
        args.ga_reference_results,
        args.ga_improved_results,
    )
    summary = summarize(
        paired, samples=args.bootstrap_samples, seed=args.bootstrap_seed
    )

    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "figure.dpi": 120,
        }
    )
    contrasts = paired_contrasts(
        paired, samples=args.bootstrap_samples, seed=args.bootstrap_seed + 1
    )
    fig, axes = plt.subplots(2, 2, figsize=(18.0, 9.0))
    for ax, metric_spec in zip(axes.flat, METRICS):
        plot_panel(ax, paired, summary, metric_spec)

    fig.suptitle(
        "Standard Track B: matched holdout comparison",
        fontsize=15,
        fontweight="bold",
        y=0.985,
    )
    fig.text(
        0.5,
        0.017,
        "Bars show method means; whiskers are 95% bootstrap confidence "
        "intervals across 10 matched EpisodeInstances; dots are individual "
        "episodes. Tardiness is directional and should be read alongside MAE.",
        ha="center",
        fontsize=8.5,
    )
    fig.tight_layout(rect=(0, 0.055, 1, 0.96), h_pad=2.3, w_pad=1.8)

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    png_path = args.output_prefix.with_suffix(".png")
    pdf_path = args.output_prefix.with_suffix(".pdf")
    csv_path = args.output_prefix.with_suffix(".csv")
    json_path = args.output_prefix.with_suffix(".json")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    write_csv(csv_path, summary)
    with json_path.open("w") as stream:
        json.dump(
            {
                "protocol": "matched_holdout_method_bar_plot_v1",
                "learned_holdout": str(args.learned_holdout),
                "baseline_results": str(args.baseline_results),
                "ga_reference_results": str(args.ga_reference_results),
                "ga_improved_results": str(args.ga_improved_results),
                "bootstrap_samples": args.bootstrap_samples,
                "bootstrap_seed": args.bootstrap_seed,
                "summary": summary,
                "paired_contrasts": contrasts,
            },
            stream,
            indent=2,
        )
        stream.write("\n")

    print(
        json.dumps(
            {
                "png": str(png_path),
                "pdf": str(pdf_path),
                "csv": str(csv_path),
                "json": str(json_path),
                "means": {
                    row["method"]: {
                        key: value
                        for key, value in row.items()
                        if key.endswith("_mean")
                    }
                    for row in summary
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
