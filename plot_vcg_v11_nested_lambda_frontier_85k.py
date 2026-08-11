#!/usr/bin/env python3
"""Plot the sampled nested-VCG lambda frontier on the 85k development panel."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import run_vcg_v11_nested_lambda_frontier_85k as frontier


def render(output_dir: Path) -> dict:
    report = json.loads((output_dir / frontier.REPORT_NAME).read_text())
    if report.get("status") != "complete" or report.get("row_count") != 180:
        raise RuntimeError("frontier report is incomplete")
    values = list(frontier.LAMBDA_GRID)
    metrics = report["aggregate_metrics"]
    mae = [metrics[frontier._key(value)]["mean_absolute_error"] for value in values]
    rehandles = [
        metrics[frontier._key(value)]["physical_rehandles_per_100"] for value in values
    ]
    returns = [metrics[frontier._key(value)]["mean_dense_return"] for value in values]
    nondominated = set(report["aggregate_nondominated_lambdas"])

    figure, axes = plt.subplots(1, 2, figsize=(11.8, 4.8))
    axes[0].plot(rehandles, mae, color="#0072B2", linewidth=2, alpha=0.75)
    for value, x, y in zip(values, rehandles, mae):
        axes[0].scatter(
            x,
            y,
            s=90 if value in nondominated else 58,
            color="#009E73" if value in nondominated else "#999999",
            edgecolor="white",
            linewidth=1,
            zorder=3,
        )
        axes[0].annotate(f"λ={value:g}", (x, y), xytext=(5, 5), textcoords="offset points")
    axes[0].set_xlabel("Physical rehandles / 100 ↓")
    axes[0].set_ylabel("MAE ↓")
    axes[0].set_title("Sampled operational–handling frontier")
    axes[0].grid(color="#DDDDDD", linewidth=0.8, alpha=0.8)

    axes[1].plot(values, returns, color="#0072B2", marker="o", label="Dense return ↑")
    second = axes[1].twinx()
    second.plot(values, rehandles, color="#D55E00", marker="D", label="Rehandles / 100 ↓")
    axes[1].set_xlabel("Handling weight λ")
    axes[1].set_ylabel("Dense return ↑", color="#0072B2")
    second.set_ylabel("Physical rehandles / 100 ↓", color="#D55E00")
    axes[1].set_title("One frozen model, different operating points")
    axes[1].grid(color="#DDDDDD", linewidth=0.8, alpha=0.8)
    handles1, labels1 = axes[1].get_legend_handles_labels()
    handles2, labels2 = second.get_legend_handles_labels()
    axes[1].legend(handles1 + handles2, labels1 + labels2, frameon=False, loc="best")
    figure.suptitle("Nested VCG fixed-λ development sweep (three model seeds)", fontsize=13)
    figure.tight_layout()
    png = output_dir / "nested-vcg-lambda-frontier.png"
    pdf = output_dir / "nested-vcg-lambda-frontier.pdf"
    figure.savefig(png, dpi=240, bbox_inches="tight")
    figure.savefig(pdf, bbox_inches="tight")
    plt.close(figure)
    result = {
        "protocol": frontier.PROTOCOL,
        "lambda_grid": values,
        "aggregate_nondominated_lambdas": sorted(nondominated),
        "advance_all_five_levels_to_unseen_89k": report[
            "advance_all_five_levels_to_unseen_89k"
        ],
        "png": str(png),
        "pdf": str(pdf),
    }
    frontier.pilot._atomic_json(output_dir / "frontier-plot-summary.json", result)
    return result


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent
        / "results/vcg-v1-1-nested-lambda-frontier-85k-development",
    )
    args = parser.parse_args(argv)
    print(json.dumps(render(args.output_dir.resolve()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
