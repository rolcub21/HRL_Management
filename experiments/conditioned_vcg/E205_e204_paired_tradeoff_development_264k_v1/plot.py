"""Plot E205's paired mechanism outcomes."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


PANELS = (
    ("physical_rehandles", "Physical rehandles"),
    ("J10", "Realized J at weight 10"),
    ("timing_cost_2T_plus_E", "Timing cost: 2T + E"),
    ("admission_wait_cost_0p5W", "Admission waiting cost: 0.5W"),
    ("steps", "Primitive steps"),
    ("peak_stored", "Peak stored blocks"),
)
COLORS = {0.2: "#2b83ba", 0.5: "#fdae61", 0.8: "#d7191c"}


def create_plot(report: dict, destination: Path) -> Path:
    rows = report["paired_rows"]
    workloads = sorted({float(row["workload"]) for row in rows})
    figure, axes = plt.subplots(2, 3, figsize=(13.5, 7.8), constrained_layout=True)
    for axis, (key, title) in zip(axes.flat, PANELS):
        for workload in workloads:
            selected = sorted(
                (row for row in rows if float(row["workload"]) == workload),
                key=lambda row: float(row["lambda"]),
            )
            axis.plot(
                [row["lambda"] for row in selected],
                [row[key] for row in selected],
                marker="o",
                linewidth=2,
                color=COLORS[workload],
                label=f"{100 * workload:.0f}% workload",
            )
        axis.axvline(10.0, color="#555555", linestyle="--", linewidth=1)
        axis.set_title(title)
        axis.set_xlabel("Deployment rehandle weight λ")
        axis.grid(alpha=0.25)
    axes[0, 0].legend(frameon=False, fontsize=9)
    figure.suptitle(
        "E205 paired E204 trade-off (same instance across λ; dashed line = E191 λ=10)",
        fontsize=13,
    )
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=180)
    plt.close(figure)
    return destination


__all__ = ["create_plot"]
