"""Plot E214's seed-0 trace-return frontier against immediate-only control."""

from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def create_plot(rows: list[dict], destination: Path) -> Path:
    groups = defaultdict(list)
    for row in rows:
        groups[(row["method"], float(row["workload"]), float(row["lambda"]))].append(row)
    workloads = sorted({float(row["workload"]) for row in rows})
    figure, axes = plt.subplots(1, 3, figsize=(12.5, 4.0), constrained_layout=True)
    styles = {
        "on_policy_trace": ("o-", "On-policy trace"),
        "immediate_only": ("s--", "Immediate only"),
    }
    for axis, workload in zip(axes, workloads):
        for method, (style, label) in styles.items():
            points = []
            for (candidate, candidate_workload, value), selected in groups.items():
                if candidate == method and candidate_workload == workload:
                    row = selected[0]
                    points.append((
                        value,
                        float(row["physical_rehandles"]),
                        float(row["timing_cost_2T_plus_E"]),
                    ))
            points.sort()
            if points:
                axis.plot(
                    [point[1] for point in points],
                    [point[2] for point in points],
                    style,
                    linewidth=2,
                    label=label,
                )
                for value, rehandles, timing in points:
                    axis.annotate(
                        f"λ={value:g}",
                        (rehandles, timing),
                        xytext=(4, 4),
                        textcoords="offset points",
                        fontsize=8,
                    )
        axis.set_title(f"{100*workload:.0f}% workload")
        axis.set_xlabel("Physical rehandles")
        axis.set_ylabel("Timing cost 2T+E")
        axis.grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    figure.suptitle("E214 8×8 trace-return Stage B versus immediate-only")
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=180)
    plt.close(figure)
    return destination


__all__ = ["create_plot"]
