"""Endpoint plot for E208."""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def create_plot(rows: list[dict], destination: Path) -> Path:
    workloads = sorted({float(row["workload"]) for row in rows})
    figure, axes = plt.subplots(1, 3, figsize=(12.5, 4.0), constrained_layout=True)
    for axis, workload in zip(axes, workloads):
        selected = sorted((row for row in rows if row["workload"] == workload), key=lambda row: row["lambda"])
        axis.plot([row["physical_rehandles"] for row in selected], [row["timing_cost_2T_plus_E"] for row in selected], marker="o", linewidth=2)
        for row in selected:
            axis.annotate(f"lambda={row['lambda']:g}", (row["physical_rehandles"], row["timing_cost_2T_plus_E"]), xytext=(4,4), textcoords="offset points", fontsize=8)
        axis.set_title(f"{100*workload:.0f}% workload")
        axis.set_xlabel("Physical rehandles")
        axis.set_ylabel("Timing cost 2T+E")
        axis.grid(alpha=0.25)
    figure.suptitle("E208 on-policy TD Stage-B endpoint")
    destination = Path(destination); destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=180); plt.close(figure)
    return destination


__all__ = ["create_plot"]
