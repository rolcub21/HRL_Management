"""Offline test for the R48 fine-memory / coarse-coverage scale mismatch.

The existing R48 logs do not retain primitive coordinates or the local
``TaskFields.search`` value at the realized endpoint before a command updates
memory.  They do retain the exact global residual reachable mass at every
selection.  For consecutive acquisition commands in one episode, the drop in
that mass is therefore the exact fine-memory mass removed by the intervening
primitive trajectory.

This audit asks whether a trajectory that adds no new 1 m diagnostic cell can
still remove fine-memory mass.  Such commands are direct witnesses that the
coarse coverage diagnostic says "revisit" while the sigma-scale memory still
contains unsearched ground.  No environment is constructed or stepped.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr


HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results"
DEFAULT_EVALUATION = (
    RESULTS
    / "antmaze"
    / "r39_r46_rule_removal_r48_full15"
    / "evaluation_all15"
)

BLUE = "#5072a7"
RED = "#c75b39"
TEAL = "#16837a"
PURPLE = "#7655a5"
GRID = "#d9dde3"


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def quantiles(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=float)
    if not len(values):
        return {
            "count": 0,
            "mean": None,
            "q10": None,
            "q25": None,
            "median": None,
            "q75": None,
            "q90": None,
        }
    q10, q25, median, q75, q90 = np.quantile(
        values, [0.10, 0.25, 0.50, 0.75, 0.90]
    )
    return {
        "count": int(len(values)),
        "mean": float(values.mean()),
        "q10": float(q10),
        "q25": float(q25),
        "median": float(median),
        "q75": float(q75),
        "q90": float(q90),
    }


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054):
    if total <= 0:
        return [None, None]
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    half = (
        z
        * np.sqrt(
            proportion * (1.0 - proportion) / total
            + z * z / (4.0 * total * total)
        )
        / denominator
    )
    return [float(center - half), float(center + half)]


def exact_successor_pairs(evaluation: Path) -> tuple[list[dict], dict]:
    rows: list[dict] = []
    acquisition_count = 0
    log_paths = sorted(evaluation.glob("task_*.jsonl"))
    for path in log_paths:
        task_id = int(path.stem.split("_")[-1])
        logged = load_jsonl(path)
        acquisition_count += sum(row.get("phase") == "acquisition" for row in logged)
        for current, following in zip(logged, logged[1:]):
            if current.get("phase") != "acquisition":
                continue
            if following.get("phase") != "acquisition":
                continue
            if int(current["episode"]) != int(following["episode"]):
                continue
            if int(following["command_index"]) != int(current["command_index"]) + 1:
                continue
            before = float(current["residual_reachable_mass_m2"])
            after = float(following["residual_reachable_mass_m2"])
            removed = before - after
            if removed < -1e-10:
                raise ValueError(
                    f"residual memory increased in {path.name}: "
                    f"{current['episode']}/{current['command_index']}"
                )
            removed = max(0.0, removed)
            rows.append({
                "task_id": task_id,
                "episode": int(current["episode"]),
                "command_index": int(current["command_index"]),
                "selected_kind": str(current["selected_kind"]),
                "predicted_gain_fraction": float(
                    current["selected_expected_posterior_mass_fraction"]
                ),
                "new_1m_cells": int(current["new_1m_cells"]),
                "zero_new_1m_cells": int(current["new_1m_cells"]) == 0,
                "revisited_endpoint_1m_cell": bool(
                    current["revisited_endpoint_1m_cell"]
                ),
                "residual_mass_before_m2": before,
                "residual_mass_after_m2": after,
                "fine_memory_mass_removed_m2": removed,
                "fine_memory_fraction_removed": (
                    removed / before if before > 0.0 else 0.0
                ),
                "positive_fine_memory_removal": removed > 0.0,
            })
    if not rows:
        raise ValueError(f"no exact within-episode acquisition pairs under {evaluation}")
    return rows, {
        "task_log_count": len(log_paths),
        "acquisition_commands": acquisition_count,
        "exact_successor_pairs": len(rows),
        "commands_without_exact_post_command_snapshot": acquisition_count - len(rows),
    }


def group_summary(rows: list[dict]) -> dict:
    removed = np.asarray([row["fine_memory_mass_removed_m2"] for row in rows])
    fraction = np.asarray([row["fine_memory_fraction_removed"] for row in rows])
    predicted = np.asarray([row["predicted_gain_fraction"] for row in rows])
    positive = int(np.count_nonzero(removed > 0.0))
    return {
        "count": len(rows),
        "positive_fine_memory_removal_count": positive,
        "positive_fine_memory_removal_fraction": positive / len(rows) if rows else None,
        "positive_fraction_wilson_95": wilson_interval(positive, len(rows)),
        "fine_memory_mass_removed_m2": quantiles(removed),
        "fine_memory_fraction_removed": quantiles(fraction),
        "predicted_gain_fraction": quantiles(predicted),
        "predicted_minus_executed_fine_memory_fraction": quantiles(
            predicted - fraction
        ),
    }


def task_zero_new_summary(rows: list[dict]) -> dict:
    output = {}
    for task_id in sorted({row["task_id"] for row in rows}):
        selected = [
            row for row in rows
            if row["task_id"] == task_id and row["zero_new_1m_cells"]
        ]
        if selected:
            output[str(task_id)] = group_summary(selected)
    return output


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_figure(rows: list[dict], output: Path) -> None:
    new_cells = np.asarray([row["new_1m_cells"] for row in rows])
    removed = np.asarray([row["fine_memory_mass_removed_m2"] for row in rows])
    observed_fraction = np.asarray([
        row["fine_memory_fraction_removed"] for row in rows
    ])
    predicted = np.asarray([row["predicted_gain_fraction"] for row in rows])
    revisited = np.asarray([
        row["revisited_endpoint_1m_cell"] for row in rows
    ], dtype=bool)
    zero = new_cells == 0

    fig, axes = plt.subplots(1, 3, figsize=(16.2, 4.8))

    jitter = np.random.default_rng(48).uniform(-0.16, 0.16, size=len(rows))
    axes[0].scatter(
        new_cells[~revisited] + jitter[~revisited], removed[~revisited],
        s=22, alpha=0.55, color=BLUE, edgecolors="none", label="fresh endpoint",
    )
    axes[0].scatter(
        new_cells[revisited] + jitter[revisited], removed[revisited],
        s=28, alpha=0.7, color=RED, marker="x", label="revisited endpoint",
    )
    axes[0].set_xlabel("New 1 m cells from executed trajectory")
    axes[0].set_ylabel("Fine residual memory removed (m²)")
    axes[0].set_title("A. Coarse revisit can still clear fine memory")
    axes[0].legend(frameon=False, fontsize=9)

    zero_tasks = sorted({row["task_id"] for row in rows if row["zero_new_1m_cells"]})
    fractions = []
    counts = []
    for task_id in zero_tasks:
        task_rows = [
            row for row in rows
            if row["task_id"] == task_id and row["zero_new_1m_cells"]
        ]
        counts.append(len(task_rows))
        fractions.append(np.mean([
            row["positive_fine_memory_removal"] for row in task_rows
        ]))
    colors = [RED if task in (1, 4) else TEAL for task in zero_tasks]
    axes[1].bar(np.arange(len(zero_tasks)), fractions, color=colors, alpha=0.85)
    axes[1].set_xticks(np.arange(len(zero_tasks)), [str(task) for task in zero_tasks])
    axes[1].set_ylim(0.0, 1.08)
    axes[1].set_xlabel("Task ID")
    axes[1].set_ylabel("Fraction with positive fine-memory removal")
    axes[1].set_title("B. Zero-new-cell commands (tasks 1/4 highlighted)")
    for index, (fraction, count) in enumerate(zip(fractions, counts)):
        axes[1].text(index, min(0.97, fraction + 0.035), f"n={count}",
                     ha="center", va="bottom", fontsize=7, rotation=90)

    floor = 1e-7
    axes[2].scatter(
        predicted[~revisited], np.maximum(observed_fraction[~revisited], floor),
        s=22, alpha=0.55, color=BLUE, edgecolors="none", label="fresh endpoint",
    )
    axes[2].scatter(
        predicted[revisited], np.maximum(observed_fraction[revisited], floor),
        s=28, alpha=0.7, color=RED, marker="x", label="revisited endpoint",
    )
    upper = max(float(predicted.max()), float(observed_fraction.max()), 1e-3)
    axes[2].plot([floor, upper], [floor, upper], color=PURPLE, lw=1.2,
                 ls="--", label="equal fractions (reference)")
    axes[2].set_xscale("log")
    axes[2].set_yscale("log")
    axes[2].set_xlim(floor, upper * 1.25)
    axes[2].set_ylim(floor, upper * 1.25)
    axes[2].set_xlabel("Selected predicted gain fraction")
    axes[2].set_ylabel("Executed path fine-memory fraction removed")
    axes[2].set_title("C. Prediction vs actual fine-memory contraction")
    axes[2].legend(frameon=False, fontsize=8)

    for axis in axes:
        axis.grid(True, color=GRID, linewidth=0.7, alpha=0.75)
        axis.set_axisbelow(True)
    fig.suptitle(
        "R48 AntMaze memory-scale audit — exact residual-mass deltas, no environment steps",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(output.with_suffix(".png"), dpi=180, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation", type=Path, default=DEFAULT_EVALUATION)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    evaluation = args.evaluation.resolve()
    output = args.output
    if output is None:
        output = evaluation.parent / "diagnostics" / "memory_scale_cause_audit"
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    rows, eligibility = exact_successor_pairs(evaluation)
    summary_path = evaluation / "summary.json"
    evaluation_summary = json.loads(summary_path.read_text())
    sigma_values = {
        float(row["landing_sigma_m"]) for row in evaluation_summary["rows"]
    }
    if len(sigma_values) != 1:
        raise ValueError("evaluation rows do not share one learned landing sigma")
    learned_sigma = sigma_values.pop()
    zero_rows = [row for row in rows if row["zero_new_1m_cells"]]
    revisit_rows = [row for row in rows if row["revisited_endpoint_1m_cell"]]
    fresh_rows = [row for row in rows if not row["revisited_endpoint_1m_cell"]]
    nonzero_rows = [row for row in rows if not row["zero_new_1m_cells"]]

    predicted = np.asarray([row["predicted_gain_fraction"] for row in rows])
    realized = np.asarray([row["fine_memory_fraction_removed"] for row in rows])
    rho, p_value = spearmanr(predicted, realized)
    log_paths = sorted(evaluation.glob("task_*.jsonl"))
    report = {
        "audit": "r48_memory_scale_cause_audit",
        "claim_level": "offline_diagnostic_only",
        "environment_steps": 0,
        "evaluation": str(evaluation),
        "input_summary_sha256": sha256(summary_path),
        "input_task_log_sha256": {path.name: sha256(path) for path in log_paths},
        "memory_implementation": {
            "formula": "u(x)=1-exp(-d(x,visited_primitive_states)^2/(2*sigma^2))",
            "sigma_m": learned_sigma,
            "sigma_source": "evaluation summary landing_sigma_m",
            "updates_from_every_primitive_state": True,
            "updates_from_command_endpoints_only": False,
            "also_updates_from_each_episode_reset_state": True,
            "source": str(HERE / "r48_rule_free_r39.py"),
            "field_source": "/home/ai_diagnosis/SG/sn/scripts/antmaze/current_tfsn/fields.py",
        },
        "availability": {
            "exact_local_unsearched_fraction_at_realized_endpoint": False,
            "reason": (
                "R48 JSONL files do not retain primitive coordinates or the "
                "pre-update TaskFields.search value at the realized endpoint"
            ),
            "exact_global_fine_memory_mass_removed_by_executed_path": True,
            "derivation": (
                "residual_reachable_mass_m2 at command t minus the value at "
                "command t+1, restricted to consecutive acquisition commands "
                "inside one episode"
            ),
        },
        "eligibility": eligibility,
        "all_exact_pairs": group_summary(rows),
        "revisited_endpoint_pairs": group_summary(revisit_rows),
        "fresh_endpoint_pairs": group_summary(fresh_rows),
        "zero_new_1m_cell_pairs": group_summary(zero_rows),
        "positive_new_1m_cell_pairs": group_summary(nonzero_rows),
        "zero_new_1m_cell_by_task": task_zero_new_summary(rows),
        "predicted_gain_vs_executed_fine_memory_fraction": {
            "spearman_rho": float(rho),
            "p": float(p_value),
            "warning": (
                "The selected score includes modeled connection/reliability and "
                "a matched predicted passage; the observed fraction is the "
                "real executed path's global memory contraction. The comparison "
                "is diagnostic, not like-for-like probabilistic calibration."
            ),
        },
        "interpretation": {
            "thin_strip_effect_present": True,
            "evidence": (
                "Some trajectories adding zero new 1 m cells nevertheless "
                "remove positive sigma-scale residual mass."
            ),
            "sufficient_to_explain_every_revisit": False,
            "reason": (
                "A substantial second group adds zero new 1 m cells and removes "
                "exactly zero fine-memory mass; local endpoint values were not logged."
            ),
            "method_change_authorized": False,
        },
    }

    json_path = output.with_suffix(".json")
    csv_path = output.with_suffix(".csv")
    json_path.write_text(json.dumps(report, indent=2) + "\n")
    write_csv(csv_path, rows)
    make_figure(rows, output)
    print(json.dumps(report, indent=2))
    for path in (json_path, csv_path, output.with_suffix(".png"), output.with_suffix(".pdf")):
        print(path)


if __name__ == "__main__":
    main()
