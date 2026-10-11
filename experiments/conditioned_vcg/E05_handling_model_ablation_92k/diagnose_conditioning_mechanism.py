#!/usr/bin/env python3
"""Explain why E5(b) preference conditioning rarely changes deployment.

The diagnostic reuses the finalized fixed candidate bank.  At each fixed
state/candidate coordinate it compares the trained prediction at lambda with
the same network's prediction at the E5(b) clamp value (.10).  It then applies
the unchanged hierarchical selector to both merit vectors.  No simulator,
training, or checkpoint selection is invoked.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
import math
from pathlib import Path
from statistics import fmean, median
import sys
from typing import Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import torch

from experiments.conditioned_vcg.E05_handling_model_ablation_92k import run as e05
from viability_graph_hierarchy import MODE_TO_ID
from viability_graph_preference_conditioned import (
    PreferenceConditionedVectorConfig,
    select_hierarchical_index,
)
import run_vcg_conditioned_final_comparison_90k as final90


PROTOCOL = "vcg_conditioned_e05b_conditioning_mechanism_v1"
REPORT_NAME = "e05b-conditioning-mechanism-report.json"
TABLE_NAME = "e05b-conditioning-mechanism-table.md"
STATE_ROWS_NAME = "e05b-conditioning-mechanism-states.csv"
FIGURE_STEM = "e05b-conditioning-mechanism"
ANCHOR_LAMBDA = e05.CLAMPED_INPUT_LAMBDA
TOLERANCE = 1.0e-7

SEED_COLORS = {0: "#0072B2", 1: "#D55E00", 2: "#009E73"}
BAR_COLORS = ("#8A929C", "#CC79A7", "#56B4E9")


class ConditioningMechanismError(RuntimeError):
    pass


def _read_csv(path: Path, *, label: str) -> list[dict]:
    path = Path(path).absolute()
    if not path.is_file() or path.is_symlink():
        raise ConditioningMechanismError(f"missing canonical {label}: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ConditioningMechanismError(f"empty {label}: {path}")
    return rows


def _atomic_csv(path: Path, rows: Sequence[Mapping]) -> None:
    if not rows:
        raise ConditioningMechanismError("cannot write an empty mechanism table")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _sources(project_root: Path) -> dict:
    d9 = e05._load_json(e05.D9_REPORT, label="D9 fixed-bank report")
    seed1 = e05._load_json(e05.SEED1_FINAL, label="final seed-1 continuation")
    try:
        final_seed1 = seed1["round_records"][-1]["fixed_bank_diagnostic"]
        paths = {
            "d9_values": Path(d9["artifacts"]["values_csv"]).resolve(),
            "d9_selections": Path(d9["artifacts"]["selections_csv"]).resolve(),
            "seed1_values": Path(final_seed1["values_csv"]).resolve(),
            "seed1_selections": Path(final_seed1["selections_csv"]).resolve(),
        }
    except (KeyError, IndexError, TypeError) as error:
        raise ConditioningMechanismError(
            "fixed-bank sources have an unexpected schema"
        ) from error

    expected_root = project_root.resolve() / "results"
    if any(expected_root not in path.parents for path in paths.values()):
        raise ConditioningMechanismError("fixed-bank CSV escaped the result root")
    if e05._sha256(paths["seed1_values"]) != final_seed1["values_sha256"]:
        raise ConditioningMechanismError("final seed-1 value bank hash changed")
    if e05._sha256(paths["seed1_selections"]) != final_seed1["selections_sha256"]:
        raise ConditioningMechanismError("final seed-1 selection bank hash changed")
    return {"d9": d9, "seed1": seed1, "final_seed1": final_seed1, **paths}


def _stitch_rows(sources: Mapping) -> tuple[list[dict], list[dict]]:
    d9_values = _read_csv(sources["d9_values"], label="D9 value bank")
    d9_selections = _read_csv(
        sources["d9_selections"], label="D9 selection bank"
    )
    values = [row for row in d9_values if int(row["seed"]) in (0, 2)]
    selections = [
        row for row in d9_selections if int(row["seed"]) in (0, 2)
    ]
    values.extend(_read_csv(sources["seed1_values"], label="final seed-1 values"))
    selections.extend(
        _read_csv(sources["seed1_selections"], label="final seed-1 selections")
    )
    expected_values = 3 * 686 * 17
    expected_selections = 3 * 34 * 17
    if len(values) != expected_values or len(selections) != expected_selections:
        raise ConditioningMechanismError(
            "stitched fixed bank is incomplete: "
            f"values={len(values)}/{expected_values}, "
            f"selections={len(selections)}/{expected_selections}"
        )
    return values, selections


def _float(row: Mapping, name: str) -> float:
    value = float(row[name])
    if not math.isfinite(value):
        raise ConditioningMechanismError(f"non-finite fixed-bank value: {name}")
    return value


def _selector(merits, rows):
    return select_hierarchical_index(
        torch.as_tensor(merits, dtype=torch.float32),
        [MODE_TO_ID[row["mode"]] for row in rows],
        PreferenceConditionedVectorConfig().within_temperatures,
        candidate_keys=[row["candidate_key"] for row in rows],
    )


def _within_mode_inversions(full, clamped, rows) -> tuple[int, int]:
    inversions = pairs = 0
    for left in range(len(rows)):
        for right in range(left + 1, len(rows)):
            if rows[left]["mode"] != rows[right]["mode"]:
                continue
            pairs += 1
            full_difference = full[left] - full[right]
            clamped_difference = clamped[left] - clamped[right]
            if full_difference * clamped_difference < 0.0:
                inversions += 1
    return inversions, pairs


def analyze_fixed_bank(
    values: Sequence[Mapping], selections: Sequence[Mapping]
) -> tuple[list[dict], dict]:
    groups = defaultdict(list)
    anchors = {}
    candidate_identity = {}
    for row in values:
        seed = int(row["seed"])
        state = row["state_id"]
        value = _float(row, "lambda")
        candidate_index = int(row["candidate_index"])
        coordinate = (seed, state, candidate_index)
        identity = (
            row["candidate_key"],
            row["action_type"],
            row["mode"],
            _float(row, "qop"),
            int(row["qn_immediate"]),
        )
        previous = candidate_identity.setdefault(coordinate, identity)
        if previous != identity:
            raise ConditioningMechanismError(
                f"candidate identity changed across lambda: {coordinate}"
            )
        groups[(seed, state, value)].append(row)
        if value == ANCHOR_LAMBDA:
            anchors[coordinate] = _float(row, "qn_future")

    if len(anchors) != 3 * 686 or len(groups) != 3 * 34 * 17:
        raise ConditioningMechanismError("fixed-bank coordinate structure changed")

    source_selections = {
        (int(row["seed"]), row["state_id"], _float(row, "lambda")): row
        for row in selections
    }
    if len(source_selections) != len(groups):
        raise ConditioningMechanismError("selection bank contains duplicate coordinates")

    state_rows = []
    reproduced = 0
    candidate_deltas = defaultdict(list)
    for coordinate, rows in sorted(groups.items()):
        seed, state, value = coordinate
        rows = sorted(rows, key=lambda row: int(row["candidate_index"]))
        if [int(row["candidate_index"]) for row in rows] != list(range(len(rows))):
            raise ConditioningMechanismError(
                f"candidate indices are not contiguous: {coordinate}"
            )
        delta = [
            _float(row, "qn_future")
            - anchors[(seed, state, int(row["candidate_index"]))]
            for row in rows
        ]
        full_merit = [_float(row, "merit") for row in rows]
        clamped_merit = [
            _float(row, "qop")
            - value
            * (
                int(row["qn_immediate"])
                + anchors[(seed, state, int(row["candidate_index"]))]
            )
            for row in rows
        ]
        full_selection = _selector(full_merit, rows)
        clamped_selection = _selector(clamped_merit, rows)
        source = source_selections.get(coordinate)
        if source is None:
            raise ConditioningMechanismError(
                f"missing source selection: {coordinate}"
            )
        full_key = rows[full_selection.selected_index]["candidate_key"]
        if full_key != source["selected_key"]:
            raise ConditioningMechanismError(
                f"failed to reproduce stored full selector: {coordinate}"
            )
        reproduced += 1

        clamped_key = rows[clamped_selection.selected_index]["candidate_key"]
        centered_mean = fmean(delta)
        centered = [item - centered_mean for item in delta]
        inversions, comparable_pairs = _within_mode_inversions(
            full_merit, clamped_merit, rows
        )
        candidate_deltas[(seed, value)].extend(delta)
        state_rows.append(
            {
                "seed": seed,
                "state_id": state,
                "lambda": value,
                "candidate_count": len(rows),
                "responding_candidates": sum(
                    abs(item) > TOLERANCE for item in delta
                ),
                "mean_delta_qn_future": centered_mean,
                "mean_abs_delta_qn_future": fmean(abs(item) for item in delta),
                "mean_abs_centered_delta": fmean(abs(item) for item in centered),
                "delta_range": max(delta) - min(delta),
                "effective_merit_span": value * (max(delta) - min(delta)),
                "within_mode_inversions": inversions,
                "within_mode_comparable_pairs": comparable_pairs,
                "within_mode_order_changed": inversions > 0,
                "full_selected_key": full_key,
                "clamped_selected_key": clamped_key,
                "selected_action_changed": (
                    full_selection.selected_index != clamped_selection.selected_index
                ),
                "selected_mode_changed": (
                    full_selection.selected_mode_id
                    != clamped_selection.selected_mode_id
                ),
            }
        )

    if reproduced != 3 * 34 * 17:
        raise ConditioningMechanismError("not every source selection was reproduced")

    lambda_grid = sorted({float(row["lambda"]) for row in state_rows})
    by_lambda = {}
    by_seed = {str(seed): {} for seed in e05.MODEL_SEEDS}
    for value in lambda_grid:
        selected = [row for row in state_rows if float(row["lambda"]) == value]
        raw = [item for seed in e05.MODEL_SEEDS for item in candidate_deltas[(seed, value)]]
        by_lambda[f"lambda_{value:.4f}"] = _summarize_coordinate(selected, raw)
        for seed in e05.MODEL_SEEDS:
            seed_rows = [row for row in selected if int(row["seed"]) == seed]
            by_seed[str(seed)][f"lambda_{value:.4f}"] = _summarize_coordinate(
                seed_rows, candidate_deltas[(seed, value)]
            )
    return state_rows, {
        "lambda_grid": lambda_grid,
        "by_lambda": by_lambda,
        "by_seed": by_seed,
        "stored_full_selections_reproduced": reproduced,
    }


def _summarize_coordinate(state_rows: Sequence[Mapping], raw) -> dict:
    raw = list(raw)
    candidate_count = len(raw)
    frontiers = len(state_rows)
    if not raw or not state_rows:
        raise ConditioningMechanismError("cannot summarize an empty coordinate")
    spans = [float(row["delta_range"]) for row in state_rows]
    merit_spans = [float(row["effective_merit_span"]) for row in state_rows]
    return {
        "candidates": candidate_count,
        "responding_candidates": sum(abs(item) > TOLERANCE for item in raw),
        "responding_candidate_fraction": fmean(
            abs(item) > TOLERANCE for item in raw
        ),
        "mean_delta_qn_future": fmean(raw),
        "mean_abs_delta_qn_future": fmean(abs(item) for item in raw),
        "minimum_delta_qn_future": min(raw),
        "maximum_delta_qn_future": max(raw),
        "frontiers": frontiers,
        "mean_abs_centered_delta": (
            sum(
                float(row["mean_abs_centered_delta"])
                * int(row["candidate_count"])
                for row in state_rows
            )
            / candidate_count
        ),
        "mean_frontier_delta_range": fmean(spans),
        "median_frontier_delta_range": median(spans),
        "maximum_frontier_delta_range": max(spans),
        "mean_effective_merit_span": fmean(merit_spans),
        "median_effective_merit_span": median(merit_spans),
        "maximum_effective_merit_span": max(merit_spans),
        "frontiers_with_within_mode_order_change": sum(
            bool(row["within_mode_order_changed"]) for row in state_rows
        ),
        "within_mode_order_change_fraction": fmean(
            bool(row["within_mode_order_changed"]) for row in state_rows
        ),
        "selected_action_changes": sum(
            bool(row["selected_action_changed"]) for row in state_rows
        ),
        "selected_action_change_fraction": fmean(
            bool(row["selected_action_changed"]) for row in state_rows
        ),
        "selected_mode_changes": sum(
            bool(row["selected_mode_changed"]) for row in state_rows
        ),
    }


def rollout_behavior_context(output_dir: Path, e04_output: Path) -> dict:
    rows = e05._all_clamped_rows(output_dir)
    expected = len(e05.MODEL_SEEDS) * len(e05.NEW_CLAMPED_LAMBDAS) * len(
        e05.INSTANCE_SEEDS
    )
    if len(rows) != expected:
        raise ConditioningMechanismError(
            f"E5(b) rollout grid is incomplete: {len(rows)}/{expected}"
        )
    result = {
        f"lambda_{ANCHOR_LAMBDA:.2f}": {
            "pairs": len(e05.MODEL_SEEDS) * len(e05.INSTANCE_SEEDS),
            "different_behavior_digests": 0,
            "difference_fraction": 0.0,
            "identity_by_construction": True,
            "by_seed": {
                str(seed): {"pairs": len(e05.INSTANCE_SEEDS), "different": 0}
                for seed in e05.MODEL_SEEDS
            },
        }
    }
    for value in e05.NEW_CLAMPED_LAMBDAS:
        selected = [row for row in rows if float(row["deployment_lambda"]) == value]
        by_seed = {}
        total_different = 0
        for seed in e05.MODEL_SEEDS:
            seed_rows = [row for row in selected if int(row["model_seed"]) == seed]
            different = 0
            for row in seed_rows:
                control = e05._control_row(
                    e04_output, seed, value, int(row["instance_seed"])
                )
                if control is None:
                    raise ConditioningMechanismError(
                        "matched E4 rollout control is missing"
                    )
                different += (
                    row.get("behavior_digest") != control.get("behavior_digest")
                )
            by_seed[str(seed)] = {"pairs": len(seed_rows), "different": different}
            total_different += different
        result[f"lambda_{value:.2f}"] = {
            "pairs": len(selected),
            "different_behavior_digests": total_different,
            "difference_fraction": total_different / len(selected),
            "identity_by_construction": False,
            "by_seed": by_seed,
        }
    return dict(sorted(result.items()))


def _table(report: Mapping) -> str:
    lines = [
        "# E5(b) conditioning mechanism diagnostic",
        "",
        "| Deployment lambda | Responding candidates | Mean delta Q_N | Mean frontier delta range | Mean effective merit span | Frontiers with within-mode order change | Selected-action changes | Rollout trajectories changed |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for value in e05.DEPLOYMENT_LAMBDAS:
        fixed = report["fixed_bank"]["by_lambda"][f"lambda_{value:.4f}"]
        rollout = report["rollout_behavior"][f"lambda_{value:.2f}"]
        lines.append(
            f"| {value:.2f} | {fixed['responding_candidates']}/{fixed['candidates']} | "
            f"{fixed['mean_delta_qn_future']:+.4f} | "
            f"{fixed['mean_frontier_delta_range']:.4f} | "
            f"{fixed['mean_effective_merit_span']:.5f} | "
            f"{fixed['frontiers_with_within_mode_order_change']}/{fixed['frontiers']} | "
            f"{fixed['selected_action_changes']}/{fixed['frontiers']} | "
            f"{rollout['different_behavior_digests']}/{rollout['pairs']} |"
        )
    lines.extend(
        [
            "",
            "Here delta Q_N = N_future(h, lambda) - N_future(h, .10). The effective merit span is lambda times the within-frontier range of delta Q_N; candidate-common shifts therefore do not inflate the selection-relevant quantity.",
            "",
            "The diagnostic establishes predictor responsiveness and its effect on stored candidate rankings. It does not test whether the response is correct against lambda-specific Monte-Carlo continuation policies, nor whether conditioned training is necessary.",
        ]
    )
    return "\n".join(lines) + "\n"


def render_figure(output_dir: Path, report: Mapping) -> list[str]:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 11,
            "legend.fontsize": 8.5,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
        }
    )
    figure, (response_axis, decision_axis) = plt.subplots(
        1, 2, figsize=(10.8, 4.45), gridspec_kw={"width_ratios": (1.15, 1.0)}
    )
    grid = report["fixed_bank"]["lambda_grid"]
    for seed in e05.MODEL_SEEDS:
        means = [
            report["fixed_bank"]["by_seed"][str(seed)][
                f"lambda_{value:.4f}"
            ]["mean_delta_qn_future"]
            for value in grid
        ]
        response_axis.plot(
            grid,
            means,
            color=SEED_COLORS[seed],
            linewidth=1.2,
            marker="o",
            markersize=3.2,
            alpha=0.62,
            label=f"seed {seed}",
        )
    aggregate = [
        report["fixed_bank"]["by_lambda"][f"lambda_{value:.4f}"][
            "mean_delta_qn_future"
        ]
        for value in grid
    ]
    response_axis.plot(
        grid,
        aggregate,
        color="#202832",
        linewidth=2.4,
        marker="o",
        markersize=4.5,
        label="three-seed mean",
        zorder=5,
    )
    response_axis.axhline(0.0, color="#7A828C", linewidth=0.9, linestyle="--")
    response_axis.axvline(
        ANCHOR_LAMBDA, color="#B2B7BD", linewidth=0.9, linestyle=":"
    )
    response_axis.set_xlabel(r"Predictor input $\lambda$")
    response_axis.set_ylabel(r"Mean $\Delta\widehat N_{future}$ from input $.10$")
    response_axis.set_title("(a) Predictor response on fixed candidates", loc="left")
    response_axis.legend(frameon=False, ncol=2, loc="upper right")

    deployment = (0.05, 0.20)
    categories = (
        ("within_mode_order_change_fraction", "Any within-mode\norder change"),
        ("selected_action_change_fraction", "Selected action\nchanged"),
        (None, "Rollout trajectory\nchanged"),
    )
    width = 0.22
    x_positions = range(len(deployment))
    for category_index, (metric, label) in enumerate(categories):
        offsets = [x + (category_index - 1) * width for x in x_positions]
        values = []
        counts = []
        for value in deployment:
            if metric is None:
                item = report["rollout_behavior"][f"lambda_{value:.2f}"]
                values.append(100.0 * item["difference_fraction"])
                counts.append(
                    f"{item['different_behavior_digests']}/{item['pairs']}"
                )
            else:
                item = report["fixed_bank"]["by_lambda"][
                    f"lambda_{value:.4f}"
                ]
                values.append(100.0 * item[metric])
                numerator = (
                    item["frontiers_with_within_mode_order_change"]
                    if metric == "within_mode_order_change_fraction"
                    else item["selected_action_changes"]
                )
                counts.append(f"{numerator}/{item['frontiers']}")
        bars = decision_axis.bar(
            offsets,
            values,
            width=width,
            color=BAR_COLORS[category_index],
            label=label,
            zorder=3,
        )
        for bar, count in zip(bars, counts):
            decision_axis.annotate(
                count,
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
                color="#4B535C",
            )
    decision_axis.set_xticks(list(x_positions), [r"$\lambda=.05$", r"$\lambda=.20$"])
    decision_axis.set_ylabel("Coordinates affected (%)")
    decision_axis.set_title("(b) Ranking response rarely reaches deployment", loc="left")
    decision_axis.legend(frameon=False, loc="upper left")
    decision_axis.set_ylim(0.0, 45.0)

    for axis in (response_axis, decision_axis):
        axis.grid(axis="y", alpha=0.20, linewidth=0.8, zorder=0)
        axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    paths = []
    for suffix in ("pdf", "png", "svg"):
        path = output_dir / f"{FIGURE_STEM}.{suffix}"
        figure.savefig(path, dpi=260, bbox_inches="tight")
        paths.append(str(path.resolve()))
    plt.close(figure)
    return paths


def run(project_root: Path, output_dir: Path, e04_output: Path) -> dict:
    contract = e05.authenticate_contract(project_root, output_dir, e04_output)
    e05_report = e05._load_json(output_dir / e05.REPORT_NAME, label="E5(b) report")
    e05._verify_hash(e05_report, "report_sha256", label="E5(b) report")
    validation = e05._load_json(
        output_dir / e05.VALIDATION_NAME, label="E5(a) validation"
    )
    e05._verify_hash(validation, "validation_sha256", label="E5(a) validation")
    if (
        e05_report.get("status") != "complete"
        or e05_report.get("paper_evidence") is not True
        or e05_report.get("contract_sha256") != contract["contract_sha256"]
        or e05_report.get("e05a_validation_sha256")
        != validation["validation_sha256"]
    ):
        raise ConditioningMechanismError(
            "mechanism analysis requires authenticated complete E5(a,b) results"
        )

    sources = _sources(project_root)
    if (
        validation["d9_report_sha256"] != e05._sha256(e05.D9_REPORT)
        or validation["seed1_final_report_sha256"] != e05._sha256(e05.SEED1_FINAL)
    ):
        raise ConditioningMechanismError("E5 fixed-bank parent reports changed")
    values, selections = _stitch_rows(sources)
    state_rows, fixed_bank = analyze_fixed_bank(values, selections)
    rollout = rollout_behavior_context(output_dir, e04_output)
    report = e05._with_hash(
        {
            "schema_version": 1,
            "protocol": PROTOCOL,
            "status": "complete",
            "new_training": False,
            "new_rollouts": 0,
            "checkpoint_selection": False,
            "anchor_lambda": ANCHOR_LAMBDA,
            "e05_contract_sha256": contract["contract_sha256"],
            "e05b_report_sha256": e05_report["report_sha256"],
            "e05a_validation_sha256": validation["validation_sha256"],
            "source_sha256": {
                "d9_values": e05._sha256(sources["d9_values"]),
                "d9_selections": e05._sha256(sources["d9_selections"]),
                "seed1_values": e05._sha256(sources["seed1_values"]),
                "seed1_selections": e05._sha256(sources["seed1_selections"]),
            },
            "fixed_bank": fixed_bank,
            "rollout_behavior": rollout,
            "established": {
                "predictor_responds_to_preference_input_on_fixed_bank": True,
                "response_can_change_candidate_relative_order": True,
                "selected_action_changes_are_sparse_on_fixed_bank": True,
                "varying_input_provided_little_additional_deployment_benefit_on_E5b_panel": True,
            },
            "not_established": {
                "correct_policy_conditioned_adaptation_against_lambda_specific_continuations": True,
                "conditioned_training_is_necessary": True,
                "conditioning_is_intrinsically_useless": True,
            },
        },
        "report_sha256",
    )
    final90._atomic_json(output_dir / REPORT_NAME, report)
    final90._atomic_text(output_dir / TABLE_NAME, _table(report))
    _atomic_csv(output_dir / STATE_ROWS_NAME, state_rows)
    figures = render_figure(output_dir, report)
    return {
        "status": "complete",
        "fixed_state_lambda_coordinates": len(state_rows),
        "stored_full_selections_reproduced": fixed_bank[
            "stored_full_selections_reproduced"
        ],
        "report": str((output_dir / REPORT_NAME).resolve()),
        "table": str((output_dir / TABLE_NAME).resolve()),
        "state_rows": str((output_dir / STATE_ROWS_NAME).resolve()),
        "figures": figures,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=e05.DEFAULT_OUTPUT)
    parser.add_argument("--e04-output", type=Path, default=e05.E04_OUTPUT)
    args = parser.parse_args()
    result = run(
        args.project_root.resolve(),
        args.output_dir.resolve(),
        args.e04_output.resolve(),
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
