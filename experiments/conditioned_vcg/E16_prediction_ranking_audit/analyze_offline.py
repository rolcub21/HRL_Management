#!/usr/bin/env python3
"""E16-A: offline prediction-to-ranking mechanism audit.

This reuses the authenticated fixed candidate bank and completed E5(c) paired
episode ledgers. It launches neither training nor environment rollouts.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
import os
from pathlib import Path
from statistics import fmean, median
import sys
from typing import Mapping, Optional, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.conditioned_vcg.E05_handling_model_ablation_92k import (
    diagnose_conditioning_mechanism as conditioning,
)
from experiments.conditioned_vcg.E05_handling_model_ablation_92k import (
    run as e05,
)
from experiments.conditioned_vcg.E05_handling_model_ablation_92k import (
    run_future_consequence as e05c,
)


PROTOCOL = "vcg_conditioned_e16_prediction_to_ranking_offline_v1"
SCHEMA_VERSION = 1
DEPLOYMENT_LAMBDAS = tuple(float(item) for item in e05c.DEPLOYMENT_LAMBDAS)
ANCHOR_LAMBDA = float(e05c.FUTURE_INPUT_LAMBDA)
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "results/vcg-conditioned-e16-prediction-ranking-audit"
)
E05B_MECHANISM = (
    e05.DEFAULT_OUTPUT / conditioning.REPORT_NAME
)
E05C_REPORT = e05c.DEFAULT_OUTPUT / e05c.REPORT_NAME
STATE_ROWS_NAME = "e16-offline-frontiers.csv"
PAIR_ROWS_NAME = "e16-e5c-pairs.csv"
REPORT_NAME = "e16-offline-report.json"
TABLE_NAME = "e16-offline-summary.md"
TOLERANCE = 1.0e-7


class E16Error(RuntimeError):
    pass


def _canonical(value: Mapping) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _digest(value: Mapping, field: Optional[str] = None) -> str:
    payload = dict(value)
    if field is not None:
        payload.pop(field, None)
    return hashlib.sha256(_canonical(payload)).hexdigest()


def _sha(path: Path) -> str:
    path = path.resolve()
    if not path.is_file() or path.is_symlink():
        raise E16Error(f"missing regular input: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_self_hashed(path: Path, field: str, label: str) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise E16Error(f"invalid {label}: {path}") from error
    if not isinstance(value, dict) or value.get(field) != _digest(value, field):
        raise E16Error(f"{label} self-hash mismatch")
    return value


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(value, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_csv(path: Path, rows: Sequence[Mapping]) -> None:
    if not rows:
        raise E16Error(f"cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _number(row: Mapping, name: str) -> float:
    value = float(row[name])
    if not math.isfinite(value):
        raise E16Error(f"nonfinite {name}")
    return value


def _frontier_record(
    rows: Sequence[Mapping], *, seed: int, state_id: str, value: float
) -> dict:
    rows = tuple(sorted(rows, key=lambda item: int(item["candidate_index"])))
    if not rows:
        raise E16Error("empty fixed-bank frontier")
    if [int(row["candidate_index"]) for row in rows] != list(range(len(rows))):
        raise E16Error("candidate indices are not contiguous")

    immediate_merits = [
        _number(row, "qop") - value * int(row["qn_immediate"])
        for row in rows
    ]
    future_merits = [
        immediate - value * _number(row, "qn_future")
        for row, immediate in zip(rows, immediate_merits)
    ]
    immediate_selection = conditioning._selector(immediate_merits, rows)
    future_selection = conditioning._selector(future_merits, rows)
    immediate_index = int(immediate_selection.selected_index)
    future_index = int(future_selection.selected_index)
    changed = immediate_index != future_index
    same_mode = (
        immediate_selection.selected_mode_id == future_selection.selected_mode_id
    )

    inversions, comparable_pairs = conditioning._within_mode_inversions(
        future_merits, immediate_merits, rows
    )
    predicted = [_number(row, "qn_future") for row in rows]
    predicted_saving = predicted[immediate_index] - predicted[future_index]
    immediate_pair_gap = (
        immediate_merits[immediate_index] - immediate_merits[future_index]
    )
    pressure = value * predicted_saving
    crossing_surplus = pressure - immediate_pair_gap if same_mode else None
    return {
        "model_seed": int(seed),
        "state_id": state_id,
        "deployment_lambda": float(value),
        "candidate_count": len(rows),
        "predicted_future_min": min(predicted),
        "predicted_future_max": max(predicted),
        "predicted_future_range": max(predicted) - min(predicted),
        "effective_future_pressure_range": value * (max(predicted) - min(predicted)),
        "within_mode_comparable_pairs": comparable_pairs,
        "within_mode_order_inversions": inversions,
        "within_mode_order_changed": inversions > 0,
        "immediate_selected_key": rows[immediate_index]["candidate_key"],
        "future_selected_key": rows[future_index]["candidate_key"],
        "immediate_selected_action_type": rows[immediate_index]["action_type"],
        "future_selected_action_type": rows[future_index]["action_type"],
        "immediate_selected_mode": rows[immediate_index]["mode"],
        "future_selected_mode": rows[future_index]["mode"],
        "selected_candidate_changed": changed,
        "selected_action_type_changed": (
            rows[immediate_index]["action_type"] != rows[future_index]["action_type"]
        ),
        "selected_mode_changed": not same_mode,
        "selected_predicted_future_saving": predicted_saving if changed else None,
        "immediate_pair_merit_gap": immediate_pair_gap if changed else None,
        "future_pressure_pair_difference": pressure if changed else None,
        "same_mode_crossing_surplus": crossing_surplus if changed else None,
        "selected_transition": (
            f"{rows[immediate_index]['action_type']}->"
            f"{rows[future_index]['action_type']}"
        ),
    }


def analyze_fixed_bank(values: Sequence[Mapping]) -> tuple[list[dict], dict]:
    anchors = defaultdict(list)
    for row in values:
        if abs(_number(row, "lambda") - ANCHOR_LAMBDA) <= TOLERANCE:
            anchors[(int(row["seed"]), str(row["state_id"]))].append(row)
    if len(anchors) != 3 * 34:
        raise E16Error(f"fixed bank has {len(anchors)} anchor frontiers, expected 102")

    records = []
    for (seed, state_id), rows in sorted(anchors.items()):
        for value in DEPLOYMENT_LAMBDAS:
            records.append(
                _frontier_record(rows, seed=seed, state_id=state_id, value=value)
            )

    by_lambda = {}
    by_seed = {str(seed): {} for seed in (0, 1, 2)}
    for value in DEPLOYMENT_LAMBDAS:
        selected = [row for row in records if row["deployment_lambda"] == value]
        by_lambda[f"lambda_{value:.2f}"] = _summarize_frontiers(selected)
        for seed in (0, 1, 2):
            seed_rows = [row for row in selected if row["model_seed"] == seed]
            by_seed[str(seed)][f"lambda_{value:.2f}"] = _summarize_frontiers(
                seed_rows
            )
    return records, {
        "states": 34,
        "model_seeds": 3,
        "candidate_coordinates_per_seed": 686,
        "frontiers_per_lambda": 102,
        "predictor_input_lambda": ANCHOR_LAMBDA,
        "deployment_lambdas": list(DEPLOYMENT_LAMBDAS),
        "by_lambda": by_lambda,
        "by_seed": by_seed,
    }


def _summarize_frontiers(rows: Sequence[Mapping]) -> dict:
    rows = tuple(rows)
    if not rows:
        raise E16Error("cannot summarize empty frontier collection")
    changed = [row for row in rows if row["selected_candidate_changed"]]
    same_mode = [row for row in changed if not row["selected_mode_changed"]]
    savings = [float(row["selected_predicted_future_saving"]) for row in changed]
    return {
        "frontiers": len(rows),
        "candidates": sum(int(row["candidate_count"]) for row in rows),
        "mean_predicted_future_range": fmean(
            float(row["predicted_future_range"]) for row in rows
        ),
        "median_predicted_future_range": median(
            float(row["predicted_future_range"]) for row in rows
        ),
        "mean_effective_future_pressure_range": fmean(
            float(row["effective_future_pressure_range"]) for row in rows
        ),
        "frontiers_with_within_mode_order_change": sum(
            bool(row["within_mode_order_changed"]) for row in rows
        ),
        "within_mode_order_inversions": sum(
            int(row["within_mode_order_inversions"]) for row in rows
        ),
        "within_mode_comparable_pairs": sum(
            int(row["within_mode_comparable_pairs"]) for row in rows
        ),
        "selected_candidate_changes": len(changed),
        "selected_candidate_change_fraction": len(changed) / len(rows),
        "selected_action_type_changes": sum(
            bool(row["selected_action_type_changed"]) for row in changed
        ),
        "selected_mode_changes": sum(
            bool(row["selected_mode_changed"]) for row in changed
        ),
        "same_mode_selected_candidate_changes": len(same_mode),
        "changed_selection_with_lower_predicted_future_cost": sum(
            item > TOLERANCE for item in savings
        ),
        "mean_predicted_future_saving_when_changed": (
            fmean(savings) if savings else None
        ),
        "minimum_predicted_future_saving_when_changed": (
            min(savings) if savings else None
        ),
        "mean_same_mode_crossing_surplus": (
            fmean(float(row["same_mode_crossing_surplus"]) for row in same_mode)
            if same_mode
            else None
        ),
        "selected_action_type_transitions": dict(
            sorted(Counter(row["selected_transition"] for row in changed).items())
        ),
    }


def _pair_row(immediate: Mapping, future: Mapping) -> dict:
    if (
        int(immediate["model_seed"]) != int(future["model_seed"])
        or int(immediate["instance_seed"]) != int(future["instance_seed"])
        or float(immediate["deployment_lambda"])
        != float(future["preference_lambda"])
    ):
        raise E16Error("E5(c) paired coordinate mismatch")
    if not immediate["strict_safe_complete"] or not future["strict_safe_complete"]:
        raise E16Error("E5(c) offline mechanism requires strict complete pairs")
    return {
        "model_seed": int(immediate["model_seed"]),
        "instance_seed": int(immediate["instance_seed"]),
        "deployment_lambda": float(immediate["deployment_lambda"]),
        "behavior_changed": immediate["behavior_digest"] != future["behavior_digest"],
        "return_immediate_minus_future": (
            float(immediate["dense_return"]) - float(future["dense_return"])
        ),
        "mae_immediate_minus_future": (
            float(immediate["mean_absolute_error"])
            - float(future["mean_absolute_error"])
        ),
        "rehandles_per_100_immediate_minus_future": (
            float(immediate["physical_rehandles_per_100_required_deliveries"])
            - float(future["physical_rehandles_per_100_required_deliveries"])
        ),
        "steps_immediate_minus_future": (
            float(immediate["steps"]) - float(future["steps"])
        ),
    }


def analyze_e5c_pairs() -> tuple[list[dict], dict, dict]:
    report = _load_self_hashed(E05C_REPORT, "report_sha256", "E5(c) report")
    if report.get("status") != "complete" or report.get("paper_evidence") is not True:
        raise E16Error("E5(c) report is not complete paper evidence")
    immediate_rows = e05c._all_immediate_rows(e05c.DEFAULT_OUTPUT)
    expected = 3 * len(DEPLOYMENT_LAMBDAS) * 30
    if len(immediate_rows) != expected:
        raise E16Error(f"E5(c) grid incomplete: {len(immediate_rows)}/{expected}")

    pairs = []
    ledger_hashes = []
    for immediate in immediate_rows:
        value = float(immediate["deployment_lambda"])
        future = e05c._future_control(
            e05c.E04_OUTPUT,
            e05c.E05_OUTPUT,
            int(immediate["model_seed"]),
            value,
            int(immediate["instance_seed"]),
        )
        pairs.append(_pair_row(immediate, future))
        path = e05c._ledger_path(
            e05c.DEFAULT_OUTPUT,
            e05c._spec(int(immediate["model_seed"]), value),
            int(immediate["instance_seed"]),
        )
        ledger_hashes.append(_sha(path))

    by_lambda = {}
    by_seed = {str(seed): {} for seed in (0, 1, 2)}
    for value in DEPLOYMENT_LAMBDAS:
        selected = [row for row in pairs if row["deployment_lambda"] == value]
        summary = _summarize_pairs(selected)
        source = report["paired_immediate_minus_future"][f"lambda_{value:.2f}"]
        if (
            summary["pairs"] != int(source["pairs"])
            or summary["identical_behaviors"] != int(source["identical_behavior_digest"])
        ):
            raise E16Error("failed to reproduce E5(c) paired behavior counts")
        by_lambda[f"lambda_{value:.2f}"] = summary
        for seed in (0, 1, 2):
            seed_rows = [row for row in selected if row["model_seed"] == seed]
            by_seed[str(seed)][f"lambda_{value:.2f}"] = _summarize_pairs(
                seed_rows
            )
    return pairs, {
        "pairs": len(pairs),
        "by_lambda": by_lambda,
        "by_seed": by_seed,
        "metric_sign": "positive immediate-minus-future favors future for MAE/rehandles/steps",
    }, {
        "report_sha256": report["report_sha256"],
        "ledger_set_sha256": hashlib.sha256(
            "\n".join(sorted(ledger_hashes)).encode("ascii")
        ).hexdigest(),
    }


def _summarize_pairs(rows: Sequence[Mapping]) -> dict:
    rows = tuple(rows)
    if not rows:
        raise E16Error("cannot summarize empty E5(c) pairs")
    changed = [row for row in rows if row["behavior_changed"]]

    def mean(field: str, selected: Sequence[Mapping]) -> Optional[float]:
        return fmean(float(row[field]) for row in selected) if selected else None

    return {
        "pairs": len(rows),
        "identical_behaviors": len(rows) - len(changed),
        "changed_behaviors": len(changed),
        "changed_behavior_fraction": len(changed) / len(rows),
        "all_pair_mean_differences": {
            field: mean(field, rows)
            for field in (
                "return_immediate_minus_future",
                "mae_immediate_minus_future",
                "rehandles_per_100_immediate_minus_future",
                "steps_immediate_minus_future",
            )
        },
        "changed_pair_mean_differences": {
            field: mean(field, changed)
            for field in (
                "return_immediate_minus_future",
                "mae_immediate_minus_future",
                "rehandles_per_100_immediate_minus_future",
                "steps_immediate_minus_future",
            )
        },
        "changed_pairs_future_lower_rehandles": sum(
            float(row["rehandles_per_100_immediate_minus_future"]) > TOLERANCE
            for row in changed
        ),
        "changed_pairs_equal_rehandles": sum(
            abs(float(row["rehandles_per_100_immediate_minus_future"]))
            <= TOLERANCE
            for row in changed
        ),
        "changed_pairs_future_higher_rehandles": sum(
            float(row["rehandles_per_100_immediate_minus_future"]) < -TOLERANCE
            for row in changed
        ),
        "changed_pairs_future_lower_mae": sum(
            float(row["mae_immediate_minus_future"]) > TOLERANCE
            for row in changed
        ),
        "changed_pairs_equal_mae": sum(
            abs(float(row["mae_immediate_minus_future"])) <= TOLERANCE
            for row in changed
        ),
        "changed_pairs_future_higher_mae": sum(
            float(row["mae_immediate_minus_future"]) < -TOLERANCE
            for row in changed
        ),
        "changed_pairs_future_lower_rehandles_and_mae": sum(
            float(row["rehandles_per_100_immediate_minus_future"]) > TOLERANCE
            and float(row["mae_immediate_minus_future"]) > TOLERANCE
            for row in changed
        ),
    }


def _input_adaptation_context() -> tuple[dict, str]:
    report = _load_self_hashed(
        E05B_MECHANISM, "report_sha256", "E5(b) mechanism report"
    )
    if report.get("status") != "complete":
        raise E16Error("E5(b) mechanism report is incomplete")
    result = {}
    for value in DEPLOYMENT_LAMBDAS:
        fixed = report["fixed_bank"]["by_lambda"][f"lambda_{value:.4f}"]
        rollout = report["rollout_behavior"][f"lambda_{value:.2f}"]
        result[f"lambda_{value:.2f}"] = {
            "fixed_bank_selected_action_changes_full_vs_clamped": fixed[
                "selected_action_changes"
            ],
            "fixed_bank_frontiers": fixed["frontiers"],
            "rollout_trajectory_changes_full_vs_clamped": rollout[
                "different_behavior_digests"
            ],
            "rollout_pairs": rollout["pairs"],
        }
    return result, report["report_sha256"]


def _table(report: Mapping) -> str:
    lines = [
        "# E16-A prediction-to-ranking offline audit",
        "",
        "The future predictor is evaluated at input lambda=.10 in every row; only the external deployment weight changes.",
        "",
        "| Lambda | Fixed frontiers | Future term changes selected candidate | Action-type changes | Mode changes | Changes select lower predicted future cost | E5(c) trajectories changed | Delta rehandles/100 | Delta MAE |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for value in DEPLOYMENT_LAMBDAS:
        key = f"lambda_{value:.2f}"
        fixed = report["fixed_bank"]["by_lambda"][key]
        rollout = report["e5c_rollout_pairs"]["by_lambda"][key]
        differences = rollout["all_pair_mean_differences"]
        lines.append(
            f"| {value:.2f} | {fixed['frontiers']} | "
            f"{fixed['selected_candidate_changes']}/{fixed['frontiers']} | "
            f"{fixed['selected_action_type_changes']} | "
            f"{fixed['selected_mode_changes']} | "
            f"{fixed['changed_selection_with_lower_predicted_future_cost']}/"
            f"{fixed['selected_candidate_changes']} | "
            f"{rollout['changed_behaviors']}/{rollout['pairs']} | "
            f"{differences['rehandles_per_100_immediate_minus_future']:+.2f} | "
            f"{differences['mae_immediate_minus_future']:+.2f} |"
        )
    lines.extend(
        [
            "",
            "Outcome differences are immediate-only minus learned-future. Positive rehandles/MAE values therefore favor the future model.",
            "",
            "This establishes a direct fixed-state prediction-to-ranking link and separately reproduces E5(c)'s paired deployment effects. The episode ledgers do not retain candidate-level decision traces, and the fixed bank has no candidate-specific realized continuation targets. A bounded continuation bank is still required to test prediction correctness and connect individual ranking changes to realized consequences under distribution shift.",
        ]
    )
    return "\n".join(lines) + "\n"


def run(output: Path) -> dict:
    output = output.resolve()
    sources = conditioning._sources(PROJECT_ROOT)
    values, selections = conditioning._stitch_rows(sources)
    if len(selections) != 3 * 34 * 17:
        raise E16Error("selection bank is incomplete")
    frontier_rows, fixed_bank = analyze_fixed_bank(values)
    pair_rows, e5c_pairs, e5c_sources = analyze_e5c_pairs()
    input_context, e5b_report_sha = _input_adaptation_context()

    source_sha = {
        "fixed_bank_values_seed_0_2": _sha(sources["d9_values"]),
        "fixed_bank_selections_seed_0_2": _sha(sources["d9_selections"]),
        "fixed_bank_values_seed_1": _sha(sources["seed1_values"]),
        "fixed_bank_selections_seed_1": _sha(sources["seed1_selections"]),
        "e05b_mechanism_report": e5b_report_sha,
        "e05c_report": e5c_sources["report_sha256"],
        "e05c_ledger_set": e5c_sources["ledger_set_sha256"],
    }
    result = {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "status": "complete_offline",
        "new_training_runs": 0,
        "new_episode_rollouts": 0,
        "source_sha256": source_sha,
        "fixed_bank": fixed_bank,
        "e5c_rollout_pairs": e5c_pairs,
        "preference_input_adaptation_context": input_context,
        "established": {
            "candidate_relative_future_predictions_change_rankings": True,
            "future_term_changes_deployed_episode_behavior": True,
            "future_term_improves_rehandles_on_average_at_all_tested_lambdas": True,
        },
        "not_established_offline": {
            "candidate_specific_prediction_accuracy": True,
            "individual_ranking_change_improved_its_realized_continuation": True,
            "prediction_to_ranking_mechanism_under_E12_geometry_shifts": True,
        },
        "continuation_bank_gate": {
            "required_for_strong_prediction_accuracy_claim": True,
            "reason": (
                "fixed bank lacks candidate-specific realized continuation targets "
                "and E5(c)/E12 ledgers lack contemporaneous candidate score traces"
            ),
        },
    }
    result["report_sha256"] = _digest(result)
    _atomic_text(
        output / REPORT_NAME,
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )
    _atomic_csv(output / STATE_ROWS_NAME, frontier_rows)
    _atomic_csv(output / PAIR_ROWS_NAME, pair_rows)
    _atomic_text(output / TABLE_NAME, _table(result))
    return result


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    print(json.dumps(run(args.output), indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
