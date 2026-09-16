#!/usr/bin/env python3
"""E16-C: decompose the learned future signal at E16-B crossings.

This analysis deterministically replays only the prefix leading to each saved
E16-B state so that the complete candidate-score vector can be reconstructed.
It launches no training and no new counterfactual continuation rollouts.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
from statistics import fmean
import sys
from typing import Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from experiments.conditioned_vcg.E16_prediction_ranking_audit import (
    continuation_bank as e16b,
)
from viability_graph_hierarchy import ID_TO_MODE
from viability_graph_preference_conditioned import select_hierarchical_index


PROTOCOL = "vcg_conditioned_e16_signal_component_decomposition_v1"
SCHEMA_VERSION = 1
REPORT_NAME = "e16c-component-decomposition-report.json"
TABLE_NAME = "e16c-component-decomposition-summary.md"


class E16CError(RuntimeError):
    pass


def _load_parent(output: Path) -> tuple[dict, dict, dict]:
    contract, manifest = e16b.authenticate(output)
    report = e16b._load_json(output / e16b.REPORT_NAME, "E16-B report")
    e16b._verify_hash(report, "report_sha256", "E16-B report")
    if (
        report.get("status") != "complete"
        or report.get("paper_evidence") is not True
        or report.get("contract_sha256") != contract["contract_sha256"]
        or report.get("manifest_sha256") != manifest["manifest_sha256"]
    ):
        raise E16CError("E16-C requires the complete authenticated E16-B report")
    return contract, manifest, report


def _selection(agent, rows: Sequence[Mapping], merits: Sequence[float]) -> dict:
    result = select_hierarchical_index(
        merits,
        [int(record["mode_id"]) for record in rows],
        agent.base_agent.within_temperatures,
        candidate_keys=[record["candidate_key"] for record in rows],
    )
    row = rows[int(result.selected_index)]
    return {
        "candidate_key": row["candidate_key"],
        "action_type": row["action_type"],
        "mode": row["mode"],
        "selected_index": int(result.selected_index),
        "selected_mode": ID_TO_MODE[int(result.selected_mode_id)],
    }


def _decompose(agent, score_rows: Sequence[Mapping]) -> dict:
    rows = [dict(row) for row in score_rows]
    if not rows:
        raise E16CError("cannot decompose an empty frontier")
    predicted = [float(row["predicted_future_rehandles"]) for row in rows]
    state_mean = fmean(predicted)
    by_type = defaultdict(list)
    for row, value in zip(rows, predicted):
        by_type[str(row["action_type"])].append(value)
    type_means = {name: fmean(values) for name, values in by_type.items()}

    total_ss = sum((value - state_mean) ** 2 for value in predicted)
    between_ss = sum(
        len(values) * (type_means[name] - state_mean) ** 2
        for name, values in by_type.items()
    )
    within_ss = sum(
        (float(row["predicted_future_rehandles"]) - type_means[row["action_type"]])
        ** 2
        for row in rows
    )
    if abs(total_ss - between_ss - within_ss) > 1.0e-8:
        raise E16CError("future-signal sum-of-squares decomposition failed")

    base = []
    common = []
    type_only = []
    residual_only = []
    reconstructed = []
    enriched = []
    for row in rows:
        action_type = str(row["action_type"])
        prediction = float(row["predicted_future_rehandles"])
        beta = type_means[action_type] - state_mean
        residual = prediction - type_means[action_type]
        immediate_merit = float(row["immediate_only_merit"])
        base.append(immediate_merit)
        common.append(immediate_merit - e16b.DEPLOYMENT_LAMBDA * state_mean)
        type_only.append(
            immediate_merit
            - e16b.DEPLOYMENT_LAMBDA * (state_mean + beta)
        )
        residual_only.append(
            immediate_merit
            - e16b.DEPLOYMENT_LAMBDA * (state_mean + residual)
        )
        reconstructed.append(
            immediate_merit
            - e16b.DEPLOYMENT_LAMBDA * (state_mean + beta + residual)
        )
        enriched.append(
            {
                **row,
                "state_common_prediction": state_mean,
                "action_type_effect": beta,
                "candidate_specific_residual": residual,
            }
        )
    maximum_error = max(
        abs(value - float(row["fixed_future_merit"]))
        for value, row in zip(reconstructed, rows)
    )
    if maximum_error > 1.0e-6:
        raise E16CError("decomposed merits do not reconstruct the full merit")

    selections = {
        "immediate_only": _selection(agent, enriched, base),
        "common_only": _selection(agent, enriched, common),
        "action_type_only": _selection(agent, enriched, type_only),
        "candidate_residual_only": _selection(agent, enriched, residual_only),
        "full_reconstructed": _selection(agent, enriched, reconstructed),
    }
    if selections["common_only"] != selections["immediate_only"]:
        raise E16CError("common candidate offset changed a shift-invariant selector")
    return {
        "candidate_count": len(rows),
        "action_type_counts": {
            name: len(values) for name, values in sorted(by_type.items())
        },
        "state_common_prediction": state_mean,
        "action_type_mean_prediction": dict(sorted(type_means.items())),
        "action_type_effect": {
            name: value - state_mean for name, value in sorted(type_means.items())
        },
        "total_candidate_variation_ss": total_ss,
        "between_action_type_ss": between_ss,
        "within_action_type_residual_ss": within_ss,
        "between_action_type_variation_fraction": (
            between_ss / total_ss if total_ss > 0.0 else None
        ),
        "within_action_type_variation_fraction": (
            within_ss / total_ss if total_ss > 0.0 else None
        ),
        "maximum_merit_reconstruction_error": maximum_error,
        "selections": selections,
        "candidate_scores": enriched,
    }


def _validate_selected_scores(observed: Mapping, expected: Mapping) -> None:
    exact = ("candidate_key", "mode", "action_type", "immediate_rehandle")
    numeric = (
        "qop",
        "predicted_future_rehandles",
        "immediate_only_merit",
        "fixed_future_merit",
    )
    if any(observed[name] != expected[name] for name in exact):
        raise E16CError("reconstructed selected-candidate identity changed")
    if any(abs(float(observed[name]) - float(expected[name])) > 1.0e-6 for name in numeric):
        raise E16CError("reconstructed selected-candidate score changed")


def _reconstruct_case(
    output: Path,
    manifest: Mapping,
    case: Mapping,
    *,
    device: torch.device,
    arms,
) -> dict:
    regime_id = str(case["regime_id"])
    model_seed = int(case["model_seed"])
    instance = e16b._instance(
        output, manifest, regime_id, int(case["instance_seed"])
    )
    env = e16b.e11.REGIME_BY_ID[regime_id].make_env()
    env.current_episode = 1
    env.reset(instance=instance)
    agent, arm = e16b._new_agent(model_seed, device, arms=arms)
    cache = e16b.reuse.TimingInvariantCertificateCache()
    consecutive_defer = 0
    pending = None
    steps = 0

    for decision_index in range(int(case["decision_index"]) + 1):
        if pending is None:
            pending, _frontier = e16b._enumerate(
                env, consecutive_defer, arm, cache
            )
        snapshot = pending
        pending = None
        scores = e16b._score_frontier(agent, snapshot)
        if decision_index == int(case["decision_index"]):
            if (
                e16b._state_digest(env) != case["state_digest"]
                or steps != int(case["primitive_steps_before_capture"])
                or scores["liveness_forced"]
            ):
                raise E16CError("reconstructed capture state does not match E16-B")
            immediate = scores["rows"][scores["immediate_index"]]
            future = scores["rows"][scores["future_index"]]
            _validate_selected_scores(
                immediate,
                case["selected_candidates"]["immediate_only_winner"],
            )
            _validate_selected_scores(
                future,
                case["selected_candidates"]["fixed_future_winner"],
            )
            records = []
            for row, record in zip(scores["rows"], scores["prepared"].records):
                records.append({**row, "mode_id": int(record.mode_id)})
            result = _decompose(agent, records)
            saved_immediate = case["selected_candidates"]["immediate_only_winner"][
                "candidate_key"
            ]
            saved_future = case["selected_candidates"]["fixed_future_winner"][
                "candidate_key"
            ]
            if (
                result["selections"]["immediate_only"]["candidate_key"]
                != saved_immediate
                or result["selections"]["full_reconstructed"]["candidate_key"]
                != saved_future
            ):
                raise E16CError("component selector does not reproduce E16-B crossing")
            return result

        decision = agent.select(
            snapshot,
            preference_lambda=e16b.DEPLOYMENT_LAMBDA,
            training=False,
            epsilon=0.0,
        )
        execution = e16b.execute_certified_macro(
            env,
            decision.candidate,
            gamma=agent.config.gamma,
            remaining_steps=e16b.MAX_STEPS_AFTER_CAPTURE - steps,
            evaluation=True,
        )
        if (
            execution.env_terminal
            or execution.truncated
            or not execution.option_success
            or execution.duration == 0
        ):
            raise E16CError("prefix replay ended before its captured crossing")
        steps += int(execution.duration)
        consecutive_defer = e16b._next_defer_count(
            consecutive_defer, decision, execution
        )
        pending, _frontier = e16b._enumerate(
            env, consecutive_defer, arm, cache
        )
        if not pending.candidates:
            raise E16CError("prefix replay lost its certified frontier")
        agent.observe_outcome(decision, next_snapshot=pending, done=False)
    raise AssertionError("capture loop did not return")


def _timing_values(branch: Mapping) -> list[float]:
    return [
        float(value)
        for decision in branch["decision_trace"]
        for value in decision["delivery_deviations"]
    ]


def _timing_summary(values: Sequence[float]) -> dict:
    values = tuple(float(value) for value in values)
    if not values:
        raise E16CError("complete branch has no delivery deviations")
    return {
        "deliveries": len(values),
        "mean_signed_error": fmean(values),
        "mean_earliness": fmean(max(-value, 0.0) for value in values),
        "mean_tardiness": fmean(max(value, 0.0) for value in values),
        "mean_absolute_error": fmean(abs(value) for value in values),
        "early_deliveries": sum(value < 0.0 for value in values),
        "on_target_deliveries": sum(value == 0.0 for value in values),
        "late_deliveries": sum(value > 0.0 for value in values),
    }


def _timing_pair(case: Mapping) -> dict:
    immediate = _timing_summary(
        _timing_values(case["branches"]["immediate_only_winner"])
    )
    future = _timing_summary(
        _timing_values(case["branches"]["fixed_future_winner"])
    )
    if immediate["deliveries"] != future["deliveries"]:
        raise E16CError("paired continuations have different delivery counts")
    delta_earliness = future["mean_earliness"] - immediate["mean_earliness"]
    delta_tardiness = future["mean_tardiness"] - immediate["mean_tardiness"]
    delta_mae = future["mean_absolute_error"] - immediate["mean_absolute_error"]
    if abs(delta_mae - delta_earliness - delta_tardiness) > 1.0e-9:
        raise E16CError("signed timing decomposition does not reconstruct MAE")
    return {
        "immediate_only": immediate,
        "fixed_future": future,
        "fixed_future_minus_immediate": {
            "mean_signed_error": (
                future["mean_signed_error"] - immediate["mean_signed_error"]
            ),
            "mean_earliness": delta_earliness,
            "mean_tardiness": delta_tardiness,
            "mean_absolute_error": delta_mae,
        },
    }


def _mechanism_label(decomposition: Mapping) -> str:
    selections = decomposition["selections"]
    full = selections["full_reconstructed"]["candidate_key"]
    type_match = selections["action_type_only"]["candidate_key"] == full
    residual_match = (
        selections["candidate_residual_only"]["candidate_key"] == full
    )
    if type_match and residual_match:
        return "both_components_individually_reproduce_full"
    if type_match:
        return "action_type_component_reproduces_full"
    if residual_match:
        return "candidate_residual_reproduces_full"
    return "both_components_or_selector_interaction_required"


def _aggregate(records: Sequence[Mapping]) -> dict:
    count = len(records)
    labels = defaultdict(int)
    for record in records:
        labels[record["mechanism_label"]] += 1
    exact = lambda component: sum(
        record["decomposition"]["selections"][component]["candidate_key"]
        == record["decomposition"]["selections"]["full_reconstructed"][
            "candidate_key"
        ]
        for record in records
    )
    action = lambda component: sum(
        record["decomposition"]["selections"][component]["action_type"]
        == record["decomposition"]["selections"]["full_reconstructed"][
            "action_type"
        ]
        for record in records
    )
    timing = [record["timing"]["fixed_future_minus_immediate"] for record in records]
    action_changed = [
        record
        for record in records
        if record["decomposition"]["selections"]["immediate_only"]["action_type"]
        != record["decomposition"]["selections"]["full_reconstructed"][
            "action_type"
        ]
    ]
    same_action = [record for record in records if record not in action_changed]

    def subset_matches(selected, component, field="candidate_key"):
        return sum(
            record["decomposition"]["selections"][component][field]
            == record["decomposition"]["selections"]["full_reconstructed"][field]
            for record in selected
        )

    total_ss = sum(
        record["decomposition"]["total_candidate_variation_ss"]
        for record in records
    )
    between_ss = sum(
        record["decomposition"]["between_action_type_ss"] for record in records
    )
    within_ss = sum(
        record["decomposition"]["within_action_type_residual_ss"]
        for record in records
    )
    return {
        "cases": count,
        "mechanism_labels": dict(sorted(labels.items())),
        "action_type_only_exact_choice_matches_full": exact("action_type_only"),
        "candidate_residual_only_exact_choice_matches_full": exact(
            "candidate_residual_only"
        ),
        "action_type_only_action_matches_full": action("action_type_only"),
        "candidate_residual_only_action_matches_full": action(
            "candidate_residual_only"
        ),
        "full_choice_changed_action_type": len(action_changed),
        "full_choice_changed_within_action_type": len(same_action),
        "among_action_type_changes": {
            "action_type_only_action_matches_full": subset_matches(
                action_changed, "action_type_only", "action_type"
            ),
            "candidate_residual_only_action_matches_full": subset_matches(
                action_changed, "candidate_residual_only", "action_type"
            ),
            "action_type_only_exact_choice_matches_full": subset_matches(
                action_changed, "action_type_only"
            ),
            "candidate_residual_only_exact_choice_matches_full": subset_matches(
                action_changed, "candidate_residual_only"
            ),
        },
        "among_within_action_type_changes": {
            "action_type_only_exact_choice_matches_full": subset_matches(
                same_action, "action_type_only"
            ),
            "candidate_residual_only_exact_choice_matches_full": subset_matches(
                same_action, "candidate_residual_only"
            ),
        },
        "pooled_between_action_type_variation_fraction": (
            between_ss / total_ss if total_ss > 0.0 else None
        ),
        "pooled_within_action_type_residual_variation_fraction": (
            within_ss / total_ss if total_ss > 0.0 else None
        ),
        "mean_fixed_future_minus_immediate_earliness": (
            fmean(item["mean_earliness"] for item in timing) if timing else None
        ),
        "mean_fixed_future_minus_immediate_tardiness": (
            fmean(item["mean_tardiness"] for item in timing) if timing else None
        ),
        "mean_fixed_future_minus_immediate_mae": (
            fmean(item["mean_absolute_error"] for item in timing) if timing else None
        ),
        "cases_with_lower_earliness": sum(
            item["mean_earliness"] < 0.0 for item in timing
        ),
        "cases_with_lower_tardiness": sum(
            item["mean_tardiness"] < 0.0 for item in timing
        ),
        "cases_with_lower_mae": sum(
            item["mean_absolute_error"] < 0.0 for item in timing
        ),
    }


def _table(report: Mapping) -> str:
    lines = [
        "# E16-C learned-signal component decomposition",
        "",
        "| Regime | Seed | Full transition | Type-only choice | Residual-only choice | Mechanism | Delta early | Delta tardy | Delta MAE |",
        "|---|---:|---|---|---|---|---:|---:|---:|",
    ]
    for record in report["records"]:
        selections = record["decomposition"]["selections"]
        immediate = selections["immediate_only"]
        full = selections["full_reconstructed"]
        type_only = selections["action_type_only"]
        residual = selections["candidate_residual_only"]
        timing = record["timing"]["fixed_future_minus_immediate"]
        lines.append(
            f"| {record['regime_id']} | {record['model_seed']} | "
            f"{immediate['action_type']} -> {full['action_type']} | "
            f"{type_only['action_type']} `{type_only['candidate_key']}` | "
            f"{residual['action_type']} `{residual['candidate_key']}` | "
            f"{record['mechanism_label']} | "
            f"{timing['mean_earliness']:+.3f} | "
            f"{timing['mean_tardiness']:+.3f} | "
            f"{timing['mean_absolute_error']:+.3f} |"
        )
    lines.extend(
        [
            "",
            "All selector replays retain the exact hierarchical mode aggregation. A component 'reproduces full' only when it selects the exact same candidate key.",
            "Timing deltas are fixed-future branch minus immediate-only branch. Negative earliness or tardiness means that component of absolute error improved.",
        ]
    )
    return "\n".join(lines) + "\n"


def analyze(output: Path, *, device_name: str) -> dict:
    output = output.resolve()
    contract, manifest, parent = _load_parent(output)
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise E16CError("CUDA requested but unavailable")
    arms = e16b._frozen_arms()
    records = []
    with e16b.reuse.path_cleanup_active():
        for index, case in enumerate(parent["cases"], 1):
            decomposition = _reconstruct_case(
                output,
                manifest,
                case,
                device=device,
                arms=arms,
            )
            record = {
                "regime_id": case["regime_id"],
                "model_seed": int(case["model_seed"]),
                "instance_seed": int(case["instance_seed"]),
                "decision_index": int(case["decision_index"]),
                "state_digest": case["state_digest"],
                "decomposition": decomposition,
                "mechanism_label": _mechanism_label(decomposition),
                "timing": _timing_pair(case),
            }
            records.append(record)
            print(
                f"E16-C {index}/{len(parent['cases'])} | "
                f"{case['regime_id']} seed={case['model_seed']} | "
                f"{record['mechanism_label']}",
                flush=True,
            )
    report = e16b._with_hash(
        {
            "schema_version": SCHEMA_VERSION,
            "protocol": PROTOCOL,
            "status": "complete",
            "paper_evidence": True,
            "e16b_contract_sha256": contract["contract_sha256"],
            "e16b_manifest_sha256": manifest["manifest_sha256"],
            "e16b_report_sha256": parent["report_sha256"],
            "analysis_source_sha256": e16b._sha(Path(__file__).resolve()),
            "training_runs": 0,
            "new_counterfactual_continuation_rollouts": 0,
            "deterministic_prefix_replays": len(records),
            "preference_and_target_alignment": {
                "deployment_lambda": e16b.DEPLOYMENT_LAMBDA,
                "predictor_input_lambda": e16b.PREDICTOR_INPUT_LAMBDA,
                "continuation_policy": "same_fixed_future_policy_in_both_branches",
                "first_decision_guard_forced": False,
                "prediction_horizon": "after_current_macro_until_episode_completion",
                "prediction_discounting": "undiscounted",
                "realization_counting": "physical_rehandles_after_forced_macro",
                "exact_immediate_rehandle_excluded_from_future_target": True,
                "single_deterministic_realization_per_frozen_EpisodeInstance": True,
            },
            "records": records,
            "aggregate": _aggregate(records),
            "claim_boundary": (
                "selector_reproduction_and_signed_timing_decomposition_at_"
                "ten_predeclared_E16B_crossings"
            ),
        },
        "report_sha256",
    )
    e16b._atomic_json(output / REPORT_NAME, report)
    e16b._atomic_text(output / TABLE_NAME, _table(report))
    return {
        "status": "complete",
        "states": len(records),
        "training_runs": 0,
        "new_counterfactual_continuation_rollouts": 0,
        "report": str(output / REPORT_NAME),
        "table": str(output / TABLE_NAME),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=e16b.DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    print(
        json.dumps(
            analyze(args.output, device_name=args.device),
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
