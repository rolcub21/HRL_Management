#!/usr/bin/env python3
"""No-rollout guarded derivation for post-hoc VCG-Dense V1.2 selection.

Validation panel A fixes the relocation ranking.  The already-open development
panel B (EpisodeInstances 80000--80029) is then used only as a sequential
operational guard: accept the first available evaluated candidate with
strict/full completion, MAE cost no larger than two, and a positive relocation
saving relative to that seed's current selected-best reference.  No
environment, checkpoint, or baseline is executed by this script.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import fmean
from typing import Mapping, Optional, Sequence
import uuid

import numpy as np

from benchmark_viability_critic_priority import _sha256_file
from compare_vcg_dense_pareto import EVALUATION_SEEDS
from compare_vcg_dense_v1_2_posthoc import (
    ENHANCED_GA_METHOD,
    EXPECTED_ARTIFACT_SHA256,
    SOURCE_SELECTED_METHODS,
    SOURCE_SEED2_FINAL_METHOD,
    _aggregate_contrast,
    _json_safe,
    _paired,
    _parse_source_row,
    _summary,
)


PROTOCOL = "vcg_dense_v1_2_panel_b_guarded_selection_development_v1"
GUARD_RULE = (
    "validation_A_relocation_rank_then_panel_B_strict_full_"
    "mae_cost_le_2_positive_relocation_saving_sequential_accept_v2"
)
GUARD_MAE_MARGIN = 2.0
EXPECTED_GUARDED_EPISODES = {0: 475, 1: 500, 2: 500}
GUARDED_GROUP = "panel_b_guarded_posthoc_v1_2_diagnostic"
OUTPUT_FILENAME = "guarded-selection-report.json"

EXPECTED_INPUT_SHA256 = {
    "posthoc-protocol-manifest.json": (
        "93ddcf3e5512bb572b06c0ee5a96b3db1f0f24b845c760ddddce1bccf8fbfbfb"
    ),
    "posthoc-runs.csv": (
        "f9b21e0a6040e0846ef10b3b228e560d06a421975111a3226e4654b81d649448"
    ),
    "posthoc-report.json": (
        "237536bd6d88a3f18bfbdc300e9dad1845bdf726a16a72f956eaef519fc820c4"
    ),
    "posthoc-audit.json": (
        "eeb9dae50a2bc6017d42be6f867e6904d709de65c6e0592ff93adfa3cb90604b"
    ),
    "source/protocol-manifest.json": (
        "c3f528f1496d77db9690cf306d9e1769446b78663dac0e884dca3778412ce4be"
    ),
    "source/instance-manifest.json": (
        "67e41f804e4845151e5e8805b968c779e743e90ffd36a51fe8e45d865bdea7eb"
    ),
    "source/pareto-runs.csv": (
        "74d646269e11fa843e8faa8ff39d03444003b220fd386e0cea990b24e104ac8a"
    ),
    "source/pareto-audit.json": (
        "15abb5d3cf406730ff5473d0bca5e708db95f0d309272303bf3561ae155cc7de"
    ),
    "source/pareto-report.json": (
        "be5c0ec4671662b3b8ae4cd16e5a4a8e5bbb6e69e6293b8a1f43dd324eb723a3"
    ),
}


def _load_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.tmp-{uuid.uuid4().hex}"
    try:
        temporary.write_text(
            json.dumps(_json_safe(value), indent=2, sort_keys=True, allow_nan=False)
            + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _verify_hashes(posthoc_dir: Path, source_dir: Path) -> dict:
    paths = {
        "posthoc-protocol-manifest.json": posthoc_dir
        / "posthoc-protocol-manifest.json",
        "posthoc-runs.csv": posthoc_dir / "posthoc-runs.csv",
        "posthoc-report.json": posthoc_dir / "posthoc-report.json",
        "posthoc-audit.json": posthoc_dir / "posthoc-audit.json",
        "source/protocol-manifest.json": source_dir / "protocol-manifest.json",
        "source/instance-manifest.json": source_dir / "instance-manifest.json",
        "source/pareto-runs.csv": source_dir / "pareto-runs.csv",
        "source/pareto-audit.json": source_dir / "pareto-audit.json",
        "source/pareto-report.json": source_dir / "pareto-report.json",
    }
    observed = {}
    for name, path in paths.items():
        sha256 = _sha256_file(path)
        expected = EXPECTED_INPUT_SHA256[name]
        if sha256 != expected:
            raise ValueError(
                f"guarded derivation input hash mismatch for {name}: "
                f"{sha256} != {expected}"
            )
        observed[name] = {
            "path": str(path.resolve()),
            "sha256": sha256,
        }
    return observed


def _read_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        return [_parse_source_row(row) for row in csv.DictReader(handle)]


def _method_rows(rows: Sequence[Mapping], method_id: str) -> list[dict]:
    selected = [dict(row) for row in rows if row["method_id"] == method_id]
    if len(selected) != len(EVALUATION_SEEDS):
        raise ValueError(f"guarded source grid for {method_id} is incomplete")
    if len({row["instance_id"] for row in selected}) != len(EVALUATION_SEEDS):
        raise ValueError(f"guarded source grid for {method_id} has duplicates")
    return selected


def _strict_full(rows: Sequence[Mapping]) -> bool:
    return bool(
        len(rows) == len(EVALUATION_SEEDS)
        and all(
            float(row["strict_method_success"]) == 1.0
            and float(row["completion_rate"]) == 1.0
            and int(row["delivery_count"]) == 8
            and int(row["relocations"]) == int(row["obstructive_moves"])
            for row in rows
        )
    )


def _validation_rank(selection: Mapping) -> tuple[dict, ...]:
    feasible = [
        row for row in selection["selection_table"] if row.get("feasible") is True
    ]
    return tuple(
        sorted(
            feasible,
            key=lambda row: (
                float(row["relocations_per_100_deliveries"]),
                -float(row["mean_dense_rescored_return"]),
                float(row["mean_absolute_error"]),
                int(row["checkpoint_episode"]),
            ),
        )
    )


def _sequential_guard(
    *,
    model_seed: int,
    selection: Mapping,
    reference_rows: Sequence[Mapping],
    available: Mapping[int, Sequence[Mapping]],
    provenance: Mapping[int, Mapping],
) -> dict:
    """Apply the frozen panel-B guard without evaluating any policy."""

    if not _strict_full(reference_rows):
        raise ValueError(f"seed {model_seed} reference is not strict/full on panel B")
    reference_summary = _summary(reference_rows)
    reference_mae = float(reference_summary["mean_absolute_error"])
    reference_relocations = float(
        reference_summary["relocations_per_100_deliveries"]
    )
    rank = _validation_rank(selection)
    trials = []
    chosen_episode = None
    chosen_rows = None
    for rank_index, rank_record in enumerate(rank, start=1):
        episode = int(rank_record["checkpoint_episode"])
        rows = available.get(episode)
        if rows is None:
            trials.append(
                {
                    "validation_A_rank": rank_index,
                    "checkpoint_episode": episode,
                    "available_authenticated_evaluation": False,
                    "accepted": False,
                    "reason": "preserved_authenticated_evaluation_unavailable",
                }
            )
            continue
        strict_full = _strict_full(rows)
        candidate_summary = _summary(rows)
        mae_cost = float(candidate_summary["mean_absolute_error"]) - reference_mae
        candidate_relocations = float(
            candidate_summary["relocations_per_100_deliveries"]
        )
        relocation_saving = reference_relocations - candidate_relocations
        accepted = bool(
            strict_full
            and mae_cost <= GUARD_MAE_MARGIN
            and relocation_saving > 0.0
        )
        trials.append(
            {
                "validation_A_rank": rank_index,
                "checkpoint_episode": episode,
                "available_authenticated_evaluation": True,
                "provenance": provenance[episode],
                "strict_full_panel_B": strict_full,
                "panel_B_reference_mae": reference_mae,
                "panel_B_candidate_mae": candidate_summary["mean_absolute_error"],
                "panel_B_mae_cost": mae_cost,
                "panel_B_mae_margin": GUARD_MAE_MARGIN,
                "panel_B_reference_relocations_per_100_deliveries": (
                    reference_relocations
                ),
                "panel_B_candidate_relocations_per_100_deliveries": (
                    candidate_relocations
                ),
                "panel_B_relocation_saving_per_100_deliveries": (
                    relocation_saving
                ),
                "panel_B_positive_relocation_saving_required": True,
                "accepted": accepted,
                "reason": (
                    "first_available_ranked_candidate_passing_guard"
                    if accepted
                    else (
                        "panel_B_strict_full_or_mae_or_relocation_guard_failed"
                    )
                ),
            }
        )
        if accepted:
            chosen_episode = episode
            chosen_rows = tuple(dict(row) for row in rows)
            break
    if chosen_episode is None or chosen_rows is None:
        chosen_episode = int(selection["reference_episode"])
        chosen_rows = tuple(dict(row) for row in reference_rows)
        trials.append(
            {
                "validation_A_rank": None,
                "checkpoint_episode": chosen_episode,
                "available_authenticated_evaluation": True,
                "strict_full_panel_B": True,
                "panel_B_reference_mae": reference_mae,
                "panel_B_candidate_mae": reference_mae,
                "panel_B_mae_cost": 0.0,
                "panel_B_mae_margin": GUARD_MAE_MARGIN,
                "panel_B_reference_relocations_per_100_deliveries": (
                    reference_relocations
                ),
                "panel_B_candidate_relocations_per_100_deliveries": (
                    reference_relocations
                ),
                "panel_B_relocation_saving_per_100_deliveries": 0.0,
                "panel_B_positive_relocation_saving_required": True,
                "accepted": False,
                "selected_as_mandatory_reference_fallback": True,
                "reason": "no_nonreference_candidate_passed_mandatory_reference_fallback",
            }
        )
    expected = EXPECTED_GUARDED_EPISODES[model_seed]
    if chosen_episode != expected:
        raise ValueError(
            f"guarded choice drifted for seed {model_seed}: "
            f"{chosen_episode} != {expected}"
        )
    return {
        "model_seed": model_seed,
        "guard_rule": GUARD_RULE,
        "reference_episode": int(selection["reference_episode"]),
        "chosen_episode": chosen_episode,
        "validation_A_relocation_rank": rank,
        "sequential_panel_B_trials": tuple(trials),
        "reference_summary_panel_B": reference_summary,
        "chosen_summary_panel_B": _summary(chosen_rows),
        "chosen_rows": chosen_rows,
    }


def _relabel_guarded(rows: Sequence[Mapping], model_seed: int, episode: int) -> list[dict]:
    method_id = f"vcg_dense_seed{model_seed}_guarded_v1_2_ep{episode}"
    output = []
    for row in rows:
        item = dict(row)
        item.update(
            {
                "protocol": PROTOCOL,
                "method": method_id,
                "method_id": method_id,
                "policy_group": GUARDED_GROUP,
                "model_seed": model_seed,
                "guarded_selected_episode": episode,
                "panel_B_selected": True,
                "posthoc": True,
                "development_only": True,
                "diagnostic_only": True,
                "deployment_checkpoint_eligible": False,
                "performance_claim_authorized": False,
            }
        )
        output.append(item)
    return output


def _aggregate_metric_summaries(per_seed: Sequence[Mapping]) -> dict:
    fields = (
        "mean_absolute_error",
        "relocations_per_100_deliveries",
        "mean_dense_rescored_return",
        "first_two_mean_absolute_error",
        "positions_three_plus_mean_absolute_error",
        "mean_tardiness",
        "mean_earliness",
        "within_target_window_rate",
        "mean_steps",
    )
    output = {}
    for field in fields:
        values = np.asarray(
            [float(item["guarded_summary"][field]) for item in per_seed],
            dtype=float,
        )
        mean = float(values.mean())
        sample_sd = float(values.std(ddof=1))
        half = 4.302652729911275 * sample_sd / math.sqrt(3.0)
        output[field] = {
            "per_training_seed": tuple(float(value) for value in values),
            "mean": mean,
            "sample_sd": sample_sd,
            "degrees_of_freedom": 2,
            "training_seed_t95_ci": [mean - half, mean + half],
        }
    return output


def _build_report(
    guarded: Sequence[Mapping],
    source_rows: Mapping[str, Sequence[Mapping]],
    input_hashes: Mapping,
    provenance: Mapping,
) -> dict:
    guarded_rows = []
    per_seed = []
    for result in guarded:
        seed = int(result["model_seed"])
        episode = int(result["chosen_episode"])
        rows = _relabel_guarded(result["chosen_rows"], seed, episode)
        guarded_rows.extend(rows)
        reference = source_rows[SOURCE_SELECTED_METHODS[seed]]
        enhanced = source_rows[ENHANCED_GA_METHOD]
        per_seed.append(
            {
                "model_seed": seed,
                "reference_episode": result["reference_episode"],
                "guarded_episode": episode,
                "guarded_summary": _summary(rows),
                "reference_summary": _summary(reference),
                "enhanced_ga_summary": _summary(enhanced),
                "guarded_vs_reference": _paired(rows, reference, "selected_best"),
                "guarded_vs_enhanced_ga": _paired(
                    rows, enhanced, ENHANCED_GA_METHOD
                ),
                "sequential_guard": {
                    key: value
                    for key, value in result.items()
                    if key != "chosen_rows"
                },
            }
        )
    if len(guarded_rows) != 3 * len(EVALUATION_SEEDS):
        raise ValueError("guarded analysis grid is incomplete")
    return {
        "protocol": PROTOCOL,
        "scope": "panel_B_selected_posthoc_development_diagnostic_only",
        "guard_rule": GUARD_RULE,
        "guard_mae_margin": GUARD_MAE_MARGIN,
        "performance_claim_authorized": False,
        "deployment_checkpoint_eligible": False,
        "new_policy_executions": 0,
        "new_baseline_executions": 0,
        "sealed_panels_opened": False,
        "input_hashes": input_hashes,
        "candidate_provenance": provenance,
        "per_training_seed": tuple(per_seed),
        "aggregate_metric_summaries": _aggregate_metric_summaries(per_seed),
        "aggregate_contrasts": {
            "guarded_vs_selected_best": _aggregate_contrast(
                guarded_rows, source_rows, comparator="selected_best"
            ),
            "guarded_vs_enhanced_ga": _aggregate_contrast(
                guarded_rows, source_rows, comparator="enhanced_ga"
            ),
        },
        "guardrails": {
            "candidate_rank_frozen_on_validation_A": True,
            "guard_panel": "already_opened_development_80000_80029",
            "guard_panel_used_for_selection": True,
            "panel_B_selected": True,
            "positive_panel_B_relocation_saving_required": True,
            "reference_is_mandatory_last_fallback": True,
            "not_an_unbiased_test_panel": True,
            "no_rollout_code_path": True,
            "no_claim": True,
        },
        "interpretation": (
            "The result is a panel-B-selected development diagnostic.  It can "
            "motivate a prospectively frozen training/selection protocol, but "
            "cannot support an unbiased performance or deployment claim."
        ),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--posthoc-dir", type=Path, required=True)
    parser.add_argument("--source-pareto-dir", type=Path, required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> dict:
    args = _build_parser().parse_args(argv)
    posthoc_dir = args.posthoc_dir.resolve()
    source_dir = args.source_pareto_dir.resolve()
    output_path = posthoc_dir / OUTPUT_FILENAME
    if output_path.exists():
        raise FileExistsError(
            f"{output_path} already exists; guarded derivation is immutable"
        )
    input_hashes = _verify_hashes(posthoc_dir, source_dir)
    posthoc_report = _load_json(posthoc_dir / "posthoc-report.json")
    posthoc_protocol = _load_json(posthoc_dir / "posthoc-protocol-manifest.json")
    if posthoc_report.get("performance_claim_authorized") is not False:
        raise ValueError("posthoc source unexpectedly authorizes a claim")
    if posthoc_protocol.get("deployment_checkpoint_eligible") is not False:
        raise ValueError("posthoc source unexpectedly marks deployment eligibility")

    source_rows_all = _read_rows(source_dir / "pareto-runs.csv")
    posthoc_rows_all = _read_rows(posthoc_dir / "posthoc-runs.csv")
    source_rows = {
        **{
            method: _method_rows(source_rows_all, method)
            for method in SOURCE_SELECTED_METHODS.values()
        },
        ENHANCED_GA_METHOD: _method_rows(source_rows_all, ENHANCED_GA_METHOD),
        "vcg_dense_seed1_episode500_final": _method_rows(
            source_rows_all, "vcg_dense_seed1_episode500_final"
        ),
        SOURCE_SEED2_FINAL_METHOD: _method_rows(
            source_rows_all, SOURCE_SEED2_FINAL_METHOD
        ),
    }
    posthoc_by_seed = {
        seed: [row for row in posthoc_rows_all if row["model_seed"] == seed]
        for seed in (0, 1, 2)
    }
    for seed, rows in posthoc_by_seed.items():
        if len(rows) != len(EVALUATION_SEEDS) or not _strict_full(rows):
            raise ValueError(f"posthoc seed-{seed} row grid is not strict/full")

    selections = {
        int(item["model_seed"]): item
        for item in posthoc_report["selection_decisions"]
    }
    if set(selections) != {0, 1, 2}:
        raise ValueError("posthoc selection decisions are incomplete")
    artifact_manifest = {
        int(item["model_seed"]): item
        for item in posthoc_report["artifact_manifest"]
    }
    for seed, expected_sha in EXPECTED_ARTIFACT_SHA256.items():
        if artifact_manifest[seed]["checkpoint_sha256"] != expected_sha:
            raise ValueError(f"posthoc artifact identity drifted for seed {seed}")

    available = {
        0: {475: posthoc_by_seed[0]},
        1: {
            425: posthoc_by_seed[1],
            500: source_rows["vcg_dense_seed1_episode500_final"],
        },
        2: {500: posthoc_by_seed[2]},
    }
    candidate_provenance = {
        0: {
            475: {
                "source": "posthoc-runs.csv",
                "checkpoint_sha256": artifact_manifest[0]["checkpoint_sha256"],
                "deployment_policy_digest": artifact_manifest[0][
                    "deployment_policy_digest"
                ],
            }
        },
        1: {
            425: {
                "source": "posthoc-runs.csv",
                "checkpoint_sha256": artifact_manifest[1]["checkpoint_sha256"],
                "deployment_policy_digest": artifact_manifest[1][
                    "deployment_policy_digest"
                ],
            },
            500: {
                "source": "source/pareto-runs.csv",
                "source_method_id": "vcg_dense_seed1_episode500_final",
                "checkpoint_sha256": source_rows[
                    "vcg_dense_seed1_episode500_final"
                ][0]["checkpoint_sha256"],
                "deployment_policy_digest": source_rows[
                    "vcg_dense_seed1_episode500_final"
                ][0]["deployment_policy_digest"],
            },
        },
        2: {
            500: {
                "source": "posthoc-runs.csv",
                "checkpoint_sha256": artifact_manifest[2]["checkpoint_sha256"],
                "deployment_policy_digest": artifact_manifest[2][
                    "deployment_policy_digest"
                ],
            }
        },
    }

    guarded = []
    for seed in (0, 1, 2):
        guarded.append(
            _sequential_guard(
                model_seed=seed,
                selection=selections[seed],
                reference_rows=source_rows[SOURCE_SELECTED_METHODS[seed]],
                available=available[seed],
                provenance=candidate_provenance[seed],
            )
        )
    report = _build_report(
        guarded,
        source_rows,
        input_hashes,
        candidate_provenance,
    )
    _atomic_json(output_path, report)
    print(
        json.dumps(
            {
                "guarded_choices": {
                    str(item["model_seed"]): item["chosen_episode"]
                    for item in guarded
                },
                "new_executions": 0,
                "report": str(output_path),
            },
            indent=2,
        ),
        flush=True,
    )
    return report


if __name__ == "__main__":
    main()
