#!/usr/bin/env python3
"""Evaluate the trained Kim2020 adaptation on the exact final 86k panel.

This is an additive, post-hoc comparison.  It reuses the three already
trained Kim checkpoints and the 30 serialized EpisodeInstances from the
four-method final panel.  No method is retrained and no baseline is rerun.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
from statistics import fmean
from typing import Any, Mapping, Optional, Sequence

import torch

import run_vcg_final86_four_method as final86
import vcg_v2_3_kim2020_supplement_evaluation as kim85


PROTOCOL = "kim2020_final86_matched_supplement_v1"
SCHEMA_VERSION = 1
MODEL_SEEDS = (0, 1, 2)
PANEL_SEEDS = tuple(range(86_000, 86_030))
ROLLOUTS = tuple(range(5))
POLICY_SEED_BASE = 633_000_000
EXPECTED_ROWS = 450

PROJECT_ROOT = Path(__file__).resolve().parent
TRAINING_ROOT = PROJECT_ROOT / "results/vcg-v2-3-kim2020-supplement-85k"
KIM85_ROOT = PROJECT_ROOT / "results/vcg-v2-3-kim2020-supplement-evaluation-85k"
FINAL86_ROOT = PROJECT_ROOT / "results/vcg-final86-four-method-confirmation"
CORRECTED_REPORT = (
    PROJECT_ROOT
    / "results/vcg-final86-baseline-correction/corrected-final-report.json"
)
OUTPUT_ROOT = PROJECT_ROOT / "results/kim2020-final86-comparison"

METRICS = (
    "dense_return",
    "mean_absolute_error",
    "mean_signed_deviation",
    "mean_tardiness",
    "mean_earliness",
    "within_target_window_rate",
    "steps",
    "physical_rehandles_per_100_required_deliveries",
)


class KimFinal86Error(ValueError):
    pass


def policy_seed(model_seed: int, panel_index: int, rollout: int) -> int:
    if model_seed not in MODEL_SEEDS:
        raise KimFinal86Error("invalid Kim model seed")
    if not 0 <= panel_index < len(PANEL_SEEDS):
        raise KimFinal86Error("invalid final-panel index")
    if rollout not in ROLLOUTS:
        raise KimFinal86Error("invalid rollout index")
    return POLICY_SEED_BASE + 1000 * model_seed + 10 * panel_index + rollout


def expected_grid() -> list[dict[str, int]]:
    rows = [
        {
            "model_seed": model_seed,
            "panel_index": panel_index,
            "instance_seed": instance_seed,
            "rollout_index": rollout,
            "policy_rng_seed": policy_seed(model_seed, panel_index, rollout),
        }
        for model_seed in MODEL_SEEDS
        for panel_index, instance_seed in enumerate(PANEL_SEEDS)
        for rollout in ROLLOUTS
    ]
    if len(rows) != EXPECTED_ROWS or len({row["policy_rng_seed"] for row in rows}) != EXPECTED_ROWS:
        raise KimFinal86Error("Kim final-grid cardinality failure")
    if any(622_000_000 <= row["policy_rng_seed"] < 623_000_000 for row in rows):
        raise KimFinal86Error("Kim policy RNG overlaps the VCG final-policy namespace")
    return rows


def _instance_record(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "instance_seed": int(record["seed"]),
        "episode_instance_id": record["episode_instance_id"],
        "schedule_id": record["schedule_id"],
        "episode_instance_sha256": record["canonical_sha256"],
        "saved_file_raw_sha256": record["raw_sha256"],
    }


def _inputs() -> tuple[dict, dict[int, dict], dict[int, dict]]:
    # Read/authenticate the serialized panel without requiring the current
    # process to expose the same CUDA device used by the earlier learned
    # evaluations. Kim itself runs on CPU; GPU availability is irrelevant to
    # the immutable EpisodeInstance bytes.
    manifest = final86._load_json(
        FINAL86_ROOT / "final-instance-manifest.json",
        name="final86 instance manifest",
    )
    final86._self_hash(manifest, "manifest_sha256", name="final86 instance manifest")
    if manifest.get("panel_opened") is not True or manifest.get("instance_count") != 30:
        raise KimFinal86Error("final86 instance manifest is incomplete")
    manifest_records = manifest.get("instances")
    if not isinstance(manifest_records, list) or len(manifest_records) != 30:
        raise KimFinal86Error("final86 instance records are incomplete")
    for expected_seed, record in zip(PANEL_SEEDS, manifest_records):
        if record.get("seed") != expected_seed:
            raise KimFinal86Error("final86 instance order mismatch")
        instance = final86._load_instance(FINAL86_ROOT, record)
        instance.validate_for(final86._new_environment())

    completions = {}
    for model_seed in MODEL_SEEDS:
        path = KIM85_ROOT / "training-completion" / f"seed-{model_seed}.json"
        completion = kim85._load_json(path, expected_type=dict)
        kim85._verify_self_hash(
            completion,
            "completion_sha256",
            label=f"Kim seed {model_seed} training completion",
        )
        checkpoint = TRAINING_ROOT / "training" / f"seed-{model_seed}" / "best.pth"
        if kim85._sha256(checkpoint) != completion.get("selected_checkpoint_raw_sha256"):
            raise KimFinal86Error(f"Kim seed {model_seed} checkpoint hash mismatch")
        completions[model_seed] = completion
    records = {int(record["seed"]): dict(record) for record in manifest["instances"]}
    if tuple(records) != PANEL_SEEDS or set(completions) != set(MODEL_SEEDS):
        raise KimFinal86Error("Kim/final-panel input grid mismatch")
    return manifest, completions, records


def _config(manifest: Mapping, completions: Mapping[int, Mapping]) -> dict:
    payload = {
        "protocol": PROTOCOL,
        "schema_version": SCHEMA_VERSION,
        "status": "prepared_no_kim_final86_rollouts_executed",
        "scientific_role": "post_hoc_matched_final_panel_supplement",
        "model_seeds": list(MODEL_SEEDS),
        "instance_seeds": list(PANEL_SEEDS),
        "rollouts_per_model_instance": len(ROLLOUTS),
        "expected_rows": EXPECTED_ROWS,
        "policy_rng_formula": "633000000 + 1000*model_seed + 10*panel_index + rollout",
        "policy_rng_range": [policy_seed(0, 0, 0), policy_seed(2, 29, 4)],
        "final_instance_manifest_sha256": manifest["manifest_sha256"],
        "checkpoints": {
            str(seed): {
                "selected_episode": completions[seed]["selected_episode"],
                "raw_sha256": completions[seed]["selected_checkpoint_raw_sha256"],
                "deployment_sha256": completions[seed][
                    "selected_checkpoint_deployment_sha256"
                ],
            }
            for seed in MODEL_SEEDS
        },
        "aggregation": (
            "five stochastic rolls within model-instance, equal three models "
            "within instance, then 30 EpisodeInstances"
        ),
        "complete_case_filtering_allowed": False,
        "baseline_reruns": 0,
        "kim_retraining": False,
    }
    payload["config_sha256"] = final86._digest(payload)
    return payload


def prepare(output_root: Path = OUTPUT_ROOT) -> dict:
    manifest, completions, _ = _inputs()
    config = _config(manifest, completions)
    root = Path(output_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    path = root / "run-config.json"
    if path.exists():
        final86._require_equal(
            "Kim final86 run config",
            final86._load_json(path, name="Kim final86 run config"),
            config,
        )
    else:
        final86._atomic_json(path, config)
    return config


def _ledger_path(root: Path, grid_row: Mapping[str, int]) -> Path:
    return (
        root
        / "run-ledger"
        / f"seed-{grid_row['model_seed']}"
        / f"instance-{grid_row['instance_seed']}-roll-{grid_row['rollout_index']}.json"
    )


def _failed_row(
    grid_row: Mapping[str, int], record: Mapping[str, Any], completion: Mapping,
    manifest_sha: str, error: Exception,
) -> dict:
    row = {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "method_id": kim85.KIM_STOCHASTIC,
        **dict(grid_row),
        "episode_instance_id": record["episode_instance_id"],
        "schedule_id": record["schedule_id"],
        "episode_instance_sha256": record["episode_instance_sha256"],
        "episode_instance_file_raw_sha256": record["saved_file_raw_sha256"],
        "evaluation_manifest_sha256": manifest_sha,
        "training_completion_sha256": completion["completion_sha256"],
        "source_checkpoint_raw_sha256": completion["selected_checkpoint_raw_sha256"],
        "source_checkpoint_deployment_sha256": completion[
            "selected_checkpoint_deployment_sha256"
        ],
        "strict_safe_complete": False,
        "safety_issues": ["execution_exception"],
        "method_failure_reason": f"{type(error).__name__}: {error}",
    }
    return kim85._add_self_hash(row, "row_sha256")


def _normalize(
    raw: Mapping[str, Any], *, grid_row: Mapping[str, int], record: Mapping,
    completion: Mapping, manifest_sha: str,
) -> dict:
    normalized = kim85._normalize_run(
        raw,
        grid_row={
            "model_seed": grid_row["model_seed"],
            "panel_index": grid_row["panel_index"],
            "instance_seed": grid_row["instance_seed"],
            "rollout": grid_row["rollout_index"],
            "policy_seed": grid_row["policy_rng_seed"],
        },
        instance_record=record,
        training_completion=completion,
        evaluation_manifest_sha256=manifest_sha,
    )
    normalized.pop("row_sha256", None)
    normalized["protocol"] = PROTOCOL
    return kim85._add_self_hash(normalized, "row_sha256")


def _validate_row(
    row: Mapping[str, Any], *, grid_row: Mapping[str, int], record: Mapping,
    completion: Mapping,
) -> None:
    kim85._verify_self_hash(row, "row_sha256", label="Kim final86 row")
    expected = {
        "protocol": PROTOCOL,
        "model_seed": grid_row["model_seed"],
        "panel_index": grid_row["panel_index"],
        "instance_seed": grid_row["instance_seed"],
        "rollout_index": grid_row["rollout_index"],
        "policy_rng_seed": grid_row["policy_rng_seed"],
        "episode_instance_id": record["episode_instance_id"],
        "schedule_id": record["schedule_id"],
        "episode_instance_sha256": record["episode_instance_sha256"],
        "episode_instance_file_raw_sha256": record["saved_file_raw_sha256"],
        "training_completion_sha256": completion["completion_sha256"],
        "source_checkpoint_raw_sha256": completion["selected_checkpoint_raw_sha256"],
    }
    for field, value in expected.items():
        if row.get(field) != value:
            raise KimFinal86Error(f"persisted Kim row binding mismatch: {field}")
    if type(row.get("strict_safe_complete")) is not bool:
        raise KimFinal86Error("Kim strict-safe flag must be boolean")
    if "method_audit" in row:
        recomputed = kim85._recompute_row_safety(
            row, training_completion=completion, instance_record=record
        )
        if row.get("safety_checks") != recomputed:
            raise KimFinal86Error("Kim row safety recomputation mismatch")
        issues = sorted(name for name, passed in recomputed.items() if passed is not True)
        if row.get("safety_issues") != issues or row["strict_safe_complete"] != (not issues):
            raise KimFinal86Error("Kim row strict-safe result mismatch")
    elif row["strict_safe_complete"]:
        raise KimFinal86Error("successful Kim row lacks primitive audit evidence")


def run(output_root: Path = OUTPUT_ROOT) -> dict:
    root = Path(output_root).resolve()
    config = prepare(root)
    manifest, completions, records = _inputs()
    kim85._configure_runtime()
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    args = kim85._evaluation_args()
    checkpoint_payloads = {
        seed: kim85._load_checkpoint_bound(
            TRAINING_ROOT / "training" / f"seed-{seed}" / "best.pth",
            completions[seed]["selected_checkpoint_raw_sha256"],
        )
        for seed in MODEL_SEEDS
    }
    executed = 0
    loaded = 0
    safe = 0
    for grid_row in expected_grid():
        model_seed = grid_row["model_seed"]
        seed = grid_row["instance_seed"]
        record = _instance_record(records[seed])
        completion = completions[model_seed]
        path = _ledger_path(root, grid_row)
        if path.exists():
            ledger = final86._load_json(path, name="Kim final86 ledger")
            final86._self_hash(ledger, "ledger_sha256", name="Kim final86 ledger")
            row = ledger["run"]
            loaded += 1
        else:
            try:
                instance = final86._load_instance(FINAL86_ROOT, records[seed])
                raw = kim85._evaluate_one(
                    args,
                    seed,
                    checkpoint_payloads[model_seed],
                    episode_instance=instance,
                    scheduler_variant=kim85.DURATION_AWARE_VARIANT,
                    assignment_source=kim85.TRACK_A_KIM2020_A3C_SPATIAL,
                    source_neutral_scheduler=True,
                    assignment_commitment=kim85.DECISION_EPOCH_RESERVED,
                    deployment_mode=kim85.DEPLOYMENT_STOCHASTIC,
                    policy_seed=grid_row["policy_rng_seed"],
                )
                row = _normalize(
                    raw,
                    grid_row=grid_row,
                    record=record,
                    completion=completion,
                    manifest_sha=manifest["manifest_sha256"],
                )
            except Exception as error:
                row = _failed_row(
                    grid_row, record, completion, manifest["manifest_sha256"], error
                )
            ledger = {
                "protocol": PROTOCOL,
                "schema_version": SCHEMA_VERSION,
                "config_sha256": config["config_sha256"],
                "grid_row": dict(grid_row),
                "run": row,
            }
            ledger["ledger_sha256"] = final86._digest(ledger)
            final86._atomic_json(path, ledger)
            ledger = final86._load_json(path, name="serialized Kim final86 ledger")
            final86._self_hash(ledger, "ledger_sha256", name="serialized Kim final86 ledger")
            row = ledger["run"]
            executed += 1
        _validate_row(row, grid_row=grid_row, record=record, completion=completion)
        safe += int(row["strict_safe_complete"])
        if (executed + loaded) % 25 == 0:
            print(
                f"Kim final86 {executed + loaded:3d}/{EXPECTED_ROWS} | safe {safe}",
                flush=True,
            )
    return {
        "status": "complete",
        "executed": executed,
        "loaded": loaded,
        "rows": EXPECTED_ROWS,
        "strict_safe_complete_rows": safe,
    }


def _load_rows(output_root: Path, completions: Mapping, records: Mapping) -> list[dict]:
    rows = []
    for grid_row in expected_grid():
        path = _ledger_path(output_root, grid_row)
        if not path.is_file():
            raise KimFinal86Error(f"missing Kim final86 ledger: {path}")
        ledger = final86._load_json(path, name="Kim final86 ledger")
        final86._self_hash(ledger, "ledger_sha256", name="Kim final86 ledger")
        row = dict(ledger["run"])
        record = _instance_record(records[grid_row["instance_seed"]])
        _validate_row(
            row,
            grid_row=grid_row,
            record=record,
            completion=completions[grid_row["model_seed"]],
        )
        rows.append(row)
    return rows


def _kim_instance_points(rows: Sequence[Mapping]) -> dict[int, dict]:
    grouped: dict[tuple[int, int], list[Mapping]] = defaultdict(list)
    for row in rows:
        grouped[(int(row["model_seed"]), int(row["instance_seed"]))].append(row)
    expected = {(model, seed) for model in MODEL_SEEDS for seed in PANEL_SEEDS}
    if set(grouped) != expected or any(len(group) != 5 for group in grouped.values()):
        raise KimFinal86Error("Kim model-instance-rollout grid is incomplete")
    model_points = {
        key: {
            metric: fmean(
                float(row[
                    "dense_objective_return" if metric == "dense_return" else metric
                ])
                for row in group
            )
            for metric in METRICS
        }
        for key, group in grouped.items()
    }
    return {
        seed: {
            metric: fmean(model_points[(model, seed)][metric] for model in MODEL_SEEDS)
            for metric in METRICS
        }
        for seed in PANEL_SEEDS
    }


def _point_dominates(left: Mapping, right: Mapping) -> bool:
    fields = ("mean_absolute_error", "physical_rehandles_per_100_required_deliveries")
    return all(float(left[field]) <= float(right[field]) for field in fields) and any(
        float(left[field]) < float(right[field]) for field in fields
    )


def analyze(output_root: Path = OUTPUT_ROOT) -> dict:
    root = Path(output_root).resolve()
    config = prepare(root)
    manifest, completions, records = _inputs()
    rows = _load_rows(root, completions, records)
    safe_count = sum(row["strict_safe_complete"] is True for row in rows)
    eligible = safe_count == EXPECTED_ROWS
    kim_points = _kim_instance_points(rows) if eligible else None
    kim_metrics = (
        {
            metric: fmean(kim_points[seed][metric] for seed in PANEL_SEEDS)
            for metric in METRICS
        }
        if kim_points is not None
        else None
    )

    corrected_outer = final86._load_json(CORRECTED_REPORT, name="corrected final86 report")
    final86._self_hash(corrected_outer, "report_sha256", name="corrected final86 report")
    corrected = corrected_outer["corrected_final_report"]
    final86._self_hash(corrected, "report_sha256", name="nested corrected final86 report")
    base_methods = [dict(record) for record in corrected["methods"]]
    kim_record = {
        "method": kim85.KIM_STOCHASTIC,
        "display_name": "Kim et al. (2020)-inspired spatial A3C",
        "expected_rows": EXPECTED_ROWS,
        "observed_rows": len(rows),
        "strict_safe_complete_rows": safe_count,
        "whole_method_eligible": eligible,
        "numeric_metrics_suppressed": not eligible,
        "metrics": kim_metrics,
        "instance_points": (
            [{"instance_seed": seed, **kim_points[seed]} for seed in PANEL_SEEDS]
            if kim_points is not None
            else None
        ),
        "failures": [
            {
                "model_seed": row["model_seed"],
                "instance_seed": row["instance_seed"],
                "rollout_index": row["rollout_index"],
                "policy_rng_seed": row["policy_rng_seed"],
                "reason": row.get("method_failure_reason"),
                "issues": row["safety_issues"],
            }
            for row in rows
            if not row["strict_safe_complete"]
        ],
    }
    methods = [*base_methods, kim_record]
    eligible_records = [record for record in methods if record["whole_method_eligible"]]
    frontier = [
        record["method"]
        for record in eligible_records
        if not any(
            other["method"] != record["method"]
            and _point_dominates(other["metrics"], record["metrics"])
            for other in eligible_records
        )
    ]
    contrasts = {}
    if eligible and kim_points is not None:
        for record in base_methods:
            if not record["whole_method_eligible"]:
                continue
            baseline_points = {
                int(point["instance_seed"]): point for point in record["instance_points"]
            }
            contrasts[record["method"]] = {
                "orientation": "Kim_minus_comparator",
                "nominal_95_paired_ci": {
                    metric: final86._mean_ci(
                        [
                            kim_points[seed][metric] - baseline_points[seed][metric]
                            for seed in PANEL_SEEDS
                        ],
                        final86.NOMINAL_T_95,
                    )
                    for metric in METRICS
                },
            }
    report = {
        "protocol": PROTOCOL,
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "scientific_role": "post_hoc_matched_final_panel_supplement",
        "config_sha256": config["config_sha256"],
        "final_instance_manifest_sha256": manifest["manifest_sha256"],
        "comparison_source": str(CORRECTED_REPORT),
        "baseline_policy_executions": 0,
        "kim_policy_executions": EXPECTED_ROWS,
        "methods": methods,
        "point_estimate_pareto_front": frontier,
        "kim_paired_contrasts": contrasts,
        "primary_coordinates": [
            "mean_absolute_error",
            "physical_rehandles_per_100_required_deliveries",
        ],
        "complete_case_filtering_used": False,
        "statistical_unit": "30 EpisodeInstances after nested stochastic/model averaging",
        "confirmatory_claim_authorized": False,
    }
    report["report_sha256"] = final86._digest(report)
    path = root / "comparison-report.json"
    if path.exists():
        final86._require_equal(
            "Kim final86 comparison report",
            final86._load_json(path, name="Kim final86 comparison report"),
            report,
        )
    else:
        final86._atomic_json(path, report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run", "analyze"))
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "prepare":
        result = prepare(args.output_root)
    elif args.command == "run":
        execution = run(args.output_root)
        result = {"execution": execution, "report": analyze(args.output_root)}
    else:
        result = analyze(args.output_root)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
