#!/usr/bin/env python3
"""Frozen 85k diagnostic evaluation of the damped conditioned-handling VCG."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Mapping, Optional, Sequence

import torch

import run_vcg_v11_conditioned_handling_seed0_85k as parent
import run_vcg_v11_nested_handling_pilot as pilot
import train_vcg_v11_conditioned_handling_damped_convergence as damped


PROTOCOL = "vcg_v1_1_conditioned_handling_damped_frozen_85k_diagnostic_v1"
SCHEMA_VERSION = 1
CONTRACT_NAME = "damped-evaluation-85k-contract.json"
LEDGER_NAME = "damped-evaluation-85k.json"
REPORT_NAME = "damped-evaluation-85k-report.json"
DAMPED_ROOT_RELATIVE = Path(
    "results/vcg-v1-1-conditioned-handling-seed0-damped-convergence"
)
PARENT_ROOT_RELATIVE = Path(
    "results/vcg-v1-1-conditioned-handling-seed0-85k-development"
)


class DampedEvaluationError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    if not path.is_file():
        raise DampedEvaluationError(f"missing required artifact: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_hash(value: Mapping) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    ).hexdigest()


def _json_form(value):
    return json.loads(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    )


def _damped_artifacts(project_root: Path) -> dict:
    root = project_root / DAMPED_ROOT_RELATIVE
    contract_path = root / damped.CONTRACT_NAME
    observed_contract = json.loads(contract_path.read_text(encoding="utf-8"))
    expected_contract = damped._contract(project_root, root, max_steps=2_000)
    if observed_contract != _json_form(expected_contract):
        raise DampedEvaluationError(
            "damped training contract, parents, or sources changed"
        )
    summary_path = root / damped.SUMMARY_NAME
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("status") != "complete":
        raise DampedEvaluationError("damped training is incomplete")
    convergence = summary.get("convergence_assessment", {})
    if convergence.get("passed") is not True or convergence.get("decision") != (
        "authorize_frozen_85k_diagnostic_evaluation"
    ):
        raise DampedEvaluationError("damped convergence gate did not authorize evaluation")
    terminal_path = root / damped.TERMINAL_NAME
    terminal = torch.load(terminal_path, map_location="cpu", weights_only=False)
    damped._validate_checkpoint(terminal, expected_contract, terminal=True)
    if terminal.get("convergence_assessment") != convergence:
        raise DampedEvaluationError("terminal convergence assessment changed")
    terminal_sha = _sha256(terminal_path)
    if terminal_sha != summary.get("terminal_checkpoint_sha256"):
        raise DampedEvaluationError("damped terminal identity changed")
    return {
        "root": root,
        "contract": observed_contract,
        "summary": summary,
        "summary_sha256": _sha256(summary_path),
        "terminal": terminal,
        "terminal_sha256": terminal_sha,
    }


def _parent_artifacts(project_root: Path) -> dict:
    root = project_root / PARENT_ROOT_RELATIVE
    contract = parent._require_contract(project_root, root)
    terminal, terminal_sha = parent._terminal(project_root, root)
    ledger_path = root / parent.LEDGER_NAME
    report_path = root / parent.REPORT_NAME
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if ledger.get("terminal_checkpoint_sha256") != terminal_sha:
        raise DampedEvaluationError("parent evaluation terminal identity changed")
    if report.get("contract_sha256") != contract["contract_sha256"]:
        raise DampedEvaluationError("parent report contract identity changed")
    return {
        "root": root,
        "contract": contract,
        "terminal": terminal,
        "terminal_sha256": terminal_sha,
        "ledger": ledger,
        "ledger_sha256": _sha256(ledger_path),
        "report": report,
        "report_sha256": _sha256(report_path),
    }


def _contract(project_root: Path, output_root: Path) -> dict:
    trained = _damped_artifacts(project_root)
    previous = _parent_artifacts(project_root)
    arm, _latest, sources, q_digest, _records = pilot._authenticate_inputs(
        project_root
    )
    identities = [
        {
            "instance_seed": seed,
            "episode_instance_id": sources.instances[seed].instance_id,
            "schedule_id": sources.instances[seed].schedule_id,
        }
        for seed in parent.INSTANCE_SEEDS
    ]
    if identities != previous["contract"]["instance_identities"]:
        raise DampedEvaluationError("85k EpisodeInstance identities changed")
    semantic = {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "scope": "opened_85k_seed0_frozen_diagnostic",
        "model_seed": parent.MODEL_SEED,
        "lambda_grid": list(parent.LAMBDA_GRID),
        "instance_seeds": list(parent.INSTANCE_SEEDS),
        "instance_identities": identities,
        "base_checkpoint_sha256": arm.checkpoint_sha256,
        "base_policy_digest": arm.deployment_policy_digest,
        "base_q_state_sha256": q_digest,
        "damped_contract_sha256": trained["contract"]["contract_sha256"],
        "damped_training_summary_sha256": trained["summary_sha256"],
        "damped_terminal_sha256": trained["terminal_sha256"],
        "damped_convergence_passed_before_evaluation": True,
        "fixed_terminal_checkpoint": True,
        "checkpoint_selection": False,
        "training_during_evaluation": False,
        "lambda_zero_new_sentinel_rollouts": 1,
        "lambda_zero_rows_reused_after_exact_sentinel": True,
        "positive_lambda_rollouts": (
            (len(parent.LAMBDA_GRID) - 1) * len(parent.INSTANCE_SEEDS)
        ),
        "total_new_rollouts": 97,
        "nested_reference_rows_reused": True,
        "parent_evaluation_ledger_sha256": previous["ledger_sha256"],
        "parent_evaluation_report_sha256": previous["report_sha256"],
        "complete_case_filtering_allowed": False,
        "frontier_contract": (
            "nondominance in MAE and physical rehandles per 100; duplicate "
            "coordinates count once"
        ),
        "source_sha256": {
            "evaluation_runner": _sha256(Path(__file__).resolve()),
            "damped_trainer": _sha256(
                project_root / "train_vcg_v11_conditioned_handling_damped_convergence.py"
            ),
            "conditioned_controller": _sha256(
                project_root / "vcg_v11_conditioned_handling.py"
            ),
            "parent_evaluator": _sha256(
                project_root / "run_vcg_v11_conditioned_handling_seed0_85k.py"
            ),
        },
        "output_root": str(output_root.resolve()),
        "advancement_scope": (
            "opened seed0 diagnostic only; matched model seeds 1 and 2 remain required"
        ),
    }
    return {**semantic, "contract_sha256": _canonical_hash(semantic)}


def prepare(project_root: Path, output_root: Path) -> dict:
    expected = _contract(project_root, output_root)
    path = output_root / CONTRACT_NAME
    if path.is_file():
        if json.loads(path.read_text(encoding="utf-8")) != expected:
            raise DampedEvaluationError(
                "evaluation contract, sources, or parent artifacts changed"
            )
    else:
        output_root.mkdir(parents=True, exist_ok=True)
        pilot._atomic_json(path, expected)
    return {
        "status": "prepared",
        "new_rollouts": 97,
        "nested_reference_rollouts_reused": 108,
        "contract": str(path.resolve()),
    }


def _require_contract(project_root: Path, output_root: Path) -> dict:
    path = output_root / CONTRACT_NAME
    if not path.is_file():
        raise DampedEvaluationError("run prepare first")
    observed = json.loads(path.read_text(encoding="utf-8"))
    expected = _contract(project_root, output_root)
    if observed != expected:
        raise DampedEvaluationError(
            "evaluation contract, sources, or parent artifacts changed"
        )
    return observed


def evaluate(project_root: Path, output_root: Path, *, device_name: str) -> dict:
    contract = _require_contract(project_root, output_root)
    trained = _damped_artifacts(project_root)
    previous = _parent_artifacts(project_root)
    path = output_root / LEDGER_NAME
    if path.is_file():
        observed = json.loads(path.read_text(encoding="utf-8"))
        if observed.get("damped_terminal_sha256") != trained["terminal_sha256"]:
            raise DampedEvaluationError("frozen evaluation terminal changed")
        return observed
    arm, _latest, sources, _q_digest, _records = pilot._authenticate_inputs(
        project_root
    )
    device = pilot._device(device_name)
    sentinel = sources.instances[parent.INSTANCE_SEEDS[0]]
    zero_raw = parent._run_conditioned(
        project_root,
        arm,
        sentinel,
        device=device,
        value=0.0,
        terminal=trained["terminal"],
    )
    zero_row = parent._behavior_row(zero_raw, sentinel)
    reference_zero = previous["ledger"]["lambda_zero_terminal_sentinel"]
    for key in (
        "behavior_digest",
        "dense_return",
        "mean_absolute_error",
        "steps",
        "physical_rehandles",
        "delivery_deviations",
    ):
        if zero_row[key] != reference_zero[key]:
            raise DampedEvaluationError(
                f"damped terminal lost lambda-zero exactness: {key}"
            )

    conditioned_rows = {
        "0.0": list(previous["ledger"]["conditioned_rows"]["0.0"])
    }
    for row in conditioned_rows["0.0"]:
        row["execution_reused_after_damped_terminal_sentinel"] = True
    for index, value in enumerate(parent.LAMBDA_GRID[1:], start=1):
        rows = []
        for seed in parent.INSTANCE_SEEDS:
            print(
                f"[damped {index}/{len(parent.LAMBDA_GRID)-1}] "
                f"lambda={value:g} seed={seed}",
                flush=True,
            )
            instance = sources.instances[seed]
            raw = parent._run_conditioned(
                project_root,
                arm,
                instance,
                device=device,
                value=value,
                terminal=trained["terminal"],
            )
            rows.append(parent._behavior_row(raw, instance))
        conditioned_rows[str(value)] = rows
    result = {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "status": "complete",
        "contract_sha256": contract["contract_sha256"],
        "damped_terminal_sha256": trained["terminal_sha256"],
        "lambda_zero_terminal_sentinel_exact": True,
        "lambda_zero_terminal_sentinel": zero_row,
        "new_rollout_count": 97,
        "conditioned_rows": conditioned_rows,
        "nested_reference_rows": previous["ledger"]["nested_reference_rows"],
        "undamped_round4_rows": previous["ledger"]["conditioned_rows"],
    }
    pilot._atomic_json(path, result)
    return result


def _coordinate(metric: Mapping) -> tuple[float, float]:
    return (
        round(float(metric["mean_absolute_error"]), 12),
        round(float(metric["physical_rehandles_per_100"]), 12),
    )


def distinct_nondominated_groups(metrics: Mapping[str, Mapping]) -> list[dict]:
    groups: dict[tuple[float, float], list[str]] = {}
    representatives = {}
    for name, metric in metrics.items():
        coordinate = _coordinate(metric)
        groups.setdefault(coordinate, []).append(str(name))
        representatives[coordinate] = metric
    result = []
    for coordinate, metric in representatives.items():
        dominated = False
        for other_coordinate, other in representatives.items():
            if other_coordinate == coordinate:
                continue
            no_worse = (
                other["mean_absolute_error"] <= metric["mean_absolute_error"]
                and other["physical_rehandles_per_100"]
                <= metric["physical_rehandles_per_100"]
            )
            strict = (
                other["mean_absolute_error"] < metric["mean_absolute_error"]
                or other["physical_rehandles_per_100"]
                < metric["physical_rehandles_per_100"]
            )
            if no_worse and strict:
                dominated = True
                break
        if not dominated:
            result.append(
                {
                    "mean_absolute_error": coordinate[0],
                    "physical_rehandles_per_100": coordinate[1],
                    "members": sorted(groups[coordinate]),
                }
            )
    return sorted(
        result,
        key=lambda item: (
            item["physical_rehandles_per_100"], item["mean_absolute_error"]
        ),
    )


def analyze(project_root: Path, output_root: Path) -> dict:
    contract = _require_contract(project_root, output_root)
    trained = _damped_artifacts(project_root)
    ledger = json.loads((output_root / LEDGER_NAME).read_text(encoding="utf-8"))
    conditioned_metrics = {}
    nested_metrics = {}
    undamped_metrics = {}
    action_profiles = {}
    behavior_changes = {}
    all_safe = True
    for value in parent.LAMBDA_GRID:
        key = str(value)
        conditioned = parent._metrics(ledger["conditioned_rows"][key])
        nested = parent._metrics(ledger["nested_reference_rows"][key])
        undamped = parent._metrics(ledger["undamped_round4_rows"][key])
        conditioned_metrics[key] = conditioned
        nested_metrics[key] = nested
        undamped_metrics[key] = undamped
        action_profiles[key] = parent._action_profile(
            ledger["conditioned_rows"][key]
        )
        all_safe &= conditioned is not None and nested is not None and undamped is not None
        comparable = [
            (current, old)
            for current, old in zip(
                ledger["conditioned_rows"][key],
                ledger["undamped_round4_rows"][key],
            )
            if current.get("behavior_digest") is not None
            and old.get("behavior_digest") is not None
        ]
        behavior_changes[key] = {
            "comparable_pairs": len(comparable),
            "changed_from_undamped_round4": sum(
                current["behavior_digest"] != old["behavior_digest"]
                for current, old in comparable
            ),
        }
    if all_safe:
        conditioned_frontier = distinct_nondominated_groups(conditioned_metrics)
        combined_frontier = distinct_nondominated_groups(
            {
                **{f"conditioned:{key}": value for key, value in conditioned_metrics.items()},
                **{f"nested:{key}": value for key, value in nested_metrics.items()},
            }
        )
        rehandles = [
            conditioned_metrics[str(value)]["physical_rehandles_per_100"]
            for value in parent.LAMBDA_GRID
        ]
        inversions = [
            {
                "lower_lambda": parent.LAMBDA_GRID[index],
                "upper_lambda": parent.LAMBDA_GRID[index + 1],
                "increase_per_100": float(rehandles[index + 1] - rehandles[index]),
            }
            for index in range(len(rehandles) - 1)
            if rehandles[index + 1] > rehandles[index]
        ]
    else:
        conditioned_frontier = None
        combined_frontier = None
        inversions = None
    high = conditioned_metrics.get("0.2")
    nested_high = nested_metrics.get("0.2")
    criteria = {
        "damped_training_convergence_precondition_passed": bool(
            trained["summary"]["convergence_assessment"]["passed"]
        ),
        "all_conditioned_and_reference_rows_strict_safe_complete": all_safe,
        "lambda_zero_exact_vcg_v1_1": bool(
            ledger["lambda_zero_terminal_sentinel_exact"]
        ),
        "lambda_0p2_rehandles_no_worse_than_nested": bool(
            all_safe
            and high["physical_rehandles_per_100"]
            <= nested_high["physical_rehandles_per_100"] + 1e-12
        ),
        "at_least_three_distinct_conditioned_frontier_coordinates": bool(
            conditioned_frontier is not None and len(conditioned_frontier) >= 3
        ),
        "no_adjacent_rehandle_inversion_above_5_per_100": bool(
            inversions is not None
            and all(item["increase_per_100"] <= 5.0 for item in inversions)
        ),
    }
    report = {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "status": "complete" if all_safe else "complete_with_metric_suppression",
        "scope": "opened_85k_seed0_frozen_diagnostic",
        "contract_sha256": contract["contract_sha256"],
        "damped_terminal_sha256": trained["terminal_sha256"],
        "lambda_grid": list(parent.LAMBDA_GRID),
        "conditioned_metrics": conditioned_metrics if all_safe else None,
        "nested_reference_metrics": nested_metrics if all_safe else None,
        "undamped_round4_metrics": undamped_metrics if all_safe else None,
        "distinct_conditioned_frontier": conditioned_frontier,
        "distinct_combined_frontier": combined_frontier,
        "adjacent_rehandle_inversions": inversions,
        "behavior_change_from_undamped_round4": behavior_changes,
        "conditioned_action_profiles": action_profiles,
        "diagnostic_gate": {
            "criteria": criteria,
            "passed": all(criteria.values()),
            "decision": (
                "prepare_matched_model_seeds_1_and_2"
                if all(criteria.values())
                else "do_not_train_more_seeds_inspect_damped_seed0_result"
            ),
        },
        "architecture_level_claim_authorized": False,
        "interpretation": (
            "This is a frozen diagnostic of the predeclared damped terminal on an "
            "already-opened seed-0 panel. Duplicate MAE-rehandle coordinates count "
            "once. Even a passing result requires matched seeds 1 and 2 before any "
            "architecture-level conclusion."
        ),
    }
    path = output_root / REPORT_NAME
    if path.is_file() and json.loads(path.read_text(encoding="utf-8")) != report:
        raise DampedEvaluationError("existing damped analysis changed")
    if not path.is_file():
        pilot._atomic_json(path, report)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("prepare", "evaluate", "analyze", "run")
    )
    parser.add_argument(
        "--project-root", type=Path, default=Path(__file__).resolve().parent
    )
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _parser().parse_args(argv)
    project_root = args.project_root.resolve()
    output_root = (
        args.output_root.resolve()
        if args.output_root is not None
        else project_root / DAMPED_ROOT_RELATIVE
    )
    torch.set_num_threads(1)
    if args.command == "prepare":
        result = prepare(project_root, output_root)
    elif args.command == "evaluate":
        result = evaluate(project_root, output_root, device_name=args.device)
    elif args.command == "analyze":
        result = analyze(project_root, output_root)
    else:
        prepare(project_root, output_root)
        evaluate(project_root, output_root, device_name=args.device)
        result = analyze(project_root, output_root)
    if "diagnostic_gate" in result:
        result = {
            "status": result["status"],
            "diagnostic_gate": result["diagnostic_gate"],
            "report": str((output_root / REPORT_NAME).resolve()),
        }
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
