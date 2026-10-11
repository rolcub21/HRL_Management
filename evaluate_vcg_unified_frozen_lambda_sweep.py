#!/usr/bin/env python3
"""Sweep lambda at evaluation time on one frozen unified-VCG checkpoint."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import hashlib
import importlib
import json
import math
from pathlib import Path
from statistics import fmean
from types import SimpleNamespace
from typing import Mapping, Optional, Sequence

import torch

import compare_unified_vcg as pair
import train_vcg_constrained_v2_1 as atomic_io
import train_vcg_constrained_v2_3 as v23
import train_vcg_unified as unified
import train_vcg_unified_fixed_lambda_sweep as fixed_training


PROTOCOL = "vcg_unified_frozen_checkpoint_lambda_sweep_85k_v1"
SCHEMA_VERSION = 1
LAMBDAS = (0.0, 0.05, 0.10, 0.20, 0.30)
EPISODE = 300
EXPECTED_PARENT_METHOD = "vcg_unified_budget_sweep_v1"
EXPECTED_PARENT_FAMILY = "vcg_unified_budget_sweep_terminal_development_v1"
FIELDS = (
    "dense_return",
    "mean_absolute_error",
    "physical_rehandles_per_100",
    "steps",
    "within_window_percentage",
    "mean_earliness",
    "mean_tardiness",
)


class FrozenLambdaSweepError(ValueError):
    """Raised when the frozen-checkpoint sweep contract is violated."""


def _read_mapping(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise FrozenLambdaSweepError(f"cannot read {path}") from error
    if not isinstance(value, dict):
        raise FrozenLambdaSweepError(f"{path} must contain an object")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _lambda_key(value: float) -> str:
    return f"lambda-{float(value):.2f}"


def _authenticate_parent(parent_dir: Path) -> dict:
    root = Path(parent_dir).resolve()
    summary_path = root / "training-summary.json"
    contract_path = root / "training-contract.json"
    checkpoint_path = root / "final-model.pth"
    summary = _read_mapping(summary_path)
    contract = _read_mapping(contract_path)
    if summary.get("status") != "complete" or summary.get("variant") != unified.VCG:
        raise FrozenLambdaSweepError("parent must be the completed lambda-zero arm")
    if summary.get("method_version") != EXPECTED_PARENT_METHOD:
        raise FrozenLambdaSweepError("parent method version mismatch")
    if summary.get("completed_training_episodes") != EPISODE:
        raise FrozenLambdaSweepError("parent must be the 300-episode terminal model")
    if float(summary.get("lambda_state", {}).get("lambda_value", -1.0)) != 0.0:
        raise FrozenLambdaSweepError("parent lambda is not zero")
    if summary.get("final_validation", {}).get("strict_integrity_gate") is not True:
        raise FrozenLambdaSweepError("parent terminal validation was not strict-safe")
    if contract.get("method_version") != EXPECTED_PARENT_METHOD:
        raise FrozenLambdaSweepError("parent training contract mismatch")
    base_contract = contract.get("base_v2_3_runtime_contract")
    if not isinstance(base_contract, dict) or base_contract.get("episodes") != EPISODE:
        raise FrozenLambdaSweepError("parent V2.3 runtime contract mismatch")
    if not checkpoint_path.is_file():
        raise FrozenLambdaSweepError("parent final checkpoint is missing")
    checkpoint_sha = _sha256(checkpoint_path)
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except Exception as error:
        raise FrozenLambdaSweepError("cannot load parent checkpoint safely") from error
    if not isinstance(checkpoint, dict):
        raise FrozenLambdaSweepError("parent checkpoint must be an object")
    expected = {
        "checkpoint_family": EXPECTED_PARENT_FAMILY,
        "method_version": EXPECTED_PARENT_METHOD,
        "unified_variant": unified.VCG,
        "completed_episodes": EPISODE,
        "terminal_checkpoint": True,
        "development_only": True,
        "deployment_checkpoint_eligible": False,
        "final_86xxx_panel_opened": False,
        "q_operational_trained": True,
        "q_physical_rehandle_trained": True,
        "q_physical_rehandle_monitored": True,
        "runtime_lambda": 0.0,
        "training_contract_sha256": contract.get("contract_sha256"),
    }
    mismatches = {
        key: (checkpoint.get(key), value)
        for key, value in expected.items()
        if checkpoint.get(key) != value
    }
    if mismatches:
        raise FrozenLambdaSweepError(f"parent checkpoint mismatch: {mismatches!r}")
    agent_checkpoint = checkpoint.get("agent_state")
    if not isinstance(agent_checkpoint, dict):
        raise FrozenLambdaSweepError("parent agent checkpoint is missing")
    return {
        "root": root,
        "summary": summary,
        "contract": contract,
        "base_contract": base_contract,
        "checkpoint": checkpoint,
        "checkpoint_path": checkpoint_path,
        "checkpoint_sha256": checkpoint_sha,
        "summary_sha256": _sha256(summary_path),
        "contract_sha256_raw": _sha256(contract_path),
    }


def _build_contract(parent: Mapping, *, device: str) -> dict:
    if str(device) != str(parent["base_contract"].get("device")):
        raise FrozenLambdaSweepError(
            "evaluation device must match the frozen parent contract"
        )
    source_path = Path(__file__).resolve()
    result = {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "scientific_role": "post_hoc_frozen_checkpoint_policy_scalarization_diagnostic",
        "parent_root": str(parent["root"]),
        "parent_checkpoint": str(parent["checkpoint_path"]),
        "parent_checkpoint_raw_sha256": parent["checkpoint_sha256"],
        "parent_training_summary_raw_sha256": parent["summary_sha256"],
        "parent_training_contract_raw_sha256": parent["contract_sha256_raw"],
        "parent_training_contract_canonical_sha256": parent["contract"]["contract_sha256"],
        "parent_model_seed": int(parent["base_contract"]["model_seed"]),
        "checkpoint_weights_frozen_across_all_rows": True,
        "training_or_learning": False,
        "lambda_values": LAMBDAS,
        "validation_episode": EPISODE,
        "validation_instance_seeds": tuple(v23.DEFAULT_VALIDATION_SEEDS),
        "action_rng_grid": tuple(
            tuple(
                unified.VALIDATION_POLICY_RNG_BASE + 4 * i + j
                for j in range(v23.VALIDATION_POLICY_RNG_COUNT)
            )
            for i in range(len(v23.DEFAULT_VALIDATION_SEEDS))
        ),
        "rows_per_lambda": 48,
        "expected_total_rows": 48 * len(LAMBDAS),
        "device": str(device),
        "source_path": str(source_path),
        "source_sha256": _sha256(source_path),
        "development_only": True,
        "budget_or_kkt_claim": False,
        "final_86xxx_panel_opened": False,
    }
    result["contract_sha256"] = v23.contract_hash(result)
    return result


def prepare(output_dir: Path, parent_dir: Path, *, device: str) -> dict:
    parent = _authenticate_parent(parent_dir)
    contract = _build_contract(parent, device=device)
    output = Path(output_dir).resolve()
    contract_path = output / "evaluation-contract.json"
    if contract_path.exists():
        if v23._json_safe(_read_mapping(contract_path)) != v23._json_safe(contract):
            raise FrozenLambdaSweepError("existing evaluation contract drifted")
    else:
        if output.exists() and any(output.iterdir()):
            raise FrozenLambdaSweepError("new evaluation root is not empty")
        atomic_io._atomic_json(contract, contract_path)
    return contract


def _runtime(parent: Mapping, *, device: str):
    runtime = v23.load_default_runtime(
        SimpleNamespace(device=device), parent["base_contract"]
    )
    core = importlib.import_module("viability_graph_constrained_v2_3")
    runtime.agent = core.ConstrainedV23HierarchyAgent.from_checkpoint(
        parent["checkpoint"]["agent_state"],
        device=device,
        resumable=False,
        seed=int(parent["base_contract"]["model_seed"]),
    )
    if runtime.agent.config.to_dict() != parent["base_contract"]["agent_config"]:
        raise FrozenLambdaSweepError("loaded agent configuration drifted")
    return runtime


def _validate_ledger(ledger: Mapping, fixed_lambda: float) -> list[dict]:
    received_sha = ledger.get("ledger_sha256")
    canonical = dict(ledger)
    canonical.pop("ledger_sha256", None)
    if (
        ledger.get("schema_version") != SCHEMA_VERSION
        or ledger.get("protocol") != PROTOCOL
        or not math.isclose(
            float(ledger.get("fixed_lambda")), fixed_lambda, rel_tol=0.0, abs_tol=1e-12
        )
        or ledger.get("row_count") != 48
        or ledger.get("training_or_learning") is not False
        or not isinstance(received_sha, str)
        or v23.contract_hash(canonical) != received_sha
        or ledger.get("validation_summary", {}).get("strict_integrity_gate") is not True
    ):
        raise FrozenLambdaSweepError(f"{_lambda_key(fixed_lambda)} ledger mismatch")
    rows = ledger.get("rows")
    if not isinstance(rows, list) or len(rows) != 48:
        raise FrozenLambdaSweepError(f"{_lambda_key(fixed_lambda)} row grid is incomplete")
    if any(
        not math.isclose(float(row.get("dual_lambda")), fixed_lambda, rel_tol=0.0, abs_tol=1e-12)
        for row in rows
    ):
        raise FrozenLambdaSweepError(f"{_lambda_key(fixed_lambda)} row lambda drifted")
    expected_grid = {
        (
            int(seed),
            rng_index,
            unified.VALIDATION_POLICY_RNG_BASE + 4 * instance_index + rng_index,
        )
        for instance_index, seed in enumerate(v23.DEFAULT_VALIDATION_SEEDS)
        for rng_index in range(v23.VALIDATION_POLICY_RNG_COUNT)
    }
    observed_grid = {
        (
            int(row.get("instance_seed")),
            int(row.get("policy_rng_index")),
            int(row.get("policy_rng_seed")),
        )
        for row in rows
    }
    if observed_grid != expected_grid:
        raise FrozenLambdaSweepError(f"{_lambda_key(fixed_lambda)} RNG grid drifted")
    if any(
        row.get("strict_method_success") is not True
        or row.get("delivery_count") != row.get("required_deliveries")
        or row.get("method_failure_reason") is not None
        or row.get("all_selected_candidates_exact_safe") is not True
        or any(int(row.get(field, -1)) != 0 for field in ("illegal_drops", "fallbacks", "witness_mismatches"))
        for row in rows
    ):
        raise FrozenLambdaSweepError(f"{_lambda_key(fixed_lambda)} contains an unsafe row")
    pair._instance_points(rows)
    return rows


def evaluate(output_dir: Path, parent_dir: Path, *, device: str) -> dict:
    contract = prepare(output_dir, parent_dir, device=device)
    parent = _authenticate_parent(parent_dir)
    output = Path(output_dir).resolve()
    completed = {}
    with ExitStack() as stack:
        stack.enter_context(fixed_training.activated_fixed_lambda_protocol())
        stack.enter_context(unified.activated_unified_seed_profile())
        runtime = _runtime(parent, device=device)
        before_local = runtime._module_signature(runtime.agent.Q_local)
        before_target = runtime._module_signature(runtime.agent.Q_target)
        schedule = v23._install_schedule(
            runtime, v23.schedule_for_episode(EPISODE, validation=True)
        )
        for fixed_lambda in LAMBDAS:
            ledger_path = output / "validation-ledger" / f"{_lambda_key(fixed_lambda)}.json"
            if ledger_path.exists():
                ledger = _read_mapping(ledger_path)
                rows = _validate_ledger(ledger, fixed_lambda)
            else:
                runtime.set_dual_lambda(fixed_lambda)
                rows, batch_audit = unified._validation_rows(
                    runtime,
                    SimpleNamespace(
                        validation_seeds=v23.DEFAULT_VALIDATION_SEEDS,
                        max_steps=2_000,
                    ),
                    schedule,
                    fixed_lambda,
                )
                summary = v23.summarize_validation(
                    rows,
                    v23.V23DualConfig(),
                    checkpoint_episode=EPISODE,
                    validation_lambda=fixed_lambda,
                    schedule_state=schedule,
                )
                if summary.get("strict_integrity_gate") is not True:
                    raise FrozenLambdaSweepError(
                        f"{_lambda_key(fixed_lambda)} was unsafe or incomplete"
                    )
                ledger = {
                    "schema_version": SCHEMA_VERSION,
                    "protocol": PROTOCOL,
                    "contract_sha256": contract["contract_sha256"],
                    "parent_checkpoint_raw_sha256": parent["checkpoint_sha256"],
                    "fixed_lambda": fixed_lambda,
                    "checkpoint_episode": EPISODE,
                    "training_or_learning": False,
                    "batch_audit": batch_audit,
                    "validation_summary": summary,
                    "rows": tuple(rows),
                    "row_count": len(rows),
                    "final_86xxx_panel_opened": False,
                }
                ledger["ledger_sha256"] = v23.contract_hash(ledger)
                atomic_io._atomic_json(ledger, ledger_path)
                rows = _validate_ledger(_read_mapping(ledger_path), fixed_lambda)
            completed[_lambda_key(fixed_lambda)] = len(rows)
        after_local = runtime._module_signature(runtime.agent.Q_local)
        after_target = runtime._module_signature(runtime.agent.Q_target)
        if (before_local, before_target) != (after_local, after_target):
            raise FrozenLambdaSweepError("frozen checkpoint weights changed")
    return {
        "status": "evaluation_complete",
        "protocol": PROTOCOL,
        "completed_rows": completed,
        "q_local_sha256": before_local,
        "q_target_sha256": before_target,
        "weights_unchanged": True,
        "final_86xxx_panel_opened": False,
    }


def _means(points: Mapping[int, Mapping]) -> dict:
    return {
        field: float(fmean(float(point[field]) for point in points.values()))
        for field in FIELDS
    }


def _dominates(left: Mapping, right: Mapping) -> bool:
    a = (left["mean_absolute_error"], left["physical_rehandles_per_100"])
    b = (right["mean_absolute_error"], right["physical_rehandles_per_100"])
    return all(x <= y for x, y in zip(a, b)) and any(x < y for x, y in zip(a, b))


def analyze(output_dir: Path, parent_dir: Path, *, device: str) -> dict:
    contract = prepare(output_dir, parent_dir, device=device)
    output = Path(output_dir).resolve()
    points = {}
    for fixed_lambda in LAMBDAS:
        ledger = _read_mapping(
            output / "validation-ledger" / f"{_lambda_key(fixed_lambda)}.json"
        )
        points[_lambda_key(fixed_lambda)] = pair._instance_points(
            _validate_ledger(ledger, fixed_lambda)
        )
    metrics = {name: _means(value) for name, value in points.items()}
    order = tuple(_lambda_key(value) for value in LAMBDAS)
    frontier = tuple(
        name
        for name in order
        if not any(
            _dominates(metrics[other], metrics[name])
            for other in order
            if other != name
        )
    )
    rehandles = tuple(metrics[name]["physical_rehandles_per_100"] for name in order)
    report = {
        "status": "complete",
        "protocol": PROTOCOL,
        "contract_sha256": contract["contract_sha256"],
        "scientific_role": "frozen_checkpoint_policy_scalarization_diagnostic",
        "table": tuple(
            {
                "arm": name,
                "fixed_lambda": fixed_lambda,
                **metrics[name],
                "developmental_point_estimate_pareto": name in frontier,
            }
            for name, fixed_lambda in zip(order, LAMBDAS)
        ),
        "paired_differences_vs_lambda0": {
            name: {
                field: pair._paired_difference(
                    points[name], points[_lambda_key(0.0)], field
                )
                for field in FIELDS
            }
            for name in order[1:]
        },
        "developmental_mae_rehandle_point_estimate_frontier": frontier,
        "rehandles_nonincreasing_as_lambda_increases": all(
            later <= earlier + 1e-12
            for earlier, later in zip(rehandles, rehandles[1:])
        ),
        "same_frozen_weights_for_all_rows": True,
        "training_or_learning": False,
        "all_240_rows_strict_safe_complete": True,
        "claim_scope": "one frozen development checkpoint on the opened 85xxx panel",
        "final_86xxx_panel_opened": False,
    }
    atomic_io._atomic_json(report, output / "frozen-lambda-comparison.json")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "run", "analyze"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--parent-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    if args.action == "prepare":
        result = prepare(args.output_dir, args.parent_dir, device=args.device)
    elif args.action == "run":
        evaluation = evaluate(args.output_dir, args.parent_dir, device=args.device)
        result = {
            "evaluation": evaluation,
            "analysis": analyze(args.output_dir, args.parent_dir, device=args.device),
        }
    else:
        result = analyze(args.output_dir, args.parent_dir, device=args.device)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
