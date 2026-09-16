#!/usr/bin/env python3
"""E5(c): learned future-handling consequence versus immediate cost only.

The control uses the frozen handling predictor at input lambda=.10 and applies
the requested deployment multiplier.  The ablation retains only the immediate
unit cost of a Reconfigure macro.  The exact-SAFE frontier, Q_op, liveness
guard, hierarchical selector, checkpoints, instances, and executor are fixed.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
from statistics import fmean
import sys
from typing import Mapping, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch import nn

from experiments.conditioned_vcg.E04_safe_frontier_ranking_92k import run as e04
from experiments.conditioned_vcg.E05_handling_model_ablation_92k import run as e05
import run_vcg_conditioned_final_comparison_90k as final90
import run_vcg_v11_conditioned_handling_seed0_85k as conditioned_seed0
import run_vcg_v11_nested_handling_pilot as pilot


PROTOCOL = "vcg_conditioned_e05c_future_consequence_ablation_92k_v1"
SCHEMA_VERSION = 1
MODEL_SEEDS = e04.MODEL_SEEDS
INSTANCE_SEEDS = e04.INSTANCE_SEEDS
DEPLOYMENT_LAMBDAS = e04.CONDITIONED_LAMBDAS
FUTURE_INPUT_LAMBDA = e05.CLAMPED_INPUT_LAMBDA
IMMEDIATE_ONLY = "immediate_reconfigure_cost_only"
FIXED_FUTURE = "fixed_learned_future_handling"

CONTRACT_NAME = "e05c-contract.json"
REPORT_NAME = "e05c-report.json"
TABLE_NAME = "e05c-results-table.md"
DEFAULT_OUTPUT = PROJECT_ROOT / "results/vcg-conditioned-e05c-future-consequence-ablation-92k"
E04_OUTPUT = e04.DEFAULT_OUTPUT
E05_OUTPUT = e05.DEFAULT_OUTPUT


class E5CError(RuntimeError):
    pass


def _canonical_bytes(value: Mapping) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _digest(value: Mapping, *, hash_field: Optional[str] = None) -> str:
    payload = dict(value)
    if hash_field is not None:
        payload.pop(hash_field, None)
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _with_hash(value: Mapping, field: str) -> dict:
    result = dict(value)
    result[field] = _digest(result)
    return result


def _sha256(path: Path) -> str:
    path = Path(path).absolute()
    if not path.is_file() or path.is_symlink():
        raise E5CError(f"missing canonical file: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_json(path: Path, *, label: str) -> dict:
    path = Path(path).absolute()
    if not path.is_file() or path.is_symlink():
        raise E5CError(f"missing canonical {label}: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise E5CError(f"invalid {label}: {path}") from error
    if not isinstance(value, dict):
        raise E5CError(f"{label} must contain an object")
    return value


def _verify_hash(value: Mapping, field: str, *, label: str) -> None:
    if value.get(field) != _digest(value, hash_field=field):
        raise E5CError(f"{label} self hash mismatch")


class ZeroFutureHandlingNetwork(nn.Module):
    """Return zero future cost, leaving the agent's immediate term intact."""

    def forward(self, features, preference_lambda):
        return torch.zeros(
            features.shape[0], dtype=features.dtype, device=features.device
        )


def _parents(project_root: Path, e04_output: Path, e05_output: Path) -> dict:
    e04_contract = e04.authenticate_contract(project_root, e04_output)
    e04_manifest = e04.authenticate_manifest(project_root, e04_output)
    e04_report = e04._load_json(e04_output / e04.REPORT_NAME, label="E4 report")
    e04._verify_hash(e04_report, "report_sha256", label="E4 report")
    e05_contract = e05.authenticate_contract(project_root, e05_output, e04_output)
    e05_report = e05._load_json(e05_output / e05.REPORT_NAME, label="E5(b) report")
    e05._verify_hash(e05_report, "report_sha256", label="E5(b) report")
    if (
        e04_report.get("status") != "complete"
        or e04_report.get("paper_evidence") is not True
        or e05_report.get("status") != "complete"
        or e05_report.get("paper_evidence") is not True
    ):
        raise E5CError("E5(c) requires complete authenticated E4 and E5(b) controls")
    return {
        "e04_contract": e04_contract,
        "e04_manifest": e04_manifest,
        "e04_report": e04_report,
        "e05_contract": e05_contract,
        "e05_report": e05_report,
    }


def _contract(project_root: Path, e04_output: Path, e05_output: Path) -> dict:
    parents = _parents(project_root, e04_output, e05_output)
    source_paths = (
        Path(__file__).resolve(),
        project_root / "vcg_v11_conditioned_handling.py",
        project_root / "run_vcg_conditioned_final_comparison_90k.py",
    )
    semantic = {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "scientific_question": (
            "does_learned_future_handling_improve_management_beyond_"
            "penalizing_the_current_reconfigure_macro"
        ),
        "model_seeds": list(MODEL_SEEDS),
        "instance_seeds": list(INSTANCE_SEEDS),
        "deployment_lambdas": list(DEPLOYMENT_LAMBDAS),
        "control": {
            "name": FIXED_FUTURE,
            "merit": "Qop-lambda*(I_reconfigure+N_future(h,0.10))",
            "lambda_005_and_020_rows_reused_from_E5b": True,
            "lambda_010_rows_reused_from_E4": True,
        },
        "ablation": {
            "name": IMMEDIATE_ONLY,
            "merit": "Qop-lambda*I_reconfigure",
            "new_rows": len(MODEL_SEEDS)
            * len(DEPLOYMENT_LAMBDAS)
            * len(INSTANCE_SEEDS),
        },
        "only_experimental_factor": "learned_future_handling_component_present",
        "held_fixed": [
            "frozen_Qop_and_checkpoint",
            "exact_SAFE_candidate_frontier",
            "candidate_generation",
            "recovery_witness_liveness_guard",
            "mode_aggregation_and_tie_breaking",
            "deployment_lambda",
            "EpisodeInstance",
            "macro_execution",
        ],
        "training_or_checkpoint_selection": False,
        "pilot_scope": {
            "model_seed": 0,
            "instance_count": e04.PILOT_INSTANCE_COUNT,
            "new_rows": e04.PILOT_INSTANCE_COUNT * len(DEPLOYMENT_LAMBDAS),
            "interpretation": "mechanism_check_only",
        },
        "e04_contract_sha256": parents["e04_contract"]["contract_sha256"],
        "e04_manifest_sha256": parents["e04_manifest"]["manifest_sha256"],
        "e04_report_sha256": parents["e04_report"]["report_sha256"],
        "e05_contract_sha256": parents["e05_contract"]["contract_sha256"],
        "e05_report_sha256": parents["e05_report"]["report_sha256"],
        "source_sha256": {
            str(path.relative_to(project_root)): _sha256(path)
            for path in source_paths
        },
    }
    return _with_hash(semantic, "contract_sha256")


def prepare(
    project_root: Path, output_dir: Path, e04_output: Path, e05_output: Path
) -> dict:
    output_dir = output_dir.absolute()
    expected = _contract(project_root, e04_output, e05_output)
    path = output_dir / CONTRACT_NAME
    if path.is_file():
        observed = _load_json(path, label="E5(c) contract")
        _verify_hash(observed, "contract_sha256", label="E5(c) contract")
        if observed != expected:
            raise E5CError("E5(c) contract, parents, inputs, or sources changed")
    else:
        if output_dir.exists() and any(output_dir.iterdir()):
            raise E5CError("nonempty E5(c) output has no contract")
        output_dir.mkdir(parents=True, exist_ok=True)
        final90._atomic_json(path, expected)
    return {
        "status": "prepared",
        "contract": str(path),
        "new_full_rows": expected["ablation"]["new_rows"],
        "reused_control_rows": expected["ablation"]["new_rows"],
    }


def authenticate_contract(
    project_root: Path, output_dir: Path, e04_output: Path, e05_output: Path
) -> dict:
    observed = _load_json(output_dir / CONTRACT_NAME, label="E5(c) contract")
    _verify_hash(observed, "contract_sha256", label="E5(c) contract")
    if observed != _contract(project_root, e04_output, e05_output):
        raise E5CError("E5(c) contract, parents, inputs, or sources changed")
    return observed


def _spec(model_seed: int, value: float) -> dict:
    return {
        "ablation_condition": IMMEDIATE_ONLY,
        "model_seed": int(model_seed),
        "deployment_lambda": float(value),
        "future_handling_component": 0.0,
    }


def _ledger_path(output_dir: Path, spec: Mapping, instance_seed: int) -> Path:
    return (
        output_dir
        / "run-ledger"
        / f"model-{int(spec['model_seed'])}"
        / f"lambda-{float(spec['deployment_lambda']):.3f}"
        / f"instance-{int(instance_seed)}.json"
    )


def _factory(project_root: Path, auth: Mapping, model_seed: int, value: float):
    def factory(base):
        agent = final90._load_conditioned_agent(
            project_root,
            auth,
            model_seed=model_seed,
            base=base,
            device=base.device,
        )
        agent.set_epsilon(0.0)
        agent.handling_network = ZeroFutureHandlingNetwork().to(base.device).eval()
        return conditioned_seed0._FixedLambdaAgent(agent, value)

    return factory


def _identity(record: Mapping) -> dict:
    return e04._identity(record)


def _row(raw: Mapping, compact: Mapping, *, identity: Mapping, spec: Mapping) -> dict:
    proxy_spec = {
        "ranking_signal": IMMEDIATE_ONLY,
        "model_seed": spec["model_seed"],
        "preference_lambda": spec["deployment_lambda"],
        "ranking_seed": None,
    }
    row = e04._row(raw, compact, identity=identity, spec=proxy_spec)
    row.update(
        {
            "protocol": PROTOCOL,
            "ablation_condition": IMMEDIATE_ONLY,
            "deployment_lambda": spec["deployment_lambda"],
            "future_handling_component": 0.0,
        }
    )
    return row


def _failed_row(error: Exception, *, identity: Mapping, spec: Mapping) -> dict:
    proxy_spec = {
        "ranking_signal": IMMEDIATE_ONLY,
        "model_seed": spec["model_seed"],
        "preference_lambda": spec["deployment_lambda"],
        "ranking_seed": None,
    }
    row = e04._failed_row(error, identity=identity, spec=proxy_spec)
    row.update(
        {
            "protocol": PROTOCOL,
            "ablation_condition": IMMEDIATE_ONLY,
            "deployment_lambda": spec["deployment_lambda"],
            "future_handling_component": 0.0,
        }
    )
    return row


def run(
    project_root: Path,
    output_dir: Path,
    e04_output: Path,
    e05_output: Path,
    *,
    selected_seed: Optional[int],
    instance_limit: Optional[int],
    device_name: str,
) -> dict:
    contract = authenticate_contract(project_root, output_dir, e04_output, e05_output)
    manifest = e04.authenticate_manifest(project_root, e04_output)
    seeds = MODEL_SEEDS if selected_seed is None else (selected_seed,)
    if any(seed not in MODEL_SEEDS for seed in seeds):
        raise E5CError("model seed must be 0, 1, or 2")
    device = pilot._device(device_name)
    if device.type != "cpu":
        raise E5CError("E5(c) is frozen to CPU execution")
    records = manifest["instances"]
    if instance_limit is not None:
        if instance_limit < 1 or instance_limit > len(records):
            raise E5CError("invalid E5(c) instance limit")
        records = records[:instance_limit]
    auth = final90._authenticate_inputs(project_root)
    conditioned = auth["conditioned"]
    complete = safe = 0
    for model_seed in seeds:
        arm = conditioned["inputs"]["arms"][model_seed]
        for value in DEPLOYMENT_LAMBDAS:
            spec = _spec(model_seed, value)
            for record in records:
                identity = _identity(record)
                path = _ledger_path(output_dir, spec, identity["instance_seed"])
                if path.is_file():
                    ledger = _load_json(path, label="E5(c) rollout ledger")
                    _verify_hash(ledger, "ledger_sha256", label="E5(c) rollout ledger")
                    if (
                        ledger.get("contract_sha256") != contract["contract_sha256"]
                        or ledger.get("e04_manifest_sha256")
                        != manifest["manifest_sha256"]
                        or ledger.get("spec") != spec
                    ):
                        raise E5CError("E5(c) rollout ledger binding changed")
                    row = ledger["run"]
                else:
                    instance = e04._load_instance(e04_output, record)
                    try:
                        raw = pilot._run_raw(
                            arm,
                            instance,
                            device=device,
                            wrapper_factory=_factory(
                                project_root, auth, model_seed, value
                            ),
                        )
                        compact = pilot._compact_row(raw, instance)
                        row = _row(raw, compact, identity=identity, spec=spec)
                    except Exception as error:
                        row = _failed_row(error, identity=identity, spec=spec)
                    ledger = _with_hash(
                        {
                            "schema_version": SCHEMA_VERSION,
                            "protocol": PROTOCOL,
                            "contract_sha256": contract["contract_sha256"],
                            "e04_manifest_sha256": manifest["manifest_sha256"],
                            "spec": spec,
                            "run": row,
                        },
                        "ledger_sha256",
                    )
                    final90._atomic_json(path, ledger)
                complete += 1
                safe += int(row["strict_safe_complete"])
                if complete % 10 == 0:
                    print(
                        f"E5(c) {complete}/{len(seeds) * len(DEPLOYMENT_LAMBDAS) * len(records)} "
                        f"| safe={safe}",
                        flush=True,
                    )
    return {
        "status": "complete",
        "new_immediate_only_rows": complete,
        "strict_safe_complete_rows": safe,
        "model_seeds": list(seeds),
        "instances": len(records),
    }


def _all_immediate_rows(output_dir: Path) -> list[dict]:
    rows = []
    root = output_dir / "run-ledger"
    if not root.is_dir():
        return rows
    for path in sorted(root.glob("**/instance-*.json")):
        ledger = _load_json(path, label="E5(c) rollout ledger")
        _verify_hash(ledger, "ledger_sha256", label="E5(c) rollout ledger")
        rows.append(ledger["run"])
    return rows


def _future_control(
    e04_output: Path,
    e05_output: Path,
    model_seed: int,
    value: float,
    instance_seed: int,
):
    if value == FUTURE_INPUT_LAMBDA:
        spec = {
            "ranking_signal": e04.CONDITIONED_SAFE,
            "model_seed": model_seed,
            "preference_lambda": value,
            "ranking_seed": None,
        }
        path = e04._ledger_path(e04_output, spec, instance_seed)
        ledger = e04._load_json(path, label="E4 fixed-future control ledger")
        e04._verify_hash(ledger, "ledger_sha256", label="E4 fixed-future control ledger")
        return ledger["run"]
    spec = e05._spec(model_seed, value)
    path = e05._ledger_path(e05_output, spec, instance_seed)
    ledger = e05._load_json(path, label="E5(b) fixed-future control ledger")
    e05._verify_hash(ledger, "ledger_sha256", label="E5(b) fixed-future control ledger")
    return ledger["run"]


METRICS = (
    "dense_return",
    "mean_absolute_error",
    "within_target_window_rate",
    "steps",
    "physical_rehandles_per_100_required_deliveries",
)


def _aggregate(rows: Sequence[Mapping]) -> dict:
    safe = [row for row in rows if row["strict_safe_complete"]]
    result = {
        "rows": len(rows),
        "strict_safe_complete": len(safe),
        "strict_completion_rate": len(safe) / len(rows) if rows else None,
        "complete_case_metrics_suppressed": len(safe) != len(rows),
    }
    for metric in METRICS:
        result[metric] = (
            fmean(float(row[metric]) for row in safe)
            if rows and len(safe) == len(rows)
            else None
        )
    return result


def _table(report: Mapping) -> str:
    lines = [
        "# E5(c) learned future-handling consequence ablation",
        "",
        "| Lambda | Condition | Strict complete | Return | MAE | Within ±20 | Steps | Rehandles/100 |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for value in DEPLOYMENT_LAMBDAS:
        for condition in (FIXED_FUTURE, IMMEDIATE_ONLY):
            item = report["aggregate"][f"{condition}_lambda_{value:.2f}"]

            def metric(name: str, digits: int = 2) -> str:
                observed = item[name]
                return "—" if observed is None else f"{observed:.{digits}f}"

            lines.append(
                f"| {value:.2f} | {condition} | "
                f"{item['strict_safe_complete']}/{item['rows']} | "
                f"{metric('dense_return')} | {metric('mean_absolute_error')} | "
                f"{metric('within_target_window_rate', 3)} | {metric('steps')} | "
                f"{metric('physical_rehandles_per_100_required_deliveries')} |"
            )
    lines.extend(
        [
            "",
            "Differences in the report are immediate-only minus fixed learned future handling on identical model-seed/EpisodeInstance coordinates.",
            "The fixed-future control evaluates the frozen predictor at input lambda=.10 for every deployment lambda.",
        ]
    )
    return "\n".join(lines) + "\n"


def analyze(
    project_root: Path,
    output_dir: Path,
    e04_output: Path,
    e05_output: Path,
    *,
    allow_partial: bool,
) -> dict:
    contract = authenticate_contract(project_root, output_dir, e04_output, e05_output)
    parents = _parents(project_root, e04_output, e05_output)
    immediate_rows = _all_immediate_rows(output_dir)
    expected = len(MODEL_SEEDS) * len(DEPLOYMENT_LAMBDAS) * len(INSTANCE_SEEDS)
    if not allow_partial and len(immediate_rows) != expected:
        raise E5CError(f"E5(c) grid is incomplete: {len(immediate_rows)}/{expected}")

    aggregate = {}
    paired = {}
    for value in DEPLOYMENT_LAMBDAS:
        immediate = [
            row for row in immediate_rows if float(row["deployment_lambda"]) == value
        ]
        controls = [
            _future_control(
                e04_output,
                e05_output,
                int(row["model_seed"]),
                value,
                int(row["instance_seed"]),
            )
            for row in immediate
        ]
        aggregate[f"{FIXED_FUTURE}_lambda_{value:.2f}"] = _aggregate(controls)
        aggregate[f"{IMMEDIATE_ONLY}_lambda_{value:.2f}"] = _aggregate(immediate)
        pairs = list(zip(controls, immediate))
        paired[f"lambda_{value:.2f}"] = {
            "pairs": len(pairs),
            "identical_behavior_digest": sum(
                control.get("behavior_digest") == ablation.get("behavior_digest")
                for control, ablation in pairs
            ),
            "strict_completion_immediate_minus_future": (
                fmean(
                    int(ablation["strict_safe_complete"])
                    - int(control["strict_safe_complete"])
                    for control, ablation in pairs
                )
                if pairs
                else None
            ),
            **{
                f"{metric}_immediate_minus_future": (
                    fmean(
                        float(ablation[metric]) - float(control[metric])
                        for control, ablation in pairs
                    )
                    if pairs
                    and all(
                        control["strict_safe_complete"]
                        and ablation["strict_safe_complete"]
                        for control, ablation in pairs
                    )
                    else None
                )
                for metric in METRICS
            },
        }

    complete = len(immediate_rows) == expected
    report = _with_hash(
        {
            "schema_version": SCHEMA_VERSION,
            "protocol": PROTOCOL,
            "status": "complete" if complete else "partial",
            "paper_evidence": complete,
            "contract_sha256": contract["contract_sha256"],
            "e04_manifest_sha256": parents["e04_manifest"]["manifest_sha256"],
            "new_immediate_only_rows": len(immediate_rows),
            "expected_new_rows": expected,
            "reused_fixed_future_control_rows": len(immediate_rows),
            "aggregate": aggregate,
            "paired_immediate_minus_future": paired,
            "causal_scope": (
                "learned_future_handling_component_given_the_final_frozen_Qop_and_SAFE_frontier"
            ),
        },
        "report_sha256",
    )
    final90._atomic_json(output_dir / REPORT_NAME, report)
    final90._atomic_text(output_dir / TABLE_NAME, _table(report))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run", "run-all", "analyze"))
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--e04-output", type=Path, default=E04_OUTPUT)
    parser.add_argument("--e05-output", type=Path, default=E05_OUTPUT)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--instance-limit", type=int)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    project_root = args.project_root.resolve()
    output_dir = args.output_dir.resolve()
    e04_output = args.e04_output.resolve()
    e05_output = args.e05_output.resolve()
    if args.command == "prepare":
        result = prepare(project_root, output_dir, e04_output, e05_output)
    elif args.command == "run":
        result = run(
            project_root,
            output_dir,
            e04_output,
            e05_output,
            selected_seed=args.seed,
            instance_limit=args.instance_limit,
            device_name=args.device,
        )
    elif args.command == "run-all":
        result = run(
            project_root,
            output_dir,
            e04_output,
            e05_output,
            selected_seed=None,
            instance_limit=None,
            device_name=args.device,
        )
    else:
        result = analyze(
            project_root,
            output_dir,
            e04_output,
            e05_output,
            allow_partial=args.allow_partial,
        )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
