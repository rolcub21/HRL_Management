#!/usr/bin/env python3
"""Single-seed development screen for anchored preference-conditioned VCG."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from statistics import fmean
from typing import Mapping, Optional, Sequence

import torch

import run_vcg_v11_nested_handling_pilot as pilot
import train_vcg_v11_anchored_preference as trainer
from vcg_v11_anchored_preference import AnchoredPreferenceAgent


PROTOCOL = "vcg_v1_1_anchored_preference_seed0_85k_screen_v1"
SCHEMA_VERSION = 1
MODEL_SEED = 0
INSTANCE_SEEDS = tuple(range(85_000, 85_012))
LAMBDA_GRID = trainer.PREFERENCE_GRID
EXPECTED_BLOCKS = 8
CONTRACT_NAME = "anchored-seed0-contract.json"
PARITY_NAME = "initial-anchor-parity.json"
LEDGER_NAME = "adapted-seed0-evaluation.json"
REPORT_NAME = "anchored-seed0-report.json"
HISTORICAL_RELATIVE_PATH = Path(
    "results/vcg-v1-1-nested-handling-seed0-85k-development/pilot-sweep.json"
)
TRAINING_RELATIVE_PATH = Path("training/seed-0")


class AnchoredScreenError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    if not path.is_file():
        raise AnchoredScreenError(f"missing required artifact: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_hash(value: Mapping) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
            "utf-8"
        )
    ).hexdigest()


def _historical(project_root: Path) -> dict:
    path = project_root / HISTORICAL_RELATIVE_PATH
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("status") != "complete":
        raise AnchoredScreenError("seed-0 nested handling pilot is incomplete")
    if tuple(float(x) for x in report.get("lambda_grid", ())) != (
        0.0,
        0.01,
        0.025,
        0.05,
        0.1,
        0.2,
    ):
        raise AnchoredScreenError("seed-0 nested handling lambda grid changed")
    for value in LAMBDA_GRID:
        rows = report.get("rows", {}).get(str(value))
        if not isinstance(rows, list) or len(rows) != len(INSTANCE_SEEDS):
            raise AnchoredScreenError(f"historical rows missing for lambda={value}")
        if tuple(int(row["instance_seed"]) for row in rows) != INSTANCE_SEEDS:
            raise AnchoredScreenError("historical instance order changed")
    return report


def _source_hashes(project_root: Path) -> dict:
    paths = {
        "screen_runner": Path(__file__).resolve(),
        "anchored_controller": project_root / "vcg_v11_anchored_preference.py",
        "anchored_trainer": project_root / "train_vcg_v11_anchored_preference.py",
    }
    return {key: _sha256(path) for key, path in sorted(paths.items())}


def _contract(project_root: Path, output_root: Path) -> dict:
    arm, _latest, sources, q_digest, _records = pilot._authenticate_inputs(project_root)
    historical_path = project_root / HISTORICAL_RELATIVE_PATH
    _historical(project_root)
    cost_path = project_root / trainer.COST_RELATIVE_PATH
    identities = [
        {
            "instance_seed": seed,
            "episode_instance_id": sources.instances[seed].instance_id,
            "schedule_id": sources.instances[seed].schedule_id,
        }
        for seed in INSTANCE_SEEDS
    ]
    semantic = {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "scope": "opened_85k_single_seed_development_screen",
        "model_seed": MODEL_SEED,
        "training_episodes": trainer.DEFAULT_EPISODES,
        "training_seed_base": trainer.TRAIN_SEED_BASE,
        "preference_grid": list(LAMBDA_GRID),
        "instance_seeds": list(INSTANCE_SEEDS),
        "base_checkpoint_sha256": arm.checkpoint_sha256,
        "base_policy_digest": arm.deployment_policy_digest,
        "base_q_state_sha256": q_digest,
        "source_cost_sha256": _sha256(cost_path),
        "historical_seed0_sweep_sha256": _sha256(historical_path),
        "training_output": str((output_root / TRAINING_RELATIVE_PATH).resolve()),
        "initial_anchor_parity": {
            "lambda_zero": "exact_vcg_v1_1_direct_delegation",
            "positive_lambdas": "exact_seed85000_behavior_match_to_nested_vcg",
            "rollouts": len(LAMBDA_GRID),
        },
        "evaluation": {
            "fixed_terminal_checkpoint": True,
            "checkpoint_selection": False,
            "positive_lambda_new_rows": (
                (len(LAMBDA_GRID) - 1) * len(INSTANCE_SEEDS)
            ),
            "lambda_zero_rows_reused_after_exact_terminal_sentinel": True,
            "lambda_zero_new_sentinel_rows": 1,
            "complete_case_filtering_allowed": False,
        },
        "architecture": {
            "frozen_vcg_v1_1_operational_anchor": True,
            "lambda_zero_exact_by_construction": True,
            "positive_lambda_warm_start_from_nested_vcg": True,
            "preference_conditioned_operational_residual": True,
            "preference_conditioned_handling_head": True,
            "handling_target": "undiscounted_physical_rehandles",
            "teacher_policy_queries": False,
        },
        "advancement_scope": (
            "seed0 behavior diagnostic only; seeds1_and_2_required_before "
            "architecture-level conclusion"
        ),
        "instance_identities": identities,
        "source_sha256": _source_hashes(project_root),
    }
    return {**semantic, "contract_sha256": _canonical_hash(semantic)}


def prepare(project_root: Path, output_root: Path) -> dict:
    contract = _contract(project_root, output_root)
    path = output_root / CONTRACT_NAME
    if path.is_file():
        if json.loads(path.read_text(encoding="utf-8")) != contract:
            raise AnchoredScreenError("anchored seed-0 contract or inputs changed")
    else:
        if output_root.exists() and any(output_root.iterdir()):
            raise AnchoredScreenError("nonempty output root has no anchored contract")
        pilot._atomic_json(path, contract)
    return {
        "status": "prepared",
        "training_episodes": trainer.DEFAULT_EPISODES,
        "initial_parity_rollouts": len(LAMBDA_GRID),
        "adapted_evaluation_rollouts": 49,
        "contract": str(path.resolve()),
    }


def _require_contract(project_root: Path, output_root: Path) -> dict:
    path = output_root / CONTRACT_NAME
    if not path.is_file():
        raise AnchoredScreenError("run prepare first")
    observed = json.loads(path.read_text(encoding="utf-8"))
    expected = _contract(project_root, output_root)
    if observed != expected:
        raise AnchoredScreenError("anchored seed-0 contract or inputs changed")
    return observed


class _FixedLambdaAgent:
    def __init__(self, agent: AnchoredPreferenceAgent, value: float) -> None:
        self.agent = agent
        self.value = float(value)
        self.config = agent.config

    def reset_episode_state(self):
        self.agent.reset_episode_state()

    def select(self, snapshot, *, training=False, epsilon=0.0):
        return self.agent.select(
            snapshot,
            preference_lambda=self.value,
            training=training,
            epsilon=epsilon,
        )

    def observe_outcome(self, decision, *, next_snapshot, done):
        return self.agent.observe_outcome(
            decision, next_snapshot=next_snapshot, done=done
        )


def _new_initial_agent(
    project_root: Path, arm, device, *, base=None
) -> AnchoredPreferenceAgent:
    base = pilot._fresh_base(arm, device) if base is None else base
    cost_path = project_root / trainer.COST_RELATIVE_PATH
    cost = pilot._load_bound_cost(
        cost_path, arm, device=device, config=base.config
    )
    config = trainer.AnchoredPreferenceConfig(
        feature_dim=(
            3 * base.config.graph_embedding_dim + base.config.action_embedding_dim
        ),
        hidden_dim=base.config.head_hidden_dim,
        lambda_max=max(LAMBDA_GRID),
        gamma_op=base.config.gamma,
        reward_scale=base.config.reward_scale,
        learning_rate=5.0e-5,
        batch_size=64,
        replay_capacity=20_000,
        target_update_every=200,
    )
    return AnchoredPreferenceAgent(
        base, config=config, seed=MODEL_SEED, warm_start_cost=cost, epsilon=0.0
    )


def _terminal(project_root: Path, output_root: Path) -> tuple[dict, str]:
    path = output_root / TRAINING_RELATIVE_PATH / "terminal.pth"
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise AnchoredScreenError("anchored terminal checkpoint is invalid")
    expected = {
        "training_protocol": trainer.TRAINING_PROTOCOL,
        "trainer_schema_version": trainer.TRAINER_SCHEMA_VERSION,
        "checkpoint_role": trainer.TERMINAL_CHECKPOINT_ROLE,
        "trainer_resumable": False,
        "fixed_terminal_checkpoint": True,
        "completed_training_episodes": trainer.DEFAULT_EPISODES,
        "development_only": True,
        "evaluation_panels_opened": False,
    }
    mismatch = {
        key: (payload.get(key), value)
        for key, value in expected.items()
        if payload.get(key) != value
    }
    if mismatch:
        raise AnchoredScreenError(f"terminal checkpoint mismatch: {mismatch}")
    state = payload.get("agent_checkpoint", {}).get("agent_state", {})
    if "replay" in state:
        raise AnchoredScreenError("terminal checkpoint unexpectedly contains replay")
    return dict(payload), _sha256(path)


def _new_terminal_agent(
    project_root: Path, arm, terminal: Mapping, device, *, base=None
):
    base = pilot._fresh_base(arm, device) if base is None else base
    return AnchoredPreferenceAgent.from_checkpoint(
        terminal["agent_checkpoint"],
        base_agent=base,
        expected_base_checkpoint_sha256=arm.checkpoint_sha256,
        expected_base_policy_digest=arm.deployment_policy_digest,
        expected_source_cost_sha256=_sha256(
            project_root / trainer.COST_RELATIVE_PATH
        ),
        seed=MODEL_SEED,
        resumable=False,
    )


def _run(project_root: Path, arm, instance, *, device, value, terminal=None):
    def factory(fresh_base):
        agent = (
            _new_initial_agent(project_root, arm, device, base=fresh_base)
            if terminal is None
            else _new_terminal_agent(
                project_root, arm, terminal, device, base=fresh_base
            )
        )
        agent.set_epsilon(0.0)
        agent.Q_local.requires_grad_(False).eval()
        agent.Q_target.requires_grad_(False).eval()
        return _FixedLambdaAgent(agent, value)

    return pilot._run_raw(
        arm, instance, device=device, wrapper_factory=factory
    )


def _behavior_row(raw: Mapping, instance) -> dict:
    """Keep a compact action-level trace without retaining full frontiers."""

    row = pilot._compact_row(raw, instance)
    action_counts: dict[str, int] = {}
    trace = []
    for decision in raw["decisions"]:
        action = str(decision["selected_action_type"])
        action_counts[action] = action_counts.get(action, 0) + 1
        trace.append(
            {
                "decision_index": int(decision["decision_index"]),
                "decision_epoch": int(decision["decision_epoch"]),
                "selected_key": decision["selected_key"],
                "selected_mode": decision["selected_mode"],
                "selected_action_type": action,
                "duration": int(decision["duration"]),
                "relocations": int(decision["relocations"]),
                "safe_direct_delivery_candidate_available": bool(
                    decision["safe_direct_delivery_candidate_available"]
                ),
                "selected_reconfigure_block_directly_deliverable": bool(
                    decision["selected_reconfigure_block_directly_deliverable"]
                ),
                "delivery_deviations": list(decision["delivery_deviations"]),
            }
        )
    row["behavior_summary"] = {
        "macro_decisions": int(raw["macro_decisions"]),
        "selected_action_counts": action_counts,
        "reconfiguration_decision_epochs": int(
            raw["reconfiguration_decision_epochs"]
        ),
        "reconfiguration_with_direct_delivery_available": int(
            raw["reconfiguration_decision_epochs_with_direct_delivery_available"]
        ),
        "reconfiguration_without_direct_delivery_available": int(
            raw["reconfiguration_decision_epochs_without_direct_delivery_available"]
        ),
        "directly_deliverable_self_reconfiguration_decisions": int(
            raw["directly_deliverable_self_reconfiguration_decision_epochs"]
        ),
    }
    row["action_trace"] = trace
    return row


def parity(project_root: Path, output_root: Path, *, device_name: str) -> dict:
    contract = _require_contract(project_root, output_root)
    path = output_root / PARITY_NAME
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    arm, _latest, sources, q_digest, _records = pilot._authenticate_inputs(project_root)
    historical = _historical(project_root)
    device = pilot._device(device_name)
    instance = sources.instances[85_000]
    initial = _new_initial_agent(project_root, arm, device)
    delta_parameters = initial.Q_local.operational_residual[-1]
    residual_zero = bool(
        torch.count_nonzero(delta_parameters.weight).item() == 0
        and torch.count_nonzero(delta_parameters.bias).item() == 0
    )
    rows = []
    for value in LAMBDA_GRID:
        raw = _run(
            project_root, arm, instance, device=device, value=value, terminal=None
        )
        row = pilot._compact_row(raw, instance)
        reference = historical["rows"][str(value)][0]
        if value == 0.0:
            expected_digest = json.loads(
                (
                    project_root
                    / "results/vcg-v1-1-nested-handling-seed0-85k-development/"
                    "lambda-zero-parity.json"
                ).read_text(encoding="utf-8")
            )["sentinel"]["behavior_digest"]
        else:
            expected_digest = reference["behavior_digest"]
        checks = {
            "behavior_digest_exact": row["behavior_digest"] == expected_digest,
            "dense_return_exact": row["dense_return"] == reference["dense_return"],
            "mae_exact": row["mean_absolute_error"] == reference["mean_absolute_error"],
            "steps_exact": row["steps"] == reference["steps"],
            "rehandles_exact": row["physical_rehandles"] == reference["physical_rehandles"],
        }
        if not all(checks.values()):
            raise AnchoredScreenError(f"initial anchor parity failed at lambda={value}: {checks}")
        rows.append({"lambda": value, "checks": checks, "row": row})
    result = {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "status": "passed",
        "contract_sha256": contract["contract_sha256"],
        "base_q_state_sha256": q_digest,
        "operational_residual_exactly_zero": residual_zero,
        "lambda_zero_direct_delegation": True,
        "positive_lambda_initial_nested_behavior_exact": True,
        "sentinel_instance_seed": 85_000,
        "rollout_count": len(rows),
        "rows": rows,
    }
    if not residual_zero:
        raise AnchoredScreenError("initial operational residual is not zero")
    pilot._atomic_json(path, result)
    return result


def _metrics(rows: Sequence[Mapping]) -> Optional[dict]:
    if len(rows) != len(INSTANCE_SEEDS) or not all(
        row["strict_safe_complete"] for row in rows
    ):
        return None
    total = sum(int(row["physical_rehandles"]) for row in rows)
    return {
        "mean_dense_return": float(fmean(float(row["dense_return"]) for row in rows)),
        "mean_absolute_error": float(
            fmean(float(row["mean_absolute_error"]) for row in rows)
        ),
        "mean_steps": float(fmean(float(row["steps"]) for row in rows)),
        "total_physical_rehandles": int(total),
        "physical_rehandles_per_100": float(
            100.0 * total / (len(rows) * EXPECTED_BLOCKS)
        ),
    }


def evaluate(project_root: Path, output_root: Path, *, device_name: str) -> dict:
    contract = _require_contract(project_root, output_root)
    parity_result = parity(project_root, output_root, device_name=device_name)
    if parity_result["status"] != "passed":
        raise AnchoredScreenError("initial anchor parity has not passed")
    terminal, terminal_sha = _terminal(project_root, output_root)
    path = output_root / LEDGER_NAME
    if path.is_file():
        observed = json.loads(path.read_text(encoding="utf-8"))
        if observed.get("terminal_checkpoint_sha256") != terminal_sha:
            raise AnchoredScreenError("adapted ledger terminal identity changed")
        return observed
    arm, _latest, sources, _q_digest, _records = pilot._authenticate_inputs(project_root)
    historical = _historical(project_root)
    device = pilot._device(device_name)
    sentinel_instance = sources.instances[85_000]
    zero_raw = _run(
        project_root,
        arm,
        sentinel_instance,
        device=device,
        value=0.0,
        terminal=terminal,
    )
    zero_row = _behavior_row(zero_raw, sentinel_instance)
    parity_zero = parity_result["rows"][0]["row"]
    zero_exact = all(
        zero_row[key] == parity_zero[key]
        for key in (
            "behavior_digest", "dense_return", "mean_absolute_error", "steps",
            "physical_rehandles", "delivery_deviations",
        )
    )
    if not zero_exact:
        raise AnchoredScreenError("trained checkpoint lost lambda-zero exactness")
    rows_by_lambda = {"0.0": list(historical["rows"]["0.0"])}
    for row in rows_by_lambda["0.0"]:
        row["execution_reused_after_terminal_sentinel"] = True
    for value in LAMBDA_GRID[1:]:
        rows = []
        for seed in INSTANCE_SEEDS:
            instance = sources.instances[seed]
            raw = _run(
                project_root,
                arm,
                instance,
                device=device,
                value=value,
                terminal=terminal,
            )
            rows.append(_behavior_row(raw, instance))
        rows_by_lambda[str(value)] = rows
    result = {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "status": "complete",
        "contract_sha256": contract["contract_sha256"],
        "terminal_checkpoint_sha256": terminal_sha,
        "lambda_zero_terminal_sentinel_exact": True,
        "lambda_zero_terminal_sentinel": zero_row,
        "new_rollout_count": 49,
        "rows": rows_by_lambda,
    }
    pilot._atomic_json(path, result)
    return result


def _nondominated(metrics: Mapping[float, Mapping]) -> list[float]:
    result = []
    for value, point in metrics.items():
        dominated = False
        for other_value, other in metrics.items():
            if other_value == value:
                continue
            no_worse = (
                other["mean_absolute_error"] <= point["mean_absolute_error"]
                and other["physical_rehandles_per_100"]
                <= point["physical_rehandles_per_100"]
            )
            strict = (
                other["mean_absolute_error"] < point["mean_absolute_error"]
                or other["physical_rehandles_per_100"]
                < point["physical_rehandles_per_100"]
            )
            if no_worse and strict:
                dominated = True
                break
        if not dominated:
            result.append(value)
    return sorted(result)


def analyze(project_root: Path, output_root: Path) -> dict:
    contract = _require_contract(project_root, output_root)
    ledger = json.loads((output_root / LEDGER_NAME).read_text(encoding="utf-8"))
    historical = _historical(project_root)
    adapted_metrics = {}
    nested_metrics = {}
    behavior = {}
    adapted_action_profiles = {}
    all_safe = True
    for value in LAMBDA_GRID:
        key = str(value)
        adapted_rows = ledger["rows"][key]
        nested_rows = historical["rows"][key]
        adapted_metrics[value] = _metrics(adapted_rows)
        nested_metrics[value] = _metrics(nested_rows)
        all_safe &= adapted_metrics[value] is not None
        comparable = [
            (a, n)
            for a, n in zip(adapted_rows, nested_rows)
            if a.get("behavior_digest") is not None and n.get("behavior_digest") is not None
        ]
        behavior[key] = {
            "comparable_pairs": len(comparable),
            "changed_behavior_pairs": sum(
                a["behavior_digest"] != n["behavior_digest"] for a, n in comparable
            ),
        }
        traced_rows = [
            row for row in adapted_rows if isinstance(row.get("behavior_summary"), Mapping)
        ]
        if not traced_rows and value == 0.0:
            traced_rows = [ledger["lambda_zero_terminal_sentinel"]]
        action_counts: dict[str, int] = {}
        for row in traced_rows:
            for action, count in row["behavior_summary"]["selected_action_counts"].items():
                action_counts[action] = action_counts.get(action, 0) + int(count)
        adapted_action_profiles[key] = {
            "traced_rows": len(traced_rows),
            "action_counts": action_counts,
            "macro_decisions": sum(
                int(row["behavior_summary"]["macro_decisions"]) for row in traced_rows
            ),
            "reconfiguration_decisions": sum(
                int(row["behavior_summary"]["reconfiguration_decision_epochs"])
                for row in traced_rows
            ),
            "reconfiguration_with_direct_delivery_available": sum(
                int(
                    row["behavior_summary"][
                        "reconfiguration_with_direct_delivery_available"
                    ]
                )
                for row in traced_rows
            ),
            "reconfiguration_without_direct_delivery_available": sum(
                int(
                    row["behavior_summary"][
                        "reconfiguration_without_direct_delivery_available"
                    ]
                )
                for row in traced_rows
            ),
        }
    if not all_safe:
        adapted_display = None
        nondominated = None
        monotonic = None
    else:
        adapted_display = {str(k): v for k, v in adapted_metrics.items()}
        nondominated = _nondominated(adapted_metrics)
        values = [adapted_metrics[value]["physical_rehandles_per_100"] for value in LAMBDA_GRID]
        monotonic = all(b <= a + 1e-12 for a, b in zip(values, values[1:]))
    lambda_zero_exact = bool(ledger["lambda_zero_terminal_sentinel_exact"])
    high_lambda_no_worse = bool(
        all_safe
        and adapted_metrics[0.2]["physical_rehandles_per_100"]
        <= nested_metrics[0.2]["physical_rehandles_per_100"]
    )
    criteria = {
        "all_60_logical_rows_strict_safe_complete": all_safe,
        "lambda_zero_exact_vcg_v1_1": lambda_zero_exact,
        "lambda_0p2_rehandles_no_worse_than_nested_seed0": high_lambda_no_worse,
        "at_least_three_seed0_nondominated_points": bool(
            nondominated is not None and len(nondominated) >= 3
        ),
        "seed0_rehandles_nonincreasing": bool(monotonic),
    }
    report = {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "status": "complete" if all_safe else "complete_with_method_suppression",
        "scope": "opened_85k_single_seed_behavior_diagnostic",
        "contract_sha256": contract["contract_sha256"],
        "model_seed": MODEL_SEED,
        "lambda_grid": LAMBDA_GRID,
        "adapted_metrics": adapted_display,
        "nested_warm_start_metrics": {str(k): v for k, v in nested_metrics.items()},
        "adapted_nondominated_lambdas": nondominated,
        "adapted_rehandles_nonincreasing": monotonic,
        "behavior_change_from_warm_start": behavior,
        "adapted_action_profiles": adapted_action_profiles,
        "diagnostic_gate": {
            "criteria": criteria,
            "passed": all(criteria.values()),
            "decision": (
                "prepare_matched_seeds_1_and_2"
                if all(criteria.values())
                else "inspect_seed0_behavior_before_more_training"
            ),
        },
        "seeds_1_and_2_run": False,
        "architecture_level_claim_authorized": False,
        "interpretation": (
            "Lambda zero is an exact immutable VCG 1.1 endpoint. Positive-lambda "
            "differences measure learned adaptation away from the authenticated nested "
            "warm start. This one-seed opened-panel result is for behavioral diagnosis "
            "only; it cannot establish seed stability or confirmation performance."
        ),
    }
    path = output_root / REPORT_NAME
    if path.is_file() and json.loads(path.read_text(encoding="utf-8")) != report:
        raise AnchoredScreenError("existing anchored analysis changed")
    if not path.is_file():
        pilot._atomic_json(path, report)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("prepare", "parity", "evaluate", "analyze", "run-analysis")
    )
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _parser().parse_args(argv)
    project_root = args.project_root.resolve()
    output_root = (
        args.output_root.resolve()
        if args.output_root is not None
        else project_root / "results/vcg-v1-1-anchored-preference-seed0-85k-development"
    )
    torch.set_num_threads(1)
    if args.command == "prepare":
        result = prepare(project_root, output_root)
    elif args.command == "parity":
        result = parity(project_root, output_root, device_name=args.device)
    elif args.command == "evaluate":
        result = evaluate(project_root, output_root, device_name=args.device)
    elif args.command == "analyze":
        result = analyze(project_root, output_root)
    else:
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
