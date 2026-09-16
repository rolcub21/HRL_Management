#!/usr/bin/env python3
"""Seed-0 development screen for policy-conditioned VCG handling cost."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import fmean
from typing import Mapping, Optional, Sequence

import torch

import run_vcg_v11_nested_handling_pilot as pilot
import train_vcg_v11_conditioned_handling_iterative as trainer
from vcg_v11_conditioned_handling import (
    ConditionedFutureHandlingNetwork,
    ConditionedHandlingAgent,
)
from vcg_v11_nested_handling import HandlingAugmentedV11Agent


PROTOCOL = "vcg_v1_1_conditioned_handling_seed0_85k_screen_v1"
SCHEMA_VERSION = 1
MODEL_SEED = 0
INSTANCE_SEEDS = tuple(range(85_000, 85_012))
LAMBDA_GRID = (0.0, 0.025, 0.0375, 0.05, 0.075, 0.1, 0.125, 0.175, 0.2)
HELD_OUT_REPORTING_LAMBDAS = (0.0375, 0.075, 0.125, 0.175)
HISTORICAL_LAMBDAS = (0.0, 0.025, 0.05, 0.1, 0.2)
EXPECTED_BLOCKS = 8
CONTRACT_NAME = "conditioned-handling-contract.json"
PARITY_NAME = "initial-warm-start-parity.json"
LEDGER_NAME = "conditioned-handling-evaluation.json"
REPORT_NAME = "conditioned-handling-report.json"
HISTORICAL_RELATIVE_PATH = Path(
    "results/vcg-v1-1-nested-handling-seed0-85k-development/pilot-sweep.json"
)
TRAINING_RELATIVE_PATH = Path("training/seed-0")


class ConditionedScreenError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    if not path.is_file():
        raise ConditionedScreenError(f"missing required artifact: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_hash(value: Mapping) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    ).hexdigest()


def _historical(project_root: Path) -> dict:
    path = project_root / HISTORICAL_RELATIVE_PATH
    result = json.loads(path.read_text(encoding="utf-8"))
    if result.get("status") != "complete":
        raise ConditionedScreenError("nested VCG seed-0 pilot is incomplete")
    for value in HISTORICAL_LAMBDAS:
        rows = result.get("rows", {}).get(str(value))
        if not isinstance(rows, list) or len(rows) != len(INSTANCE_SEEDS):
            raise ConditionedScreenError(f"historical rows missing at lambda={value}")
        if tuple(int(row["instance_seed"]) for row in rows) != INSTANCE_SEEDS:
            raise ConditionedScreenError("historical instance order changed")
    return result


def _source_hashes(project_root: Path) -> dict:
    paths = {
        "screen_runner": Path(__file__).resolve(),
        "conditioned_controller": project_root / "vcg_v11_conditioned_handling.py",
        "conditioned_trainer": (
            project_root / "train_vcg_v11_conditioned_handling_iterative.py"
        ),
    }
    return {name: _sha256(path) for name, path in sorted(paths.items())}


def _contract(project_root: Path, output_root: Path) -> dict:
    arm, _latest, sources, q_digest, _records = pilot._authenticate_inputs(
        project_root
    )
    historical_path = project_root / HISTORICAL_RELATIVE_PATH
    historical = _historical(project_root)
    cost_path = project_root / trainer.COST_RELATIVE_PATH
    schedules = [
        trainer.stratified_lambda_schedule(index)
        for index in range(trainer.ROUNDS)
    ]
    scheduled_values = [
        float(record["behavior_lambda"])
        for schedule in schedules
        for record in schedule
    ]
    if any(value in scheduled_values for value in LAMBDA_GRID):
        raise ConditionedScreenError("a reporting coordinate entered training exactly")
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
        "lambda_interval": [0.0, 0.2],
        "reporting_lambda_grid": list(LAMBDA_GRID),
        "held_out_reporting_lambdas": list(HELD_OUT_REPORTING_LAMBDAS),
        "no_reporting_coordinate_sampled_exactly_during_training": True,
        "instance_seeds": list(INSTANCE_SEEDS),
        "base_checkpoint_sha256": arm.checkpoint_sha256,
        "base_policy_digest": arm.deployment_policy_digest,
        "base_q_state_sha256": q_digest,
        "source_cost_sha256": _sha256(cost_path),
        "historical_nested_sweep_sha256": _sha256(historical_path),
        "historical_nested_protocol": historical["protocol"],
        "training": {
            "output": str((output_root / TRAINING_RELATIVE_PATH).resolve()),
            "rounds": trainer.ROUNDS,
            "episodes_per_round": trainer.EPISODES_PER_ROUND,
            "total_episodes": trainer.TOTAL_EPISODES,
            "continuous_stratified_lambda_sampling": True,
            "policy_frozen_within_collection_round": True,
            "complete_episode_future_rehandle_mc_labels": True,
            "current_rehandle_excluded_from_network_target": True,
            "operational_critic_frozen": True,
            "checkpoint_selection": False,
        },
        "initial_parity": {
            "sentinel_seed": INSTANCE_SEEDS[0],
            "lambda_zero_vcg_v1_1": True,
            "nested_warm_start_lambdas": [0.025, 0.2],
            "rollouts": 3,
        },
        "evaluation": {
            "fixed_terminal_checkpoint": True,
            "conditioned_positive_lambda_rollouts": (
                (len(LAMBDA_GRID) - 1) * len(INSTANCE_SEEDS)
            ),
            "conditioned_lambda_zero_new_sentinel_rollouts": 1,
            "conditioned_lambda_zero_rows_reused_after_sentinel": True,
            "new_nested_interpolation_rollouts": (
                len(HELD_OUT_REPORTING_LAMBDAS) * len(INSTANCE_SEEDS)
            ),
            "complete_case_filtering_allowed": False,
        },
        "architecture": {
            "lambda_zero_direct_vcg_v1_1_delegation": True,
            "frozen_vcg_v1_1_operational_critic": True,
            "policy_conditioned_future_handling_only": True,
            "known_immediate_reconfigure_term": True,
            "nonnegative_future_handling_softplus": True,
            "teacher_policy_queries": False,
        },
        "instance_identities": identities,
        "source_sha256": _source_hashes(project_root),
        "advancement_scope": (
            "seed0 behavior and calibration diagnostic only; matched seeds 1 and 2 "
            "are required before an architecture-level conclusion"
        ),
    }
    return {**semantic, "contract_sha256": _canonical_hash(semantic)}


def prepare(project_root: Path, output_root: Path) -> dict:
    contract = _contract(project_root, output_root)
    path = output_root / CONTRACT_NAME
    if path.is_file():
        if json.loads(path.read_text(encoding="utf-8")) != contract:
            raise ConditionedScreenError("conditioned handling contract changed")
    else:
        if output_root.exists() and any(output_root.iterdir()):
            raise ConditionedScreenError("nonempty output root has no contract")
        pilot._atomic_json(path, contract)
    return {
        "status": "prepared",
        "training_episodes": trainer.TOTAL_EPISODES,
        "training_rounds": trainer.ROUNDS,
        "initial_parity_rollouts": 3,
        "post_training_new_rollouts": 145,
        "contract": str(path.resolve()),
    }


def _require_contract(project_root: Path, output_root: Path) -> dict:
    path = output_root / CONTRACT_NAME
    if not path.is_file():
        raise ConditionedScreenError("run prepare first")
    observed = json.loads(path.read_text(encoding="utf-8"))
    expected = _contract(project_root, output_root)
    if observed != expected:
        raise ConditionedScreenError("contract, source, or parent artifact changed")
    return observed


class _FixedLambdaAgent:
    def __init__(self, agent: ConditionedHandlingAgent, value: float) -> None:
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


def _new_initial_agent(project_root: Path, arm, device, *, base=None):
    base = pilot._fresh_base(arm, device) if base is None else base
    cost_path = project_root / trainer.COST_RELATIVE_PATH
    old_cost = pilot._load_bound_cost(
        cost_path, arm, device=device, config=base.config
    )
    config = trainer.ConditionedHandlingConfig(
        feature_dim=(
            3 * base.config.graph_embedding_dim + base.config.action_embedding_dim
        ),
        hidden_dim=base.config.head_hidden_dim,
        lambda_max=0.2,
        gamma_op=base.config.gamma,
        reward_scale=base.config.reward_scale,
    )
    network = ConditionedFutureHandlingNetwork(config, seed=MODEL_SEED).to(device)
    network.initialize_from_nested_head(old_cost)
    return ConditionedHandlingAgent(
        base, network, config=config, seed=MODEL_SEED, epsilon=0.0
    )


def _terminal(project_root: Path, output_root: Path) -> tuple[dict, str]:
    path = output_root / TRAINING_RELATIVE_PATH / "terminal.pth"
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise ConditionedScreenError("conditioned terminal checkpoint is invalid")
    expected = {
        "training_protocol": trainer.TRAINING_PROTOCOL,
        "trainer_schema_version": trainer.TRAINER_SCHEMA_VERSION,
        "checkpoint_role": trainer.TERMINAL_CHECKPOINT_ROLE,
        "completed_rounds": trainer.ROUNDS,
        "completed_training_episodes": trainer.TOTAL_EPISODES,
        "fixed_terminal_checkpoint": True,
        "development_only": True,
        "evaluation_panels_opened": False,
    }
    mismatch = {
        key: (payload.get(key), value)
        for key, value in expected.items()
        if payload.get(key) != value
    }
    if mismatch:
        raise ConditionedScreenError(f"terminal checkpoint mismatch: {mismatch}")
    return dict(payload), _sha256(path)


def _new_terminal_agent(
    project_root: Path, arm, terminal: Mapping, device, *, base=None
):
    base = pilot._fresh_base(arm, device) if base is None else base
    return ConditionedHandlingAgent.from_checkpoint(
        terminal["agent_checkpoint"],
        base_agent=base,
        expected_base_checkpoint_sha256=arm.checkpoint_sha256,
        expected_base_policy_digest=arm.deployment_policy_digest,
        expected_source_cost_sha256=_sha256(
            project_root / trainer.COST_RELATIVE_PATH
        ),
        seed=MODEL_SEED,
    )


def _run_conditioned(
    project_root: Path, arm, instance, *, device, value: float, terminal=None
):
    def factory(fresh_base):
        agent = (
            _new_initial_agent(project_root, arm, device, base=fresh_base)
            if terminal is None
            else _new_terminal_agent(
                project_root, arm, terminal, device, base=fresh_base
            )
        )
        agent.set_epsilon(0.0)
        return _FixedLambdaAgent(agent, value)

    return pilot._run_raw(arm, instance, device=device, wrapper_factory=factory)


def _run_nested(project_root: Path, arm, instance, *, device, value: float):
    def factory(fresh_base):
        cost = pilot._load_bound_cost(
            project_root / trainer.COST_RELATIVE_PATH,
            arm,
            device=device,
            config=fresh_base.config,
        )
        return HandlingAugmentedV11Agent(
            fresh_base, cost, handling_lambda=float(value)
        )

    return pilot._run_raw(arm, instance, device=device, wrapper_factory=factory)


def _behavior_row(raw: Mapping, instance) -> dict:
    row = pilot._compact_row(raw, instance)
    counts: dict[str, int] = {}
    trace = []
    for decision in raw["decisions"]:
        action = str(decision["selected_action_type"])
        counts[action] = counts.get(action, 0) + 1
        trace.append(
            {
                "decision_index": int(decision["decision_index"]),
                "decision_epoch": int(decision["decision_epoch"]),
                "selected_key": decision["selected_key"],
                "selected_mode": decision["selected_mode"],
                "selected_action_type": action,
                "duration": int(decision["duration"]),
                "relocations": int(decision["relocations"]),
                "direct_delivery_available": bool(
                    decision["safe_direct_delivery_candidate_available"]
                ),
                "selected_reconfigure_block_directly_deliverable": bool(
                    decision["selected_reconfigure_block_directly_deliverable"]
                ),
            }
        )
    row["behavior_summary"] = {
        "macro_decisions": int(raw["macro_decisions"]),
        "selected_action_counts": counts,
        "reconfiguration_decisions": int(raw["reconfiguration_decision_epochs"]),
        "reconfiguration_with_direct_delivery_available": int(
            raw["reconfiguration_decision_epochs_with_direct_delivery_available"]
        ),
        "directly_deliverable_self_reconfigurations": int(
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
    arm, _latest, sources, q_digest, _records = pilot._authenticate_inputs(
        project_root
    )
    historical = _historical(project_root)
    device = pilot._device(device_name)
    instance = sources.instances[INSTANCE_SEEDS[0]]
    initial = _new_initial_agent(project_root, arm, device)
    lambda_column_zero = bool(
        torch.count_nonzero(initial.handling_network.future_head[0].weight[:, -1])
        .detach()
        .cpu()
        .item()
        == 0
    )
    checks_by_lambda = {}
    for value in (0.0, 0.025, 0.2):
        raw = _run_conditioned(
            project_root,
            arm,
            instance,
            device=device,
            value=value,
            terminal=None,
        )
        row = pilot._compact_row(raw, instance)
        reference = historical["rows"][str(value)][0]
        expected_digest = (
            json.loads(
                (
                    project_root
                    / "results/vcg-v1-1-nested-handling-seed0-85k-development/"
                    "lambda-zero-parity.json"
                ).read_text(encoding="utf-8")
            )["sentinel"]["behavior_digest"]
            if value == 0.0
            else reference["behavior_digest"]
        )
        checks = {
            "behavior_digest_exact": row["behavior_digest"]
            == expected_digest,
            "dense_return_exact": row["dense_return"] == reference["dense_return"],
            "mae_exact": row["mean_absolute_error"]
            == reference["mean_absolute_error"],
            "steps_exact": row["steps"] == reference["steps"],
            "rehandles_exact": row["physical_rehandles"]
            == reference["physical_rehandles"],
        }
        if not all(checks.values()):
            raise ConditionedScreenError(
                f"initial warm-start parity failed at lambda={value}: {checks}"
            )
        checks_by_lambda[str(value)] = {"checks": checks, "row": row}
    if not lambda_column_zero:
        raise ConditionedScreenError("warm-start lambda column is not zero")
    result = {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "status": "passed",
        "contract_sha256": contract["contract_sha256"],
        "base_q_state_sha256": q_digest,
        "sentinel_instance_seed": INSTANCE_SEEDS[0],
        "lambda_input_column_exactly_zero": True,
        "lambda_zero_direct_vcg_v1_1_exact": True,
        "positive_lambda_initial_nested_behavior_exact": True,
        "rollout_count": 3,
        "rows": checks_by_lambda,
    }
    pilot._atomic_json(path, result)
    return result


def _metrics(rows: Sequence[Mapping]) -> Optional[dict]:
    if len(rows) != len(INSTANCE_SEEDS) or not all(
        bool(row["strict_safe_complete"]) for row in rows
    ):
        return None
    rehandles = sum(int(row["physical_rehandles"]) for row in rows)
    return {
        "mean_dense_return": float(
            fmean(float(row["dense_return"]) for row in rows)
        ),
        "mean_absolute_error": float(
            fmean(float(row["mean_absolute_error"]) for row in rows)
        ),
        "mean_steps": float(fmean(float(row["steps"]) for row in rows)),
        "total_physical_rehandles": int(rehandles),
        "physical_rehandles_per_100": float(
            100.0 * rehandles / (len(rows) * EXPECTED_BLOCKS)
        ),
    }


def evaluate(project_root: Path, output_root: Path, *, device_name: str) -> dict:
    contract = _require_contract(project_root, output_root)
    parity_result = parity(project_root, output_root, device_name=device_name)
    if parity_result["status"] != "passed":
        raise ConditionedScreenError("initial parity did not pass")
    terminal, terminal_sha = _terminal(project_root, output_root)
    path = output_root / LEDGER_NAME
    if path.is_file():
        observed = json.loads(path.read_text(encoding="utf-8"))
        if observed.get("terminal_checkpoint_sha256") != terminal_sha:
            raise ConditionedScreenError("evaluation terminal identity changed")
        return observed
    arm, _latest, sources, _q_digest, _records = pilot._authenticate_inputs(
        project_root
    )
    historical = _historical(project_root)
    device = pilot._device(device_name)
    sentinel = sources.instances[INSTANCE_SEEDS[0]]
    zero_raw = _run_conditioned(
        project_root,
        arm,
        sentinel,
        device=device,
        value=0.0,
        terminal=terminal,
    )
    zero_row = _behavior_row(zero_raw, sentinel)
    reference_zero = parity_result["rows"]["0.0"]["row"]
    for key in (
        "behavior_digest",
        "dense_return",
        "mean_absolute_error",
        "steps",
        "physical_rehandles",
        "delivery_deviations",
    ):
        if zero_row[key] != reference_zero[key]:
            raise ConditionedScreenError(
                f"terminal checkpoint lost lambda-zero exactness: {key}"
            )

    conditioned_rows = {"0.0": list(historical["rows"]["0.0"])}
    for row in conditioned_rows["0.0"]:
        row["execution_reused_after_terminal_sentinel"] = True
    for index, value in enumerate(LAMBDA_GRID[1:], start=1):
        rows = []
        for seed in INSTANCE_SEEDS:
            print(
                f"[conditioned {index}/{len(LAMBDA_GRID)-1}] "
                f"lambda={value:g} seed={seed}",
                flush=True,
            )
            instance = sources.instances[seed]
            raw = _run_conditioned(
                project_root,
                arm,
                instance,
                device=device,
                value=value,
                terminal=terminal,
            )
            rows.append(_behavior_row(raw, instance))
        conditioned_rows[str(value)] = rows

    nested_rows = {
        str(value): list(historical["rows"][str(value)])
        for value in HISTORICAL_LAMBDAS
    }
    for index, value in enumerate(HELD_OUT_REPORTING_LAMBDAS, start=1):
        rows = []
        for seed in INSTANCE_SEEDS:
            print(
                f"[nested interpolation {index}/{len(HELD_OUT_REPORTING_LAMBDAS)}] "
                f"lambda={value:g} seed={seed}",
                flush=True,
            )
            instance = sources.instances[seed]
            raw = _run_nested(
                project_root, arm, instance, device=device, value=value
            )
            rows.append(_behavior_row(raw, instance))
        nested_rows[str(value)] = rows
    nested_rows = {str(value): nested_rows[str(value)] for value in LAMBDA_GRID}
    result = {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "status": "complete",
        "contract_sha256": contract["contract_sha256"],
        "terminal_checkpoint_sha256": terminal_sha,
        "lambda_zero_terminal_sentinel_exact": True,
        "lambda_zero_terminal_sentinel": zero_row,
        "new_rollout_count": 145,
        "conditioned_rows": conditioned_rows,
        "nested_reference_rows": nested_rows,
    }
    pilot._atomic_json(path, result)
    return result


def _nondominated(points: Mapping[str, Mapping]) -> list[str]:
    result = []
    for name, point in points.items():
        dominated = False
        for other_name, other in points.items():
            if other_name == name:
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
            result.append(name)
    return sorted(result)


def _action_profile(rows: Sequence[Mapping]) -> dict:
    counts: dict[str, int] = {}
    for row in rows:
        summary = row.get("behavior_summary")
        if not isinstance(summary, Mapping):
            continue
        for action, count in summary["selected_action_counts"].items():
            counts[action] = counts.get(action, 0) + int(count)
    return {
        "traced_rows": sum(
            isinstance(row.get("behavior_summary"), Mapping) for row in rows
        ),
        "selected_action_counts": counts,
        "reconfiguration_decisions": sum(
            int(row.get("behavior_summary", {}).get("reconfiguration_decisions", 0))
            for row in rows
        ),
        "reconfiguration_with_direct_delivery_available": sum(
            int(
                row.get("behavior_summary", {}).get(
                    "reconfiguration_with_direct_delivery_available", 0
                )
            )
            for row in rows
        ),
        "directly_deliverable_self_reconfigurations": sum(
            int(
                row.get("behavior_summary", {}).get(
                    "directly_deliverable_self_reconfigurations", 0
                )
            )
            for row in rows
        ),
    }


def analyze(project_root: Path, output_root: Path) -> dict:
    contract = _require_contract(project_root, output_root)
    ledger = json.loads((output_root / LEDGER_NAME).read_text(encoding="utf-8"))
    training = json.loads(
        (output_root / TRAINING_RELATIVE_PATH / "training-summary.json").read_text(
            encoding="utf-8"
        )
    )
    conditioned_metrics = {}
    nested_metrics = {}
    action_profiles = {}
    all_safe = True
    for value in LAMBDA_GRID:
        key = str(value)
        conditioned = _metrics(ledger["conditioned_rows"][key])
        nested = _metrics(ledger["nested_reference_rows"][key])
        conditioned_metrics[key] = conditioned
        nested_metrics[key] = nested
        action_profiles[key] = _action_profile(ledger["conditioned_rows"][key])
        all_safe &= conditioned is not None and nested is not None

    if all_safe:
        conditioned_points = {
            f"conditioned:{key}": metric
            for key, metric in conditioned_metrics.items()
        }
        combined_points = {
            **conditioned_points,
            **{
                f"nested:{key}": metric
                for key, metric in nested_metrics.items()
            },
        }
        conditioned_frontier = _nondominated(conditioned_points)
        combined_frontier = _nondominated(combined_points)
        rehandles = [
            conditioned_metrics[str(value)]["physical_rehandles_per_100"]
            for value in LAMBDA_GRID
        ]
        adjacent_inversions = [
            {
                "lower_lambda": LAMBDA_GRID[index],
                "upper_lambda": LAMBDA_GRID[index + 1],
                "increase_per_100": float(rehandles[index + 1] - rehandles[index]),
            }
            for index in range(len(rehandles) - 1)
            if rehandles[index + 1] > rehandles[index]
        ]
    else:
        conditioned_frontier = None
        combined_frontier = None
        adjacent_inversions = None

    behavior_variation = {}
    for seed_index, seed in enumerate(INSTANCE_SEEDS):
        digests = [
            ledger["conditioned_rows"][str(value)][seed_index]["behavior_digest"]
            for value in LAMBDA_GRID[1:]
        ]
        behavior_variation[str(seed)] = {
            "distinct_behaviors_across_positive_lambda": len(set(digests)),
            "positive_preference_changes_behavior": len(set(digests)) > 1,
        }
    fit_diagnostics = []
    for record in training["round_records"]:
        fit = record["fit"]
        fit_diagnostics.append(
            {
                "round_number": record["round_number"],
                "training_samples": fit["training_samples"],
                "validation_samples": fit["validation_samples"],
                "initial_validation": fit["initial_validation"],
                "final_validation": fit["final_validation"],
                "validation_mae_improved": (
                    fit["final_validation"]["mae"]
                    <= fit["initial_validation"]["mae"]
                ),
                "probe_behavior_changes_from_previous_round": record[
                    "probe_behavior_changes_from_previous_round"
                ],
            }
        )

    high = conditioned_metrics.get("0.2")
    high_reference = nested_metrics.get("0.2")
    lambda_zero_exact = bool(ledger["lambda_zero_terminal_sentinel_exact"])
    criteria = {
        "all_conditioned_and_reference_rows_strict_safe_complete": all_safe,
        "lambda_zero_exact_vcg_v1_1": lambda_zero_exact,
        "lambda_0p2_rehandles_no_worse_than_original_nested": bool(
            all_safe
            and high["physical_rehandles_per_100"]
            <= high_reference["physical_rehandles_per_100"] + 1e-12
        ),
        "at_least_three_conditioned_operating_points_nondominated": bool(
            conditioned_frontier is not None and len(conditioned_frontier) >= 3
        ),
        "no_adjacent_rehandle_inversion_above_5_per_100": bool(
            adjacent_inversions is not None
            and all(item["increase_per_100"] <= 5.0 for item in adjacent_inversions)
        ),
        "preference_changes_behavior_on_at_least_half_the_instances": bool(
            sum(
                item["positive_preference_changes_behavior"]
                for item in behavior_variation.values()
            )
            >= len(INSTANCE_SEEDS) // 2
        ),
    }
    report = {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "status": "complete" if all_safe else "complete_with_metric_suppression",
        "scope": "opened_85k_single_seed_behavior_and_calibration_diagnostic",
        "contract_sha256": contract["contract_sha256"],
        "model_seed": MODEL_SEED,
        "lambda_grid": list(LAMBDA_GRID),
        "held_out_reporting_lambdas": list(HELD_OUT_REPORTING_LAMBDAS),
        "conditioned_metrics": conditioned_metrics if all_safe else None,
        "nested_reference_metrics": nested_metrics if all_safe else None,
        "conditioned_nondominated_points": conditioned_frontier,
        "combined_nondominated_points": combined_frontier,
        "adjacent_rehandle_inversions": adjacent_inversions,
        "behavior_variation": behavior_variation,
        "conditioned_action_profiles": action_profiles,
        "iterative_fit_diagnostics": fit_diagnostics,
        "diagnostic_gate": {
            "criteria": criteria,
            "passed": all(criteria.values()),
            "decision": (
                "prepare_matched_seeds_1_and_2"
                if all(criteria.values())
                else "inspect_seed0_policy_and_calibration_before_more_training"
            ),
        },
        "seeds_1_and_2_run": False,
        "architecture_level_claim_authorized": False,
        "interpretation": (
            "VCG 1.1 supplies the immutable operational evaluator. The only learned "
            "quantity is nonnegative future physical handling under the policy induced "
            "by lambda; the current Reconfigure cost is exact and outside the network. "
            "Held-out coordinates test interpolation of the continuous preference input. "
            "This opened seed-0 panel is diagnostic and does not establish seed stability."
        ),
    }
    path = output_root / REPORT_NAME
    if path.is_file() and json.loads(path.read_text(encoding="utf-8")) != report:
        raise ConditionedScreenError("existing conditioned analysis changed")
    if not path.is_file():
        pilot._atomic_json(path, report)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("prepare", "parity", "evaluate", "analyze", "run-analysis")
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
        else project_root
        / "results/vcg-v1-1-conditioned-handling-seed0-85k-development"
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
