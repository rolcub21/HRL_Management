#!/usr/bin/env python3
"""Strict evaluation for fully learned reserved-macro checkpoints only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from example.episode_instance import EpisodeInstance
from example.small_rooms_env import SmallRoomsEnv
from example.yard_geometry import geometry_metadata
from fully_learned_hierarchy import (
    FULLY_LEARNED_ACTION_INTERFACE,
    FULLY_LEARNED_CHECKPOINT_SCHEMA_VERSION,
    validate_fully_learned_checkpoint_metadata,
)
from train_fully_learned_track_b import (
    CURRICULUM_CONTRACT,
    FAILURE_CONTRACT,
    FAILURE_PENALTY_CONTRACT,
    FULLY_LEARNED_TRAINER_VERSION,
    METHOD,
    MODEL_SELECTION_CONTRACT,
    RAW_MACRO_RETURN_CONTRACT,
    TRUNCATION_CONTRACT,
    build_stack_from_payload,
    json_safe,
    resolve_device,
    run_episode,
    seed_everything,
    summarize_runs,
)


def load_instance(args, env, seed: int) -> EpisodeInstance:
    if args.instance is not None:
        path = args.instance
    elif args.instances_dir is not None:
        path = args.instances_dir / f"seed-{seed}.json"
    else:
        return env.sample_episode_instance(seed)
    instance = EpisodeInstance.from_json(path.read_text())
    if instance.seed is not None and int(instance.seed) != int(seed):
        raise ValueError(
            f"instance seed mismatch: expected {seed}, found {instance.seed}"
        )
    instance.validate_for(env)
    return instance


def validate_deployment_payload(payload: dict) -> None:
    schema = payload.get("fully_learned_checkpoint_schema_version")
    if schema != FULLY_LEARNED_CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(
            "unsupported fully learned checkpoint schema: "
            f"{schema!r}; expected {FULLY_LEARNED_CHECKPOINT_SCHEMA_VERSION!r}. "
            "Historical gated/relational checkpoints are not compatible."
        )
    validate_fully_learned_checkpoint_metadata(payload)
    if payload.get("fully_learned_trainer_version") != FULLY_LEARNED_TRAINER_VERSION:
        raise ValueError("checkpoint trainer curriculum is incompatible")
    if payload.get("curriculum_contract") != CURRICULUM_CONTRACT:
        raise ValueError("checkpoint curriculum contract is incompatible")
    if payload.get("checkpoint_kind") != "deployment_best":
        raise ValueError("strict evaluation requires deployment_best, not latest")
    if payload.get("resumable") is not False:
        raise ValueError("deployment checkpoint is incorrectly marked resumable")
    if payload.get("controller_action_interface") != FULLY_LEARNED_ACTION_INTERFACE:
        raise ValueError("checkpoint does not use the reserved-cell v2 interface")
    if payload.get("macro_return_contract") != RAW_MACRO_RETURN_CONTRACT:
        raise ValueError("checkpoint macro-return semantics are incompatible")
    if payload.get("truncation_contract") != TRUNCATION_CONTRACT:
        raise ValueError("checkpoint truncation semantics are incompatible")
    if payload.get("failure_contract") != FAILURE_CONTRACT:
        raise ValueError("checkpoint strict-failure semantics are incompatible")
    if payload.get("failure_penalty_contract") != FAILURE_PENALTY_CONTRACT:
        raise ValueError("checkpoint boundary-penalty semantics are incompatible")
    if payload.get("model_selection_contract") != MODEL_SELECTION_CONTRACT:
        raise ValueError("checkpoint model-selection semantics are incompatible")
    if payload.get("selector_frozen_independent_replay_disabled") is not True:
        raise ValueError("checkpoint does not prove isolated REG teacher semantics")
    if payload.get("completed_joint_episodes", 0) <= 0:
        raise ValueError(
            "deployment evaluation requires a checkpoint trained in joint phase"
        )
    expected_joint = {
        "training_phase": "joint",
        "teacher_action_mixture": 0.0,
        "teacher_coefficient": 0.0,
        "accept_policy_lock": False,
        "spatial_trainable": True,
        "frozen_reg_runtime_policy_query": False,
        "common_continuation": True,
        "teacher_prior_application": "behavior_selection_only_v1",
        "td_teacher_prior": False,
        "assignment_source_independent_replay": False,
    }
    mismatches = {
        key: {"expected": value, "found": payload.get(key)}
        for key, value in expected_joint.items()
        if payload.get(key) != value
    }
    agent_state = payload.get("agent_checkpoint_state")
    if not isinstance(agent_state, dict):
        mismatches["agent_checkpoint_state"] = "missing"
    else:
        if agent_state.get("training_phase") != "joint":
            mismatches["agent_training_phase"] = agent_state.get(
                "training_phase"
            )
        forbidden = set(agent_state).intersection(
            {"optimizer", "replay", "rng_state"}
        )
        if forbidden:
            mismatches["deployment_state"] = {
                "unexpected_resumable_fields": sorted(forbidden)
            }
    config = payload.get("fully_learned_config", {})
    failure_penalties = (
        payload.get("failure_penalty"),
        payload.get("failure_boundary_penalty"),
        config.get("failure_penalty") if isinstance(config, dict) else None,
    )
    if (
        any(value is None for value in failure_penalties)
        or len(set(failure_penalties)) != 1
    ):
        mismatches["failure_penalty"] = failure_penalties
    if mismatches:
        raise ValueError(f"checkpoint is not a joint deployment artifact: {mismatches}")


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--lambda", dest="lam", type=float, required=True)
    parser.add_argument("--mu", type=float, required=True)
    parser.add_argument("--grid-rows", type=int, default=10)
    parser.add_argument("--grid-cols", type=int, default=10)
    parser.add_argument("--exit-width", type=int)
    parser.add_argument("--number-blocks", type=int, default=40)
    parser.add_argument(
        "--generalization-axis",
        action="append",
        choices=("load", "grid_size", "exit_width", "block_count"),
        default=[],
        help=(
            "Explicitly authorize a deployment-only domain shift. May be "
            "repeated; resume and warm-start checks remain exact."
        ),
    )
    parser.add_argument("--eval-seeds", type=int, nargs="+", required=True)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--instance", type=Path)
    source.add_argument("--instances-dir", type=Path)
    parser.add_argument("--save-instances-dir", type=Path)
    parser.add_argument("--max-steps", type=int, default=4000)
    parser.add_argument(
        "--target-window",
        type=float,
        default=SmallRoomsEnv.DELIVERY_TARGET_WINDOW,
    )
    parser.add_argument("--include-decision-audit", action="store_true")
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda"), default="auto"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.instance is not None and len(args.eval_seeds) != 1:
        parser.error("--instance requires exactly one evaluation seed")
    if not args.eval_seeds:
        parser.error("at least one evaluation seed is required")
    if args.max_steps <= 0:
        parser.error("max-steps must be positive")
    if args.number_blocks <= 0:
        parser.error("number-blocks must be positive")
    args.generalization_axis = tuple(dict.fromkeys(args.generalization_axis))
    return args


def main(argv=None):
    args = parse_args(argv)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError("checkpoint payload must be a mapping")
    validate_deployment_payload(payload)
    permitted = set(args.generalization_axis)
    load_shift = bool(
        payload.get("training_lambda") != args.lam
        or payload.get("training_mu") != args.mu
    )
    if load_shift and "load" not in permitted:
        raise ValueError(
            "checkpoint load does not match evaluation load; explicitly pass "
            "--generalization-axis load for deployment-only transfer"
        )
    expected_geometry = payload.get("geometry", {})
    requested_by_axis = {
        "grid_size": {
            "grid_rows": args.grid_rows,
            "grid_cols": args.grid_cols,
        },
        "exit_width": {"requested_exit_width": args.exit_width},
        "block_count": {"block_count": args.number_blocks},
    }
    mismatched = {}
    for axis, requested in requested_by_axis.items():
        for key, value in requested.items():
            if expected_geometry.get(key) != value and axis not in permitted:
                mismatched[key] = {
                    "saved": expected_geometry.get(key),
                    "requested": value,
                    "required_generalization_axis": axis,
                }
    if mismatched:
        raise ValueError(f"checkpoint geometry mismatch: {mismatched!r}")

    device = resolve_device(args.device)
    env, selector, agent = build_stack_from_payload(
        args, payload, device=device, for_evaluation=True
    )
    target_geometry = geometry_metadata(
        env, requested_exit_width=args.exit_width
    )
    observed_axes = []
    if load_shift:
        observed_axes.append("load")
    if (
        expected_geometry.get("grid_rows") != args.grid_rows
        or expected_geometry.get("grid_cols") != args.grid_cols
    ):
        observed_axes.append("grid_size")
    if expected_geometry.get("requested_exit_width") != args.exit_width:
        observed_axes.append("exit_width")
    if expected_geometry.get("block_count") != args.number_blocks:
        observed_axes.append("block_count")
    runs = []
    for seed in args.eval_seeds:
        seed_everything(seed)
        instance = load_instance(args, env, seed)
        result = run_episode(
            agent,
            selector,
            env,
            instance,
            max_steps=args.max_steps,
            target_window=args.target_window,
            training=False,
            phase="joint",
            include_decisions=args.include_decision_audit,
        )
        result.update(
            {
                "method": METHOD,
                "track": "B",
                "eval_seed": int(seed),
                "instance_id": instance.instance_id,
                "schedule_id": instance.schedule_id,
                "checkpoint": str(args.checkpoint.resolve()),
                "checkpoint_episode": payload.get("completed_training_episodes"),
            }
        )
        runs.append(result)
        if args.save_instances_dir is not None:
            args.save_instances_dir.mkdir(parents=True, exist_ok=True)
            destination = args.save_instances_dir / f"seed-{seed}.json"
            destination.write_text(instance.to_json() + "\n")

    summary = summarize_runs(runs)
    if summary["reservation_integrity_rate"] != 1.0:
        raise RuntimeError(
            "fully learned evaluation violated exact reservation integrity"
        )
    if summary["selector_independent_replay_isolation_rate"] != 1.0:
        raise RuntimeError(
            "frozen REG teacher mutated or retained independent replay state"
        )
    if (
        summary["total_invalid_assignments"] != 0
        or summary["total_fallbacks"] != 0
        or summary["total_reservation_invalidations"] != 0
    ):
        raise RuntimeError(
            "fully learned evaluation violated strict no-fallback assignment"
        )
    output = json_safe(
        {
            "protocol": {
                "track": "B_complete_system",
                "method": METHOD,
                "lambda": args.lam,
                "mu": args.mu,
                "eval_seeds": args.eval_seeds,
                "max_steps": args.max_steps,
                "checkpoint": str(args.checkpoint.resolve()),
                "checkpoint_schema": payload[
                    "fully_learned_checkpoint_schema_version"
                ],
                "checkpoint_kind": payload["checkpoint_kind"],
                "instance_pairing_key": "instance_id",
                "information_regime": "online_arrived_only",
                "assignment_commitment": "decision_epoch_reserved",
                "strict_no_fallback": True,
                "evaluation_contract": (
                    "explicit_deployment_domain_shift_v1"
                    if observed_axes
                    else "exact_training_domain_v1"
                ),
                "authorized_generalization_axes": list(
                    args.generalization_axis
                ),
                "observed_generalization_axes": observed_axes,
                "source_domain": {
                    "lambda": payload.get("training_lambda"),
                    "mu": payload.get("training_mu"),
                    "geometry": expected_geometry,
                },
                "target_domain": {
                    "lambda": args.lam,
                    "mu": args.mu,
                    "geometry": target_geometry,
                },
            },
            "summary": summary,
            "runs": runs,
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    print(json.dumps(output["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
