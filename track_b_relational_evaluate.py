#!/usr/bin/env python3
"""Evaluate a relational residual scheduler checkpoint on matched instances."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from example.Options.selector_v5 import StorageSelectOptionV5
from example.controller_observation import OnlineManifestTimingObservationEncoder
from example.controller_options import build_controller_options
from example.episode_instance import EpisodeInstance
from example.small_rooms_env import SmallRoomsEnv
from PSLAP.checkpoint_identity import selector_deployment_digest
from relational_scheduler import (
    RELATIONAL_ACTION_INTERFACE,
    RELATIONAL_POLICY_REALIZATIONS,
    RelationalResidualSchedulerAgent,
    RelationalSchedulerConfig,
    validate_relational_checkpoint_metadata,
)
from train_relational_track_b import (
    RELATIONAL_CHECKPOINT_SCHEMA_VERSION,
    json_safe,
    resolve_device,
    run_episode,
    seed_everything,
    summarize_runs,
)


METHOD = "relational_residual_reg_v5_frozen"


def load_instance(args, env, seed):
    if args.instance is not None:
        path = args.instance
    elif args.instances_dir is not None:
        path = args.instances_dir / f"seed-{seed}.json"
    else:
        return env.sample_episode_instance(seed)
    instance = EpisodeInstance.from_json(path.read_text())
    if instance.seed is not None and instance.seed != seed:
        raise ValueError(
            f"instance seed mismatch: expected {seed}, found {instance.seed}"
        )
    instance.validate_for(env)
    return instance


def build_from_checkpoint(args, payload, device):
    if payload.get("relational_checkpoint_schema_version") != (
        RELATIONAL_CHECKPOINT_SCHEMA_VERSION
    ):
        raise ValueError("unsupported relational checkpoint schema")
    if (payload.get("training_lambda"), payload.get("training_mu")) != (
        args.lam,
        args.mu,
    ):
        raise ValueError(
            "relational checkpoint regime does not match evaluation regime"
        )
    selector_payload = payload.get("selector_checkpoint")
    if not isinstance(selector_payload, dict):
        raise ValueError("relational checkpoint does not embed frozen REG-v5")
    if selector_deployment_digest(selector_payload) != payload.get(
        "selector_deployment_digest"
    ):
        raise ValueError("embedded selector deployment digest mismatch")

    env = SmallRoomsEnv(
        choose_storage=False, arrival_rate=args.lam, proc_mean=args.mu
    )
    selector = StorageSelectOptionV5.from_checkpoint(
        env,
        selector_payload,
        device=device,
        seed=int(payload["training_seed"]),
        learning_enabled=False,
    )
    build_controller_options(
        env,
        selector,
        controller_action_interface=RELATIONAL_ACTION_INTERFACE,
        max_defer_steps=int(payload.get("max_defer_steps", 10)),
    )
    initial = env.sample_episode_instance(args.eval_seeds[0])
    env.reset(instance=initial)
    encoder = OnlineManifestTimingObservationEncoder(env)
    validate_relational_checkpoint_metadata(payload, env, encoder)
    config = RelationalSchedulerConfig.from_dict(payload["relational_config"])
    agent = RelationalResidualSchedulerAgent(
        env,
        encoder,
        config=config,
        seed=int(payload["controller_initialization_seed"]),
        device=device,
        gamma=float(payload["gamma"]),
        learning_rate=float(payload.get("learning_rate", 5e-5)),
        batch_size=int(payload.get("batch_size", 128)),
        buffer_size=max(
            int(payload.get("buffer_size", 100_000)),
            int(payload.get("batch_size", 128)),
        ),
        update_every=int(payload.get("update_every", 100)),
        target_tau=float(payload.get("target_tau", 1e-3)),
        grad_clip=float(payload.get("grad_clip", 5.0)),
        reward_clip=float(payload.get("reward_clip", 100.0)),
        epsilon=0.0,
    )
    agent.Q_local.load_state_dict(payload["relational_q_state_dict"])
    agent.Q_target.load_state_dict(
        payload.get(
            "relational_target_state_dict", payload["relational_q_state_dict"]
        )
    )
    agent.Q_local.eval()
    agent.Q_target.eval()
    agent.set_policy_realization(args.policy)
    return env, selector, agent


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--lambda", dest="lam", type=float, required=True)
    parser.add_argument("--mu", type=float, required=True)
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
    parser.add_argument(
        "--policy",
        choices=RELATIONAL_POLICY_REALIZATIONS,
        default="residual_map",
    )
    parser.add_argument(
        "--include-decision-audit", action="store_true"
    )
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda"), default="auto"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.instance is not None and len(args.eval_seeds) != 1:
        parser.error("--instance requires exactly one evaluation seed")
    if args.max_steps <= 0:
        parser.error("max-steps must be positive")
    return args


def main():
    args = parse_args()
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    device = resolve_device(args.device)
    env, selector, agent = build_from_checkpoint(args, payload, device)
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
            include_decisions=args.include_decision_audit,
        )
        result.update(
            {
                "method": METHOD,
                "track": "B",
                "eval_seed": seed,
                "instance_id": instance.instance_id,
                "policy_realization": args.policy,
            }
        )
        runs.append(result)
        if args.save_instances_dir is not None:
            args.save_instances_dir.mkdir(parents=True, exist_ok=True)
            destination = args.save_instances_dir / f"seed-{seed}.json"
            destination.write_text(instance.to_json() + "\n")

    output = json_safe(
        {
            "protocol": {
                "track": "B_complete_system",
                "method": METHOD,
                "lambda": args.lam,
                "mu": args.mu,
                "eval_seeds": args.eval_seeds,
                "max_steps": args.max_steps,
                "policy_realization": args.policy,
                "checkpoint": str(args.checkpoint.resolve()),
                "checkpoint_episode": payload.get(
                    "completed_training_episodes"
                ),
                "instance_pairing_key": "instance_id",
                "information_regime": "online_arrived_only",
            },
            "summary": summarize_runs(runs),
            "runs": runs,
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    print(json.dumps(output["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
