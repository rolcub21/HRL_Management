#!/usr/bin/env python3
"""Matched-instance Track A storage-assignment evaluator."""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import json
import os
import random

import numpy as np
import torch

from example.episode_instance import EpisodeInstance
from example.Options.selector import StorageSelectOption, TinyQ
from example.helper.timing_metrics import summarize_delivery_timing
from example.small_rooms_env import SmallRoomsEnv
from PSLAP.ga_optimizer import GAConfig
from PSLAP.ga_policy import DEFAULT_ROLLING_GA_CONFIG
from PSLAP.kim2020_a3c_spatial import (
    DEPLOYMENT_MODES as KIM2020_DEPLOYMENT_MODES,
    DEPLOYMENT_STOCHASTIC as KIM2020_DEPLOYMENT_STOCHASTIC,
    Kim2020A3CSpatialSource,
)
from PSLAP.reg_selector_v4 import REGV4AssignmentSource
from PSLAP.reg_selector_v5 import REGV5AssignmentSource
from PSLAP.track_a import (
    REGSelectorAssignmentSource,
    TRACK_A_KIM2020_A3C_SPATIAL,
    TRACK_A_METHODS,
    TRACK_A_REG_SELECTOR,
    TRACK_A_REG_SELECTOR_V4,
    TRACK_A_REG_SELECTOR_V5,
    run_track_a_episode,
)


FIELDNAMES = [
    "track",
    "method",
    "information_regime",
    "lambda",
    "mu",
    "episode_seed",
    "ga_seed",
    "instance_id",
    "checkpoint_id",
    "return",
    "success",
    "timeout",
    "completion_fraction",
    "steps",
    "delivery_count",
    "mean_signed_deviation",
    "mean_absolute_error",
    "mean_tardiness",
    "within_target_window_rate",
    "tardy_delivery_rate",
    "obstructive_moves",
    "obstructive_moves_per_delivered_block",
    "illegal_drops",
    "strict_method_success",
    "strict_completion_fraction",
    "strict_return_before_failure",
    "strict_delivery_count",
    "strict_mean_absolute_error",
    "strict_mean_tardiness",
    "strict_tardy_delivery_rate",
    "strict_obstructive_moves",
    "assignment_decision_count",
    "valid_assignment_count",
    "invalid_assignment_count",
    "invalid_assignment_rate",
    "invalid_reasons",
    "infeasible_epoch_count",
    "infeasible_block_counts",
    "retrieval_live_plan_failure_count",
    "inbound_approach_defer_count",
    "exact_recovery_search_count",
    "exact_recovery_fallback_count",
    "exact_recovery_failure_count",
    "exact_recovery_explored_nodes",
    "fallback_count",
    "fallback_contaminated",
    "first_invalid_step",
    "contaminated_steps",
    "contaminated_deliveries",
    "empty_travel_steps",
    "loaded_travel_steps",
    "wait_actions",
    "pickup_actions",
    "putdown_actions",
    "source_setup_seconds",
    "online_assignment_planning_seconds",
    "episode_loop_seconds",
    "ga_predicted_infeasible_events",
    "ga_predicted_obstructive_moves",
    "ga_predicted_route_steps",
    "assignment_decision_audit",
]


def make_env(lam, mu):
    return SmallRoomsEnv(
        choose_storage=False,
        arrival_rate=lam,
        proc_mean=mu,
    )


def _load_or_sample_instance(env, episode_seed, path):
    if path:
        with open(path) as handle:
            instance = EpisodeInstance.from_json(handle.read())
        if instance.seed is not None and instance.seed != episode_seed:
            raise ValueError(
                "episode seed does not match the seed recorded in the instance"
            )
    else:
        instance = env.sample_episode_instance(episode_seed)
    instance.validate_for(env)
    return instance


def format_row(
    result, lam, mu, episode_seed, ga_seed, max_steps, checkpoint_id
):
    operational = summarize_delivery_timing(
        result["delivery_errors"], SmallRoomsEnv.DELIVERY_TARGET_WINDOW
    )
    strict = summarize_delivery_timing(
        result["strict_delivery_errors"], SmallRoomsEnv.DELIVERY_TARGET_WINDOW
    )
    return {
        "track": "A_assignment_isolation",
        "method": result["method"],
        "information_regime": result["information_regime"],
        "lambda": lam,
        "mu": mu,
        "episode_seed": episode_seed,
        "ga_seed": ga_seed,
        "instance_id": result["instance_id"],
        "checkpoint_id": checkpoint_id,
        "return": result["return"],
        "success": result["success"],
        "timeout": int(not result["success"] and result["steps"] >= max_steps),
        "completion_fraction": result["completion_fraction"],
        "steps": result["steps"],
        "delivery_count": result["delivery_count"],
        "mean_signed_deviation": operational["mean_signed_deviation"],
        "mean_absolute_error": operational["mean_absolute_error"],
        "mean_tardiness": operational["mean_tardiness"],
        "within_target_window_rate": operational[
            "within_target_window_rate"
        ],
        "tardy_delivery_rate": operational["tardy_delivery_rate"],
        "obstructive_moves": result["obstructive_moves"],
        "obstructive_moves_per_delivered_block": result[
            "obstructive_moves_per_delivered_block"
        ],
        "illegal_drops": result["illegal_drops"],
        "strict_method_success": result["strict_method_success"],
        "strict_completion_fraction": result["strict_completion_fraction"],
        "strict_return_before_failure": result[
            "strict_return_before_failure"
        ],
        "strict_delivery_count": strict["delivery_count"],
        "strict_mean_absolute_error": strict["mean_absolute_error"],
        "strict_mean_tardiness": strict["mean_tardiness"],
        "strict_tardy_delivery_rate": strict["tardy_delivery_rate"],
        "strict_obstructive_moves": result["strict_obstructive_moves"],
        "assignment_decision_count": result["assignment_decision_count"],
        "valid_assignment_count": result["valid_assignment_count"],
        "invalid_assignment_count": result["invalid_assignment_count"],
        "invalid_assignment_rate": result["invalid_assignment_rate"],
        "invalid_reasons": json.dumps(
            result["invalid_reasons"], sort_keys=True, separators=(",", ":")
        ),
        "infeasible_epoch_count": result["infeasible_epoch_count"],
        "infeasible_block_counts": json.dumps(
            result["infeasible_block_counts"],
            sort_keys=True,
            separators=(",", ":"),
        ),
        "retrieval_live_plan_failure_count": result[
            "retrieval_live_plan_failure_count"
        ],
        "inbound_approach_defer_count": result[
            "inbound_approach_defer_count"
        ],
        "exact_recovery_search_count": result[
            "exact_recovery_search_count"
        ],
        "exact_recovery_fallback_count": result[
            "exact_recovery_fallback_count"
        ],
        "exact_recovery_failure_count": result[
            "exact_recovery_failure_count"
        ],
        "exact_recovery_explored_nodes": result[
            "exact_recovery_explored_nodes"
        ],
        "fallback_count": result["fallback_count"],
        "fallback_contaminated": result["fallback_contaminated"],
        "first_invalid_step": result["first_invalid_step"],
        "contaminated_steps": result["contaminated_steps"],
        "contaminated_deliveries": result["contaminated_deliveries"],
        "empty_travel_steps": result["empty_travel_steps"],
        "loaded_travel_steps": result["loaded_travel_steps"],
        "wait_actions": result["wait_actions"],
        "pickup_actions": result["pickup_actions"],
        "putdown_actions": result["putdown_actions"],
        "source_setup_seconds": result["source_setup_seconds"],
        "online_assignment_planning_seconds": result[
            "assignment_planning_seconds"
        ],
        "episode_loop_seconds": result["episode_loop_seconds"],
        "ga_predicted_infeasible_events": result[
            "ga_predicted_infeasible_events"
        ],
        "ga_predicted_obstructive_moves": result[
            "ga_predicted_obstructive_moves"
        ],
        "ga_predicted_route_steps": result["ga_predicted_route_steps"],
        "assignment_decision_audit": json.dumps(
            result["assignment_decisions_compact"], separators=(",", ":")
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=TRACK_A_METHODS, required=True)
    parser.add_argument("--lambda", dest="lam", type=float, required=True)
    parser.add_argument("--mu", type=float, required=True)
    parser.add_argument("--episode-seed", type=int, required=True)
    parser.add_argument("--ga-seed", type=int, default=0)
    parser.add_argument(
        "--checkpoint",
        help="required checkpoint for a learned assignment source",
    )
    parser.add_argument(
        "--deployment-mode",
        choices=KIM2020_DEPLOYMENT_MODES,
        default=KIM2020_DEPLOYMENT_STOCHASTIC,
        help="Kim spatial realization; ignored by other methods",
    )
    parser.add_argument(
        "--policy-seed",
        type=int,
        help="Kim policy-sampling seed, separate from the episode seed",
    )
    parser.add_argument("--instance")
    parser.add_argument("--save-instance")
    parser.add_argument(
        "--audit-output",
        help="optional JSON file with every full candidate mask and decision",
    )
    parser.add_argument("--max-steps", type=int, default=4000)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    random.seed(args.episode_seed + 10_000)
    np.random.seed(args.episode_seed + 20_000)
    env = make_env(args.lam, args.mu)
    instance = _load_or_sample_instance(
        env, args.episode_seed, args.instance
    )
    if args.save_instance:
        with open(args.save_instance, "w") as handle:
            handle.write(instance.to_json() + "\n")

    assignment_source = None
    checkpoint_id = "not_applicable"
    if args.method in (
        TRACK_A_KIM2020_A3C_SPATIAL,
        TRACK_A_REG_SELECTOR,
        TRACK_A_REG_SELECTOR_V4,
        TRACK_A_REG_SELECTOR_V5,
    ):
        if not args.checkpoint:
            parser.error(f"--checkpoint is required for {args.method}")
        # Resetting the same immutable instance here prepares selector geometry;
        # run_track_a_episode replays the identical instance before execution.
        env.reset(instance=instance)
        payload = torch.load(
            args.checkpoint, map_location="cpu", weights_only=False
        )
        if args.method == TRACK_A_KIM2020_A3C_SPATIAL:
            assignment_source = Kim2020A3CSpatialSource.from_checkpoint(
                env,
                payload,
                learning_enabled=False,
                device="cpu",
                seed=(
                    args.episode_seed
                    if args.policy_seed is None
                    else args.policy_seed
                ),
                policy_seed=args.policy_seed,
                deployment_mode=args.deployment_mode,
            )
        elif args.method == TRACK_A_REG_SELECTOR:
            saved_version = payload.get("selector_feature_version")
            if saved_version != StorageSelectOption.FEATURE_VERSION:
                raise ValueError(
                    "Track A REG selector requires feature version "
                    f"{StorageSelectOption.FEATURE_VERSION}; checkpoint has "
                    f"{saved_version!r}. Retrain rather than using a future-"
                    "leaking or unverified checkpoint."
                )
            weights = payload.get("selector_state_dict")
            if weights is None:
                raise ValueError("checkpoint has no selector_state_dict")
            selector = StorageSelectOption(
                env, gamma=float(payload.get("selector_gamma", 0.99))
            )
            hidden_1 = int(weights["net.0.weight"].shape[0])
            hidden_2 = int(weights["net.3.weight"].shape[0])
            selector.q = TinyQ(
                int(weights["net.0.weight"].shape[1]),
                selector.n_cells,
                hidden_1,
                hidden_2,
            ).to(selector.device)
            selector.q.load_state_dict(weights)
            assignment_source = REGSelectorAssignmentSource(env, selector)
        elif args.method == TRACK_A_REG_SELECTOR_V4:
            assignment_source = REGV4AssignmentSource.from_checkpoint(
                env,
                payload,
                learning_enabled=False,
                device="cpu",
                seed=args.episode_seed,
            )
        else:
            assignment_source = REGV5AssignmentSource.from_checkpoint(
                env,
                payload,
                learning_enabled=False,
                device="cpu",
                seed=args.episode_seed,
            )
        checkpoint_id = os.path.basename(args.checkpoint)

    result = run_track_a_episode(
        env,
        args.method,
        max_steps=args.max_steps,
        episode_instance=instance,
        offline_config=replace(GAConfig(), seed=args.ga_seed),
        rolling_config=replace(
            DEFAULT_ROLLING_GA_CONFIG, seed=args.ga_seed
        ),
        assignment_source=assignment_source,
    )
    row = format_row(
        result,
        args.lam,
        args.mu,
        args.episode_seed,
        args.ga_seed,
        args.max_steps,
        checkpoint_id,
    )
    if args.audit_output:
        audit_payload = {
            "track": "A_assignment_isolation",
            "method": result["method"],
            "information_regime": result["information_regime"],
            "instance_id": result["instance_id"],
            "episode_seed": args.episode_seed,
            "ga_seed": args.ga_seed,
            "decisions": result["assignment_decisions"],
        }
        with open(args.audit_output, "w") as handle:
            json.dump(audit_payload, handle, indent=2, sort_keys=True)
            handle.write("\n")

    new_file = not os.path.exists(args.output)
    if not new_file:
        with open(args.output, newline="") as handle:
            header = next(csv.reader(handle), None)
        if header != FIELDNAMES:
            raise ValueError(
                "Existing Track A CSV uses a different schema; choose a new path"
            )
    with open(args.output, "a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        if new_file:
            writer.writeheader()
        writer.writerow(row)
    printed = dict(row)
    printed["assignment_decision_audit"] = (
        f"{result['assignment_decision_count']} compact records"
    )
    print(printed)


if __name__ == "__main__":
    main()
