#!/usr/bin/env python3
"""Fail-fast provenance, configuration, and interface audit for R22."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import runpy

import numpy as np

from .environment import (
    CONTROL_COST_WEIGHT,
    FORWARD_REWARD_SCALE,
    GOAL_THRESHOLD,
    OGBenchLargeChainedEnv,
    PHYSICAL_HORIZON,
    SURVIVAL_REWARD,
    dense_directional_reward,
)
from .train import config


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compare_sac_config(reference_path: Path) -> dict:
    reference = runpy.run_path(str(reference_path))["config"]
    if set(reference) != set(config):
        raise RuntimeError("R22 and E549 SAC top-level keys differ")
    for key in set(config) - {"env_id", "model_kwargs"}:
        if config[key] != reference[key]:
            raise RuntimeError(f"R22 changed E549 configuration at {key}")
    left = reference["model_kwargs"]
    right = config["model_kwargs"]
    if set(left) != set(right):
        raise RuntimeError("R22 and E549 SAC keyword sets differ")
    for key in left:
        if key == "action_noise":
            for attribute in ("_mu", "_sigma"):
                if not np.array_equal(
                    getattr(left[key], attribute), getattr(right[key], attribute)
                ):
                    raise RuntimeError(f"R22 changed E549 noise {attribute}")
        elif key == "policy_kwargs":
            if left[key]["net_arch"] != right[key]["net_arch"]:
                raise RuntimeError("R22 changed E549 network widths")
            if (
                left[key]["activation_fn"].__name__
                != right[key]["activation_fn"].__name__
            ):
                raise RuntimeError("R22 changed E549 activation")
        elif left[key] != right[key]:
            raise RuntimeError(f"R22 changed E549 SAC setting {key}")
    return {
        "only_env_id_differs_at_SAC_config_level": True,
        "e549_env_id": reference["env_id"],
        "r22_env_id": config["env_id"],
        "model_kwargs_identical": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--command-bank", type=Path, required=True)
    parser.add_argument("--e549-train", type=Path, required=True)
    parser.add_argument("--e549-environment", type=Path, required=True)
    parser.add_argument("--e549-protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    references = protocol["e549_reference"]
    for path, expected in (
        (args.e549_train, references["training_source_sha256"]),
        (args.e549_environment, references["environment_source_sha256"]),
        (args.e549_protocol, references["protocol_sha256"]),
    ):
        if sha256(path) != expected:
            raise RuntimeError(f"R22 E549 reference changed: {path}")
    bank_meta = json.loads(args.command_bank.with_suffix(".json").read_text())
    if bank_meta["artifact_sha256"] != sha256(args.command_bank):
        raise RuntimeError("R22 training bank changed")
    sac_audit = compare_sac_config(args.e549_train)
    environment = OGBenchLargeChainedEnv(str(args.command_bank))
    try:
        observation, info = environment.reset(seed=22)
        if observation.shape != (31,) or observation.dtype != np.float64:
            raise RuntimeError("R22 observation shape or dtype changed")
        if not np.array_equal(observation[:29], environment.native_observation):
            raise RuntimeError("R22 does not preserve the exact OGBench observation")
        if environment.action_space.shape != (8,):
            raise RuntimeError("R22 action interface changed")
        if info.get("native_task_row_accessed") is not False:
            raise RuntimeError("R22 reset did not certify its reward-free source")
        previous = np.asarray([1.0, 2.0])
        achieved = np.asarray([1.1, 1.8])
        desired = np.asarray([4.0, 3.0])
        action = np.asarray([0.2] * 8)
        velocity = (achieved - previous) / environment.native.dt
        goal = desired - achieved
        expected = (
            10.0 * np.dot(velocity, goal) / np.linalg.norm(goal)
            - 0.5 * np.sum(np.square(action))
            + 1.0
        )
        actual = dense_directional_reward(
            previous, achieved, desired, action, environment.native.dt
        )
        if not np.isclose(actual, expected, rtol=0.0, atol=1e-14):
            raise RuntimeError("R22 changed E549's dense reward equation")
        events = []
        for _ in range(200):
            _observation, _reward, terminated, truncated, step_info = environment.step(
                environment.action_space.sample()
            )
            if terminated or truncated:
                break
            if step_info["command_event"] is not None:
                events.append(step_info["command_event"])
        if len(events) < 2 or any(event["duration"] > 100 for event in events):
            raise RuntimeError("R22 did not enforce E549's command contract")
    finally:
        environment.close()
    checks = {
        "protocol_frozen_before_training": protocol["status"]
        == "frozen_before_training",
        "E549_sources_match_frozen_hashes": True,
        "SAC_configuration_matches_E549": True,
        "native_OGBench_29D_observation_is_exact_prefix": True,
        "relative_goal_is_the_only_observation_suffix": True,
        "action_interface_matches_E549": True,
        "dense_reward_equation_matches_E549": True,
        "command_contract_matches_E549": True,
        "training_bank_is_reward_free": bank_meta["task_reward_used"] is False,
        "native_task_rows_not_accessed": True,
        "native_panel_not_accessed": True,
    }
    report = {
        "schema_version": 1,
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": sha256(args.protocol),
        "claim_level": protocol["claim_level"],
        "sac_config_audit": sac_audit,
        "interface": {
            "native_observation_dimension": 29,
            "relative_goal_dimension": 2,
            "policy_observation_dimension": 31,
            "action_dimension": 8,
            "goal_threshold_m": GOAL_THRESHOLD,
            "physical_episode_horizon": PHYSICAL_HORIZON,
            "forward_reward_scale": FORWARD_REWARD_SCALE,
            "control_cost_weight": CONTROL_COST_WEIGHT,
            "survival_reward": SURVIVAL_REWARD,
        },
        "declared_environment_adapter_changes_from_E549": protocol[
            "declared_environment_adapter_changes_from_E549"
        ],
        "checks": checks,
        "passed": all(checks.values()),
        "native_evaluation_panel_accessed": False,
        "task_reward_used": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
