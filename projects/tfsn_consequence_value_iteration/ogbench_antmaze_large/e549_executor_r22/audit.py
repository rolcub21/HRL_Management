#!/usr/bin/env python3
"""Evaluate an R22 checkpoint on the immutable R21 adaptive-reach bank."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from stable_baselines3 import SAC

from .. import audit_executor as base
from .. import audit_executor_adaptive_reach as r21
from .environment import OGBenchLargeChainedEnv
from .train import sha256


def evaluate(args: argparse.Namespace) -> dict:
    protocol = json.loads(args.protocol.read_text())
    development = json.loads(args.development_protocol.read_text())
    summary = json.loads(args.training_summary.read_text())
    r21_meta = json.loads(args.r21_bank_meta.read_text())
    if summary["protocol_id"] != protocol["protocol_id"]:
        raise RuntimeError("R22 checkpoint belongs to another protocol")
    checkpoint = Path(summary["checkpoint"])
    if sha256(checkpoint) != summary["checkpoint_sha256"]:
        raise RuntimeError("R22 checkpoint changed")
    if base.sha256(args.r21_bank) != r21_meta["bank_sha256"]:
        raise RuntimeError("frozen R21 bank changed")
    model = SAC.load(str(checkpoint), device=args.device)
    if model.observation_space.shape != (31,) or model.action_space.shape != (8,):
        raise RuntimeError("R22 checkpoint has another interface")
    with np.load(args.r21_bank, allow_pickle=False) as stored:
        bank = {key: np.asarray(stored[key]) for key in stored.files}
    standardized = r21.standardized_sources(bank)
    tree = cKDTree(standardized)
    with np.load(args.validation, allow_pickle=False) as data:
        validation_observation = np.asarray(data["observations"], dtype=np.float32)
    command_steps = int(protocol["training"]["fixed_command_horizon"])
    commands_per_chain = int(protocol["training"]["physical_episode_horizon"] // command_steps)
    evaluator_radius = float(development["benchmark"]["success_radius_m"])
    environment = OGBenchLargeChainedEnv(
        str(args.training_bank), switch_on_contact=False
    )
    rows: list[dict] = []
    executed_steps = 0
    try:
        for chain, reset_index in enumerate(bank["reset_index"]):
            reset_xy = validation_observation[int(reset_index), :2]
            observation, _info = environment.reset(
                seed=42 + chain, options={"reset_xy": reset_xy}
            )
            for command_index in range(commands_per_chain):
                native_start = environment.native_observation
                match_distance, selected = tree.query(
                    r21.standardize_query(native_start, bank), k=1, eps=0.0
                )
                selected = int(selected)
                observation = environment.set_external_goal(bank["target_xy"][selected])
                initial_xy = native_start[:2].copy()
                initial_distance = float(
                    np.linalg.norm(initial_xy - bank["target_xy"][selected])
                )
                recorded_distance = float(
                    np.linalg.norm(
                        bank["source_observation"][selected, :2]
                        - bank["target_xy"][selected]
                    )
                )
                best_distance = initial_distance
                minimum_uprightness = environment.torso_uprightness
                endpoint_height = environment.torso_height
                for step in range(command_steps):
                    action, _state = model.predict(observation, deterministic=True)
                    observation, _reward, terminated, truncated, _step_info = (
                        environment.step(action)
                    )
                    executed_steps += 1
                    native = environment.native_observation
                    best_distance = min(
                        best_distance,
                        float(np.linalg.norm(native[:2] - bank["target_xy"][selected])),
                    )
                    minimum_uprightness = min(
                        minimum_uprightness, environment.torso_uprightness
                    )
                    endpoint_height = environment.torso_height
                    if (terminated or truncated) and not (
                        command_index == commands_per_chain - 1
                        and step == command_steps - 1
                    ):
                        raise RuntimeError("R22 audit ended before the fixed R21 contract")
                native = environment.native_observation
                endpoint_error = float(
                    np.linalg.norm(native[:2] - bank["target_xy"][selected])
                )
                endpoint_uprightness = environment.torso_uprightness
                denominator = max(initial_distance, np.finfo(np.float32).eps)
                rows.append(
                    {
                        "chain": chain,
                        "command_index": command_index,
                        "selected_source_validation_index": int(
                            bank["source_index"][selected]
                        ),
                        "selected_target_validation_index": int(
                            bank["target_index"][selected]
                        ),
                        "source_match_standardized_distance": float(match_distance),
                        "recorded_command_distance_m": recorded_distance,
                        "initial_distance_m": initial_distance,
                        "endpoint_error_m": endpoint_error,
                        "best_error_m": best_distance,
                        "displacement_m": float(np.linalg.norm(native[:2] - initial_xy)),
                        "best_progress_fraction": float(
                            (initial_distance - best_distance) / denominator
                        ),
                        "positive_progress": bool(best_distance < initial_distance),
                        "contact": bool(best_distance <= evaluator_radius),
                        "minimum_uprightness": minimum_uprightness,
                        "endpoint_uprightness": endpoint_uprightness,
                        "endpoint_torso_height": endpoint_height,
                        "inverted_during_command": bool(minimum_uprightness <= 0.0),
                        "inverted_at_endpoint": bool(endpoint_uprightness <= 0.0),
                    }
                )
            print(
                f"R22 checkpoint={model.num_timesteps} chain={chain + 1}/{len(bank['reset_index'])}",
                flush=True,
            )
    finally:
        environment.close()
    inversion = base.chain_inversion_summary(rows)
    overall = base.aggregate(rows)
    maximum_inversion = float(protocol["gate"]["maximum_chain_inversion_rate"])
    minimum_contact = float(protocol["gate"]["minimum_contact_rate_at_evaluator_radius"])
    inversion_pass = inversion["chain_inversion_rate"] <= maximum_inversion
    contact_pass = overall["contact_rate_at_evaluator_radius"] >= minimum_contact
    checks = {
        "checkpoint_matches_training_summary": True,
        "R21_bank_hash_matches": True,
        "all_R21_chains_evaluated": len(bank["reset_index"]) == r21_meta["chains"],
        "fixed_R21_command_contract_executed": executed_steps
        == len(rows) * command_steps,
        "native_task_rows_not_accessed": True,
        "task_reward_not_used_by_auditor": True,
    }
    payload = {
        "schema_version": 1,
        "protocol_id": protocol["protocol_id"],
        "claim_level": protocol["claim_level"],
        "scope": "R22_executor_only_on_frozen_R21_bank_no_TF_SN_or_native_task_verdict",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": summary["checkpoint_sha256"],
        "checkpoint_environment_steps": int(model.num_timesteps),
        "R21_bank": str(args.r21_bank.resolve()),
        "R21_bank_sha256": r21_meta["bank_sha256"],
        "chains": int(len(bank["reset_index"])),
        "commands_per_chain": commands_per_chain,
        "primitive_steps": executed_steps,
        "overall": overall,
        "chain_inversion": inversion,
        "gate": {
            "maximum_chain_inversion_rate": maximum_inversion,
            "minimum_contact_rate_at_evaluator_radius": minimum_contact,
            "chain_inversion_pass": inversion_pass,
            "contact_pass": contact_pass,
            "executor_accepted": bool(inversion_pass and contact_pass),
            "stopping_rule": protocol["gate"]["stopping_rule"],
        },
        "rows": rows,
        "checks": checks,
        "passed": all(checks.values()),
        "native_evaluation_panel_accessed": False,
        "task_reward_used": False,
        "task_gradient_updates": 0,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--development-protocol", type=Path, required=True)
    parser.add_argument("--training-summary", type=Path, required=True)
    parser.add_argument("--training-bank", type=Path, required=True)
    parser.add_argument("--r21-bank", type=Path, required=True)
    parser.add_argument("--r21-bank-meta", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    evaluate(parser.parse_args())


if __name__ == "__main__":
    main()
