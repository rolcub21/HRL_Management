#!/usr/bin/env python3
"""Static, runtime, and dataset checks for OGBench AntMaze Large development."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PROTOCOL = HERE / "DEVELOPMENT_PROTOCOL.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_protocol() -> dict:
    payload = json.loads(PROTOCOL.read_text())
    if payload["protocol_id"] != "tfsn_r74_ogbench_antmaze_large_development_v1":
        raise RuntimeError("unexpected AntMaze Large development protocol")
    return payload


def static_checks(protocol: dict) -> dict:
    method = ROOT / protocol["method_reference"]
    panel = ROOT / protocol["panel_reference"]
    checks = {
        "method_reference_exists": method.is_file(),
        "method_reference_frozen": method.is_file()
        and sha256(method) == protocol["method_reference_sha256"],
        "panel_reference_exists": panel.is_file(),
        "panel_reference_frozen": panel.is_file()
        and sha256(panel) == protocol["panel_reference_sha256"],
        "no_declared_method_deviation": protocol["declared_deviations"] == [],
        "no_oracle_inputs": protocol["oracle_inputs"] == [],
        "native_panel_unopened": protocol["native_evaluation_panel_accessed"] is False,
        "success_radius_evaluator_only": protocol["benchmark"]["success_radius_role"]
        == "evaluator_only",
        "horizon_is_derived": protocol["frozen_method_instantiation"]["episode_commands"]
        * protocol["frozen_method_instantiation"]["command_duration_primitive_steps"]
        == protocol["benchmark"]["primitive_episode_steps"],
        "no_environment_specific_planner_constants": protocol[
            "frozen_method_instantiation"
        ]["environment_specific_planner_constants"]
        == [],
        "executor_is_shared": protocol["executor"]["use_after_freeze"]
        == "identical_checkpoint_for_TF_SN_SGM_and_direct_executor",
    }
    return checks


def runtime_checks(
    protocol: dict, vendor: Path, require_gpu: bool
) -> tuple[dict, dict]:
    import gymnasium
    import jax
    import ogbench

    env = ogbench.make_env_and_datasets(
        protocol["benchmark"]["dataset"], env_only=True
    )
    unwrapped = env.unwrapped
    devices = jax.devices()
    checks = {
        "ogbench_source_is_pinned_vendor": Path(ogbench.__file__).resolve().is_relative_to(
            vendor.resolve()
        ),
        "environment_id_matches": env.spec.id == protocol["benchmark"]["environment"],
        "maze_type_is_large": unwrapped._maze_type == "large",
        "native_observation_dimension_matches": env.observation_space.shape
        == (protocol["benchmark"]["observation_dimension"],),
        "native_action_dimension_matches": env.action_space.shape
        == (protocol["benchmark"]["action_dimension"],),
        "native_horizon_matches": env.spec.max_episode_steps
        == protocol["benchmark"]["primitive_episode_steps"],
        "success_radius_matches_evaluator_record": float(unwrapped._goal_tol)
        == protocol["benchmark"]["success_radius_m"],
        "xy_projection_is_native_qpos_xy": tuple(unwrapped.get_xy().shape) == (2,),
    }
    if require_gpu:
        checks["jax_gpu_available_for_training"] = any(
            device.platform == "gpu" for device in devices
        )
    details = {
        "ogbench_module": str(Path(ogbench.__file__).resolve()),
        "gymnasium_version": gymnasium.__version__,
        "jax_devices": [str(device) for device in devices],
        "gpu_required_by_this_check": require_gpu,
        "environment_id": env.spec.id,
        "observation_shape": list(env.observation_space.shape),
        "action_shape": list(env.action_space.shape),
        "max_episode_steps": env.spec.max_episode_steps,
        "native_task_reset_performed": False,
        "native_task_goal_read": False,
    }
    env.close()
    return checks, details


def dataset_checks(protocol: dict, dataset_dir: Path) -> tuple[dict, dict]:
    name = protocol["benchmark"]["dataset"]
    train_path = dataset_dir / f"{name}.npz"
    val_path = dataset_dir / f"{name}-val.npz"
    checks = {
        "train_dataset_exists": train_path.is_file(),
        "validation_dataset_exists": val_path.is_file(),
    }
    details: dict[str, object] = {}
    if all(checks.values()):
        for label, path in (("train", train_path), ("validation", val_path)):
            with np.load(path, mmap_mode="r") as data:
                keys = sorted(data.files)
                observations = data["observations"]
                actions = data["actions"]
                terminals = data["terminals"]
                details[label] = {
                    "path": str(path.resolve()),
                    "sha256": sha256(path),
                    "keys": keys,
                    "rows": int(observations.shape[0]),
                    "observation_dimension": int(observations.shape[1]),
                    "action_dimension": int(actions.shape[1]),
                    "terminal_rows": int(np.count_nonzero(terminals)),
                }
                checks[f"{label}_required_arrays"] = {
                    "observations",
                    "actions",
                    "terminals",
                }.issubset(keys)
                checks[f"{label}_observation_dimension"] = (
                    observations.shape[1]
                    == protocol["benchmark"]["observation_dimension"]
                )
                checks[f"{label}_action_dimension"] = (
                    actions.shape[1] == protocol["benchmark"]["action_dimension"]
                )
                checks[f"{label}_has_no_reward_array"] = "rewards" not in keys
    return checks, details


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("verify", "runtime", "dataset", "all"))
    parser.add_argument("--vendor", type=Path)
    parser.add_argument("--dataset-dir", type=Path)
    parser.add_argument("--require-gpu", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    protocol = load_protocol()
    checks = static_checks(protocol)
    details: dict[str, object] = {"protocol": str(PROTOCOL)}
    if args.mode in ("runtime", "all"):
        if args.vendor is None:
            parser.error("runtime checks require --vendor")
        runtime, runtime_details = runtime_checks(
            protocol, args.vendor, require_gpu=args.require_gpu
        )
        checks.update(runtime)
        details["runtime"] = runtime_details
    if args.mode in ("dataset", "all"):
        if args.dataset_dir is None:
            parser.error("dataset checks require --dataset-dir")
        dataset, dataset_details = dataset_checks(protocol, args.dataset_dir)
        checks.update(dataset)
        details["dataset"] = dataset_details

    payload = {
        "schema_version": 1,
        "protocol_id": protocol["protocol_id"],
        "mode": args.mode,
        "claim_level": protocol["claim_level"],
        "checks": checks,
        "details": details,
        "native_evaluation_panel_accessed": False,
        "task_reward_used": False,
        "passed": all(checks.values()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not payload["passed"]:
        raise RuntimeError(f"AntMaze Large preflight failed: {checks}")


if __name__ == "__main__":
    main()
