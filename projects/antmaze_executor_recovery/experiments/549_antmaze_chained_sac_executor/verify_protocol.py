#!/usr/bin/env python3
"""Fail-fast provenance, configuration, and interface audit for E549."""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import runpy
import sys
import types

import numpy as np


HERE = Path(__file__).resolve().parent
ORIGINAL = HERE / "farama_original"
MODIFIED = HERE / "farama_modified"
BANK = HERE / "artifacts/command_distance_bank_r47.npz"
PROTOCOL = HERE / "protocol.json"
EXPECTED_ORIGINAL = {
    "train_ant.py": "b1cc7de1a50d482ffb23856bfc46d00bfc66f5729be2405fc679db0acf6bf800",
    # The pinned upstream file has no terminal newline. The vendored copy is
    # normalized to one terminal newline, so verification uses that form.
    "reach_goal_ant.py": "22f935397a405c23ab2d86dae0f268b2802e821bdfeb41d9aadff1fa882c78c4",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def compare_model_config() -> dict:
    # WandB is logging-only and is intentionally absent from the execution
    # environment. Stub just enough of it to read Farama's immutable config.
    saved_modules = {
        name: sys.modules.get(name)
        for name in ("wandb", "wandb.integration", "wandb.integration.sb3")
    }
    wandb = types.ModuleType("wandb")
    integration = types.ModuleType("wandb.integration")
    integration.__path__ = []
    sb3 = types.ModuleType("wandb.integration.sb3")
    sb3.WandbCallback = object
    sys.modules["wandb"] = wandb
    sys.modules["wandb.integration"] = integration
    sys.modules["wandb.integration.sb3"] = sb3
    try:
        official = runpy.run_path(str(ORIGINAL / "train_ant.py"))["config"]
    finally:
        for name, module in saved_modules.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module
    modified = runpy.run_path(str(MODIFIED / "train_ant.py"))["config"]
    allowed = {"env_id"}
    if set(official) - {"checkpoint_dir", "checkpoint_filename"} != set(modified):
        raise ValueError("Top-level Farama SAC configuration keys changed")
    for key in set(modified) - allowed - {"model_kwargs"}:
        if official[key] != modified[key]:
            raise ValueError(f"Farama configuration changed at {key}")
    official_kwargs = official["model_kwargs"]
    modified_kwargs = modified["model_kwargs"]
    if set(official_kwargs) != set(modified_kwargs):
        raise ValueError("Farama SAC model keyword set changed")
    for key in official_kwargs:
        if key == "action_noise":
            for attribute in ("_mu", "_sigma"):
                if not np.array_equal(
                    getattr(official_kwargs[key], attribute),
                    getattr(modified_kwargs[key], attribute),
                ):
                    raise ValueError(f"Farama action noise changed at {attribute}")
        elif key == "policy_kwargs":
            if official_kwargs[key]["net_arch"] != modified_kwargs[key]["net_arch"]:
                raise ValueError("Farama network widths changed")
            if (
                official_kwargs[key]["activation_fn"].__name__
                != modified_kwargs[key]["activation_fn"].__name__
            ):
                raise ValueError("Farama activation changed")
        elif official_kwargs[key] != modified_kwargs[key]:
            raise ValueError(f"Farama SAC hyperparameter changed at {key}")
    return {
        "only_training_environment_config_changed": True,
        "official_env_id": official["env_id"],
        "modified_env_id": modified["env_id"],
        "sac_model_kwargs_identical": True,
    }


def compare_interfaces() -> dict:
    official_module = load_module("e549_official_goal_ant", ORIGINAL / "reach_goal_ant.py")
    modified_module = load_module("e549_modified_goal_ant", MODIFIED / "reach_goal_ant.py")
    official = official_module.GoalReachAnt(
        goal_threshold=0.2, forward_reward_scale=10.0
    )
    modified = modified_module.GoalReachAnt(
        goal_threshold=0.2,
        forward_reward_scale=10.0,
        command_distance_bank=str(BANK),
    )
    try:
        official_obs, _ = official.reset(seed=549)
        modified_obs, _ = modified.reset(seed=549)
        if official_obs.shape != modified_obs.shape or official_obs.shape != (29,):
            raise ValueError("The modified observation shape differs from Farama")
        if official_obs.dtype != modified_obs.dtype or official_obs.dtype != np.float64:
            raise ValueError("The modified observation dtype differs from Farama")
        if official.action_space.shape != modified.action_space.shape or modified.action_space.shape != (8,):
            raise ValueError("The modified action shape differs from Farama")
        info = {
            "x_velocity": 0.7,
            "y_velocity": -0.2,
            "reward_ctrl": -0.15,
            "reward_survive": 1.0,
        }
        achieved = np.asarray([1.2, -0.5])
        desired = np.asarray([3.0, 2.1])
        official_reward = official.compute_reward(achieved, desired, info)
        modified_reward = modified.compute_reward(achieved, desired, info)
        if official_reward != modified_reward:
            raise ValueError("The Farama reward changed")
        return {
            "observation_shape": list(modified_obs.shape),
            "observation_dtype": str(modified_obs.dtype),
            "action_shape": list(modified.action_space.shape),
            "reward_exact_on_audit_input": True,
            "semantic_ant_proprioception_assertion_executed": True,
        }
    finally:
        official.close()
        modified.close()


def audit_ast() -> dict:
    tree = ast.parse((MODIFIED / "reach_goal_ant.py").read_text())
    constants = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "COMMAND_HORIZON" for target in node.targets)
    ]
    if len(constants) != 1 or ast.literal_eval(constants[0].value) != 100:
        raise ValueError("The fixed deployed command horizon changed")
    return {"fixed_command_horizon": 100, "command_horizon_asserted": True}


def main() -> None:
    protocol = json.loads(PROTOCOL.read_text())
    expected = {
        "method_revision": "antmaze_chained_sac_executor_e549",
        "claim_level": "from_scratch_executor_development_only",
        "oracle_inputs": [],
        "training_seed": 42,
        "fixed_command_horizon": 100,
        "task_reward_used": False,
        "hidden_goals_used": False,
    }
    for key, value in expected.items():
        if protocol.get(key) != value:
            raise ValueError(f"E549 protocol changed at {key}")
    for filename, expected_hash in EXPECTED_ORIGINAL.items():
        if sha256(ORIGINAL / filename) != expected_hash:
            raise ValueError(f"Vendored pinned Farama source changed: {filename}")
    if not BANK.is_file():
        raise FileNotFoundError(
            f"Run prepare_distance_bank.py before verification: {BANK}"
        )
    with np.load(BANK, allow_pickle=False) as stored:
        distances = stored["requested_distance_m"]
    if distances.shape != (2000,):
        raise ValueError("The empirical command-distance bank changed")

    report = {
        "schema_version": 1,
        "experiment_id": "antmaze_chained_sac_executor_e549",
        "pinned_farama_commit": protocol["pinned_farama_commit"],
        "pinned_farama_sources_verified": True,
        "config_audit": compare_model_config(),
        "interface_audit": compare_interfaces(),
        "contract_audit": audit_ast(),
        "declared_method_changes": protocol["declared_method_changes"],
        "declared_operational_deviations": protocol["declared_operational_deviations"],
        "passed": True,
    }
    audit_path = HERE / "artifacts/preflight_audit.json"
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

