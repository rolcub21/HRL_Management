"""Train the E549 executor from scratch with Farama's SAC configuration."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import time

import gymnasium as gym
import gymnasium_robotics
import stable_baselines3
import torch
import torch.nn as nn
from gymnasium.envs.registration import register
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise


HERE = Path(__file__).resolve().parent
EXPERIMENT = HERE.parent
DEFAULT_OUTPUT = EXPERIMENT.parents[1] / "results/experiments/549_antmaze_chained_sac_executor"

project_name = "GoalAntMazeChained"
checkpoint_filename = f"{project_name}_model"

# Apart from env_id, these are copied literally from pinned Farama train_ant.py.
config = {
    "policy_type": "MlpPolicy",
    "total_timesteps": 30_000_000,
    "env_id": "GoalReachAntMazeChained-v0",
    "goal_threshold": 0.2,
    "forward_reward_scale": 10.0,
    "n_envs": 16,
    "max_episode_steps": 1000,
    "model_kwargs": {
        "seed": 42,
        "verbose": 1,
        "batch_size": 256,
        "buffer_size": 1000000,
        "ent_coef": "auto_0.1",
        "gamma": 0.991,
        "gradient_steps": 1,
        "learning_rate": 0.0003,
        "learning_starts": 10000,
        "policy_kwargs": {"net_arch": [256] * 4, "activation_fn": nn.ReLU},
        "tau": 0.03,
        "train_freq": 1,
        "use_sde": False,
        "action_noise": NormalActionNoise(mean=[0.0], sigma=[0.3]),
    },
}


def register_environment() -> None:
    if config["env_id"] not in gym.registry:
        register(
            id=config["env_id"],
            entry_point="reach_goal_ant:GoalReachAnt",
            max_episode_steps=config["max_episode_steps"],
        )


def make_env(command_distance_bank: str):
    """Construct the environment without changing Farama's training wrappers."""
    env = gym.make(
        config["env_id"],
        render_mode=None,
        forward_reward_scale=config["forward_reward_scale"],
        goal_threshold=config["goal_threshold"],
        command_distance_bank=command_distance_bank,
    )
    return Monitor(env)


def replay_path_for(checkpoint: Path) -> Path:
    stem = checkpoint.name.removesuffix(".zip")
    marker = f"{checkpoint_filename}_"
    if not stem.startswith(marker) or not stem.endswith("_steps"):
        raise ValueError(f"Unrecognized E549 checkpoint name: {checkpoint}")
    steps = stem[len(marker): -len("_steps")]
    return checkpoint.with_name(f"{checkpoint_filename}_replay_buffer_{steps}_steps.pkl")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--command-distance-bank", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--target-timesteps", type=int, default=config["total_timesteps"])
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if args.target_timesteps <= 0:
        raise ValueError("target timesteps must be positive")
    bank = args.command_distance_bank.resolve()
    if not bank.is_file():
        raise FileNotFoundError(bank)
    output = args.output.resolve()
    checkpoints = output / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    register_environment()

    env = make_vec_env(
        make_env,
        n_envs=config["n_envs"],
        env_kwargs={"command_distance_bank": str(bank)},
        seed=int(config["model_kwargs"]["seed"]),
    )
    started = time.time()
    resumed_from = None
    try:
        if args.resume_checkpoint is None:
            if any(checkpoints.glob(f"{checkpoint_filename}_*_steps.zip")):
                raise FileExistsError(
                    "E549 checkpoints already exist; use --resume-checkpoint explicitly"
                )
            model = SAC(
                config["policy_type"],
                env,
                tensorboard_log=str(output / "tensorboard"),
                device=args.device,
                **config["model_kwargs"],
            )
        else:
            checkpoint = args.resume_checkpoint.resolve()
            resumed_from = str(checkpoint)
            replay = replay_path_for(checkpoint)
            if not checkpoint.is_file() or not replay.is_file():
                raise FileNotFoundError(f"Missing checkpoint/replay pair: {checkpoint}, {replay}")
            model = SAC.load(
                str(checkpoint),
                env=env,
                tensorboard_log=str(output / "tensorboard"),
                device=args.device,
            )
            model.load_replay_buffer(str(replay))

        remaining = int(args.target_timesteps) - int(model.num_timesteps)
        if remaining <= 0:
            raise ValueError(
                f"Checkpoint already has {model.num_timesteps} steps, target is "
                f"{args.target_timesteps}"
            )
        # 0.5M is the requested first gate; 1M is therefore also preserved.
        checkpoint_callback = CheckpointCallback(
            save_freq=500_000 // config["n_envs"],
            save_path=str(checkpoints),
            name_prefix=checkpoint_filename,
            save_replay_buffer=True,
        )
        model.learn(
            total_timesteps=remaining,
            callback=[checkpoint_callback],
            reset_num_timesteps=False,
            tb_log_name="e549_from_scratch",
        )
        checkpoint = checkpoints / f"{checkpoint_filename}_{model.num_timesteps}_steps.zip"
        replay = replay_path_for(checkpoint)
        if not checkpoint.is_file() or not replay.is_file():
            raise RuntimeError("The expected model/replay checkpoint pair was not saved")
        summary = {
            "schema_version": 1,
            "experiment_id": "antmaze_chained_sac_executor_e549",
            "method_revision": "antmaze_chained_sac_executor_e549",
            "claim_level": "from_scratch_executor_development_only",
            "declared_method_changes": [
                "AntMaze_Medium_Diverse_GR-v4_geometry_and_resets",
                "contact_or_exact_100_step_chained_command_handoffs",
                "reward_free_R47_empirical_command_distance_distribution",
            ],
            "oracle_inputs": [],
            "training_initialization": "scratch",
            "invocation_resumed_existing_scratch_run": resumed_from is not None,
            "resumed_from": resumed_from,
            "primitive_training_steps_total": int(model.num_timesteps),
            "primitive_training_steps_this_invocation": int(remaining),
            "primitive_training_steps_count_toward_method_budget": True,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256(checkpoint),
            "replay_buffer": str(replay),
            "replay_buffer_sha256": sha256(replay),
            "command_distance_bank": str(bank),
            "command_distance_bank_sha256": sha256(bank),
            "training_seed": int(config["model_kwargs"]["seed"]),
            "n_envs": int(config["n_envs"]),
            "fixed_command_horizon": 100,
            "physical_episode_horizon": int(config["max_episode_steps"]),
            "goal_threshold_m": float(config["goal_threshold"]),
            "forward_reward_scale": float(config["forward_reward_scale"]),
            "algorithm": "Stable_Baselines3_SAC",
            "stable_baselines3_version": stable_baselines3.__version__,
            "torch_version": torch.__version__,
            "gymnasium_version": gym.__version__,
            "gymnasium_robotics_version": gymnasium_robotics.__version__,
            "platform": platform.platform(),
            "wall_clock_seconds_this_invocation": time.time() - started,
            "executor_training_reward_used": True,
            "task_reward_used": False,
            "hidden_goals_used": False,
        }
        summary_path = output / f"training_summary_{model.num_timesteps}_steps.json"
        summary_path.write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary, indent=2))
    finally:
        env.close()


if __name__ == "__main__":
    main()

