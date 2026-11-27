#!/usr/bin/env python3
"""Train R22 from scratch with E549's literal SAC configuration."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import time

import gymnasium as gym
import gymnasium_robotics
import stable_baselines3
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise
import torch
import torch.nn as nn

from .environment import OGBenchLargeChainedEnv


CHECKPOINT_NAME = "GoalOGBenchAntMazeLargeChained_model"
# Copied literally from E549; verify.py compares every field before training.
config = {
    "policy_type": "MlpPolicy",
    "total_timesteps": 30_000_000,
    "env_id": "OGBenchLargeChainedR22-v0",
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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def replay_path(checkpoint: Path) -> Path:
    stem = checkpoint.name.removesuffix(".zip")
    marker = f"{CHECKPOINT_NAME}_"
    if not stem.startswith(marker) or not stem.endswith("_steps"):
        raise ValueError(f"Unrecognized R22 checkpoint: {checkpoint}")
    steps = stem[len(marker) : -len("_steps")]
    return checkpoint.with_name(
        f"{CHECKPOINT_NAME}_replay_buffer_{steps}_steps.pkl"
    )


def make_environment(command_bank: str):
    return Monitor(OGBenchLargeChainedEnv(command_bank=command_bank))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--command-bank", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-timesteps", type=int, required=True)
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    if args.target_timesteps not in protocol["training"][
        "checkpoints_environment_steps"
    ]:
        raise ValueError("target timestep is not a frozen R22 gate")
    bank = args.command_bank.resolve()
    output = args.output.resolve()
    checkpoints = output / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    environment = make_vec_env(
        make_environment,
        n_envs=int(config["n_envs"]),
        env_kwargs={"command_bank": str(bank)},
        seed=int(config["model_kwargs"]["seed"]),
    )
    started = time.time()
    resumed_from = None
    try:
        if args.resume_checkpoint is None:
            if any(checkpoints.glob(f"{CHECKPOINT_NAME}_*_steps.zip")):
                raise FileExistsError(
                    "R22 checkpoints already exist; explicit resume is required"
                )
            model = SAC(
                config["policy_type"],
                environment,
                tensorboard_log=str(output / "tensorboard"),
                device=args.device,
                **config["model_kwargs"],
            )
        else:
            checkpoint = args.resume_checkpoint.resolve()
            replay = replay_path(checkpoint)
            if not checkpoint.is_file() or not replay.is_file():
                raise FileNotFoundError(f"Missing checkpoint/replay pair: {checkpoint}")
            resumed_from = str(checkpoint)
            model = SAC.load(
                str(checkpoint),
                env=environment,
                tensorboard_log=str(output / "tensorboard"),
                device=args.device,
            )
            model.load_replay_buffer(str(replay))
        remaining = int(args.target_timesteps) - int(model.num_timesteps)
        if remaining <= 0:
            raise ValueError("checkpoint has already reached the requested gate")
        callback = CheckpointCallback(
            save_freq=1_000_000 // int(config["n_envs"]),
            save_path=str(checkpoints),
            name_prefix=CHECKPOINT_NAME,
            save_replay_buffer=True,
        )
        model.learn(
            total_timesteps=remaining,
            callback=[callback],
            reset_num_timesteps=False,
            tb_log_name="r22_e549_style_from_scratch",
        )
        checkpoint = checkpoints / f"{CHECKPOINT_NAME}_{model.num_timesteps}_steps.zip"
        replay = replay_path(checkpoint)
        if not checkpoint.is_file():
            model.save(str(checkpoint))
        if not replay.is_file():
            model.save_replay_buffer(str(replay))
        summary = {
            "schema_version": 1,
            "protocol_id": protocol["protocol_id"],
            "protocol_sha256": sha256(args.protocol),
            "claim_level": protocol["claim_level"],
            "training_initialization": "scratch",
            "invocation_resumed_existing_scratch_run": resumed_from is not None,
            "resumed_from": resumed_from,
            "primitive_training_steps_total": int(model.num_timesteps),
            "primitive_training_steps_this_invocation": remaining,
            "primitive_training_steps_count_toward_method_budget": True,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256(checkpoint),
            "replay_buffer": str(replay),
            "replay_buffer_sha256": sha256(replay),
            "command_bank": str(bank),
            "command_bank_sha256": sha256(bank),
            "training_seed": int(config["model_kwargs"]["seed"]),
            "n_envs": int(config["n_envs"]),
            "observation_dimension": 31,
            "action_dimension": 8,
            "fixed_command_horizon": 100,
            "physical_episode_horizon": 1000,
            "algorithm": "Stable_Baselines3_SAC",
            "stable_baselines3_version": stable_baselines3.__version__,
            "torch_version": torch.__version__,
            "gymnasium_version": gym.__version__,
            "gymnasium_robotics_version": gymnasium_robotics.__version__,
            "platform": platform.platform(),
            "wall_clock_seconds_this_invocation": time.time() - started,
            "executor_training_reward_used": True,
            "task_reward_used": False,
            "native_task_rows_accessed": False,
            "hidden_goals_used": False,
        }
        summary_path = output / f"training_summary_{model.num_timesteps}_steps.json"
        summary_path.write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary, indent=2))
    finally:
        environment.close()


if __name__ == "__main__":
    main()
