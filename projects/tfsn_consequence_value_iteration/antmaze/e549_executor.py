"""Hash-pinned adapter for the frozen E549 5M AntMaze executor."""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np

from scripts.antmaze.current_tfsn.constants import ACTION_DIM, STATE_DIM


HERE = Path(__file__).resolve().parent
MANAGEMENT_ROOT = HERE.parents[2]
E549_CHECKPOINT = (
    MANAGEMENT_ROOT
    / "projects/antmaze_executor_recovery/results/experiments"
    / "549_antmaze_chained_sac_executor/checkpoints"
    / "GoalAntMazeChained_model_5000000_steps.zip"
)
E549_CHECKPOINT_SHA256 = (
    "3318bf1ea6e30811285a1e92a0beb7824201b27466e18412359a22131b474b6b"
)
E549_TRAINING_STEPS = 5_000_000
E549_TRAINING_SEED = 42
E549_10M_CHECKPOINT = (
    MANAGEMENT_ROOT
    / "projects/antmaze_executor_recovery/results/experiments"
    / "549_antmaze_chained_sac_executor/checkpoints"
    / "GoalAntMazeChained_model_10000000_steps.zip"
)
E549_10M_CHECKPOINT_SHA256 = (
    "d966bd714c5d25edaaf1db89ee9c870065d6d92c09fbf3f2b40f8ca4dd3c65f0"
)
E549_10M_TRAINING_STEPS = 10_000_000


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _inspect_checkpoint(
    path: Path, expected_sha256: str, expected_steps: int,
) -> dict:
    """Verify identity and the actor I/O contract without loading Torch."""
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"missing frozen E549 checkpoint: {path}")
    actual_hash = sha256(path)
    with zipfile.ZipFile(path) as archive:
        metadata = json.loads(archive.read("data"))
        sb3_version = archive.read("_stable_baselines3_version").decode().strip()
    checks = {
        "sha256": actual_hash == expected_sha256,
        "num_timesteps": int(metadata.get("num_timesteps", -1))
        == int(expected_steps),
        "observation_shape": metadata.get("observation_space", {}).get("_shape")
        == [STATE_DIM],
        "action_shape": metadata.get("action_space", {}).get("_shape")
        == [ACTION_DIM],
        "seed": int(metadata.get("seed", -1)) == E549_TRAINING_SEED,
    }
    if not all(checks.values()):
        raise ValueError(f"E549 checkpoint contract failed: {checks}")
    return {
        "path": str(path),
        "sha256": actual_hash,
        "num_timesteps": int(metadata["num_timesteps"]),
        "training_seed": int(metadata["seed"]),
        "observation_shape": metadata["observation_space"]["_shape"],
        "action_shape": metadata["action_space"]["_shape"],
        "stable_baselines3_version": sb3_version,
        "checks": checks,
    }


def inspect_e549_checkpoint(path: Path = E549_CHECKPOINT) -> dict:
    return _inspect_checkpoint(
        path, E549_CHECKPOINT_SHA256, E549_TRAINING_STEPS
    )


def inspect_e549_10m_checkpoint(path: Path = E549_10M_CHECKPOINT) -> dict:
    return _inspect_checkpoint(
        path, E549_10M_CHECKPOINT_SHA256, E549_10M_TRAINING_STEPS
    )


class E549SACExecutor:
    """Use E549 through the same absolute-command interface as TF-SN."""

    def __init__(self, checkpoint: Path = E549_CHECKPOINT, device: str = "cuda"):
        from stable_baselines3 import SAC

        self.audit = inspect_e549_checkpoint(checkpoint)
        self.model = SAC.load(str(checkpoint), device=device)
        if self.model.observation_space.shape != (STATE_DIM,):
            raise ValueError(
                f"unexpected E549 observation space {self.model.observation_space}"
            )
        if self.model.action_space.shape != (ACTION_DIM,):
            raise ValueError(
                f"unexpected E549 action space {self.model.action_space}"
            )

    def action(self, state: np.ndarray, command_xy: np.ndarray) -> np.ndarray:
        state = np.asarray(state, np.float32).reshape(-1)
        command_xy = np.asarray(command_xy, np.float32).reshape(-1)
        if state.shape != (STATE_DIM,) or command_xy.shape != (2,):
            raise ValueError(
                f"bad E549 input: state={state.shape}, command={command_xy.shape}"
            )
        # E549 retained Farama's 29D observation: 27D proprioception followed
        # by the command displacement.  TF-SN stores absolute xy first.
        actor_input = np.concatenate(
            (state[2:], command_xy - state[:2])
        ).astype(np.float32, copy=False)
        action, _ = self.model.predict(actor_input, deterministic=True)
        action = np.asarray(action, np.float32).reshape(-1)
        if action.shape != (ACTION_DIM,) or not np.isfinite(action).all():
            raise ValueError(f"invalid E549 action: {action}")
        return action


class E54910MSACExecutor(E549SACExecutor):
    """The independently hash-pinned E549 10M continuation."""

    def __init__(
        self, checkpoint: Path = E549_10M_CHECKPOINT, device: str = "cuda"
    ):
        from stable_baselines3 import SAC

        self.audit = inspect_e549_10m_checkpoint(checkpoint)
        self.model = SAC.load(str(checkpoint), device=device)
        if self.model.observation_space.shape != (STATE_DIM,):
            raise ValueError(
                f"unexpected E549-10M observation space "
                f"{self.model.observation_space}"
            )
        if self.model.action_space.shape != (ACTION_DIM,):
            raise ValueError(
                f"unexpected E549-10M action space {self.model.action_space}"
            )
