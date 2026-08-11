"""Cardinality-invariant REG-v4 assignment selector for Track A.

V4 is deliberately separate from the historical flat-cell v3 selector.  It is
trained and evaluated under the same neutral Track A executor and exact shared
candidate mask.  All observable blocks are encoded as a set, while one shared
network scores every admissible candidate cell.
"""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
import math
import random
from typing import Optional

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from PSLAP.dynamic_yard import (
    BlockView,
    Cell,
    YardSnapshot,
    find_min_obstruction_route,
    shortest_clear_path,
)


FEATURE_VERSION = 4
CHECKPOINT_SCHEMA_VERSION = 1
ARCHITECTURE_NAME = "reg_v4_deepset_candidate_scorer"
BLOCK_FEATURE_DIM = 15
GLOBAL_FEATURE_DIM = 8
CANDIDATE_FEATURE_DIM = 12


@dataclass(frozen=True)
class REGV4Config:
    block_embedding_dim: int = 64
    candidate_embedding_dim: int = 64
    context_dim: int = 128
    learning_rate: float = 3e-4
    weight_decay: float = 1e-5
    replay_size: int = 100_000
    batch_size: int = 128
    min_replay_size: int = 256
    updates_per_episode: int = 10
    gamma: float = 0.99
    grad_clip: float = 5.0
    huber_delta: float = 1.0
    fail_reward: float = -50.0
    epsilon_start: float = 0.90
    epsilon_end: float = 0.05
    epsilon_warmup_assignments: int = 300
    epsilon_decay_assignments: int = 3_000

    def __post_init__(self):
        positive = (
            "block_embedding_dim",
            "candidate_embedding_dim",
            "context_dim",
            "replay_size",
            "batch_size",
            "min_replay_size",
            "updates_per_episode",
        )
        for name in positive:
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.min_replay_size < self.batch_size:
            raise ValueError("min_replay_size must be at least batch_size")
        if not 0.0 < self.gamma <= 1.0:
            raise ValueError("gamma must be in (0, 1]")
        if self.epsilon_decay_assignments <= 0:
            raise ValueError("epsilon_decay_assignments must be positive")


class RunningMoments:
    """Serializable Welford moments used only to scale regression targets."""

    def __init__(self):
        self.count = 0
        self.mean = 0.0
        self.m2 = 0.0

    def update(self, value: float) -> None:
        self.count += 1
        delta = float(value) - self.mean
        self.mean += delta / self.count
        self.m2 += delta * (float(value) - self.mean)

    @property
    def std(self) -> float:
        if self.count < 2:
            return 1.0
        return max(math.sqrt(self.m2 / (self.count - 1)), 1.0)

    def normalize(self, values: torch.Tensor) -> torch.Tensor:
        return (values - self.mean) / self.std

    def state_dict(self) -> dict:
        return {"count": self.count, "mean": self.mean, "m2": self.m2}

    def load_state_dict(self, state: dict) -> None:
        self.count = int(state["count"])
        self.mean = float(state["mean"])
        self.m2 = float(state["m2"])


class REGV4Network(nn.Module):
    """Permutation-invariant block context plus a shared candidate scorer."""

    def __init__(self, config: REGV4Config):
        super().__init__()
        block_dim = config.block_embedding_dim
        candidate_dim = config.candidate_embedding_dim
        context_dim = config.context_dim
        self.block_encoder = nn.Sequential(
            nn.Linear(BLOCK_FEATURE_DIM, block_dim),
            nn.LayerNorm(block_dim),
            nn.ReLU(),
            nn.Linear(block_dim, block_dim),
            nn.ReLU(),
        )
        self.global_encoder = nn.Sequential(
            nn.Linear(GLOBAL_FEATURE_DIM, block_dim),
            nn.ReLU(),
        )
        self.context_encoder = nn.Sequential(
            nn.Linear(block_dim * 4, context_dim),
            nn.LayerNorm(context_dim),
            nn.ReLU(),
            nn.Linear(context_dim, context_dim),
            nn.ReLU(),
        )
        self.candidate_encoder = nn.Sequential(
            nn.Linear(CANDIDATE_FEATURE_DIM, candidate_dim),
            nn.LayerNorm(candidate_dim),
            nn.ReLU(),
            nn.Linear(candidate_dim, candidate_dim),
            nn.ReLU(),
        )
        self.scorer = nn.Sequential(
            nn.Linear(context_dim + candidate_dim, context_dim),
            nn.ReLU(),
            nn.Linear(context_dim, max(32, context_dim // 2)),
            nn.ReLU(),
            nn.Linear(max(32, context_dim // 2), 1),
        )

    def forward(
        self,
        block_features: torch.Tensor,
        block_mask: torch.Tensor,
        current_block_features: torch.Tensor,
        global_features: torch.Tensor,
        candidate_features: torch.Tensor,
    ) -> torch.Tensor:
        """Return one score per candidate.

        Shapes are ``blocks=(B,N,D_b)``, ``mask=(B,N)``, current/global with
        batch dimension, and ``candidates=(B,M,D_c)``.
        """

        encoded = self.block_encoder(block_features)
        mask = block_mask.unsqueeze(-1).to(encoded.dtype)
        count = mask.sum(dim=1).clamp_min(1.0)
        mean_pool = (encoded * mask).sum(dim=1) / count
        masked = encoded.masked_fill(mask == 0, torch.finfo(encoded.dtype).min)
        max_pool = masked.max(dim=1).values
        no_blocks = block_mask.sum(dim=1) == 0
        if bool(no_blocks.any()):
            max_pool = torch.where(
                no_blocks.unsqueeze(-1), torch.zeros_like(max_pool), max_pool
            )
        current = self.block_encoder(current_block_features)
        global_context = self.global_encoder(global_features)
        context = self.context_encoder(
            torch.cat((mean_pool, max_pool, current, global_context), dim=-1)
        )
        candidates = self.candidate_encoder(candidate_features)
        expanded = context.unsqueeze(1).expand(-1, candidates.shape[1], -1)
        return self.scorer(torch.cat((expanded, candidates), dim=-1)).squeeze(-1)


@dataclass(frozen=True)
class REGV4Observation:
    block_features: np.ndarray
    block_mask: np.ndarray
    current_block_features: np.ndarray
    global_features: np.ndarray
    candidate_features: np.ndarray


@dataclass(frozen=True)
class REGV4Transition:
    block_features: np.ndarray
    block_mask: np.ndarray
    current_block_features: np.ndarray
    global_features: np.ndarray
    chosen_candidate_features: np.ndarray
    event_return: float


def _normalise_cell(cell: Optional[Cell], rows: int, cols: int) -> tuple[float, float]:
    if cell is None:
        return 0.0, 0.0
    return (
        float(cell[0]) / max(1, rows - 1),
        float(cell[1]) / max(1, cols - 1),
    )


def _block_feature(env, item, current_label: str) -> np.ndarray:
    arrived = item.position is not None
    is_current = item.label == current_label
    if item.delivered:
        status = (0.0, 0.0, 0.0, 0.0, 1.0)
    elif item.carrying:
        status = (0.0, 0.0, 0.0, 1.0, 0.0)
    elif item.stored:
        status = (0.0, 0.0, 1.0, 0.0, 0.0)
    elif arrived:
        status = (0.0, 1.0, 0.0, 0.0, 0.0)
    else:
        status = (1.0, 0.0, 0.0, 0.0, 0.0)
    position = item.position if arrived else None
    pos_row, pos_col = _normalise_cell(position, env.grid_rows, env.grid_cols)
    # Assignment/duration become observable only after arrival. This guards the
    # online information contract even if a future environment pre-populates them.
    assigned = arrived and item.storage_location is not None
    assigned_row, assigned_col = _normalise_cell(
        item.storage_location if assigned else None,
        env.grid_rows,
        env.grid_cols,
    )
    remaining = 0.0
    age = 0.0
    if arrived and not item.delivered:
        remaining = float(item.get_remaining_storage_time()) / max(1.0, env.MAX_T)
        age = max(0.0, env.time_steps - float(item.arrival_step)) / max(
            1.0, len(env.blocks) * env.MAX_T
        )
    exit_distance = 0.0
    if position is not None:
        exit_distance = min(
            env.manhattan_distance(position, exit_cell)
            for exit_cell in env.exit_cells
        ) / max(1.0, env.grid_rows + env.grid_cols)
    return np.asarray(
        (
            *status,
            float(is_current),
            float(arrived),
            pos_row,
            pos_col,
            float(assigned),
            assigned_row,
            assigned_col,
            float(np.clip(remaining, 0.0, 1.0)),
            float(np.clip(age, 0.0, 1.0)),
            float(np.clip(exit_distance, 0.0, 1.0)),
        ),
        dtype=np.float32,
    )


def build_observation(
    env,
    yard: YardSnapshot,
    block: BlockView,
    source: Cell,
    valid_candidates: tuple[Cell, ...],
) -> REGV4Observation:
    """Build an online-safe, cardinality-invariant Track A observation."""

    if not valid_candidates:
        raise ValueError("REG-v4 observation requires at least one candidate")
    blocks = np.stack(
        [_block_feature(env, item, block.label) for item in env.blocks], axis=0
    )
    current_index = next(
        index for index, item in enumerate(env.blocks) if item.label == block.label
    )
    current = blocks[current_index].copy()
    block_mask = np.ones(len(env.blocks), dtype=np.float32)
    count = max(1, len(env.blocks))
    arrived = sum(item.position is not None for item in env.blocks)
    active_stored = sum(item.stored and not item.delivered for item in env.blocks)
    delivered = sum(item.delivered for item in env.blocks)
    waiting = sum(
        item.position in (env.waiting_cell, env.pickup_cell)
        and not item.stored
        and not item.delivered
        for item in env.blocks
    )
    global_features = np.asarray(
        (
            min(1.0, env.time_steps / max(1.0, count * env.MAX_T)),
            arrived / count,
            active_stored / count,
            delivered / count,
            waiting / count,
            len(yard.blocks) / max(1, len(yard.storage_cells)),
            env.arrival_rate / (1.0 + env.arrival_rate),
            min(1.0, env.proc_mean / max(1.0, env.MAX_T)),
        ),
        dtype=np.float32,
    )

    occupied = set(yard.occupancy())
    max_distance = max(1.0, yard.rows + yard.cols)
    max_route = max(1.0, yard.rows * yard.cols)
    candidate_rows = []
    for candidate in valid_candidates:
        source_path = shortest_clear_path(
            yard, source, candidate, ignore_labels=(block.label,)
        )
        if source_path is None:
            raise ValueError("shared candidate mask contained an unreachable cell")
        placed = yard.with_block(block, candidate)
        route = find_min_obstruction_route(
            placed,
            candidate,
            placed.exits,
            ignore_labels=(block.label,),
        )
        route_distance = len(route.cells) - 1 if route is not None else max_route
        blockers = route.obstruction_count if route is not None else len(env.blocks)
        row, col = candidate
        radius_1 = sum(
            (row + dr, col + dc) in occupied
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))
        ) / 4.0
        radius_2_cells = (
            (row + dr, col + dc)
            for dr in range(-2, 3)
            for dc in range(-2, 3)
            if 0 < abs(dr) + abs(dc) <= 2
        )
        radius_2 = sum(cell in occupied for cell in radius_2_cells) / 12.0
        cell_row, cell_col = _normalise_cell(candidate, yard.rows, yard.cols)
        exit_distance = min(
            abs(row - exit_cell[0]) + abs(col - exit_cell[1])
            for exit_cell in yard.exits
        )
        door_distance = env.manhattan_distance(candidate, env.door_cell)
        usage = env.storage_counts.get(candidate, 0) / max(1.0, len(env.blocks))
        candidate_rows.append(
            (
                cell_row,
                cell_col,
                (len(source_path) - 1) / max_route,
                env.manhattan_distance(candidate, source) / max_distance,
                exit_distance / max_distance,
                route_distance / max_route,
                blockers / max(1.0, len(env.blocks)),
                radius_1,
                radius_2,
                usage,
                door_distance / max_distance,
                float(blockers == 0),
            )
        )
    candidates = np.asarray(candidate_rows, dtype=np.float32)
    return REGV4Observation(
        block_features=blocks,
        block_mask=block_mask,
        current_block_features=current,
        global_features=global_features,
        candidate_features=candidates,
    )


class REGV4AssignmentSource:
    """Trainable or frozen Track A-native REG-v4 assignment source."""

    feature_version = FEATURE_VERSION
    architecture_name = ARCHITECTURE_NAME

    def __init__(
        self,
        env,
        config: REGV4Config = REGV4Config(),
        *,
        seed: int = 0,
        learning_enabled: bool = True,
        device: str | torch.device = "cpu",
    ):
        self.env = env
        self.config = config
        self.device = torch.device(device)
        self.network = REGV4Network(config).to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.network.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self.replay: deque[REGV4Transition] = deque(maxlen=config.replay_size)
        self.return_moments = RunningMoments()
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)
        self.learning_enabled = bool(learning_enabled)
        self.pending: dict[str, dict] = {}
        self.assignment_count = 0
        self.gradient_steps = 0
        self.completed_events = 0
        self.failed_events = 0
        self.loss_history: list[float] = []

    @property
    def epsilon(self) -> float:
        if not self.learning_enabled:
            return 0.0
        warmup = self.config.epsilon_warmup_assignments
        if self.assignment_count <= warmup:
            return self.config.epsilon_start
        fraction = min(
            1.0,
            (self.assignment_count - warmup)
            / self.config.epsilon_decay_assignments,
        )
        return self.config.epsilon_start + fraction * (
            self.config.epsilon_end - self.config.epsilon_start
        )

    def set_learning_enabled(self, enabled: bool) -> None:
        self.learning_enabled = bool(enabled)
        self.network.train(self.learning_enabled)
        if not self.learning_enabled:
            self.pending.clear()

    def on_episode_start(self) -> None:
        self.pending.clear()

    def _scores(self, observation: REGV4Observation) -> torch.Tensor:
        # Action selection never participates in the regression graph. Replay
        # stores the observation and reconstructs gradients during _learn_once.
        with torch.no_grad():
            return self.network(
                torch.from_numpy(observation.block_features)
                .unsqueeze(0)
                .to(self.device),
                torch.from_numpy(observation.block_mask)
                .unsqueeze(0)
                .to(self.device),
                torch.from_numpy(observation.current_block_features)
                .unsqueeze(0)
                .to(self.device),
                torch.from_numpy(observation.global_features)
                .unsqueeze(0)
                .to(self.device),
                torch.from_numpy(observation.candidate_features)
                .unsqueeze(0)
                .to(self.device),
            )[0]

    def propose(self, yard, block, source, valid_candidates):
        observation = build_observation(
            self.env, yard, block, source, valid_candidates
        )
        if self.learning_enabled and self.rng.random() < self.epsilon:
            chosen_index = self.rng.randrange(len(valid_candidates))
        else:
            self.network.eval()
            with torch.no_grad():
                scores = self._scores(observation)
            chosen_index = max(
                range(len(valid_candidates)),
                key=lambda index: (float(scores[index].item()), -index),
            )
            self.network.train(self.learning_enabled)
        if self.learning_enabled:
            self.pending[block.label] = {
                "block_features": observation.block_features.copy(),
                "block_mask": observation.block_mask.copy(),
                "current_block_features": observation.current_block_features.copy(),
                "global_features": observation.global_features.copy(),
                "chosen_candidate_features": observation.candidate_features[
                    chosen_index
                ].copy(),
                "acc": 0.0,
                "disc": 1.0,
                "steps": 0,
            }
            self.assignment_count += 1
        return valid_candidates[chosen_index]

    def _close(self, label: str, *, failed: bool) -> None:
        entry = self.pending.pop(label, None)
        if entry is None:
            return
        event_return = float(entry["acc"])
        if failed:
            event_return += float(entry["disc"]) * self.config.fail_reward
            self.failed_events += 1
        else:
            self.completed_events += 1
        self.return_moments.update(event_return)
        self.replay.append(
            REGV4Transition(
                block_features=entry["block_features"],
                block_mask=entry["block_mask"],
                current_block_features=entry["current_block_features"],
                global_features=entry["global_features"],
                chosen_candidate_features=entry["chosen_candidate_features"],
                event_return=event_return,
            )
        )

    def on_step(self, reward: float, info: dict) -> None:
        if not self.learning_enabled:
            return
        for entry in self.pending.values():
            entry["acc"] += entry["disc"] * float(reward)
            entry["disc"] *= self.config.gamma
            entry["steps"] += 1
        delivered = info.get("delivered_block")
        if delivered:
            self._close(str(delivered), failed=False)

    def on_episode_end(self, *, success: bool, truncated: bool) -> None:
        if not self.learning_enabled:
            return
        for label in list(self.pending):
            self._close(label, failed=True)
        if len(self.replay) >= self.config.min_replay_size:
            replay_snapshot = list(self.replay)
            for _ in range(self.config.updates_per_episode):
                self._learn_once(replay_snapshot)

    @staticmethod
    def _pad_blocks(transitions: list[REGV4Transition]):
        max_blocks = max(item.block_features.shape[0] for item in transitions)
        blocks = np.zeros(
            (len(transitions), max_blocks, BLOCK_FEATURE_DIM), dtype=np.float32
        )
        masks = np.zeros((len(transitions), max_blocks), dtype=np.float32)
        for index, item in enumerate(transitions):
            count = item.block_features.shape[0]
            blocks[index, :count] = item.block_features
            masks[index, :count] = item.block_mask
        return blocks, masks

    def _learn_once(
        self, replay_snapshot: Optional[list[REGV4Transition]] = None
    ) -> float:
        replay_list = (
            list(self.replay) if replay_snapshot is None else replay_snapshot
        )
        indices = self.np_rng.choice(
            len(replay_list), size=self.config.batch_size, replace=False
        )
        batch = [replay_list[int(index)] for index in indices]
        blocks, masks = self._pad_blocks(batch)
        current = np.stack([item.current_block_features for item in batch])
        globals_ = np.stack([item.global_features for item in batch])
        candidates = np.stack(
            [item.chosen_candidate_features for item in batch]
        )[:, None, :]
        returns = torch.as_tensor(
            [item.event_return for item in batch],
            dtype=torch.float32,
            device=self.device,
        )
        prediction = self.network(
            torch.from_numpy(blocks).to(self.device),
            torch.from_numpy(masks).to(self.device),
            torch.from_numpy(current).to(self.device),
            torch.from_numpy(globals_).to(self.device),
            torch.from_numpy(candidates).to(self.device),
        )[:, 0]
        target = self.return_moments.normalize(returns)
        loss = F.huber_loss(
            prediction, target, delta=self.config.huber_delta
        )
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.network.parameters(), self.config.grad_clip)
        self.optimizer.step()
        value = float(loss.item())
        self.loss_history.append(value)
        self.gradient_steps += 1
        return value

    def checkpoint(self, **metadata) -> dict:
        return {
            "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
            "selector_feature_version": FEATURE_VERSION,
            "selector_architecture": ARCHITECTURE_NAME,
            "block_feature_dim": BLOCK_FEATURE_DIM,
            "global_feature_dim": GLOBAL_FEATURE_DIM,
            "candidate_feature_dim": CANDIDATE_FEATURE_DIM,
            "config": asdict(self.config),
            "network_state_dict": self.network.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "return_moments": self.return_moments.state_dict(),
            "assignment_count": self.assignment_count,
            "gradient_steps": self.gradient_steps,
            "completed_events": self.completed_events,
            "failed_events": self.failed_events,
            **metadata,
        }

    @classmethod
    def from_checkpoint(
        cls,
        env,
        payload: dict,
        *,
        learning_enabled: bool = False,
        device: str | torch.device = "cpu",
        seed: int = 0,
    ) -> "REGV4AssignmentSource":
        expected = {
            "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
            "selector_feature_version": FEATURE_VERSION,
            "selector_architecture": ARCHITECTURE_NAME,
            "block_feature_dim": BLOCK_FEATURE_DIM,
            "global_feature_dim": GLOBAL_FEATURE_DIM,
            "candidate_feature_dim": CANDIDATE_FEATURE_DIM,
        }
        mismatches = [
            key for key, value in expected.items() if payload.get(key) != value
        ]
        if mismatches:
            raise ValueError(
                "Incompatible REG-v4 checkpoint metadata: "
                + ", ".join(mismatches)
            )
        source = cls(
            env,
            REGV4Config(**payload["config"]),
            seed=seed,
            learning_enabled=learning_enabled,
            device=device,
        )
        source.network.load_state_dict(payload["network_state_dict"])
        if learning_enabled and payload.get("optimizer_state_dict"):
            source.optimizer.load_state_dict(payload["optimizer_state_dict"])
        source.return_moments.load_state_dict(payload["return_moments"])
        source.assignment_count = int(payload.get("assignment_count", 0))
        source.gradient_steps = int(payload.get("gradient_steps", 0))
        source.completed_events = int(payload.get("completed_events", 0))
        source.failed_events = int(payload.get("failed_events", 0))
        source.set_learning_enabled(learning_enabled)
        return source


__all__ = [
    "ARCHITECTURE_NAME",
    "BLOCK_FEATURE_DIM",
    "CANDIDATE_FEATURE_DIM",
    "CHECKPOINT_SCHEMA_VERSION",
    "FEATURE_VERSION",
    "GLOBAL_FEATURE_DIM",
    "REGV4AssignmentSource",
    "REGV4Config",
    "REGV4Network",
    "REGV4Observation",
    "RunningMoments",
    "build_observation",
]
