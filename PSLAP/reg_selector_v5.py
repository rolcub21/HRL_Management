"""Decision-epoch Double-DQN REG selector for Track A.

REG-v5 deliberately retains the v4 observation and cardinality-invariant
candidate scorer.  The controlled ablation is the learning semantics: one
transition spans two consecutive storage-assignment decisions, so environment
rewards never overlap across replay records.  Continuing placement effects are
represented by the next yard state and a variable-duration Double-DQN backup.
"""

from __future__ import annotations

from collections import deque
from copy import deepcopy
from dataclasses import asdict, dataclass
import random
from typing import Optional

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from PSLAP.reg_selector_v4 import (
    BLOCK_FEATURE_DIM,
    CANDIDATE_FEATURE_DIM,
    GLOBAL_FEATURE_DIM,
    REGV4Config,
    REGV4Network,
    REGV4Observation,
    build_observation,
)


FEATURE_VERSION = 5
CHECKPOINT_SCHEMA_VERSION = 1
ARCHITECTURE_NAME = "reg_v5_assignment_epoch_double_dqn"


def build_assignment_observation(
    env,
    yard,
    block,
    source,
    valid_candidates,
) -> REGV4Observation:
    """Build the canonical observation for a pending storage assignment.

    Track A requests an assignment while the inbound block is still resting at
    the pickup cell.  Track B historically requests the same assignment one
    primitive step later, immediately after pickup.  ``carrying`` is therefore
    an executor-phase detail, not information that should change the selector's
    assignment state.  Canonicalising the current block to the arrived/pending
    phase makes a frozen Track A checkpoint receive the same status semantics in
    both tracks while preserving the live time, queue, yard, and candidate data.
    """

    observation = build_observation(
        env, yard, block, source, valid_candidates
    )
    current_index = next(
        index for index, item in enumerate(env.blocks) if item.label == block.label
    )
    block_features = observation.block_features.copy()
    current = observation.current_block_features.copy()
    assignment_pending_status = np.asarray(
        (0.0, 1.0, 0.0, 0.0, 0.0), dtype=np.float32
    )
    block_features[current_index, :5] = assignment_pending_status
    current[:5] = assignment_pending_status
    return REGV4Observation(
        block_features=block_features,
        block_mask=observation.block_mask,
        current_block_features=current,
        global_features=observation.global_features,
        candidate_features=observation.candidate_features,
    )


@dataclass(frozen=True)
class REGV5Config:
    block_embedding_dim: int = 64
    candidate_embedding_dim: int = 64
    context_dim: int = 128
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    replay_size: int = 20_000
    batch_size: int = 128
    min_replay_size: int = 256
    updates_per_episode: int = 20
    gamma: float = 1.0
    reward_scale: float = 0.01
    grad_clip: float = 5.0
    huber_delta: float = 1.0
    truncation_penalty: float = 0.0
    target_update_every: int = 250
    ema_tau: float = 0.01
    epsilon_start: float = 0.90
    epsilon_end: float = 0.05
    epsilon_warmup_assignments: int = 300
    epsilon_decay_assignments: int = 10_000

    def __post_init__(self):
        positive = (
            "block_embedding_dim",
            "candidate_embedding_dim",
            "context_dim",
            "replay_size",
            "batch_size",
            "min_replay_size",
            "updates_per_episode",
            "reward_scale",
            "grad_clip",
            "huber_delta",
            "target_update_every",
        )
        for name in positive:
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.min_replay_size < self.batch_size:
            raise ValueError("min_replay_size must be at least batch_size")
        if not 0.0 < self.gamma <= 1.0:
            raise ValueError("gamma must be in (0, 1]")
        if not 0.0 < self.ema_tau <= 1.0:
            raise ValueError("ema_tau must be in (0, 1]")
        if not 0.0 <= self.epsilon_end <= self.epsilon_start <= 1.0:
            raise ValueError("epsilon must satisfy 0 <= end <= start <= 1")
        if self.epsilon_decay_assignments <= 0:
            raise ValueError("epsilon_decay_assignments must be positive")

    def network_config(self) -> REGV4Config:
        return REGV4Config(
            block_embedding_dim=self.block_embedding_dim,
            candidate_embedding_dim=self.candidate_embedding_dim,
            context_dim=self.context_dim,
        )


@dataclass(frozen=True)
class REGV5Transition:
    block_features: np.ndarray
    block_mask: np.ndarray
    current_block_features: np.ndarray
    global_features: np.ndarray
    chosen_candidate_features: np.ndarray
    interval_return: float
    duration: int
    next_observation: Optional[REGV4Observation]
    terminal: bool
    truncated: bool


class REGV5AssignmentSource:
    """Trainable or frozen Track A-native assignment-epoch controller."""

    feature_version = FEATURE_VERSION
    architecture_name = ARCHITECTURE_NAME

    def __init__(
        self,
        env,
        config: REGV5Config = REGV5Config(),
        *,
        seed: int = 0,
        learning_enabled: bool = True,
        device: str | torch.device = "cpu",
    ):
        self.env = env
        self.config = config
        self.device = torch.device(device)
        network_config = config.network_config()
        self.network = REGV4Network(network_config).to(self.device)
        self.target_network = deepcopy(self.network).to(self.device)
        self.ema_network = deepcopy(self.network).to(self.device)
        self.target_network.requires_grad_(False)
        self.ema_network.requires_grad_(False)
        self.target_network.eval()
        self.ema_network.eval()
        self.optimizer = torch.optim.AdamW(
            self.network.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self.replay: deque[REGV5Transition] = deque(maxlen=config.replay_size)
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)
        self.learning_enabled = bool(learning_enabled)
        self.pending: Optional[dict] = None
        self.assignment_count = 0
        self.gradient_steps = 0
        self.target_updates = 0
        self.completed_intervals = 0
        self.terminal_intervals = 0
        self.truncated_intervals = 0
        self.loss_history: list[float] = []
        self.q_history: list[float] = []
        self.target_history: list[float] = []
        self.set_learning_enabled(self.learning_enabled)

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
        self.target_network.eval()
        self.ema_network.eval()
        if not self.learning_enabled:
            self.pending = None

    def on_episode_start(self) -> None:
        self.pending = None

    def _scores(
        self,
        observation: REGV4Observation,
        *,
        network: Optional[nn.Module] = None,
    ) -> torch.Tensor:
        scorer = self.network if network is None else network
        with torch.no_grad():
            return scorer(
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

    def greedy_proposal(self, yard, block, source, valid_candidates):
        """Return the deterministic deployment choice without side effects.

        This method never advances exploration RNG, opens/closes a replay
        interval, increments assignment counters, or mutates the yard.  It is
        therefore suitable for read-only duration estimates at a scheduler
        decision epoch.  A trainable caller may inspect its greedy action, but
        that action is intentionally distinct from an exploratory proposal.
        """

        if not valid_candidates:
            raise ValueError("greedy proposal requires at least one candidate")
        observation = build_assignment_observation(
            self.env, yard, block, source, valid_candidates
        )
        scores = self._scores(observation, network=self.network)
        chosen_index = max(
            range(len(valid_candidates)),
            key=lambda index: (float(scores[index].item()), -index),
        )
        return valid_candidates[chosen_index]

    def score_candidates(self, yard, block, source, valid_candidates):
        """Return deployment scores without RNG, replay, or counter effects."""

        if not valid_candidates:
            return np.empty((0,), dtype=np.float32)
        observation = build_assignment_observation(
            self.env, yard, block, source, valid_candidates
        )
        scores = self._scores(observation, network=self.network)
        return scores.detach().cpu().numpy().astype(np.float32, copy=True)

    def preview(self, yard, block, source, valid_candidates):
        """Expose the frozen greedy proposal through the neutral preview ABI."""

        if self.learning_enabled:
            raise RuntimeError("REG-v5 preview requires a frozen selector")
        return self.greedy_proposal(yard, block, source, valid_candidates)

    def _finish_interval(
        self,
        *,
        next_observation: Optional[REGV4Observation],
        terminal: bool,
        truncated: bool,
    ) -> None:
        entry = self.pending
        if entry is None:
            return
        interval_return = float(entry["acc"])
        if truncated and self.config.truncation_penalty:
            interval_return += (
                float(entry["disc"]) * self.config.truncation_penalty
            )
        self.replay.append(
            REGV5Transition(
                block_features=entry["block_features"],
                block_mask=entry["block_mask"],
                current_block_features=entry["current_block_features"],
                global_features=entry["global_features"],
                chosen_candidate_features=entry["chosen_candidate_features"],
                interval_return=interval_return,
                duration=int(entry["duration"]),
                next_observation=next_observation,
                terminal=bool(terminal),
                truncated=bool(truncated),
            )
        )
        self.completed_intervals += 1
        self.terminal_intervals += int(terminal)
        self.truncated_intervals += int(truncated)
        self.pending = None

    def propose(self, yard, block, source, valid_candidates):
        observation = build_assignment_observation(
            self.env, yard, block, source, valid_candidates
        )
        if self.learning_enabled and self.pending is not None:
            self._finish_interval(
                next_observation=observation,
                terminal=False,
                truncated=False,
            )

        if self.learning_enabled and self.rng.random() < self.epsilon:
            chosen_index = self.rng.randrange(len(valid_candidates))
        else:
            scores = self._scores(observation, network=self.network)
            chosen_index = max(
                range(len(valid_candidates)),
                key=lambda index: (float(scores[index].item()), -index),
            )

        if self.learning_enabled:
            self.pending = {
                "block_features": observation.block_features.copy(),
                "block_mask": observation.block_mask.copy(),
                "current_block_features": observation.current_block_features.copy(),
                "global_features": observation.global_features.copy(),
                "chosen_candidate_features": observation.candidate_features[
                    chosen_index
                ].copy(),
                "acc": 0.0,
                "disc": 1.0,
                "duration": 0,
            }
            self.assignment_count += 1
        return valid_candidates[chosen_index]

    def on_step(self, reward: float, info: dict) -> None:
        if not self.learning_enabled or self.pending is None:
            return
        self.pending["acc"] += self.pending["disc"] * float(reward)
        self.pending["disc"] *= self.config.gamma
        self.pending["duration"] += 1

    def on_episode_end(self, *, success: bool, truncated: bool) -> None:
        if not self.learning_enabled:
            return
        self._finish_interval(
            next_observation=None,
            terminal=True,
            truncated=bool(truncated or not success),
        )
        if len(self.replay) >= self.config.min_replay_size:
            replay_snapshot = list(self.replay)
            for _ in range(self.config.updates_per_episode):
                self._learn_once(replay_snapshot)

    @staticmethod
    def _pad_current_states(transitions: list[REGV5Transition]):
        max_blocks = max(item.block_features.shape[0] for item in transitions)
        blocks = np.zeros(
            (len(transitions), max_blocks, BLOCK_FEATURE_DIM), dtype=np.float32
        )
        block_masks = np.zeros((len(transitions), max_blocks), dtype=np.float32)
        current = np.stack(
            [item.current_block_features for item in transitions]
        ).astype(np.float32)
        globals_ = np.stack([item.global_features for item in transitions]).astype(
            np.float32
        )
        candidates = np.stack(
            [item.chosen_candidate_features for item in transitions]
        ).astype(np.float32)[:, None, :]
        for index, item in enumerate(transitions):
            count = item.block_features.shape[0]
            blocks[index, :count] = item.block_features
            block_masks[index, :count] = item.block_mask
        return blocks, block_masks, current, globals_, candidates

    @staticmethod
    def _pad_next_states(transitions: list[REGV5Transition]):
        observations = [item.next_observation for item in transitions]
        max_blocks = max(
            (obs.block_features.shape[0] for obs in observations if obs is not None),
            default=1,
        )
        max_candidates = max(
            (obs.candidate_features.shape[0] for obs in observations if obs is not None),
            default=1,
        )
        size = len(transitions)
        blocks = np.zeros(
            (size, max_blocks, BLOCK_FEATURE_DIM), dtype=np.float32
        )
        block_masks = np.zeros((size, max_blocks), dtype=np.float32)
        current = np.zeros((size, BLOCK_FEATURE_DIM), dtype=np.float32)
        globals_ = np.zeros((size, GLOBAL_FEATURE_DIM), dtype=np.float32)
        candidates = np.zeros(
            (size, max_candidates, CANDIDATE_FEATURE_DIM), dtype=np.float32
        )
        candidate_masks = np.zeros((size, max_candidates), dtype=bool)
        nonterminal = np.zeros(size, dtype=bool)
        for index, observation in enumerate(observations):
            if observation is None:
                continue
            block_count = observation.block_features.shape[0]
            candidate_count = observation.candidate_features.shape[0]
            blocks[index, :block_count] = observation.block_features
            block_masks[index, :block_count] = observation.block_mask
            current[index] = observation.current_block_features
            globals_[index] = observation.global_features
            candidates[index, :candidate_count] = observation.candidate_features
            candidate_masks[index, :candidate_count] = True
            nonterminal[index] = True
        return (
            blocks,
            block_masks,
            current,
            globals_,
            candidates,
            candidate_masks,
            nonterminal,
        )

    def _td_targets(self, transitions: list[REGV5Transition]) -> torch.Tensor:
        (
            blocks,
            block_masks,
            current,
            globals_,
            candidates,
            candidate_masks,
            nonterminal,
        ) = self._pad_next_states(transitions)
        with torch.no_grad():
            online_scores = self.network(
                torch.from_numpy(blocks).to(self.device),
                torch.from_numpy(block_masks).to(self.device),
                torch.from_numpy(current).to(self.device),
                torch.from_numpy(globals_).to(self.device),
                torch.from_numpy(candidates).to(self.device),
            )
            mask = torch.from_numpy(candidate_masks).to(self.device)
            online_scores = online_scores.masked_fill(~mask, -torch.inf)
            next_actions = online_scores.argmax(dim=1)
            target_scores = self.target_network(
                torch.from_numpy(blocks).to(self.device),
                torch.from_numpy(block_masks).to(self.device),
                torch.from_numpy(current).to(self.device),
                torch.from_numpy(globals_).to(self.device),
                torch.from_numpy(candidates).to(self.device),
            )
            next_values = target_scores.gather(1, next_actions[:, None]).squeeze(1)
            next_values = torch.where(
                torch.from_numpy(nonterminal).to(self.device),
                next_values,
                torch.zeros_like(next_values),
            )
            rewards = torch.as_tensor(
                [item.interval_return for item in transitions],
                dtype=torch.float32,
                device=self.device,
            ) * self.config.reward_scale
            discounts = torch.as_tensor(
                [self.config.gamma ** item.duration for item in transitions],
                dtype=torch.float32,
                device=self.device,
            )
            return rewards + discounts * next_values

    def _update_ema(self) -> None:
        tau = self.config.ema_tau
        with torch.no_grad():
            for ema_parameter, parameter in zip(
                self.ema_network.parameters(), self.network.parameters()
            ):
                ema_parameter.lerp_(parameter, tau)
            for ema_buffer, buffer in zip(
                self.ema_network.buffers(), self.network.buffers()
            ):
                ema_buffer.copy_(buffer)

    def _learn_once(
        self, replay_snapshot: Optional[list[REGV5Transition]] = None
    ) -> float:
        replay_list = (
            list(self.replay) if replay_snapshot is None else replay_snapshot
        )
        indices = self.np_rng.choice(
            len(replay_list), size=self.config.batch_size, replace=False
        )
        batch = [replay_list[int(index)] for index in indices]
        blocks, masks, current, globals_, candidates = self._pad_current_states(batch)
        prediction = self.network(
            torch.from_numpy(blocks).to(self.device),
            torch.from_numpy(masks).to(self.device),
            torch.from_numpy(current).to(self.device),
            torch.from_numpy(globals_).to(self.device),
            torch.from_numpy(candidates).to(self.device),
        )[:, 0]
        target = self._td_targets(batch)
        loss = F.huber_loss(
            prediction, target, delta=self.config.huber_delta
        )
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.network.parameters(), self.config.grad_clip)
        self.optimizer.step()
        self.gradient_steps += 1
        self._update_ema()
        if self.gradient_steps % self.config.target_update_every == 0:
            self.target_network.load_state_dict(self.network.state_dict())
            self.target_updates += 1
        value = float(loss.item())
        self.loss_history.append(value)
        self.q_history.append(float(prediction.detach().mean().item()))
        self.target_history.append(float(target.detach().mean().item()))
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
            "target_network_state_dict": self.target_network.state_dict(),
            "ema_network_state_dict": self.ema_network.state_dict(),
            "deployment_state_dict": self.ema_network.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "assignment_count": self.assignment_count,
            "gradient_steps": self.gradient_steps,
            "target_updates": self.target_updates,
            "completed_intervals": self.completed_intervals,
            "terminal_intervals": self.terminal_intervals,
            "truncated_intervals": self.truncated_intervals,
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
    ) -> "REGV5AssignmentSource":
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
                "Incompatible REG-v5 checkpoint metadata: "
                + ", ".join(mismatches)
            )
        source = cls(
            env,
            REGV5Config(**payload["config"]),
            seed=seed,
            learning_enabled=learning_enabled,
            device=device,
        )
        if learning_enabled:
            source.network.load_state_dict(payload["network_state_dict"])
            source.target_network.load_state_dict(
                payload.get("target_network_state_dict", payload["network_state_dict"])
            )
            source.ema_network.load_state_dict(
                payload.get("ema_network_state_dict", payload["network_state_dict"])
            )
            if payload.get("optimizer_state_dict"):
                source.optimizer.load_state_dict(payload["optimizer_state_dict"])
        else:
            deployment = payload.get(
                "deployment_state_dict",
                payload.get("ema_network_state_dict", payload["network_state_dict"]),
            )
            source.network.load_state_dict(deployment)
            source.target_network.load_state_dict(deployment)
            source.ema_network.load_state_dict(deployment)
        source.assignment_count = int(payload.get("assignment_count", 0))
        source.gradient_steps = int(payload.get("gradient_steps", 0))
        source.target_updates = int(payload.get("target_updates", 0))
        source.completed_intervals = int(payload.get("completed_intervals", 0))
        source.terminal_intervals = int(payload.get("terminal_intervals", 0))
        source.truncated_intervals = int(payload.get("truncated_intervals", 0))
        source.set_learning_enabled(learning_enabled)
        return source


__all__ = [
    "ARCHITECTURE_NAME",
    "CHECKPOINT_SCHEMA_VERSION",
    "FEATURE_VERSION",
    "REGV5AssignmentSource",
    "REGV5Config",
    "REGV5Transition",
    "build_assignment_observation",
]
