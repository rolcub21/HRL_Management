"""Kim et al. (2020)-inspired spatial placement actor-critic.

This module is an explicit adaptation, not a reproduction.  The paper's
locating agent is retained at the level that is observable from the article:
a spatial relative-dwell representation, one legal yard-cell action, a
stochastic actor-critic policy, and delayed credit when the selected block is
eventually retrieved.  The repository's shared candidate mask and strict
retrieval executor remain authoritative.

Each placement receives the negative number of obstructing blocks relocated
while retrieving that same block.  Once a complete episode has resolved those
delayed rewards, they are put back into placement order and accumulated into
the discounted episodic returns used by A3C.  Placements whose retrieval is
not observed are censored and never silently assigned zero reward.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Optional

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from PSLAP.dynamic_yard import BlockView, Cell, YardSnapshot


METHOD_NAME = "kim2020_a3c_spatial_adapted"
FEATURE_VERSION = "kim2020_relative_dwell_spatial_v1"
ARCHITECTURE_NAME = "kim2020_grid_actor_critic_adapted_v1"
REWARD_CONTRACT = "kim2020_delayed_target_retrieval_relocations_v1"
RETURN_CONTRACT = "kim2020_episodic_discounted_placement_return_v1"
ACTION_MAPPING = "row_major_grid_cell_masked_by_shared_candidates_v1"
CHECKPOINT_SCHEMA_VERSION = 2
INPUT_CHANNELS = 5

DEPLOYMENT_STOCHASTIC = "stochastic"
DEPLOYMENT_MAP = "map"
DEPLOYMENT_MODES = (DEPLOYMENT_STOCHASTIC, DEPLOYMENT_MAP)


@dataclass(frozen=True)
class Kim2020Config:
    """Hyperparameters for the adapted single-worker actor-critic.

    ``gamma`` discounts successive *placement decisions*, not primitive travel
    steps.  All rewards are resolved before the terminal Monte Carlo return is
    calculated, so no value bootstrap is used at episode end.
    """

    hidden_channels: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    gamma: float = 0.99
    reward_scale: float = 1.0
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    grad_clip: float = 5.0
    updates_per_episode: int = 1

    def __post_init__(self) -> None:
        if self.hidden_channels <= 0:
            raise ValueError("hidden_channels must be positive")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if self.weight_decay < 0.0:
            raise ValueError("weight_decay must be nonnegative")
        if not 0.0 < self.gamma <= 1.0:
            raise ValueError("gamma must be in (0, 1]")
        if self.reward_scale <= 0.0:
            raise ValueError("reward_scale must be positive")
        if self.entropy_coef < 0.0 or self.value_coef < 0.0:
            raise ValueError("loss coefficients must be nonnegative")
        if self.grad_clip <= 0.0:
            raise ValueError("grad_clip must be positive")
        if self.updates_per_episode <= 0:
            raise ValueError("updates_per_episode must be positive")


@dataclass(frozen=True)
class Kim2020Observation:
    """Paper-inspired grid plus the repository-authoritative legal mask."""

    spatial_features: np.ndarray
    candidate_mask: np.ndarray
    candidates: tuple[Cell, ...]

    @property
    def features(self) -> np.ndarray:
        """Short compatibility alias used by analysis utilities."""

        return self.spatial_features


@dataclass
class Kim2020PlacementRecord:
    """One selected cell and its later target-specific retrieval outcome."""

    instance_id: str
    block_label: str
    spatial_features: np.ndarray
    candidate_mask: np.ndarray
    chosen_cell: Cell
    chosen_flat_index: int
    selection_time_step: int
    stored: bool = False
    storage_time_step: Optional[int] = None
    relocation_count: Optional[int] = None
    reward: Optional[float] = None
    discounted_return: Optional[float] = None
    censored: bool = False
    censor_reason: Optional[str] = None

    def audit_dict(self) -> dict:
        return {
            "instance_id": self.instance_id,
            "block_label": self.block_label,
            "chosen_cell": tuple(self.chosen_cell),
            "chosen_flat_index": int(self.chosen_flat_index),
            "selection_time_step": int(self.selection_time_step),
            "stored": bool(self.stored),
            "storage_time_step": self.storage_time_step,
            "relocation_count": self.relocation_count,
            "reward": self.reward,
            "discounted_return": self.discounted_return,
            "censored": bool(self.censored),
            "censor_reason": self.censor_reason,
        }


def _canonical_geometry_from_yard(yard: YardSnapshot) -> dict:
    return {
        "rows": int(yard.rows),
        "cols": int(yard.cols),
        "traversable": [list(cell) for cell in sorted(yard.traversable)],
        "storage_cells": [list(cell) for cell in sorted(yard.storage_cells)],
        "exits": [list(cell) for cell in sorted(yard.exits)],
    }


def geometry_contract(env) -> dict:
    """Return the exact geometry governed by one checkpoint."""

    traversable = frozenset(
        (row, col)
        for row in range(env.grid_rows)
        for col in range(env.grid_cols)
        if env.rooms[row, col] != "#"
    )
    yard = YardSnapshot(
        rows=int(env.grid_rows),
        cols=int(env.grid_cols),
        traversable=traversable,
        storage_cells=frozenset(tuple(cell) for cell in env.storage_positions),
        exits=tuple(tuple(cell) for cell in env.exit_cells),
        blocks=(),
    )
    return _canonical_geometry_from_yard(yard)


def build_spatial_observation(
    yard: YardSnapshot,
    block: BlockView,
    valid_candidates,
) -> Kim2020Observation:
    """Encode a legal placement decision without future schedule information.

    Channel 0 follows the paper's relative-storage-period encoding: empty cells
    are -1 and occupied cells are ``clip(remaining / incoming, 0, 2)``.  Four
    explicit channels disambiguate occupancy, storage layout, traversability,
    and road/exit direction; those channels are adaptation details required by
    this repository's walled and potentially rectangular geometries.
    """

    rows, cols = int(yard.rows), int(yard.cols)
    relative_dwell = np.full((rows, cols), -1.0, dtype=np.float32)
    occupied = np.zeros((rows, cols), dtype=np.float32)
    storage = np.zeros((rows, cols), dtype=np.float32)
    traversable = np.zeros((rows, cols), dtype=np.float32)
    exit_proximity = np.zeros((rows, cols), dtype=np.float32)

    for row, col in yard.storage_cells:
        storage[row, col] = 1.0
    for row, col in yard.traversable:
        traversable[row, col] = 1.0

    denominator = max(float(block.remaining_time), 1.0)
    for stored_block in yard.blocks:
        row, col = stored_block.position
        if not (0 <= row < rows and 0 <= col < cols):
            raise ValueError("yard block lies outside the declared geometry")
        occupied[row, col] = 1.0
        relative_dwell[row, col] = np.float32(
            np.clip(
                max(float(stored_block.remaining_time), 0.0) / denominator,
                0.0,
                2.0,
            )
        )

    if yard.exits:
        max_distance = max(rows + cols - 2, 1)
        for row, col in yard.traversable:
            distance = min(
                abs(row - exit_row) + abs(col - exit_col)
                for exit_row, exit_col in yard.exits
            )
            exit_proximity[row, col] = np.float32(
                1.0 - min(distance, max_distance) / max_distance
            )

    candidates = tuple(tuple(cell) for cell in valid_candidates)
    if len(candidates) != len(set(candidates)):
        raise ValueError("valid_candidates contains duplicate cells")
    candidate_mask = np.zeros((rows, cols), dtype=bool)
    for row, col in candidates:
        if not (0 <= row < rows and 0 <= col < cols):
            raise ValueError("candidate lies outside the declared geometry")
        if (row, col) not in yard.storage_cells:
            raise ValueError("candidate is not a storage cell")
        candidate_mask[row, col] = True

    return Kim2020Observation(
        spatial_features=np.stack(
            (
                relative_dwell,
                occupied,
                storage,
                traversable,
                exit_proximity,
            )
        ).astype(np.float32, copy=False),
        candidate_mask=candidate_mask,
        candidates=candidates,
    )


class Kim2020SpatialActorCritic(nn.Module):
    """Fully convolutional actor with a pooled contextual critic."""

    def __init__(self, hidden_channels: int = 32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(INPUT_CHANNELS, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.actor = nn.Conv2d(hidden_channels, 1, kernel_size=1)
        self.critic = nn.Linear(hidden_channels, 1)

    def forward(self, spatial_features: torch.Tensor):
        encoded = self.encoder(spatial_features)
        logits = self.actor(encoded).squeeze(1).flatten(start_dim=1)
        pooled = F.adaptive_avg_pool2d(encoded, output_size=1).flatten(start_dim=1)
        value = self.critic(pooled).squeeze(1)
        return logits, value


def _state_dict_digest(state_dict: dict) -> str:
    digest = hashlib.sha256()
    for name in sorted(state_dict):
        tensor = state_dict[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def kim2020_deployment_digest(payload: dict) -> str:
    """Stable identity for a frozen adapted placement policy."""

    state = payload.get("deployment_state_dict", payload.get("network_state_dict"))
    if state is None:
        raise ValueError("Kim checkpoint has no deployment weights")
    identity = {
        "method": payload.get("assignment_source"),
        "architecture": payload.get("selector_architecture"),
        "feature_version": payload.get("selector_feature_version"),
        "geometry": payload.get("geometry_contract"),
        "config": payload.get("config"),
        "weights": _state_dict_digest(state),
    }
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


class Kim2020A3CSpatialSource:
    """Trainable or frozen source implementing the neutral assignment ABI."""

    feature_version = FEATURE_VERSION
    architecture_name = ARCHITECTURE_NAME
    reward_contract = REWARD_CONTRACT
    return_contract = RETURN_CONTRACT

    def __init__(
        self,
        env,
        config: Kim2020Config = Kim2020Config(),
        *,
        seed: int = 0,
        policy_seed: Optional[int] = None,
        learning_enabled: bool = True,
        deployment_mode: str = DEPLOYMENT_STOCHASTIC,
        device: str | torch.device = "cpu",
    ):
        if deployment_mode not in DEPLOYMENT_MODES:
            raise ValueError(
                f"deployment_mode must be one of {DEPLOYMENT_MODES}"
            )
        self.env = env
        self.config = config
        self.device = torch.device(device)
        self.network = Kim2020SpatialActorCritic(config.hidden_channels).to(
            self.device
        )
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self.training_rng = np.random.default_rng(seed)
        self.training_seed = int(seed)
        self.policy_seed = int(seed if policy_seed is None else policy_seed)
        self.deployment_mode = deployment_mode
        self.learning_enabled = bool(learning_enabled)
        self.geometry = geometry_contract(env)

        self.assignment_count = 0
        self.gradient_steps = 0
        self.completed_outcome_count = 0
        self.censored_outcome_count = 0
        self.training_episode_count = 0
        self.skipped_episode_count = 0
        self.loss_history: list[float] = []
        self.actor_loss_history: list[float] = []
        self.value_loss_history: list[float] = []
        self.entropy_history: list[float] = []
        self.return_history: list[float] = []

        self.provisional_records: dict[str, Kim2020PlacementRecord] = {}
        self.pending_records: dict[str, Kim2020PlacementRecord] = {}
        self.completed_records: list[Kim2020PlacementRecord] = []
        self.last_episode_records: list[Kim2020PlacementRecord] = []
        self._episode_relocation_events = 0
        self._open_retrieval_relocations = 0
        self._unmatched_store_count = 0
        self._unmatched_delivery_count = 0
        self._episode_instance_id: Optional[str] = None
        self.last_episode_audit: dict = {}
        self._deployment_digest = ""
        self.set_learning_enabled(self.learning_enabled)

    def set_learning_enabled(self, enabled: bool) -> None:
        self.learning_enabled = bool(enabled)
        self.network.train(self.learning_enabled)
        if not self.learning_enabled:
            self._deployment_digest = _state_dict_digest(self.network.state_dict())

    def set_deployment_mode(self, mode: str, *, policy_seed: Optional[int] = None):
        if mode not in DEPLOYMENT_MODES:
            raise ValueError(f"deployment mode must be one of {DEPLOYMENT_MODES}")
        if self.learning_enabled:
            raise RuntimeError("deployment mode can only be set on a frozen source")
        self.deployment_mode = mode
        if policy_seed is not None:
            self.policy_seed = int(policy_seed)

    def on_episode_start(self) -> None:
        self.provisional_records = {}
        self.pending_records = {}
        self.completed_records = []
        self._episode_relocation_events = 0
        self._open_retrieval_relocations = 0
        self._unmatched_store_count = 0
        self._unmatched_delivery_count = 0
        instance = getattr(self.env, "current_episode_instance", None)
        self._episode_instance_id = getattr(instance, "instance_id", None)

    @staticmethod
    def _flat_index(cell: Cell, cols: int) -> int:
        return int(cell[0]) * int(cols) + int(cell[1])

    def _forward(self, observation: Kim2020Observation):
        features = torch.from_numpy(observation.spatial_features).unsqueeze(0)
        with torch.no_grad():
            return self.network(features.to(self.device))

    def score_candidates(self, yard, block, source, valid_candidates):
        if not valid_candidates:
            return np.empty((0,), dtype=np.float32)
        observation = build_spatial_observation(yard, block, valid_candidates)
        logits, _ = self._forward(observation)
        flat = logits[0]
        return np.asarray(
            [float(flat[self._flat_index(cell, yard.cols)].item()) for cell in observation.candidates],
            dtype=np.float32,
        )

    def _stateless_uniform(self, observation, block, source) -> float:
        instance = getattr(self.env, "current_episode_instance", None)
        instance_id = getattr(instance, "instance_id", self._episode_instance_id)
        features_digest = hashlib.sha256(
            observation.spatial_features.tobytes()
            + observation.candidate_mask.tobytes()
        ).hexdigest()
        key = {
            "deployment_digest": self._deployment_digest,
            "policy_seed": self.policy_seed,
            "instance_id": instance_id,
            "time_step": int(getattr(self.env, "time_steps", 0)),
            "block_label": block.label,
            "source": tuple(source),
            "candidates": observation.candidates,
            "observation": features_digest,
        }
        encoded = json.dumps(key, sort_keys=True, separators=(",", ":")).encode()
        integer = int.from_bytes(hashlib.sha256(encoded).digest()[:8], "big")
        return (integer + 0.5) / 2**64

    def _choose_index(self, observation, block, source) -> int:
        logits, _ = self._forward(observation)
        flat = logits[0]
        action_indices = [
            self._flat_index(cell, observation.candidate_mask.shape[1])
            for cell in observation.candidates
        ]
        legal_logits = torch.stack([flat[index] for index in action_indices])
        if self.learning_enabled:
            probabilities = torch.softmax(legal_logits, dim=0).cpu().numpy()
            return int(self.training_rng.choice(len(action_indices), p=probabilities))
        if self.deployment_mode == DEPLOYMENT_MAP:
            return max(
                range(len(action_indices)),
                key=lambda index: (float(legal_logits[index].item()), -index),
            )
        probabilities = torch.softmax(legal_logits, dim=0).cpu().numpy()
        uniform = self._stateless_uniform(observation, block, source)
        return min(
            int(np.searchsorted(np.cumsum(probabilities), uniform, side="right")),
            len(action_indices) - 1,
        )

    def _select(self, yard, block, source, valid_candidates, *, record: bool):
        if not valid_candidates:
            raise ValueError("Kim spatial selection requires a legal candidate")
        if record:
            instance = getattr(self.env, "current_episode_instance", None)
            instance_id = getattr(instance, "instance_id", None)
            if instance_id is None:
                raise RuntimeError("training selection requires an active instance")
            if block.label in self.pending_records:
                raise RuntimeError(f"duplicate placement record for {block.label}")
            provisional = self.provisional_records.get(block.label)
            if provisional is not None:
                if provisional.instance_id != instance_id:
                    raise RuntimeError(
                        f"cross-instance provisional placement for {block.label}"
                    )
                if provisional.chosen_cell not in valid_candidates:
                    raise RuntimeError(
                        f"stale provisional placement for {block.label}: "
                        f"{provisional.chosen_cell} is no longer legal"
                    )
                # A generator retry of the same unresolved storage macro is
                # not a second policy decision.  Preserve the sampled action,
                # RNG state, observation, and assignment counter exactly.
                return provisional.chosen_cell
        observation = build_spatial_observation(yard, block, valid_candidates)
        chosen_index = self._choose_index(observation, block, source)
        chosen = observation.candidates[chosen_index]
        if record:
            self.provisional_records[block.label] = Kim2020PlacementRecord(
                instance_id=instance_id,
                block_label=block.label,
                spatial_features=observation.spatial_features.copy(),
                candidate_mask=observation.candidate_mask.copy(),
                chosen_cell=chosen,
                chosen_flat_index=self._flat_index(chosen, yard.cols),
                selection_time_step=int(getattr(self.env, "time_steps", 0)),
            )
            self.assignment_count += 1
        return chosen

    def propose(self, yard, block, source, valid_candidates):
        return self._select(
            yard,
            block,
            source,
            valid_candidates,
            record=self.learning_enabled,
        )

    def preview(self, yard, block, source, valid_candidates):
        if self.learning_enabled:
            raise RuntimeError("Kim spatial preview requires a frozen source")
        return self._select(
            yard, block, source, valid_candidates, record=False
        )

    def on_assignment_committed(self, **_metadata) -> None:
        """Reservation commit is intentionally not treated as storage success."""

    def on_step(self, reward: float, info: dict) -> None:
        del reward
        # Frozen deployment is intentionally inference-only.  In particular,
        # a reservation preview must not create a hidden training/audit record.
        # Operational relocation counts remain available from the common
        # evaluator and retrieval executor audits.
        if not self.learning_enabled:
            return
        if info.get("relocated_block"):
            self._episode_relocation_events += 1
            self._open_retrieval_relocations += 1

        stored_label = info.get("stored_block")
        if stored_label:
            record = self.provisional_records.pop(str(stored_label), None)
            if record is None:
                self._unmatched_store_count += 1
            else:
                record.stored = True
                record.storage_time_step = int(getattr(self.env, "time_steps", 0))
                self.pending_records[record.block_label] = record

        delivered_label = info.get("delivered_block")
        if delivered_label:
            record = self.pending_records.pop(str(delivered_label), None)
            if record is None:
                self._unmatched_delivery_count += 1
            else:
                record.relocation_count = int(self._open_retrieval_relocations)
                record.reward = -float(record.relocation_count)
                self.completed_records.append(record)
            self._open_retrieval_relocations = 0

    def _learn_records(self, records: list[Kim2020PlacementRecord]) -> float:
        records = sorted(records, key=lambda record: record.selection_time_step)
        features = torch.from_numpy(
            np.stack([record.spatial_features for record in records])
        ).to(self.device)
        masks = torch.from_numpy(
            np.stack([record.candidate_mask for record in records])
        ).to(self.device).flatten(start_dim=1)
        actions = torch.as_tensor(
            [record.chosen_flat_index for record in records],
            dtype=torch.long,
            device=self.device,
        )
        running_return = 0.0
        return_targets = []
        for record in reversed(records):
            running_return = (
                float(record.reward) + self.config.gamma * running_return
            )
            record.discounted_return = running_return
            return_targets.append(running_return)
        return_targets.reverse()
        targets = torch.as_tensor(
            return_targets,
            dtype=torch.float32,
            device=self.device,
        ) * self.config.reward_scale

        final_loss = 0.0
        for _ in range(self.config.updates_per_episode):
            logits, values = self.network(features)
            masked_logits = logits.masked_fill(~masks, -torch.inf)
            distribution = torch.distributions.Categorical(logits=masked_logits)
            advantages = targets - values
            actor_loss = -(distribution.log_prob(actions) * advantages.detach()).mean()
            value_loss = F.mse_loss(values, targets)
            entropy = distribution.entropy().mean()
            loss = (
                actor_loss
                + self.config.value_coef * value_loss
                - self.config.entropy_coef * entropy
            )
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(self.network.parameters(), self.config.grad_clip)
            self.optimizer.step()
            self.gradient_steps += 1
            final_loss = float(loss.item())
            self.loss_history.append(final_loss)
            self.actor_loss_history.append(float(actor_loss.item()))
            self.value_loss_history.append(float(value_loss.item()))
            self.entropy_history.append(float(entropy.item()))
        return final_loss

    def on_episode_end(self, *, success: bool, truncated: bool) -> None:
        if not self.learning_enabled:
            self.last_episode_records = []
            self.last_episode_audit = {
                "instance_id": self._episode_instance_id,
                "success": bool(success),
                "truncated": bool(truncated),
                "outcome_tracking": False,
                "integrity": None,
                "trained": False,
                "selection_count": 0,
                "completed_count": 0,
                "censored_count": 0,
                "records": [],
            }
            self.provisional_records = {}
            self.pending_records = {}
            self.completed_records = []
            return
        censored = []
        reason = "episode_truncated" if truncated else "retrieval_not_observed"
        for record in (*self.provisional_records.values(), *self.pending_records.values()):
            record.censored = True
            record.censor_reason = reason
            censored.append(record)
        all_records = sorted(
            [*self.completed_records, *censored],
            key=lambda record: record.selection_time_step,
        )
        completed_relocations = sum(
            int(record.relocation_count or 0) for record in self.completed_records
        )
        integrity = bool(
            success
            and not truncated
            and not censored
            and self._unmatched_store_count == 0
            and self._unmatched_delivery_count == 0
            and self._open_retrieval_relocations == 0
            and completed_relocations == self._episode_relocation_events
        )
        # Episodic Monte Carlo returns are valid only when every placement
        # reward in the trajectory is observed.  Censored episodes are never
        # partially spliced into a shorter return sequence.
        trained = bool(
            self.learning_enabled and self.completed_records and integrity
        )
        if trained:
            self._learn_records(self.completed_records)
            self.training_episode_count += 1
        elif self.learning_enabled:
            self.skipped_episode_count += 1

        self.completed_outcome_count += len(self.completed_records)
        self.censored_outcome_count += len(censored)
        episode_return = -float(completed_relocations)
        self.return_history.append(episode_return)
        self.last_episode_records = all_records
        self.last_episode_audit = {
            "instance_id": self._episode_instance_id,
            "success": bool(success),
            "truncated": bool(truncated),
            "outcome_tracking": True,
            "integrity": integrity,
            "trained": trained,
            "selection_count": len(all_records),
            "completed_count": len(self.completed_records),
            "censored_count": len(censored),
            "relocation_event_count": self._episode_relocation_events,
            "credited_relocation_count": completed_relocations,
            "unmatched_store_count": self._unmatched_store_count,
            "unmatched_delivery_count": self._unmatched_delivery_count,
            "open_retrieval_relocation_count": self._open_retrieval_relocations,
            "records": [record.audit_dict() for record in all_records],
        }
        self.provisional_records = {}
        self.pending_records = {}
        self.completed_records = []

    def audit(self) -> dict:
        return {
            "assignment_source": METHOD_NAME,
            "assignment_source_family": "learned_spatial_assignment_policy",
            "assignment_source_version": ARCHITECTURE_NAME,
            "selector_feature_version": FEATURE_VERSION,
            "reward_contract": REWARD_CONTRACT,
            "return_contract": RETURN_CONTRACT,
            "action_mapping": ACTION_MAPPING,
            "learning_enabled": self.learning_enabled,
            "deployment_mode": self.deployment_mode,
            "policy_seed": self.policy_seed,
            "assignment_count": self.assignment_count,
            "gradient_steps": self.gradient_steps,
            "completed_outcome_count": self.completed_outcome_count,
            "censored_outcome_count": self.censored_outcome_count,
            "last_episode": dict(self.last_episode_audit),
        }

    def checkpoint(self, **metadata) -> dict:
        state = self.network.state_dict()
        return {
            "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
            "assignment_source": METHOD_NAME,
            "selector_feature_version": FEATURE_VERSION,
            "selector_architecture": ARCHITECTURE_NAME,
            "adaptation_status": "adaptation_not_exact_reproduction",
            "paper_doi": "10.1080/00207543.2020.1748247",
            "reward_contract": REWARD_CONTRACT,
            "return_contract": RETURN_CONTRACT,
            "action_mapping": ACTION_MAPPING,
            "input_channels": INPUT_CHANNELS,
            "geometry_contract": self.geometry,
            "deployment_mode": self.deployment_mode,
            "config": asdict(self.config),
            "network_state_dict": state,
            "deployment_state_dict": state,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "training_rng_state": self.training_rng.bit_generator.state,
            "training_seed": self.training_seed,
            "assignment_count": self.assignment_count,
            "gradient_steps": self.gradient_steps,
            "completed_outcome_count": self.completed_outcome_count,
            "censored_outcome_count": self.censored_outcome_count,
            "training_episode_count": self.training_episode_count,
            "skipped_episode_count": self.skipped_episode_count,
            "loss_history": list(self.loss_history),
            "actor_loss_history": list(self.actor_loss_history),
            "value_loss_history": list(self.value_loss_history),
            "entropy_history": list(self.entropy_history),
            "return_history": list(self.return_history),
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
        policy_seed: Optional[int] = None,
        deployment_mode: Optional[str] = None,
        allow_geometry_shift: bool = False,
    ) -> "Kim2020A3CSpatialSource":
        expected = {
            "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
            "assignment_source": METHOD_NAME,
            "selector_feature_version": FEATURE_VERSION,
            "selector_architecture": ARCHITECTURE_NAME,
            "reward_contract": REWARD_CONTRACT,
            "return_contract": RETURN_CONTRACT,
            "action_mapping": ACTION_MAPPING,
            "input_channels": INPUT_CHANNELS,
        }
        mismatches = [
            key for key, value in expected.items() if payload.get(key) != value
        ]
        if mismatches:
            raise ValueError(
                "Incompatible Kim spatial checkpoint metadata: "
                + ", ".join(mismatches)
            )
        live_geometry = geometry_contract(env)
        if (
            not allow_geometry_shift
            and payload.get("geometry_contract") != live_geometry
        ):
            raise ValueError("Kim spatial checkpoint geometry does not match environment")
        mode = deployment_mode or payload.get(
            "deployment_mode", DEPLOYMENT_STOCHASTIC
        )
        source = cls(
            env,
            Kim2020Config(**payload["config"]),
            seed=seed,
            policy_seed=seed if policy_seed is None else policy_seed,
            learning_enabled=learning_enabled,
            deployment_mode=mode,
            device=device,
        )
        deployment = payload.get(
            "deployment_state_dict", payload["network_state_dict"]
        )
        source.network.load_state_dict(
            payload["network_state_dict"] if learning_enabled else deployment
        )
        if learning_enabled and payload.get("optimizer_state_dict"):
            source.optimizer.load_state_dict(payload["optimizer_state_dict"])
            for optimizer_state in source.optimizer.state.values():
                for key, value in optimizer_state.items():
                    if isinstance(value, torch.Tensor):
                        optimizer_state[key] = value.to(source.device)
        if learning_enabled and payload.get("training_rng_state"):
            source.training_rng.bit_generator.state = payload["training_rng_state"]
        source.assignment_count = int(payload.get("assignment_count", 0))
        source.gradient_steps = int(payload.get("gradient_steps", 0))
        source.completed_outcome_count = int(
            payload.get("completed_outcome_count", 0)
        )
        source.censored_outcome_count = int(
            payload.get("censored_outcome_count", 0)
        )
        source.training_episode_count = int(
            payload.get("training_episode_count", 0)
        )
        source.skipped_episode_count = int(payload.get("skipped_episode_count", 0))
        source.loss_history = [float(value) for value in payload.get("loss_history", ())]
        source.actor_loss_history = [
            float(value) for value in payload.get("actor_loss_history", ())
        ]
        source.value_loss_history = [
            float(value) for value in payload.get("value_loss_history", ())
        ]
        source.entropy_history = [
            float(value) for value in payload.get("entropy_history", ())
        ]
        source.return_history = [
            float(value) for value in payload.get("return_history", ())
        ]
        source.set_learning_enabled(learning_enabled)
        return source


__all__ = [
    "ACTION_MAPPING",
    "ARCHITECTURE_NAME",
    "CHECKPOINT_SCHEMA_VERSION",
    "DEPLOYMENT_MAP",
    "DEPLOYMENT_MODES",
    "DEPLOYMENT_STOCHASTIC",
    "FEATURE_VERSION",
    "INPUT_CHANNELS",
    "Kim2020A3CSpatialSource",
    "Kim2020Config",
    "Kim2020Observation",
    "Kim2020PlacementRecord",
    "Kim2020SpatialActorCritic",
    "METHOD_NAME",
    "REWARD_CONTRACT",
    "RETURN_CONTRACT",
    "build_spatial_observation",
    "geometry_contract",
    "kim2020_deployment_digest",
]
