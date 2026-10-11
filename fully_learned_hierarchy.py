"""Fully learned, parameterized, mode-regularized Track-B hierarchy.

This controller is deliberately separate from ``relational_scheduler`` v1.
It represents storage as ``AcceptStore(block, cell)``, executes the selected
cell through the strict reservation contract, and trains every macro against
one cardinality-normalized SMDP continuation value.  The repaired curriculum
keeps teacher information out of that Bellman value: teacher mixing, behavior
priors, and retained imitation are optimization aids only.
"""

from __future__ import annotations

from collections import Counter, deque
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
import math
import random
from typing import Optional, Sequence

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from example.Options.AcceptStoreOption import ReservedAcceptStoreOption
from example.Options.RetrieveDeliverOption import StrictRetrieveDeliverOption
from example.Options.StrategicDeferOption import StrategicDeferOption
from example.Options.selector_v5 import ExplicitCellAssignmentRegistry
from example.controller_observation import (
    BLOCK_FEATURE_NAMES,
    GLOBAL_FEATURE_NAMES,
    OnlineManifestTimingObservationEncoder,
)
from example.controller_options import FULLY_LEARNED_RESERVED_ACTION_INTERFACE
from options_agent import canonical_manager_options, option_identifier
from PSLAP.dynamic_yard import BlockView, YardSnapshot, shortest_clear_path
from PSLAP.reg_selector_v4 import CANDIDATE_FEATURE_DIM as REG_CELL_FEATURE_DIM
from PSLAP.reg_selector_v5 import build_assignment_observation
from PSLAP.retrieval_context import retrieval_planning_context
from PSLAP.retrieval_dispatch import plan_retrieval
from PSLAP.track_a import shared_candidate_mask


FULLY_LEARNED_CHECKPOINT_SCHEMA_VERSION = 4
FULLY_LEARNED_CONTROLLER_ARCHITECTURE = (
    "relational_parameterized_mode_regularized_smdp_v4"
)
FULLY_LEARNED_ACTION_INTERFACE = FULLY_LEARNED_RESERVED_ACTION_INTERFACE
FULLY_LEARNED_NETWORK_ARCHITECTURE = "shared_deepset_candidate_q_v2"
FULLY_LEARNED_REPLAY_VERSION = (
    "variable_candidate_mode_balanced_common_continuation_smdp_v4"
)
FULLY_LEARNED_BACKUP_VERSION = "nested_logmeanexp_raw_q_common_smdp_v2"
FULLY_LEARNED_CANDIDATE_FEATURE_VERSION = 2
TRAINING_PHASES = ("imitation", "temporal", "spatial", "joint")
FULLY_LEARNED_POLICY_REALIZATIONS = ("hierarchical_map",)
FULLY_LEARNED_REPLAY_SAMPLING = (
    "uniform",
    "mode_balanced",
    "regime_mode_balanced",
)

MODE_NAMES = ("accept", "retrieve", "defer")
MODE_TO_ID = {name: index for index, name in enumerate(MODE_NAMES)}

HISTORY_FEATURE_NAMES = (
    "recent_accept_fraction",
    "recent_retrieve_fraction",
    "recent_defer_fraction",
    "recent_mean_duration",
    "recent_mean_discounted_return",
    "recent_failure_fraction",
)

CANDIDATE_FEATURE_NAMES = (
    "mode_accept",
    "mode_retrieve",
    "mode_defer",
    "block_present",
    *(f"block_{name}" for name in BLOCK_FEATURE_NAMES),
    "cell_present",
    *(f"cell_geometry_{index}" for index in range(REG_CELL_FEATURE_DIM)),
    "duration_valid",
    "pickup_duration",
    "storage_duration",
    "total_duration",
    "retrieval_plan_valid",
    "retrieval_eta",
    "retrieval_signed_slack",
    "retrieval_due",
    "retrieval_relocation_fraction",
    "retrieval_live_first_leg",
    "inbound_present",
    "lookahead_valid",
    "lookahead_error_now",
    "lookahead_error_after_accept",
    "lookahead_cost_delta",
    "defer_horizon",
)

_BLOCK_OFFSET = 4
_CELL_PRESENT_INDEX = _BLOCK_OFFSET + len(BLOCK_FEATURE_NAMES)
_CELL_OFFSET = _CELL_PRESENT_INDEX + 1
CELL_FEATURE_SLICE = slice(_CELL_OFFSET, _CELL_OFFSET + REG_CELL_FEATURE_DIM)
_DURATION_OFFSET = CELL_FEATURE_SLICE.stop
_PLAN_OFFSET = _DURATION_OFFSET + 4
_INBOUND_INDEX = _PLAN_OFFSET + 6
_LOOKAHEAD_OFFSET = _INBOUND_INDEX + 1
_DEFER_INDEX = _LOOKAHEAD_OFFSET + 4


class FullyLearnedInfeasible(RuntimeError):
    """Raised when no strict macro is available at a decision epoch."""


def masked_logmeanexp(
    values: torch.Tensor,
    mask: torch.Tensor,
    temperature: float | torch.Tensor,
    dim: int = -1,
) -> torch.Tensor:
    """Uniform-reference log-sum-exp over valid entries.

    Subtracting ``log(valid_count)`` makes the result invariant to replicating
    the complete representation of a control set. Empty reductions are errors,
    rather than finite sentinels that could silently enter a Bellman target.
    """

    if not torch.is_tensor(values) or not torch.is_tensor(mask):
        raise TypeError("values and mask must be tensors")
    if values.shape != mask.shape:
        raise ValueError("values and mask must have the same shape")
    tau = torch.as_tensor(temperature, dtype=values.dtype, device=values.device)
    if tau.numel() != 1 or not bool(torch.isfinite(tau)) or float(tau) <= 0:
        raise ValueError("temperature must be finite and positive")
    valid = mask.to(dtype=torch.bool)
    count = valid.sum(dim=dim)
    if bool((count == 0).any()):
        raise ValueError("masked log-mean-exp received an empty valid set")
    scaled = (values / tau).masked_fill(~valid, -torch.inf)
    return tau * (
        torch.logsumexp(scaled, dim=dim)
        - torch.log(count.to(dtype=values.dtype))
    )


def _mode_values_1d(
    q_values: torch.Tensor,
    candidate_mask: torch.Tensor,
    mode_ids: torch.Tensor,
    within_temperatures: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if q_values.ndim != 1:
        raise ValueError("mode helper expects one candidate vector")
    if q_values.shape != candidate_mask.shape or q_values.shape != mode_ids.shape:
        raise ValueError("candidate values, masks, and modes must align")
    temperatures = torch.as_tensor(
        within_temperatures, dtype=q_values.dtype, device=q_values.device
    ).flatten()
    live_values = []
    live_modes = []
    for mode in range(int(temperatures.numel())):
        mask = candidate_mask.to(torch.bool) & (mode_ids == mode)
        if not bool(mask.any()):
            continue
        live_values.append(
            masked_logmeanexp(q_values, mask, temperatures[mode], dim=0)
        )
        live_modes.append(mode)
    if not live_values:
        raise ValueError("no live temporal mode")
    return (
        torch.stack(live_values),
        torch.as_tensor(live_modes, dtype=torch.long, device=q_values.device),
    )


def mode_regularized_value(
    q_values: torch.Tensor,
    candidate_mask: torch.Tensor,
    mode_ids: torch.Tensor,
    within_temperatures: torch.Tensor | Sequence[float],
    mode_temperature: float,
    mode_priors: Optional[torch.Tensor | Sequence[float]] = None,
) -> torch.Tensor:
    """Nested normalized continuation value for one variable action set."""

    values, live_modes = _mode_values_1d(
        q_values,
        candidate_mask,
        mode_ids.to(dtype=torch.long, device=q_values.device),
        torch.as_tensor(within_temperatures, device=q_values.device),
    )
    tau = torch.as_tensor(
        mode_temperature, dtype=q_values.dtype, device=q_values.device
    )
    if tau.numel() != 1 or not bool(torch.isfinite(tau)) or float(tau) <= 0:
        raise ValueError("mode temperature must be finite and positive")
    if mode_priors is None:
        priors = torch.ones_like(values)
    else:
        all_priors = torch.as_tensor(
            mode_priors, dtype=q_values.dtype, device=q_values.device
        ).flatten()
        if int(all_priors.numel()) <= int(live_modes.max()):
            raise ValueError("mode prior vector is too short")
        priors = all_priors[live_modes]
    if bool((priors <= 0).any()) or not bool(torch.isfinite(priors).all()):
        raise ValueError("live mode priors must be finite and positive")
    priors = priors / priors.sum()
    return tau * torch.logsumexp(values / tau + torch.log(priors), dim=0)


def smdp_target(
    reward: torch.Tensor,
    duration: torch.Tensor,
    done: torch.Tensor,
    next_value: torch.Tensor,
    gamma: float,
    reward_scale: float = 1.0,
) -> torch.Tensor:
    """One common variable-duration target for every temporal mode."""

    if not 0.0 < float(gamma) <= 1.0:
        raise ValueError("gamma must be in (0, 1]")
    reward = torch.as_tensor(reward)
    device = reward.device
    dtype = reward.dtype
    duration = torch.as_tensor(duration, device=device)
    done = torch.as_tensor(done, dtype=torch.bool, device=device)
    next_value = torch.as_tensor(next_value, dtype=dtype, device=device)
    discount = torch.pow(
        torch.as_tensor(float(gamma), dtype=dtype, device=device),
        duration.to(dtype=dtype),
    )
    return float(reward_scale) * reward + discount * (~done).to(dtype) * next_value


@dataclass(frozen=True)
class FullyLearnedConfig:
    block_embedding_dim: int = 64
    global_embedding_dim: int = 64
    candidate_embedding_dim: int = 64
    context_dim: int = 128
    history_length: int = 8
    tau_accept: float = 0.1
    tau_retrieve: float = 0.1
    tau_defer: float = 0.1
    tau_mode: float = 1.0
    teacher_coefficient: float = 1.0
    teacher_prior_strength: float = 5.0
    teacher_score_temperature: float = 1.0
    spatial_distillation_weight: float = 1.0
    joint_accept_exploration_top_k: int = 8
    lookahead_margin_steps: float = 2.0
    failure_penalty: float = -50.0
    huber_delta: float = 1.0

    def __post_init__(self):
        for name in (
            "block_embedding_dim",
            "global_embedding_dim",
            "candidate_embedding_dim",
            "context_dim",
            "history_length",
        ):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        for name in (
            "tau_accept",
            "tau_retrieve",
            "tau_defer",
            "tau_mode",
            "teacher_score_temperature",
            "huber_delta",
        ):
            if not math.isfinite(float(getattr(self, name))) or float(
                getattr(self, name)
            ) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in (
            "teacher_coefficient",
            "teacher_prior_strength",
            "spatial_distillation_weight",
            "lookahead_margin_steps",
        ):
            if not math.isfinite(float(getattr(self, name))) or float(
                getattr(self, name)
            ) < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if int(self.joint_accept_exploration_top_k) <= 0:
            raise ValueError("joint_accept_exploration_top_k must be positive")
        if not math.isfinite(float(self.failure_penalty)) or self.failure_penalty > 0:
            raise ValueError("failure_penalty must be finite and nonpositive")

    @property
    def within_temperatures(self) -> tuple[float, float, float]:
        return (self.tau_accept, self.tau_retrieve, self.tau_defer)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict):
        allowed = cls.__dataclass_fields__
        return cls(**{key: values[key] for key in allowed if key in values})


@dataclass(frozen=True)
class FullyLearnedState:
    global_features: np.ndarray
    block_features: np.ndarray
    block_mask: np.ndarray

    def copy(self):
        return FullyLearnedState(
            self.global_features.astype(np.float32, copy=True),
            self.block_features.astype(np.float32, copy=True),
            self.block_mask.astype(bool, copy=True),
        )


@dataclass(frozen=True)
class FullyLearnedCandidate:
    key: str
    mode: str
    mode_id: int
    features: np.ndarray
    option: object
    teacher_prior: float = 0.0
    target_label: Optional[str] = None
    chosen_cell: Optional[tuple[int, int]] = None
    predicted_duration: Optional[int] = None
    teacher_score: Optional[float] = None


@dataclass(frozen=True)
class FullyLearnedSnapshot:
    state: FullyLearnedState
    candidates: tuple[FullyLearnedCandidate, ...]
    teacher_key: str


@dataclass(frozen=True)
class FullyLearnedTransition:
    state: FullyLearnedState
    candidate_features: np.ndarray
    candidate_prior: float
    candidate_mode: int
    reward: float
    duration: int
    next_state: Optional[FullyLearnedState]
    next_candidate_features: np.ndarray
    next_candidate_priors: np.ndarray
    next_candidate_modes: np.ndarray
    done: bool
    terminal: bool
    truncated: bool
    failed: bool
    boundary_penalty: float = 0.0
    regime_id: str = "legacy"

    def numeric_copy(self):
        return FullyLearnedTransition(
            state=self.state.copy(),
            candidate_features=self.candidate_features.astype(
                np.float32, copy=True
            ),
            candidate_prior=float(self.candidate_prior),
            candidate_mode=int(self.candidate_mode),
            reward=float(self.reward),
            duration=int(self.duration),
            next_state=None if self.next_state is None else self.next_state.copy(),
            next_candidate_features=self.next_candidate_features.astype(
                np.float32, copy=True
            ),
            next_candidate_priors=self.next_candidate_priors.astype(
                np.float32, copy=True
            ),
            next_candidate_modes=self.next_candidate_modes.astype(
                np.int64, copy=True
            ),
            done=bool(self.done),
            terminal=bool(self.terminal),
            truncated=bool(self.truncated),
            failed=bool(self.failed),
            boundary_penalty=float(self.boundary_penalty),
            regime_id=str(self.regime_id),
        )


class FullyLearnedReplayBuffer:
    def __init__(self, capacity: int):
        if int(capacity) <= 0:
            raise ValueError("replay capacity must be positive")
        self.capacity = int(capacity)
        self.memory = deque(maxlen=self.capacity)

    def add(self, transition: FullyLearnedTransition):
        self.memory.append(transition.numeric_copy())

    def sample(self, batch_size: int, rng: random.Random):
        return rng.sample(list(self.memory), int(batch_size))

    def sample_mode_balanced(self, batch_size: int, rng: random.Random):
        """Sample without replacement while equalizing represented modes.

        Buckets are rebuilt from the live deque on every sample.  This keeps
        capacity eviction and checkpoint restoration free of stale indices.
        Scarce buckets contribute every available item; the remainder is then
        filled uniformly from transitions not selected yet.
        """

        batch_size = int(batch_size)
        memory = list(self.memory)
        if batch_size <= 0:
            raise ValueError("batch size must be positive")
        if batch_size > len(memory):
            raise ValueError("batch size exceeds replay size")
        groups = {}
        for index, transition in enumerate(memory):
            groups.setdefault(int(transition.candidate_mode), []).append(index)
        live_modes = sorted(groups)
        base, remainder = divmod(batch_size, len(live_modes))
        selected = []
        selected_set = set()
        for position, mode in enumerate(live_modes):
            quota = base + int(position < remainder)
            take = min(quota, len(groups[mode]))
            chosen = rng.sample(groups[mode], take)
            selected.extend(chosen)
            selected_set.update(chosen)
        missing = batch_size - len(selected)
        if missing:
            remaining = [
                index for index in range(len(memory)) if index not in selected_set
            ]
            selected.extend(rng.sample(remaining, missing))
        rng.shuffle(selected)
        return [memory[index] for index in selected]

    def sample_regime_mode_balanced(
        self, batch_size: int, rng: random.Random
    ):
        """Balance first across regimes, then temporal modes.

        Each live ``(regime, mode)`` cell receives an equal provisional quota.
        Sparse cells contribute all available records and the unfilled portion
        is sampled uniformly without replacement.  This prevents regimes with
        more blocks (and therefore more macro transitions per episode) from
        dominating mixed-domain replay.
        """

        batch_size = int(batch_size)
        memory = list(self.memory)
        if batch_size <= 0:
            raise ValueError("batch size must be positive")
        if batch_size > len(memory):
            raise ValueError("batch size exceeds replay size")
        groups = {}
        for index, transition in enumerate(memory):
            key = (
                str(getattr(transition, "regime_id", "legacy")),
                int(transition.candidate_mode),
            )
            groups.setdefault(key, []).append(index)
        live_regimes = sorted({key[0] for key in groups})
        regime_base, regime_remainder = divmod(
            batch_size, len(live_regimes)
        )
        selected = []
        selected_set = set()
        for regime_position, regime in enumerate(live_regimes):
            regime_quota = regime_base + int(
                regime_position < regime_remainder
            )
            modes = sorted(key[1] for key in groups if key[0] == regime)
            mode_base, mode_remainder = divmod(regime_quota, len(modes))
            for mode_position, mode in enumerate(modes):
                quota = mode_base + int(mode_position < mode_remainder)
                bucket = groups[(regime, mode)]
                chosen = rng.sample(bucket, min(quota, len(bucket)))
                selected.extend(chosen)
                selected_set.update(chosen)
        missing = batch_size - len(selected)
        if missing:
            remaining = [
                index for index in range(len(memory))
                if index not in selected_set
            ]
            selected.extend(rng.sample(remaining, missing))
        rng.shuffle(selected)
        return [memory[index] for index in selected]

    def mode_counts(self):
        return dict(Counter(int(item.candidate_mode) for item in self.memory))

    def regime_mode_counts(self):
        return dict(
            Counter(
                (
                    str(getattr(item, "regime_id", "legacy")),
                    int(item.candidate_mode),
                )
                for item in self.memory
            )
        )

    def clear(self):
        self.memory.clear()

    def state_dict(self):
        return {"capacity": self.capacity, "memory": list(self.memory)}

    def load_state_dict(self, state):
        if int(state["capacity"]) != self.capacity:
            raise ValueError("replay capacity mismatch")
        self.memory = deque(
            (item.numeric_copy() for item in state["memory"]),
            maxlen=self.capacity,
        )

    def __len__(self):
        return len(self.memory)


# Compatibility aliases used by focused tests and earlier design notes.
FullyLearnedReplay = FullyLearnedReplayBuffer
FullyLearnedQTransition = FullyLearnedTransition


class FullyLearnedQNetwork(nn.Module):
    """One shared scalar Q scorer for all parameterized macro candidates."""

    def __init__(self, config: FullyLearnedConfig, spatial_network, *, seed: int):
        super().__init__()
        self.config = config
        global_dim = len(GLOBAL_FEATURE_NAMES) + len(HISTORY_FEATURE_NAMES)
        block_dim = len(BLOCK_FEATURE_NAMES)
        candidate_dim = len(CANDIDATE_FEATURE_NAMES)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(seed))
            self.block_encoder = nn.Sequential(
                nn.Linear(block_dim, config.block_embedding_dim),
                nn.LayerNorm(config.block_embedding_dim),
                nn.ReLU(),
                nn.Linear(config.block_embedding_dim, config.block_embedding_dim),
                nn.ReLU(),
            )
            self.global_encoder = nn.Sequential(
                nn.Linear(global_dim, config.global_embedding_dim),
                nn.LayerNorm(config.global_embedding_dim),
                nn.ReLU(),
            )
            self.state_encoder = nn.Sequential(
                nn.Linear(
                    config.global_embedding_dim
                    + 2 * config.block_embedding_dim,
                    config.context_dim,
                ),
                nn.LayerNorm(config.context_dim),
                nn.ReLU(),
            )
            self.candidate_encoder = nn.Sequential(
                nn.Linear(candidate_dim, config.candidate_embedding_dim),
                nn.LayerNorm(config.candidate_embedding_dim),
                nn.ReLU(),
                nn.Linear(
                    config.candidate_embedding_dim,
                    config.candidate_embedding_dim,
                ),
                nn.ReLU(),
            )

        if spatial_network is None or not hasattr(
            spatial_network, "candidate_encoder"
        ):
            raise ValueError("joint controller requires a REG candidate encoder")
        # Build the composite module on CPU, then let the owning agent move the
        # complete network once. This avoids a CPU/CUDA mismatch during shape
        # inference when the frozen REG checkpoint was loaded directly on GPU.
        self.spatial_encoder = deepcopy(
            spatial_network.candidate_encoder
        ).to("cpu")
        with torch.no_grad():
            spatial_dim = int(
                self.spatial_encoder(torch.zeros(1, 1, REG_CELL_FEATURE_DIM)).shape[-1]
            )
        joint_dim = (
            config.context_dim + config.candidate_embedding_dim + spatial_dim
        )
        self.fusion = nn.Sequential(
            nn.Linear(joint_dim, config.context_dim),
            nn.LayerNorm(config.context_dim),
            nn.ReLU(),
            nn.Linear(config.context_dim, config.context_dim),
            nn.ReLU(),
        )
        self.q_head = nn.Linear(config.context_dim, 1)

    def set_spatial_trainable(self, enabled: bool):
        self.spatial_encoder.requires_grad_(bool(enabled))
        self.spatial_encoder.train(bool(enabled) and self.training)

    def forward(
        self,
        global_features: torch.Tensor,
        block_features: torch.Tensor,
        block_mask: torch.Tensor,
        candidate_features: torch.Tensor,
        candidate_priors: Optional[torch.Tensor] = None,
        teacher_coefficient: float = 0.0,
    ) -> torch.Tensor:
        encoded_blocks = self.block_encoder(block_features)
        mask = block_mask.unsqueeze(-1).to(encoded_blocks.dtype)
        count = mask.sum(dim=1).clamp_min(1.0)
        mean_pool = (encoded_blocks * mask).sum(dim=1) / count
        negative = torch.finfo(encoded_blocks.dtype).min
        max_pool = encoded_blocks.masked_fill(mask == 0, negative).max(dim=1).values
        no_blocks = ~block_mask.to(torch.bool).any(dim=1)
        max_pool = torch.where(
            no_blocks.unsqueeze(-1), torch.zeros_like(max_pool), max_pool
        )
        context = self.state_encoder(
            torch.cat((self.global_encoder(global_features), mean_pool, max_pool), -1)
        )
        candidate_context = self.candidate_encoder(candidate_features)
        cell = candidate_features[..., CELL_FEATURE_SLICE]
        spatial = self.spatial_encoder(cell)
        accept_mask = candidate_features[..., 0:1]
        spatial = spatial * accept_mask
        expanded = context.unsqueeze(1).expand(-1, candidate_features.shape[1], -1)
        hidden = self.fusion(torch.cat((expanded, candidate_context, spatial), -1))
        q = self.q_head(hidden).squeeze(-1)
        if candidate_priors is not None and float(teacher_coefficient) != 0.0:
            q = q + float(teacher_coefficient) * candidate_priors
        return q


# Short alias requested by downstream experiments.
FullyLearnedNetwork = FullyLearnedQNetwork


class FullyLearnedCandidateBuilder:
    """Capture one immutable, online-safe parameterized macro action set."""

    def __init__(self, env, encoder, config: FullyLearnedConfig):
        if not isinstance(encoder, OnlineManifestTimingObservationEncoder):
            raise ValueError("fully learned controller requires online observations")
        self.env = env
        self.encoder = encoder
        self.config = config
        managers = canonical_manager_options(env.options)
        accepts = [item for item in managers if isinstance(item, ReservedAcceptStoreOption)]
        defers = [item for item in managers if isinstance(item, StrategicDeferOption)]
        retrievals = [
            item for item in managers if isinstance(item, StrictRetrieveDeliverOption)
        ]
        if len(accepts) != 1 or len(defers) != 1:
            raise ValueError("v2 interface requires one reserved AcceptStore and Defer")
        if len(retrievals) != len(env.blocks):
            raise ValueError("v2 interface requires one strict retrieval per block")
        self.accept_option = accepts[0]
        self.defer_option = defers[0]
        self.retrieval_options = tuple(sorted(retrievals, key=lambda x: x.block_index))
        self.registry = self.accept_option.selector
        if not isinstance(self.registry, ExplicitCellAssignmentRegistry):
            raise ValueError("v2 requires the explicit-cell proposal registry")
        self.relocation_selector = self.retrieval_options[0].relocation_selector

    def capture_state(self, state, history_features) -> FullyLearnedState:
        observation = self.encoder.capture(state)
        history = np.asarray(history_features, dtype=np.float32)
        if history.shape != (len(HISTORY_FEATURE_NAMES),):
            raise ValueError("history feature contract changed")
        return FullyLearnedState(
            global_features=np.concatenate(
                (observation.global_features, history)
            ).astype(np.float32, copy=True),
            block_features=observation.block_features.astype(np.float32, copy=True),
            block_mask=observation.block_mask.astype(bool, copy=True),
        )

    def _inbound(self):
        return next(
            (
                block
                for block in self.env.blocks
                if block.position == self.env.pickup_cell
                and not block.carrying
                and not block.stored
                and not block.delivered
                and block.storage_location is None
            ),
            None,
        )

    @staticmethod
    def _retrieval_score(item):
        option, plan = item
        return (
            float(plan.slack),
            len(plan.relocations),
            int(plan.estimated_steps),
            int(option.block_index),
        )

    def _projected_plan(self, inbound, cell, duration, target_label, yard):
        aged = tuple(
            replace(
                block,
                remaining_time=float(block.remaining_time) - int(duration),
            )
            for block in yard.blocks
        )
        projected = replace(
            yard,
            blocks=aged
            + (
                BlockView(
                    inbound.label,
                    tuple(cell),
                    float(inbound.storage_steps_needed),
                ),
            ),
        )
        return plan_retrieval(
            projected,
            tuple(cell),
            target_label,
            relocation_selector=self.relocation_selector,
        )

    def _features(
        self,
        mode,
        block_row,
        *,
        cell_features=None,
        pickup_steps=None,
        storage_steps=None,
        plan=None,
        inbound=False,
        lookahead=None,
    ):
        values = np.zeros(len(CANDIDATE_FEATURE_NAMES), dtype=np.float32)
        values[MODE_TO_ID[mode]] = 1.0
        if block_row is not None:
            values[3] = 1.0
            values[_BLOCK_OFFSET : _BLOCK_OFFSET + len(BLOCK_FEATURE_NAMES)] = block_row
        if cell_features is not None:
            values[_CELL_PRESENT_INDEX] = 1.0
            values[CELL_FEATURE_SLICE] = cell_features
        if pickup_steps is not None and storage_steps is not None:
            scale = max(1.0, float(self.env.MAX_T))
            values[_DURATION_OFFSET] = 1.0
            values[_DURATION_OFFSET + 1] = np.clip(pickup_steps / scale, 0, 1)
            values[_DURATION_OFFSET + 2] = np.clip(storage_steps / scale, 0, 1)
            values[_DURATION_OFFSET + 3] = np.clip(
                (pickup_steps + storage_steps) / scale, 0, 1
            )
        if plan is not None:
            scale = max(1.0, float(self.env.MAX_T))
            values[_PLAN_OFFSET] = 1.0
            values[_PLAN_OFFSET + 1] = np.clip(plan.estimated_steps / scale, 0, 1)
            values[_PLAN_OFFSET + 2] = np.clip(plan.slack / scale, -1, 1)
            values[_PLAN_OFFSET + 3] = float(plan.slack <= 0)
            values[_PLAN_OFFSET + 4] = len(plan.relocations) / max(1, len(self.env.blocks))
            values[_PLAN_OFFSET + 5] = 1.0
        values[_INBOUND_INDEX] = float(inbound)
        if lookahead is not None:
            scale = max(1.0, float(self.env.MAX_T))
            error_now, error_after = lookahead
            values[_LOOKAHEAD_OFFSET] = float(error_after is not None)
            values[_LOOKAHEAD_OFFSET + 1] = np.clip(error_now / scale, -1, 1)
            if error_after is not None:
                values[_LOOKAHEAD_OFFSET + 2] = np.clip(error_after / scale, -1, 1)
                values[_LOOKAHEAD_OFFSET + 3] = np.clip(
                    (abs(error_after) - abs(error_now)) / scale, -1, 1
                )
        if mode == "defer":
            values[_DEFER_INDEX] = np.clip(
                self.defer_option.max_defer_steps / max(1.0, float(self.env.MAX_T)),
                0,
                1,
            )
        return values

    def capture(
        self,
        state,
        history_features,
        phase: str,
        *,
        teacher_supervision: bool = False,
    ) -> FullyLearnedSnapshot:
        if phase not in TRAINING_PHASES:
            raise ValueError(f"unknown phase: {phase!r}")
        if any(block.carrying for block in self.env.blocks):
            raise FullyLearnedInfeasible("decision requested while carrying")
        learned_state = self.capture_state(state, history_features)
        inbound = self._inbound()
        inbound_present = inbound is not None
        planning = retrieval_planning_context(
            self.env, relocation_selector=self.relocation_selector
        )
        raw_plans = []
        executable = []
        for option in self.retrieval_options:
            plan = planning.plan(option.target_label)
            if plan is None:
                continue
            raw_plans.append((option, plan))
            if option.initiation(state):
                executable.append((option, plan))
        raw_plans.sort(key=self._retrieval_score)
        executable.sort(key=self._retrieval_score)

        candidates: list[FullyLearnedCandidate] = []
        teacher_cell = None
        teacher_cell_score = None
        accept_durations = {}
        accept_rows = {}
        yard = planning.yard
        if inbound is not None and self.accept_option.initiation(state):
            view = BlockView(
                inbound.label,
                tuple(inbound.position),
                float(inbound.get_remaining_storage_time()),
            )
            cells = tuple(shared_candidate_mask(yard, view, tuple(inbound.position)))
            if cells:
                reg_observation = build_assignment_observation(
                    self.env, yard, view, tuple(inbound.position), cells
                )
                needs_reg_scores = bool(
                    phase in {"imitation", "temporal"} or teacher_supervision
                )
                if needs_reg_scores:
                    scores = self.registry.source.score_candidates(
                        yard, view, tuple(inbound.position), cells
                    )
                    teacher_index = max(
                        range(len(cells)),
                        key=lambda index: (float(scores[index]), -index),
                    )
                    teacher_cell = tuple(cells[teacher_index])
                    teacher_cell_score = float(scores[teacher_index])
                    visible_indices = (
                        (teacher_index,)
                        if phase == "temporal"
                        else range(len(cells))
                    )
                else:
                    # Spatial/joint deployment uses REG's geometric feature
                    # definition and the controller-owned encoder, but never
                    # calls the frozen REG assignment policy. Teacher scores
                    # are available only in explicitly supervised training.
                    scores = None
                    teacher_index = None
                    visible_indices = range(len(cells))
                pickup_actions = self.env.plan_path_heuristic(
                    self.env.current_state, inbound.position, ignore_block=inbound
                )
                pickup_steps = len(pickup_actions) + 1
                head = executable[0] if executable else None
                for index in visible_indices:
                    cell = tuple(cells[index])
                    path = shortest_clear_path(
                        yard,
                        tuple(inbound.position),
                        cell,
                        ignore_labels=(inbound.label,),
                    )
                    if path is None:
                        continue
                    storage_steps = len(path) - 1 + 1
                    duration = pickup_steps + storage_steps
                    lookahead = None
                    if head is not None:
                        _, current_plan = head
                        # Per-cell exact retrieval replanning is quadratic in
                        # the candidate count and dominated long runs. Use the
                        # observable delay approximation as a feature for every
                        # cell; the teacher's actual mode label below still
                        # receives one exact projected plan for its chosen cell.
                        error_now = -float(current_plan.slack)
                        lookahead = (
                            error_now,
                            error_now + float(duration),
                        )
                    key = f"accept:{inbound.label}:{cell[0]}:{cell[1]}"
                    accept_durations[cell] = duration
                    accept_rows[cell] = reg_observation.candidate_features[index]
                    candidates.append(
                        FullyLearnedCandidate(
                            key=key,
                            mode="accept",
                            mode_id=MODE_TO_ID["accept"],
                            features=self._features(
                                "accept",
                                learned_state.block_features[
                                    self.env.blocks.index(inbound)
                                ],
                                cell_features=reg_observation.candidate_features[index],
                                pickup_steps=pickup_steps,
                                storage_steps=storage_steps,
                                inbound=True,
                                lookahead=lookahead,
                            ),
                            option=self.accept_option,
                            target_label=inbound.label,
                            chosen_cell=cell,
                            predicted_duration=duration,
                            teacher_score=(
                                None
                                if scores is None
                                else float(scores[index])
                            ),
                        )
                    )

        for option, plan in executable:
            candidates.append(
                FullyLearnedCandidate(
                    key=f"retrieve:{option.target_label}",
                    mode="retrieve",
                    mode_id=MODE_TO_ID["retrieve"],
                    features=self._features(
                        "retrieve",
                        learned_state.block_features[option.block_index],
                        plan=plan,
                        inbound=inbound_present,
                    ),
                    option=option,
                    target_label=option.target_label,
                    predicted_duration=int(plan.estimated_steps),
                )
            )

        allow_defer = not (
            inbound_present and not any(item.mode == "accept" for item in candidates)
        )
        if allow_defer and self.defer_option.initiation(state):
            candidates.append(
                FullyLearnedCandidate(
                    key="defer",
                    mode="defer",
                    mode_id=MODE_TO_ID["defer"],
                    features=self._features(
                        "defer", None, inbound=inbound_present
                    ),
                    option=self.defer_option,
                )
            )
        if not candidates:
            raise FullyLearnedInfeasible("no executable strict macro")

        due = [item for item in executable if item[1].slack <= 0]
        raw_due = [item for item in raw_plans if item[1].slack <= 0]
        if phase in {"spatial", "joint"} and not teacher_supervision:
            if inbound is None and raw_due and not due:
                raise FullyLearnedInfeasible(
                    "due retrieval has no executable first leg"
                )
            teacher_key = candidates[0].key
        elif due:
            teacher_key = f"retrieve:{due[0][0].target_label}"
        elif inbound is not None and teacher_cell is not None:
            teacher_accept_key = (
                f"accept:{inbound.label}:{teacher_cell[0]}:{teacher_cell[1]}"
            )
            if not executable:
                teacher_key = teacher_accept_key
            else:
                option, current_plan = executable[0]
                duration = accept_durations.get(teacher_cell)
                if duration is None:
                    # Temporal phase always includes the teacher cell; other phases
                    # enumerate all cells. Reaching this point is a contract error.
                    raise FullyLearnedInfeasible("teacher cell has no live duration")
                projected = self._projected_plan(
                    inbound, teacher_cell, duration, option.target_label, yard
                )
                cost_now = abs(-float(current_plan.slack))
                cost_after = (
                    math.inf if projected is None else abs(-float(projected.slack))
                )
                teacher_key = (
                    f"retrieve:{option.target_label}"
                    if cost_now
                    <= cost_after + self.config.lookahead_margin_steps
                    else teacher_accept_key
                )
        elif inbound is not None:
            if not executable:
                raise FullyLearnedInfeasible(
                    "inbound placement infeasible with no capacity release"
                )
            teacher_key = f"retrieve:{executable[0][0].target_label}"
        elif raw_due:
            raise FullyLearnedInfeasible("due retrieval has no executable first leg")
        elif any(item.mode == "defer" for item in candidates):
            teacher_key = "defer"
        elif executable:
            teacher_key = f"retrieve:{executable[0][0].target_label}"
        else:
            raise FullyLearnedInfeasible("teacher has no strict action")

        keys = [item.key for item in candidates]
        if len(keys) != len(set(keys)):
            raise RuntimeError("duplicate parameterized candidate identity")
        if teacher_key not in set(keys):
            raise RuntimeError(f"teacher candidate is absent: {teacher_key}")
        candidates = [
            replace(
                item,
                teacher_prior=(
                    self.config.teacher_prior_strength
                    if phase == "temporal" and item.key == teacher_key
                    else 0.0
                ),
            )
            for item in candidates
        ]
        return FullyLearnedSnapshot(
            state=learned_state,
            candidates=tuple(candidates),
            teacher_key=teacher_key,
        )

    def bind(self, candidate: FullyLearnedCandidate):
        if candidate.mode != "accept":
            return candidate.option
        inbound = self._inbound()
        if inbound is None or inbound.label != candidate.target_label:
            raise FullyLearnedInfeasible("selected inbound candidate became stale")
        preview = self.registry.preview_assignment_for_cell(
            inbound, candidate.chosen_cell, source_cell=inbound.position
        )
        estimate = self.accept_option.estimate_duration_for_preview(preview)
        if estimate is None:
            raise FullyLearnedInfeasible("selected cell has no exact duration")
        if (
            candidate.predicted_duration is not None
            and int(estimate.total_steps) != int(candidate.predicted_duration)
        ):
            raise FullyLearnedInfeasible("selected cell duration changed at bind")
        self.accept_option.bind_estimate(estimate)
        return self.accept_option


class FullyLearnedHierarchyAgent:
    """Shared-Q parameterized controller with phased teacher warm start."""

    def __init__(
        self,
        env,
        observation_encoder,
        *,
        config: Optional[FullyLearnedConfig] = None,
        spatial_network=None,
        seed: int = 0,
        device="cpu",
        gamma: float = 0.99,
        learning_rate: float = 5e-5,
        spatial_learning_rate: float = 5e-6,
        batch_size: int = 128,
        buffer_size: int = 20_000,
        update_every: int = 100,
        target_tau: float = 1e-3,
        grad_clip: float = 5.0,
        reward_scale: float = 0.01,
        epsilon: float = 0.9,
        replay_sampling: str = "mode_balanced",
    ):
        self.env = env
        self.observation_encoder = observation_encoder
        self.config = config or FullyLearnedConfig()
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        if not 0 < float(gamma) <= 1:
            raise ValueError("gamma must be in (0, 1]")
        self.gamma = float(gamma)
        self.batch_size = int(batch_size)
        self.update_every = int(update_every)
        self.target_tau = float(target_tau)
        self.grad_clip = float(grad_clip)
        self.reward_scale = float(reward_scale)
        self.epsilon = float(epsilon)
        self.replay_sampling = str(replay_sampling)
        if self.batch_size <= 0 or self.update_every <= 0:
            raise ValueError("batch size and update interval must be positive")
        if self.replay_sampling not in FULLY_LEARNED_REPLAY_SAMPLING:
            raise ValueError(
                "replay_sampling must be one of "
                f"{FULLY_LEARNED_REPLAY_SAMPLING!r}"
            )
        self.seed = int(seed)
        self.regime_id = "legacy"
        self.rng = random.Random(self.seed)
        self.builder = FullyLearnedCandidateBuilder(
            env, observation_encoder, self.config
        )
        self.Q_local = FullyLearnedQNetwork(
            self.config, spatial_network, seed=self.seed
        ).to(self.device)
        self.Q_target = deepcopy(self.Q_local).to(self.device)
        self.Q_target.requires_grad_(False)
        self.Q_target.eval()
        spatial_ids = {id(p) for p in self.Q_local.spatial_encoder.parameters()}
        main_parameters = [
            parameter
            for parameter in self.Q_local.parameters()
            if id(parameter) not in spatial_ids
        ]
        self.optimizer = torch.optim.AdamW(
            (
                {
                    "params": main_parameters,
                    "lr": float(learning_rate),
                    "group_name": "main",
                },
                {
                    "params": self.Q_local.spatial_encoder.parameters(),
                    "lr": float(spatial_learning_rate),
                    "group_name": "spatial",
                },
            )
        )
        self.base_learning_rates = {
            "main": float(learning_rate),
            "spatial": float(spatial_learning_rate),
        }
        self.learning_rate_scales = {"main": 1.0, "spatial": 1.0}
        self.replay = FullyLearnedReplayBuffer(buffer_size)
        self.training_phase = "imitation"
        self.policy_realization = "hierarchical_map"
        self.step_count = 0
        self.decision_count = 0
        self.gradient_steps = 0
        self.imitation_steps = 0
        self.retention_steps = 0
        self.spatial_distillation_steps = 0
        self.optimizer_resets = 0
        self.phase_transitions = []
        self.replay_clears = 0
        self.loss_history = []
        self.imitation_loss_history = []
        self.distillation_loss_history = []
        self.history = deque(maxlen=self.config.history_length)
        self.current_option = None
        self.last_decision_was_forced = False
        self.decision_counts = Counter()
        self.mode_counts = Counter()
        self.decisions = []
        self.selection_source_counts = Counter()
        self.current_teacher_mixture = 0.0
        self.current_bc_weight = 0.0
        self.teacher_supervision_enabled = False
        self._active_state = None
        self._active_candidate = None
        self._active_return = 0.0
        self._active_discount = 1.0
        self._active_duration = 0
        self._active_boundary_penalty = 0.0
        self._pending_imitation = None
        self.last_boundary_failure_reason = None
        self.set_training_phase("imitation", clear_replay=False)

    def bind_runtime(self, env, observation_encoder, *, regime_id: str):
        """Bind a new episode runtime without replacing learned state.

        Networks, optimizer, replay, clocks, and RNG remain owned by this
        agent.  Only environment-dependent observation/candidate plumbing and
        episode-local state are replaced.
        """

        if (
            self.current_option is not None
            or self._active_candidate is not None
            or self._active_state is not None
            or self._pending_imitation is not None
        ):
            raise RuntimeError("cannot rebind a fully learned agent mid-macro")
        regime_id = str(regime_id)
        if not regime_id:
            raise ValueError("regime_id must be nonempty")
        if getattr(observation_encoder, "env", None) is not env:
            raise ValueError(
                "observation encoder must be bound to the runtime environment"
            )
        builder = FullyLearnedCandidateBuilder(
            env, observation_encoder, self.config
        )
        self.env = env
        self.observation_encoder = observation_encoder
        self.builder = builder
        self.regime_id = regime_id
        self.reset_episode()

    @property
    def effective_teacher_coefficient(self):
        # Imitation executes teacher labels directly and must validate the
        # transferred network without label-prior assistance. The optional
        # prior is confined to the locked-cell temporal stabilization phase.
        return (
            float(self.config.teacher_coefficient)
            if self.training_phase == "temporal"
            else 0.0
        )

    @property
    def within_temperatures(self):
        return torch.as_tensor(
            self.config.within_temperatures,
            dtype=torch.float32,
            device=self.device,
        )

    def set_training_phase(self, phase: str, *, clear_replay: bool = True):
        if phase not in TRAINING_PHASES:
            raise ValueError(f"unknown training phase: {phase!r}")
        previous = getattr(self, "training_phase", None)
        if previous != phase and previous is not None:
            self.phase_transitions.append(
                {
                    "from": previous,
                    "to": phase,
                    "decision_count": self.decision_count,
                }
            )
            if clear_replay and len(self.replay):
                self.replay.clear()
                self.replay_clears += 1
        self.training_phase = phase
        self.Q_local.train()
        self.Q_local.set_spatial_trainable(phase == "joint")
        self.Q_target.requires_grad_(False)
        self.Q_target.eval()

    def set_teacher_supervision(self, enabled: bool):
        """Enable frozen-REG labels for training snapshots only."""

        self.teacher_supervision_enabled = bool(enabled)

    def set_learning_rate_scales(self, *, main: float = 1.0, spatial: float = 1.0):
        scales = {"main": float(main), "spatial": float(spatial)}
        if any(not math.isfinite(value) or value <= 0 for value in scales.values()):
            raise ValueError("learning-rate scales must be finite and positive")
        for group in self.optimizer.param_groups:
            name = group.get("group_name")
            if name not in self.base_learning_rates:
                raise RuntimeError(f"unknown optimizer parameter group: {name!r}")
            group["lr"] = self.base_learning_rates[name] * scales[name]
        self.learning_rate_scales = scales

    def reset_optimizer_state(self, *, sync_target: bool = True):
        """Reset objective-specific Adam moments at a phase boundary."""

        self.optimizer.state.clear()
        if sync_target:
            self.Q_target.load_state_dict(self.Q_local.state_dict())
        self.optimizer_resets += 1

    # Compatibility spelling used in early design notes.
    set_phase = set_training_phase

    def reset_episode(self):
        self.history.clear()
        self.current_option = None
        self.last_decision_was_forced = False
        self.decision_counts.clear()
        self.mode_counts.clear()
        self.decisions.clear()
        self.selection_source_counts.clear()
        self.current_teacher_mixture = 0.0
        self.current_bc_weight = 0.0
        self.teacher_supervision_enabled = False
        self._clear_active()
        self._pending_imitation = None
        self.last_boundary_failure_reason = None

    def _clear_active(self):
        self._active_state = None
        self._active_candidate = None
        self._active_return = 0.0
        self._active_discount = 1.0
        self._active_duration = 0
        self._active_boundary_penalty = 0.0

    def _history_features(self):
        if not self.history:
            return np.zeros(len(HISTORY_FEATURE_NAMES), dtype=np.float32)
        total = float(len(self.history))
        counts = Counter(item["mode"] for item in self.history)
        return np.asarray(
            (
                counts["accept"] / total,
                counts["retrieve"] / total,
                counts["defer"] / total,
                np.mean([item["duration"] for item in self.history])
                / max(1.0, float(self.env.MAX_T)),
                np.clip(
                    np.mean([item["return"] for item in self.history]) / 100.0,
                    -1,
                    1,
                ),
                np.mean([item["failed"] for item in self.history]),
            ),
            dtype=np.float32,
        )

    def capture_snapshot(self, state):
        return self.builder.capture(
            state,
            self._history_features(),
            self.training_phase,
            teacher_supervision=self.teacher_supervision_enabled,
        )

    @staticmethod
    def _candidate_arrays(candidates):
        return (
            np.stack([item.features for item in candidates]).astype(np.float32),
            np.asarray([item.teacher_prior for item in candidates], dtype=np.float32),
            np.asarray([item.mode_id for item in candidates], dtype=np.int64),
        )

    def _score_snapshot(
        self,
        snapshot,
        *,
        grad=False,
        teacher_coefficient=None,
    ):
        features, priors, _ = self._candidate_arrays(snapshot.candidates)
        context = torch.enable_grad() if grad else torch.no_grad()
        with context:
            q = self.Q_local(
                torch.from_numpy(snapshot.state.global_features).unsqueeze(0).to(self.device),
                torch.from_numpy(snapshot.state.block_features).unsqueeze(0).to(self.device),
                torch.from_numpy(snapshot.state.block_mask).unsqueeze(0).to(self.device),
                torch.from_numpy(features).unsqueeze(0).to(self.device),
                torch.from_numpy(priors).unsqueeze(0).to(self.device),
                (
                    self.effective_teacher_coefficient
                    if teacher_coefficient is None
                    else float(teacher_coefficient)
                ),
            )[0]
        return q

    def _hierarchical_map(self, q, candidates):
        modes = torch.as_tensor(
            [item.mode_id for item in candidates], device=q.device
        )
        values, live_modes = _mode_values_1d(
            q,
            torch.ones_like(q, dtype=torch.bool),
            modes,
            self.within_temperatures.to(dtype=q.dtype),
        )
        mode_position = max(
            range(len(values)), key=lambda i: (float(values[i]), -int(live_modes[i]))
        )
        selected_mode = int(live_modes[mode_position])
        indices = [i for i, item in enumerate(candidates) if item.mode_id == selected_mode]
        return max(indices, key=lambda i: (float(q[i]), -i))

    def _explore(self, candidates):
        grouped = {}
        for index, candidate in enumerate(candidates):
            grouped.setdefault(candidate.mode_id, []).append(index)
        mode = self.rng.choice(sorted(grouped))
        indices = grouped[mode]
        if (
            mode == MODE_TO_ID["accept"]
            and self.training_phase == "joint"
            and self.teacher_supervision_enabled
            and all(candidates[index].teacher_score is not None for index in indices)
        ):
            # Exploration must not seed fresh joint replay with arbitrary bad
            # placements. It remains stochastic, but only inside the frozen
            # teacher's highest-ranked feasible cells.
            indices = sorted(
                indices,
                key=lambda index: (
                    -float(candidates[index].teacher_score),
                    index,
                ),
            )[: self.config.joint_accept_exploration_top_k]
        return self.rng.choice(indices)

    def select_action(
        self,
        state,
        eps=0.0,
        teacher_forcing=False,
        teacher_probability=0.0,
        retain_teacher_label=False,
    ):
        if self._active_candidate is not None:
            raise RuntimeError("new decision requested before macro closure")
        teacher_probability = float(teacher_probability)
        if not 0.0 <= teacher_probability <= 1.0:
            raise ValueError("teacher_probability must be in [0, 1]")
        snapshot = self.capture_snapshot(state)
        teacher_index = next(
            index
            for index, item in enumerate(snapshot.candidates)
            if item.key == snapshot.teacher_key
        )
        imitation_forced = bool(teacher_forcing)
        teacher_mixed = bool(
            not imitation_forced
            and teacher_probability > 0.0
            and self.rng.random() < teacher_probability
        )
        forced = bool(imitation_forced or teacher_mixed)
        explored = False
        if forced:
            index = teacher_index
        elif float(eps) > 0 and self.rng.random() < float(eps):
            index = self._explore(snapshot.candidates)
            explored = True
        else:
            q = self._score_snapshot(snapshot)
            index = self._hierarchical_map(q, snapshot.candidates)
        candidate = snapshot.candidates[index]
        option = self.builder.bind(candidate)
        self._active_state = snapshot.state.copy()
        self._active_candidate = candidate
        self._active_return = 0.0
        self._active_discount = 1.0
        self._active_duration = 0
        self._active_boundary_penalty = 0.0
        self._pending_imitation = (
            (snapshot, teacher_index)
            if forced or bool(retain_teacher_label)
            else None
        )
        self.last_decision_was_forced = bool(forced or len(snapshot.candidates) == 1)
        self.decision_count += 1
        self.decision_counts[candidate.key] += 1
        self.mode_counts[candidate.mode] += 1
        if imitation_forced:
            selection_source = "imitation_teacher"
        elif teacher_mixed:
            selection_source = f"{self.training_phase}_teacher_mixture"
        elif explored:
            selection_source = "epsilon"
        elif len(snapshot.candidates) == 1:
            selection_source = "singleton"
        else:
            selection_source = "greedy"
        self.selection_source_counts[selection_source] += 1
        self.decisions.append(
            {
                "decision_index": self.decision_count - 1,
                "time_step": int(self.env.time_steps),
                "phase": self.training_phase,
                "candidate_count": len(snapshot.candidates),
                "mode_counts": dict(Counter(item.mode for item in snapshot.candidates)),
                "teacher_key": snapshot.teacher_key,
                "selected_key": candidate.key,
                "selected_mode": candidate.mode,
                "teacher_forced": forced,
                "imitation_teacher_forced": imitation_forced,
                "teacher_mixed": teacher_mixed,
                "explored": explored,
                "selection_source": selection_source,
                "chosen_cell": candidate.chosen_cell,
            }
        )
        return option

    def learn_imitation(self, *, weight: float = 1.0):
        pending = self._pending_imitation
        self._pending_imitation = None
        if pending is None:
            return None
        weight = float(weight)
        if not math.isfinite(weight) or weight < 0.0:
            raise ValueError("imitation weight must be finite and nonnegative")
        self.current_bc_weight = weight
        if weight == 0.0:
            return None
        snapshot, teacher_index = pending
        # The teacher chooses the label, but its prior logit is deliberately
        # removed from the cloning loss. Otherwise a large fixed prior would
        # make the loss nearly zero without transferring the policy into the
        # controller's trainable parameters before joint deployment.
        q = self._score_snapshot(
            snapshot, grad=True, teacher_coefficient=0.0
        )
        modes = torch.as_tensor(
            [item.mode_id for item in snapshot.candidates], device=self.device
        )
        teacher_mode = int(snapshot.candidates[teacher_index].mode_id)
        mode_values, live_modes = _mode_values_1d(
            q,
            torch.ones_like(q, dtype=torch.bool),
            modes,
            self.within_temperatures.to(dtype=q.dtype),
        )
        target_mode_position = int(
            (live_modes == teacher_mode).nonzero(as_tuple=False)[0].item()
        )
        mode_loss = F.cross_entropy(
            (mode_values / self.config.tau_mode).unsqueeze(0),
            torch.as_tensor([target_mode_position], device=self.device),
        )
        within_indices = (modes == teacher_mode).nonzero(as_tuple=False).flatten()
        target_within = int(
            (within_indices == teacher_index).nonzero(as_tuple=False)[0].item()
        )
        within_loss = F.cross_entropy(
            (q[within_indices] / self.config.within_temperatures[teacher_mode]).unsqueeze(0),
            torch.as_tensor([target_within], device=self.device),
        )
        distillation_loss = torch.zeros((), dtype=q.dtype, device=self.device)
        accept_indices = [
            index
            for index, candidate in enumerate(snapshot.candidates)
            if candidate.mode_id == MODE_TO_ID["accept"]
            and candidate.teacher_score is not None
        ]
        if len(accept_indices) > 1:
            accept_index_tensor = torch.as_tensor(
                accept_indices, dtype=torch.long, device=self.device
            )
            teacher_scores = torch.as_tensor(
                [
                    float(snapshot.candidates[index].teacher_score)
                    for index in accept_indices
                ],
                dtype=q.dtype,
                device=self.device,
            )
            # Standardization makes the listwise target insensitive to the
            # arbitrary affine scale of a frozen REG critic while preserving
            # its ordering. The hard within-mode CE above still anchors the
            # exact teacher argmax.
            teacher_scores = teacher_scores - teacher_scores.mean()
            scale = teacher_scores.std(unbiased=False).clamp_min(1.0e-6)
            teacher_logits = (
                teacher_scores / scale / self.config.teacher_score_temperature
            )
            teacher_distribution = torch.softmax(teacher_logits, dim=0).detach()
            student_log_distribution = torch.log_softmax(
                q[accept_index_tensor] / self.config.tau_accept,
                dim=0,
            )
            distillation_loss = torch.sum(
                teacher_distribution
                * (
                    torch.log(teacher_distribution.clamp_min(1.0e-12))
                    - student_log_distribution
                )
            )
            self.spatial_distillation_steps += 1
        loss = weight * (
            mode_loss
            + within_loss
            + self.config.spatial_distillation_weight * distillation_loss
        )
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.Q_local.parameters(), self.grad_clip)
        self.optimizer.step()
        self.imitation_steps += 1
        if self.training_phase in {"temporal", "joint"}:
            self.retention_steps += 1
        self._soft_update()
        value = float(loss.detach().item())
        self.imitation_loss_history.append(value)
        self.distillation_loss_history.append(
            float(distillation_loss.detach().item())
        )
        return value

    def _finish_transition(
        self,
        next_snapshot,
        *,
        done,
        env_terminal,
        truncated,
        failed,
        store_transition,
    ):
        candidate = self._active_candidate
        if candidate is None or self._active_state is None:
            return False
        if self._active_duration <= 0:
            self._clear_active()
            return False
        if next_snapshot is None:
            next_features = np.empty((0, len(CANDIDATE_FEATURE_NAMES)), np.float32)
            next_priors = np.empty((0,), np.float32)
            next_modes = np.empty((0,), np.int64)
            next_state = None
        else:
            next_features, next_priors, next_modes = self._candidate_arrays(
                next_snapshot.candidates
            )
            next_state = next_snapshot.state.copy()
        transition = FullyLearnedTransition(
            state=self._active_state.copy(),
            candidate_features=candidate.features.copy(),
            candidate_prior=float(candidate.teacher_prior),
            candidate_mode=int(candidate.mode_id),
            reward=float(self._active_return),
            duration=int(self._active_duration),
            next_state=next_state,
            next_candidate_features=next_features,
            next_candidate_priors=next_priors,
            next_candidate_modes=next_modes,
            done=bool(done),
            terminal=bool(env_terminal),
            truncated=bool(truncated),
            failed=bool(failed),
            boundary_penalty=float(self._active_boundary_penalty),
            regime_id=self.regime_id,
        )
        if store_transition:
            self.replay.add(transition)
        self.history.append(
            {
                "mode": candidate.mode,
                "duration": self._active_duration,
                "return": self._active_return,
                "failed": bool(failed),
            }
        )
        self._clear_active()
        return bool(store_transition)

    def process_step(
        self,
        next_state,
        reward,
        *,
        done,
        terminated,
        failed=False,
        env_terminal=None,
        truncated=False,
        store_transition=True,
    ):
        if self._active_candidate is None:
            raise RuntimeError("environment reward arrived without an active macro")
        self._active_return += self._active_discount * float(reward)
        self._active_discount *= self.gamma
        self._active_duration += 1
        # A time-limit cut is not a method failure, but an unfinished episode
        # must still expose its training-only completion penalty.  Keep the
        # flags distinct and assign the boundary penalty exactly once.
        if failed or (truncated and not bool(env_terminal)):
            self._active_boundary_penalty = (
                self._active_discount * float(self.config.failure_penalty)
            )
        if not terminated:
            return False
        terminal = bool(done or failed or truncated)
        if env_terminal is None:
            env_terminal = bool(done and not truncated and not failed)
        # History is part of the learned decision state. Append the completed
        # macro before capturing the next genuine decision epoch.
        candidate = self._active_candidate
        completed_summary = {
            "mode": candidate.mode,
            "duration": self._active_duration,
            "return": self._active_return,
            "failed": bool(failed),
        }
        self.history.append(completed_summary)
        next_snapshot = None
        try:
            if not terminal:
                try:
                    next_snapshot = self.capture_snapshot(next_state)
                except FullyLearnedInfeasible as exc:
                    # The selected macro completed, but the resulting state is
                    # not a valid scheduler decision epoch. Close exactly this
                    # positive-duration record as a strict method failure with
                    # no continuation; the next select reports the same typed
                    # infeasibility without creating a synthetic transition.
                    terminal = True
                    failed = True
                    self._active_boundary_penalty = (
                        self._active_discount
                        * float(self.config.failure_penalty)
                    )
                    self.last_boundary_failure_reason = str(exc)
        finally:
            # _finish_transition appends the definitive history entry.
            self.history.pop()
        return self._finish_transition(
            next_snapshot,
            done=terminal,
            env_terminal=bool(env_terminal),
            truncated=bool(truncated),
            failed=bool(failed),
            store_transition=bool(store_transition),
        )

    def flush_failure(
        self,
        state,
        *,
        failure_penalty=-50.0,
        store_transition=True,
    ):
        del state
        if self._active_candidate is None:
            return False
        if self._active_duration <= 0:
            self._clear_active()
            return False
        self._active_boundary_penalty = (
            self._active_discount * float(failure_penalty)
        )
        return self._finish_transition(
            None,
            done=True,
            env_terminal=False,
            truncated=False,
            failed=True,
            store_transition=bool(store_transition),
        )

    @staticmethod
    def _pad_states(states):
        batch = len(states)
        max_blocks = max(item.block_features.shape[0] for item in states)
        globals_ = np.stack([item.global_features for item in states]).astype(np.float32)
        blocks = np.zeros(
            (batch, max_blocks, len(BLOCK_FEATURE_NAMES)), dtype=np.float32
        )
        masks = np.zeros((batch, max_blocks), dtype=bool)
        for index, state in enumerate(states):
            count = state.block_features.shape[0]
            blocks[index, :count] = state.block_features
            masks[index, :count] = state.block_mask
        return globals_, blocks, masks

    def _td_batch(self, transitions):
        globals_, blocks, block_masks = self._pad_states(
            [item.state for item in transitions]
        )
        chosen = np.stack([item.candidate_features for item in transitions]).astype(
            np.float32
        )[:, None, :]
        priors = np.asarray(
            [item.candidate_prior for item in transitions], dtype=np.float32
        )[:, None]
        prediction = self.Q_local(
            torch.from_numpy(globals_).to(self.device),
            torch.from_numpy(blocks).to(self.device),
            torch.from_numpy(block_masks).to(self.device),
            torch.from_numpy(chosen).to(self.device),
            torch.from_numpy(priors).to(self.device),
            0.0,
        )[:, 0]

        nonterminal = [item for item in transitions if item.next_state is not None]
        next_values = torch.zeros(len(transitions), dtype=torch.float32, device=self.device)
        if nonterminal:
            max_blocks = max(item.next_state.block_features.shape[0] for item in nonterminal)
            max_candidates = max(item.next_candidate_features.shape[0] for item in nonterminal)
            size = len(transitions)
            global_dim = len(GLOBAL_FEATURE_NAMES) + len(HISTORY_FEATURE_NAMES)
            next_globals = np.zeros((size, global_dim), np.float32)
            next_blocks = np.zeros(
                (size, max_blocks, len(BLOCK_FEATURE_NAMES)), np.float32
            )
            next_block_masks = np.zeros((size, max_blocks), bool)
            next_features = np.zeros(
                (size, max_candidates, len(CANDIDATE_FEATURE_NAMES)), np.float32
            )
            next_priors = np.zeros((size, max_candidates), np.float32)
            next_modes = np.zeros((size, max_candidates), np.int64)
            next_masks = np.zeros((size, max_candidates), bool)
            for index, item in enumerate(transitions):
                if item.next_state is None:
                    continue
                block_count = item.next_state.block_features.shape[0]
                candidate_count = item.next_candidate_features.shape[0]
                next_globals[index] = item.next_state.global_features
                next_blocks[index, :block_count] = item.next_state.block_features
                next_block_masks[index, :block_count] = item.next_state.block_mask
                next_features[index, :candidate_count] = item.next_candidate_features
                next_priors[index, :candidate_count] = item.next_candidate_priors
                next_modes[index, :candidate_count] = item.next_candidate_modes
                next_masks[index, :candidate_count] = True
            with torch.no_grad():
                q_next = self.Q_target(
                    torch.from_numpy(next_globals).to(self.device),
                    torch.from_numpy(next_blocks).to(self.device),
                    torch.from_numpy(next_block_masks).to(self.device),
                    torch.from_numpy(next_features).to(self.device),
                    torch.from_numpy(next_priors).to(self.device),
                    0.0,
                )
                modes = torch.from_numpy(next_modes).to(self.device)
                masks = torch.from_numpy(next_masks).to(self.device)
                temperatures = self.within_temperatures
                for index, item in enumerate(transitions):
                    if item.next_state is None:
                        continue
                    next_values[index] = mode_regularized_value(
                        q_next[index],
                        masks[index],
                        modes[index],
                        temperatures,
                        self.config.tau_mode,
                    )
        rewards = torch.as_tensor(
            [item.reward + item.boundary_penalty for item in transitions],
            dtype=torch.float32,
            device=self.device,
        )
        durations = torch.as_tensor(
            [item.duration for item in transitions], dtype=torch.long, device=self.device
        )
        dones = torch.as_tensor(
            [item.done for item in transitions], dtype=torch.bool, device=self.device
        )
        target = smdp_target(
            rewards,
            durations,
            dones,
            next_values,
            self.gamma,
            self.reward_scale,
        )
        return prediction, target

    def learn(self):
        if len(self.replay) < self.batch_size:
            return None
        if self.step_count % self.update_every != 0:
            return None
        if self.replay_sampling == "mode_balanced":
            transitions = self.replay.sample_mode_balanced(
                self.batch_size, self.rng
            )
        elif self.replay_sampling == "regime_mode_balanced":
            transitions = self.replay.sample_regime_mode_balanced(
                self.batch_size, self.rng
            )
        else:
            transitions = self.replay.sample(self.batch_size, self.rng)
        prediction, target = self._td_batch(transitions)
        loss = F.smooth_l1_loss(
            prediction, target, beta=self.config.huber_delta
        )
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.Q_local.parameters(), self.grad_clip)
        self.optimizer.step()
        self.gradient_steps += 1
        self._soft_update()
        value = float(loss.detach().item())
        self.loss_history.append(value)
        return value

    def _soft_update(self):
        with torch.no_grad():
            for target, local in zip(
                self.Q_target.parameters(), self.Q_local.parameters()
            ):
                target.mul_(1.0 - self.target_tau).add_(
                    local, alpha=self.target_tau
                )

    def checkpoint_metadata(self):
        return {
            "controller_architecture": FULLY_LEARNED_CONTROLLER_ARCHITECTURE,
            "controller_action_interface": FULLY_LEARNED_ACTION_INTERFACE,
            "network_architecture": FULLY_LEARNED_NETWORK_ARCHITECTURE,
            "replay_version": FULLY_LEARNED_REPLAY_VERSION,
            "backup_version": FULLY_LEARNED_BACKUP_VERSION,
            "candidate_feature_version": FULLY_LEARNED_CANDIDATE_FEATURE_VERSION,
            "candidate_feature_names": CANDIDATE_FEATURE_NAMES,
            "candidate_feature_dim": len(CANDIDATE_FEATURE_NAMES),
            "state_global_feature_names": (
                *GLOBAL_FEATURE_NAMES,
                *HISTORY_FEATURE_NAMES,
            ),
            "state_block_feature_names": BLOCK_FEATURE_NAMES,
            "modes": MODE_NAMES,
            "within_temperatures": self.config.within_temperatures,
            "mode_temperature": self.config.tau_mode,
            "reference_distribution": "uniform_renormalized_over_live_controls_v1",
            "common_continuation": True,
            "training_phase": self.training_phase,
            "teacher_coefficient": self.effective_teacher_coefficient,
            "teacher_prior_application": "behavior_selection_only_v1",
            "td_teacher_prior": False,
            "teacher_action_mixture": (
                1.0
                if self.training_phase == "imitation"
                else (
                    float(self.current_teacher_mixture)
                    if self.training_phase == "temporal"
                    else 0.0
                )
            ),
            "training_teacher_action_mixture": (
                float(self.current_teacher_mixture)
                if self.training_phase in {"temporal", "joint"}
                else float(self.training_phase in {"imitation", "spatial"})
            ),
            "training_reg_supervision": bool(
                self.teacher_supervision_enabled
            ),
            "training_reg_policy_query_active": bool(
                self.teacher_supervision_enabled
            ),
            "temporal_bc_retention_weight": (
                float(self.current_bc_weight)
                if self.training_phase == "temporal"
                else 0.0
            ),
            "training_bc_retention_weight": float(self.current_bc_weight),
            "spatial_distillation": "standardized_reg_score_kl_plus_argmax_ce_v1",
            "teacher_score_temperature": self.config.teacher_score_temperature,
            "spatial_distillation_weight": self.config.spatial_distillation_weight,
            "joint_accept_exploration_top_k": (
                self.config.joint_accept_exploration_top_k
            ),
            "replay_sampling": self.replay_sampling,
            "accept_policy_lock": self.training_phase == "temporal",
            "spatial_trainable": self.training_phase == "joint",
            "training_spatial_encoder_requires_grad": bool(
                any(
                    parameter.requires_grad
                    for parameter in self.Q_local.spatial_encoder.parameters()
                )
            ),
            "training_spatial_frozen": bool(
                self.training_phase == "joint"
                and not any(
                    parameter.requires_grad
                    for parameter in self.Q_local.spatial_encoder.parameters()
                )
            ),
            "spatial_warm_start": "reg_v5_deployment_candidate_encoder_copy_v1",
            "frozen_reg_runtime_policy_query": self.training_phase
            not in {"spatial", "joint"},
            "frozen_reg_runtime_policy_query_semantics": "deployment_only_v2",
            "deployment_reg_policy_query": False,
            "learning_rate_scales": dict(self.learning_rate_scales),
            "assignment_source_independent_replay": False,
            "reward_scale": self.reward_scale,
            "failure_boundary_penalty": self.config.failure_penalty,
            "failure_penalty_contract": (
                "separate_gamma_k_failure_or_horizon_boundary_penalty_v2"
            ),
            "gamma": self.gamma,
        }

    def checkpoint_state(self, *, include_replay=True):
        state = {
            "Q_local": self.Q_local.state_dict(),
            "Q_target": self.Q_target.state_dict(),
            "training_phase": self.training_phase,
            "epsilon": self.epsilon,
            "step_count": self.step_count,
            "decision_count": self.decision_count,
            "gradient_steps": self.gradient_steps,
            "imitation_steps": self.imitation_steps,
            "retention_steps": self.retention_steps,
            "spatial_distillation_steps": self.spatial_distillation_steps,
            "optimizer_resets": self.optimizer_resets,
            "learning_rate_scales": dict(self.learning_rate_scales),
            "phase_transitions": list(self.phase_transitions),
            "replay_clears": self.replay_clears,
        }
        if include_replay:
            state.update(
                {
                    "optimizer": self.optimizer.state_dict(),
                    "replay": self.replay.state_dict(),
                    "rng_state": self.rng.getstate(),
                }
            )
        return state

    def load_checkpoint_state(self, state, *, resumable=False):
        self.Q_local.load_state_dict(state["Q_local"])
        self.Q_target.load_state_dict(state.get("Q_target", state["Q_local"]))
        self.set_training_phase(state["training_phase"], clear_replay=False)
        self.epsilon = float(state.get("epsilon", self.epsilon))
        self.step_count = int(state.get("step_count", 0))
        self.decision_count = int(state.get("decision_count", 0))
        self.gradient_steps = int(state.get("gradient_steps", 0))
        self.imitation_steps = int(state.get("imitation_steps", 0))
        self.retention_steps = int(state.get("retention_steps", 0))
        self.spatial_distillation_steps = int(
            state.get("spatial_distillation_steps", 0)
        )
        self.optimizer_resets = int(state.get("optimizer_resets", 0))
        scales = state.get("learning_rate_scales", {"main": 1.0, "spatial": 1.0})
        self.set_learning_rate_scales(
            main=float(scales.get("main", 1.0)),
            spatial=float(scales.get("spatial", 1.0)),
        )
        self.phase_transitions = list(state.get("phase_transitions", ()))
        self.replay_clears = int(state.get("replay_clears", 0))
        if resumable:
            if not all(key in state for key in ("optimizer", "replay", "rng_state")):
                raise ValueError("resumable checkpoint lacks optimizer/replay/RNG")
            self.optimizer.load_state_dict(state["optimizer"])
            self.replay.load_state_dict(state["replay"])
            self.rng.setstate(state["rng_state"])

    def audit(self, *, include_decisions=False):
        value = {
            **self.checkpoint_metadata(),
            "step_count": self.step_count,
            "decision_count": self.decision_count,
            "gradient_steps": self.gradient_steps,
            "imitation_steps": self.imitation_steps,
            "retention_steps": self.retention_steps,
            "spatial_distillation_steps": self.spatial_distillation_steps,
            "optimizer_resets": self.optimizer_resets,
            "learning_rate_scales": dict(self.learning_rate_scales),
            "replay_size": len(self.replay),
            "replay_mode_counts": self.replay.mode_counts(),
            "replay_regime_mode_counts": self.replay.regime_mode_counts(),
            "replay_clears": self.replay_clears,
            "phase_transitions": list(self.phase_transitions),
            "control_decisions": dict(self.decision_counts),
            "mode_decisions": dict(self.mode_counts),
            "selection_sources": dict(self.selection_source_counts),
            "last_loss": self.loss_history[-1] if self.loss_history else None,
            "last_imitation_loss": (
                self.imitation_loss_history[-1]
                if self.imitation_loss_history
                else None
            ),
            "last_distillation_loss": (
                self.distillation_loss_history[-1]
                if self.distillation_loss_history
                else None
            ),
            "last_boundary_failure_reason": self.last_boundary_failure_reason,
        }
        if include_decisions:
            value["decisions"] = list(self.decisions)
        return value


# Public aliases used by scripts written during the design audit.
FullyLearnedAgent = FullyLearnedHierarchyAgent
FullyLearnedCandidateSetBuilder = FullyLearnedCandidateBuilder


def validate_fully_learned_checkpoint_metadata(payload: dict):
    expected = {
        "controller_architecture": FULLY_LEARNED_CONTROLLER_ARCHITECTURE,
        "controller_action_interface": FULLY_LEARNED_ACTION_INTERFACE,
        "network_architecture": FULLY_LEARNED_NETWORK_ARCHITECTURE,
        "replay_version": FULLY_LEARNED_REPLAY_VERSION,
        "backup_version": FULLY_LEARNED_BACKUP_VERSION,
        "candidate_feature_version": FULLY_LEARNED_CANDIDATE_FEATURE_VERSION,
    }
    mismatches = {
        key: {"expected": value, "found": payload.get(key)}
        for key, value in expected.items()
        if payload.get(key) != value
    }
    if mismatches:
        raise ValueError(f"incompatible fully learned checkpoint: {mismatches!r}")
    return True


__all__ = [
    "CANDIDATE_FEATURE_NAMES",
    "CELL_FEATURE_SLICE",
    "FULLY_LEARNED_ACTION_INTERFACE",
    "FULLY_LEARNED_BACKUP_VERSION",
    "FULLY_LEARNED_CANDIDATE_FEATURE_VERSION",
    "FULLY_LEARNED_CHECKPOINT_SCHEMA_VERSION",
    "FULLY_LEARNED_CONTROLLER_ARCHITECTURE",
    "FULLY_LEARNED_NETWORK_ARCHITECTURE",
    "FULLY_LEARNED_POLICY_REALIZATIONS",
    "FULLY_LEARNED_REPLAY_VERSION",
    "FULLY_LEARNED_REPLAY_SAMPLING",
    "FullyLearnedAgent",
    "FullyLearnedCandidate",
    "FullyLearnedCandidateBuilder",
    "FullyLearnedConfig",
    "FullyLearnedHierarchyAgent",
    "FullyLearnedInfeasible",
    "FullyLearnedQNetwork",
    "FullyLearnedReplayBuffer",
    "FullyLearnedSnapshot",
    "FullyLearnedState",
    "FullyLearnedTransition",
    "MODE_NAMES",
    "MODE_TO_ID",
    "TRAINING_PHASES",
    "masked_logmeanexp",
    "mode_regularized_value",
    "smdp_target",
    "validate_fully_learned_checkpoint_metadata",
]
