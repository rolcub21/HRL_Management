"""Learning-augmented relational macro scheduling for Track B.

This module is intentionally independent of ``GatedAtomicInboundSchedulerAgent``.
The historical v5 controller has a fixed 42-output ABI and remains reproducible.
The controller below instead evaluates a variable feasible candidate set with one
shared action-conditional scorer.
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import asdict, dataclass
import copy
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from example.Options.AcceptStoreOption import AcceptStoreOption
from example.Options.RetrieveDeliverOption import StrictRetrieveDeliverOption
from example.Options.StrategicDeferOption import StrategicDeferOption
from example.controller_observation import (
    BLOCK_FEATURE_NAMES,
    GLOBAL_FEATURE_NAMES,
    OnlineManifestTimingObservationEncoder,
)
from options_agent import canonical_manager_options, controller_ids, option_identifier
from PSLAP.retrieval_context import retrieval_planning_context


RELATIONAL_CONTROLLER_ARCHITECTURE = "relational_residual_macro_scheduler_v1"
RELATIONAL_ACTION_INTERFACE = "relational_interleaved_atomic_scheduler_v1"
RELATIONAL_NETWORK_ARCHITECTURE = "deepset_action_conditional_residual_q_v1"
RELATIONAL_CANDIDATE_FEATURE_VERSION = 1
RELATIONAL_REPLAY_VERSION = "variable_candidate_double_dqn_smdp_v1"
RELATIONAL_BASELINE_VERSION = "eta_slack_urgency_priority_v1"
RELATIONAL_HISTORY_VERSION = "recent_macro_summary_v1"
RELATIONAL_POLICY_REALIZATIONS = (
    "residual_map",
    "kind_safe_residual_map",
    "baseline",
)

HISTORY_FEATURE_NAMES = (
    "recent_accept_fraction",
    "recent_retrieve_fraction",
    "recent_defer_fraction",
    "recent_mean_duration",
    "recent_mean_discounted_return",
    "recent_failure_fraction",
)

CANDIDATE_FEATURE_NAMES = (
    "kind_accept",
    "kind_retrieve",
    "kind_defer",
    *(f"block_{name}" for name in BLOCK_FEATURE_NAMES),
    "plan_valid",
    "plan_eta",
    "plan_signed_slack",
    "plan_due",
    "plan_relocation_fraction",
    "live_first_leg_feasible",
    "inbound_present",
    "baseline_score_scaled",
)


class RelationalSchedulerInfeasible(RuntimeError):
    """Raised when the strict relational controller has no executable macro."""


@dataclass(frozen=True)
class RelationalSchedulerConfig:
    block_embedding_dim: int = 64
    global_embedding_dim: int = 64
    candidate_embedding_dim: int = 64
    context_dim: int = 128
    residual_scale: float = 1.0
    uncertainty_penalty: float = 0.05
    override_margin: float = 1.0
    baseline_score_scale: float = 40.0
    history_length: int = 8
    risk_loss_weight: float = 0.1

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
            "residual_scale",
            "uncertainty_penalty",
            "override_margin",
            "baseline_score_scale",
            "risk_loss_weight",
        ):
            if float(getattr(self, name)) < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.baseline_score_scale == 0:
            raise ValueError("baseline_score_scale must be positive")
        if self.context_dim < 2:
            raise ValueError("context_dim must be at least 2")

    @classmethod
    def from_dict(cls, value):
        allowed = cls.__dataclass_fields__
        return cls(**{key: value[key] for key in allowed if key in value})


@dataclass(frozen=True)
class RelationalState:
    global_features: np.ndarray
    block_features: np.ndarray
    block_mask: np.ndarray

    def copy(self):
        return RelationalState(
            self.global_features.copy(),
            self.block_features.copy(),
            self.block_mask.copy(),
        )


@dataclass(frozen=True)
class RelationalCandidate:
    option: object
    option_id: str
    kind: str
    features: np.ndarray
    baseline_score: float
    target_label: str | None = None
    slack: float | None = None
    eta: int | None = None
    relocations: int | None = None


@dataclass(frozen=True)
class RelationalTransition:
    state: RelationalState
    candidate_features: np.ndarray
    candidate_baseline: float
    reward: float
    next_state: RelationalState
    next_candidate_features: np.ndarray
    next_candidate_baselines: np.ndarray
    done: bool
    duration: int


class RelationalReplayBuffer:
    def __init__(self, capacity: int):
        if capacity <= 0:
            raise ValueError("replay capacity must be positive")
        self.memory = deque(maxlen=int(capacity))

    def add(self, transition: RelationalTransition):
        self.memory.append(transition)

    def sample(self, batch_size: int):
        return random.sample(self.memory, k=batch_size)

    def __len__(self):
        return len(self.memory)


class RelationalResidualQNetwork(nn.Module):
    """Deep-Set yard context plus a shared candidate residual and risk head."""

    def __init__(self, config: RelationalSchedulerConfig, *, seed: int):
        super().__init__()
        self.config = config
        global_size = len(GLOBAL_FEATURE_NAMES) + len(HISTORY_FEATURE_NAMES)
        block_size = len(BLOCK_FEATURE_NAMES)
        candidate_size = len(CANDIDATE_FEATURE_NAMES)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(seed))
            self.block_encoder = nn.Sequential(
                nn.Linear(block_size, config.block_embedding_dim),
                nn.LayerNorm(config.block_embedding_dim),
                nn.ReLU(),
                nn.Linear(config.block_embedding_dim, config.block_embedding_dim),
                nn.ReLU(),
            )
            self.global_encoder = nn.Sequential(
                nn.Linear(global_size, config.global_embedding_dim),
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
                nn.Linear(candidate_size, config.candidate_embedding_dim),
                nn.LayerNorm(config.candidate_embedding_dim),
                nn.ReLU(),
                nn.Linear(
                    config.candidate_embedding_dim,
                    config.candidate_embedding_dim,
                ),
                nn.ReLU(),
            )
            joint_size = config.context_dim + config.candidate_embedding_dim
            self.joint = nn.Sequential(
                nn.Linear(joint_size, config.context_dim),
                nn.LayerNorm(config.context_dim),
                nn.ReLU(),
                nn.Linear(config.context_dim, config.context_dim // 2),
                nn.ReLU(),
            )
            self.residual_head = nn.Linear(config.context_dim // 2, 1)
            self.risk_head = nn.Linear(config.context_dim // 2, 1)

        # The untrained policy is exactly the deterministic baseline.  This is
        # important for a conservative learning-augmented experiment.
        nn.init.zeros_(self.residual_head.weight)
        nn.init.zeros_(self.residual_head.bias)
        nn.init.zeros_(self.risk_head.weight)
        nn.init.zeros_(self.risk_head.bias)

    def forward(
        self,
        global_features,
        block_features,
        block_mask,
        candidate_features,
        candidate_baselines,
    ):
        if global_features.dim() != 2 or block_features.dim() != 3:
            raise ValueError("relational network requires explicit batch dimensions")
        if candidate_features.dim() != 3 or candidate_baselines.dim() != 2:
            raise ValueError("candidate tensors must have shapes (B,M,D) and (B,M)")
        block_mask = block_mask.bool()
        encoded_blocks = self.block_encoder(block_features)
        mask = block_mask.unsqueeze(-1)
        count = mask.sum(dim=1).clamp(min=1)
        mean_pool = (encoded_blocks * mask).sum(dim=1) / count
        negative = torch.finfo(encoded_blocks.dtype).min
        max_pool = encoded_blocks.masked_fill(~mask, negative).max(dim=1).values
        any_block = block_mask.any(dim=1, keepdim=True)
        max_pool = torch.where(any_block, max_pool, torch.zeros_like(max_pool))
        global_context = self.global_encoder(global_features)
        context = self.state_encoder(
            torch.cat((global_context, mean_pool, max_pool), dim=-1)
        )
        candidates = self.candidate_encoder(candidate_features)
        expanded = context.unsqueeze(1).expand(-1, candidates.shape[1], -1)
        joint = self.joint(torch.cat((expanded, candidates), dim=-1))
        residual = self.residual_head(joint).squeeze(-1)
        risk = F.softplus(self.risk_head(joint).squeeze(-1))
        q_values = (
            candidate_baselines + self.config.residual_scale * residual
        )
        return q_values, risk, residual


class RelationalCandidateBuilder:
    """Construct online-safe executable macro candidates and baseline scores."""

    def __init__(self, env, observation_encoder, config):
        if not isinstance(
            observation_encoder, OnlineManifestTimingObservationEncoder
        ):
            raise ValueError(
                "relational scheduling requires online manifest observations"
            )
        self.env = env
        self.observation_encoder = observation_encoder
        self.config = config
        managers = canonical_manager_options(env.options)
        accepts = [item for item in managers if isinstance(item, AcceptStoreOption)]
        defers = [
            item for item in managers if isinstance(item, StrategicDeferOption)
        ]
        retrievals = [
            item
            for item in managers
            if isinstance(item, StrictRetrieveDeliverOption)
        ]
        if len(accepts) != 1 or len(defers) != 1:
            raise ValueError("relational interface needs one AcceptStore and Defer")
        if len(retrievals) != len(env.blocks):
            raise ValueError("relational interface needs one strict job per block")
        self.accept_option = accepts[0]
        self.defer_option = defers[0]
        self.retrieval_options = tuple(
            sorted(retrievals, key=lambda item: item.block_index)
        )
        self.manager_options = tuple(managers)
        self.relocation_selector = self.retrieval_options[0].relocation_selector

    def capture_state(self, state, history_features):
        observation = self.observation_encoder.capture(state)
        history = np.asarray(history_features, dtype=np.float32)
        if history.shape != (len(HISTORY_FEATURE_NAMES),):
            raise ValueError("history feature shape changed")
        return RelationalState(
            global_features=np.concatenate(
                (observation.global_features, history)
            ).astype(np.float32, copy=False),
            block_features=observation.block_features.astype(
                np.float32, copy=True
            ),
            block_mask=observation.block_mask.astype(bool, copy=True),
        )

    def _inbound_block(self):
        return next(
            (
                block
                for block in self.env.blocks
                if block.position == self.env.pickup_cell
                and not block.carrying
                and not block.stored
                and not block.delivered
            ),
            None,
        )

    def _baseline(self, kind, *, slack=None, inbound=False):
        # Scores use approximate macro-reward units.  A due retrieval starts
        # above AcceptStore; early retrievals remain below Defer.
        if kind == "retrieve":
            if slack is None:
                return -self.config.baseline_score_scale
            if slack <= 0:
                overdue = min(
                    1.0, -float(slack) / max(1.0, float(self.env.MAX_T))
                )
                return 30.0 + 10.0 * overdue
            early = min(
                1.0, float(slack) / max(1.0, float(self.env.MAX_T))
            )
            return -10.0 * early
        if kind == "accept":
            return 10.0 if inbound else -self.config.baseline_score_scale
        if kind == "defer":
            return 0.0
        raise ValueError(f"unknown candidate kind: {kind!r}")

    def _features(
        self,
        kind,
        block_row,
        *,
        plan=None,
        inbound=False,
        baseline_score,
    ):
        values = np.zeros(len(CANDIDATE_FEATURE_NAMES), dtype=np.float32)
        values[("accept", "retrieve", "defer").index(kind)] = 1.0
        offset = 3
        if block_row is not None:
            values[offset : offset + len(BLOCK_FEATURE_NAMES)] = block_row
        offset += len(BLOCK_FEATURE_NAMES)
        if plan is not None:
            values[offset] = 1.0
            values[offset + 1] = np.clip(
                plan.estimated_steps / max(1.0, float(self.env.MAX_T)), 0.0, 1.0
            )
            values[offset + 2] = np.clip(
                plan.slack / max(1.0, float(self.env.MAX_T)), -1.0, 1.0
            )
            values[offset + 3] = float(plan.slack <= 0)
            values[offset + 4] = len(plan.relocations) / max(1, len(self.env.blocks))
            values[offset + 5] = 1.0
        values[offset + 6] = float(inbound)
        values[offset + 7] = np.clip(
            baseline_score / self.config.baseline_score_scale, -1.0, 1.0
        )
        return values

    def build(self, state, relational_state):
        if any(block.carrying for block in self.env.blocks):
            return ()
        inbound = self._inbound_block()
        inbound_present = inbound is not None
        candidates = []
        if self.accept_option.initiation(state):
            baseline = self._baseline("accept", inbound=True)
            candidates.append(
                RelationalCandidate(
                    option=self.accept_option,
                    option_id=option_identifier(self.accept_option),
                    kind="accept",
                    features=self._features(
                        "accept",
                        relational_state.block_features[
                            self.env.blocks.index(inbound)
                        ],
                        inbound=True,
                        baseline_score=baseline,
                    ),
                    baseline_score=baseline,
                    target_label=inbound.label,
                )
            )

        context = retrieval_planning_context(
            self.env, relocation_selector=self.relocation_selector
        )
        for option in self.retrieval_options:
            plan = context.plan(option.target_label)
            if plan is None or not option.initiation(state):
                continue
            baseline = self._baseline("retrieve", slack=plan.slack)
            candidates.append(
                RelationalCandidate(
                    option=option,
                    option_id=option_identifier(option),
                    kind="retrieve",
                    features=self._features(
                        "retrieve",
                        relational_state.block_features[option.block_index],
                        plan=plan,
                        inbound=inbound_present,
                        baseline_score=baseline,
                    ),
                    baseline_score=baseline,
                    target_label=option.target_label,
                    slack=float(plan.slack),
                    eta=int(plan.estimated_steps),
                    relocations=len(plan.relocations),
                )
            )

        # If inbound inventory cannot be stored, waiting is not allowed to hide
        # a capacity failure; an executable retrieval must release capacity.
        allow_defer = not (
            inbound_present and not any(item.kind == "accept" for item in candidates)
        )
        if allow_defer and self.defer_option.initiation(state):
            baseline = self._baseline("defer")
            candidates.append(
                RelationalCandidate(
                    option=self.defer_option,
                    option_id=option_identifier(self.defer_option),
                    kind="defer",
                    features=self._features(
                        "defer",
                        None,
                        inbound=inbound_present,
                        baseline_score=baseline,
                    ),
                    baseline_score=baseline,
                )
            )
        return tuple(candidates)


class RelationalResidualSchedulerAgent:
    """Variable-candidate Double-DQN SMDP with a conservative baseline gate."""

    def __init__(
        self,
        env,
        observation_encoder,
        *,
        config=None,
        seed=0,
        device="cpu",
        gamma=0.99,
        learning_rate=5e-5,
        batch_size=128,
        buffer_size=100_000,
        update_every=100,
        target_tau=1e-3,
        grad_clip=5.0,
        reward_clip=100.0,
        epsilon=0.9,
    ):
        self.env = env
        self.observation_encoder = observation_encoder
        self.config = config or RelationalSchedulerConfig()
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        self.seed = int(seed)
        self.gamma = float(gamma)
        self.batch_size = int(batch_size)
        self.update_every = int(update_every)
        self.target_tau = float(target_tau)
        self.grad_clip = float(grad_clip)
        self.reward_clip = float(reward_clip)
        self.epsilon = float(epsilon)
        self.builder = RelationalCandidateBuilder(
            env, observation_encoder, self.config
        )
        self.manager_options = list(self.builder.manager_options)
        self.primitive_options = []
        self.Q_local = RelationalResidualQNetwork(
            self.config, seed=self.seed
        ).to(self.device)
        self.Q_target = copy.deepcopy(self.Q_local).to(self.device)
        self.Q_target.eval()
        self.optimizer = optim.Adam(self.Q_local.parameters(), lr=learning_rate)
        self.replay = RelationalReplayBuffer(buffer_size)
        self.step_count = 0
        self.policy_realization = "residual_map"
        self.history = deque(maxlen=self.config.history_length)
        self.decision_counts = Counter()
        self.control_decisions = Counter()
        self.gate_decisions = Counter()
        self.decision_audit = []
        self.current_option = None
        self.last_decision_was_forced = False
        self._active_state = None
        self._active_candidate = None
        self._active_rewards = []

    def set_policy_realization(self, value):
        if value not in RELATIONAL_POLICY_REALIZATIONS:
            choices = ", ".join(repr(item) for item in RELATIONAL_POLICY_REALIZATIONS)
            raise ValueError(f"policy must be one of: {choices}")
        self.policy_realization = value

    def reset_episode(self):
        self.history.clear()
        self.decision_counts.clear()
        self.control_decisions.clear()
        self.gate_decisions.clear()
        self.decision_audit.clear()
        self.current_option = None
        self.last_decision_was_forced = False
        self._active_state = None
        self._active_candidate = None
        self._active_rewards = []

    def _history_features(self):
        if not self.history:
            return np.zeros(len(HISTORY_FEATURE_NAMES), dtype=np.float32)
        total = float(len(self.history))
        kinds = Counter(item["kind"] for item in self.history)
        return np.asarray(
            (
                kinds["accept"] / total,
                kinds["retrieve"] / total,
                kinds["defer"] / total,
                np.mean([item["duration"] for item in self.history])
                / max(1.0, float(self.env.MAX_T)),
                np.clip(
                    np.mean([item["return"] for item in self.history])
                    / max(1.0, self.reward_clip),
                    -1.0,
                    1.0,
                ),
                np.mean([item["failed"] for item in self.history]),
            ),
            dtype=np.float32,
        )

    def capture_state(self, state):
        return self.builder.capture_state(state, self._history_features())

    @staticmethod
    def _candidate_arrays(candidates):
        return (
            np.stack([item.features for item in candidates]).astype(np.float32),
            np.asarray(
                [item.baseline_score for item in candidates], dtype=np.float32
            ),
        )

    def _score(self, relational_state, candidates, network=None):
        if not candidates:
            empty = np.empty(0, dtype=np.float32)
            return empty, empty, empty
        network = network or self.Q_local
        candidate_features, baselines = self._candidate_arrays(candidates)
        with torch.no_grad():
            q_values, risks, residuals = network(
                torch.from_numpy(relational_state.global_features)
                .unsqueeze(0)
                .to(self.device),
                torch.from_numpy(relational_state.block_features)
                .unsqueeze(0)
                .to(self.device),
                torch.from_numpy(relational_state.block_mask)
                .unsqueeze(0)
                .to(self.device),
                torch.from_numpy(candidate_features).unsqueeze(0).to(self.device),
                torch.from_numpy(baselines).unsqueeze(0).to(self.device),
            )
        q_values = q_values[0].cpu().numpy()
        risks = risks[0].cpu().numpy()
        residuals = residuals[0].cpu().numpy()
        selection = q_values - self.config.uncertainty_penalty * risks
        return selection, q_values, residuals

    @staticmethod
    def _best_index(values, candidates):
        return max(
            range(len(candidates)),
            key=lambda index: (float(values[index]), -index),
        )

    def _deployment_choice(self, selection, candidates, baseline_index):
        """Apply the selected deployment policy to learned candidate scores."""
        if self.policy_realization == "baseline":
            return baseline_index, "baseline_policy"

        learned_index = self._best_index(selection, candidates)
        if (
            self.policy_realization == "kind_safe_residual_map"
            and candidates[learned_index].kind != candidates[baseline_index].kind
        ):
            return baseline_index, "cross_kind_override_rejected"

        advantage = float(
            selection[learned_index] - selection[baseline_index]
        )
        if (
            learned_index != baseline_index
            and advantage <= self.config.override_margin
        ):
            return baseline_index, "override_rejected"
        if learned_index != baseline_index:
            return learned_index, "learned_override"
        return baseline_index, "baseline_agreement"

    @staticmethod
    def _explore(candidates):
        groups = {}
        for index, candidate in enumerate(candidates):
            groups.setdefault(candidate.kind, []).append(index)
        return random.choice(groups[random.choice(sorted(groups))])

    def select_action(self, state, eps=None):
        if self._active_candidate is not None:
            raise RuntimeError("a relational macro is already active")
        relational_state = self.capture_state(state)
        candidates = self.builder.build(state, relational_state)
        if not candidates:
            raise RelationalSchedulerInfeasible(
                "no executable relational macro is available"
            )
        baselines = np.asarray(
            [item.baseline_score for item in candidates], dtype=np.float32
        )
        baseline_index = self._best_index(baselines, candidates)
        selection, q_values, residuals = self._score(
            relational_state, candidates
        )
        epsilon = self.epsilon if eps is None else float(eps)
        self.last_decision_was_forced = len(candidates) == 1
        if self.last_decision_was_forced:
            selected_index = 0
            reason = "forced_singleton"
        elif epsilon > 0 and random.random() < epsilon:
            selected_index = self._explore(candidates)
            reason = "balanced_exploration"
        else:
            selected_index, reason = self._deployment_choice(
                selection, candidates, baseline_index
            )

        selected = candidates[selected_index]
        self._active_state = relational_state.copy()
        self._active_candidate = selected
        self._active_rewards = []
        self.decision_counts["option"] += 1
        self.control_decisions[selected.option_id] += 1
        self.gate_decisions[reason] += 1
        self.decision_audit.append(
            {
                "decision_index": len(self.decision_audit),
                "time_step": int(self.env.time_steps),
                "reason": reason,
                "selected": selected.option_id,
                "baseline": candidates[baseline_index].option_id,
                "candidate_ids": [item.option_id for item in candidates],
                "baseline_scores": baselines.tolist(),
                "q_values": q_values.tolist(),
                "residuals": residuals.tolist(),
            }
        )
        return selected.option

    def _discounted_return(self):
        return sum(
            (self.gamma**index) * reward
            for index, reward in enumerate(self._active_rewards)
        )

    def process_step(
        self,
        next_state,
        reward,
        *,
        done,
        terminated,
        failed=False,
        store_transition=True,
    ):
        if self._active_candidate is None:
            raise RuntimeError("no relational macro is active")
        self._active_rewards.append(
            float(np.clip(reward, -self.reward_clip, self.reward_clip))
        )
        if not terminated:
            return
        duration = len(self._active_rewards)
        macro_return = float(self._discounted_return())
        candidate = self._active_candidate
        self.history.append(
            {
                "kind": candidate.kind,
                "duration": duration,
                "return": macro_return,
                "failed": float(failed),
            }
        )
        next_relational = self.capture_state(next_state)
        next_candidates = () if done or failed else self.builder.build(
            next_state, next_relational
        )
        terminal = bool(done or failed or not next_candidates)
        if next_candidates:
            next_features, next_baselines = self._candidate_arrays(next_candidates)
        else:
            next_features = np.empty(
                (0, len(CANDIDATE_FEATURE_NAMES)), dtype=np.float32
            )
            next_baselines = np.empty(0, dtype=np.float32)
        if store_transition:
            self.replay.add(
                RelationalTransition(
                    state=self._active_state.copy(),
                    candidate_features=candidate.features.copy(),
                    candidate_baseline=float(candidate.baseline_score),
                    reward=macro_return,
                    next_state=next_relational.copy(),
                    next_candidate_features=next_features.copy(),
                    next_candidate_baselines=next_baselines.copy(),
                    done=terminal,
                    duration=duration,
                )
            )
        self._active_state = None
        self._active_candidate = None
        self._active_rewards = []

    def flush_truncation(self, next_state, *, store_transition=True):
        if self._active_candidate is None:
            return
        self.process_step(
            next_state,
            0.0,
            done=True,
            terminated=True,
            failed=True,
            store_transition=store_transition,
        )

    @staticmethod
    def _pad_next_candidates(transitions):
        maximum = max(1, max(len(item.next_candidate_baselines) for item in transitions))
        size = len(transitions)
        features = np.zeros(
            (size, maximum, len(CANDIDATE_FEATURE_NAMES)), dtype=np.float32
        )
        baselines = np.zeros((size, maximum), dtype=np.float32)
        mask = np.zeros((size, maximum), dtype=bool)
        for row, transition in enumerate(transitions):
            count = len(transition.next_candidate_baselines)
            if count:
                features[row, :count] = transition.next_candidate_features
                baselines[row, :count] = transition.next_candidate_baselines
                mask[row, :count] = True
        return features, baselines, mask

    def learn(self):
        if self.step_count % self.update_every != 0:
            return None
        if len(self.replay) < self.batch_size:
            return None
        transitions = self.replay.sample(self.batch_size)
        globals_ = torch.from_numpy(
            np.stack([item.state.global_features for item in transitions])
        ).to(self.device)
        blocks = torch.from_numpy(
            np.stack([item.state.block_features for item in transitions])
        ).to(self.device)
        block_masks = torch.from_numpy(
            np.stack([item.state.block_mask for item in transitions])
        ).to(self.device)
        chosen_features = torch.from_numpy(
            np.stack([item.candidate_features for item in transitions])
        ).unsqueeze(1).to(self.device)
        chosen_baselines = torch.tensor(
            [[item.candidate_baseline] for item in transitions],
            dtype=torch.float32,
            device=self.device,
        )
        rewards = torch.tensor(
            [item.reward for item in transitions],
            dtype=torch.float32,
            device=self.device,
        )
        dones = torch.tensor(
            [item.done for item in transitions],
            dtype=torch.bool,
            device=self.device,
        )
        durations = torch.tensor(
            [item.duration for item in transitions],
            dtype=torch.float32,
            device=self.device,
        )
        estimates, risks, _ = self.Q_local(
            globals_, blocks, block_masks, chosen_features, chosen_baselines
        )
        estimates = estimates[:, 0]
        risks = risks[:, 0]

        next_features, next_baselines, next_mask = self._pad_next_candidates(
            transitions
        )
        next_globals = torch.from_numpy(
            np.stack([item.next_state.global_features for item in transitions])
        ).to(self.device)
        next_blocks = torch.from_numpy(
            np.stack([item.next_state.block_features for item in transitions])
        ).to(self.device)
        next_block_masks = torch.from_numpy(
            np.stack([item.next_state.block_mask for item in transitions])
        ).to(self.device)
        next_features_t = torch.from_numpy(next_features).to(self.device)
        next_baselines_t = torch.from_numpy(next_baselines).to(self.device)
        next_mask_t = torch.from_numpy(next_mask).to(self.device)
        with torch.no_grad():
            local_q, local_risk, _ = self.Q_local(
                next_globals,
                next_blocks,
                next_block_masks,
                next_features_t,
                next_baselines_t,
            )
            local_selection = (
                local_q - self.config.uncertainty_penalty * local_risk
            ).masked_fill(~next_mask_t, -torch.inf)
            next_indices = local_selection.argmax(dim=1, keepdim=True)
            target_q, _, _ = self.Q_target(
                next_globals,
                next_blocks,
                next_block_masks,
                next_features_t,
                next_baselines_t,
            )
            continuation = target_q.gather(1, next_indices).squeeze(1)
            continuation = torch.where(
                next_mask_t.any(dim=1), continuation, torch.zeros_like(continuation)
            )
            targets = rewards + (
                self.gamma**durations
            ) * continuation * (~dones).float()
        td_error = targets - estimates
        q_loss = F.smooth_l1_loss(estimates, targets)
        risk_target = td_error.detach().abs().clamp(max=self.reward_clip)
        risk_loss = F.smooth_l1_loss(risks, risk_target)
        loss = q_loss + self.config.risk_loss_weight * risk_loss
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.Q_local.parameters(), self.grad_clip)
        self.optimizer.step()
        with torch.no_grad():
            for target, local in zip(
                self.Q_target.parameters(), self.Q_local.parameters()
            ):
                target.mul_(1.0 - self.target_tau).add_(
                    local, alpha=self.target_tau
                )
        return {
            "loss": float(loss.item()),
            "q_loss": float(q_loss.item()),
            "risk_loss": float(risk_loss.item()),
            "mean_abs_td_error": float(td_error.abs().mean().item()),
        }

    def checkpoint_metadata(self):
        return {
            "controller_architecture": RELATIONAL_CONTROLLER_ARCHITECTURE,
            "controller_action_interface": RELATIONAL_ACTION_INTERFACE,
            "relational_network_architecture": RELATIONAL_NETWORK_ARCHITECTURE,
            "relational_candidate_feature_version": (
                RELATIONAL_CANDIDATE_FEATURE_VERSION
            ),
            "relational_candidate_feature_names": CANDIDATE_FEATURE_NAMES,
            "relational_history_version": RELATIONAL_HISTORY_VERSION,
            "relational_history_feature_names": HISTORY_FEATURE_NAMES,
            "relational_replay_version": RELATIONAL_REPLAY_VERSION,
            "relational_baseline_version": RELATIONAL_BASELINE_VERSION,
            "relational_config": asdict(self.config),
            "controller_backup": "variable_candidate_double_dqn_smdp_v1",
            "decision_epoch_contract": (
                "interleaved_accept_retrieve_defer_strict_live_v1"
            ),
            "retrieve_deliver_option_version": (
                StrictRetrieveDeliverOption.VERSION
            ),
            "retrieval_executor_version": "named_atomic_retrieval_executor_v3",
            "primitive_decisions_enabled": False,
            "manager_option_ids": controller_ids(self.manager_options),
            "primitive_action_ids": (),
            "controller_initialization_seed": self.seed,
            "gamma": self.gamma,
            **self.observation_encoder.metadata(),
        }

    def audit(self, *, include_decisions=True):
        result = {
            **self.checkpoint_metadata(),
            "policy_realization": self.policy_realization,
            "decision_counts": dict(self.decision_counts),
            "control_decisions": dict(self.control_decisions),
            "gate_decisions": dict(self.gate_decisions),
        }
        if include_decisions:
            result["decisions"] = list(self.decision_audit)
        return result


def validate_relational_checkpoint_metadata(payload, env, observation_encoder):
    expected = {
        "controller_architecture": RELATIONAL_CONTROLLER_ARCHITECTURE,
        "controller_action_interface": RELATIONAL_ACTION_INTERFACE,
        "relational_network_architecture": RELATIONAL_NETWORK_ARCHITECTURE,
        "relational_candidate_feature_version": (
            RELATIONAL_CANDIDATE_FEATURE_VERSION
        ),
        "relational_candidate_feature_names": CANDIDATE_FEATURE_NAMES,
        "relational_history_version": RELATIONAL_HISTORY_VERSION,
        "relational_history_feature_names": HISTORY_FEATURE_NAMES,
        "relational_replay_version": RELATIONAL_REPLAY_VERSION,
        "relational_baseline_version": RELATIONAL_BASELINE_VERSION,
        "retrieve_deliver_option_version": StrictRetrieveDeliverOption.VERSION,
        "retrieval_executor_version": "named_atomic_retrieval_executor_v3",
        "primitive_decisions_enabled": False,
        "manager_option_ids": controller_ids(
            canonical_manager_options(env.options)
        ),
        "primitive_action_ids": (),
        **observation_encoder.metadata(),
    }
    mismatches = {}
    for key, value in expected.items():
        saved = payload.get(key)
        matches = (
            tuple(saved or ()) == value
            if isinstance(value, tuple)
            else saved == value
        )
        if not matches:
            mismatches[key] = {"saved": saved, "expected": value}
    if mismatches:
        raise ValueError(f"relational checkpoint contract mismatch: {mismatches!r}")


__all__ = [
    "CANDIDATE_FEATURE_NAMES",
    "HISTORY_FEATURE_NAMES",
    "RELATIONAL_ACTION_INTERFACE",
    "RELATIONAL_BASELINE_VERSION",
    "RELATIONAL_CANDIDATE_FEATURE_VERSION",
    "RELATIONAL_CONTROLLER_ARCHITECTURE",
    "RELATIONAL_NETWORK_ARCHITECTURE",
    "RELATIONAL_POLICY_REALIZATIONS",
    "RelationalCandidate",
    "RelationalCandidateBuilder",
    "RelationalReplayBuffer",
    "RelationalResidualQNetwork",
    "RelationalResidualSchedulerAgent",
    "RelationalSchedulerConfig",
    "RelationalSchedulerInfeasible",
    "RelationalState",
    "RelationalTransition",
    "validate_relational_checkpoint_metadata",
]
