"""Production temporal-mode controller with a common regularized backup."""

from __future__ import annotations

from collections import Counter, namedtuple
import math
import random

import numpy as np
import torch
import torch.nn.functional as F

from example.Options.waitOption import WaitOption
from example.small_rooms_env import SmallRoomsEnv
from example.helper.tools import flat
from options_agent import (
    DQNAgent,
    DUELING_UNCENTERED,
    LAYER_NORM,
    ManagerReplayBuffer,
    WorkerReplayBuffer,
    option_identifier,
)


FULL_REGULARIZED_CONTROLLER_ARCHITECTURE = (
    "cardinality_invariant_temporal_mode_v1"
)
MODE_ONLY_CONTROLLER_ARCHITECTURE = (
    "cardinality_invariant_temporal_mode_max_within_v2"
)
FOUNDATION_CONTROLLER_ARCHITECTURE = (
    "cardinality_invariant_temporal_mode_foundation_v3"
)
SCHEDULER_CONTROLLER_ARCHITECTURE = (
    "cardinality_invariant_temporal_mode_scheduler_v4"
)
ATOMIC_INBOUND_SCHEDULER_ARCHITECTURE = "atomic_inbound_scheduler_v5"
LEGACY_SPLIT_CONTROLLER_ACTION_INTERFACE = "legacy_split_retrieval_v1"
SCHEDULER_CONTROLLER_ACTION_INTERFACE = "fixed_block_scheduler_v1"
ATOMIC_INBOUND_CONTROLLER_ACTION_INTERFACE = "atomic_inbound_scheduler_v2"
# Backward-compatible name for the original production implementation.
CONTROLLER_ARCHITECTURE = FULL_REGULARIZED_CONTROLLER_ARCHITECTURE
SUPPORTED_CONTROLLER_ARCHITECTURES = (
    FULL_REGULARIZED_CONTROLLER_ARCHITECTURE,
    MODE_ONLY_CONTROLLER_ARCHITECTURE,
    FOUNDATION_CONTROLLER_ARCHITECTURE,
    SCHEDULER_CONTROLLER_ARCHITECTURE,
    ATOMIC_INBOUND_SCHEDULER_ARCHITECTURE,
)


def normalized_logsumexp(values, mask, temperature, reference=None):
    """KL-regularized value under a reference conditioned on valid controls."""

    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if values.shape != mask.shape:
        raise ValueError("values and mask must have identical shapes")
    mask = mask.bool()
    if reference is None:
        log_mass = torch.zeros_like(values)
    else:
        reference = torch.as_tensor(
            reference, dtype=values.dtype, device=values.device
        )
        if reference.ndim == 1:
            reference = reference.unsqueeze(0).expand_as(values)
        else:
            reference = torch.broadcast_to(reference, values.shape)
        if bool((reference < 0).any()):
            raise ValueError("reference masses must be non-negative")
        positive = reference > 0
        mask = mask & positive
        log_mass = torch.where(
            positive, reference.log(), torch.full_like(reference, -torch.inf)
        )
    valid = mask.any(dim=1, keepdim=True)
    masked_log_mass = log_mass.masked_fill(~mask, -torch.inf)
    normalizer = torch.logsumexp(masked_log_mass, dim=1, keepdim=True)
    terms = (values / temperature + log_mass).masked_fill(~mask, -torch.inf)
    result = temperature * (
        torch.logsumexp(terms, dim=1, keepdim=True) - normalizer
    )
    return torch.where(valid, result, torch.zeros_like(result))


def temporal_mode_value(
    option_values,
    primitive_values,
    option_mask,
    primitive_mask,
    tau_option,
    tau_primitive,
    tau_mode,
):
    """Aggregate controls within modes, then aggregate the two modes."""

    option_mode = normalized_logsumexp(option_values, option_mask, tau_option)
    primitive_mode = normalized_logsumexp(
        primitive_values, primitive_mask, tau_primitive
    )
    mode_values = torch.cat((primitive_mode, option_mode), dim=1)
    mode_mask = torch.cat(
        (
            primitive_mask.any(dim=1, keepdim=True),
            option_mask.any(dim=1, keepdim=True),
        ),
        dim=1,
    )
    value = normalized_logsumexp(mode_values, mode_mask, tau_mode)
    return value, primitive_mode, option_mode, mode_mask


def masked_max(values, mask):
    """Maximum over valid controls, returning zero for an unavailable mode."""

    if values.shape != mask.shape:
        raise ValueError("values and mask must have identical shapes")
    mask = mask.bool()
    valid = mask.any(dim=1, keepdim=True)
    result = values.masked_fill(~mask, -torch.inf).max(
        dim=1, keepdim=True
    ).values
    return torch.where(valid, result, torch.zeros_like(result))


def mode_only_temporal_value(
    option_values,
    primitive_values,
    option_mask,
    primitive_mask,
    tau_mode,
):
    """Hard within-mode control and KL-regularized temporal-mode value."""

    option_mode = masked_max(option_values, option_mask)
    primitive_mode = masked_max(primitive_values, primitive_mask)
    mode_values = torch.cat((primitive_mode, option_mode), dim=1)
    mode_mask = torch.cat(
        (
            primitive_mask.any(dim=1, keepdim=True),
            option_mask.any(dim=1, keepdim=True),
        ),
        dim=1,
    )
    value = normalized_logsumexp(mode_values, mode_mask, tau_mode)
    return value, primitive_mode, option_mode, mode_mask


class GatedManagerReplayBuffer(ManagerReplayBuffer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.experience = namedtuple(
            "GatedManagerExperience",
            (
                "state",
                "option_idx",
                "reward",
                "next_state",
                "done",
                "k",
                "next_option_mask",
                "next_primitive_mask",
            ),
        )

    def add(
        self,
        state,
        option_idx,
        reward,
        next_state,
        done,
        k,
        next_option_mask,
        next_primitive_mask,
    ):
        self.memory.append(
            self.experience(
                state,
                option_idx,
                reward,
                next_state,
                done,
                k,
                next_option_mask,
                next_primitive_mask,
            )
        )
        self.priorities.append(float(self.max_priority))

    def sample(self):
        if len(self.memory) < self.batch_size:
            return None
        priorities = np.asarray(self.priorities, dtype=float)
        probabilities = priorities**self.alpha
        probabilities /= probabilities.sum()
        indices = np.random.choice(
            len(self.memory), self.batch_size, p=probabilities, replace=True
        )
        experiences = [self.memory[index] for index in indices]
        self.beta = min(1.0, self.beta + self.beta_increment_per_sampling)
        weights = (len(self.memory) * probabilities[indices]) ** -self.beta
        weights /= weights.max()
        return (
            torch.from_numpy(np.vstack([e.state for e in experiences]))
            .float()
            .to(self.device),
            torch.tensor(
                [[e.option_idx] for e in experiences],
                dtype=torch.long,
                device=self.device,
            ),
            torch.from_numpy(np.vstack([e.reward for e in experiences]))
            .float()
            .to(self.device),
            torch.from_numpy(np.vstack([e.next_state for e in experiences]))
            .float()
            .to(self.device),
            torch.from_numpy(
                np.vstack([e.done for e in experiences]).astype(np.uint8)
            )
            .float()
            .to(self.device),
            torch.tensor(
                [[e.k] for e in experiences], dtype=torch.float, device=self.device
            ),
            torch.from_numpy(weights).float().to(self.device).unsqueeze(1),
            indices,
            torch.from_numpy(
                np.vstack([e.next_option_mask for e in experiences])
            )
            .bool()
            .to(self.device),
            torch.from_numpy(
                np.vstack([e.next_primitive_mask for e in experiences])
            )
            .bool()
            .to(self.device),
        )


class GatedWorkerReplayBuffer(WorkerReplayBuffer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.experience = namedtuple(
            "GatedWorkerExperience",
            (
                "state",
                "action",
                "reward",
                "next_state",
                "done",
                "next_option_mask",
                "next_primitive_mask",
            ),
        )

    def add(
        self,
        state,
        action,
        reward,
        next_state,
        done,
        next_option_mask,
        next_primitive_mask,
    ):
        self.memory.append(
            self.experience(
                state,
                action,
                reward,
                next_state,
                done,
                next_option_mask,
                next_primitive_mask,
            )
        )

    def sample(self):
        if len(self.memory) < self.batch_size:
            return None
        experiences = random.sample(self.memory, k=self.batch_size)
        return (
            torch.from_numpy(np.vstack([e.state for e in experiences]))
            .float()
            .to(self.device),
            torch.from_numpy(np.vstack([e.action for e in experiences]))
            .long()
            .to(self.device),
            torch.from_numpy(np.vstack([e.reward for e in experiences]))
            .float()
            .to(self.device),
            torch.from_numpy(np.vstack([e.next_state for e in experiences]))
            .float()
            .to(self.device),
            torch.from_numpy(
                np.vstack([e.done for e in experiences]).astype(np.uint8)
            )
            .float()
            .to(self.device),
            torch.from_numpy(
                np.vstack([e.next_option_mask for e in experiences])
            )
            .bool()
            .to(self.device),
            torch.from_numpy(
                np.vstack([e.next_primitive_mask for e in experiences])
            )
            .bool()
            .to(self.device),
        )


class GatedRegularizedAgent(DQNAgent):
    """Two-stage controller with one regularized continuation for both heads."""

    controller_architecture = FULL_REGULARIZED_CONTROLLER_ARCHITECTURE
    controller_backup = "common_temporal_mode_regularized"

    def __init__(self, *args, **kwargs):
        self.tau_option = float(kwargs.pop("tau_option", 0.1))
        self.tau_primitive = float(kwargs.pop("tau_primitive", 0.1))
        self.tau_mode = float(kwargs.pop("tau_mode", 1.0))
        policy = kwargs.pop("training_policy", "regularized")
        super().__init__(*args, **kwargs)
        buffer_size = kwargs.get("buffer_size", 1_000_000)
        self.ManagerBuffer = GatedManagerReplayBuffer(
            buffer_size // 10,
            self.batch_size,
            device=self.device,
            verbose=self.verbose,
        )
        self.WorkerBuffer = GatedWorkerReplayBuffer(
            buffer_size, self.batch_size, device=self.device
        )
        self.target_audit = {
            "manager_updates": Counter(),
            "primitive_updates": Counter(),
        }
        self.decision_counts = Counter()
        self.control_decisions = Counter()
        self.gate_decisions = Counter()
        self.set_policy_realization(policy)

    def checkpoint_metadata(self):
        return {
            **super().checkpoint_metadata(),
            "controller_architecture": self.controller_architecture,
            "controller_backup": self.controller_backup,
            "within_mode_aggregation": "normalized_logsumexp",
            "mode_aggregation": "kl_regularized",
            "tau_option": self.tau_option,
            "tau_primitive": self.tau_primitive,
            "tau_mode": self.tau_mode,
            "training_policy": self.policy_realization,
        }

    def _temporal_value(
        self,
        option_values,
        primitive_values,
        option_mask,
        primitive_mask,
    ):
        return temporal_mode_value(
            option_values,
            primitive_values,
            option_mask,
            primitive_mask,
            self.tau_option,
            self.tau_primitive,
            self.tau_mode,
        )

    def set_policy_realization(self, policy):
        if policy not in ("map", "mode_regularized", "regularized"):
            raise ValueError(f"Unknown gated policy realization: {policy}")
        self.policy_realization = policy
        self.sample_regularized_mode = policy in (
            "mode_regularized",
            "regularized",
        )
        self.sample_regularized_controls = policy == "regularized"

    def _availability_masks(self, next_state, done):
        if done:
            return (
                np.zeros(len(self.manager_options), dtype=np.bool_),
                np.zeros(len(self.primitive_options), dtype=np.bool_),
            )
        return (
            np.asarray(
                [option.initiation(next_state) for option in self.manager_options],
                dtype=np.bool_,
            ),
            np.asarray(
                [option.initiation(next_state) for option in self.primitive_options],
                dtype=np.bool_,
            ),
        )

    def _next_value(self, next_states, option_masks, primitive_masks, audit_key):
        with torch.no_grad():
            option_values = self.Q_manager_target(next_states)
            primitive_values = self.Q_worker_target(next_states)
            value, primitive_mode, option_mode, mode_mask = self._temporal_value(
                option_values,
                primitive_values,
                option_masks,
                primitive_masks,
            )
        audit = self.target_audit[audit_key]
        active = mode_mask.any(dim=1)
        both = mode_mask.all(dim=1)
        audit["samples"] += int(active.sum().item())
        audit["both_modes_available"] += int(both.sum().item())
        audit["option_gate_wins"] += int(
            ((option_mode > primitive_mode).squeeze(1) & both).sum().item()
        )
        audit["primitive_gate_wins"] += int(
            ((primitive_mode >= option_mode).squeeze(1) & both).sum().item()
        )
        return value

    def select_action(self, state, eps, state_features=None):
        encoded = (
            self.encode_state(state)
            if state_features is None
            else np.asarray(state_features, dtype=np.float32)
        )
        state_tensor = torch.from_numpy(encoded).float().to(self.device)
        self.Q_manager_local.eval()
        self.Q_worker_local.eval()
        with torch.no_grad():
            option_values = self.Q_manager_local(state_tensor).unsqueeze(0)
            primitive_values = self.Q_worker_local(state_tensor).unsqueeze(0)
        self.Q_manager_local.train()
        self.Q_worker_local.train()

        option_valid, primitive_valid = self._availability_masks(state, False)
        option_valid = option_valid.tolist()
        primitive_valid = primitive_valid.tolist()
        option_mask = torch.tensor(
            [option_valid], dtype=torch.bool, device=self.device
        )
        primitive_mask = torch.tensor(
            [primitive_valid], dtype=torch.bool, device=self.device
        )
        option_indices = [i for i, valid in enumerate(option_valid) if valid]
        primitive_indices = [i for i, valid in enumerate(primitive_valid) if valid]
        if not option_indices and not primitive_indices:
            raise RuntimeError("No controller action is available")

        if random.random() < eps:
            if option_indices and primitive_indices:
                choose_option = random.random() < 0.5
            else:
                choose_option = bool(option_indices)
            if choose_option:
                choice = self.manager_options[random.choice(option_indices)]
            else:
                choice = self.primitive_options[random.choice(primitive_indices)]
        else:
            _, primitive_mode, option_mode, _ = self._temporal_value(
                option_values,
                primitive_values,
                option_mask,
                primitive_mask,
            )
            if option_indices and primitive_indices:
                option_probability = float(
                    torch.sigmoid(
                        (option_mode - primitive_mode) / self.tau_mode
                    ).item()
                )
                self.gate_decisions["scored_states"] += 1
                self.gate_decisions["u_option_sum"] += float(option_mode.item())
                self.gate_decisions["u_primitive_sum"] += float(
                    primitive_mode.item()
                )
                self.gate_decisions["option_probability_sum"] += option_probability
            elif option_indices:
                option_probability = 1.0
            else:
                option_probability = 0.0
            if option_indices and not primitive_indices:
                choose_option = True
            elif primitive_indices and not option_indices:
                choose_option = False
            else:
                choose_option = (
                    random.random() < option_probability
                    if self.sample_regularized_mode
                    else bool((option_mode > primitive_mode).item())
                )
            if choose_option:
                if self.sample_regularized_controls:
                    logits = (option_values[0] / self.tau_option).masked_fill(
                        ~option_mask[0], -torch.inf
                    )
                    index = int(
                        torch.multinomial(torch.softmax(logits, dim=0), 1).item()
                    )
                else:
                    index = max(
                        option_indices,
                        key=lambda i: (float(option_values[0, i].item()), -i),
                    )
                choice = self.manager_options[index]
            else:
                if self.sample_regularized_controls:
                    logits = (primitive_values[0] / self.tau_primitive).masked_fill(
                        ~primitive_mask[0], -torch.inf
                    )
                    index = int(
                        torch.multinomial(torch.softmax(logits, dim=0), 1).item()
                    )
                else:
                    index = max(
                        primitive_indices,
                        key=lambda i: (float(primitive_values[0, i].item()), -i),
                    )
                choice = self.primitive_options[index]

        mode = "primitive" if choice.is_primitive else "option"
        self.decision_counts[mode] += 1
        self.gate_decisions[mode] += 1
        self.control_decisions[option_identifier(choice)] += 1
        return choice

    def process_step(
        self,
        state,
        action,
        reward,
        next_state,
        done,
        term,
        *,
        state_features=None,
        next_state_features=None,
    ):
        reward = np.clip(reward, -self.reward_clip, self.reward_clip)
        primitive_wait = (
            self.current_option
            and self.current_option.is_primitive
            and getattr(self.current_option, "action", None)
            == SmallRoomsEnv.ACTION_IDS["WAIT"]
        )
        if isinstance(self.current_option, WaitOption) or primitive_wait:
            reward -= self.wait_training_penalty
        state_flat = (
            self.encode_state(state)
            if state_features is None
            else np.asarray(state_features, dtype=np.float32)
        )
        next_state_flat = (
            self.encode_state(next_state)
            if next_state_features is None
            else np.asarray(next_state_features, dtype=np.float32)
        )
        if self.current_option and self.current_option.is_primitive:
            option_mask, primitive_mask = self._availability_masks(next_state, done)
            self.WorkerBuffer.add(
                state_flat,
                action,
                reward,
                next_state_flat,
                done,
                option_mask,
                primitive_mask,
            )
        if self.current_option and not self.current_option.is_primitive:
            self.option_reward_traj.append(reward)
            if term:
                option_mask, primitive_mask = self._availability_masks(next_state, done)
                option_index = self.manager_options.index(self.current_option)
                duration = len(self.option_reward_traj)
                cumulative = np.clip(
                    self._discounted_return(self.option_reward_traj, self.gamma),
                    -self.reward_clip,
                    self.reward_clip,
                )
                self.ManagerBuffer.add(
                    (
                        self.option_start_features
                        if getattr(self, "option_start_features", None) is not None
                        else self.encode_state(self.option_start_state)
                    ),
                    option_index,
                    cumulative,
                    next_state_flat,
                    done,
                    duration,
                    option_mask,
                    primitive_mask,
                )
                self.current_option = None

    def _learn_worker(self):
        sample = self.WorkerBuffer.sample()
        if sample is None:
            return None
        states, actions, rewards, next_states, dones, option_masks, primitive_masks = sample
        continuation = self._next_value(
            next_states, option_masks, primitive_masks, "primitive_updates"
        )
        target = rewards + self.gamma * continuation * (1 - dones)
        estimate = self.Q_worker_local(states).gather(1, actions)
        loss = F.mse_loss(estimate, target)
        self.optimizer_worker.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.Q_worker_local.parameters(), self.grad_clip
        )
        self.optimizer_worker.step()
        self._soft_update(self.Q_worker_local, self.Q_worker_target, self.tau)
        return loss.item()

    def _learn_manager(self):
        sample = self.ManagerBuffer.sample()
        if sample is None:
            return None
        (
            states,
            options,
            rewards,
            next_states,
            dones,
            durations,
            weights,
            indices,
            option_masks,
            primitive_masks,
        ) = sample
        continuation = self._next_value(
            next_states, option_masks, primitive_masks, "manager_updates"
        )
        target = rewards + (self.gamma**durations) * continuation * (1 - dones)
        estimate = self.Q_manager_local(states).gather(1, options)
        td_error = target - estimate
        loss = (weights * F.mse_loss(estimate, target, reduction="none")).mean()
        self.optimizer_manager.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.Q_manager_local.parameters(), self.grad_clip
        )
        self.optimizer_manager.step()
        self.ManagerBuffer.update_priorities(indices, td_error)
        self._soft_update(self.Q_manager_local, self.Q_manager_target, self.tau)
        return loss.item()


class GatedModeOnlyAgent(GatedRegularizedAgent):
    """Greedy within each mode with a KL-regularized temporal-mode gate."""

    controller_architecture = MODE_ONLY_CONTROLLER_ARCHITECTURE
    controller_backup = "common_temporal_mode_only_regularized"

    def checkpoint_metadata(self):
        return {
            **super().checkpoint_metadata(),
            "controller_architecture": self.controller_architecture,
            "controller_backup": self.controller_backup,
            "within_mode_aggregation": "masked_max",
            "mode_aggregation": "kl_regularized",
            "tau_mode": self.tau_mode,
            "training_policy": self.policy_realization,
        }

    def set_policy_realization(self, policy):
        if policy not in ("map", "mode_regularized"):
            raise ValueError(
                "Mode-only controller policy must be map or mode_regularized"
            )
        self.policy_realization = policy
        self.sample_regularized_mode = policy == "mode_regularized"
        self.sample_regularized_controls = False

    def _temporal_value(
        self,
        option_values,
        primitive_values,
        option_mask,
        primitive_mask,
    ):
        return mode_only_temporal_value(
            option_values,
            primitive_values,
            option_mask,
            primitive_mask,
            self.tau_mode,
        )


class GatedModeOnlyFoundationAgent(GatedModeOnlyAgent):
    """Corrected v3 foundation prior to the parameterized scheduler.

    The Bellman operator is unchanged from v2.  This version isolates critic
    normalization, removes representation-dependent WAIT shaping, and closes
    episode-end replay records so subsequent scheduler work has a sound base.
    """

    controller_architecture = FOUNDATION_CONTROLLER_ARCHITECTURE

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("network_normalization", LAYER_NORM)
        kwargs.setdefault("wait_training_penalty", 0.0)
        kwargs.setdefault("terminal_on_truncation", True)
        kwargs.setdefault("close_options_on_episode_end", True)
        super().__init__(*args, **kwargs)


class GatedSchedulingAgent(GatedModeOnlyFoundationAgent):
    """Online temporal scheduler with atomic named retrieval jobs.

    This transitional fixed-output controller keeps the v3 Bellman operator
    and corrected replay semantics, but replaces the hand-written ripe pickup
    rule with one learnable retrieval decision per manifest block and a single
    event-driven defer control.  It intentionally receives a new architecture
    and action-interface version because prior manager weights are not shape-
    or meaning-compatible.
    """

    controller_architecture = SCHEDULER_CONTROLLER_ARCHITECTURE

    def __init__(self, *args, **kwargs):
        self.max_defer_steps = int(kwargs.pop("max_defer_steps", 10))
        if self.max_defer_steps <= 0:
            raise ValueError("max_defer_steps must be positive")
        kwargs.setdefault("dueling_centering", DUELING_UNCENTERED)
        super().__init__(*args, **kwargs)
        if self.controller_observation_metadata.get(
            "controller_observation_version"
        ) != "online_manifest_timing_v3":
            raise ValueError(
                "Scheduling controller requires online_manifest_timing_v3"
            )
        if self.controller_observation_metadata.get(
            "controller_observation_block_row_contract"
        ) != "episode_manifest_slot_v1":
            raise ValueError(
                "Scheduling controller observation rows must use manifest slots"
            )

    def checkpoint_metadata(self):
        return {
            **super().checkpoint_metadata(),
            "controller_action_interface": (
                SCHEDULER_CONTROLLER_ACTION_INTERFACE
            ),
            "block_identity_contract": "episode_manifest_slot_v1",
            "block_count": len(self.env.blocks),
            "block_slot_ids": tuple(block.label for block in self.env.blocks),
            "retrieve_deliver_option_version": "retrieve_deliver_option_v1",
            "retrieval_executor_version": (
                "named_atomic_retrieval_executor_v1"
            ),
            "retrieval_termination_contract": (
                "complete_delivery_or_safe_explicit_failure_v1"
            ),
            "defer_option_version": "strategic_defer_until_event_v1",
            "relocation_selector_version": (
                "nearest_feasible_path_then_cell_v1"
            ),
            "max_defer_steps": self.max_defer_steps,
        }


class GatedAtomicInboundSchedulerAgent(GatedSchedulingAgent):
    """Option-only scheduler with a mandatory atomic inbound workflow."""

    controller_architecture = ATOMIC_INBOUND_SCHEDULER_ARCHITECTURE

    def _availability_masks(self, state, done):
        if done:
            return (
                np.zeros(len(self.manager_options), dtype=np.bool_),
                np.zeros(len(self.primitive_options), dtype=np.bool_),
            )
        option_mask = np.asarray(
            [option.initiation(state) for option in self.manager_options],
            dtype=np.bool_,
        )
        accept_indices = [
            index
            for index, option in enumerate(self.manager_options)
            if getattr(option, "controller_identifier", None)
            == "option:AcceptStore:v1"
            and option_mask[index]
        ]
        inbound_waiting = any(
            block.position == self.env.pickup_cell
            and not block.carrying
            and not block.stored
            and not block.delivered
            for block in self.env.blocks
        )
        if accept_indices:
            option_mask[:] = False
            option_mask[accept_indices] = True
        elif inbound_waiting:
            # A waiting inbound block is a mandatory decision epoch.  If the
            # atomic workflow is temporarily infeasible, retrieval may create
            # capacity; Defer is deliberately unavailable so failure cannot be
            # hidden by an endless WAIT policy.
            for index, option in enumerate(self.manager_options):
                identifier = option_identifier(option)
                option_mask[index] = bool(
                    option_mask[index]
                    and identifier.startswith("option:RetrieveDeliver:")
                )
        return (
            option_mask,
            np.zeros(len(self.primitive_options), dtype=np.bool_),
        )

    def select_action(self, state, eps, state_features=None):
        option_mask, primitive_mask = self._availability_masks(state, False)
        if bool(primitive_mask.any()):
            raise RuntimeError(
                "Atomic inbound scheduler exposed a primitive decision"
            )
        valid_indices = np.flatnonzero(option_mask).tolist()
        if not valid_indices:
            raise RuntimeError(
                "No feasible scheduler macro is available; the episode is "
                "method-infeasible under the strict protocol"
            )

        self.last_decision_was_forced = len(valid_indices) == 1
        if self.last_decision_was_forced:
            choice_index = valid_indices[0]
            self.gate_decisions["forced_singleton"] += 1
        elif random.random() < eps:
            retrieval_indices = [
                index
                for index in valid_indices
                if option_identifier(
                    self.manager_options[index]
                ).startswith("option:RetrieveDeliver:")
            ]
            defer_indices = [
                index
                for index in valid_indices
                if option_identifier(self.manager_options[index])
                == "option:DeferUntilEvent:v1"
            ]
            if retrieval_indices and defer_indices:
                # Explore temporal intent, not raw output cardinality: the
                # dispatch group and Defer each receive half of the mass.
                group = (
                    retrieval_indices
                    if random.random() < 0.5
                    else defer_indices
                )
            else:
                group = retrieval_indices or defer_indices or valid_indices
            choice_index = random.choice(group)
            self.gate_decisions["exploratory_scheduler"] += 1
        else:
            encoded = (
                self.encode_state(state)
                if state_features is None
                else np.asarray(state_features, dtype=np.float32)
            )
            state_tensor = torch.from_numpy(encoded).float().to(self.device)
            self.Q_manager_local.eval()
            with torch.no_grad():
                option_values = self.Q_manager_local(state_tensor)
            self.Q_manager_local.train()
            choice_index = max(
                valid_indices,
                key=lambda index: (
                    float(option_values[index].item()),
                    -index,
                ),
            )
            self.gate_decisions["greedy_scheduler"] += 1

        choice = self.manager_options[choice_index]
        self.decision_counts["option"] += 1
        self.gate_decisions["option"] += 1
        self.control_decisions[option_identifier(choice)] += 1
        return choice

    def checkpoint_metadata(self):
        return {
            **super().checkpoint_metadata(),
            "controller_architecture": self.controller_architecture,
            "controller_backup": "masked_macro_smdp_optimality_v1",
            "within_mode_aggregation": "masked_max",
            "mode_aggregation": "single_macro_mode_identity_v1",
            "controller_action_interface": (
                ATOMIC_INBOUND_CONTROLLER_ACTION_INTERFACE
            ),
            "decision_epoch_contract": "mandatory_inbound_then_schedule_v1",
            "scheduler_decision_state_contract": (
                "forced_inbound_else_retrieve_or_defer_v1"
            ),
            "active_decision_modes": ("macro_option",),
            "primitive_decision_mode": "masked_at_scheduler_epochs_v1",
            "primitive_decisions_enabled": False,
            "exploration_reference": (
                "half_dispatch_group_half_defer_then_uniform_block_v1"
            ),
            "forced_decision_epsilon_clock": "does_not_advance_v1",
            "accept_store_option_version": "accept_store_option_v1",
            "selector_assignment_duration": "zero_environment_steps_v1",
        }


__all__ = [
    "CONTROLLER_ARCHITECTURE",
    "FULL_REGULARIZED_CONTROLLER_ARCHITECTURE",
    "MODE_ONLY_CONTROLLER_ARCHITECTURE",
    "FOUNDATION_CONTROLLER_ARCHITECTURE",
    "SCHEDULER_CONTROLLER_ARCHITECTURE",
    "ATOMIC_INBOUND_SCHEDULER_ARCHITECTURE",
    "LEGACY_SPLIT_CONTROLLER_ACTION_INTERFACE",
    "SCHEDULER_CONTROLLER_ACTION_INTERFACE",
    "ATOMIC_INBOUND_CONTROLLER_ACTION_INTERFACE",
    "SUPPORTED_CONTROLLER_ARCHITECTURES",
    "GatedModeOnlyAgent",
    "GatedModeOnlyFoundationAgent",
    "GatedSchedulingAgent",
    "GatedAtomicInboundSchedulerAgent",
    "GatedRegularizedAgent",
    "masked_max",
    "mode_only_temporal_value",
    "normalized_logsumexp",
    "temporal_mode_value",
]
