"""Strictly nested handling-cost augmentation for frozen VCG 1.1.

The operational controller is retained as an opaque, frozen base agent.  The
cost critic consumes detached features from its frozen encoders but owns an
entirely separate cost tower and optimizer.  At ``lambda=0`` selection
delegates directly to the base agent, making operational parity a structural
invariant rather than a training outcome.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
import hashlib
import json
import math
import random
from typing import Iterable, Mapping, Optional, Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from PSLAP.viability_candidates import ViabilityMode
from viability_graph_hierarchy import (
    ID_TO_MODE,
    NoCertifiedViableAction,
    ViabilityGraphCandidateRecord,
    ViabilityGraphDecision,
    ViabilityGraphHierarchyAgent,
    encode_recovery_state,
    pad_yard_graphs,
    regularized_mode_values,
)


METHOD_VERSION = "vcg_v1_1_detached_handling_augmentation_v1"
COST_ARCHITECTURE = (
    "frozen_v1_detached_embedding_exact_immediate_reconfigure_"
    "softplus_future_cost_head_v1"
)


def _config_binding(config) -> str:
    if is_dataclass(config):
        value = asdict(config)
    elif hasattr(config, "__dict__"):
        value = dict(config.__dict__)
    else:
        raise TypeError("V1 configuration cannot be bound canonically")
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


class DetachedHandlingCostNetwork(nn.Module):
    """Cost-only tower over frozen, detached V1.1 candidate embeddings."""

    def __init__(self, config, *, seed: int) -> None:
        super().__init__()
        self.config = config
        self.seed = int(seed)
        width = 3 * config.graph_embedding_dim + config.action_embedding_dim
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.seed)
            self.cost_head = nn.Sequential(
                nn.Linear(width, config.head_hidden_dim),
                nn.SiLU(),
                nn.Linear(config.head_hidden_dim, config.head_hidden_dim),
                nn.SiLU(),
                nn.Linear(config.head_hidden_dim, 1),
            )

    def forward(self, detached_features: Tensor) -> Tensor:
        if detached_features.ndim != 2:
            raise ValueError("detached cost features must be a matrix")
        return F.softplus(self.cost_head(detached_features).squeeze(-1))


def _record_tensors(records: Sequence[ViabilityGraphCandidateRecord]):
    records = tuple(records)
    if not records:
        raise NoCertifiedViableAction("handling critic received an empty frontier")
    return (
        tuple(record.current_state for record in records),
        tuple(record.successor_state for record in records),
        tuple(record.action_features for record in records),
    )


def detached_v11_features(base_network, records) -> Tensor:
    """Mirror V1.1's encoder prefix without retaining an autograd path."""

    current_states, successor_states, action_features = _record_tensors(records)
    parameter = next(base_network.parameters())
    action_features = torch.as_tensor(
        action_features, dtype=parameter.dtype, device=parameter.device
    )
    with torch.no_grad():
        graphs = tuple(
            encode_recovery_state(
                state, timing_scale=base_network.config.timing_scale
            )
            for state in current_states + successor_states
        )
        embedding = base_network.graph_encoder(pad_yard_graphs(graphs))
        count = len(current_states)
        current = embedding[:count]
        successor = embedding[count:]
        action = base_network.action_encoder(action_features)
        features = torch.cat(
            (current, successor, successor - current, action), dim=-1
        )
    return features.detach()


def score_cost_records(
    network: DetachedHandlingCostNetwork,
    base_network,
    records: Sequence[ViabilityGraphCandidateRecord],
    *,
    grad: bool = False,
) -> Tensor:
    features = detached_v11_features(base_network, records)
    immediate = torch.as_tensor(
        [1.0 if record.action_type == "reconfigure" else 0.0 for record in records],
        dtype=features.dtype,
        device=features.device,
    )
    if grad:
        return immediate + network(features)
    with torch.no_grad():
        return immediate + network(features)


class HandlingAugmentedV11Agent:
    """Frozen VCG 1.1 plus an optional detached handling-cost merit."""

    def __init__(
        self,
        base_agent: ViabilityGraphHierarchyAgent,
        cost_network: DetachedHandlingCostNetwork,
        *,
        handling_lambda: float,
    ) -> None:
        self.base_agent = base_agent
        for base_network in (self.base_agent.Q_local, self.base_agent.Q_target):
            base_network.requires_grad_(False).eval()
        self.cost_network = cost_network.to(base_agent.device)
        self.cost_network.eval()
        self.handling_lambda = self._validate_lambda(handling_lambda)
        # Fail immediately if an implementation accidentally shares storage.
        operational_pointers = {
            parameter.data_ptr()
            for base_network in (self.base_agent.Q_local, self.base_agent.Q_target)
            for parameter in base_network.parameters()
        }
        cost_pointers = {
            parameter.data_ptr() for parameter in self.cost_network.parameters()
        }
        if operational_pointers.intersection(cost_pointers):
            raise RuntimeError("operational and handling critics share parameters")

    @staticmethod
    def _validate_lambda(value: float) -> float:
        value = float(value)
        if not math.isfinite(value) or value < 0.0:
            raise ValueError("handling lambda must be finite and nonnegative")
        return value

    @property
    def config(self):
        return self.base_agent.config

    @property
    def device(self):
        return self.base_agent.device

    def set_handling_lambda(self, value: float) -> None:
        self.handling_lambda = self._validate_lambda(value)

    def reset_episode_state(self) -> None:
        self.base_agent.reset_episode_state()

    on_episode_reset = reset_episode_state

    def observe_outcome(self, *args, **kwargs):
        return self.base_agent.observe_outcome(*args, **kwargs)

    def remember(self, *args, **kwargs):
        raise RuntimeError("handling-augmented V1.1 pilot is evaluation-only")

    def learn(self, *args, **kwargs):
        raise RuntimeError("handling-augmented V1.1 pilot cannot update Qop")

    def audit(self, *args, **kwargs) -> dict:
        result = self.base_agent.audit(*args, **kwargs)
        result.update(
            {
                "method_version": METHOD_VERSION,
                "handling_cost_architecture": COST_ARCHITECTURE,
                "handling_lambda": self.handling_lambda,
                "operational_controller_frozen": True,
                "cost_parameters_disjoint": True,
                "lambda_zero_direct_delegation": True,
            }
        )
        return result

    def select(
        self,
        snapshot,
        *,
        training: bool = False,
        epsilon: Optional[float] = None,
    ) -> ViabilityGraphDecision:
        if training:
            raise ValueError("handling-augmented V1.1 pilot is evaluation-only")
        # This direct call is the central nesting guarantee.  No cost-network
        # forward pass, merit reconstruction, or alternate tie-breaking occurs.
        if self.handling_lambda == 0.0:
            return self.base_agent.select(
                snapshot, training=training, epsilon=epsilon
            )
        epsilon_value = 0.0 if epsilon is None else float(epsilon)
        if epsilon_value != 0.0:
            raise ValueError("positive-lambda deployment requires epsilon=0")

        prepared, liveness_forced = self.base_agent._admissible_prepared(snapshot)
        if not prepared.records:
            raise NoCertifiedViableAction("no exact SAFE candidate to score")
        operational = self.base_agent._score_records(prepared.records)
        # A liveness witness is a hard-layer decision.  The learned cost tower
        # is deliberately bypassed so it cannot veto or crash that mandate.
        handling = (
            torch.zeros_like(operational)
            if liveness_forced
            else score_cost_records(
                self.cost_network, self.base_agent.Q_local, prepared.records
            )
        )
        if not bool(torch.isfinite(handling).all()):
            raise ValueError("handling-cost critic produced a nonfinite value")
        merit = operational - self.handling_lambda * handling
        modes = torch.as_tensor(
            prepared.mode_ids, dtype=torch.long, device=merit.device
        )
        mode_values, live_modes = regularized_mode_values(
            merit, modes, self.base_agent.within_temperatures
        )

        if liveness_forced:
            selected_index = 0
            selection_source = "exact_recovery_witness_guard"
        else:
            maximum_mode_value = float(mode_values.max().item())
            best_mode_positions = [
                index
                for index, value in enumerate(mode_values.tolist())
                if value == maximum_mode_value
            ]
            mode_position = min(
                best_mode_positions,
                key=lambda index: int(live_modes[index].item()),
            )
            selected_mode = int(live_modes[mode_position].item())
            mode_indices = [
                index
                for index, mode_id in enumerate(prepared.mode_ids)
                if mode_id == selected_mode
            ]
            maximum_merit = max(
                float(merit[index].item()) for index in mode_indices
            )
            selected_index = min(
                (
                    index
                    for index in mode_indices
                    if float(merit[index].item()) == maximum_merit
                ),
                key=lambda index: prepared.records[index].key,
            )
            selection_source = (
                "singleton_safe"
                if len(prepared.records) == 1
                else "regularized_mode_map_candidate_map_handling_augmented"
            )

        source_index = prepared.source_indices[selected_index]
        candidate = snapshot.candidates[source_index]
        record = prepared.records[selected_index]
        exact_rank_progress = bool(
            snapshot.audit.recovery_rank_exact
            and candidate.mode is ViabilityMode.RECOVER
            and candidate.rank_delta is not None
            and candidate.rank_delta > 0
        )
        mode_pairs = tuple(
            (ID_TO_MODE[int(mode.item())], float(value.item()))
            for value, mode in zip(mode_values, live_modes)
        )
        decision = ViabilityGraphDecision(
            candidate=candidate,
            record=record,
            prepared_snapshot=prepared,
            q_values=tuple(float(value) for value in merit.tolist()),
            mode_values=mode_pairs,
            explored=False,
            selection_source=selection_source,
            liveness_forced=liveness_forced,
            exact_rank_progress=exact_rank_progress,
        )

        base = self.base_agent
        base.decision_count += 1
        base.mode_decisions[candidate.mode.value] += 1
        base.action_decisions[candidate.action_type.value] += 1
        base.selection_sources[selection_source] += 1
        base.safe_candidates_scored += len(prepared.records)
        base.interface_rejections += prepared.interface_rejection_count
        base.exact_rejections_seen += int(snapshot.audit.fail_closed_rejection_count)
        base.decision_log.append(
            {
                "decision_index": base.decision_count - 1,
                "episode_instance_id": snapshot.episode_instance_id,
                "decision_epoch": int(snapshot.decision_epoch),
                "exact_safe_candidate_count": prepared.exact_safe_candidate_count,
                "admissible_candidate_count": len(prepared.records),
                "liveness_restricted": prepared.liveness_restricted,
                "interface_rejection_count": prepared.interface_rejection_count,
                "verifier_fail_closed_rejection_count": int(
                    snapshot.audit.fail_closed_rejection_count
                ),
                "live_modes": tuple(name for name, _ in mode_pairs),
                "selected_key": candidate.key,
                "selected_mode": candidate.mode.value,
                "selected_action_type": candidate.action_type.value,
                "selection_source": selection_source,
                "explored": False,
                "handling_lambda": self.handling_lambda,
                "selected_operational_q": float(operational[selected_index]),
                "selected_handling_q": float(handling[selected_index]),
                "selected_merit": float(merit[selected_index]),
                "liveness_forced": liveness_forced,
                "exact_rank_progress": exact_rank_progress,
            }
        )
        return decision

    def __getattr__(self, name):
        # Preserve the executor-facing V1 surface without copying mutable state.
        return getattr(self.base_agent, name)


@dataclass(frozen=True)
class MonteCarloCostSample:
    record: ViabilityGraphCandidateRecord
    remaining_physical_rehandles: float

    def __post_init__(self) -> None:
        value = float(self.remaining_physical_rehandles)
        if not math.isfinite(value) or value < 0.0:
            raise ValueError("cost target must be finite and nonnegative")


def replay_episodes(transitions: Iterable) -> tuple[tuple, ...]:
    """Recover complete episodes from the chronological V1.1 replay.

    V1.1's replay does not store an episode identifier, but it retains every
    transition in chronological order and marks each terminal boundary.  A
    partial leading/trailing episode would invalidate a Monte-Carlo label and
    is therefore rejected rather than silently used.
    """

    episodes = []
    current = []
    for index, transition in enumerate(tuple(transitions)):
        if not isinstance(getattr(transition, "done", None), bool):
            raise TypeError(f"replay transition {index} has no strict done flag")
        chosen = getattr(transition, "chosen", None)
        if not isinstance(chosen, ViabilityGraphCandidateRecord):
            raise TypeError(f"replay transition {index} has no V1 candidate record")
        next_candidates = tuple(getattr(transition, "next_candidates", ()))
        if transition.done and next_candidates:
            raise ValueError("terminal replay transition retained next candidates")
        if not transition.done and not next_candidates:
            raise ValueError("nonterminal replay transition lacks a next frontier")
        current.append(transition)
        if transition.done:
            episodes.append(tuple(current))
            current = []
    if current:
        raise ValueError("V1.1 replay ends with a partial episode")
    if not episodes:
        raise ValueError("V1.1 replay contains no complete episode")
    return tuple(episodes)


def monte_carlo_samples_from_episodes(
    episodes: Sequence[Sequence],
) -> tuple[tuple[MonteCarloCostSample, ...], ...]:
    """Label each behavior decision by remaining physical relocations.

    In the frozen V1.1 executor, physical storage relocation occurs exactly
    in a standalone ``reconfigure`` macro.  The pilot authenticates this
    identity against the source training summaries before fitting the head.
    """

    output = []
    for episode_index, episode in enumerate(episodes):
        if not episode or not episode[-1].done:
            raise ValueError(f"episode {episode_index} is incomplete")
        remaining = 0
        reversed_samples = []
        for transition in reversed(tuple(episode)):
            immediate = int(transition.chosen.action_type == "reconfigure")
            remaining += immediate
            reversed_samples.append(
                MonteCarloCostSample(transition.chosen, float(remaining))
            )
        output.append(tuple(reversed(reversed_samples)))
    return tuple(output)


def split_episode_samples(
    episode_samples: Sequence[Sequence[MonteCarloCostSample]],
    *,
    validation_fraction: float,
    seed: int,
) -> tuple[tuple[MonteCarloCostSample, ...], tuple[MonteCarloCostSample, ...], dict]:
    """Deterministically split whole episodes, never individual transitions."""

    episodes = tuple(tuple(episode) for episode in episode_samples)
    if len(episodes) < 2:
        raise ValueError("episode split requires at least two episodes")
    fraction = float(validation_fraction)
    if not 0.0 < fraction < 1.0:
        raise ValueError("validation fraction must lie strictly inside (0, 1)")
    order = list(range(len(episodes)))
    random.Random(int(seed)).shuffle(order)
    validation_count = max(1, min(len(episodes) - 1, round(len(episodes) * fraction)))
    validation_indices = frozenset(order[:validation_count])
    train = tuple(
        sample
        for index, episode in enumerate(episodes)
        if index not in validation_indices
        for sample in episode
    )
    validation = tuple(
        sample
        for index, episode in enumerate(episodes)
        if index in validation_indices
        for sample in episode
    )
    return train, validation, {
        "episode_count": len(episodes),
        "training_episode_count": len(episodes) - validation_count,
        "validation_episode_count": validation_count,
        "training_transition_count": len(train),
        "validation_transition_count": len(validation),
        "validation_episode_indices": tuple(sorted(validation_indices)),
        "split_seed": int(seed),
    }


@dataclass(frozen=True)
class CachedCostDataset:
    features: Tensor
    immediate_cost: Tensor
    targets: Tensor

    def __post_init__(self) -> None:
        count = int(self.features.shape[0])
        if self.features.ndim != 2:
            raise ValueError("cached cost features must be a matrix")
        if self.immediate_cost.shape != (count,) or self.targets.shape != (count,):
            raise ValueError("cached cost tensors have inconsistent shapes")

    def __len__(self) -> int:
        return int(self.features.shape[0])


def cache_cost_dataset(
    base_network,
    samples: Sequence[MonteCarloCostSample],
    *,
    batch_size: int = 256,
    storage_device: str | torch.device = "cpu",
) -> CachedCostDataset:
    """Encode frozen V1.1 features once; later optimization is cost-head only."""

    samples = tuple(samples)
    if not samples:
        raise ValueError("cannot cache an empty cost dataset")
    if batch_size <= 0:
        raise ValueError("cache batch size must be positive")
    feature_batches = []
    for start in range(0, len(samples), int(batch_size)):
        records = tuple(
            sample.record for sample in samples[start : start + int(batch_size)]
        )
        feature_batches.append(detached_v11_features(base_network, records).cpu())
    features = torch.cat(feature_batches, dim=0).to(storage_device)
    immediate = torch.as_tensor(
        [1.0 if sample.record.action_type == "reconfigure" else 0.0 for sample in samples],
        dtype=features.dtype,
        device=storage_device,
    )
    targets = torch.as_tensor(
        [sample.remaining_physical_rehandles for sample in samples],
        dtype=features.dtype,
        device=storage_device,
    )
    if bool(torch.any(targets + 1e-12 < immediate)):
        raise ValueError("remaining-cost target is below exact immediate cost")
    return CachedCostDataset(features, immediate, targets)


def _dataset_metrics(
    network: DetachedHandlingCostNetwork,
    dataset: CachedCostDataset,
    *,
    device: torch.device,
    batch_size: int,
) -> dict:
    predictions = []
    network.eval()
    with torch.no_grad():
        for start in range(0, len(dataset), int(batch_size)):
            stop = start + int(batch_size)
            features = dataset.features[start:stop].to(device)
            immediate = dataset.immediate_cost[start:stop].to(device)
            predictions.append((immediate + network(features)).cpu())
    prediction = torch.cat(predictions)
    target = dataset.targets.cpu()
    return {
        "mae_rehandles": float(torch.mean(torch.abs(prediction - target)).item()),
        "smooth_l1": float(F.smooth_l1_loss(prediction, target, beta=1.0).item()),
        "prediction_mean": float(prediction.mean().item()),
        "target_mean": float(target.mean().item()),
    }


def fit_cached_cost_network(
    network: DetachedHandlingCostNetwork,
    training: CachedCostDataset,
    validation: CachedCostDataset,
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    grad_clip: float = 5.0,
) -> dict:
    """Fit only the detached future-cost head and retain best validation state."""

    if epochs <= 0 or batch_size <= 0:
        raise ValueError("epochs and batch size must be positive")
    device = next(network.parameters()).device
    optimizer = torch.optim.Adam(network.parameters(), lr=float(learning_rate))
    rng = random.Random(int(seed))
    best_state = None
    best_validation = math.inf
    history = []
    optimizer_steps = 0
    for epoch in range(1, int(epochs) + 1):
        order = list(range(len(training)))
        rng.shuffle(order)
        network.train()
        losses = []
        for start in range(0, len(order), int(batch_size)):
            indices = order[start : start + int(batch_size)]
            features = training.features[indices].to(device)
            immediate = training.immediate_cost[indices].to(device)
            target = training.targets[indices].to(device)
            prediction = immediate + network(features)
            loss = F.smooth_l1_loss(prediction, target, beta=1.0)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(network.parameters(), float(grad_clip))
            optimizer.step()
            optimizer_steps += 1
            losses.append(float(loss.detach().item()))
        validation_metrics = _dataset_metrics(
            network, validation, device=device, batch_size=batch_size
        )
        history.append(
            {
                "epoch": epoch,
                "training_batch_loss_mean": float(sum(losses) / len(losses)),
                "validation": validation_metrics,
            }
        )
        if validation_metrics["smooth_l1"] < best_validation:
            best_validation = validation_metrics["smooth_l1"]
            best_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in network.state_dict().items()
            }
    if best_state is None:
        raise RuntimeError("cost fitting did not produce a validation state")
    network.load_state_dict(best_state, strict=True)
    network.eval()
    return {
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "learning_rate": float(learning_rate),
        "optimizer_steps": optimizer_steps,
        "selected_epoch": min(
            history,
            key=lambda row: row["validation"]["smooth_l1"],
        )["epoch"],
        "training": _dataset_metrics(
            network, training, device=device, batch_size=batch_size
        ),
        "validation": _dataset_metrics(
            network, validation, device=device, batch_size=batch_size
        ),
        "history": tuple(history),
    }


class CostTrajectoryCollector:
    """V1 behavior wrapper that retains chosen records and exact macro costs."""

    def __init__(self, base_agent: ViabilityGraphHierarchyAgent, *, epsilon: float) -> None:
        self.base_agent = base_agent
        self.epsilon = float(epsilon)
        if not 0.0 <= self.epsilon <= 1.0:
            raise ValueError("collection epsilon must lie in [0, 1]")
        self._chosen: list[ViabilityGraphCandidateRecord] = []
        self._costs: list[int] = []
        self._pending_decision: Optional[ViabilityGraphDecision] = None
        self._pending_execution = None

    @property
    def config(self):
        return self.base_agent.config

    def reset_episode_state(self) -> None:
        if self._pending_decision is not None or self._pending_execution is not None:
            raise RuntimeError("collector reset with an unfinished macro")
        self._chosen.clear()
        self._costs.clear()
        self.base_agent.reset_episode_state()

    def select(self, snapshot, *, training=False, epsilon=None):
        if self._pending_decision is not None:
            raise RuntimeError("collector received two selections without an outcome")
        decision = self.base_agent.select(
            snapshot, training=True, epsilon=self.epsilon
        )
        self._pending_decision = decision
        return decision

    def record_execution(self, execution) -> None:
        if self._pending_decision is None or self._pending_execution is not None:
            raise RuntimeError("collector execution is out of sequence")
        self._pending_execution = execution

    def observe_outcome(self, decision, *, next_snapshot, done):
        if decision is not self._pending_decision or self._pending_execution is None:
            raise RuntimeError("collector outcome is out of sequence")
        result = self.base_agent.observe_outcome(
            decision, next_snapshot=next_snapshot, done=done
        )
        self._chosen.append(decision.record)
        self._costs.append(int(self._pending_execution.relocations))
        self._pending_decision = None
        self._pending_execution = None
        return result

    def samples(self) -> tuple[MonteCarloCostSample, ...]:
        if self._pending_decision is not None or self._pending_execution is not None:
            raise RuntimeError("collector episode has an unfinished macro")
        remaining = 0
        reversed_samples = []
        for record, cost in zip(reversed(self._chosen), reversed(self._costs)):
            remaining += int(cost)
            reversed_samples.append(
                MonteCarloCostSample(record, float(remaining))
            )
        return tuple(reversed(reversed_samples))

    def __getattr__(self, name):
        return getattr(self.base_agent, name)


def fit_cost_network(
    network: DetachedHandlingCostNetwork,
    base_network,
    samples: Sequence[MonteCarloCostSample],
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    grad_clip: float = 5.0,
) -> dict:
    samples = tuple(samples)
    if not samples:
        raise ValueError("cost training requires at least one sample")
    if epochs <= 0 or batch_size <= 0:
        raise ValueError("epochs and batch size must be positive")
    device = next(network.parameters()).device
    network.train()
    optimizer = torch.optim.Adam(network.parameters(), lr=float(learning_rate))
    rng = random.Random(int(seed))
    losses = []
    for _ in range(int(epochs)):
        order = list(range(len(samples)))
        rng.shuffle(order)
        for start in range(0, len(order), int(batch_size)):
            batch = [samples[index] for index in order[start : start + batch_size]]
            prediction = score_cost_records(
                network,
                base_network,
                tuple(sample.record for sample in batch),
                grad=True,
            )
            target = torch.as_tensor(
                [sample.remaining_physical_rehandles for sample in batch],
                dtype=prediction.dtype,
                device=device,
            )
            loss = F.smooth_l1_loss(prediction, target, beta=1.0)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(network.parameters(), float(grad_clip))
            optimizer.step()
            losses.append(float(loss.detach().item()))
    network.eval()
    with torch.no_grad():
        prediction = score_cost_records(
            network,
            base_network,
            tuple(sample.record for sample in samples),
        )
        target = torch.as_tensor(
            [sample.remaining_physical_rehandles for sample in samples],
            dtype=prediction.dtype,
            device=device,
        )
        mae = float(torch.mean(torch.abs(prediction - target)).item())
    return {
        "sample_count": len(samples),
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "optimizer_steps": len(losses),
        "last_loss": losses[-1],
        "training_mae_rehandles": mae,
        "target_mean": float(
            sum(sample.remaining_physical_rehandles for sample in samples)
            / len(samples)
        ),
    }


def cost_checkpoint(
    network: DetachedHandlingCostNetwork,
    *,
    source_checkpoint_sha256: str,
    source_policy_digest: str,
    model_seed: int,
    training_record: Mapping,
) -> dict:
    return {
        "method_version": METHOD_VERSION,
        "cost_architecture": COST_ARCHITECTURE,
        "source_vcg_1_1_checkpoint_sha256": source_checkpoint_sha256,
        "source_vcg_1_1_policy_digest": source_policy_digest,
        "model_seed": int(model_seed),
        "v1_config_sha256": _config_binding(network.config),
        "cost_seed": int(network.seed),
        "operational_controller_frozen": True,
        "cost_parameters_disjoint": True,
        "lambda_zero_direct_delegation": True,
        "training_record": dict(training_record),
        "cost_state_dict": network.state_dict(),
    }


def load_cost_checkpoint(
    payload: Mapping,
    *,
    config,
    device,
    expected_source_checkpoint_sha256: str,
    expected_source_policy_digest: str,
    expected_model_seed: int,
) -> DetachedHandlingCostNetwork:
    if (
        payload.get("method_version") != METHOD_VERSION
        or payload.get("cost_architecture") != COST_ARCHITECTURE
        or payload.get("operational_controller_frozen") is not True
        or payload.get("cost_parameters_disjoint") is not True
        or payload.get("lambda_zero_direct_delegation") is not True
    ):
        raise ValueError("incompatible detached handling-cost checkpoint")
    expected = {
        "source_vcg_1_1_checkpoint_sha256": str(
            expected_source_checkpoint_sha256
        ),
        "source_vcg_1_1_policy_digest": str(expected_source_policy_digest),
        "model_seed": int(expected_model_seed),
        "v1_config_sha256": _config_binding(config),
    }
    for name, value in expected.items():
        if payload.get(name) != value:
            raise ValueError(f"cost checkpoint/base binding mismatch: {name}")
    network = DetachedHandlingCostNetwork(config, seed=int(payload["cost_seed"]))
    network.load_state_dict(payload["cost_state_dict"], strict=True)
    network.to(device).requires_grad_(False).eval()
    return network
