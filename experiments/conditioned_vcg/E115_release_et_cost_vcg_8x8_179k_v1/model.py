"""Factorized queue-observed critic with conservative ranking updates."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import math
from typing import Optional, Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from viability_graph_hierarchy import (
    ACTION_FEATURE_DIM,
    NoCertifiedViableAction,
    ViabilityGraphCandidateRecord,
    ViabilityGraphDecision,
    ViabilityGraphTransition,
    regularized_mode_values,
)
from viability_graph_preference_conditioned import select_hierarchical_index
from viability_graph_episodic_audit import (
    EPISODIC_VIABILITY_GRAPH_CHECKPOINT_FAMILY,
    EPISODIC_VIABILITY_GRAPH_CHECKPOINT_SCHEMA_VERSION,
    EpisodicViabilityGraphConfig,
)
from experiments.conditioned_vcg.E100_accelerated_release_clock_163k_v1 import (
    fast_inputs,
)
from experiments.conditioned_vcg.E107_queue_context_stage_a_8x8_171k_v1.representation import (
    QUEUE_CONTEXT_DIM,
    QUEUE_CONTEXT_NAMES,
    QueueContextAgent,
    QueueContextQNetwork,
    TIME_CONTEXT_CONTRACT,
)
from .rewards import (
    COMPONENT_NAMES,
    ReleaseETCostEnv,
    SCALAR_WEIGHTS,
    scalarized_components,
)


CONTROLLER_ARCHITECTURE = "e115_exact_safe_release_et_cost_graph_smdp_q_v1"
NETWORK_ARCHITECTURE = "e115_due_observed_shared_four_cost_q_v1"
BACKUP_VERSION = "e115_deterministic_shared_candidate_vector_double_q_v1"
CONSERVATIVE_UPDATE = "frozen_target_confident_pairwise_margin_v1"

LR_PHASES = (
    (18_000, 5.0e-5),
    (24_000, 1.0e-5),
    (math.inf, 3.0e-6),
)
ANCHOR_CONFIDENCE = 0.02
ANCHOR_MARGIN = 0.01
ANCHOR_LOSS_WEIGHT = 0.25
# Match each raw component's marginal coefficient in the unscaled objective.
# This prevents a one-unit rehandle error from being treated like a one-unit
# step error merely because the heads are fitted separately.
COMPONENT_LOSS_WEIGHTS = (2.0, 1.0, 0.1, 10.0)
MAX_ANCHOR_FRONTIERS_PER_UPDATE = 8
MAX_TARGET_CACHE_ENTRIES = 500_000


def learning_rate_at(decision_count: int) -> float:
    if isinstance(decision_count, bool) or int(decision_count) < 0:
        raise ValueError("decision_count must be nonnegative")
    count = int(decision_count)
    for boundary, value in LR_PHASES:
        if count < boundary:
            return float(value)
    raise AssertionError("learning-rate schedule has no terminal phase")


def scalarize_tensor(vectors: Tensor) -> Tensor:
    if not torch.is_tensor(vectors) or vectors.ndim < 1:
        raise ValueError("component vectors must be a tensor")
    if vectors.shape[-1] != len(COMPONENT_NAMES):
        raise ValueError("component vector tensor has the wrong width")
    weights = vectors.new_tensor(SCALAR_WEIGHTS)
    return torch.sum(vectors * weights, dim=-1)


class FactorizedQNetwork(QueueContextQNetwork):
    """One shared graph/context trunk with four auditable return heads."""

    def __init__(self, config) -> None:
        super().__init__(config)
        del self.q_head
        input_width = (
            3 * config.graph_embedding_dim
            + config.action_embedding_dim
            + config.action_embedding_dim
        )
        self.consequence_trunk = nn.Sequential(
            nn.Linear(input_width, config.head_hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(config.head_hidden_dim),
            nn.Linear(config.head_hidden_dim, config.head_hidden_dim),
            nn.SiLU(),
        )
        self.service_head = nn.Linear(config.head_hidden_dim, 1)
        self.wait_head = nn.Linear(config.head_hidden_dim, 1)
        self.step_head = nn.Linear(config.head_hidden_dim, 1)
        self.rehandle_head = nn.Linear(config.head_hidden_dim, 1)
        # Cost heads initially contribute the same value to every action, so
        # they create no arbitrary preference.  A zero softplus preactivation
        # retains useful gradient; a large negative bias would nearly freeze
        # a cost head exactly when early credit is most important.
        with torch.no_grad():
            for head in (
                self.service_head,
                self.wait_head,
                self.step_head,
                self.rehandle_head,
            ):
                head.weight.zero_()
                head.bias.zero_()

    def score_component_embeddings(
        self,
        current_embedding: Tensor,
        successor_embedding: Tensor,
        action_features,
        queue_context: Tensor,
    ) -> Tensor:
        parameter = next(self.parameters())
        action = torch.as_tensor(
            action_features, dtype=parameter.dtype, device=parameter.device
        )
        if action.ndim == 1:
            action = action.unsqueeze(0)
        count = current_embedding.shape[0]
        if action.shape != (count, ACTION_FEATURE_DIM):
            raise ValueError("E115 action feature matrix has the wrong shape")
        if queue_context.shape != (count, QUEUE_CONTEXT_DIM):
            raise ValueError("E115 queue context matrix has the wrong shape")
        latent = self.consequence_trunk(torch.cat((
            current_embedding,
            successor_embedding,
            successor_embedding - current_embedding,
            self.action_encoder(action),
            self.context_encoder(queue_context),
        ), dim=-1))
        return torch.cat((
            F.softplus(self.service_head(latent)),
            F.softplus(self.wait_head(latent)),
            F.softplus(self.step_head(latent)),
            F.softplus(self.rehandle_head(latent)),
        ), dim=-1)

    def score_embeddings(
        self,
        current_embedding: Tensor,
        successor_embedding: Tensor,
        action_features,
        queue_context: Tensor,
    ) -> Tensor:
        return scalarize_tensor(self.score_component_embeddings(
            current_embedding, successor_embedding, action_features, queue_context
        ))

    def forward_components(self, current_states, successor_states, action_features):
        current = tuple(current_states)
        successor = tuple(successor_states)
        if not current or len(current) != len(successor):
            raise ValueError("E115 candidate state counts must be nonempty and equal")
        embeddings = fast_inputs._embed(self, current + successor)
        count = len(current)
        return self.score_component_embeddings(
            embeddings[:count],
            embeddings[count:],
            action_features,
            self.context_tensor(current),
        )


@dataclass(frozen=True)
class FactorizedTransition(ViabilityGraphTransition):
    component_returns: tuple[float, ...]

    def __post_init__(self) -> None:
        super().__post_init__()
        values = tuple(float(value) for value in self.component_returns)
        if len(values) != len(COMPONENT_NAMES):
            raise ValueError("E115 transition component width differs")
        if not all(math.isfinite(value) for value in values):
            raise ValueError("E115 transition components must be finite")
        if any(value < 0.0 for value in values):
            raise ValueError("E115 cost components cannot be negative")
        expected = scalarized_components(values)
        if not math.isclose(float(self.reward) * 0.01, expected,
                            rel_tol=0.0, abs_tol=2.0e-6):
            raise ValueError("E115 transition scalar and components disagree")

    def numeric_copy(self) -> "FactorizedTransition":
        return FactorizedTransition(
            chosen=self.chosen,
            reward=float(self.reward),
            duration=int(self.duration),
            next_candidates=tuple(self.next_candidates),
            done=bool(self.done),
            component_returns=tuple(float(value) for value in self.component_returns),
        )


class FactorizedConservativeAgent(QueueContextAgent):
    """Queue-visible vector Double-Q agent with a conservative margin term."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.seed)
            self.Q_local = FactorizedQNetwork(self.config).to(self.device)
        self.Q_target = deepcopy(self.Q_local).to(self.device)
        self.Q_target.requires_grad_(False).eval()
        self.optimizer = torch.optim.Adam(
            self.Q_local.parameters(), lr=learning_rate_at(0)
        )
        self.factorized_environment = None
        self.component_cursor = 0
        self.component_loss_history: list[tuple[float, ...]] = []
        self.anchor_loss_history: list[float] = []
        self.anchor_pairs_used = 0
        self._target_component_cache: dict[int, tuple[object, tuple[float, ...]]] = {}
        self._target_component_cache_generation = -1
        self._target_component_cache_hits = 0
        self._target_component_cache_misses = 0

    def bind_context_environment(self, env, **kwargs) -> None:
        super().bind_context_environment(env, **kwargs)
        if not isinstance(env, ReleaseETCostEnv):
            raise TypeError("E115 requires ReleaseETCostEnv")
        self.factorized_environment = env

    def reset_episode_state(self) -> None:
        super().reset_episode_state()
        if self.factorized_environment is not None:
            self.component_cursor = len(
                self.factorized_environment.factorized_component_records
            )

    def _component_scores(
        self,
        records: Sequence[ViabilityGraphCandidateRecord],
        *,
        network: Optional[FactorizedQNetwork] = None,
        grad: bool = False,
    ) -> Tensor:
        records = tuple(records)
        if not records:
            raise NoCertifiedViableAction("no exact SAFE candidate to score")
        network = network or self.Q_local
        if not isinstance(network, FactorizedQNetwork):
            raise TypeError("E115 received another network family")
        current = tuple(record.current_state for record in records)
        successor = tuple(record.successor_state for record in records)
        features = tuple(record.action_features for record in records)
        if grad:
            return network.forward_components(current, successor, features)
        with torch.no_grad():
            return network.forward_components(current, successor, features)

    def _score_records(self, records, *, network=None, grad=False) -> Tensor:
        return scalarize_tensor(self._component_scores(
            records, network=network, grad=grad
        ))

    def _macro_components(self, duration: int) -> tuple[float, ...]:
        if self.factorized_environment is None:
            raise RuntimeError("E115 agent is not bound to an environment")
        records = self.factorized_environment.factorized_component_records
        start = self.component_cursor
        stop = len(records)
        if stop - start != int(duration):
            raise RuntimeError("E115 primitive component count differs from macro duration")
        totals = [0.0] * len(COMPONENT_NAMES)
        for offset, values in enumerate(records[start:stop]):
            discount = float(self.config.gamma) ** offset
            for index, value in enumerate(values):
                totals[index] += discount * float(value)
        self.component_cursor = stop
        return tuple(totals)

    def _target_components(self, records) -> Tensor:
        """Exact frozen-target vector cache, invalidated at every target refresh."""

        records = tuple(records)
        generation = self.gradient_steps // self.config.target_update_every
        if self._target_component_cache_generation != generation:
            self._target_component_cache.clear()
            self._target_component_cache_generation = generation
        cache = self._target_component_cache
        missing = []
        seen = set()
        hits = 0
        for record in records:
            identity = id(record)
            entry = cache.get(identity)
            if entry is not None and entry[0] is record:
                hits += 1
            elif identity not in seen:
                missing.append(record)
                seen.add(identity)
        if missing and len(cache) + len(missing) > MAX_TARGET_CACHE_ENTRIES:
            cache.clear()
            missing = []
            seen.clear()
            hits = 0
            for record in records:
                if id(record) not in seen:
                    missing.append(record)
                    seen.add(id(record))
        if missing:
            values = self._component_scores(
                missing, network=self.Q_target
            ).detach().cpu().tolist()
            for record, vector in zip(missing, values):
                cache[id(record)] = (
                    record, tuple(float(value) for value in vector)
                )
        self._target_component_cache_hits += hits
        self._target_component_cache_misses += len(records) - hits
        parameter = next(self.Q_target.parameters())
        return torch.as_tensor(
            [cache[id(record)][1] for record in records],
            dtype=parameter.dtype,
            device=parameter.device,
        )

    def _anchor_term(self, records, target_vectors: Tensor) -> tuple[Tensor, int]:
        """Preserve confident frozen-target mode/candidate preferences locally.

        This is deliberately a one-target-period trust region.  It is not a
        counterfactual label and is not described as proof that the incumbent
        action is operationally better.
        """

        target_merit = scalarize_tensor(target_vectors)
        mode_ids = torch.as_tensor(
            [record.mode_id for record in records],
            dtype=torch.long,
            device=self.device,
        )
        target_mode_values, live_modes = regularized_mode_values(
            target_merit, mode_ids, self.within_temperatures
        )
        selected_mode_position = int(torch.argmax(target_mode_values).item())
        selected_mode = int(live_modes[selected_mode_position].item())
        terms = []
        used = 0
        online_merit = None

        if len(target_mode_values) >= 2:
            ordered = torch.sort(target_mode_values, descending=True).values
            if float(ordered[0] - ordered[1]) >= ANCHOR_CONFIDENCE:
                online_merit = scalarize_tensor(self._component_scores(
                    records, network=self.Q_local, grad=True
                ))
                online_mode_values, online_live_modes = regularized_mode_values(
                    online_merit, mode_ids, self.within_temperatures
                )
                position = int((online_live_modes == selected_mode).nonzero()[0].item())
                alternatives = torch.cat((
                    online_mode_values[:position], online_mode_values[position + 1:]
                ))
                terms.append(F.relu(
                    online_merit.new_tensor(ANCHOR_MARGIN)
                    - (online_mode_values[position] - alternatives.max())
                ))
                used += 1

        selected_indices = torch.nonzero(
            mode_ids == selected_mode, as_tuple=False
        ).flatten()
        if len(selected_indices) >= 2:
            selected_target_merits = target_merit.index_select(0, selected_indices)
            ordered_positions = torch.argsort(selected_target_merits, descending=True)
            if float(
                selected_target_merits[ordered_positions[0]]
                - selected_target_merits[ordered_positions[1]]
            ) >= ANCHOR_CONFIDENCE:
                if online_merit is None:
                    online_merit = scalarize_tensor(self._component_scores(
                        records, network=self.Q_local, grad=True
                    ))
                incumbent = int(selected_indices[ordered_positions[0]].item())
                alternatives = selected_indices[selected_indices != incumbent]
                terms.append(F.relu(
                    online_merit.new_tensor(ANCHOR_MARGIN)
                    - (
                        online_merit[incumbent]
                        - online_merit.index_select(0, alternatives).max()
                    )
                ))
                used += 1

        if not terms:
            return target_merit.new_tensor(0.0), 0
        return torch.stack(terms).mean(), used

    def remember(
        self,
        decision: ViabilityGraphDecision,
        *,
        reward: float,
        duration: int,
        next_snapshot,
        done: bool,
        outcome_already_observed: bool = False,
    ) -> FactorizedTransition:
        if not bool(done):
            boundary = int(decision.prepared_snapshot.decision_epoch) + int(duration)
            if next_snapshot is not None:
                boundary = max(boundary, int(next_snapshot.decision_epoch))
            if boundary >= self.config.episode_horizon_steps:
                raise ValueError("E115 horizon-reaching transition must be terminal")
        if outcome_already_observed:
            if done:
                next_prepared = None
            else:
                next_prepared, _ = self._admissible_prepared(next_snapshot)
                if not next_prepared.records:
                    raise NoCertifiedViableAction("E115 next frontier is empty")
        else:
            next_prepared = self.observe_outcome(
                decision, next_snapshot=next_snapshot, done=done
            )
        components = self._macro_components(duration)
        expected_raw_reward = scalarized_components(components) / 0.01
        if not math.isclose(float(reward), expected_raw_reward,
                            rel_tol=0.0, abs_tol=2.0e-4):
            raise RuntimeError("E115 macro scalar reward differs from components")
        transition = FactorizedTransition(
            chosen=decision.record,
            reward=float(reward),
            duration=int(duration),
            next_candidates=(
                () if next_prepared is None else tuple(next_prepared.records)
            ),
            done=bool(done),
            component_returns=components,
        )
        self.replay.add(transition)
        self.transition_count += 1
        return transition

    def _td_batch(self, transitions: Sequence[FactorizedTransition]):
        transitions = tuple(transitions)
        if not all(isinstance(item, FactorizedTransition) for item in transitions):
            raise TypeError("E115 replay contains a non-factorized transition")
        prediction = self._component_scores(
            tuple(item.chosen for item in transitions),
            network=self.Q_local,
            grad=True,
        )
        next_vectors = torch.zeros_like(prediction)
        anchor_terms = []
        anchor_pairs = 0
        nonterminal_records = tuple(
            record
            for transition in transitions
            if not transition.done
            for record in transition.next_candidates
        )
        with torch.no_grad():
            online_all = (
                self._component_scores(nonterminal_records, network=self.Q_local)
                if nonterminal_records else prediction.new_zeros((0, len(COMPONENT_NAMES)))
            )
            target_all = (
                self._target_components(nonterminal_records)
                if nonterminal_records else prediction.new_zeros((0, len(COMPONENT_NAMES)))
            )
            offset = 0
            for index, transition in enumerate(transitions):
                if transition.done:
                    continue
                count = len(transition.next_candidates)
                online = online_all[offset:offset + count]
                target = target_all[offset:offset + count]
                selected = select_hierarchical_index(
                    scalarize_tensor(online),
                    [record.mode_id for record in transition.next_candidates],
                    self.within_temperatures,
                    candidate_keys=tuple(
                        record.key for record in transition.next_candidates
                    ),
                ).selected_index
                next_vectors[index] = target[selected]
                offset += count

        # Re-score at most eight confident frontiers with gradients.  The
        # frozen target supplies a local trust region, not an oracle label.
        offset = 0
        for transition in transitions:
            if transition.done:
                continue
            count = len(transition.next_candidates)
            if count < 2:
                offset += count
                continue
            anchor, used = self._anchor_term(
                transition.next_candidates,
                target_all[offset:offset + count],
            )
            offset += count
            if used:
                anchor_terms.append(anchor)
                anchor_pairs += used
            if len(anchor_terms) >= MAX_ANCHOR_FRONTIERS_PER_UPDATE:
                break

        immediate = torch.as_tensor(
            [item.component_returns for item in transitions],
            dtype=prediction.dtype,
            device=self.device,
        )
        durations = torch.as_tensor(
            [item.duration for item in transitions],
            dtype=prediction.dtype,
            device=self.device,
        )
        dones = torch.as_tensor(
            [item.done for item in transitions],
            dtype=prediction.dtype,
            device=self.device,
        )
        discount = torch.pow(prediction.new_tensor(self.config.gamma), durations)
        target = immediate + (
            discount * (1.0 - dones)
        ).unsqueeze(-1) * next_vectors
        anchor = (
            torch.stack(anchor_terms).mean()
            if anchor_terms else prediction.new_tensor(0.0)
        )
        return prediction, target.detach(), anchor, anchor_pairs

    def learn(self) -> Optional[float]:
        if len(self.replay) < self.config.batch_size:
            return None
        if self.transition_count % self.config.update_every:
            return None
        transitions = self.replay.sample_mode_balanced(
            self.config.batch_size, self.rng
        )
        prediction, target, anchor, anchor_pairs = self._td_batch(transitions)
        component_losses = tuple(
            F.smooth_l1_loss(
                prediction[:, index], target[:, index], beta=self.config.huber_delta
            )
            for index in range(len(COMPONENT_NAMES))
        )
        loss = sum(
            weight * item
            for weight, item in zip(COMPONENT_LOSS_WEIGHTS, component_losses)
        ) + ANCHOR_LOSS_WEIGHT * anchor
        learning_rate = learning_rate_at(self.decision_count)
        for group in self.optimizer.param_groups:
            group["lr"] = learning_rate
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.Q_local.parameters(), self.config.grad_clip)
        self.optimizer.step()
        self.gradient_steps += 1
        if self.gradient_steps % self.config.target_update_every == 0:
            self.Q_target.load_state_dict(self.Q_local.state_dict())
            self.Q_target.requires_grad_(False).eval()
            self.target_updates += 1
        value = float(loss.detach().item())
        self.loss_history.append(value)
        self.component_loss_history.append(tuple(
            float(item.detach().item()) for item in component_losses
        ))
        anchor_value = float(anchor.detach().item())
        self.anchor_loss_history.append(anchor_value)
        self.anchor_pairs_used += int(anchor_pairs)
        return value

    def checkpoint_metadata(self) -> dict:
        value = super().checkpoint_metadata()
        value.update({
            "controller_architecture": CONTROLLER_ARCHITECTURE,
            "network_architecture": NETWORK_ARCHITECTURE,
            "backup_version": BACKUP_VERSION,
            "factorized_component_names": COMPONENT_NAMES,
            "factorized_scalar_weights": SCALAR_WEIGHTS,
            "component_loss_weights": COMPONENT_LOSS_WEIGHTS,
            "cost_heads_nonnegative": True,
            "conservative_update": CONSERVATIVE_UPDATE,
            "anchor_confidence": ANCHOR_CONFIDENCE,
            "anchor_margin": ANCHOR_MARGIN,
            "anchor_loss_weight": ANCHOR_LOSS_WEIGHT,
            "maximum_anchor_frontiers_per_update": (
                MAX_ANCHOR_FRONTIERS_PER_UPDATE
            ),
            "learning_rate_schedule_by_decisions": LR_PHASES,
        })
        return value

    def checkpoint_state(self, *, include_replay: bool = True) -> dict:
        value = super().checkpoint_state(include_replay=include_replay)
        value.update({
            "component_loss_history": tuple(self.component_loss_history),
            "anchor_loss_history": tuple(self.anchor_loss_history),
            "anchor_pairs_used": int(self.anchor_pairs_used),
        })
        return value

    def load_checkpoint_state(self, state: dict, *, resumable: bool = False) -> None:
        super().load_checkpoint_state(state, resumable=resumable)
        self.component_loss_history = list(state.get("component_loss_history", ()))
        self.anchor_loss_history = list(state.get("anchor_loss_history", ()))
        self.anchor_pairs_used = int(state.get("anchor_pairs_used", 0))

    @classmethod
    def from_checkpoint(
        cls,
        payload: dict,
        *,
        device: str | torch.device = "cpu",
        resumable: bool = False,
        seed: int = 0,
    ) -> "FactorizedConservativeAgent":
        expected = {
            "checkpoint_schema_version": (
                EPISODIC_VIABILITY_GRAPH_CHECKPOINT_SCHEMA_VERSION
            ),
            "checkpoint_family": EPISODIC_VIABILITY_GRAPH_CHECKPOINT_FAMILY,
            "controller_architecture": CONTROLLER_ARCHITECTURE,
            "network_architecture": NETWORK_ARCHITECTURE,
            "backup_version": BACKUP_VERSION,
            "time_context_contract": TIME_CONTEXT_CONTRACT,
            "queue_context_names": QUEUE_CONTEXT_NAMES,
            "queue_context_dim": QUEUE_CONTEXT_DIM,
            "queue_context_stored_in_replay": True,
            "factorized_component_names": COMPONENT_NAMES,
            "factorized_scalar_weights": SCALAR_WEIGHTS,
        }
        mismatches = {}
        for name, value in expected.items():
            observed = payload.get(name)
            if isinstance(value, tuple):
                equal = tuple(observed or ()) == value
            else:
                equal = observed == value
            if not equal:
                mismatches[name] = (payload.get(name), value)
        if mismatches:
            raise ValueError(f"incompatible E115 checkpoint: {mismatches!r}")
        if payload.get("future_schedule_visible_to_policy", False) is not False:
            raise ValueError("E115 checkpoint exposes the future schedule")
        agent = cls(
            config=EpisodicViabilityGraphConfig.from_dict(payload["config"]),
            seed=seed,
            device=device,
            epsilon=float(payload.get("agent_state", {}).get("epsilon", 0.0)),
        )
        agent.load_checkpoint_state(payload["agent_state"], resumable=resumable)
        return agent


__all__ = [
    "ANCHOR_CONFIDENCE",
    "ANCHOR_LOSS_WEIGHT",
    "ANCHOR_MARGIN",
    "BACKUP_VERSION",
    "COMPONENT_LOSS_WEIGHTS",
    "CONSERVATIVE_UPDATE",
    "CONTROLLER_ARCHITECTURE",
    "FactorizedConservativeAgent",
    "FactorizedQNetwork",
    "FactorizedTransition",
    "LR_PHASES",
    "MAX_ANCHOR_FRONTIERS_PER_UPDATE",
    "NETWORK_ARCHITECTURE",
    "learning_rate_at",
    "scalarize_tensor",
]
