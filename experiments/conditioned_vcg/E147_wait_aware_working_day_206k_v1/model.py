"""Warm-started five-head residual critic for the wait-aware day objective."""

from copy import deepcopy
from dataclasses import dataclass, fields, replace
import math

import torch
from torch import nn
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
from viability_graph_episodic_audit import EpisodicViabilityGraphConfig
from experiments.conditioned_vcg.E115_release_et_cost_vcg_8x8_179k_v1 import (
    model as factor,
)
from experiments.conditioned_vcg.E123_absolute_progress_8x8_187k_v1.model import (
    INPUT_DIM,
    ProgressObservedState,
)
from experiments.conditioned_vcg.E144_warm_start_mixture import arrival as old_arrival
from experiments.conditioned_vcg.E144_warm_start_mixture import run as warm
from .objective import (
    COMPONENT_NAMES,
    OBJECTIVE_WEIGHTS,
    SCALAR_WEIGHTS,
    nominal_components,
    scalarized_components,
)


WINDOW = 120
ARRIVAL_FEATURES = (
    "incoming_arrivals_last_120_primitive_steps_per_step",
    "primitive_steps_since_last_observed_incoming_arrival_or_episode_start",
    "working_day_phase_clipped_at_one",
)
NETWORK = "e147_arrival_observed_signed_five_head_residual_v1"


def scalarize_tensor(vectors):
    if not torch.is_tensor(vectors) or vectors.shape[-1] != len(COMPONENT_NAMES):
        raise ValueError("E147 component tensor has the wrong width")
    return torch.sum(vectors * vectors.new_tensor(SCALAR_WEIGHTS), dim=-1)


def observed_arrivals(env):
    now = int(env.time_steps)
    times = [
        int(block.arrival_step)
        for block in env.blocks
        if block.label not in env.initial_labels and int(block.arrival_step) <= now
    ]
    rate = sum(value > now - WINDOW for value in times) / float(WINDOW)
    age = now - max(times) if times else now
    day_steps = int(getattr(env, "_e147_day_steps", 5400))
    return (float(rate), float(age), min(now / float(day_steps), 1.0))


@dataclass(frozen=True)
class WorkdayState(ProgressObservedState):
    arrival_context: tuple[float, ...] = (0.0, 0.0, 0.0)

    def __post_init__(self):
        super().__post_init__()
        if len(self.arrival_context) != len(ARRIVAL_FEATURES):
            raise ValueError("E147 arrival context width differs")
        if not all(math.isfinite(value) and value >= 0.0 for value in self.arrival_context):
            raise ValueError("E147 arrival context must be finite and nonnegative")


class WaitNetwork(old_arrival.ArrivalNetwork):
    def __init__(self, config):
        super().__init__(config)
        self.arrival_projection = nn.Linear(
            len(ARRIVAL_FEATURES), config.action_embedding_dim, bias=False
        )
        self.admission_wait_head = nn.Linear(config.head_hidden_dim, 1)
        with torch.no_grad():
            self.arrival_projection.weight.zero_()
            self.admission_wait_head.weight.zero_()
            self.admission_wait_head.bias.zero_()

    def context_tensor(self, states):
        states = tuple(states)
        if not all(isinstance(state, WorkdayState) for state in states):
            raise TypeError("E147 replay lacks working-day arrival observations")
        parameter = next(self.parameters())
        return torch.as_tensor(
            [
                state.queue_context + state.progress_context + state.arrival_context
                for state in states
            ],
            dtype=parameter.dtype,
            device=parameter.device,
        )

    def score_component_embeddings(
        self,
        current_embedding,
        successor_embedding,
        action_features,
        queue_context,
    ):
        parameter = next(self.parameters())
        action = torch.as_tensor(
            action_features, dtype=parameter.dtype, device=parameter.device
        )
        if action.ndim == 1:
            action = action.unsqueeze(0)
        count = current_embedding.shape[0]
        expected = INPUT_DIM + len(ARRIVAL_FEATURES)
        if action.shape != (count, ACTION_FEATURE_DIM) or queue_context.shape != (count, expected):
            raise ValueError("E147 network input dimensions differ")
        context = self.context_encoder(queue_context[:, :INPUT_DIM])
        context = context + self.arrival_projection(queue_context[:, INPUT_DIM:])
        latent = self.consequence_trunk(
            torch.cat(
                (
                    current_embedding,
                    successor_embedding,
                    successor_embedding - current_embedding,
                    self.action_encoder(action),
                    context,
                ),
                dim=-1,
            )
        )
        return torch.cat(
            (
                self.service_head(latent),
                self.wait_head(latent),
                self.admission_wait_head(latent),
                self.step_head(latent),
                self.rehandle_head(latent),
            ),
            dim=-1,
        )


@dataclass(frozen=True)
class WaitCostRecord(ViabilityGraphCandidateRecord):
    nominal_components: tuple[float, ...]
    nominal_duration: int

    def __post_init__(self):
        super().__post_init__()
        if len(self.nominal_components) != len(COMPONENT_NAMES):
            raise ValueError("E147 nominal component width differs")
        if not all(math.isfinite(value) and value >= 0.0 for value in self.nominal_components):
            raise ValueError("E147 nominal components must be finite and nonnegative")
        if int(self.nominal_duration) < 1:
            raise ValueError("E147 nominal duration must be positive")


@dataclass(frozen=True)
class WaitTransition(ViabilityGraphTransition):
    component_returns: tuple[float, ...]

    def __post_init__(self):
        super().__post_init__()
        values = tuple(float(value) for value in self.component_returns)
        if len(values) != len(COMPONENT_NAMES):
            raise ValueError("E147 transition component width differs")
        scalarized_components(values)
        if not math.isclose(
            float(self.reward) * 0.01,
            scalarized_components(values),
            rel_tol=0.0,
            abs_tol=2.0e-6,
        ):
            raise ValueError("E147 transition reward and components disagree")

    def numeric_copy(self):
        return WaitTransition(
            chosen=self.chosen,
            reward=float(self.reward),
            duration=int(self.duration),
            next_candidates=tuple(self.next_candidates),
            done=bool(self.done),
            component_returns=tuple(float(value) for value in self.component_returns),
        )


def known_tensor(records, like):
    if not all(isinstance(record, WaitCostRecord) for record in records):
        raise TypeError("E147 records lack five-component nominal costs")
    return torch.as_tensor(
        [record.nominal_components for record in records],
        dtype=like.dtype,
        device=like.device,
    )


class WaitAgent(old_arrival.ArrivalAgent):
    """One shared selected successor backs all five residual cost heads."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.seed)
            self.Q_local = WaitNetwork(self.config).to(self.device)
        self.Q_target = deepcopy(self.Q_local).requires_grad_(False).eval()
        self.optimizer = torch.optim.Adam(self.Q_local.parameters(), lr=5e-5)

    def _admissible_prepared(self, snapshot):
        prepared, forced = super()._admissible_prepared(snapshot)
        arrival = observed_arrivals(self.context_environment)
        records = []
        for record, source_index in zip(prepared.records, prepared.source_indices):
            current = WorkdayState(
                **{
                    item.name: getattr(record.current_state, item.name)
                    for item in fields(ProgressObservedState)
                },
                arrival_context=arrival,
            )
            prediction = nominal_components(
                self.context_environment,
                snapshot.candidates[source_index],
                self.config.gamma,
            )
            records.append(
                WaitCostRecord(
                    **{
                        item.name: getattr(record, item.name)
                        for item in fields(ViabilityGraphCandidateRecord)
                        if item.name != "current_state"
                    },
                    current_state=current,
                    nominal_components=tuple(prediction["components"]),
                    nominal_duration=int(prediction["duration"]),
                )
            )
        return replace(prepared, records=tuple(records)), forced

    def _component_scores(self, records, *, network=None, grad=False):
        records = tuple(records)
        learned = factor.FactorizedConservativeAgent._component_scores(
            self, records, network=network, grad=grad
        )
        return learned + known_tensor(records, learned)

    def _score_records(self, records, *, network=None, grad=False):
        return scalarize_tensor(
            self._component_scores(records, network=network, grad=grad)
        )

    def _macro_components(self, duration):
        records = self.factorized_environment.factorized_component_records
        start = self.component_cursor
        stop = len(records)
        if stop - start != int(duration):
            raise RuntimeError("E147 primitive component count differs from macro duration")
        totals = [0.0] * len(COMPONENT_NAMES)
        for offset, values in enumerate(records[start:stop]):
            if len(values) != len(COMPONENT_NAMES):
                raise RuntimeError("E147 environment component width differs")
            discount = float(self.config.gamma) ** offset
            for index, value in enumerate(values):
                totals[index] += discount * float(value)
        self.component_cursor = stop
        return tuple(totals)

    def _anchor_term(self, records, target_vectors):
        target_merit = scalarize_tensor(target_vectors)
        mode_ids = torch.as_tensor(
            [record.mode_id for record in records], dtype=torch.long, device=self.device
        )
        target_modes, live_modes = regularized_mode_values(
            target_merit, mode_ids, self.within_temperatures
        )
        selected_position = int(torch.argmax(target_modes).item())
        selected_mode = int(live_modes[selected_position].item())
        terms = []
        used = 0
        online_merit = None
        if len(target_modes) >= 2:
            ordered = torch.sort(target_modes, descending=True).values
            if float(ordered[0] - ordered[1]) >= factor.ANCHOR_CONFIDENCE:
                online_merit = scalarize_tensor(
                    self._component_scores(records, network=self.Q_local, grad=True)
                )
                online_modes, online_live = regularized_mode_values(
                    online_merit, mode_ids, self.within_temperatures
                )
                position = int((online_live == selected_mode).nonzero()[0].item())
                alternatives = torch.cat((online_modes[:position], online_modes[position + 1 :]))
                terms.append(
                    F.relu(
                        online_merit.new_tensor(factor.ANCHOR_MARGIN)
                        - (online_modes[position] - alternatives.max())
                    )
                )
                used += 1
        selected_indices = torch.nonzero(mode_ids == selected_mode, as_tuple=False).flatten()
        if len(selected_indices) >= 2:
            selected_target = target_merit.index_select(0, selected_indices)
            order = torch.argsort(selected_target, descending=True)
            if float(selected_target[order[0]] - selected_target[order[1]]) >= factor.ANCHOR_CONFIDENCE:
                if online_merit is None:
                    online_merit = scalarize_tensor(
                        self._component_scores(records, network=self.Q_local, grad=True)
                    )
                incumbent = int(selected_indices[order[0]].item())
                alternatives = selected_indices[selected_indices != incumbent]
                terms.append(
                    F.relu(
                        online_merit.new_tensor(factor.ANCHOR_MARGIN)
                        - (
                            online_merit[incumbent]
                            - online_merit.index_select(0, alternatives).max()
                        )
                    )
                )
                used += 1
        if not terms:
            return target_merit.new_tensor(0.0), 0
        return torch.stack(terms).mean(), used

    def remember(
        self,
        decision: ViabilityGraphDecision,
        *,
        reward,
        duration,
        next_snapshot,
        done,
        outcome_already_observed=False,
    ):
        if not bool(done):
            boundary = int(decision.prepared_snapshot.decision_epoch) + int(duration)
            if next_snapshot is not None:
                boundary = max(boundary, int(next_snapshot.decision_epoch))
            if boundary >= self.config.episode_horizon_steps:
                raise ValueError("E147 horizon-reaching transition must be terminal")
        if outcome_already_observed:
            next_prepared = None if done else self._admissible_prepared(next_snapshot)[0]
        else:
            next_prepared = self.observe_outcome(
                decision, next_snapshot=next_snapshot, done=bool(done)
            )
        components = self._macro_components(duration)
        expected_reward = scalarized_components(components) / 0.01
        if not math.isclose(float(reward), expected_reward, abs_tol=2.0e-4):
            raise RuntimeError("E147 macro reward differs from five-component record")
        transition = WaitTransition(
            chosen=decision.record,
            reward=float(reward),
            duration=int(duration),
            next_candidates=() if next_prepared is None else tuple(next_prepared.records),
            done=bool(done),
            component_returns=components,
        )
        self.replay.add(transition)
        self.transition_count += 1
        return transition

    def _td_batch(self, transitions):
        transitions = tuple(transitions)
        if not all(isinstance(item, WaitTransition) for item in transitions):
            raise TypeError("E147 replay contains another transition family")
        prediction = self._component_scores(
            tuple(item.chosen for item in transitions), network=self.Q_local, grad=True
        )
        next_vectors = torch.zeros_like(prediction)
        records = tuple(
            record
            for transition in transitions
            if not transition.done
            for record in transition.next_candidates
        )
        anchors = []
        anchor_pairs = 0
        with torch.no_grad():
            online_all = (
                self._component_scores(records, network=self.Q_local)
                if records
                else prediction.new_zeros((0, len(COMPONENT_NAMES)))
            )
            target_all = (
                self._target_components(records)
                if records
                else prediction.new_zeros((0, len(COMPONENT_NAMES)))
            )
            offset = 0
            for index, transition in enumerate(transitions):
                if transition.done:
                    continue
                count = len(transition.next_candidates)
                online = online_all[offset : offset + count]
                target = target_all[offset : offset + count]
                selected = select_hierarchical_index(
                    scalarize_tensor(online),
                    [record.mode_id for record in transition.next_candidates],
                    self.within_temperatures,
                    candidate_keys=tuple(record.key for record in transition.next_candidates),
                ).selected_index
                next_vectors[index] = target[selected]
                offset += count
        offset = 0
        for transition in transitions:
            if transition.done:
                continue
            count = len(transition.next_candidates)
            if count >= 2:
                anchor, used = self._anchor_term(
                    transition.next_candidates, target_all[offset : offset + count]
                )
                if used:
                    anchors.append(anchor)
                    anchor_pairs += used
            offset += count
            if len(anchors) >= factor.MAX_ANCHOR_FRONTIERS_PER_UPDATE:
                break
        immediate = torch.as_tensor(
            [item.component_returns for item in transitions],
            dtype=prediction.dtype,
            device=self.device,
        )
        durations = torch.as_tensor(
            [item.duration for item in transitions], dtype=prediction.dtype, device=self.device
        )
        dones = torch.as_tensor(
            [item.done for item in transitions], dtype=prediction.dtype, device=self.device
        )
        discount = torch.pow(prediction.new_tensor(self.config.gamma), durations)
        target = immediate + (discount * (1.0 - dones)).unsqueeze(-1) * next_vectors
        anchor = torch.stack(anchors).mean() if anchors else prediction.new_tensor(0.0)
        if len(self.td_diagnostics) < 20 or self.gradient_steps % 200 == 0:
            learned = prediction - known_tensor(
                [item.chosen for item in transitions], prediction
            )
            self.td_diagnostics.append(
                {
                    "gradient_step": self.gradient_steps,
                    "max_abs_total_prediction": float(prediction.detach().abs().max()),
                    "max_abs_learned_prediction": float(learned.detach().abs().max()),
                    "max_abs_target": float(target.detach().abs().max()),
                    "max_abs_td_error": float((target - prediction.detach()).abs().max()),
                }
            )
        return prediction, target.detach(), anchor, anchor_pairs

    def learn(self):
        if len(self.replay) < self.config.batch_size:
            return None
        if self.transition_count % self.config.update_every:
            return None
        transitions = self.replay.sample_mode_balanced(self.config.batch_size, self.rng)
        prediction, target, anchor, anchor_pairs = self._td_batch(transitions)
        losses = tuple(
            F.smooth_l1_loss(
                prediction[:, index], target[:, index], beta=self.config.huber_delta
            )
            for index in range(len(COMPONENT_NAMES))
        )
        loss = sum(weight * value for weight, value in zip(OBJECTIVE_WEIGHTS, losses))
        loss = loss + factor.ANCHOR_LOSS_WEIGHT * anchor
        learning_rate = factor.learning_rate_at(self.decision_count)
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
        self.component_loss_history.append(
            tuple(float(item.detach().item()) for item in losses)
        )
        self.anchor_loss_history.append(float(anchor.detach().item()))
        self.anchor_pairs_used += int(anchor_pairs)
        return value

    def checkpoint_metadata(self):
        value = super().checkpoint_metadata()
        value.update(
            network_architecture=NETWORK,
            factorized_component_names=COMPONENT_NAMES,
            factorized_scalar_weights=SCALAR_WEIGHTS,
            component_loss_weights=OBJECTIVE_WEIGHTS,
            e147_wait_aware_working_day=True,
            e144_arrival_v2=False,
            arrival_context_names=ARRIVAL_FEATURES,
            arrival_window_steps=WINDOW,
            arrival_context_in_replay=True,
            future_release_schedule_visible=False,
            regime_identifier_visible=False,
            workload_target_visible=False,
            same_selected_successor_for_all_heads=True,
        )
        return value

    @classmethod
    def from_checkpoint(cls, payload, *, device="cpu", resumable=False, seed=0):
        expected = {
            "network_architecture": NETWORK,
            "factorized_component_names": COMPONENT_NAMES,
            "factorized_scalar_weights": SCALAR_WEIGHTS,
            "e147_wait_aware_working_day": True,
            "arrival_context_names": ARRIVAL_FEATURES,
            "future_release_schedule_visible": False,
            "regime_identifier_visible": False,
            "same_selected_successor_for_all_heads": True,
        }
        mismatches = {}
        for key, value in expected.items():
            observed = payload.get(key)
            equal = tuple(observed or ()) == value if isinstance(value, tuple) else observed == value
            if not equal:
                mismatches[key] = (observed, value)
        if mismatches:
            raise ValueError(f"E147 checkpoint differs: {mismatches!r}")
        agent = cls(
            config=EpisodicViabilityGraphConfig.from_dict(payload["config"]),
            seed=seed,
            device=device,
            epsilon=float(payload["agent_state"].get("epsilon", 0.0)),
            variant="residual_q",
            arm="A_outflow_bounded",
            absolute_progress=False,
        )
        agent.load_checkpoint_state(payload["agent_state"], resumable=resumable)
        return agent


def initialize(source_payload, *, device="cpu", horizon=16200):
    config = replace(
        EpisodicViabilityGraphConfig.from_dict(source_payload["config"]),
        episode_horizon_steps=int(horizon),
    )
    agent = WaitAgent(
        config=config,
        seed=0,
        device=device,
        epsilon=0.9,
        variant="residual_q",
        arm="A_outflow_bounded",
        absolute_progress=False,
    )
    expected_missing = {
        "arrival_projection.weight",
        "admission_wait_head.weight",
        "admission_wait_head.bias",
    }
    for network, key in ((agent.Q_local, "Q_local"), (agent.Q_target, "Q_target")):
        missing, unexpected = network.load_state_dict(
            source_payload["agent_state"][key], strict=False
        )
        if set(missing) != expected_missing or unexpected:
            raise ValueError(
                f"E147 source migration differs: missing={missing}, unexpected={unexpected}"
            )
        if bool(network.arrival_projection.weight.count_nonzero()):
            raise ValueError("E147 arrival connections must initialize at zero")
        if bool(network.admission_wait_head.weight.count_nonzero()) or bool(
            network.admission_wait_head.bias.count_nonzero()
        ):
            raise ValueError("E147 admission-wait head must initialize at zero")
    return agent

