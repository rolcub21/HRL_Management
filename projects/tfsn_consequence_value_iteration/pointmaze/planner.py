"""Current PointMaze TF-SN road-map runtime used for alignment and evaluation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from projects.tfsn_consequence_value_iteration.shared.factorized import (
    FactorizedConsequenceEnsemble,
)
from projects.tfsn_consequence_value_iteration.shared.reliability import (
    posterior_mean,
)
from projects.tfsn_consequence_value_iteration.shared.reach import EmpiricalReachQuery

from .core import load_json, protocol, sha256
from .passages import PassageGeometry
from .prepare_bridges import MobilityModel
from .roadmap import (
    RecordedRoadmap,
    alias_kernel,
    bounded_search_values,
    persistent_values,
)
from projects.tfsn_consequence_value_iteration.shared.settings import (
    CONSUMPTIVE_LOOKAHEAD_BEAM_WIDTH,
    CONSUMPTIVE_LOOKAHEAD_DEPTH,
    finite_horizon,
)
from projects.tfsn_consequence_value_iteration.shared.frozen_hub_method import (
    posterior_probability,
)


class ConsumedSearchField:
    """Remove an imagined passage using the deployed search observation model."""

    def __init__(self, base, passage):
        self.base = base
        self.points = np.asarray(passage, np.float32).reshape(-1, 2)
        self.tree = cKDTree(self.points)

    def search(self, points):
        points = np.asarray(points, np.float32)
        shape = points.shape[:-1]
        flat = points.reshape(-1, 2)
        residual = np.asarray(self.base.search(flat), np.float32)
        distance = self.tree.query(flat, k=1, workers=-1)[0]
        newly_unsearched = 1.0 - np.exp(
            -np.square(distance) / (2.0 * self.base.sigma**2)
        ).astype(np.float32)
        return (residual * newly_unsearched).reshape(shape)


class PointMazeRoadmapPlanner:
    """R22 road map plus R28 bounded acquisition and R14 aliasing."""

    def __init__(
        self,
        graph,
        landmarks,
        targets,
        reach_offsets,
        member_weight,
        consequence,
        mobility,
        passages,
        sigma,
        horizon,
        seed,
        advance_weighting=False,
        consumptive_lookahead=False,
    ):
        self.graph = graph
        self.landmarks = np.asarray(landmarks, np.int32)
        self.targets = np.asarray(targets, np.int32)
        self.member_weight = np.asarray(member_weight, np.float32)
        self.weight = self.member_weight.min(axis=2)
        self.consequence = consequence
        self.mobility = mobility
        self.passages = passages
        self.sigma = float(sigma)
        self.horizon = int(horizon)
        self.seed = int(seed)
        self.advance_weighting = bool(advance_weighting)
        self.consumptive_lookahead = bool(consumptive_lookahead)
        self.reach_query = EmpiricalReachQuery(
            graph.nodes[self.landmarks, :2], reach_offsets
        )
        order = np.argsort(graph.source_index, kind="stable")
        counts = np.bincount(graph.source_index, minlength=graph.node_count)
        self.outgoing_order = order.astype(np.int32)
        self.outgoing_offsets = np.concatenate(
            ([0], np.cumsum(counts, dtype=np.int64))
        )
        self.values = None
        self.node_field = None
        self.recorded_gain = None
        self.bridge_gain = None
        self.acquisition_active = None
        self._field_cache_token = None
        self.destination_arrival_correction = False
        self.reset_task_memory()

    def enable_destination_arrival_correction(self):
        """Install the frozen R74 link outcome without changing preparation."""
        self.destination_arrival_correction = True
        self.reset_task_memory()

    def reset_task_memory(self):
        """Reset reward-free online consequence evidence for a new task."""
        self.online_starts = []
        self.online_commands = []
        self.online_endpoints = []
        self.online_movement = []
        self.online_endpoint_mobility = []
        self.pending_online_record = None
        self._last_online_command_support = None
        self._last_online_endpoint_support = None
        self.arrival_starts = []
        self.arrival_commands = []
        self.arrival_destinations = []
        self.arrival_outcomes = []
        self.pending_arrival = None
        self._arrival_bridge_support = None
        self._arrival_bridge_outcome_mass = None
        self._arrival_bridge_processed = 0

    def begin_episode(self):
        """Keep task memory but do not connect evidence across a physical reset."""
        self.pending_online_record = None
        self.pending_arrival = None

    def observe_transition(self, start_state, command, endpoint_state):
        """Record movement now and endpoint mobility on the next command."""
        start = np.asarray(start_state, np.float32).reshape(4)
        command = np.asarray(command, np.float32).reshape(2)
        endpoint = np.asarray(endpoint_state, np.float32).reshape(4)
        if getattr(self, "destination_arrival_correction", False):
            if self.pending_arrival is None:
                raise RuntimeError("selected link destination was not recorded")
            pending_start, pending_command, destination = self.pending_arrival
            if not np.array_equal(pending_start, start) or not np.array_equal(
                pending_command, command
            ):
                raise RuntimeError("executed command differs from selected link")
            outcome = float(
                np.linalg.norm(endpoint[:2] - destination) <= self.sigma
            )
            self.arrival_starts.append(start.copy())
            self.arrival_commands.append(command.copy())
            self.arrival_destinations.append(destination.copy())
            self.arrival_outcomes.append(outcome)
            self.pending_arrival = None
            return
        movement = float(
            1.0 - alias_kernel(
                np.linalg.norm(endpoint[:2] - start[:2]), self.sigma
            )
        )
        if self.pending_online_record is not None:
            previous = int(self.pending_online_record)
            if np.allclose(
                self.online_endpoints[previous], start, rtol=0.0, atol=1e-5
            ):
                self.online_endpoint_mobility[previous] = movement
        self.online_starts.append(start.copy())
        self.online_commands.append(command.copy())
        self.online_endpoints.append(endpoint.copy())
        self.online_movement.append(movement)
        self.online_endpoint_mobility.append(np.nan)
        self.pending_online_record = len(self.online_starts) - 1

    def _arrival_mass(self, state, commands, destinations):
        commands = np.asarray(commands, np.float32).reshape(-1, 2)
        destinations = np.asarray(destinations, np.float32).reshape(-1, 2)
        if not self.arrival_starts:
            zeros = np.zeros(len(commands), np.float32)
            return zeros, zeros
        observed_start = np.asarray(self.arrival_starts, np.float32)[:, :2]
        observed_command = np.asarray(self.arrival_commands, np.float32)
        observed_destination = np.asarray(
            self.arrival_destinations, np.float32
        )
        outcome = np.asarray(self.arrival_outcomes, np.float32)
        source_distance = np.square(
            (observed_start - np.asarray(state, np.float32)[None, :2])
            / self.sigma
        ).sum(axis=1, dtype=np.float64)
        command_distance = np.square(
            (observed_command[:, None] - commands[None]) / self.sigma
        ).sum(axis=2, dtype=np.float64)
        destination_distance = np.square(
            (observed_destination[:, None] - destinations[None]) / self.sigma
        ).sum(axis=2, dtype=np.float64)
        similarity = np.exp(-0.5 * (
            source_distance[:, None] + command_distance + destination_distance
        ))
        return (
            similarity.sum(axis=0).astype(np.float32),
            (similarity * outcome[:, None]).sum(axis=0).astype(np.float32),
        )

    def corrected_bridge_weight(self):
        """R71 link evidence generalized to all modeled graph bridges."""
        if not getattr(self, "destination_arrival_correction", False) or not self.arrival_starts:
            return self.weight
        if self._arrival_bridge_support is None:
            self._arrival_bridge_support = np.zeros(
                self.weight.shape, np.float32
            )
            self._arrival_bridge_outcome_mass = np.zeros(
                self.weight.shape, np.float32
            )
        node_xy = self.graph.nodes[:, :2].astype(np.float64)
        target_xy = node_xy[self.targets]
        denominator = float(self.sigma) ** 2
        for index in range(
            self._arrival_bridge_processed, len(self.arrival_starts)
        ):
            start = self.arrival_starts[index]
            command = self.arrival_commands[index]
            destination = self.arrival_destinations[index]
            outcome = self.arrival_outcomes[index]
            source_distance = np.square(
                node_xy - np.asarray(start, np.float64)[None, :2]
            ).sum(axis=1) / denominator
            command_distance = np.square(
                target_xy - np.asarray(command, np.float64)[None, None]
            ).sum(axis=2) / denominator
            destination_distance = np.square(
                target_xy - np.asarray(destination, np.float64)[None, None]
            ).sum(axis=2) / denominator
            similarity = np.exp(-0.5 * (
                source_distance[:, None]
                + command_distance
                + destination_distance
            )).astype(np.float32)
            self._arrival_bridge_support += similarity
            if float(outcome) != 0.0:
                self._arrival_bridge_outcome_mass += (
                    similarity * float(outcome)
                )
        self._arrival_bridge_processed = len(self.arrival_starts)
        return posterior_probability(
            self.weight,
            self._arrival_bridge_support,
            self._arrival_bridge_outcome_mass,
        ).astype(np.float32)

    def _online_state_similarity(self, states):
        states = np.asarray(states, np.float32).reshape(-1, 4)
        if not self.online_starts:
            return np.zeros((len(states), 0), np.float32)
        observed = np.asarray(self.online_starts, np.float32)
        difference = (
            states[:, None] - observed[None]
        ) / self.graph.state_scale[None, None]
        return np.exp(
            -0.5 * np.square(difference).sum(axis=2)
        ).astype(np.float32)

    def _online_command_similarity(self, states, commands, known):
        states = np.asarray(states, np.float32).reshape(-1, 4)
        commands = np.asarray(commands, np.float32).reshape(-1, 2)
        observed_state = np.asarray(self.online_starts, np.float32)[known]
        observed_command = np.asarray(self.online_commands, np.float32)[known]
        state_difference = (
            states[:, None] - observed_state[None]
        ) / self.graph.state_scale[None, None]
        command_difference = (
            commands[:, None] - observed_command[None]
        ) / self.sigma
        return np.exp(-0.5 * (
            np.square(state_difference).sum(axis=2)
            + np.square(command_difference).sum(axis=2)
        )).astype(np.float32)

    def _correct_endpoint_mobility(self, endpoints, prior):
        endpoints = np.asarray(endpoints, np.float32)
        prior = np.asarray(prior, np.float32)
        if not self.online_starts:
            return prior, np.zeros(prior.shape, np.float32)
        flat = endpoints.reshape(-1, 4)
        similarity = self._online_state_similarity(flat)
        corrected, support = posterior_mean(
            prior.reshape(-1),
            similarity,
            np.asarray(self.online_movement, np.float32),
        )
        return corrected.reshape(prior.shape), support.reshape(prior.shape)

    def _correct_command_viability(self, states, commands, prior):
        endpoint_mobile = np.asarray(self.online_endpoint_mobility, np.float32)
        known = np.flatnonzero(np.isfinite(endpoint_mobile))
        if not len(known):
            return prior, np.zeros(len(prior), np.float32)
        similarity = self._online_command_similarity(states, commands, known)
        outcome = (
            np.asarray(self.online_movement, np.float32)[known]
            * endpoint_mobile[known]
        )
        return posterior_mean(prior, similarity, outcome)

    def _consumed_tail(self, fields, passage, remaining, target_index):
        if remaining <= 1:
            return 0.0
        consumed = ConsumedSearchField(fields, passage)
        recorded_gain, bridge_gain = self.passages.graph_gains(consumed)
        values = bounded_search_values(
            self.graph,
            self.targets,
            self.weight,
            recorded_gain,
            bridge_gain,
            remaining - 1,
        )
        return float(values[remaining - 1, int(target_index)])

    def _set_field(self, fields):
        cache_token = getattr(fields, "planner_cache_token", None)
        if cache_token is not None and cache_token == self._field_cache_token:
            return
        if fields.contacted:
            self.acquisition_active = False
            self.node_field = fields.reward_rate(self.graph.nodes[:, :2])
            self.values = persistent_values(
                self.graph,
                self.targets,
                self.weight,
                self.node_field,
                self.horizon,
            )
            self.recorded_gain = self.bridge_gain = None
        else:
            self.acquisition_active = True
            self.node_field = fields.search(self.graph.nodes[:, :2])
            self.recorded_gain, self.bridge_gain = self.passages.graph_gains(fields)
            self.values = bounded_search_values(
                self.graph,
                self.targets,
                self.weight,
                self.recorded_gain,
                self.bridge_gain,
                self.horizon,
            )
        self._field_cache_token = cache_token

    def _member_connection(self, state, commands):
        state = np.asarray(state, np.float32).reshape(4)
        commands = np.asarray(commands, np.float32).reshape(-1, 2)
        source = np.repeat(state[None], len(commands), axis=0)
        rows = []
        command_support = []
        endpoint_support = []
        for member in range(len(self.consequence.spatial_members)):
            weights, endpoint_state = (
                self.consequence.component_mean_endpoint_outcomes(
                    source, commands, member
                )
            )
            distance = np.linalg.norm(
                endpoint_state[..., :2] - commands[:, None], axis=2
            )
            landing = np.exp(
                -np.square(distance) / (2.0 * self.sigma**2)
            )
            endpoint_mobility_prior = self.mobility.predict(
                endpoint_state.reshape(-1, 4)
            ).reshape(endpoint_state.shape[:2])
            endpoint_mobility, local_endpoint_support = (
                self._correct_endpoint_mobility(
                    endpoint_state, endpoint_mobility_prior
                )
            )
            base = (weights * landing * endpoint_mobility).sum(axis=1)
            movement_prior = 1.0 - self.consequence.stall_probability(
                source, commands, self.sigma, member
            )
            viability, local_command_support = self._correct_command_viability(
                source, commands, movement_prior
            )
            rows.append(np.minimum(base, viability))
            command_support.append(local_command_support)
            endpoint_support.append(
                (weights * local_endpoint_support).sum(axis=1)
            )
        self._last_online_command_support = np.stack(command_support, axis=1)
        self._last_online_endpoint_support = np.stack(endpoint_support, axis=1)
        return np.stack(rows, axis=1).astype(np.float32)

    def _candidate_q(
        self, state, commands, target_index, remaining, execution_weight, fields
    ):
        commands = np.asarray(commands, np.float32).reshape(-1, 2)
        target_index = np.asarray(target_index, np.int32).reshape(-1)
        execution_weight = np.asarray(execution_weight, np.float32).reshape(-1)
        member_connection = self._member_connection(state, commands)
        previous = self.values[remaining - 1]
        source_index = int(self.graph.project(np.asarray(state)[None])[0][0])
        failure = float(previous[source_index])
        if self.acquisition_active:
            gain = self.passages.live_gains(state[:2], commands, fields)
            success = gain + (1.0 - gain) * previous[target_index]
        else:
            success = (
                fields.reward_rate(commands) + previous[target_index]
            )
        probability = member_connection * execution_weight[:, None]
        if getattr(self, "destination_arrival_correction", False):
            destinations = self.graph.nodes[target_index, :2]
            support, outcome_mass = self._arrival_mass(
                state, commands, destinations
            )
            probability = posterior_probability(
                probability, support[:, None], outcome_mass[:, None]
            ).astype(np.float32)
        member_q = (
            probability * success[:, None]
            + (1.0 - probability) * failure
        )
        return member_q.min(axis=1), member_connection.min(axis=1), member_q

    def _greedy_outgoing(self, node_index, remaining):
        node_index = np.asarray(node_index, np.int32).reshape(-1)
        previous = self.values[remaining - 1]
        model_target = self.targets[node_index]
        if self.acquisition_active:
            success = self.bridge_gain[node_index] + (
                1.0 - self.bridge_gain[node_index]
            ) * previous[model_target]
        else:
            success = self.node_field[model_target] + previous[model_target]
        model_distance = np.linalg.norm(
            self.graph.nodes[model_target, :2]
            - self.graph.nodes[node_index, None, :2],
            axis=2,
        )
        model_advance = (
            1.0 - alias_kernel(model_distance, self.sigma)
            if self.advance_weighting
            else np.ones_like(model_distance, np.float32)
        )
        model_execution = self.weight[node_index] * model_advance
        model_q = (
            model_execution * success
            + (1.0 - model_execution) * previous[node_index, None]
        )
        choice = np.argmax(model_q, axis=1)
        row = np.arange(len(node_index))
        target = model_target[row, choice].astype(np.int32)
        command = self.graph.nodes[target, :2].copy()
        score = model_q[row, choice].astype(np.float32)
        advance = model_advance[row, choice].astype(np.float32)
        reliability = self.weight[node_index, choice].astype(np.float32)
        kind = np.full(len(node_index), "alias_next_model_bridge", dtype=object)
        for output, node in enumerate(node_index):
            begin = self.outgoing_offsets[node]
            stop = self.outgoing_offsets[node + 1]
            if begin == stop:
                continue
            edges = self.outgoing_order[begin:stop]
            endpoints = self.graph.endpoint_index[edges]
            if self.acquisition_active:
                edge_success = self.recorded_gain[edges] + (
                    1.0 - self.recorded_gain[edges]
                ) * previous[endpoints]
            else:
                edge_success = self.node_field[endpoints] + previous[endpoints]
            edge_distance = np.linalg.norm(
                self.graph.commands[edges]
                - self.graph.nodes[node, :2],
                axis=1,
            )
            edge_advance = (
                1.0 - alias_kernel(edge_distance, self.sigma)
                if self.advance_weighting
                else np.ones_like(edge_distance, np.float32)
            )
            graph_reliability = self.graph.edge_reliability[edges]
            edge_execution = graph_reliability * edge_advance
            edge_q = (
                edge_execution * edge_success
                + (1.0 - edge_execution) * previous[node]
            )
            local = int(np.argmax(edge_q))
            if edge_q[local] <= score[output]:
                continue
            edge = int(edges[local])
            target[output] = self.graph.endpoint_index[edge]
            command[output] = self.graph.commands[edge]
            score[output] = edge_q[local]
            advance[output] = edge_advance[local]
            reliability[output] = graph_reliability[local]
            kind[output] = "alias_next_recorded_edge"
        return command, target, score, advance, reliability, kind

    def choose(self, state, command_index, fields):
        remaining = self.horizon - int(command_index)
        if remaining <= 0:
            raise ValueError("command index exceeds the physical episode horizon")
        state = np.asarray(state, np.float32).reshape(4)
        self._set_field(fields)
        distance, local = self.reach_query.query(state[:2], workers=-1)
        bridge_target = self.landmarks[np.asarray(local, np.int64)]
        bridge_command = self.graph.nodes[bridge_target, :2]
        association = alias_kernel(distance, self.sigma)
        bridge_q, bridge_connection, _ = self._candidate_q(
            state,
            bridge_command,
            bridge_target,
            remaining,
            1.0 - association,
            fields,
        )
        (
            advance_command,
            advance_target,
            offline,
            advance_factor,
            advance_reliability,
            advance_kind,
        ) = self._greedy_outgoing(bridge_target, remaining)
        advance_q, advance_connection, _ = self._candidate_q(
            state,
            advance_command,
            advance_target,
            remaining,
            association * advance_factor * advance_reliability,
            fields,
        )
        commands = [bridge_command, advance_command]
        scores = [bridge_q, advance_q]
        connections = [bridge_connection, advance_connection]
        kinds = [
            np.full(len(bridge_command), "model_bridge", dtype=object),
            advance_kind,
        ]
        associations = [np.zeros(len(bridge_command), np.float32), association]
        offline_rows = [np.zeros(len(bridge_command), np.float32), offline]
        advance_rows = [
            np.ones(len(bridge_command), np.float32), advance_factor
        ]
        targets = [bridge_target, advance_target]
        execution_rows = [1.0 - association, association * advance_factor]
        execution_rows[-1] = execution_rows[-1] * advance_reliability

        # Positive online evidence is part of the experience set.  It is an
        # exact return candidate, scored by the same model and continuation.
        if fields.contacted:
            reward = np.asarray(fields.rewards, np.float32) > 0.0
            evidence = np.unique(np.asarray(fields.coords, np.float32)[reward], axis=0)
            if len(evidence):
                target = self.graph.project(
                    np.concatenate(
                        (evidence, np.zeros((len(evidence), 2), np.float32)), axis=1
                    )
                )[0]
                value, connection, _ = self._candidate_q(
                    state,
                    evidence,
                    target,
                    remaining,
                    np.ones(len(evidence), np.float32),
                    fields,
                )
                commands.append(evidence)
                scores.append(value)
                connections.append(connection)
                kinds.append(np.full(len(evidence), "online_positive_evidence", object))
                associations.append(np.zeros(len(evidence), np.float32))
                offline_rows.append(np.zeros(len(evidence), np.float32))
                advance_rows.append(np.ones(len(evidence), np.float32))
                targets.append(target)
                execution_rows.append(np.ones(len(evidence), np.float32))

        command = np.concatenate(commands)
        score = np.concatenate(scores)
        connection = np.concatenate(connections)
        kind = np.concatenate(kinds)
        association_all = np.concatenate(associations)
        offline_all = np.concatenate(offline_rows)
        advance_all = np.concatenate(advance_rows)
        target_all = np.concatenate(targets).astype(np.int32)
        execution_all = np.concatenate(execution_rows).astype(np.float32)
        baseline_winner = int(np.argmax(score))
        lookahead = {}
        if self.acquisition_active and self.consumptive_lookahead:
            beam_count = min(
                CONSUMPTIVE_LOOKAHEAD_BEAM_WIDTH, len(score)
            )
            beam = np.argpartition(score, -beam_count)[-beam_count:]
            beam = beam[np.argsort(score[beam])[::-1]]
            first_gain = self.passages.live_gains(
                state[:2], command[beam], fields
            )
            passages = self.passages.live_passage_samples(
                state[:2], command[beam]
            )
            member_probability = self._member_connection(
                state, command[beam]
            ) * execution_all[beam, None]
            source_index = int(self.graph.project(state[None])[0][0])
            failure = float(self.values[remaining - 1, source_index])
            consumed_tail = np.zeros(beam_count, np.float32)
            consumed_score = np.zeros(beam_count, np.float32)
            for local, candidate in enumerate(beam):
                tail = self._consumed_tail(
                    fields,
                    passages[local],
                    remaining,
                    int(target_all[candidate]),
                )
                consumed_tail[local] = tail
                success = float(first_gain[local]) + (
                    1.0 - float(first_gain[local])
                ) * tail
                member_q = (
                    member_probability[local] * success
                    + (1.0 - member_probability[local]) * failure
                )
                consumed_score[local] = float(member_q.min())
            local_winner = int(np.argmax(consumed_score))
            winner = int(beam[local_winner])
            lookahead = {
                "selected_baseline_score": float(score[winner]),
                "baseline_selected_kind": str(kind[baseline_winner]),
                "baseline_selected_command_distance_m": float(
                    np.linalg.norm(command[baseline_winner] - state[:2])
                ),
                "selected_first_passage_gain": float(
                    first_gain[local_winner]
                ),
                "selected_consumed_tail_value": float(
                    consumed_tail[local_winner]
                ),
                "consumptive_lookahead_depth_global": int(
                    CONSUMPTIVE_LOOKAHEAD_DEPTH
                ),
                "consumptive_lookahead_beam_width_global": int(
                    CONSUMPTIVE_LOOKAHEAD_BEAM_WIDTH
                ),
                "lookahead_changed_baseline_choice": bool(
                    winner != baseline_winner
                ),
                "lookahead_beam_baseline_scores": score[beam].tolist(),
                "lookahead_beam_consumed_scores": consumed_score.tolist(),
            }
            selected_score = float(consumed_score[local_winner])
            score_median = float(np.median(consumed_score))
        else:
            winner = baseline_winner
            selected_score = float(score[winner])
            score_median = float(np.median(score))
        selected_destination = (
            command[winner].copy()
            if str(kind[winner]) == "online_positive_evidence"
            else self.graph.nodes[target_all[winner], :2].copy()
        )
        if getattr(self, "destination_arrival_correction", False):
            self.pending_arrival = (
                state.copy(), command[winner].copy(), selected_destination.copy()
            )
        return command[winner].copy(), {
            "phase": "acquisition" if self.acquisition_active else "return",
            "remaining": int(remaining),
            "selected_score": selected_score,
            "selected_kind": str(kind[winner]),
            "selected_connection_weight": float(connection[winner]),
            "selected_command_distance_m": float(
                np.linalg.norm(command[winner] - state[:2])
            ),
            "selected_alias_association": float(association_all[winner]),
            "selected_alias_offline_score": float(offline_all[winner]),
            "selected_advance_factor": float(advance_all[winner]),
            "selected_target_index": int(target_all[winner]),
            "selected_link_destination_xy": (
                selected_destination.astype(float).tolist()
            ),
            "candidate_score_maximum": selected_score,
            "candidate_score_median": score_median,
            "search_value_maximum": (
                float(self.values.max()) if self.acquisition_active else None
            ),
            "ensemble_reduction": "pessimistic_minimum",
            "online_consequence_records": int(len(self.online_starts)),
            "online_command_support_maximum": float(
                self._last_online_command_support.max()
            ) if self._last_online_command_support is not None else 0.0,
            "online_endpoint_support_maximum": float(
                self._last_online_endpoint_support.max()
            ) if self._last_online_endpoint_support is not None else 0.0,
            **lookahead,
        }


def load_planner(
    models,
    roadmap,
    planner,
    device,
    seed,
    advance_weighting=False,
    consumptive_lookahead=False,
):
    models, roadmap, planner = Path(models), Path(roadmap), Path(planner)
    metrics = load_json(models / "factorized_consequence_metrics.json")
    roadmeta = load_json(roadmap / "roadmap.json")
    bridge_meta = load_json(planner / "bridges.json")
    if bridge_meta["factorized_consequence_sha256"] != metrics["artifact_sha256"]:
        raise ValueError("planner bridge cache uses another consequence model")
    if bridge_meta["recorded_graph_sha256"] != roadmeta["graph_sha256"]:
        raise ValueError("planner bridge cache uses another graph")
    graph = RecordedRoadmap.load(roadmeta["graph"])
    with np.load(roadmeta["reach_candidates"], allow_pickle=False) as stored:
        landmarks = stored["landmark_index"]
        targets = stored["target_index"]
        offsets = stored["reach_offsets"]
    with np.load(planner / "bridges.npz", allow_pickle=False) as stored:
        member_weight = stored["member_weight"]
    consequence = FactorizedConsequenceEnsemble(metrics["artifact"], device)
    mobility = MobilityModel(planner / "mobility.pt", device)
    passages = PassageGeometry(
        roadmap / "passages", graph, targets, metrics["kernel_sigma_m"]
    )
    contract = protocol()
    return PointMazeRoadmapPlanner(
        graph,
        landmarks,
        targets,
        offsets,
        member_weight,
        consequence,
        mobility,
        passages,
        metrics["kernel_sigma_m"],
        finite_horizon(contract["evaluator"]["episode_primitive_steps"]),
        seed,
        advance_weighting=advance_weighting,
        consumptive_lookahead=consumptive_lookahead,
    )
