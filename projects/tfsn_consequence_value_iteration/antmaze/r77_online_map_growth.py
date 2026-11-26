#!/usr/bin/env python3
"""R75 task-6 replay with task-local recorded-link map growth."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from projects.tfsn_consequence_value_iteration.antmaze import (
    r48_rule_free_r39 as r48,
)
from projects.tfsn_consequence_value_iteration.antmaze import (
    r52_e549_preparation as r52,
)
from projects.tfsn_consequence_value_iteration.antmaze import (
    r67_e549_hub_routed_exploration as r67,
)
from projects.tfsn_consequence_value_iteration.antmaze import (
    r71_sigma_generalized_correction as r71,
)
from projects.tfsn_consequence_value_iteration.antmaze import (
    r74_reach_scale_route_correction as r74,
)
from projects.tfsn_consequence_value_iteration.antmaze import (
    r75_r74_twenty_episode as r75,
)
from projects.tfsn_consequence_value_iteration.antmaze import (
    r76_return_route_correction as r76,
)
from projects.tfsn_consequence_value_iteration.antmaze.successor_reductions import (
    finite_horizon_edge_values,
    value_coverage,
)
from projects.tfsn_consequence_value_iteration.shared.settings import (
    RECORDED_EVIDENCE_PRIOR_WEIGHT,
)


HERE = Path(__file__).resolve().parent
MANAGEMENT_ROOT = HERE.parents[2]
REFERENCE = HERE / "R77_ONLINE_MAP_GROWTH_REFERENCE.md"
PROTOCOL = HERE / "r77_online_map_growth_protocol.json"
METHOD_REVISION = "antmaze_e549_5m_task6_online_map_growth_return_r77"
TASK_ID = 6
HORIZON = r75.HORIZON
EPISODES = r75.EPISODES
R75_OUTPUT = r75.DEFAULT_OUTPUT
R76_OUTPUT = r76.DEFAULT_OUTPUT
DEFAULT_OUTPUT = (
    MANAGEMENT_ROOT
    / "projects/tfsn_consequence_value_iteration/results/antmaze"
    / "e549_5m_task6_online_map_growth_return_r77"
)


def load_protocol() -> dict:
    value = json.loads(PROTOCOL.read_text())
    expected = {
        "method_revision": METHOD_REVISION,
        "base_preparation_revision": r52.METHOD_REVISION,
        "base_evaluation_revision": r75.METHOD_REVISION,
        "task_ids": [TASK_ID],
        "episodes": EPISODES,
        "commands_per_episode": HORIZON,
        "primitive_steps_per_episode": HORIZON * r48.COMMAND_STEPS,
        "fixed_command_duration": r48.COMMAND_STEPS,
        "evaluation_seed": 607,
        "return_route_prior_weight": float(RECORDED_EVIDENCE_PRIOR_WEIGHT),
        "return_route_similarity_cutoff": None,
        "route_source_scale": "R52_empirical_reach_covariance",
        "route_target_scale": "R52_empirical_reach_covariance",
        "acquisition_change_from_R75": "none",
        "hidden_goal_visible_to_planner_executor_or_models": False,
        "success_radius_read_by_planner": False,
        "task_gradient_updates": 0,
        "confirmation_panel_accessed": False,
    }
    for key, expected_value in expected.items():
        if value.get(key) != expected_value:
            raise ValueError(f"R77 protocol has another {key}")
    reference = Path(value["method_reference"])
    if not reference.is_absolute():
        reference = MANAGEMENT_ROOT / reference
    if reference.resolve() != REFERENCE.resolve():
        raise ValueError("R77 method reference changed")
    return value


PROTOCOL_VALUE = load_protocol()


def landing_association(distance, sigma):
    distance = np.asarray(distance, np.float64)
    return np.exp(-0.5 * np.square(distance / float(sigma)))


def project_online_links(node_xy, starts_xy, endpoints_xy, sigma):
    """Associate exact online nodes continuously with the offline partition."""
    nodes = np.asarray(node_xy, np.float64).reshape(-1, 2)
    starts = np.asarray(starts_xy, np.float64).reshape(-1, 2)
    endpoints = np.asarray(endpoints_xy, np.float64).reshape(-1, 2)
    if len(starts) != len(endpoints):
        raise ValueError("online starts and endpoints must align")
    if not len(starts):
        empty_i = np.empty(0, np.int32)
        empty_f = np.empty(0, np.float32)
        return empty_i, empty_i.copy(), empty_f, empty_f.copy(), empty_f.copy()
    tree = cKDTree(nodes)
    source_distance, source = tree.query(starts, k=1)
    endpoint_distance, endpoint = tree.query(endpoints, k=1)
    source_association = landing_association(source_distance, sigma)
    endpoint_association = landing_association(endpoint_distance, sigma)
    reliability = source_association * endpoint_association
    return (
        np.asarray(source, np.int32),
        np.asarray(endpoint, np.int32),
        np.asarray(reliability, np.float32),
        np.asarray(source_association, np.float32),
        np.asarray(endpoint_association, np.float32),
    )


def route_reliability_layers(
    node_xy, target_xy, records, inverse_covariance, horizon,
):
    """Posterior route reliability, with one successful prior observation."""
    nodes = np.asarray(node_xy, np.float64).reshape(-1, 2)
    target = np.asarray(target_xy, np.float64).reshape(2)
    inverse = np.asarray(inverse_covariance, np.float64).reshape(2, 2)
    reliability = np.ones((int(horizon) + 1, len(nodes)), np.float32)
    support = np.zeros_like(reliability)
    arrivals = np.zeros_like(reliability)
    for remaining in range(1, int(horizon) + 1):
        matching = [
            row for row in records
            if int(row["remaining_commands"]) == remaining
        ]
        if not matching:
            continue
        observed_source = np.asarray(
            [row["source_xy"] for row in matching], np.float64
        )
        observed_target = np.asarray(
            [row["target_xy"] for row in matching], np.float64
        )
        outcome = np.asarray(
            [row["arrived"] for row in matching], np.float64
        )
        source_delta = nodes[None] - observed_source[:, None]
        target_delta = target[None] - observed_target
        source_distance = np.einsum(
            "rni,ij,rnj->rn", source_delta, inverse, source_delta
        )
        target_distance = np.einsum(
            "ri,ij,rj->r", target_delta, inverse, target_delta
        )
        similarity = np.exp(
            -0.5 * (source_distance + target_distance[:, None])
        )
        support[remaining] = similarity.sum(axis=0).astype(np.float32)
        arrivals[remaining] = (
            similarity * outcome[:, None]
        ).sum(axis=0).astype(np.float32)
        reliability[remaining] = np.clip(
            (
                float(RECORDED_EVIDENCE_PRIOR_WEIGHT)
                + arrivals[remaining]
            )
            / (
                float(RECORDED_EVIDENCE_PRIOR_WEIGHT)
                + support[remaining]
            ),
            0.0,
            1.0,
        )
    return reliability, support, arrivals


def r75_values_with_online_links(
    graph, targets, bridge_weight, node_field, horizon,
    online_source=(), online_endpoint=(), online_reliability=(),
    route_reliability=None,
):
    """The exact R75 max backup plus additional recorded online links."""
    targets = np.asarray(targets, np.int32)
    bridge_weight = np.asarray(bridge_weight, np.float32)
    node_field = np.asarray(node_field, np.float32).reshape(-1)
    source = np.asarray(online_source, np.int32).reshape(-1)
    endpoint = np.asarray(online_endpoint, np.int32).reshape(-1)
    reliability = np.asarray(online_reliability, np.float32).reshape(-1)
    if not (len(source) == len(endpoint) == len(reliability)):
        raise ValueError("online link arrays must align")
    if route_reliability is None:
        route_reliability = np.ones(
            (int(horizon) + 1, graph.node_count), np.float32
        )
    route_reliability = np.asarray(route_reliability, np.float32)
    expected_shape = (int(horizon) + 1, graph.node_count)
    if route_reliability.shape != expected_shape:
        raise ValueError("route reliability has the wrong shape")

    values = np.zeros(expected_shape, np.float32)
    for remaining in range(1, int(horizon) + 1):
        previous = values[remaining - 1]
        recorded_q = graph.edge_reliability * (
            node_field[graph.endpoint_index]
            + previous[graph.endpoint_index]
        )
        row = np.zeros(graph.node_count, np.float32)
        np.maximum.at(row, graph.source_index, recorded_q)
        if len(source):
            online_q = reliability * (
                node_field[endpoint] + previous[endpoint]
            )
            np.maximum.at(row, source, online_q)
        bridge_q = bridge_weight * (
            node_field[targets] + previous[targets]
        )
        row = np.maximum(row, bridge_q.max(axis=1))
        values[remaining] = row * route_reliability[remaining]
    return values


def exact_reachability_with_online_links(
    target, horizon, node_count, source, endpoint, edge_weight,
    bridge_target, bridge_weight, online_source=(), online_endpoint=(),
    online_reliability=(),
):
    """R67 exact target audit with added task-local recorded links."""
    online_source = np.asarray(online_source, np.int32).reshape(-1)
    online_endpoint = np.asarray(online_endpoint, np.int32).reshape(-1)
    online_reliability = np.asarray(
        online_reliability, np.float32
    ).reshape(-1)
    values = np.zeros((int(horizon) + 1, int(node_count)), np.float32)
    values[0, int(target)] = 1.0
    for remaining in range(1, int(horizon) + 1):
        previous = values[remaining - 1]
        bridge_q = (
            bridge_weight * previous[bridge_target]
            + (1.0 - bridge_weight) * previous[:, None]
        )
        best = bridge_q.max(axis=1)
        recorded_q = (
            edge_weight * previous[endpoint]
            + (1.0 - edge_weight) * previous[source]
        )
        np.maximum.at(best, source, recorded_q)
        if len(online_source):
            online_q = (
                online_reliability * previous[online_endpoint]
                + (1.0 - online_reliability) * previous[online_source]
            )
            np.maximum.at(best, online_source, online_q)
        best[int(target)] = 1.0
        values[remaining] = best
    return values


def _rows(path):
    return [
        json.loads(line) for line in Path(path).read_text().splitlines()
        if line.strip()
    ]


def offline_map_growth_certificate(args) -> dict:
    """Use only already observed R75/R76 task evidence for the preflight."""
    arrays = r67._graph_arrays(Path(args.preparation).resolve())
    (_paths, consequence, nodes, source, endpoint, edge_weight,
     bridge_target, bridge_weight) = arrays
    r75_log = R75_OUTPUT / f"task_{TASK_ID:03d}.jsonl"
    r76_log = R76_OUTPUT / f"task_{TASK_ID:03d}.jsonl"
    acquisition = [row for row in _rows(r75_log) if row["phase"] == "acquisition"]
    return_rows = [row for row in _rows(r76_log) if row["phase"] == "return"]
    if not acquisition or not return_rows:
        raise FileNotFoundError("R75 acquisition or R76 observed evidence is absent")
    target_xy = np.asarray(return_rows[0]["return_target_xy"], np.float64)
    starts = np.asarray([row["start_xy"] for row in acquisition], np.float64)
    endpoints = np.asarray(
        [row["actual_endpoint"] for row in acquisition], np.float64
    )
    online_source, online_endpoint, reliability, source_alias, endpoint_alias = (
        project_online_links(
            nodes[:, :2], starts, endpoints,
            float(consequence["landing_sigma_m"]),
        )
    )
    tree = cKDTree(nodes[:, :2])
    reset_index = int(tree.query(starts[0], k=1)[1])
    target_index = int(tree.query(target_xy, k=1)[1])
    rewarded = [row for row in acquisition if bool(row.get("reward"))]
    if not rewarded:
        raise RuntimeError("R75 acquisition log has no observed contact command")
    contact_command_endpoint = np.asarray(
        rewarded[0]["actual_endpoint"], np.float64
    )
    contact_endpoint_index = int(
        tree.query(contact_command_endpoint, k=1)[1]
    )
    before = r67._one_target_layers(
        target_index, HORIZON, len(nodes), source, endpoint, edge_weight,
        bridge_target, bridge_weight,
    )
    after = exact_reachability_with_online_links(
        target_index, HORIZON, len(nodes), source, endpoint, edge_weight,
        bridge_target, bridge_weight, online_source, online_endpoint,
        reliability,
    )
    endpoint_before = r67._one_target_layers(
        contact_endpoint_index, HORIZON, len(nodes), source, endpoint,
        edge_weight, bridge_target, bridge_weight,
    )
    endpoint_after = exact_reachability_with_online_links(
        contact_endpoint_index, HORIZON, len(nodes), source, endpoint,
        edge_weight, bridge_target, bridge_weight, online_source,
        online_endpoint, reliability,
    )
    contact_distance = float(np.linalg.norm(
        contact_command_endpoint - target_xy
    ))
    contact_association = float(landing_association(
        contact_distance, float(consequence["landing_sigma_m"])
    ))
    endpoint_reach_before = float(endpoint_before[HORIZON, reset_index])
    endpoint_reach_after = float(endpoint_after[HORIZON, reset_index])
    report = {
        "schema_version": 1,
        "source_artifact": str(r75_log),
        "source_artifact_sha256": r52.sha256(r75_log),
        "observed_evidence_artifact": str(r76_log),
        "observed_evidence_artifact_sha256": r52.sha256(r76_log),
        "online_links": int(len(reliability)),
        "online_link_reliability_mean": float(reliability.mean()),
        "online_link_reliability_minimum": float(reliability.min()),
        "source_association_mean": float(source_alias.mean()),
        "endpoint_association_mean": float(endpoint_alias.mean()),
        "reset_xy": starts[0].tolist(),
        "observed_contact_xy": target_xy.tolist(),
        "contact_command_endpoint_xy": contact_command_endpoint.tolist(),
        "contact_to_command_endpoint_distance_m": contact_distance,
        "contact_to_command_endpoint_landing_association": contact_association,
        "reset_node": reset_index,
        "contact_node": target_index,
        "offline_exact_reachability_before_growth": float(
            before[HORIZON, reset_index]
        ),
        "exact_reachability_after_growth": float(
            after[HORIZON, reset_index]
        ),
        "offline_exact_reachability_to_contact_command_endpoint_before_growth": (
            endpoint_reach_before
        ),
        "exact_reachability_to_contact_command_endpoint_after_growth": (
            endpoint_reach_after
        ),
        "offline_contact_via_observed_endpoint_association_before_growth": (
            endpoint_reach_before * contact_association
        ),
        "missing_route_hypothesis_supported": bool(
            endpoint_reach_before == 0.0
        ),
        "map_growth_makes_observed_route_positive": bool(
            after[HORIZON, reset_index] > before[HORIZON, reset_index]
            and after[HORIZON, reset_index] > 0.0
        ),
        "hidden_goal_used": False,
        "task_reward_used": False,
        "contact_coordinate_source": "first_observed_positive_evidence",
        "interpretation": (
            "the_exact_single_contact_node_is_disconnected_but_the_recorded_"
            "command_endpoint_is_already_reachable_and_is_within_the_deployed_"
            "landing_kernel_of_contact"
        ),
    }
    return report


class OnlineMapGrowthReturnPlanner(
    r71.SigmaGeneralizedDestinationArrivalPlanner
):
    """R75 plus task-local recorded links and route reliability."""

    method_revision = METHOD_REVISION
    planner_name = "E549_R75_task_local_recorded_link_return"

    def configure_online_map_growth(self, reach_covariance):
        self.online_map_tree = cKDTree(self.graph.nodes[:, :2])
        self.route_inverse_covariance = np.linalg.inv(
            np.asarray(reach_covariance, np.float64).reshape(2, 2)
        )
        self._reset_online_map_growth()

    def _reset_online_map_growth(self):
        self.map_starts = []
        self.map_commands = []
        self.map_endpoints = []
        self.map_source_index = []
        self.map_endpoint_index = []
        self.map_edge_reliability = []
        self.map_endpoint_association = []
        self.map_pending_start = None
        self.map_pending_command = None
        self.map_pending_online_length = 0
        self.return_route_records = []
        self.return_route_prefixes = []
        self.return_episode_active = False
        self.return_episode_positive_count = 0
        self.return_target_xy = None
        self.return_route_support = None
        self.return_route_arrivals = None
        self.return_route_reliability = None
        self.return_route_attempts = 0
        self.return_route_successes = 0

    def reset_online_correction_task(self):
        super().reset_online_correction_task()
        if hasattr(self, "online_map_tree"):
            self._reset_online_map_growth()

    def _ingest_map_pending(self, online_states):
        if (
            self.map_pending_start is None
            or len(online_states) <= int(self.map_pending_online_length)
        ):
            return False
        endpoint = np.asarray(online_states[-1], np.float32).copy()
        start = np.asarray(self.map_pending_start, np.float32).copy()
        command = np.asarray(self.map_pending_command, np.float32).copy()
        projected = project_online_links(
            self.graph.nodes[:, :2], start[None, :2], endpoint[None, :2],
            float(r48.SIGMA),
        )
        self.map_starts.append(start)
        self.map_commands.append(command)
        self.map_endpoints.append(endpoint)
        self.map_source_index.append(int(projected[0][0]))
        self.map_endpoint_index.append(int(projected[1][0]))
        self.map_edge_reliability.append(float(projected[2][0]))
        self.map_endpoint_association.append(float(projected[4][0]))
        self.map_pending_start = None
        self.map_pending_command = None
        return True

    @staticmethod
    def _observed_contact(fields):
        return r76.has_observed_positive_evidence(fields)

    @staticmethod
    def _positive_xy(fields):
        return r76.ReturnRouteCorrectedPlanner._first_positive_xy(fields)

    def _finish_return_episode(self, arrived):
        if not self.return_episode_active:
            return
        for source_xy, remaining in self.return_route_prefixes:
            self.return_route_records.append({
                "source_xy": np.asarray(source_xy, np.float32).tolist(),
                "target_xy": np.asarray(
                    self.return_target_xy, np.float32
                ).tolist(),
                "remaining_commands": int(remaining),
                "arrived": bool(arrived),
            })
        self.return_route_attempts += 1
        self.return_route_successes += int(bool(arrived))
        self.return_route_prefixes = []
        self.return_episode_active = False

    def finalize_return_episode(self, arrived):
        self._finish_return_episode(bool(arrived))

    def _begin_return_episode(self, command_index, fields):
        if int(command_index) != 0:
            return
        if self.return_episode_active:
            positive = int(np.count_nonzero(np.asarray(fields.rewards) > 0.0))
            self._finish_return_episode(
                positive > int(self.return_episode_positive_count)
            )
        self.return_episode_active = True
        self.return_route_prefixes = []
        self.return_episode_positive_count = int(
            np.count_nonzero(np.asarray(fields.rewards) > 0.0)
        )

    def _set_goal(self, fields):
        super()._set_goal(fields)
        if not self._observed_contact(fields):
            return
        target = self._positive_xy(fields)
        if self.return_target_xy is None:
            self.return_target_xy = target.copy()
        elif not np.array_equal(self.return_target_xy, target):
            raise RuntimeError("first observed return evidence changed")
        route, support, arrivals = route_reliability_layers(
            self.graph.nodes[:, :2], target, self.return_route_records,
            self.route_inverse_covariance, int(self.horizon),
        )
        self.values = r75_values_with_online_links(
            self.graph, self.targets, self.weight, self.node_field,
            int(self.horizon), self.map_source_index,
            self.map_endpoint_index, self.map_edge_reliability, route,
        )
        self.coverage = value_coverage(self.values)
        self.return_route_reliability = route
        self.return_route_support = support
        self.return_route_arrivals = arrivals

    def _online_map_candidates(self, state, remaining, command_index):
        if not self.map_commands:
            return None
        starts = np.asarray(self.map_starts, np.float32)
        commands = np.asarray(self.map_commands, np.float32)
        targets = np.asarray(self.map_endpoint_index, np.int32)
        distance = np.linalg.norm(starts[:, :2] - state[None, :2], axis=1)
        association = landing_association(distance, r48.SIGMA).astype(np.float32)
        reliability = association * np.asarray(
            self.map_endpoint_association, np.float32
        )
        q, connection = self._candidate_value(
            state, commands, targets, remaining, reliability,
            self.seed + int(command_index) * 10_007 + 769,
        )
        winner = int(np.argmax(q))
        return {
            "command": commands[winner].copy(),
            "target": int(targets[winner]),
            "score": float(q[winner]),
            "connection": float(connection[winner]),
            "association": float(association[winner]),
            "reliability": float(reliability[winner]),
            "edge_index": winner,
        }

    def choose(
        self, state, command_index, fields, online_states=(),
        online_trajectories=(),
    ):
        self._ingest_map_pending(online_states)
        observed_contact = self._observed_contact(fields)
        if observed_contact:
            self._begin_return_episode(command_index, fields)
        command, diagnostic = super().choose(
            state, command_index, fields, online_states, online_trajectories
        )
        state = np.asarray(state, np.float32)
        command = np.asarray(command, np.float32)
        if observed_contact:
            remaining = int(self.horizon) - int(command_index)
            candidate = self._online_map_candidates(
                state, remaining, command_index
            )
            if candidate is not None and candidate["score"] > float(
                diagnostic["selected_score"]
            ):
                command = candidate["command"]
                destination = self.graph.nodes[candidate["target"], :2].copy()
                self.pending_command = command.copy()
                self.pending_destination = destination
                diagnostic.update({
                    "selected_score": candidate["score"],
                    "selected_kind": "task_local_recorded_edge",
                    "selected_connection_weight": candidate["connection"],
                    "selected_command_distance": float(
                        np.linalg.norm(command - state[:2])
                    ),
                    "selected_alias_association": candidate["association"],
                    "selected_execution_partition_weight": candidate[
                        "reliability"
                    ],
                    "selected_alias_offline_score": candidate["score"],
                    "selected_link_destination_xy": destination.tolist(),
                    "online_map_edge_index": candidate["edge_index"],
                })
                diagnostic.update(self._selected_online_diagnostic(
                    command, candidate["connection"]
                ))
            current_node = int(self.online_map_tree.query(state[:2], k=1)[1])
            self.return_route_prefixes.append((state[:2].copy(), remaining))
            diagnostic.update({
                "online_map_nodes_retained": int(2 * len(self.map_commands)),
                "online_map_recorded_links": int(len(self.map_commands)),
                "return_value_base": "unchanged_R75_Bellman_backup",
                "return_route_reliability_at_current_state": float(
                    self.return_route_reliability[remaining, current_node]
                ),
                "return_route_support_at_current_state": float(
                    self.return_route_support[remaining, current_node]
                ),
                "return_route_similarity_cutoff": None,
            })
        self.map_pending_start = state.copy()
        self.map_pending_command = command.copy()
        self.map_pending_online_length = len(online_states)
        return command, diagnostic

    def online_map_summary(self):
        return {
            "online_map_nodes_retained": int(2 * len(self.map_commands)),
            "online_map_recorded_links": int(len(self.map_commands)),
            "online_map_link_reliability_mean": (
                float(np.mean(self.map_edge_reliability))
                if self.map_edge_reliability else None
            ),
            "return_route_attempts": int(self.return_route_attempts),
            "return_route_arrivals": int(self.return_route_successes),
            "return_route_failures": int(
                self.return_route_attempts - self.return_route_successes
            ),
        }


def _bind_reporting_contract():
    r67.REFERENCE = REFERENCE
    r67.PROTOCOL = PROTOCOL
    r67.METHOD_REVISION = METHOD_REVISION
    r67.PROTOCOL_VALUE = PROTOCOL_VALUE
    r67.EPISODES = EPISODES


def _load_runtime(args):
    loaded = list(r74._load_runtime(args))
    _bind_reporting_contract()
    planner = loaded[0]
    planner.__class__ = OnlineMapGrowthReturnPlanner
    planner.method_revision = METHOD_REVISION
    planner.planner_name = "E549_R75_task_local_recorded_link_return"
    planner.horizon = HORIZON
    reach = r74.empirical_reach_covariance(args.preparation)
    planner.configure_online_map_growth(reach["covariance_m2"])
    loaded[0] = planner
    return tuple(loaded)


def acquisition_signature(path):
    rows = [row for row in _rows(path) if row.get("phase") == "acquisition"]
    digest = hashlib.sha256()
    for row in rows:
        digest.update(np.asarray(
            [int(row["episode"]), int(row["command_index"])], np.int32
        ).tobytes())
        digest.update(np.asarray(row["command"], np.float32).tobytes())
        digest.update(np.asarray(row["actual_endpoint"], np.float32).tobytes())
        digest.update(np.asarray([bool(row["reward"])], np.uint8).tobytes())
    return {"commands": len(rows), "sha256": digest.hexdigest()}


def command_audit(args):
    with contextlib.redirect_stdout(io.StringIO()):
        base = r74.command_audit(args)
    _bind_reporting_contract()
    baseline_task = R75_OUTPUT / f"task_{TASK_ID:03d}.json"
    baseline_log = R75_OUTPUT / f"task_{TASK_ID:03d}.jsonl"
    baseline = json.loads(baseline_task.read_text())
    certificate = offline_map_growth_certificate(args)
    report = {
        "schema_version": 1,
        "method_reference": str(REFERENCE),
        "method_reference_sha256": r52.sha256(REFERENCE),
        "method_revision": METHOD_REVISION,
        "claim_level": PROTOCOL_VALUE["claim_level"],
        "declared_deviations": PROTOCOL_VALUE["declared_deviations"],
        "oracle_inputs": [],
        "task_ids": [TASK_ID],
        "episodes": EPISODES,
        "commands_per_episode": HORIZON,
        "base_R75_task": str(baseline_task),
        "base_R75_first_contact_episode": baseline["first_contact_episode"],
        "base_R75_conditional_return": baseline["conditional_return"],
        "online_map_growth_certificate": certificate,
        "r75_value_identity_gate": "covered_by_unit_test_exact_array_equality",
        "acquisition_change_from_R75": "none",
        "ready_for_evaluation": bool(
            base.get("ready_for_evaluation") is True
            and baseline_log.is_file()
            and certificate["map_growth_makes_observed_route_positive"]
        ),
        "hidden_goal_visible_to_planner_executor_or_models": False,
        "success_radius_read_by_planner": False,
        "task_gradient_updates": 0,
        "confirmation_panel_accessed": False,
    }
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    r48.dump(output / "audit.json", report)
    r48.dump(output / "online_map_growth_certificate.json", certificate)
    print(json.dumps(report, indent=2))
    return report


def command_evaluate(args):
    if int(args.episodes) != EPISODES or int(args.seed) != 607:
        raise ValueError("R77 requires seed 607 and twenty episodes")
    audit = command_audit(args)
    if not audit["ready_for_evaluation"]:
        raise RuntimeError("R77 offline gates did not pass")
    output = Path(args.output).resolve()
    summary_path = output / "summary.json"
    if summary_path.is_file():
        saved = json.loads(summary_path.read_text())
        if saved.get("method_revision") != METHOD_REVISION:
            raise ValueError("saved R77 summary has another method")
        print(json.dumps(saved, indent=2))
        return saved
    planner, executor, env, hub_nodes, hub_xy, nodes, reachability = (
        _load_runtime(args)
    )
    try:
        row = r67._evaluate_task(
            args, planner, executor, env, TASK_ID, output, hub_nodes, hub_xy,
            reachability, nodes, float(r48.SIGMA),
        )
    finally:
        env.close()
    planner.finalize_return_episode(bool(row["successes"][-1]))
    row.update(planner.online_map_summary())
    task_path = output / f"task_{TASK_ID:03d}.json"
    r48.dump(task_path, row)
    baseline = json.loads(
        (R75_OUTPUT / f"task_{TASK_ID:03d}.json").read_text()
    )
    baseline_signature = acquisition_signature(
        R75_OUTPUT / f"task_{TASK_ID:03d}.jsonl"
    )
    replay_signature = acquisition_signature(
        output / f"task_{TASK_ID:03d}.jsonl"
    )
    result = {
        "schema_version": 1,
        "method_reference": str(REFERENCE),
        "method_reference_sha256": r52.sha256(REFERENCE),
        "method_revision": METHOD_REVISION,
        "claim_level": PROTOCOL_VALUE["claim_level"],
        "declared_deviations": PROTOCOL_VALUE["declared_deviations"],
        "oracle_inputs": [],
        "task_id": TASK_ID,
        "R75_first_contact_episode": baseline["first_contact_episode"],
        "R77_first_contact_episode": row["first_contact_episode"],
        "R75_conditional_return": baseline["conditional_return"],
        "R77_conditional_return": row["conditional_return"],
        "R75_successes": baseline["successes"],
        "R77_successes": row["successes"],
        "acquisition_replay_R75": baseline_signature,
        "acquisition_replay_R77": replay_signature,
        "acquisition_replay_exact": baseline_signature == replay_signature,
        **planner.online_map_summary(),
        "hidden_goal_visible_to_planner_executor_or_models": False,
        "success_radius_read_by_planner": False,
        "task_gradient_updates": 0,
        "confirmation_panel_accessed": False,
    }
    r48.dump(summary_path, result)
    print(json.dumps(result, indent=2))
    return result


def parser():
    value = argparse.ArgumentParser()
    value.add_argument(
        "--root", type=Path, default=Path("/home/ai_diagnosis/SG/e377")
    )
    value.add_argument("--device", default="cuda")
    value.add_argument("--seed", type=int, default=607)
    value.add_argument("--episodes", type=int, default=EPISODES)
    value.add_argument("--preparation", type=Path, default=r52.DEFAULT_OUTPUT)
    value.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    value.add_argument("--graph", type=Path)
    commands = value.add_subparsers(dest="command", required=True)
    commands.add_parser("audit").set_defaults(func=command_audit)
    commands.add_parser("evaluate").set_defaults(func=command_evaluate)
    return value


def main():
    args = parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
