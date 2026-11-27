"""PointMaze adapter for the frozen cross-domain R74 method."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from projects.tfsn_consequence_value_iteration.shared.frozen_hub_method import (
    ExactBatchedReachability,
    alias_hub_candidates,
    area_weights,
    empirical_reach_covariance,
    exact_hub_argmax,
    latent_hub_coverage_torch,
    route_mass,
)
from projects.tfsn_consequence_value_iteration.shared.latent_detection import (
    LatentRadiusPosterior,
)

from .fields import SuppliedGoalField
from .planner import load_planner


HERE = Path(__file__).resolve().parent
PROTOCOL = HERE / "frozen_r74_port_protocol.json"
METHOD_REVISION = "pointmaze_frozen_antmaze_r74_port_r15"


def load_protocol():
    value = json.loads(PROTOCOL.read_text())
    expected = {
        "method_revision": METHOD_REVISION,
        "frozen_source_method": (
            "antmaze_e549_5m_reach_scale_hub_route_correction_r74"
        ),
        "link_similarity_scale": "deployed_landing_sigma",
        "link_similarity_cutoff": None,
        "route_source_scale": "empirical_reach_covariance",
        "route_hub_scale": "empirical_reach_covariance",
        "route_similarity_cutoff": None,
        "hub_evaluation_cutoff": None,
        "hub_target_batch_computational": 64,
        "recorded_evidence_prior_weight": 1.0,
        "hidden_goal_visible_to_planner_executor_or_models": False,
        "success_radius_read_by_planner": False,
        "confirmation_panel_accessed": False,
    }
    for key, expected_value in expected.items():
        if value.get(key) != expected_value:
            raise ValueError(f"PointMaze frozen R74 port has another {key}")
    return value


PROTOCOL_VALUE = load_protocol()


class FrozenR74PointPlanner:
    """Latent hubs plus sigma-link and empirical-reach route correction."""

    def __init__(
        self, base, device, *, method_revision=METHOD_REVISION,
        protocol_value=None,
    ):
        self.base = base
        self.graph = base.graph
        self.goal_dim = int(self.graph.goal_dim)
        self.sigma = float(base.sigma)
        self.horizon = int(base.horizon)
        self.method_revision = str(method_revision)
        self.protocol_value = (
            PROTOCOL_VALUE if protocol_value is None else dict(protocol_value)
        )
        self.device = device
        self.base.enable_destination_arrival_correction()

        support = np.unique(
            self.graph.nodes[:, : self.goal_dim].astype(np.float64), axis=0
        )
        prior, occupied, origin = area_weights(support, self.sigma)
        self.support_xy = support
        self.area_prior = prior
        self.area_occupied_cells = occupied
        self.area_origin = origin

        self.hub_nodes, _inverse, self.hub_origin = alias_hub_candidates(
            self.graph.nodes[:, : self.goal_dim], self.sigma
        )
        self.hub_xy = self.graph.nodes[
            self.hub_nodes, : self.goal_dim
        ].astype(np.float64)
        reach = empirical_reach_covariance(
            self.graph.nodes[:, : self.goal_dim],
            self.graph.source_index,
            self.graph.endpoint_index,
            self.graph.commands,
            self.sigma,
        )
        self.reach = reach
        self.reach_inverse = reach["inverse_covariance_per_m2"]
        self.solver = ExactBatchedReachability(
            self.graph.source_index,
            self.graph.endpoint_index,
            self.graph.edge_reliability,
            self.base.targets,
            device,
            target_batch=int(
                self.protocol_value["hub_target_batch_computational"]
            ),
        )
        self.reset_task_memory()

    def _new_posterior(self):
        posterior = LatentRadiusPosterior(self.support_xy, self.sigma)
        if not np.array_equal(posterior.goal_xy, self.support_xy):
            raise RuntimeError("latent support ordering changed")
        posterior._prior_per_goal = self.area_prior.copy()
        posterior._refresh()
        return posterior

    def reset_task_memory(self):
        self.base.reset_task_memory()
        self.posterior = self._new_posterior()
        self.posterior_field_rows = 0
        self.route_records = []
        self.active_attempt = None
        self.pending_passage = None
        self.hub_selections = 0
        self.hub_arrivals = 0
        self.hub_failures = 0
        self.hub_censored = 0
        self.last_hub_diagnostic = {}

    def begin_episode(self):
        self.base.begin_episode()

    def _finish_attempt(self, arrived, endpoint_xy, censored=False, reason=None):
        if self.active_attempt is None:
            return None
        record = {
            **self.active_attempt,
            "arrived": None if censored else bool(arrived),
            "endpoint_xy": np.asarray(endpoint_xy, np.float32).tolist(),
            "censored": bool(censored),
        }
        if reason is not None:
            record["censoring_reason"] = str(reason)
        if censored:
            self.hub_censored += 1
        else:
            self.route_records.append(record)
            self.hub_arrivals += int(arrived)
            self.hub_failures += int(not arrived)
        self.active_attempt = None
        return record

    def end_episode(self, contacted, endpoint_xy):
        if self.active_attempt is None:
            return
        if contacted:
            self._finish_attempt(
                False, endpoint_xy, censored=True,
                reason="task_contact_before_hub_arrival",
            )
        else:
            self._finish_attempt(False, endpoint_xy)

    def _sync_negative_observations(self, fields):
        if fields.contacted:
            return
        coords = np.asarray(fields.coords, np.float64).reshape(
            -1, self.goal_dim
        )
        rewards = np.asarray(fields.rewards, np.float32).reshape(-1)
        if self.posterior_field_rows > len(coords):
            raise RuntimeError("task field shrank within one task")
        new_xy = coords[self.posterior_field_rows:]
        new_reward = rewards[self.posterior_field_rows:]
        self.posterior_field_rows = len(coords)
        positive = np.flatnonzero(new_reward > 0.0)
        if len(positive):
            new_xy = new_xy[: int(positive[0])]
        if not len(new_xy):
            return
        distance = np.linalg.norm(
            new_xy[:, None] - self.posterior.goal_xy[None], axis=2
        ).min(axis=0)
        self.posterior.upper = np.minimum(self.posterior.upper, distance)
        self.posterior.negative_observations += len(new_xy)
        self.posterior._refresh()
        self.posterior.assert_not_exhausted(
            "after PointMaze frozen-R74 traversed passage"
        )

    def _select_hub(self, state, remaining):
        coverage = latent_hub_coverage_torch(
            self.posterior,
            self.hub_xy,
            self.device,
            chunk_size=int(
                self.protocol_value["hub_target_batch_computational"]
            ),
        )
        current_node = int(self.graph.project(np.asarray(state)[None])[0][0])
        route_support, route_arrivals = route_mass(
            self.route_records,
            np.asarray(state, np.float64)[: self.goal_dim],
            self.hub_xy,
            int(remaining),
            self.reach_inverse,
        )
        bridge_weight = self.base.corrected_bridge_weight()
        decision = exact_hub_argmax(
            coverage,
            self.hub_nodes,
            int(remaining),
            current_node,
            self.solver,
            bridge_weight,
            route_support,
            route_arrivals,
            prior=float(
                self.protocol_value["recorded_evidence_prior_weight"]
            ),
        )
        selected = int(decision["selected"])
        evaluated = decision["evaluated_indices"]
        reaches = decision["reaches"]
        selected_position = np.flatnonzero(evaluated == selected)
        if not len(selected_position):
            raise RuntimeError("selected hub was not exactly evaluated")
        selected_reach = float(reaches[int(selected_position[0])])
        self.active_attempt = {
            "source_xy": np.asarray(
                state, np.float32
            )[: self.goal_dim].tolist(),
            "hub_bank_index": selected,
            "hub_node_index": int(self.hub_nodes[selected]),
            "hub_xy": self.hub_xy[selected].astype(float).tolist(),
            "remaining_commands": int(remaining),
            "predicted_reachability": selected_reach,
        }
        self.hub_selections += 1
        self.last_hub_diagnostic = {
            "hub_bank_index": selected,
            "hub_node_index": int(self.hub_nodes[selected]),
            "hub_xy": self.hub_xy[selected].astype(float).tolist(),
            "hub_contact_probability": float(coverage[selected]),
            "hub_route_reachability": selected_reach,
            "hub_combined_score": float(decision["score"]),
            "hub_candidates": int(len(self.hub_nodes)),
            "hub_candidates_exactly_evaluated": int(len(evaluated)),
            "route_evidence_records": int(len(self.route_records)),
        }
        return selected

    def choose(self, state, command_index, fields):
        if fields.contacted:
            command, diagnostic = self.base.choose(state, command_index, fields)
            diagnostic.update({
                "method_revision": self.method_revision,
                "acquisition_controller": "inactive_after_contact",
            })
            return command, diagnostic

        self._sync_negative_observations(fields)
        remaining = self.horizon - int(command_index)
        if self.active_attempt is None:
            self._select_hub(state, remaining)
        hub = np.asarray(self.active_attempt["hub_xy"], np.float32)
        route_field = SuppliedGoalField(self.sigma, hub)
        command, diagnostic = self.base.choose(
            state, command_index, route_field
        )
        diagnostic.update({
            "phase": "acquisition",
            "method_revision": self.method_revision,
            "acquisition_controller": (
                "area_weighted_latent_radius_dynamic_hub_route"
            ),
            **self.last_hub_diagnostic,
        })
        return command, diagnostic

    def observe_passage(self, xy, rewards):
        if self.active_attempt is None:
            return
        path = np.asarray(xy, np.float32).reshape(-1, self.goal_dim)
        rewards = np.asarray(rewards, np.float32).reshape(-1)
        hub = np.asarray(self.active_attempt["hub_xy"], np.float32)
        arrived = bool(np.any(np.linalg.norm(path - hub[None], axis=1) <= self.sigma))
        if arrived:
            self._finish_attempt(True, path[-1])
        elif np.any(rewards > 0.0):
            self._finish_attempt(
                False, path[-1], censored=True,
                reason="task_contact_before_hub_arrival",
            )

    def observe_transition(self, start_state, command, endpoint_state):
        self.base.observe_transition(start_state, command, endpoint_state)

    def diagnostics(self):
        return {
            "hub_candidates_data_derived": int(len(self.hub_nodes)),
            "area_prior_occupied_sigma_cells": int(self.area_occupied_cells),
            "rho_min_data_derived_m": float(self.posterior.rho_min),
            "rho_max_data_derived_m": float(self.posterior.rho_max),
            "empirical_reach_covariance_m2": self.reach[
                "covariance_m2"
            ].astype(float).tolist(),
            "online_destination_arrival_observations": int(
                len(self.base.arrival_outcomes)
            ),
            "online_destination_arrival_rate": (
                float(np.mean(self.base.arrival_outcomes))
                if self.base.arrival_outcomes else None
            ),
            "hub_route_attempts": int(
                self.hub_arrivals + self.hub_failures
            ),
            "hub_route_arrivals": int(self.hub_arrivals),
            "hub_route_failures": int(self.hub_failures),
            "hub_route_censored_attempts": int(self.hub_censored),
        }


def load_frozen_r74_planner(
    *, models, roadmap, planner, device, seed,
):
    base = load_planner(
        models,
        roadmap,
        planner,
        device,
        seed,
        advance_weighting=True,
        consumptive_lookahead=True,
    )
    return FrozenR74PointPlanner(base, device)
