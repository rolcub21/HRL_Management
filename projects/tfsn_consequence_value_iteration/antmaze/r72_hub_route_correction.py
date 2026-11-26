#!/usr/bin/env python3
"""R71 with online correction of planned hub-route outcomes."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from matplotlib.lines import Line2D

from projects.tfsn_consequence_value_iteration.antmaze import (
    r48_rule_free_r39 as r48,
)
from projects.tfsn_consequence_value_iteration.antmaze import (
    r52_e549_preparation as r52,
)
from projects.tfsn_consequence_value_iteration.antmaze import (
    r53_e549_r51_evaluation as r53,
)
from projects.tfsn_consequence_value_iteration.antmaze import (
    r67_e549_hub_routed_exploration as r67,
)
from projects.tfsn_consequence_value_iteration.antmaze import (
    r68_e549_dynamic_hubs as r68,
)
from projects.tfsn_consequence_value_iteration.antmaze import (
    r69_e549_dynamic_hub_long_budget as r69,
)
from projects.tfsn_consequence_value_iteration.antmaze import (
    r70_corrected_hub_reachability as r70,
)
from projects.tfsn_consequence_value_iteration.antmaze import (
    r71_sigma_generalized_correction as r71,
)
from projects.tfsn_consequence_value_iteration.shared.settings import (
    RECORDED_EVIDENCE_PRIOR_WEIGHT,
)


HERE = Path(__file__).resolve().parent
MANAGEMENT_ROOT = HERE.parents[2]
REFERENCE = HERE / "R72_HUB_ROUTE_CORRECTION_REFERENCE.md"
PROTOCOL = HERE / "r72_hub_route_correction_protocol.json"
METHOD_REVISION = "antmaze_e549_5m_hub_route_online_correction_r72"
HORIZON = r71.HORIZON
EPISODES = r71.EPISODES
DEFAULT_OUTPUT = (
    MANAGEMENT_ROOT
    / "projects/tfsn_consequence_value_iteration/results/antmaze"
    / "e549_5m_hub_route_online_correction_r72/original9_long_budget"
)
PLOT_TITLE = (
    "R72 E549-5M hub-route correction — 20 commands × 10 episodes"
)
PLOT_STEM = "r72_hub_route_correction_original9"
EQUIVALENCE_FILE = "computational_equivalence.json"
EQUIVALENCE_COVERAGE_ATOL = 1e-11
EQUIVALENCE_REACH_ATOL = 2e-6


def load_protocol() -> dict:
    value = json.loads(PROTOCOL.read_text())
    expected = {
        "method_revision": METHOD_REVISION,
        "base_preparation_revision": r52.METHOD_REVISION,
        "base_evaluation_revision": r71.METHOD_REVISION,
        "task_ids": list(range(9)),
        "episodes": EPISODES,
        "commands_per_episode": HORIZON,
        "primitive_steps_per_episode": HORIZON * r48.COMMAND_STEPS,
        "fixed_command_duration": r48.COMMAND_STEPS,
        "evaluation_seed": 607,
        "hub_route_outcome": (
            "binary_arrival_within_landing_sigma_before_the_selected_"
            "attempt_horizon_expires"
        ),
        "hub_route_failure": (
            "physical_episode_ends_before_arrival_and_before_task_contact"
        ),
        "hub_route_censoring": "task_contact_before_hub_arrival",
        "hub_route_similarity": (
            "continuous_gaussian_over_attempt_source_xy_and_hub_xy_each_"
            "scaled_by_deployed_landing_sigma_with_exact_remaining_horizon"
        ),
        "hub_route_similarity_cutoff": None,
        "hub_route_prior_weight": float(RECORDED_EVIDENCE_PRIOR_WEIGHT),
        "hub_route_prior": (
            "current_R71_link_corrected_reachability_counted_as_one_"
            "observation"
        ),
        "reachability_computational_backend": (
            "exact_batched_rolling_torch"
        ),
        "coverage_computational_backend": "exact_chunked_torch",
        "computational_target_batch_source": (
            "declared_R69_hub_scoring_chunk_64"
        ),
        "bridge_evidence_computational_backend": "unchanged_R71_numpy",
        "corrected_hub_selection": (
            "exact_global_argmax_by_coverage_upper_bound_branch_and_bound"
        ),
        "hub_evaluation_cutoff": None,
        "hidden_goal_visible_to_planner_executor_or_models": False,
        "success_radius_read_by_planner": False,
        "task_gradient_updates": 0,
        "confirmation_panel_accessed": False,
    }
    for key, expected_value in expected.items():
        if value.get(key) != expected_value:
            raise ValueError(f"R72 protocol has another {key}")
    reference = Path(value["method_reference"])
    if not reference.is_absolute():
        reference = MANAGEMENT_ROOT / reference
    if reference.resolve() != REFERENCE.resolve():
        raise ValueError("R72 method reference changed")
    return value


PROTOCOL_VALUE = load_protocol()
_LATENT_NUMPY_COVERAGE = r68._all_hub_coverage


def hub_route_similarity(
    source_xy, hub_xy, observed_source_xy, observed_hub_xy, sigma,
) -> np.ndarray:
    """Continuous spatial similarity between planned route outcomes."""
    sigma = float(sigma)
    if not np.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("landing sigma must be positive and finite")
    source_distance = np.square(
        (
            np.asarray(source_xy, np.float64)
            - np.asarray(observed_source_xy, np.float64)
        ) / sigma
    ).sum(axis=-1)
    hub_distance = np.square(
        (
            np.asarray(hub_xy, np.float64)
            - np.asarray(observed_hub_xy, np.float64)
        ) / sigma
    ).sum(axis=-1)
    return np.exp(-0.5 * (source_distance + hub_distance))


def correct_hub_route_probability(
    prior_probability, support, arrival_mass,
) -> np.ndarray:
    """Bernoulli posterior with the shared one-observation model prior."""
    prior = float(RECORDED_EVIDENCE_PRIOR_WEIGHT)
    return np.clip(
        (
            prior * np.asarray(prior_probability, np.float64)
            + np.asarray(arrival_mass, np.float64)
        ) / (prior + np.asarray(support, np.float64)),
        0.0,
        1.0,
    )


def latent_hub_coverage_torch(
    posterior, hub_xy, device, chunk_size,
) -> np.ndarray:
    """Exact R68 latent-radius coverage in computational GPU chunks."""
    posterior.assert_not_exhausted("before R72 batched hub scoring")
    hubs = np.asarray(hub_xy, np.float64).reshape(-1, 2)
    device = torch.device(device)
    goal = torch.as_tensor(
        posterior.goal_xy, dtype=torch.float64, device=device
    )
    probability = torch.as_tensor(
        posterior.goal_probability, dtype=torch.float64, device=device
    )
    lower = torch.as_tensor(
        posterior.lower, dtype=torch.float64, device=device
    )
    upper = torch.as_tensor(
        posterior.upper, dtype=torch.float64, device=device
    )
    feasible = torch.as_tensor(
        posterior.feasible, dtype=torch.bool, device=device
    )
    log_width = torch.zeros_like(lower)
    log_width[feasible] = torch.log(upper[feasible] / lower[feasible])
    output = np.empty(len(hubs), np.float64)
    with torch.inference_mode():
        for begin in range(0, len(hubs), int(chunk_size)):
            stop = min(begin + int(chunk_size), len(hubs))
            target = torch.as_tensor(
                hubs[begin:stop], dtype=torch.float64, device=device
            )
            distance = torch.linalg.vector_norm(
                goal[:, None] - target[None], dim=2
            )
            certain = feasible[:, None] & (distance <= lower[:, None])
            possible = (
                feasible[:, None]
                & (distance > lower[:, None])
                & (distance < upper[:, None])
            )
            conditional = torch.zeros_like(distance)
            conditional[certain] = 1.0
            row, column = torch.nonzero(possible, as_tuple=True)
            conditional[row, column] = (
                torch.log(upper[row] / distance[row, column])
                / log_width[row]
            )
            output[begin:stop] = (
                probability @ conditional
            ).detach().cpu().numpy()
    return np.clip(output, 0.0, 1.0)


class ExactBatchedReachability:
    """Exact multi-target Bellman recursion with two rolling layers."""

    def __init__(
        self, source, endpoint, edge_weight, bridge_target, device,
        target_batch,
    ):
        self.device = torch.device(device)
        self.target_batch = int(target_batch)
        if self.target_batch <= 0:
            raise ValueError("target batch must be positive")
        self.source = torch.as_tensor(
            source, dtype=torch.long, device=self.device
        )
        self.endpoint = torch.as_tensor(
            endpoint, dtype=torch.long, device=self.device
        )
        self.edge_weight = torch.as_tensor(
            edge_weight, dtype=torch.float32, device=self.device
        )
        target = np.asarray(bridge_target, np.int64)
        self.node_count, self.bridge_count = target.shape
        self.bridge_target = torch.as_tensor(
            target.reshape(-1), dtype=torch.long, device=self.device
        )
        self.bridge_weight = None
        self.bridge_stay_weight = None

    def set_bridge_weight(self, bridge_weight) -> None:
        weight = torch.as_tensor(
            np.asarray(bridge_weight, np.float32),
            dtype=torch.float32,
            device=self.device,
        )
        if tuple(weight.shape) != (self.node_count, self.bridge_count):
            raise ValueError("batched bridge weight shape changed")
        self.bridge_weight = weight
        self.bridge_stay_weight = 1.0 - weight

    def solve(
        self, targets, horizon, current_node,
    ) -> np.ndarray:
        if self.bridge_weight is None:
            raise RuntimeError("set bridge weights before batched solving")
        targets = np.asarray(targets, np.int64).reshape(-1)
        if len(targets) > self.target_batch:
            raise ValueError("target batch exceeds the declared chunk")
        if not len(targets):
            return np.empty(0, np.float32)
        target = torch.as_tensor(
            targets, dtype=torch.long, device=self.device
        )
        batch = len(targets)
        row = torch.arange(batch, dtype=torch.long, device=self.device)
        previous = torch.zeros(
            (batch, self.node_count),
            dtype=torch.float32,
            device=self.device,
        )
        previous[row, target] = 1.0
        source_index = self.source.unsqueeze(0).expand(batch, -1)
        with torch.inference_mode():
            for _remaining in range(1, int(horizon) + 1):
                destination = previous.index_select(
                    1, self.bridge_target
                ).reshape(batch, self.node_count, self.bridge_count)
                destination.mul_(self.bridge_weight.unsqueeze(0))
                destination.add_(
                    previous.unsqueeze(2)
                    * self.bridge_stay_weight.unsqueeze(0)
                )
                best = destination.amax(dim=2)
                recorded = (
                    self.edge_weight.unsqueeze(0)
                    * previous.index_select(1, self.endpoint)
                    + (1.0 - self.edge_weight).unsqueeze(0)
                    * previous.index_select(1, self.source)
                )
                best.scatter_reduce_(
                    1,
                    source_index,
                    recorded,
                    reduce="amax",
                    include_self=True,
                )
                best[row, target] = 1.0
                previous = best
        return (
            previous[:, int(current_node)].detach().cpu().numpy()
            .astype(np.float32, copy=False)
        )


def _array_sha256(value: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(value)
    return hashlib.sha256(contiguous.view(np.uint8)).hexdigest()


def _scalar_branch_and_bound_selection(
    coverage,
    hub_nodes,
    remaining,
    current_node,
    node_count,
    source,
    endpoint,
    edge_weight,
    bridge_target,
    bridge_weight,
    route_support,
    route_arrivals,
) -> dict:
    """The pre-optimization R72 selection, retained for exact auditing."""
    coverage = np.asarray(coverage, np.float64)
    order = np.argsort(-coverage, kind="stable")
    selected = int(order[0])
    best_score = -1.0
    evaluated_indices = []
    reaches = []
    for index in order:
        if best_score >= float(coverage[index]):
            break
        base = r67._one_target_layers(
            int(hub_nodes[index]),
            int(remaining),
            int(node_count),
            source,
            endpoint,
            edge_weight,
            bridge_target,
            bridge_weight,
        )[int(remaining), int(current_node)]
        corrected = float(correct_hub_route_probability(
            np.asarray([base]),
            np.asarray([route_support[index]]),
            np.asarray([route_arrivals[index]]),
        )[0])
        score = float(coverage[index]) * corrected
        evaluated_indices.append(int(index))
        reaches.append(float(base))
        if score > best_score:
            best_score = score
            selected = int(index)
    return {
        "selected": selected,
        "score": float(best_score),
        "evaluated_indices": np.asarray(evaluated_indices, np.int64),
        "reaches": np.asarray(reaches, np.float32),
    }


def _batched_branch_and_bound_selection(
    coverage,
    hub_nodes,
    remaining,
    current_node,
    solver,
    bridge_weight,
    route_support,
    route_arrivals,
) -> dict:
    """Batched realization of the same ordered exact selection."""
    coverage = np.asarray(coverage, np.float64)
    order = np.argsort(-coverage, kind="stable")
    selected = int(order[0])
    best_score = -1.0
    evaluated_indices = []
    reaches = []
    solver.set_bridge_weight(bridge_weight)
    stop = False
    for begin in range(0, len(order), solver.target_batch):
        if best_score >= float(coverage[order[begin]]):
            break
        indices = order[begin:begin + solver.target_batch]
        base_values = solver.solve(
            np.asarray(hub_nodes)[indices], remaining, current_node
        )
        for index, base in zip(indices, base_values):
            if best_score >= float(coverage[index]):
                stop = True
                break
            corrected = float(correct_hub_route_probability(
                np.asarray([base]),
                np.asarray([route_support[index]]),
                np.asarray([route_arrivals[index]]),
            )[0])
            score = float(coverage[index]) * corrected
            evaluated_indices.append(int(index))
            reaches.append(float(base))
            if score > best_score:
                best_score = score
                selected = int(index)
        if stop:
            break
    return {
        "selected": selected,
        "score": float(best_score),
        "evaluated_indices": np.asarray(evaluated_indices, np.int64),
        "reaches": np.asarray(reaches, np.float32),
    }


class HubRouteAttemptObserver:
    """Lifecycle and task-local evidence for selected hub pursuits."""

    def __init__(self):
        self.reset_task()

    def reset_task(self) -> None:
        self.route_sources = []
        self.route_hubs = []
        self.route_remaining = []
        self.route_outcomes = []
        self.route_records = []
        self.censored_records = []
        self.active_attempt = None

    def start_attempt(
        self,
        source_xy,
        hub_bank_index,
        hub_xy,
        remaining_commands,
        predicted_reachability,
    ) -> dict:
        if self.active_attempt is not None:
            raise RuntimeError("a hub-route attempt is already active")
        self.active_attempt = {
            "source_xy": np.asarray(source_xy, np.float32).tolist(),
            "hub_bank_index": int(hub_bank_index),
            "hub_xy": np.asarray(hub_xy, np.float32).tolist(),
            "remaining_commands": int(remaining_commands),
            "predicted_reachability": float(predicted_reachability),
        }
        return dict(self.active_attempt)

    def finish_attempt(self, arrived, endpoint_xy) -> dict:
        if self.active_attempt is None:
            raise RuntimeError("no active hub-route attempt to finish")
        record = {
            **self.active_attempt,
            "arrived": bool(arrived),
            "endpoint_xy": np.asarray(endpoint_xy, np.float32).tolist(),
            "censored": False,
        }
        self.route_sources.append(record["source_xy"])
        self.route_hubs.append(record["hub_xy"])
        self.route_remaining.append(record["remaining_commands"])
        self.route_outcomes.append(float(record["arrived"]))
        self.route_records.append(record)
        self.active_attempt = None
        return dict(record)

    def cancel_attempt(self, reason) -> dict:
        if self.active_attempt is None:
            raise RuntimeError("no active hub-route attempt to cancel")
        record = {
            **self.active_attempt,
            "arrived": None,
            "endpoint_xy": None,
            "censored": True,
            "censoring_reason": str(reason),
        }
        self.censored_records.append(record)
        self.active_attempt = None
        return dict(record)

    def summary(self) -> dict:
        outcomes = np.asarray(self.route_outcomes, np.float32)
        return {
            "hub_route_attempts": int(len(outcomes)),
            "hub_route_arrivals": int(outcomes.sum()) if len(outcomes) else 0,
            "hub_route_failures": (
                int(len(outcomes) - outcomes.sum()) if len(outcomes) else 0
            ),
            "hub_route_arrival_rate": (
                float(outcomes.mean()) if len(outcomes) else None
            ),
            "hub_route_censored_attempts": int(len(self.censored_records)),
        }


class HubRouteCorrectedScorer(
    r71.SigmaGeneralizedHubScorer, HubRouteAttemptObserver
):
    """R71 link correction plus route-level correction of hub outcomes."""

    def __init__(self, planner, arrays, hub_nodes, device="cuda"):
        r71.SigmaGeneralizedHubScorer.__init__(
            self, planner, arrays, hub_nodes
        )
        HubRouteAttemptObserver.__init__(self)
        self.computational_batch = int(
            r68.PROTOCOL_VALUE["hub_scoring_chunk_computational_only"]
        )
        self.optimized_latent_backend = bool(
            PROTOCOL_VALUE.get("reachability_computational_backend")
            == "exact_batched_rolling_torch"
            and PROTOCOL_VALUE.get("coverage_computational_backend")
            == "exact_chunked_torch"
        )
        self.computational_device = None
        self.batched_reachability = None
        if self.optimized_latent_backend:
            requested = torch.device(device)
            if requested.type == "cuda" and not torch.cuda.is_available():
                raise RuntimeError(
                    "R72 exact batched backend requires CUDA; run the "
                    "computational-equivalence audit on the evaluation host"
                )
            self.computational_device = requested
            self.batched_reachability = ExactBatchedReachability(
                self.source,
                self.endpoint,
                self.edge_weight,
                self.bridge_target,
                requested,
                self.computational_batch,
            )

    def reset_task(self) -> None:
        HubRouteAttemptObserver.reset_task(self)
        if hasattr(self, "support"):
            self.support.fill(0.0)
            self.outcome_mass.fill(0.0)
            self.processed_observations = 0

    def _route_mass(self, source_xy, hub_xy, remaining):
        support = np.zeros(len(hub_xy), np.float64)
        arrivals = np.zeros(len(hub_xy), np.float64)
        if not self.route_sources:
            return support, arrivals
        sources = np.asarray(self.route_sources, np.float64)
        hubs = np.asarray(self.route_hubs, np.float64)
        horizons = np.asarray(self.route_remaining, np.int32)
        outcomes = np.asarray(self.route_outcomes, np.float64)
        same_event = horizons == int(remaining)
        if not same_event.any():
            return support, arrivals
        sources = sources[same_event]
        hubs = hubs[same_event]
        outcomes = outcomes[same_event]
        source_distance = np.square(
            (
                sources - np.asarray(source_xy, np.float64)[None]
            ) / float(r48.SIGMA)
        ).sum(axis=1)
        hub_distance = np.square(
            (
                hubs[:, None] - np.asarray(hub_xy, np.float64)[None]
            ) / float(r48.SIGMA)
        ).sum(axis=2)
        similarity = np.exp(
            -0.5 * (source_distance[:, None] + hub_distance)
        )
        support = similarity.sum(axis=0)
        arrivals = (similarity * outcomes[:, None]).sum(axis=0)
        return support, arrivals

    def __call__(
        self, posterior, hub_xy, offline_reachability, graph, state,
        remaining, online_states,
    ):
        started = time.perf_counter()
        update_seconds = self._synchronize(online_states)
        if self.optimized_latent_backend:
            coverage = latent_hub_coverage_torch(
                posterior,
                hub_xy,
                self.computational_device,
                self.computational_batch,
            )
            coverage_backend = "exact_chunked_torch"
        else:
            # R73 intentionally remains on its matched fixed-sigma backend.
            coverage = r68._all_hub_coverage(posterior, hub_xy)
            coverage_backend = "legacy_domain_backend"
        current_node = int(graph.project(np.asarray(state)[None])[0][0])
        if len(self.hub_nodes) != len(hub_xy):
            raise ValueError("R72 hub-node contract changed")
        route_support, route_arrivals = self._route_mass(
            np.asarray(state)[:2], hub_xy, remaining
        )

        link_reach = np.zeros(len(hub_xy), np.float64)
        corrected_reach = np.zeros(len(hub_xy), np.float64)
        score = np.zeros(len(hub_xy), np.float64)
        solver_seconds = 0.0
        if self.processed_observations == 0:
            link_reach = np.asarray(
                offline_reachability[:, int(remaining), current_node],
                np.float64,
            )
            corrected_reach = correct_hub_route_probability(
                link_reach, route_support, route_arrivals
            )
            score = coverage * corrected_reach
            selected = int(np.argmax(score))
            evaluated = 0
            score_semantics = (
                "complete_exact_cached_R71_link_reach_with_route_posterior"
            )
        else:
            corrected_weight = r70.corrected_bridge_weights(
                self.bridge_weight, self.support, self.outcome_mass
            )
            solve_started = time.perf_counter()
            if self.optimized_latent_backend:
                decision = _batched_branch_and_bound_selection(
                    coverage,
                    self.hub_nodes,
                    int(remaining),
                    current_node,
                    self.batched_reachability,
                    corrected_weight,
                    route_support,
                    route_arrivals,
                )
                reachability_backend = "exact_batched_rolling_torch"
            else:
                decision = _scalar_branch_and_bound_selection(
                    coverage,
                    self.hub_nodes,
                    int(remaining),
                    current_node,
                    len(self.nodes),
                    self.source,
                    self.endpoint,
                    self.edge_weight,
                    self.bridge_target,
                    corrected_weight,
                    route_support,
                    route_arrivals,
                )
                reachability_backend = "legacy_scalar_numpy"
            selected = int(decision["selected"])
            indices = decision["evaluated_indices"]
            bases = decision["reaches"]
            link_reach[indices] = bases
            corrected_reach[indices] = correct_hub_route_probability(
                bases,
                route_support[indices],
                route_arrivals[indices],
            )
            score[indices] = coverage[indices] * corrected_reach[indices]
            evaluated = int(len(indices))
            solver_seconds = time.perf_counter() - solve_started
            score_semantics = (
                "evaluated_exact_R71_link_and_route_corrected_scores_else_"
                "zero_lower_bound;global_argmax_exact_by_coverage_upper_bound"
            )

        if self.processed_observations == 0:
            reachability_backend = "exact_cached_R69_table"

        offline_selected = float(
            offline_reachability[selected, int(remaining), current_node]
        )
        route_empirical = (
            float(route_arrivals[selected] / route_support[selected])
            if route_support[selected] > 0.0
            else None
        )
        self.planner.hub_selection_extra = {
            "offline_cached_reachability_same_hub": offline_selected,
            "sigma_generalized_corrected_reachability_same_hub": float(
                link_reach[selected]
            ),
            "hub_route_corrected_reachability_same_hub": float(
                corrected_reach[selected]
            ),
            "hub_route_evidence_support_same_hub": float(
                route_support[selected]
            ),
            "hub_route_evidence_arrival_mass_same_hub": float(
                route_arrivals[selected]
            ),
            "hub_route_empirical_arrival_same_hub": route_empirical,
            "hub_route_prior_weight": float(
                RECORDED_EVIDENCE_PRIOR_WEIGHT
            ),
            "hub_route_observations_used": int(len(self.route_outcomes)),
            "hub_route_similarity": PROTOCOL_VALUE["hub_route_similarity"],
            "hub_route_similarity_cutoff": None,
            "corrected_hubs_evaluated": int(evaluated),
            "corrected_hub_candidates_total": int(len(hub_xy)),
            "branch_and_bound_rule": (
                "stop_only_when_next_coverage_upper_bound_le_best_exact_score"
            ),
            "hub_evaluation_cutoff": None,
            "score_vector_semantics": score_semantics,
            "online_observations_used_by_hub_scorer": int(
                self.processed_observations
            ),
            "online_bridge_update_seconds": float(update_seconds),
            "corrected_reachability_solve_seconds": float(solver_seconds),
            "coverage_computational_backend": coverage_backend,
            "reachability_computational_backend": reachability_backend,
            "computational_target_batch": int(self.computational_batch),
            "computational_device": (
                str(self.computational_device)
                if self.computational_device is not None else None
            ),
        }
        return (
            selected,
            coverage,
            corrected_reach,
            score,
            current_node,
            time.perf_counter() - started,
        )


def _bind_r72_contract() -> None:
    r67.REFERENCE = REFERENCE
    r67.PROTOCOL = PROTOCOL
    r67.METHOD_REVISION = METHOD_REVISION
    r67.PROTOCOL_VALUE = PROTOCOL_VALUE
    r67._score_hubs = r68._score_all_hubs


def _r69_args(args):
    value = copy.copy(args)
    value.output = r69.DEFAULT_OUTPUT
    return value


def _jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line) for line in path.read_text().splitlines()
        if line.strip()
    ]


def _reconstruct_link_evidence(
    logs, count, nodes, bridge_target_xy, sigma,
) -> tuple[np.ndarray, np.ndarray]:
    support = np.zeros(bridge_target_xy.shape[:2], np.float32)
    outcome_mass = np.zeros_like(support)
    used = 0
    for row in logs:
        if used >= int(count):
            break
        destination = row.get("selected_link_destination_xy")
        if destination is None:
            raise ValueError("saved R72 command lacks its graph destination")
        endpoint = np.asarray(row["actual_endpoint"], np.float32)
        destination = np.asarray(destination, np.float32)
        outcome = float(np.linalg.norm(endpoint - destination) <= sigma)
        r71.add_sigma_observation_mass(
            support,
            outcome_mass,
            nodes,
            bridge_target_xy,
            np.asarray(row["start_xy"], np.float32),
            np.asarray(row["command"], np.float32),
            destination,
            outcome,
            sigma,
        )
        used += 1
    if used != int(count):
        raise ValueError("saved R72 log has too few online observations")
    return support, outcome_mass


def _saved_route_mass(logs, stop, source_xy, hub_xy, remaining, sigma):
    support = np.zeros(len(hub_xy), np.float64)
    arrivals = np.zeros(len(hub_xy), np.float64)
    for row in logs[:int(stop)]:
        record = row.get("hub_route_attempt_outcome")
        if not record or record.get("censored"):
            continue
        if int(record["remaining_commands"]) != int(remaining):
            continue
        similarity = hub_route_similarity(
            source_xy,
            hub_xy,
            np.asarray(record["source_xy"], np.float64),
            np.asarray(record["hub_xy"], np.float64),
            sigma,
        )
        support += similarity
        arrivals += similarity * float(record["arrived"])
    return support, arrivals


def _equivalence_report_path(args) -> Path:
    return Path(args.output).resolve() / EQUIVALENCE_FILE


def _valid_equivalence_report(args) -> tuple[bool, dict | None]:
    path = _equivalence_report_path(args)
    if not path.is_file():
        return False, None
    report = json.loads(path.read_text())
    valid = bool(
        report.get("passed") is True
        and report.get("method_revision") == METHOD_REVISION
        and report.get("implementation_sha256")
        == r52.sha256(Path(__file__))
        and report.get("method_reference_sha256") == r52.sha256(REFERENCE)
    )
    return valid, report


def command_equivalence(args) -> dict:
    """Gate the optimized latent backend against saved scalar decisions."""
    valid, saved = _valid_equivalence_report(args)
    if valid:
        print(json.dumps(saved, indent=2))
        return saved
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError(
            "R72 computational equivalence must exercise CUDA on the "
            "evaluation host"
        )
    table_args = _r69_args(args)
    table_report = r69._audit(table_args)
    if not table_report["ready_for_evaluation"]:
        raise FileNotFoundError("R69 horizon table is incomplete")
    contract, hub_nodes, hub_xy, arrays = r68._candidate_contract(
        table_args
    )
    artifact = r68._artifact_paths(r69.DEFAULT_OUTPUT)
    r68._validate_metadata(artifact["metadata"], contract)
    reachability = np.load(artifact["table"], mmap_mode="r")
    (
        _paths, consequence, nodes, source, endpoint, edge_weight,
        bridge_target, bridge_weight,
    ) = arrays
    sigma = float(consequence["landing_sigma_m"])
    batch = int(r68.PROTOCOL_VALUE[
        "hub_scoring_chunk_computational_only"
    ])

    # Three saved R71 reset decisions validate the exact latent coverage
    # realization and the resulting cached-table argmax.
    reset_checks = []
    for task_id in (0, 4, 8):
        path = r71.DEFAULT_OUTPUT / f"task_{task_id:03d}.jsonl"
        first = next(
            row for row in _jsonl(path)
            if row.get("hub_selected_this_command")
        )
        posterior = r67._area_posterior(nodes, sigma)
        r67._negative_path_update(
            posterior, np.asarray(first["start_xy"])[None]
        )
        cpu_coverage = _LATENT_NUMPY_COVERAGE(posterior, hub_xy)
        gpu_coverage = latent_hub_coverage_torch(
            posterior, hub_xy, device, batch
        )
        current_node = int(first["hub_selection"]["current_graph_node"])
        remaining = int(first["remaining"])
        cpu_score = cpu_coverage * np.asarray(
            reachability[:, remaining, current_node], np.float64
        )
        gpu_score = gpu_coverage * np.asarray(
            reachability[:, remaining, current_node], np.float64
        )
        expected = int(first["hub_selection"]["hub_bank_index"])
        reset_checks.append({
            "source": str(path),
            "task_id": int(task_id),
            "saved_hub": expected,
            "scalar_hub": int(np.argmax(cpu_score)),
            "batched_hub": int(np.argmax(gpu_score)),
            "coverage_max_abs_difference": float(
                np.max(np.abs(cpu_coverage - gpu_coverage))
            ),
        })

    # Replay a saved scalar R72 selection after nine physical commands.  This
    # reconstructs the actual latent posterior from all primitive states and
    # the exact R71 bridge-evidence arrays from the command log.
    replay_task = 1
    replay_index = 9
    replay_path = DEFAULT_OUTPUT / f"task_{replay_task:03d}.jsonl"
    logs = _jsonl(replay_path)
    saved_selection = logs[replay_index]
    selection = saved_selection["hub_selection"]
    evidence_count = int(
        selection["online_observations_used_by_hub_scorer"]
    )
    posterior = r67._area_posterior(nodes, sigma)
    task = r48.r31.r22._task(args.root, replay_task)
    goal = np.asarray(task["goal"], np.float32)
    success_radius = float(
        r48.PROTOCOL_VALUE["evaluator"][
            "success_radius_m_evaluator_only"
        ]
    )
    r52.configure_r52_runtime(args.preparation)
    executor = r53.E549SACExecutor(r53.E549_CHECKPOINT, args.device)
    env = r48.make_env(max_episode_steps=10_000_000)
    endpoint_errors = []
    try:
        observation, _ = env.reset(seed=int(task["reset_seed"]))
        state = r48.full_state(observation)
        r67._negative_path_update(posterior, state[:2][None])
        for row in logs[:replay_index]:
            trajectory, rewards = r48._execute_evaluator_command(
                env,
                executor,
                state,
                np.asarray(row["command"], np.float32),
                goal,
                success_radius,
            )
            endpoint_errors.append(float(np.linalg.norm(
                trajectory[-1, :2]
                - np.asarray(row["actual_endpoint"], np.float32)
            )))
            positive = np.flatnonzero(rewards)
            negative_stop = (
                int(positive[0]) if len(positive) else len(rewards)
            )
            r67._negative_path_update(
                posterior, trajectory[1:1 + negative_stop, :2]
            )
            state = trajectory[-1]
    finally:
        env.close()
    replay_start_error = float(np.linalg.norm(
        state[:2] - np.asarray(saved_selection["start_xy"], np.float32)
    ))
    cpu_coverage = _LATENT_NUMPY_COVERAGE(posterior, hub_xy)
    gpu_coverage = latent_hub_coverage_torch(
        posterior, hub_xy, device, batch
    )
    bridge_target_xy = nodes[bridge_target, :2]
    support, outcome_mass = _reconstruct_link_evidence(
        logs, evidence_count, nodes, bridge_target_xy, sigma
    )
    corrected_weight = r70.corrected_bridge_weights(
        bridge_weight, support, outcome_mass
    )
    remaining = int(saved_selection["remaining"])
    current_node = int(selection["current_graph_node"])
    route_support, route_arrivals = _saved_route_mass(
        logs,
        replay_index,
        state[:2],
        hub_xy,
        remaining,
        sigma,
    )
    scalar = _scalar_branch_and_bound_selection(
        cpu_coverage,
        hub_nodes,
        remaining,
        current_node,
        len(nodes),
        source,
        endpoint,
        edge_weight,
        bridge_target,
        corrected_weight,
        route_support,
        route_arrivals,
    )
    solver = ExactBatchedReachability(
        source,
        endpoint,
        edge_weight,
        bridge_target,
        device,
        batch,
    )
    batched = _batched_branch_and_bound_selection(
        gpu_coverage,
        hub_nodes,
        remaining,
        current_node,
        solver,
        corrected_weight,
        route_support,
        route_arrivals,
    )
    same_evaluation_order = np.array_equal(
        scalar["evaluated_indices"], batched["evaluated_indices"]
    )
    reach_max_abs = (
        float(np.max(np.abs(scalar["reaches"] - batched["reaches"])))
        if same_evaluation_order and len(scalar["reaches"]) else None
    )
    saved_hub = int(selection["hub_bank_index"])
    full_replay = {
        "source": str(replay_path),
        "task_id": replay_task,
        "saved_log_row": replay_index,
        "saved_hub": saved_hub,
        "scalar_hub": int(scalar["selected"]),
        "batched_hub": int(batched["selected"]),
        "scalar_hubs_evaluated": int(len(scalar["evaluated_indices"])),
        "batched_hubs_evaluated": int(len(batched["evaluated_indices"])),
        "same_evaluation_order": bool(same_evaluation_order),
        "coverage_max_abs_difference": float(
            np.max(np.abs(cpu_coverage - gpu_coverage))
        ),
        "reachability_max_abs_difference": reach_max_abs,
        "score_abs_difference": float(abs(
            scalar["score"] - batched["score"]
        )),
        "maximum_replay_endpoint_error_m": float(max(endpoint_errors)),
        "selection_start_replay_error_m": replay_start_error,
        "bridge_evidence_backend": "shared_unchanged_R71_numpy_arrays",
        "bridge_support_sha256": _array_sha256(support),
        "bridge_outcome_mass_sha256": _array_sha256(outcome_mass),
        "corrected_bridge_weight_sha256": _array_sha256(corrected_weight),
    }
    reset_passed = all(
        item["saved_hub"] == item["scalar_hub"]
        == item["batched_hub"]
        and item["coverage_max_abs_difference"]
        <= EQUIVALENCE_COVERAGE_ATOL
        for item in reset_checks
    )
    replay_passed = bool(
        full_replay["saved_hub"] == full_replay["scalar_hub"]
        == full_replay["batched_hub"]
        and full_replay["same_evaluation_order"]
        and full_replay["coverage_max_abs_difference"]
        <= EQUIVALENCE_COVERAGE_ATOL
        and full_replay["reachability_max_abs_difference"] is not None
        and full_replay["reachability_max_abs_difference"]
        <= EQUIVALENCE_REACH_ATOL
        and full_replay["maximum_replay_endpoint_error_m"] <= 1e-5
        and full_replay["selection_start_replay_error_m"] <= 1e-5
    )
    report = {
        "schema_version": 1,
        "method_revision": METHOD_REVISION,
        "method_reference": str(REFERENCE),
        "method_reference_sha256": r52.sha256(REFERENCE),
        "implementation": str(Path(__file__).resolve()),
        "implementation_sha256": r52.sha256(Path(__file__)),
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device),
        "target_batch": batch,
        "coverage_absolute_tolerance": EQUIVALENCE_COVERAGE_ATOL,
        "reachability_absolute_tolerance": EQUIVALENCE_REACH_ATOL,
        "saved_R71_reset_selection_checks": reset_checks,
        "saved_R72_full_replay_check": full_replay,
        "same_hub_selected_every_time": bool(
            reset_passed
            and full_replay["saved_hub"] == full_replay["scalar_hub"]
            == full_replay["batched_hub"]
        ),
        "bridge_evidence_spread_identical": True,
        "exact_global_argmax_retained": True,
        "shortlist_used": False,
        "passed": bool(reset_passed and replay_passed),
    }
    path = _equivalence_report_path(args)
    path.parent.mkdir(parents=True, exist_ok=True)
    r48.dump(path, report)
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise RuntimeError("R72 computational equivalence gate failed")
    return report


def command_audit(args) -> dict:
    table_report = r69._audit(_r69_args(args))
    consequence = r67._graph_arrays(args.preparation)[1]
    paths = r52.output_paths(args.preparation)
    required = [
        paths["consequence"],
        paths["graph"],
        paths["runtime"] / "planner/bridges.npz",
        r68._artifact_paths(r69.DEFAULT_OUTPUT)["table"],
        r68._artifact_paths(r69.DEFAULT_OUTPUT)["metadata"],
    ]
    optimized_latent_backend = bool(
        PROTOCOL_VALUE.get("reachability_computational_backend")
        == "exact_batched_rolling_torch"
    )
    equivalence_passed, equivalence = _valid_equivalence_report(args)
    if not optimized_latent_backend:
        equivalence_passed = True
    report = {
        "schema_version": 1,
        "method_reference": str(REFERENCE),
        "method_reference_sha256": r52.sha256(REFERENCE),
        "method_revision": METHOD_REVISION,
        "claim_level": PROTOCOL_VALUE["claim_level"],
        "declared_deviations": PROTOCOL_VALUE["declared_deviations"],
        "oracle_inputs": [],
        "task_ids": PROTOCOL_VALUE["task_ids"],
        "episodes": EPISODES,
        "commands_per_episode": HORIZON,
        "primitive_steps_per_episode": HORIZON * r48.COMMAND_STEPS,
        "landing_sigma_m": float(consequence["landing_sigma_m"]),
        "hub_route_outcome": PROTOCOL_VALUE["hub_route_outcome"],
        "hub_route_prior_weight": float(
            RECORDED_EVIDENCE_PRIOR_WEIGHT
        ),
        "hub_route_similarity": PROTOCOL_VALUE["hub_route_similarity"],
        "hub_route_similarity_cutoff": None,
        "hub_evaluation_cutoff": None,
        "reachability_computational_backend": PROTOCOL_VALUE.get(
            "reachability_computational_backend", "legacy_domain_backend"
        ),
        "coverage_computational_backend": PROTOCOL_VALUE.get(
            "coverage_computational_backend", "legacy_domain_backend"
        ),
        "computational_equivalence_required": optimized_latent_backend,
        "computational_equivalence_report": (
            str(_equivalence_report_path(args))
            if optimized_latent_backend else None
        ),
        "computational_equivalence_passed": bool(equivalence_passed),
        "computational_equivalence_summary": (
            {
                "same_hub_selected_every_time": equivalence.get(
                    "same_hub_selected_every_time"
                ),
                "bridge_evidence_spread_identical": equivalence.get(
                    "bridge_evidence_spread_identical"
                ),
            }
            if equivalence_passed and equivalence is not None else None
        ),
        "required_artifacts": [
            {"path": str(path), "exists": path.is_file()}
            for path in required
        ],
        "ready_for_evaluation": bool(
            table_report["ready_for_evaluation"]
            and all(path.is_file() for path in required)
            and equivalence_passed
        ),
        "hidden_goal_visible_to_planner_executor_or_models": False,
        "success_radius_read_by_planner": False,
        "task_gradient_updates": 0,
        "confirmation_panel_accessed": False,
    }
    print(json.dumps(report, indent=2))
    return report


def _load_runtime(args):
    loaded = list(r71._load_runtime(args))
    _bind_r72_contract()
    planner = loaded[0]
    planner.method_revision = METHOD_REVISION
    planner.planner_name = (
        "E549_R51_sigma_link_and_hub_route_outcome_correction"
    )
    scorer = HubRouteCorrectedScorer(
        planner, r67._graph_arrays(args.preparation), loaded[3], args.device
    )
    planner.hub_score_override = scorer
    planner.hub_attempt_observer = scorer
    loaded[0] = planner
    return tuple(loaded)


def _plot(output: Path, rows: list[dict]) -> dict:
    colors = r67.plt.get_cmap("tab10")(np.linspace(0.0, 0.9, EPISODES))
    figure, axes = r67.plt.subplots(
        3, 3, figsize=(17.0, 16.0), constrained_layout=True
    )
    for axis, task in zip(axes.ravel(), rows):
        r67.maze_plot.draw_maze(axis)
        logs = [
            json.loads(line)
            for line in (
                output / f"task_{int(task['task_id']):03d}.jsonl"
            ).read_text().splitlines()
            if line.strip()
        ]
        goal = np.asarray(task["goal"], np.float64)
        axis.scatter(*goal, marker="X", s=90, color="#b74343", zorder=12)
        for episode in range(EPISODES):
            episode_rows = [
                row for row in logs if int(row["episode"]) == episode
            ]
            if not episode_rows:
                continue
            color = colors[episode]
            previous = np.asarray(episode_rows[0]["start_xy"], np.float64)
            for row in episode_rows:
                endpoint = np.asarray(row["actual_endpoint"], np.float64)
                axis.plot(
                    [previous[0], endpoint[0]],
                    [previous[1], endpoint[1]],
                    color=color,
                    linestyle="-" if row["phase"] == "acquisition" else "--",
                    linewidth=1.15,
                    alpha=0.88,
                    zorder=3,
                )
                axis.scatter(*endpoint, s=8, color=color, zorder=5)
                if row.get("hub_selected_this_command"):
                    hub = np.asarray(row["active_hub_xy"], np.float64)
                    axis.scatter(
                        *hub, marker="D", s=38, facecolor="none",
                        edgecolor=color, linewidth=1.0, zorder=8,
                    )
                previous = endpoint
        contact = task["first_contact_episode"]
        status = f"contact E{contact + 1}" if contact is not None else "no contact"
        closest = task.get("closest_goal_distance_m")
        closest_text = "n/a" if closest is None else f"{closest:.2f} m"
        route_rate = task.get("hub_route_arrival_rate")
        route_text = "n/a" if route_rate is None else f"{route_rate:.2f}"
        axis.set_title(
            f"T{int(task['task_id'])}: {status}; closest {closest_text}\n"
            f"{task['acquisition_new_1m_cells_per_command']:.2f} cells/cmd; "
            f"route arrival {route_text}",
            fontsize=9,
        )
        axis.set_xlabel("x (m)")
        axis.set_ylabel("y (m)")
    handles = [
        Line2D([0], [0], color=colors[index], linewidth=2, label=f"E{index + 1}")
        for index in range(EPISODES)
    ]
    handles.extend([
        Line2D([0], [0], color="black", linestyle="-", label="acquisition"),
        Line2D([0], [0], color="black", linestyle="--", label="return"),
        Line2D(
            [0], [0], marker="D", markerfacecolor="none",
            markeredgecolor="black", linestyle="none", label="selected hub",
        ),
    ])
    figure.legend(
        handles=handles, loc="lower center", ncol=13,
        bbox_to_anchor=(0.5, -0.01), fontsize=8,
    )
    figure.suptitle(PLOT_TITLE, fontsize=14, fontweight="bold")
    stem = output / PLOT_STEM
    figure.savefig(stem.with_suffix(".png"), dpi=220, bbox_inches="tight")
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    r67.plt.close(figure)
    return {
        "png": str(stem.with_suffix(".png")),
        "pdf": str(stem.with_suffix(".pdf")),
    }


def command_evaluate(args) -> dict:
    if int(args.episodes) != EPISODES:
        raise ValueError("R72 requires ten episodes per development task")
    if int(args.seed) != int(PROTOCOL_VALUE["evaluation_seed"]):
        raise ValueError("R72 requires evaluation seed 607")
    audit = command_audit(args)
    if not audit["ready_for_evaluation"]:
        raise FileNotFoundError(
            "R52/R69 artifacts are incomplete or the R72 exact "
            "computational-equivalence gate has not passed"
        )
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    summary_path = output / "summary.json"
    if summary_path.is_file():
        saved = json.loads(summary_path.read_text())
        if saved.get("method_revision") != METHOD_REVISION:
            raise ValueError("saved R72 summary has another method")
        print(json.dumps(saved, indent=2))
        return saved
    optimized_latent_backend = bool(
        PROTOCOL_VALUE.get("reachability_computational_backend")
        == "exact_batched_rolling_torch"
    )
    pre_optimization_scalar_task_ids = (
        [
            int(task_id) for task_id in PROTOCOL_VALUE["task_ids"]
            if (output / f"task_{int(task_id):03d}.json").is_file()
        ]
        if optimized_latent_backend else []
    )

    (
        planner, executor, env, hub_nodes, hub_xy, nodes, reachability,
    ) = _load_runtime(args)
    rows = []
    try:
        for task_id in PROTOCOL_VALUE["task_ids"]:
            row = r67._evaluate_task(
                args, planner, executor, env, int(task_id), output,
                hub_nodes, hub_xy, reachability, nodes, float(r48.SIGMA),
            )
            log_path = output / f"task_{int(task_id):03d}.jsonl"
            row = r68._normalize_task_record(row, log_path, hub_nodes)
            row["closest_goal_distance_m"] = r69._closest_goal_distance(
                log_path, row["goal"]
            )
            r48.dump(log_path.with_suffix(".json"), row)
            rows.append(row)
    finally:
        env.close()

    contacts = [row["first_contact_episode"] for row in rows]
    returns = [
        success
        for row in rows
        if row["first_contact_episode"] is not None
        for success in row["successes"][row["first_contact_episode"] + 1 :]
    ]
    figures = _plot(output, rows)
    total_route_attempts = sum(row["hub_route_attempts"] for row in rows)
    total_route_arrivals = sum(row["hub_route_arrivals"] for row in rows)
    result = {
        "schema_version": 1,
        "experiment_id": PROTOCOL_VALUE["experiment_id"],
        "method_reference": str(REFERENCE),
        "method_reference_sha256": r52.sha256(REFERENCE),
        "method_revision": METHOD_REVISION,
        "claim_level": PROTOCOL_VALUE["claim_level"],
        "declared_deviations": PROTOCOL_VALUE["declared_deviations"],
        "oracle_inputs": [],
        "tasks": len(rows),
        "task_ids": PROTOCOL_VALUE["task_ids"],
        "episodes_per_task": EPISODES,
        "commands_per_episode": HORIZON,
        "primitive_steps_per_episode": HORIZON * r48.COMMAND_STEPS,
        "acquired_tasks": int(sum(value is not None for value in contacts)),
        "acquisition_at_1": float(np.mean([value == 0 for value in contacts])),
        "acquisition_at_5": float(np.mean([
            value is not None and value < 5 for value in contacts
        ])),
        "acquisition_at_10": float(np.mean([
            value is not None for value in contacts
        ])),
        "pooled_conditional_return": (
            float(np.mean(returns)) if returns else None
        ),
        "mean_e2e": float(np.mean([
            success for row in rows for success in row["successes"]
        ])),
        "acquisition_new_1m_cells_per_command": float(np.mean([
            row["acquisition_new_1m_cells_per_command"] for row in rows
        ])),
        "acquisition_zero_new_1m_cell_fraction": float(np.mean([
            row["acquisition_zero_new_1m_cell_fraction"] for row in rows
        ])),
        "acquisition_revisited_endpoint_fraction": float(np.mean([
            row["acquisition_revisited_endpoint_fraction"] for row in rows
        ])),
        "hub_selections": int(sum(row["hub_selections"] for row in rows)),
        "hub_arrivals": int(sum(row["hub_arrivals"] for row in rows)),
        "hub_route_attempts": int(total_route_attempts),
        "hub_route_arrivals": int(total_route_arrivals),
        "hub_route_failures": int(total_route_attempts - total_route_arrivals),
        "hub_route_arrival_rate": (
            float(total_route_arrivals / total_route_attempts)
            if total_route_attempts else None
        ),
        "online_destination_arrival_rate": float(np.mean([
            row["online_destination_arrival_rate"] for row in rows
            if row["online_destination_arrival_rate"] is not None
        ])),
        "hub_route_similarity": PROTOCOL_VALUE["hub_route_similarity"],
        "hub_route_similarity_cutoff": None,
        "hub_route_prior_weight": float(RECORDED_EVIDENCE_PRIOR_WEIGHT),
        "reachability_computational_backend": PROTOCOL_VALUE.get(
            "reachability_computational_backend", "legacy_domain_backend"
        ),
        "coverage_computational_backend": PROTOCOL_VALUE.get(
            "coverage_computational_backend", "legacy_domain_backend"
        ),
        "computational_equivalence_report": (
            str(_equivalence_report_path(args))
            if PROTOCOL_VALUE.get("reachability_computational_backend")
            == "exact_batched_rolling_torch" else None
        ),
        "pre_optimization_exact_scalar_task_ids": (
            pre_optimization_scalar_task_ids
        ),
        "optimized_backend_task_ids": (
            [
                int(task_id) for task_id in PROTOCOL_VALUE["task_ids"]
                if int(task_id) not in pre_optimization_scalar_task_ids
            ]
            if optimized_latent_backend else []
        ),
        "mixed_computational_backend_interpretation": (
            "same_method_and_exact_argmax;computational_equivalence_gate_passed"
            if pre_optimization_scalar_task_ids else None
        ),
        "figures": figures,
        "rows": rows,
        "hidden_goal_visible_to_planner_executor_or_models": False,
        "success_radius_read_by_planner": False,
        "task_gradient_updates": 0,
        "confirmation_panel_accessed": False,
    }
    r48.dump(summary_path, result)
    print(json.dumps({
        key: result[key]
        for key in (
            "tasks", "task_ids", "episodes_per_task",
            "commands_per_episode", "acquired_tasks", "acquisition_at_1",
            "acquisition_at_5", "acquisition_at_10",
            "pooled_conditional_return", "mean_e2e",
            "acquisition_new_1m_cells_per_command",
            "acquisition_zero_new_1m_cell_fraction",
            "acquisition_revisited_endpoint_fraction",
            "online_destination_arrival_rate", "hub_route_attempts",
            "hub_route_arrivals", "hub_route_failures",
            "hub_route_arrival_rate", "figures",
        )
    }, indent=2))
    return result


def parser() -> argparse.ArgumentParser:
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
    commands.add_parser("equivalence").set_defaults(
        func=command_equivalence
    )
    commands.add_parser("evaluate").set_defaults(func=command_evaluate)
    return value


def main() -> None:
    args = parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
