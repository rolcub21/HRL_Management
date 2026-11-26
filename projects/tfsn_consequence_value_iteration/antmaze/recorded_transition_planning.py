"""Finite-horizon planning on reward-free transitions that actually occurred.

R4 projected every learned transition onto a sparse state graph.  R6 removes
that projection from the graph itself: each edge is one recorded
start--command--endpoint triple and both incident nodes are the exact recorded
states.  The frozen consequence ensemble is used only to connect the live
state to this empirical graph.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from scripts.antmaze.current_tfsn.collect import evaluate_supplied_goal_task
from scripts.antmaze.current_tfsn.constants import (
    CANDIDATES,
    ENDPOINT_SAMPLES,
    EPISODE_STEPS,
    LANDING_SIGMA,
    STATE_DIM,
)
from scripts.antmaze.current_tfsn.data import command_records
from scripts.antmaze.current_tfsn.executor import FaramaSACExecutor, make_env
from scripts.antmaze.current_tfsn.main import dump, panel, paths
from scripts.antmaze.current_tfsn.mdn import MDNEnsemble


METHOD_REVISION = "antmaze_recorded_transition_value_iteration_r6_repeatability"
# This affects only the numerical approximation to the continuous kernel
# support estimator.  It is not a distance, success, or environment rule.
SUPPORT_NEIGHBORS = 256


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def consequence_inputs(states, commands):
    states = np.asarray(states, np.float32).reshape(-1, STATE_DIM)
    commands = np.asarray(commands, np.float32).reshape(-1, 2)
    if len(states) != len(commands):
        raise ValueError("states and commands must be aligned")
    delta = commands - states[:, :2]
    return np.concatenate(
        (states, commands, delta, np.abs(delta).sum(axis=1, keepdims=True)),
        axis=1,
    ).astype(np.float32)


def exact_goal_field(xy, goal, sigma):
    xy = np.asarray(xy, np.float32).reshape(-1, 2)
    goal = np.asarray(goal, np.float32).reshape(2)
    squared_distance = np.square(xy - goal[None]).sum(axis=1)
    return np.exp(-squared_distance / (2.0 * float(sigma) ** 2)).astype(
        np.float32
    )


class RecordedTransitionGraph:
    """A graph whose edges and endpoints are exact reward-free observations."""

    def __init__(
        self,
        nodes,
        source_index,
        endpoint_index,
        commands,
        edge_reliability,
        state_scale,
        metadata=None,
    ):
        self.nodes = np.asarray(nodes, np.float32)
        self.source_index = np.asarray(source_index, np.int32)
        self.endpoint_index = np.asarray(endpoint_index, np.int32)
        self.commands = np.asarray(commands, np.float32)
        self.edge_reliability = np.asarray(edge_reliability, np.float32)
        self.state_scale = np.asarray(state_scale, np.float32).reshape(STATE_DIM)
        self.metadata = dict(metadata or {})

        edge_count = len(self.source_index)
        if self.nodes.ndim != 2 or self.nodes.shape[1] != STATE_DIM:
            raise ValueError("nodes must have shape [N, STATE_DIM]")
        if self.commands.shape != (edge_count, 2):
            raise ValueError("commands must have one xy coordinate per edge")
        for name, value in (
            ("endpoint_index", self.endpoint_index),
            ("edge_reliability", self.edge_reliability),
        ):
            if value.shape != (edge_count,):
                raise ValueError(f"{name} must have one value per edge")
        if np.any(self.source_index < 0) or np.any(
            self.source_index >= len(self.nodes)
        ):
            raise ValueError("source index falls outside graph")
        if np.any(self.endpoint_index < 0) or np.any(
            self.endpoint_index >= len(self.nodes)
        ):
            raise ValueError("endpoint index falls outside graph")
        if not np.isfinite(self.state_scale).all() or np.any(
            self.state_scale <= 0
        ):
            raise ValueError("state scale must be finite and positive")
        if np.any(self.edge_reliability < 0) or np.any(
            self.edge_reliability > 1
        ):
            raise ValueError("edge reliability must lie in [0, 1]")

        self.node_tree = cKDTree(self.nodes / self.state_scale[None])
        self.edge_starts = self.nodes[self.source_index]
        self.joint_features = np.concatenate(
            (
                self.edge_starts / self.state_scale[None],
                self.commands / LANDING_SIGMA,
            ),
            axis=1,
        )
        self.joint_tree = cKDTree(self.joint_features)

    @property
    def node_count(self):
        return len(self.nodes)

    @property
    def edge_count(self):
        return len(self.source_index)

    def project(self, states):
        states = np.asarray(states, np.float32).reshape(-1, STATE_DIM)
        distance, index = self.node_tree.query(
            states / self.state_scale[None], k=1, workers=-1
        )
        return index.astype(np.int32), np.asarray(distance, np.float32)

    def sample_commands(self, rng, current_state, count):
        """Draw commands demonstrated from continuously similar states."""
        state = np.asarray(current_state, np.float32).reshape(STATE_DIM)
        difference = (
            self.edge_starts - state[None]
        ) / self.state_scale[None]
        squared_distance = np.square(difference).sum(axis=1, dtype=np.float64)
        log_weight = -0.5 * squared_distance
        weight = np.exp(log_weight - log_weight.max())
        weight_sum = float(weight.sum())
        cdf = np.cumsum(weight)
        cdf[-1] = weight_sum
        selected = np.searchsorted(
            cdf, rng.random(int(count)) * weight_sum, side="right"
        )
        return self.commands[selected].copy(), selected, {
            "proposal_source": "recorded_commands_from_similar_recorded_states",
            "proposal_effective_sample_size": float(
                weight_sum**2 / np.square(weight).sum()
            ),
            "sampled_source_standardized_distance_median": float(
                np.median(np.sqrt(squared_distance[selected]))
            ),
            "sampled_source_xy_distance_median": float(
                np.median(
                    np.linalg.norm(
                        self.edge_starts[selected, :2] - state[None, :2], axis=1
                    )
                )
            ),
        }

    def command_support(self, current_state, commands):
        """Continuous support posterior for live state-command queries."""
        state = np.asarray(current_state, np.float32).reshape(STATE_DIM)
        commands = np.asarray(commands, np.float32).reshape(-1, 2)
        query = np.concatenate(
            (
                np.repeat((state / self.state_scale)[None], len(commands), axis=0),
                commands / LANDING_SIGMA,
            ),
            axis=1,
        )
        neighbors = min(SUPPORT_NEIGHBORS, self.edge_count)
        distance, index = self.joint_tree.query(
            query, k=neighbors, workers=-1
        )
        if neighbors == 1:
            distance = distance[:, None]
            index = index[:, None]
        weight = np.exp(-0.5 * np.square(distance))
        aimed_mass = weight.sum(axis=1)
        reliability_mass = (
            weight * self.edge_reliability[index]
        ).sum(axis=1)
        local_reliability = np.divide(
            reliability_mass,
            aimed_mass,
            out=np.zeros_like(reliability_mass),
            where=aimed_mass > 0,
        )
        return {
            "aimed_support_mass": aimed_mass.astype(np.float32),
            "local_recorded_reliability": local_reliability.astype(np.float32),
            "pessimistic_success": (
                local_reliability * aimed_mass / (aimed_mass + 1.0)
            ).astype(np.float32),
        }

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            nodes=self.nodes,
            source_index=self.source_index,
            endpoint_index=self.endpoint_index,
            commands=self.commands,
            edge_reliability=self.edge_reliability,
            state_scale=self.state_scale,
        )
        path.with_suffix(".json").write_text(
            json.dumps(self.metadata, indent=2) + "\n"
        )

    @classmethod
    def load(cls, path):
        path = Path(path)
        with np.load(path) as data:
            arrays = {name: data[name] for name in data.files}
        return cls(
            metadata=json.loads(path.with_suffix(".json").read_text()),
            **arrays,
        )


def kernel_edge_reliability(
    starts,
    commands,
    endpoints,
    state_scale,
    neighbors=SUPPORT_NEIGHBORS,
):
    """Estimate repeatability from similar observed state-command queries."""
    starts = np.asarray(starts, np.float32)
    commands = np.asarray(commands, np.float32)
    endpoints = np.asarray(endpoints, np.float32)
    state_scale = np.asarray(state_scale, np.float32).reshape(STATE_DIM)
    feature = np.concatenate(
        (starts / state_scale[None], commands / LANDING_SIGMA), axis=1
    )
    tree = cKDTree(feature)
    neighbors = min(int(neighbors), len(feature))
    distance, index = tree.query(feature, k=neighbors, workers=-1)
    if neighbors == 1:
        distance = distance[:, None]
        index = index[:, None]
    weight = np.exp(-0.5 * np.square(distance))
    # Reliability concerns the recorded transition, not whether a fixed-time
    # command happened to finish at a potentially distant requested target.
    # Compare realized full-state deltas among similar state-command records.
    realized_delta = endpoints - starts
    neighbor_delta = realized_delta[index]
    outcome_difference = (
        neighbor_delta - realized_delta[:, None]
    ) / state_scale[None, None]
    outcome_agreement = np.exp(
        -0.5 * np.square(outcome_difference).sum(axis=2, dtype=np.float64)
    )
    aimed_mass = weight.sum(axis=1)
    agreement_mass = (weight * outcome_agreement).sum(axis=1)
    empirical_agreement = agreement_mass / aimed_mass
    reliability = agreement_mass / (aimed_mass + 1.0)
    return (
        reliability.astype(np.float32),
        empirical_agreement.astype(np.float32),
        aimed_mass.astype(np.float32),
    )


def build_recorded_graph(starts, commands, endpoints, state_scale):
    starts = np.asarray(starts, np.float32).reshape(-1, STATE_DIM)
    commands = np.asarray(commands, np.float32).reshape(-1, 2)
    endpoints = np.asarray(endpoints, np.float32).reshape(-1, STATE_DIM)
    if not (len(starts) == len(commands) == len(endpoints)):
        raise ValueError("recorded starts, commands, and endpoints must align")

    # Exact row identity is deliberate.  No spatial radius, clustering
    # tolerance, or predicted-to-recorded projection enters graph creation.
    nodes, inverse = np.unique(
        np.concatenate((starts, endpoints), axis=0),
        axis=0,
        return_inverse=True,
    )
    edge_count = len(starts)
    source_index = inverse[:edge_count]
    endpoint_index = inverse[edge_count:]
    reliability, empirical_agreement, support_mass = kernel_edge_reliability(
        starts, commands, endpoints, state_scale
    )
    out_degree = np.bincount(source_index, minlength=len(nodes))
    exact_start_error = np.abs(nodes[source_index] - starts).max()
    exact_endpoint_error = np.abs(nodes[endpoint_index] - endpoints).max()
    metadata = {
        "method_revision": METHOD_REVISION,
        "node_source": "all_exact_reward_free_command_boundary_states",
        "edge_source": "all_exact_recorded_reward_free_command_transitions",
        "transition_model_inside_graph": False,
        "predicted_endpoint_projection_inside_graph": False,
        "live_connector": "one_step_frozen_distributional_consequence_ensemble",
        "live_continuation_support": (
            "gaussian_of_nearest_recorded_full_state_standardized_distance"
        ),
        "edge_reliability": (
            "continuous_kernel_posterior_of_realized_delta_repeatability"
        ),
        "support_neighbors_computational_only": int(
            min(SUPPORT_NEIGHBORS, edge_count)
        ),
        "backup": "finite_horizon_undiscounted_value_iteration",
        "node_count": int(len(nodes)),
        "edge_count": int(edge_count),
        "field_sigma_m": float(LANDING_SIGMA),
        "exact_recorded_start_reconstruction_max_error": float(
            exact_start_error
        ),
        "exact_recorded_endpoint_reconstruction_max_error": float(
            exact_endpoint_error
        ),
        "fraction_edges_with_recorded_continuation": float(
            np.mean(out_degree[endpoint_index] > 0)
        ),
        "nodes_with_multiple_recorded_actions": int(np.sum(out_degree > 1)),
        "edge_reliability_mean": float(reliability.mean()),
        "edge_reliability_median": float(np.median(reliability)),
        "edge_reliability_p10": float(np.quantile(reliability, 0.1)),
        "empirical_outcome_agreement_median": float(
            np.median(empirical_agreement)
        ),
        "edge_support_mass_median": float(np.median(support_mass)),
    }
    return RecordedTransitionGraph(
        nodes,
        source_index,
        endpoint_index,
        commands,
        reliability,
        state_scale,
        metadata,
    )


def finite_horizon_values(graph, goal, sigma, horizon):
    """Undiscounted finite-horizon values over exact recorded edges."""
    endpoint_field = exact_goal_field(
        graph.nodes[graph.endpoint_index, :2], goal, sigma
    )
    values = np.zeros((int(horizon) + 1, graph.node_count), np.float32)
    for remaining in range(1, int(horizon) + 1):
        edge_value = graph.edge_reliability * (
            endpoint_field + values[remaining - 1][graph.endpoint_index]
        )
        np.maximum.at(values[remaining], graph.source_index, edge_value)
    return values


class RecordedTransitionPlanner:
    def __init__(self, consequence, graph, rng, horizon=10):
        self.consequence = consequence
        self.graph = graph
        self.rng = rng
        self.horizon = int(horizon)
        self._goal_key = None
        self._values = None

    def _set_field(self, fields):
        if not hasattr(fields, "goal"):
            raise ValueError("R6 currently requires an exact supplied goal")
        key = (tuple(np.asarray(fields.goal, np.float32)), float(fields.sigma))
        if key != self._goal_key:
            self._values = finite_horizon_values(
                self.graph, fields.goal, fields.sigma, self.horizon
            )
            self._goal_key = key

    def choose(
        self, state, command_index, fields, online_states=(),
        online_trajectories=(),
    ):
        del online_states, online_trajectories
        self._set_field(fields)
        remaining = self.horizon - int(command_index)
        if remaining <= 0:
            raise ValueError("command index exceeds planning horizon")
        state = np.asarray(state, np.float32).reshape(STATE_DIM)
        commands, _source_edge, proposal = self.graph.sample_commands(
            self.rng, state, CANDIDATES
        )
        support = self.graph.command_support(state, commands)
        inputs = consequence_inputs(
            np.repeat(state[None], len(commands), axis=0), commands
        )

        member_q = []
        member_projection_distance = []
        member_connection_support = []
        samples_per_member = ENDPOINT_SAMPLES // len(self.consequence.members)
        if samples_per_member * len(self.consequence.members) != ENDPOINT_SAMPLES:
            raise ValueError("endpoint samples must divide over ensemble members")
        for member in range(len(self.consequence.members)):
            sampled_delta = self.consequence.sample_member(
                inputs, member, samples_per_member, self.rng
            )
            absolute = state[None, None] + sampled_delta
            projected, distance = self.graph.project(
                absolute.reshape(-1, STATE_DIM)
            )
            projected = projected.reshape(len(commands), samples_per_member)
            distance = distance.reshape(len(commands), samples_per_member)
            # This is a smooth support weight, not a projection threshold.  A
            # predicted state unlike every recorded graph state cannot inherit
            # a fictitious recorded continuation.
            connection_support = np.exp(-0.5 * np.square(distance))
            immediate = fields.field(
                absolute[..., :2].reshape(-1, 2)
            ).reshape(len(commands), samples_per_member)
            continuation = self._values[remaining - 1][projected]
            member_q.append(
                (immediate + connection_support * continuation).mean(axis=1)
            )
            member_projection_distance.append(distance.mean(axis=1))
            member_connection_support.append(connection_support.mean(axis=1))

        member_q = np.stack(member_q, axis=1)
        # The model ensemble supplies epistemic pessimism.  Continuous
        # state-command support supplies an additional no-threshold guard on
        # the live connector only.
        model_value = member_q.min(axis=1)
        scores = model_value * support["pessimistic_success"]
        winner = int(np.argmax(scores))
        top = np.argsort(-scores, kind="stable")[:5]
        projection = np.stack(member_projection_distance, axis=1)
        connection = np.stack(member_connection_support, axis=1)
        diagnostic = {
            "planner": "recorded_transition_value_iteration",
            "method_revision": METHOD_REVISION,
            "phase": "supplied_goal",
            "remaining": int(remaining),
            "selected_score": float(scores[winner]),
            "selected_model_value": float(model_value[winner]),
            "selected_pessimistic_command_support": float(
                support["pessimistic_success"][winner]
            ),
            "selected_aimed_support_mass": float(
                support["aimed_support_mass"][winner]
            ),
            "selected_connection_support": float(connection[winner].mean()),
            "selected_projection_standardized_distance": float(
                projection[winner].mean()
            ),
            "selected_command_distance": float(
                np.linalg.norm(commands[winner] - state[:2])
            ),
            "top_scores": scores[top].tolist(),
            "top_commands": commands[top].tolist(),
            "top_member_values": member_q[top].tolist(),
            "top_command_support": support["pessimistic_success"][top].tolist(),
            "graph_nodes": int(self.graph.node_count),
            "graph_edges": int(self.graph.edge_count),
            "graph_edges_are_recorded": True,
            "search_enabled": False,
            "acquisition_search_mode": "supplied_goal",
            "fine_search_sigma": float(fields.sigma),
            "coarse_search_sigma": float(fields.sigma),
            "coarse_search_weight": 0.0,
            "fine_unsearched_mass": None,
            "coarse_unsearched_mass": None,
        }
        diagnostic.update(proposal)
        return commands[winner].copy(), diagnostic


def load_frozen_data(root, round_id, device):
    checkpoint = paths(root, round_id)["consequence"]
    consequence = MDNEnsemble(checkpoint, device)
    round_dirs = [paths(root, index)["round"] for index in range(round_id + 1)]
    starts, commands, endpoints = command_records(round_dirs)
    state_scale = consequence.input_std[:STATE_DIM].copy()
    state_scale[:2] = LANDING_SIGMA
    return consequence, starts, commands, endpoints, state_scale


def command_prepare(args):
    consequence, starts, commands, endpoints, state_scale = load_frozen_data(
        args.root, args.round, args.device
    )
    del consequence
    graph = build_recorded_graph(
        starts, commands, endpoints, state_scale
    )
    checkpoint = paths(args.root, args.round)["consequence"]
    manifests = [
        paths(args.root, index)["round"] / "manifest.json"
        for index in range(args.round + 1)
    ]
    graph.metadata.update({
        "round": int(args.round),
        "seed": int(args.seed),
        "consequence_checkpoint": str(checkpoint.resolve()),
        "consequence_sha256": sha256_file(checkpoint),
        "reward_free_manifests": [
            {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for path in manifests
        ],
        "task_reward_used": False,
        "hidden_goals_used": False,
        "oracle_inputs": [],
    })
    graph.save(args.graph)
    result = dict(graph.metadata)
    result.update({
        "artifact": str(Path(args.graph).resolve()),
        "artifact_sha256": sha256_file(args.graph),
    })
    print(json.dumps(result, indent=2))


def command_evaluate(args):
    graph = RecordedTransitionGraph.load(args.graph)
    checkpoint = paths(args.root, args.round)["consequence"]
    expected = graph.metadata.get("consequence_sha256")
    actual = sha256_file(checkpoint)
    if expected != actual:
        raise ValueError(
            f"graph consequence hash mismatch: expected {expected}, got {actual}"
        )
    consequence = MDNEnsemble(checkpoint, args.device)
    planner = RecordedTransitionPlanner(
        consequence, graph, np.random.default_rng(args.seed)
    )
    executor = FaramaSACExecutor(paths(args.root)["executor"], args.device)
    selected = panel(args.root, "development", 9, 37701)
    output = (
        paths(args.root)["eval"]
        / "development"
        / "recorded_transition_value_iteration_r6_repeatability"
    )
    output.mkdir(parents=True, exist_ok=True)
    env = make_env(max_episode_steps=10_000)
    rows = []
    for task in selected["tasks"]:
        planner.rng = np.random.default_rng(
            int(args.seed) + 1_000_003 * int(task["task_id"])
        )
        row = evaluate_supplied_goal_task(
            env,
            executor,
            planner,
            np.asarray(task["goal"], np.float32),
            task["reset_seed"],
            LANDING_SIGMA,
            max_steps=EPISODE_STEPS,
            log_path=output / f"task_{task['task_id']:03d}.jsonl",
        )
        row.update(task)
        rows.append(row)
        print(json.dumps(row), flush=True)
    env.close()
    result = {
        "method_revision": METHOD_REVISION,
        "claim_level": "consumed_development_supplied_goal_diagnostic",
        "panel": "development",
        "tasks": len(rows),
        "success_rate": float(np.mean([row["success"] for row in rows])),
        "mean_steps_to_goal_success": (
            float(np.mean([
                row["steps_to_goal"] for row in rows if row["success"]
            ]))
            if any(row["success"] for row in rows) else None
        ),
        "search_enabled": False,
        "task_gradient_updates": 0,
        "graph": str(Path(args.graph).resolve()),
        "graph_sha256": sha256_file(args.graph),
        "reachable_reference": {"oracle_successes": 7, "oracle_tasks": 9},
        "comparison_references": {
            "direct_executor_successes": 2,
            "behavior_successor_tfsn_successes": 3,
            "r4_projected_consequence_graph_successes": 1,
        },
        "rows": rows,
    }
    dump(output / "summary.json", result)
    print(json.dumps({
        "success_rate": result["success_rate"],
        "mean_steps_to_goal_success": result["mean_steps_to_goal_success"],
    }, indent=2))


def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=Path, default=Path("/home/ai_diagnosis/SG/e377")
    )
    parser.add_argument("--graph", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=605)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--round", type=int, default=2)
    prepare.set_defaults(func=command_prepare)
    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--round", type=int, default=2)
    evaluate.set_defaults(func=command_evaluate)
    return parser


def main():
    args = parser().parse_args()
    if args.graph is None:
        args.graph = (
            args.root
            / "model_planning"
            / "round2_recorded_transition_graph_r6.npz"
        )
    args.func(args)


if __name__ == "__main__":
    main()
