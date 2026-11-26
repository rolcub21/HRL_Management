"""R12: share recorded command coordinates and evaluate them from each state.

R10's mobility-weighted bridges connected the graph, but a state sometimes
spent a complete command entering a same-position landmark merely to inherit
that landmark's outgoing action.  R12 preserves the R10 bridges and all exact
recorded edges.  It additionally gives every exact or live state every command
recorded at a source coordinate within the already-defined landing scale.
Each shared command is evaluated by the factorized consequence model from the
state that would actually issue it.  Its continuation is read at the landmark
nearest the commanded coordinate, not at the endpoint of the old execution.
Thus the command is detached from the recorded state, endpoint, and empirical
edge reliability; no recorded posture or old outcome is copied.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from scripts.antmaze.current_tfsn.collect import evaluate_supplied_goal_task
from scripts.antmaze.current_tfsn.constants import (
    ENDPOINT_SAMPLES,
    EPISODE_STEPS,
    LANDING_SIGMA,
    STATE_DIM,
)
from scripts.antmaze.current_tfsn.executor import FaramaSACExecutor, make_env
from scripts.antmaze.current_tfsn.main import dump, panel, paths

from projects.tfsn_consequence_value_iteration.antmaze.mobility_stitched_planning import (
    CLAIM_LEVEL,
    LANDMARK_NEIGHBORS,
    NODE_SHARD_SIZE,
    PAIR_INFERENCE_BATCH,
    MobilityModel,
    MobilityStitchedPlanner,
    _artifact_paths as r10_artifact_paths,
    _bind_shards,
    _bridge_weight_chunk,
    _load_base,
    _prepare_landmarks,
    _save_npz_atomic,
    _valid_npz,
    _verify_bridges,
)
from projects.tfsn_consequence_value_iteration.antmaze.recorded_transition_planning import (
    sha256_file,
)


METHOD_REVISION = "antmaze_command_coordinate_sharing_value_iteration_r12"


def artifact_paths(root):
    directory = (
        paths(root)["root"]
        / "model_planning"
        / "command_coordinate_sharing_r12"
    )
    return {
        "directory": directory,
        "neighborhoods": directory / "shared_action_neighborhoods.npz",
        "neighborhood_meta": directory / "shared_action_neighborhoods.json",
        "weight_shards": directory / "weight_shards",
        "actions": directory / "shared_actions.npz",
        "action_meta": directory / "shared_actions.json",
    }


def shared_command_neighborhoods(graph, radius=LANDING_SIGMA):
    """Return CSR edge indices whose recorded source is spatially nearby."""
    source_tree = cKDTree(graph.nodes[graph.source_index, :2])
    neighbors = source_tree.query_ball_point(
        graph.nodes[:, :2], r=float(radius), workers=-1
    )
    offsets = np.zeros(graph.node_count + 1, np.int64)
    offsets[1:] = np.cumsum([len(row) for row in neighbors], dtype=np.int64)
    edge_index = np.empty(int(offsets[-1]), np.int32)
    for node, row in enumerate(neighbors):
        edge_index[offsets[node]:offsets[node + 1]] = np.sort(
            np.asarray(row, np.int32)
        )
    return offsets, edge_index


def _prepare_neighborhoods(args, graph):
    artifact = artifact_paths(args.root)
    artifact["directory"].mkdir(parents=True, exist_ok=True)
    graph_hash = sha256_file(args.graph)
    if artifact["neighborhoods"].exists() and artifact[
        "neighborhood_meta"
    ].exists():
        metadata = json.loads(artifact["neighborhood_meta"].read_text())
        if metadata.get("base_graph_sha256") != graph_hash:
            raise ValueError("saved shared-action neighborhoods use another graph")
        with np.load(artifact["neighborhoods"]) as data:
            offsets = data["offsets"]
            edge_index = data["edge_index"]
        return artifact, offsets, edge_index, metadata

    offsets, edge_index = shared_command_neighborhoods(graph)
    counts = np.diff(offsets)
    _save_npz_atomic(
        artifact["neighborhoods"], offsets=offsets, edge_index=edge_index
    )
    metadata = {
        "method_revision": METHOD_REVISION,
        "claim_level": CLAIM_LEVEL,
        "hidden_goals_accessed": False,
        "task_reward_used": False,
        "oracle_inputs": [],
        "base_graph": str(Path(args.graph).resolve()),
        "base_graph_sha256": graph_hash,
        "sharing_rule": (
            "all_commands_whose_recorded_source_coordinate_is_within_sigma"
        ),
        "sharing_radius_m": float(LANDING_SIGMA),
        "per_environment_distance_rule_added": False,
        "nodes": int(graph.node_count),
        "recorded_edges_preserved": int(graph.edge_count),
        "shared_state_command_pairs": int(len(edge_index)),
        "commands_per_state_minimum": int(counts.min()),
        "commands_per_state_median": float(np.median(counts)),
        "commands_per_state_p90": float(np.quantile(counts, 0.9)),
        "commands_per_state_p99": float(np.quantile(counts, 0.99)),
        "commands_per_state_maximum": int(counts.max()),
        "states_without_shared_commands": int(np.sum(counts == 0)),
    }
    dump(artifact["neighborhood_meta"], metadata)
    return artifact, offsets, edge_index, metadata


def _source_index(offsets):
    return np.repeat(
        np.arange(len(offsets) - 1, dtype=np.int32), np.diff(offsets)
    )


def command_prepare(args):
    graph, consequence, consequence_path = _load_base(args)
    artifact, offsets, edge_index, neighborhood_meta = _prepare_neighborhoods(
        args, graph
    )
    r10 = r10_artifact_paths(args.root)
    if not r10["mobility"].exists() or not r10["bridges"].exists():
        raise FileNotFoundError("complete R10 mobility and bridges first")
    mobility = MobilityModel(r10["mobility"], args.device)
    _r10_artifact, landmark_index, _bridge_targets, _distance, _meta = (
        _prepare_landmarks(args, graph, consequence_path)
    )
    landmark_tree = cKDTree(graph.nodes[landmark_index, :2])
    command_distance, command_local_target = landmark_tree.query(
        graph.commands, k=1, workers=-1
    )
    command_target = landmark_index[command_local_target].astype(np.int32)

    _bind_shards(artifact["weight_shards"], {
        "method_revision": METHOD_REVISION,
        "stage": "state_conditioned_local_shared_action_weights",
        "base_graph_sha256": sha256_file(args.graph),
        "factorized_consequence_sha256": sha256_file(consequence_path),
        "mobility_sha256": sha256_file(r10["mobility"]),
        "landmarks_sha256": sha256_file(r10["landmarks"]),
        "neighborhoods_sha256": sha256_file(artifact["neighborhoods"]),
        "continuation_target": "nearest_sigma_net_landmark_to_command_coordinate",
        "recorded_endpoint_used_as_shared_target": False,
        "recorded_edge_reliability_applied_to_shared_command": False,
        "endpoint_samples": int(ENDPOINT_SAMPLES),
        "landing_sigma_m": float(LANDING_SIGMA),
        "seed": int(args.seed),
    })

    for begin in range(0, graph.node_count, NODE_SHARD_SIZE):
        stop = min(begin + NODE_SHARD_SIZE, graph.node_count)
        pair_begin = int(offsets[begin])
        pair_stop = int(offsets[stop])
        shard = (
            artifact["weight_shards"]
            / f"nodes_{begin:06d}_{stop:06d}.npz"
        )
        if shard.exists() and _valid_npz(
            shard, "weight", (pair_stop - pair_begin,)
        ):
            continue
        local_edges = edge_index[pair_begin:pair_stop]
        counts = np.diff(offsets[begin:stop + 1])
        source_node = np.repeat(
            np.arange(begin, stop, dtype=np.int32), counts
        )
        source = graph.nodes[source_node]
        target_state = graph.nodes[command_target[local_edges]]
        command = graph.commands[local_edges]
        values = []
        for batch_begin in range(0, len(local_edges), PAIR_INFERENCE_BATCH):
            batch_stop = min(
                batch_begin + PAIR_INFERENCE_BATCH, len(local_edges)
            )
            values.append(_bridge_weight_chunk(
                consequence,
                mobility,
                source[batch_begin:batch_stop],
                target_state[batch_begin:batch_stop],
                command[batch_begin:batch_stop],
                args.seed + pair_begin + batch_begin,
            ))
        if values:
            weight = np.concatenate(values).astype(np.float32)
        else:
            weight = np.empty(0, np.float32)
        _save_npz_atomic(shard, weight=weight)
        print(
            f"shared action weights nodes={stop}/{graph.node_count} "
            f"pairs={pair_stop}/{len(edge_index)}",
            flush=True,
        )

    weights = []
    for begin in range(0, graph.node_count, NODE_SHARD_SIZE):
        stop = min(begin + NODE_SHARD_SIZE, graph.node_count)
        shard = (
            artifact["weight_shards"]
            / f"nodes_{begin:06d}_{stop:06d}.npz"
        )
        with np.load(shard) as data:
            weights.append(data["weight"])
    weight = np.concatenate(weights).astype(np.float32)
    _save_npz_atomic(
        artifact["actions"],
        offsets=offsets,
        edge_index=edge_index,
        target_index=command_target[edge_index],
        weight=weight,
    )
    metadata = {
        "method_revision": METHOD_REVISION,
        "claim_level": CLAIM_LEVEL,
        "hidden_goals_accessed": False,
        "task_reward_used": False,
        "oracle_inputs": [],
        "artifact": str(artifact["actions"].resolve()),
        "artifact_sha256": sha256_file(artifact["actions"]),
        "base_graph_sha256": sha256_file(args.graph),
        "factorized_consequence_sha256": sha256_file(consequence_path),
        "mobility_sha256": sha256_file(r10["mobility"]),
        "landmarks_sha256": sha256_file(r10["landmarks"]),
        "r10_bridges_sha256": sha256_file(r10["bridges"]),
        "neighborhoods_sha256": sha256_file(artifact["neighborhoods"]),
        "recorded_nodes_preserved": int(graph.node_count),
        "recorded_edges_preserved": int(graph.edge_count),
        "r10_model_bridges_preserved": True,
        "shared_state_command_pairs": int(len(edge_index)),
        "shared_target_definition": (
            "nearest_sigma_net_landmark_to_command_coordinate"
        ),
        "recorded_endpoint_used_as_shared_target": False,
        "recorded_edge_reliability_applied_to_shared_command": False,
        "command_to_target_landmark_distance_median": float(
            np.median(command_distance)
        ),
        "command_to_target_landmark_distance_p99": float(
            np.quantile(command_distance, 0.99)
        ),
        "command_to_target_landmark_distance_maximum": float(
            command_distance.max()
        ),
        "shared_weight_mean": float(weight.mean()),
        "shared_weight_median": float(np.median(weight)),
        "shared_weight_p90": float(np.quantile(weight, 0.9)),
        "neighborhood_metadata": neighborhood_meta,
    }
    dump(artifact["action_meta"], metadata)
    print(json.dumps(metadata, indent=2))


def shared_action_values(
    graph,
    bridge_targets,
    bridge_weight,
    offsets,
    shared_edges,
    shared_target,
    shared_weight,
    goal,
    horizon=10,
):
    goal = np.asarray(goal, np.float32)
    field = np.exp(
        -np.square(graph.nodes[:, :2] - goal[None]).sum(axis=1)
        / (2.0 * LANDING_SIGMA**2)
    ).astype(np.float32)
    values = np.zeros((horizon + 1, graph.node_count), np.float32)
    shared_source = _source_index(offsets)
    shared_target = np.asarray(shared_target, np.int32)
    coverage = []
    for remaining in range(1, horizon + 1):
        row = np.zeros(graph.node_count, np.float32)
        recorded_q = graph.edge_reliability * (
            field[graph.endpoint_index]
            + values[remaining - 1, graph.endpoint_index]
        )
        np.maximum.at(row, graph.source_index, recorded_q)
        bridge_q = bridge_weight * (
            field[bridge_targets] + values[remaining - 1, bridge_targets]
        )
        row = np.maximum(row, bridge_q.max(axis=1))
        shared_q = shared_weight * (
            field[shared_target] + values[remaining - 1, shared_target]
        )
        np.maximum.at(row, shared_source, shared_q)
        values[remaining] = row
        coverage.append({
            "remaining": remaining,
            "positive_nodes": int(np.sum(row > 0)),
            "positive_fraction": float(np.mean(row > 0)),
            "meaningful_nodes": int(np.sum(row > 1e-6)),
            "meaningful_fraction": float(np.mean(row > 1e-6)),
            "maximum": float(row.max()),
        })
    return values, coverage, field


class SharedActionPlanner(MobilityStitchedPlanner):
    def __init__(
        self,
        consequence,
        mobility,
        graph,
        landmark_index,
        bridge_targets,
        bridge_weight,
        shared_offsets,
        shared_edges,
        shared_target,
        shared_weight,
        rng,
        seed,
    ):
        super().__init__(
            consequence,
            mobility,
            graph,
            landmark_index,
            bridge_targets,
            bridge_weight,
            rng,
            seed,
        )
        self.shared_offsets = np.asarray(shared_offsets, np.int64)
        self.shared_edges = np.asarray(shared_edges, np.int32)
        self.shared_target = np.asarray(shared_target, np.int32)
        self.shared_weight = np.asarray(shared_weight, np.float32)
        self.recorded_source_tree = cKDTree(
            graph.nodes[graph.source_index, :2]
        )

    def _set_goal(self, fields):
        key = tuple(np.asarray(fields.goal, np.float32))
        if key != self.goal_key:
            self.values, self.coverage, self.node_field = shared_action_values(
                self.graph,
                self.targets,
                self.weight,
                self.shared_offsets,
                self.shared_edges,
                self.shared_target,
                self.shared_weight,
                fields.goal,
            )
            self.goal_key = key

    def choose(
        self,
        state,
        command_index,
        fields,
        online_states=(),
        online_trajectories=(),
    ):
        del online_states, online_trajectories
        self._set_goal(fields)
        remaining = 10 - int(command_index)
        if remaining <= 0:
            raise ValueError("command index exceeds planning horizon")
        state = np.asarray(state, np.float32).reshape(STATE_DIM)

        _distance, local = self.landmark_tree.query(
            state[:2],
            k=min(LANDMARK_NEIGHBORS, len(self.landmark_index)),
        )
        bridge_target = self.landmark_index[np.atleast_1d(local)]
        bridge_command = self.graph.nodes[bridge_target, :2]
        (
            bridge_q,
            bridge_connection,
            bridge_member_q,
            bridge_member_connection,
        ) = self._candidate_value(
            state,
            bridge_command,
            bridge_target,
            remaining,
            np.ones(len(bridge_target), np.float32),
            self.seed + command_index * 10_007,
            return_members=True,
        )

        shared_edge = np.sort(np.asarray(
            self.recorded_source_tree.query_ball_point(
                state[:2], r=LANDING_SIGMA
            ),
            np.int32,
        ))
        if len(shared_edge):
            shared_command = self.graph.commands[shared_edge]
            _shared_distance, shared_local_target = self.landmark_tree.query(
                shared_command, k=1
            )
            shared_target = self.landmark_index[
                np.asarray(shared_local_target, np.int32)
            ]
            (
                shared_q,
                shared_connection,
                shared_member_q,
                shared_member_connection,
            ) = self._candidate_value(
                state,
                shared_command,
                shared_target,
                remaining,
                np.ones(len(shared_edge), np.float32),
                self.seed + command_index * 10_007 + 97,
                return_members=True,
            )
        else:
            shared_command = np.empty((0, 2), np.float32)
            shared_target = np.empty(0, np.int32)
            shared_q = np.empty(0, np.float32)
            shared_connection = np.empty(0, np.float32)
            shared_member_q = np.empty(
                (0, len(self.consequence.members)), np.float32
            )
            shared_member_connection = shared_member_q.copy()

        command = np.concatenate((bridge_command, shared_command), axis=0)
        target = np.concatenate((bridge_target, shared_target), axis=0)
        score = np.concatenate((bridge_q, shared_q), axis=0)
        connection = np.concatenate(
            (bridge_connection, shared_connection), axis=0
        )
        member_q = np.concatenate(
            (bridge_member_q, shared_member_q), axis=0
        )
        member_connection = np.concatenate(
            (bridge_member_connection, shared_member_connection), axis=0
        )
        kind = np.asarray(
            ["model_bridge"] * len(bridge_command)
            + ["locally_shared_recorded_command"] * len(shared_command)
        )
        winner = int(np.argmax(score))
        diagnostic = {
            "planner": "command_coordinate_sharing_value_iteration",
            "method_revision": METHOD_REVISION,
            "phase": "supplied_goal",
            "remaining": int(remaining),
            "selected_score": float(score[winner]),
            "selected_kind": str(kind[winner]),
            "selected_target_index": int(target[winner]),
            "selected_connection_weight": float(connection[winner]),
            "selected_member_connection_weights": (
                member_connection[winner].tolist()
            ),
            "selected_member_scores": member_q[winner].tolist(),
            "selected_ensemble_connection_range": float(
                member_connection[winner].max()
                - member_connection[winner].min()
            ),
            "selected_command_distance": float(
                np.linalg.norm(command[winner] - state[:2])
            ),
            "locally_shared_commands_considered": int(len(shared_command)),
            "landmark_bridges_considered": int(len(bridge_command)),
            "positive_value_fraction": self.coverage[-1][
                "positive_fraction"
            ],
            "meaningful_value_fraction": self.coverage[-1][
                "meaningful_fraction"
            ],
            "top_scores": np.sort(score)[-5:][::-1].tolist(),
            "search_enabled": False,
            "acquisition_search_mode": "supplied_goal",
            "fine_search_sigma": float(fields.sigma),
            "coarse_search_sigma": float(fields.sigma),
            "coarse_search_weight": 0.0,
            "fine_unsearched_mass": None,
            "coarse_unsearched_mass": None,
        }
        return command[winner].copy(), diagnostic


def _load_artifacts(args):
    graph, consequence, consequence_path = _load_base(args)
    r10 = r10_artifact_paths(args.root)
    artifact, landmark_index, bridge_targets, _distance, _meta = (
        _prepare_landmarks(args, graph, consequence_path)
    )
    with np.load(r10["bridges"]) as data:
        saved_landmarks = data["landmark_index"]
        saved_targets = data["target_index"]
        bridge_weight = data["weight"]
    if not np.array_equal(landmark_index, saved_landmarks):
        raise ValueError("R10 landmark set changed")
    if not np.array_equal(bridge_targets, saved_targets):
        raise ValueError("R10 bridge targets changed")
    _verify_bridges(
        args,
        graph,
        consequence_path,
        artifact,
        landmark_index,
        bridge_targets,
        bridge_weight,
    )
    shared = artifact_paths(args.root)
    if not shared["actions"].exists() or not shared["action_meta"].exists():
        raise FileNotFoundError("run prepare-shared-actions first")
    metadata = json.loads(shared["action_meta"].read_text())
    if metadata.get("base_graph_sha256") != sha256_file(args.graph):
        raise ValueError("shared actions refer to another graph")
    if metadata.get("factorized_consequence_sha256") != sha256_file(
        consequence_path
    ):
        raise ValueError("shared actions refer to another consequence model")
    if metadata.get("artifact_sha256") != sha256_file(shared["actions"]):
        raise ValueError("shared action artifact hash mismatch")
    with np.load(shared["actions"]) as data:
        offsets = data["offsets"]
        shared_edges = data["edge_index"]
        shared_target = data["target_index"]
        shared_weight = data["weight"]
    if len(offsets) != graph.node_count + 1:
        raise ValueError("shared action offsets have the wrong shape")
    if (
        len(shared_edges) != len(shared_weight)
        or len(shared_target) != len(shared_edges)
        or offsets[-1] != len(shared_edges)
    ):
        raise ValueError("shared action arrays do not align")
    if np.any(shared_target < 0) or np.any(shared_target >= graph.node_count):
        raise ValueError("shared action target falls outside the graph")
    if (
        not np.isfinite(shared_weight).all()
        or np.any(shared_weight < 0)
        or np.any(shared_weight > 1)
    ):
        raise ValueError("shared action weights must be finite probabilities")
    mobility = MobilityModel(r10["mobility"], args.device)
    return (
        graph,
        consequence,
        consequence_path,
        mobility,
        landmark_index,
        bridge_targets,
        bridge_weight,
        offsets,
        shared_edges,
        shared_target,
        shared_weight,
        r10,
        shared,
    )


def command_evaluate(args):
    (
        graph,
        consequence,
        consequence_path,
        mobility,
        landmark_index,
        bridge_targets,
        bridge_weight,
        offsets,
        shared_edges,
        shared_target,
        shared_weight,
        r10,
        shared,
    ) = _load_artifacts(args)
    planner = SharedActionPlanner(
        consequence,
        mobility,
        graph,
        landmark_index,
        bridge_targets,
        bridge_weight,
        offsets,
        shared_edges,
        shared_target,
        shared_weight,
        np.random.default_rng(args.seed),
        args.seed,
    )
    executor = FaramaSACExecutor(paths(args.root)["executor"], args.device)
    selected = panel(args.root, "development", 9, 37701)
    output = (
        paths(args.root)["eval"]
        / "development"
        / "command_coordinate_sharing_value_iteration_r12"
    )
    output.mkdir(parents=True, exist_ok=True)
    env = make_env(max_episode_steps=10_000)
    rows = []
    for task in selected["tasks"]:
        task_seed = int(args.seed) + 1_000_003 * int(task["task_id"])
        planner.rng = np.random.default_rng(task_seed)
        planner.seed = task_seed
        planner.goal_key = None
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
        row["value_coverage"] = planner.coverage
        rows.append(row)
        print(json.dumps(row), flush=True)
    env.close()
    result = {
        "method_revision": METHOD_REVISION,
        "claim_level": CLAIM_LEVEL,
        "declared_deviations": [
            "supplied_goal_consumed_development_diagnostic",
            "recorded_commands_shared_within_existing_sigma_scale",
            "shared_continuation_at_command_coordinate_landmark",
        ],
        "oracle_inputs": [],
        "panel": "development",
        "tasks": len(rows),
        "successes": int(sum(row["success"] for row in rows)),
        "success_rate": float(np.mean([row["success"] for row in rows])),
        "mean_steps_to_goal_success": (
            float(np.mean([
                row["steps_to_goal"] for row in rows if row["success"]
            ]))
            if any(row["success"] for row in rows) else None
        ),
        "mean_meaningful_value_coverage": float(np.mean([
            row["value_coverage"][-1]["meaningful_fraction"] for row in rows
        ])),
        "recorded_nodes_preserved": int(graph.node_count),
        "recorded_edges_preserved": int(graph.edge_count),
        "r10_model_bridges_preserved": int(bridge_weight.size),
        "shared_state_command_pairs": int(len(shared_edges)),
        "sharing_radius_is_existing_sigma": True,
        "graph_sha256": sha256_file(args.graph),
        "factorized_consequence_sha256": sha256_file(consequence_path),
        "mobility_sha256": sha256_file(r10["mobility"]),
        "r10_bridges_sha256": sha256_file(r10["bridges"]),
        "shared_actions_sha256": sha256_file(shared["actions"]),
        "fixed_contract_oracle_reference": {"successes": 8, "tasks": 9},
        "r10_reference": {"successes": 5, "tasks": 9},
        "fresh_panel_accessed": False,
        "task_gradient_updates": 0,
        "rows": rows,
    }
    dump(output / "summary.json", result)
    print(json.dumps({
        "successes": result["successes"],
        "tasks": result["tasks"],
        "success_rate": result["success_rate"],
        "mean_steps_to_goal_success": result[
            "mean_steps_to_goal_success"
        ],
        "mean_meaningful_value_coverage": result[
            "mean_meaningful_value_coverage"
        ],
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
    prepare.set_defaults(func=command_prepare)
    evaluate = subparsers.add_parser("evaluate")
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
