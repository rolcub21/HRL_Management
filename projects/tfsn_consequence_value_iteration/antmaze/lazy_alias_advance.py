"""Narrow R10 test for commands wasted on spatial graph aliases.

R10 can spend a physical command connecting the live Ant to a recorded state
at essentially the same place, only to use that recorded state's outgoing
action at the next decision.  R12 exposed every locally recorded command and
thereby created a much larger, exploitable action set.  This diagnostic keeps
R10 unchanged and adds only one alternative for each R10 landmark candidate:
the single action that R10's frozen value function would take next at that
landmark.

The alternative is evaluated by the frozen consequence model from the live
full state.  Its association with the recorded state is weighted continuously
by the existing landing kernel, so there is no new distance threshold, mask,
or AntMaze-specific setting.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.antmaze.current_tfsn.collect import evaluate_supplied_goal_task
from scripts.antmaze.current_tfsn.constants import (
    CANDIDATES,
    EPISODE_STEPS,
    LANDING_SIGMA,
    STATE_DIM,
)
from scripts.antmaze.current_tfsn.executor import FaramaSACExecutor, make_env
from scripts.antmaze.current_tfsn.main import dump, paths

from projects.tfsn_consequence_value_iteration.antmaze.expanded_development import (
    PANEL_REVISION,
    output_paths as expanded_output_paths,
    prepare_panel,
)
from projects.tfsn_consequence_value_iteration.antmaze.mobility_stitched_planning import (
    LANDMARK_NEIGHBORS,
    MobilityModel,
    MobilityStitchedPlanner,
    _artifact_paths,
    _load_base,
    _prepare_landmarks,
    _verify_bridges,
    stitched_values,
)
from projects.tfsn_consequence_value_iteration.antmaze.recorded_transition_planning import (
    sha256_file,
)


METHOD_REVISION = "antmaze_r10_lazy_alias_advance_r13"
CLAIM_LEVEL = "consumed_development_long_route_diagnostic"
LONG_ROUTE_TASK_IDS = (9, 13, 14)


def alias_kernel(distance, sigma=LANDING_SIGMA):
    """Continuous probability-like association at the existing map scale."""
    distance = np.asarray(distance, np.float32)
    return np.exp(-np.square(distance) / (2.0 * float(sigma) ** 2)).astype(
        np.float32
    )


class LazyAliasAdvancePlanner(MobilityStitchedPlanner):
    """R10 plus one value-greedy outgoing action per landmark candidate."""

    method_revision = METHOD_REVISION
    planner_name = "r10_lazy_alias_advance"
    horizon = 10

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        order = np.argsort(self.graph.source_index, kind="stable")
        counts = np.bincount(
            self.graph.source_index, minlength=self.graph.node_count
        )
        self.outgoing_order = order.astype(np.int32)
        self.outgoing_offsets = np.concatenate(
            ([0], np.cumsum(counts, dtype=np.int64))
        )

    def _association_weights(self, association):
        """R13 adds the advance option without suppressing the R10 bridge."""
        return np.ones_like(association), association

    def _set_goal(self, fields):
        key = tuple(np.asarray(fields.goal, np.float32))
        if key != self.goal_key:
            self.values, self.coverage, self.node_field = stitched_values(
                self.graph,
                self.targets,
                self.weight,
                fields.goal,
                horizon=self.horizon,
            )
            self.goal_key = key

    def _greedy_outgoing(self, node_index, remaining):
        """Return R10's single frozen-graph action at each supplied node."""
        node_index = np.asarray(node_index, np.int32).reshape(-1)
        if remaining <= 0:
            raise ValueError("remaining horizon must be positive")

        model_target = self.targets[node_index]
        model_q = self.weight[node_index] * (
            self.node_field[model_target]
            + self.values[remaining - 1, model_target]
        )
        model_choice = np.argmax(model_q, axis=1)
        row = np.arange(len(node_index))
        target = model_target[row, model_choice].astype(np.int32)
        command = self.graph.nodes[target, :2].copy()
        reliability = np.ones(len(node_index), np.float32)
        offline_score = model_q[row, model_choice].astype(np.float32)
        kind = np.full(len(node_index), "alias_next_model_bridge", dtype=object)

        for output_index, node in enumerate(node_index):
            begin = self.outgoing_offsets[node]
            stop = self.outgoing_offsets[node + 1]
            if begin == stop:
                continue
            edges = self.outgoing_order[begin:stop]
            endpoints = self.graph.endpoint_index[edges]
            recorded_q = self.graph.edge_reliability[edges] * (
                self.node_field[endpoints]
                + self.values[remaining - 1, endpoints]
            )
            local = int(np.argmax(recorded_q))
            if recorded_q[local] <= offline_score[output_index]:
                continue
            edge = int(edges[local])
            command[output_index] = self.graph.commands[edge]
            target[output_index] = self.graph.endpoint_index[edge]
            reliability[output_index] = self.graph.edge_reliability[edge]
            offline_score[output_index] = recorded_q[local]
            kind[output_index] = "alias_next_recorded_edge"
        return command, target, reliability, offline_score, kind

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
        remaining = self.horizon - int(command_index)
        if remaining <= 0:
            raise ValueError("command index exceeds planning horizon")
        state = np.asarray(state, np.float32).reshape(STATE_DIM)

        distance, local = self.landmark_tree.query(
            state[:2],
            k=min(LANDMARK_NEIGHBORS, len(self.landmark_index)),
        )
        distance = np.atleast_1d(distance).astype(np.float32)
        bridge_target = self.landmark_index[np.atleast_1d(local)]
        bridge_command = self.graph.nodes[bridge_target, :2]
        association = alias_kernel(distance)
        bridge_reliability, advance_association = self._association_weights(
            association
        )
        bridge_q, bridge_weight = self._candidate_value(
            state,
            bridge_command,
            bridge_target,
            remaining,
            bridge_reliability,
            self.seed + command_index * 10_007,
        )

        recorded_command, recorded_edge, proposal = self.graph.sample_commands(
            self.rng, state, CANDIDATES
        )
        recorded_target = self.graph.endpoint_index[recorded_edge]
        recorded_q, recorded_weight = self._candidate_value(
            state,
            recorded_command,
            recorded_target,
            remaining,
            self.graph.edge_reliability[recorded_edge],
            self.seed + command_index * 10_007 + 97,
        )

        (
            advance_command,
            advance_target,
            advance_reliability,
            advance_offline_score,
            advance_kind,
        ) = self._greedy_outgoing(bridge_target, remaining)
        advance_q, advance_weight = self._candidate_value(
            state,
            advance_command,
            advance_target,
            remaining,
            advance_reliability * advance_association,
            self.seed + command_index * 10_007 + 193,
        )

        command = np.concatenate(
            (bridge_command, recorded_command, advance_command), axis=0
        )
        score = np.concatenate((bridge_q, recorded_q, advance_q), axis=0)
        connection = np.concatenate(
            (bridge_weight, recorded_weight, advance_weight), axis=0
        )
        kind = np.concatenate((
            np.full(len(bridge_command), "model_bridge", dtype=object),
            np.full(len(recorded_command), "recorded_edge", dtype=object),
            advance_kind,
        ))
        association_all = np.concatenate((
            np.zeros(len(bridge_command) + len(recorded_command), np.float32),
            association,
        ))
        offline_all = np.concatenate((
            np.zeros(len(bridge_command) + len(recorded_command), np.float32),
            advance_offline_score,
        ))
        winner = int(np.argmax(score))
        score_median = float(np.median(score))
        score_best = float(score[winner])
        score_ratio = (
            float(score_best / score_median) if score_median > 0 else None
        )
        diagnostic = {
            "planner": self.planner_name,
            "method_revision": self.method_revision,
            "phase": "supplied_goal",
            "remaining": int(remaining),
            "selected_score": float(score[winner]),
            "selected_kind": str(kind[winner]),
            "selected_connection_weight": float(connection[winner]),
            "selected_command_distance": float(
                np.linalg.norm(command[winner] - state[:2])
            ),
            "selected_alias_association": float(association_all[winner]),
            "selected_execution_partition_weight": float(
                np.concatenate((
                    bridge_reliability,
                    np.ones(len(recorded_command), np.float32),
                    advance_association,
                ))[winner]
            ),
            "selected_alias_offline_score": float(offline_all[winner]),
            "alias_advance_candidates": int(len(advance_command)),
            "maximum_alias_association": float(association.max()),
            "positive_value_fraction": self.coverage[-1]["positive_fraction"],
            "meaningful_value_fraction": self.coverage[-1][
                "meaningful_fraction"
            ],
            "top_scores": np.sort(score)[-5:][::-1].tolist(),
            "candidate_score_best": score_best,
            "candidate_score_median": score_median,
            "candidate_score_best_to_median_ratio": score_ratio,
            "candidate_score_median_is_zero": bool(score_median == 0),
            "search_enabled": False,
            "acquisition_search_mode": "supplied_goal",
            "fine_search_sigma": float(fields.sigma),
            "coarse_search_sigma": float(fields.sigma),
            "coarse_search_weight": 0.0,
            "fine_unsearched_mass": None,
            "coarse_unsearched_mass": None,
        }
        diagnostic.update(proposal)
        return command[winner].copy(), diagnostic


def output_directory(root):
    return (
        expanded_output_paths(root)["directory"]
        / "r10_lazy_alias_advance_r13_long3"
    )


def _load_planner(args, planner_class=LazyAliasAdvancePlanner):
    graph, consequence, consequence_path = _load_base(args)
    artifact, landmark_index, targets, _distance, _metadata = (
        _prepare_landmarks(args, graph, consequence_path)
    )
    with np.load(artifact["bridges"]) as data:
        saved_landmarks = data["landmark_index"]
        saved_targets = data["target_index"]
        weight = data["weight"]
    if not np.array_equal(landmark_index, saved_landmarks):
        raise ValueError("the frozen R10 landmark set changed")
    if not np.array_equal(targets, saved_targets):
        raise ValueError("the frozen R10 bridge targets changed")
    _verify_bridges(
        args,
        graph,
        consequence_path,
        artifact,
        landmark_index,
        targets,
        weight,
    )
    mobility = MobilityModel(artifact["mobility"], args.device)
    planner = planner_class(
        consequence,
        mobility,
        graph,
        landmark_index,
        targets,
        weight,
        np.random.default_rng(args.seed),
        args.seed,
    )
    hashes = {
        "graph_sha256": sha256_file(args.graph),
        "factorized_consequence_sha256": sha256_file(consequence_path),
        "mobility_sha256": sha256_file(artifact["mobility"]),
        "r10_bridges_sha256": sha256_file(artifact["bridges"]),
    }
    return planner, hashes


def _task(root, task_id):
    selected = prepare_panel(root)
    task_id = int(task_id)
    if task_id not in LONG_ROUTE_TASK_IDS:
        raise ValueError(f"task id must be one of {LONG_ROUTE_TASK_IDS}")
    return next(row for row in selected["tasks"] if row["task_id"] == task_id)


def command_task(args):
    task = _task(args.root, args.task_id)
    output = output_directory(args.root)
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / f"task_{task['task_id']:03d}.json"
    if result_path.exists():
        saved = json.loads(result_path.read_text())
        expected = (task["goal"], task["reset_seed"], METHOD_REVISION)
        actual = (
            saved.get("goal"),
            saved.get("reset_seed"),
            saved.get("method_revision"),
        )
        if actual != expected:
            raise ValueError(f"existing task result has another contract: {result_path}")
        print(json.dumps(saved))
        return

    planner, hashes = _load_planner(args)
    task_seed = int(args.seed) + 1_000_003 * int(task["task_id"])
    planner.rng = np.random.default_rng(task_seed)
    planner.seed = task_seed
    planner.goal_key = None
    executor = FaramaSACExecutor(paths(args.root)["executor"], args.device)
    env = make_env(max_episode_steps=10_000)
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
    env.close()
    row.update(task)
    row.update({
        "evaluation_arm": "r10_lazy_alias_advance",
        "method_revision": METHOD_REVISION,
        "claim_level": CLAIM_LEVEL,
        "task_gradient_updates": 0,
        "confirmation_panel_accessed": False,
        "fresh_confirmation_panel_accessed": False,
        "value_coverage": planner.coverage,
        **hashes,
    })
    dump(result_path, row)
    print(json.dumps(row))


def command_summarize(args):
    output = output_directory(args.root)
    rows = []
    for task_id in LONG_ROUTE_TASK_IDS:
        path = output / f"task_{task_id:03d}.json"
        if not path.exists():
            raise FileNotFoundError(path)
        rows.append(json.loads(path.read_text()))
    r10_root = expanded_output_paths(args.root)["r10"]
    references = [
        json.loads((r10_root / f"task_{task_id:03d}.json").read_text())
        for task_id in LONG_ROUTE_TASK_IDS
    ]
    result = {
        "panel_revision": PANEL_REVISION,
        "method_revision": METHOD_REVISION,
        "claim_level": CLAIM_LEVEL,
        "declared_deviations": [
            "three_consumed_long_route_failures_only",
            "one_value_greedy_outgoing_action_per_r10_landmark",
            "continuous_existing_sigma_alias_association",
        ],
        "oracle_inputs": [],
        "tasks": len(rows),
        "task_ids": list(LONG_ROUTE_TASK_IDS),
        "successes": int(sum(row["success"] for row in rows)),
        "r10_reference_successes": int(
            sum(row["success"] for row in references)
        ),
        "minimum_distance": {
            str(row["task_id"]): float(row["minimum_distance"])
            for row in rows
        },
        "r10_reference_minimum_distance": {
            str(row["task_id"]): float(row["minimum_distance"])
            for row in references
        },
        "confirmation_panel_accessed": False,
        "fresh_confirmation_panel_accessed": False,
        "task_gradient_updates": 0,
        "rows": rows,
    }
    dump(output / "summary.json", result)
    print(json.dumps(result, indent=2))


def parser():
    value = argparse.ArgumentParser()
    value.add_argument(
        "--root", type=Path, default=Path("/home/ai_diagnosis/SG/e377")
    )
    value.add_argument("--graph", type=Path)
    value.add_argument("--device", default="cuda")
    value.add_argument("--seed", type=int, default=605)
    subparsers = value.add_subparsers(dest="command", required=True)
    task = subparsers.add_parser("task")
    task.add_argument("--task-id", type=int, required=True)
    task.set_defaults(func=command_task)
    summarize = subparsers.add_parser("summarize")
    summarize.set_defaults(func=command_summarize)
    return value


def main():
    args = parser().parse_args()
    if args.graph is None:
        args.graph = (
            Path(args.root)
            / "model_planning"
            / "round2_recorded_transition_graph_r6.npz"
        )
    args.func(args)


if __name__ == "__main__":
    main()
