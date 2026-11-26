"""Bellman-consistent live endpoint valuation for the frozen R10 graph.

R10 values a live command by whether its predicted endpoint can inherit the
value of one intended recorded node. R14 repairs the resulting spatial alias
with a continuous partition. This module removes both mechanisms from the live
backup. A sampled endpoint receives value only through commands that the
planner can evaluate from that endpoint's own full predicted state.

For each live candidate g and consequence-ensemble member:

    s' ~ P_member(. | s, g)
    Q(s, g) = F(phi(s')) + max_g' Q_graph(s', g')

The second action is evaluated by the same consequence ensemble from s'. Its
realized s'' receives the exact field F(phi(s'')); only then can it connect to
the frozen finite-horizon graph value with the remaining budget. Thus a stall
does not inherit another recorded state's options. There is no alias distance
test, suppression term, or bridge/advance special case.

One stratified model draw per ensemble member is a global computational Monte
Carlo setting. The candidate count remains the existing global CANDIDATES for
both landmark and recorded-command proposals.
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

from projects.tfsn_consequence_value_iteration.antmaze.alias_partition_advance import (
    ALL_DEVELOPMENT_TASK_IDS,
    _r10_reference_rows,
    _task,
)
from projects.tfsn_consequence_value_iteration.antmaze.expanded_development import (
    PANEL_REVISION,
    output_paths as expanded_output_paths,
)
from projects.tfsn_consequence_value_iteration.antmaze.lazy_alias_advance import (
    LONG_ROUTE_TASK_IDS,
    LazyAliasAdvancePlanner,
    _load_planner,
)
from projects.tfsn_consequence_value_iteration.antmaze.mobility_stitched_planning import (
    LANDMARK_NEIGHBORS,
)
from projects.tfsn_consequence_value_iteration.antmaze.recorded_transition_planning import (
    consequence_inputs,
)


METHOD_REVISION = "antmaze_bellman_consistent_live_endpoints_r16"
CLAIM_LEVEL = "expanded_consumed_development_planner_diagnostic"
SAMPLES_PER_ENSEMBLE_MEMBER = 1
PAIR_BATCH_SIZE = 2048


def bellman_endpoint_backup(immediate_field, own_option_value):
    """Undiscounted finite-horizon backup at an endpoint's own state."""
    immediate_field = np.asarray(immediate_field, np.float32)
    own_option_value = np.asarray(own_option_value, np.float32)
    if immediate_field.shape != own_option_value.shape:
        raise ValueError("Bellman backup terms must align")
    return immediate_field + own_option_value


class BellmanConsistentPlanner(LazyAliasAdvancePlanner):
    method_revision = METHOD_REVISION
    planner_name = "bellman_consistent_live_endpoints"

    def _terminal_graph_values(
        self,
        states,
        commands,
        target_index,
        reliability,
        graph_remaining,
        fields,
        seed,
    ):
        """Evaluate second actions from their own sources into graph values."""
        states = np.asarray(states, np.float32).reshape(-1, STATE_DIM)
        commands = np.asarray(commands, np.float32).reshape(-1, 2)
        target_index = np.asarray(target_index, np.int32).reshape(-1)
        reliability = np.asarray(reliability, np.float32).reshape(-1)
        if not (
            len(states)
            == len(commands)
            == len(target_index)
            == len(reliability)
        ):
            raise ValueError("second-action arrays do not align")
        values = np.empty(len(states), np.float32)
        for begin in range(0, len(states), PAIR_BATCH_SIZE):
            stop = min(begin + PAIR_BATCH_SIZE, len(states))
            source = states[begin:stop]
            command = commands[begin:stop]
            target = target_index[begin:stop]
            inputs = consequence_inputs(source, command)
            target_state = self.graph.nodes[target]
            member_q = []
            for member in range(len(self.consequence.members)):
                rng = np.random.default_rng(
                    seed + begin * 1009 + member * 1_000_003
                )
                delta = self.consequence.sample_member(
                    inputs,
                    member,
                    SAMPLES_PER_ENSEMBLE_MEMBER,
                    rng,
                )
                endpoint = source[:, None] + delta
                immediate = fields.field(
                    endpoint[..., :2].reshape(-1, 2)
                ).reshape(len(source), SAMPLES_PER_ENSEMBLE_MEMBER)
                if graph_remaining > 0:
                    spatial = np.exp(
                        -np.square(
                            endpoint[..., :2] - target_state[:, None, :2]
                        ).sum(axis=2)
                        / (2.0 * LANDING_SIGMA**2)
                    )
                    endpoint_mobility = self.mobility.predict(
                        endpoint.reshape(-1, STATE_DIM)
                    ).reshape(
                        len(source), SAMPLES_PER_ENSEMBLE_MEMBER
                    )
                    continuation = (
                        spatial
                        * endpoint_mobility
                        * self.values[graph_remaining, target, None]
                    )
                else:
                    continuation = np.zeros_like(immediate)
                member_q.append(
                    (immediate + continuation).mean(axis=1)
                    * reliability[begin:stop]
                )
            values[begin:stop] = np.stack(member_q, axis=1).min(axis=1)
        return values

    def _own_option_values(self, endpoints, remaining, fields, seed):
        """Max over commands evaluated from each endpoint's full state."""
        endpoints = np.asarray(endpoints, np.float32).reshape(-1, STATE_DIM)
        if remaining <= 0:
            return np.zeros(len(endpoints), np.float32)

        _distance, local = self.landmark_tree.query(
            endpoints[:, :2],
            k=min(LANDMARK_NEIGHBORS, len(self.landmark_index)),
        )
        if np.asarray(local).ndim == 1:
            local = np.asarray(local)[:, None]
        landmark_target = self.landmark_index[np.asarray(local, np.int32)]
        landmark_command = self.graph.nodes[landmark_target, :2]

        recorded_commands = []
        recorded_edges = []
        proposal_rng = np.random.default_rng(seed + 37)
        for endpoint in endpoints:
            command, edge, _diagnostic = self.graph.sample_commands(
                proposal_rng, endpoint, CANDIDATES
            )
            recorded_commands.append(command)
            recorded_edges.append(edge)
        recorded_command = np.stack(recorded_commands)
        recorded_edge = np.stack(recorded_edges)
        recorded_target = self.graph.endpoint_index[recorded_edge]

        command = np.concatenate(
            (landmark_command, recorded_command), axis=1
        )
        target = np.concatenate(
            (landmark_target, recorded_target), axis=1
        )
        reliability = np.concatenate((
            np.ones(landmark_target.shape, np.float32),
            self.graph.edge_reliability[recorded_edge],
        ), axis=1)
        sources = np.repeat(
            endpoints[:, None], command.shape[1], axis=1
        )
        candidate_value = self._terminal_graph_values(
            sources.reshape(-1, STATE_DIM),
            command.reshape(-1, 2),
            target.reshape(-1),
            reliability.reshape(-1),
            graph_remaining=remaining - 1,
            fields=fields,
            seed=seed + 101,
        ).reshape(len(endpoints), command.shape[1])
        return candidate_value.max(axis=1)

    def _bellman_candidate_values(
        self, state, commands, remaining, fields, seed
    ):
        state = np.asarray(state, np.float32).reshape(STATE_DIM)
        commands = np.asarray(commands, np.float32).reshape(-1, 2)
        source = np.repeat(state[None], len(commands), axis=0)
        inputs = consequence_inputs(source, commands)
        member_q = []
        member_immediate = []
        member_own = []
        for member in range(len(self.consequence.members)):
            rng = np.random.default_rng(seed + member * 1_000_003)
            delta = self.consequence.sample_member(
                inputs,
                member,
                SAMPLES_PER_ENSEMBLE_MEMBER,
                rng,
            )
            endpoints = (source[:, None] + delta).reshape(-1, STATE_DIM)
            immediate = fields.field(endpoints[:, :2])
            own = self._own_option_values(
                endpoints,
                remaining=remaining - 1,
                fields=fields,
                seed=seed + member * 2_000_003 + 503,
            )
            q = bellman_endpoint_backup(immediate, own)
            member_q.append(q.reshape(len(commands), -1).mean(axis=1))
            member_immediate.append(
                immediate.reshape(len(commands), -1).mean(axis=1)
            )
            member_own.append(
                own.reshape(len(commands), -1).mean(axis=1)
            )
        member_q = np.stack(member_q, axis=1)
        return (
            member_q.min(axis=1),
            member_q,
            np.stack(member_immediate, axis=1),
            np.stack(member_own, axis=1),
        )

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

        _distance, local = self.landmark_tree.query(
            state[:2],
            k=min(LANDMARK_NEIGHBORS, len(self.landmark_index)),
        )
        landmark_target = self.landmark_index[np.atleast_1d(local)]
        landmark_command = self.graph.nodes[landmark_target, :2]
        recorded_command, _recorded_edge, proposal = self.graph.sample_commands(
            self.rng, state, CANDIDATES
        )
        command = np.concatenate((landmark_command, recorded_command), axis=0)
        kind = np.asarray(
            ["landmark_command"] * len(landmark_command)
            + ["recorded_command"] * len(recorded_command)
        )
        score, member_q, member_immediate, member_own = (
            self._bellman_candidate_values(
                state,
                command,
                remaining,
                fields,
                self.seed + command_index * 10_007,
            )
        )
        winner = int(np.argmax(score))
        top = np.argsort(-score, kind="stable")[:5]
        diagnostic = {
            "planner": self.planner_name,
            "method_revision": self.method_revision,
            "phase": "supplied_goal",
            "remaining": int(remaining),
            "selected_score": float(score[winner]),
            "selected_kind": str(kind[winner]),
            "selected_command_distance": float(
                np.linalg.norm(command[winner] - state[:2])
            ),
            "selected_member_scores": member_q[winner].tolist(),
            "selected_member_immediate": member_immediate[winner].tolist(),
            "selected_member_own_option": member_own[winner].tolist(),
            "top_scores": score[top].tolist(),
            "top_commands": command[top].tolist(),
            "first_endpoint_samples_per_member": int(
                SAMPLES_PER_ENSEMBLE_MEMBER
            ),
            "second_endpoint_samples_per_member": int(
                SAMPLES_PER_ENSEMBLE_MEMBER
            ),
            "alias_suppression_used": False,
            "alias_distance_test_used": False,
            "predicted_endpoint_inherits_intended_node_value": False,
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
        / "bellman_consistent_live_endpoints_r16"
    )


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
            raise ValueError(
                f"existing task result has another contract: {result_path}"
            )
        print(json.dumps(saved))
        return

    planner, hashes = _load_planner(args, BellmanConsistentPlanner)
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
        "evaluation_arm": "bellman_consistent_live_endpoints",
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
    for task_id in ALL_DEVELOPMENT_TASK_IDS:
        task_path = output / f"task_{task_id:03d}.json"
        if not task_path.exists():
            raise FileNotFoundError(task_path)
        rows.append(json.loads(task_path.read_text()))
    references = _r10_reference_rows(args.root)
    r14 = json.loads((
        expanded_output_paths(args.root)["directory"]
        / "r10_partitioned_alias_advance_r14_long3"
        / "summary_all15.json"
    ).read_text())["rows"]
    successes = int(sum(row["success"] for row in rows))
    result = {
        "panel_revision": PANEL_REVISION,
        "method_revision": METHOD_REVISION,
        "claim_level": CLAIM_LEVEL,
        "declared_deviations": [
            "fifteen_consumed_development_goals",
            "one_stratified_endpoint_draw_per_ensemble_member",
        ],
        "oracle_inputs": [],
        "tasks": len(rows),
        "successes": successes,
        "success_rate": float(successes / len(rows)),
        "r10_reference_successes": int(
            sum(row["success"] for row in references)
        ),
        "r14_target_successes": int(sum(row["success"] for row in r14)),
        "paired_vs_r14": {
            "gained_task_ids": [
                int(row["task_id"])
                for row, reference in zip(rows, r14)
                if row["success"] and not reference["success"]
            ],
            "lost_task_ids": [
                int(row["task_id"])
                for row, reference in zip(rows, r14)
                if not row["success"] and reference["success"]
            ],
        },
        "alias_suppression_used": False,
        "alias_distance_test_used": False,
        "confirmation_panel_accessed": False,
        "fresh_confirmation_panel_accessed": False,
        "task_gradient_updates": 0,
        "rows": rows,
    }
    dump(output / "summary.json", result)
    print(json.dumps({
        key: result[key]
        for key in (
            "successes",
            "tasks",
            "success_rate",
            "r14_target_successes",
            "paired_vs_r14",
        )
    }, indent=2))


def command_summarize_long3(args):
    output = output_directory(args.root)
    rows = []
    for task_id in LONG_ROUTE_TASK_IDS:
        task_path = output / f"task_{task_id:03d}.json"
        if not task_path.exists():
            raise FileNotFoundError(task_path)
        rows.append(json.loads(task_path.read_text()))
    r14_path = (
        expanded_output_paths(args.root)["directory"]
        / "r10_partitioned_alias_advance_r14_long3"
        / "summary.json"
    )
    r14 = json.loads(r14_path.read_text())["rows"]
    result = {
        "panel_revision": PANEL_REVISION,
        "method_revision": METHOD_REVISION,
        "claim_level": "consumed_development_long_route_diagnostic",
        "declared_deviations": [
            "three_consumed_long_route_failures_only",
            "one_stratified_endpoint_draw_per_ensemble_member",
        ],
        "oracle_inputs": [],
        "tasks": len(rows),
        "task_ids": list(LONG_ROUTE_TASK_IDS),
        "successes": int(sum(row["success"] for row in rows)),
        "r14_reference_successes": int(
            sum(row["success"] for row in r14)
        ),
        "minimum_distance": {
            str(row["task_id"]): float(row["minimum_distance"])
            for row in rows
        },
        "r14_reference_minimum_distance": {
            str(row["task_id"]): float(row["minimum_distance"])
            for row in r14
        },
        "alias_suppression_used": False,
        "alias_distance_test_used": False,
        "confirmation_panel_accessed": False,
        "fresh_confirmation_panel_accessed": False,
        "task_gradient_updates": 0,
        "rows": rows,
    }
    dump(output / "summary_long3.json", result)
    print(json.dumps({
        key: result[key]
        for key in (
            "successes",
            "tasks",
            "minimum_distance",
            "r14_reference_minimum_distance",
        )
    }, indent=2))


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
    summarize_long3 = subparsers.add_parser("summarize-long3")
    summarize_long3.set_defaults(func=command_summarize_long3)
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
