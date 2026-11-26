"""Hidden-goal full loop for the frozen R22 AntMaze planner.

The supplied-goal R22 graph, model, candidate set, finite-horizon max backup,
and continuous alias partition are unchanged.  This protocol supplies dynamic
TF-SN fields instead of a known goal:

* acquisition reads the residual-search field;
* after evaluator-only contact, return reads the empirical reward-rate field;
* task memory persists across physical resets and no task gradients occur.

The deployed factorized consequence model's landing precision (R22 sigma) is
used unchanged in both fields.  One-meter coverage cells are diagnostics only;
they never enter a score, candidate filter, or stopping rule.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from projects.tfsn_consequence_value_iteration.antmaze import (
    reach_based_r14 as r22,
)
from projects.tfsn_consequence_value_iteration.antmaze.successor_reductions import (
    finite_horizon_edge_values,
    value_coverage,
)
from scripts.antmaze.current_tfsn.collect import (
    execute_command,
    task_fields,
)
from scripts.antmaze.current_tfsn.constants import (
    COMMANDS_PER_EPISODE,
    COMMAND_STEPS,
)
from scripts.antmaze.current_tfsn.executor import (
    FaramaSACExecutor,
    full_state,
    make_env,
)
from scripts.antmaze.current_tfsn.main import dump, paths


METHOD_REVISION = "antmaze_hidden_goal_frozen_r22_r25"
BASE_METHOD_REVISION = r22.METHOD_REVISION
CLAIM_LEVEL = "consumed_development_hidden_goal_full_loop"
ALL_TASK_IDS = tuple(range(15))
# Frozen-reference selection: one supplied-goal success requiring a detour and
# one supplied-goal success also solved by the direct executor.
GATE_TASK_IDS = (3, 5)
DEFAULT_EPISODES = 5


def frozen_search_sigma():
    return float(r22.SIGMA)


def dynamic_stitched_values(graph, targets, weight, node_field, horizon=10):
    values = finite_horizon_edge_values(
        graph,
        targets,
        weight,
        np.asarray(node_field, np.float32),
        int(horizon),
        reduction="max",
    )
    return values, value_coverage(values)


class HiddenGoalR22Planner(r22.ReachBasedR14Planner):
    method_revision = METHOD_REVISION
    planner_name = "hidden_goal_frozen_r22"

    def configure_hidden_fields(self, search_sigma):
        if float(search_sigma) != float(r22.SIGMA):
            raise ValueError("hidden acquisition must use frozen R22 sigma")
        self.acquisition_search_mode = "fine"
        self.coarse_search_sigma = float(search_sigma)
        self.coverage_reference_xy = None

    def _set_goal(self, fields):
        # Dynamic task memory changes the field after every command, so the
        # frozen R22 max backup is reread rather than cached by a goal key.
        self.node_field = fields.field(self.graph.nodes[:, :2]).astype(
            np.float32
        )
        self.values, self.coverage = dynamic_stitched_values(
            self.graph,
            self.targets,
            self.weight,
            self.node_field,
            horizon=self.horizon,
        )
        self.goal_key = (
            "dynamic_return" if fields.contacted else "dynamic_acquisition",
            len(fields.coords),
        )

    def choose(
        self,
        state,
        command_index,
        fields,
        online_states=(),
        online_trajectories=(),
    ):
        command, diagnostic = super().choose(
            state,
            command_index,
            fields,
            online_states,
            online_trajectories,
        )
        diagnostic.update({
            "planner": self.planner_name,
            "method_revision": self.method_revision,
            "phase": "return" if fields.contacted else "acquisition",
            "search_enabled": not fields.contacted,
            "acquisition_search_mode": fields.search_mode,
            "fine_search_sigma": float(fields.sigma),
            "coarse_search_sigma": float(fields.coarse_sigma),
            "coarse_search_weight": float(fields.coarse_weight),
            "fine_unsearched_mass": fields.fine_unsearched_mass,
            "coarse_unsearched_mass": fields.coarse_unsearched_mass,
        })
        return command, diagnostic


def output_directory(root, output_name):
    return paths(root)["eval"] / str(output_name)


def _coverage_cell(xy):
    """One-meter cell for reporting only; never read by the planner."""
    # Return native Python integers so the diagnostic can be written directly
    # as JSON.  This representation is reporting-only and never reaches the
    # planner.
    return tuple(
        int(value)
        for value in np.floor(np.asarray(xy, np.float32)).astype(np.int64)
    )


def evaluate_hidden_task(
    env, executor, planner, goal, reset_seed, episodes, sigma, log_path,
    commands_per_episode=COMMANDS_PER_EPISODE,
    planner_horizon_commands=COMMANDS_PER_EPISODE,
):
    fields, online, completed_trajectories = (
        task_fields(planner, sigma), [], []
    )
    first_contact_episode = None
    first_contact_step = None
    successes, logs, elapsed_steps = [], [], 0
    visited_cells = set()
    acquisition_endpoint_cells = []
    acquisition_commands = 0
    acquisition_new_cells = 0
    acquisition_zero_new = 0
    acquisition_two_cycles = 0
    acquisition_revisited_endpoints = 0

    for episode in range(int(episodes)):
        observation, _ = env.reset(seed=int(reset_seed))
        state = full_state(observation)
        current_trajectory = [state.copy()]
        fields.add(state[:2], 0)
        visited_cells.add(_coverage_cell(state[:2]))
        episode_success = False
        for command_index in range(int(commands_per_episode)):
            fields.rebuild()
            phase_before = "return" if fields.contacted else "acquisition"
            trajectories = completed_trajectories + [current_trajectory]
            planner_command_index = (
                int(command_index) % int(planner_horizon_commands)
            )
            command, diagnostic = planner.choose(
                state, planner_command_index, fields, online, trajectories
            )
            trajectory, rewards = execute_command(
                env, executor, state, command, goal
            )
            command_cells = {
                _coverage_cell(xy) for xy in trajectory[1:, :2]
            }
            new_cells = command_cells - visited_cells
            endpoint_cell = _coverage_cell(trajectory[-1, :2])
            revisited_endpoint = endpoint_cell in visited_cells
            two_cycle = bool(
                len(acquisition_endpoint_cells) >= 2
                and endpoint_cell == acquisition_endpoint_cells[-2]
                and endpoint_cell != acquisition_endpoint_cells[-1]
            )
            if phase_before == "acquisition":
                acquisition_commands += 1
                acquisition_new_cells += len(new_cells)
                acquisition_zero_new += int(not new_cells)
                acquisition_revisited_endpoints += int(revisited_endpoint)
                acquisition_two_cycles += int(two_cycle)
                acquisition_endpoint_cells.append(endpoint_cell)
            visited_cells.update(command_cells)
            for next_state, reward in zip(trajectory[1:], rewards):
                fields.add(next_state[:2], reward)
                online.append(next_state.copy())
                current_trajectory.append(next_state.copy())
            if first_contact_step is None and rewards.any():
                first_contact_step = (
                    elapsed_steps + int(np.flatnonzero(rewards)[0]) + 1
                )
            elapsed_steps += COMMAND_STEPS
            episode_success = episode_success or bool(rewards.any())
            diagnostic.update({
                "episode": int(episode),
                "command_index": int(command_index),
                "planner_command_index": int(planner_command_index),
                "command": np.asarray(command).tolist(),
                "actual_endpoint": trajectory[-1, :2].tolist(),
                "moved_distance": float(np.linalg.norm(
                    trajectory[-1, :2] - state[:2]
                )),
                "reward": bool(rewards.any()),
                "coverage_cell_size_m_diagnostic_only": 1.0,
                "new_1m_cells": int(len(new_cells)),
                "cumulative_1m_cells": int(len(visited_cells)),
                "revisited_endpoint_1m_cell": bool(revisited_endpoint),
                "endpoint_1m_cell": list(endpoint_cell),
                "acquisition_two_cycle_1m_cell": (
                    bool(two_cycle) if phase_before == "acquisition" else None
                ),
            })
            logs.append(diagnostic)
            state = trajectory[-1]
            if episode_success:
                if first_contact_episode is None:
                    first_contact_episode = episode
                break
        completed_trajectories.append(np.asarray(current_trajectory))
        successes.append(episode_success)

    Path(log_path).write_text(
        "\n".join(json.dumps(row) for row in logs) + "\n"
    )
    return {
        "successes": successes,
        "first_contact_episode": first_contact_episode,
        "contact_latency": (
            first_contact_step
            if first_contact_step is not None
            else int(episodes) * int(commands_per_episode) * COMMAND_STEPS
        ),
        "mean_e2e": float(np.mean(successes)),
        "acq_at_1": first_contact_episode == 0,
        "acq_at_2": (
            first_contact_episode is not None and first_contact_episode < 2
        ),
        "acq_at_5": (
            first_contact_episode is not None and first_contact_episode < 5
        ),
        "conditional_return": (
            float(np.mean(successes[first_contact_episode + 1:]))
            if first_contact_episode is not None
            and first_contact_episode + 1 < int(episodes)
            else None
        ),
        "acquisition_commands": int(acquisition_commands),
        "acquisition_new_1m_cells": int(acquisition_new_cells),
        "acquisition_new_1m_cells_per_command": (
            float(acquisition_new_cells / acquisition_commands)
            if acquisition_commands else None
        ),
        "acquisition_zero_new_1m_cell_fraction": (
            float(acquisition_zero_new / acquisition_commands)
            if acquisition_commands else None
        ),
        "acquisition_revisited_endpoint_fraction": (
            float(acquisition_revisited_endpoints / acquisition_commands)
            if acquisition_commands else None
        ),
        "acquisition_two_cycle_count": int(acquisition_two_cycles),
        "coverage_cell_size_m_diagnostic_only": 1.0,
    }


def _load_runtime(args):
    graph = r22._verify_graph(args)
    planner, hashes = r22.lazy._load_planner(args, HiddenGoalR22Planner)
    search_sigma = frozen_search_sigma()
    planner.configure_hidden_fields(search_sigma)
    planner.artifact_hashes = {
        **hashes,
        "reach_landmarks_sha256": r22.sha256_file(
            r22.artifact_paths(args.root)["landmarks"]
        ),
    }
    executor = FaramaSACExecutor(paths(args.root)["executor"], args.device)
    env = make_env(max_episode_steps=10_000_000)
    return planner, executor, env, search_sigma


def _evaluate_task(args, planner, executor, env, search_sigma, task_id):
    task = r22._task(args.root, task_id)
    output = output_directory(args.root, args.output_name)
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / f"task_{task_id:03d}.json"
    if result_path.exists():
        saved = json.loads(result_path.read_text())
        expected = (
            task["goal"], task["reset_seed"], METHOD_REVISION,
            int(args.episodes), float(search_sigma), float(r22.SIGMA),
        )
        actual = (
            saved.get("goal"), saved.get("reset_seed"),
            saved.get("method_revision"), saved.get("episodes"),
            saved.get("acquisition_search_sigma_m"),
            saved.get("return_field_sigma_m"),
        )
        if actual != expected:
            raise ValueError(f"existing task has another contract: {result_path}")
        print(json.dumps(saved))
        return saved

    task_seed = int(args.seed) + 1_000_003 * int(task_id)
    planner.rng = np.random.default_rng(task_seed)
    planner.seed = task_seed
    planner.goal_key = None
    row = evaluate_hidden_task(
        env,
        executor,
        planner,
        np.asarray(task["goal"], np.float32),
        task["reset_seed"],
        int(args.episodes),
        r22.SIGMA,
        output / f"task_{task_id:03d}.jsonl",
    )
    row.update(task)
    row.update({
        "evaluation_arm": "hidden_goal_frozen_r22_full_loop",
        "method_revision": METHOD_REVISION,
        "base_method_revision": BASE_METHOD_REVISION,
        "claim_level": CLAIM_LEVEL,
        "episodes": int(args.episodes),
        "hidden_goal_visible_to_planner_executor_or_models": False,
        "evaluator_uses_hidden_goal_for_contact_only": True,
        "task_memory_persisted_across_physical_resets": True,
        "acquisition_search_mode": "fine",
        "acquisition_search_sigma_m": float(search_sigma),
        "acquisition_search_sigma_estimator": (
            "frozen_R22_deployed_factorized_consequence_landing_precision"
        ),
        "return_field_sigma_m": float(r22.SIGMA),
        "return_field_sigma_estimator": r22.ESTIMATOR,
        "task_gradient_updates": 0,
        "confirmation_panel_accessed": False,
        "fresh_confirmation_panel_accessed": False,
        **planner.artifact_hashes,
    })
    dump(result_path, row)
    print(json.dumps(row), flush=True)
    return row


def _task_ids(args):
    if args.set_name == "gate2":
        return GATE_TASK_IDS
    if args.set_name == "all15":
        return ALL_TASK_IDS
    raise ValueError(args.set_name)


def command_evaluate_set(args):
    task_ids = _task_ids(args)
    planner, executor, env, search_sigma = _load_runtime(args)
    try:
        for task_id in task_ids:
            _evaluate_task(
                args, planner, executor, env, search_sigma, task_id
            )
    finally:
        env.close()
    command_summarize(args)


def command_summarize(args):
    task_ids = _task_ids(args)
    output = output_directory(args.root, args.output_name)
    rows = []
    for task_id in task_ids:
        path = output / f"task_{task_id:03d}.json"
        if not path.exists():
            raise FileNotFoundError(path)
        rows.append(json.loads(path.read_text()))
    episodes = int(args.episodes)
    first_contact = [row["first_contact_episode"] for row in rows]
    all_successes = [
        bool(value) for row in rows for value in row["successes"]
    ]
    post_contact = []
    for row in rows:
        contact = row["first_contact_episode"]
        if contact is not None:
            post_contact.extend(row["successes"][int(contact) + 1:])
    result = {
        "schema_version": 1,
        "method_revision": METHOD_REVISION,
        "base_method_revision": BASE_METHOD_REVISION,
        "claim_level": CLAIM_LEVEL,
        "declared_deviations": [
            "consumed_development_goals",
            "dynamic_hidden_goal_fields_over_frozen_r22_planner",
            "one_meter_coverage_cells_are_diagnostic_only",
        ],
        "oracle_inputs": [],
        "set_name": args.set_name,
        "task_ids": list(task_ids),
        "tasks": len(rows),
        "episodes_per_task": episodes,
        "physical_episodes": len(rows) * episodes,
        "acquired_tasks": int(sum(value is not None for value in first_contact)),
        "acquisition_at_1": float(np.mean([
            value == 0 for value in first_contact
        ])),
        "acquisition_at_2": float(np.mean([
            value is not None and value < 2 for value in first_contact
        ])),
        "acquisition_at_5": float(np.mean([
            value is not None and value < 5 for value in first_contact
        ])),
        "mean_e2e": float(np.mean(all_successes)),
        "successful_episodes": int(sum(all_successes)),
        "pooled_conditional_return": (
            float(np.mean(post_contact)) if post_contact else None
        ),
        "post_contact_episode_opportunities": len(post_contact),
        "contact_latency_mean_steps": float(np.mean([
            row["contact_latency"] for row in rows
        ])),
        "acquisition_new_1m_cells_per_command": float(np.mean([
            row["acquisition_new_1m_cells_per_command"] for row in rows
            if row["acquisition_new_1m_cells_per_command"] is not None
        ])),
        "acquisition_zero_new_1m_cell_fraction": float(np.mean([
            row["acquisition_zero_new_1m_cell_fraction"] for row in rows
            if row["acquisition_zero_new_1m_cell_fraction"] is not None
        ])),
        "acquisition_revisited_endpoint_fraction": float(np.mean([
            row["acquisition_revisited_endpoint_fraction"] for row in rows
            if row["acquisition_revisited_endpoint_fraction"] is not None
        ])),
        "acquisition_two_cycle_count": int(sum(
            row["acquisition_two_cycle_count"] for row in rows
        )),
        "acquisition_search_sigma_m": rows[0][
            "acquisition_search_sigma_m"
        ],
        "return_field_sigma_m": rows[0]["return_field_sigma_m"],
        "hidden_goal_visible_to_planner_executor_or_models": False,
        "task_gradient_updates": 0,
        "confirmation_panel_accessed": False,
        "fresh_confirmation_panel_accessed": False,
        "rows": rows,
    }
    dump(output / "summary.json", result)
    print(json.dumps({
        key: result[key]
        for key in (
            "tasks", "physical_episodes", "acquired_tasks",
            "acquisition_at_1", "acquisition_at_2", "acquisition_at_5",
            "pooled_conditional_return", "mean_e2e",
            "contact_latency_mean_steps",
            "acquisition_new_1m_cells_per_command",
            "acquisition_zero_new_1m_cell_fraction",
            "acquisition_revisited_endpoint_fraction",
            "acquisition_two_cycle_count",
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
    evaluate = subparsers.add_parser("evaluate-set")
    evaluate.add_argument("--set-name", choices=("gate2", "all15"), required=True)
    evaluate.add_argument("--episodes", type=int, default=DEFAULT_EPISODES)
    evaluate.add_argument("--output-name", required=True)
    evaluate.set_defaults(func=command_evaluate_set)
    summarize = subparsers.add_parser("summarize")
    summarize.add_argument("--set-name", choices=("gate2", "all15"), required=True)
    summarize.add_argument("--episodes", type=int, default=DEFAULT_EPISODES)
    summarize.add_argument("--output-name", required=True)
    summarize.set_defaults(func=command_summarize)
    return value


def main():
    args = parser().parse_args()
    if args.graph is None:
        args.graph = r22.graph_path(args.root)
    if int(args.episodes) <= 0:
        raise ValueError("episodes must be positive")
    args.func(args)


if __name__ == "__main__":
    main()
