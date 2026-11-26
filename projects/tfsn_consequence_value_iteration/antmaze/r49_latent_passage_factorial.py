"""R49 AntMaze adapter for the shared latent-detection/passage 2x2.

This is development-only and leaves R48 immutable.  It preserves R48's live
candidate families, exact consequence integration, and R46 correction.  The
advance-family prefilter is made observation-neutral so no old Gaussian memory
can leak into a latent-radius arm.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from projects.tfsn_consequence_value_iteration.antmaze import r48_rule_free_r39 as r48
from projects.tfsn_consequence_value_iteration.shared.factorial_acquisition import (
    FactorialAcquisition,
    OBSERVATION_LEVELS,
    PASSAGE_LEVELS,
)
from projects.tfsn_consequence_value_iteration.shared.latent_detection import (
    PosteriorFalsifiedError,
)
from projects.tfsn_consequence_value_iteration.shared.learned_passage import (
    LearnedPassageEnsemble,
)
from projects.tfsn_consequence_value_iteration.shared.r12_settings import (
    R12_CANDIDATE_BUDGET,
)
from scripts.antmaze.current_tfsn.executor import FaramaSACExecutor, full_state, make_env
from scripts.antmaze.current_tfsn.fields import TaskFields
from scripts.antmaze.current_tfsn.main import dump, paths


HERE = Path(__file__).resolve().parent
MANIFEST = HERE.parent / "shared" / "method_manifest_r12.json"
PROTOCOL = HERE / "r49_latent_passage_factorial_protocol.json"
REFERENCE = HERE / "R49_LATENT_PASSAGE_FACTORIAL_REFERENCE.md"
METHOD_REVISION = "antmaze_latent_detection_passage_factorial_r49"
CLAIM_LEVEL = "consumed_development_factorial_only"
RUN_LABEL = "r49"
DEFAULT_BASE = (
    HERE.parent / "results" / "antmaze" / "r39_r46_rule_removal_r48_full15"
)
DEFAULT_OUTPUT = HERE.parent / "results" / "latent_detection_passage_r12" / "antmaze"
TASK_IDS = (1, 4)
REQUIRED_EPISODES = 3
REQUIRE_BASELINE_MISSES = False


def verify_baseline_miss_selection(args, selected):
    if not REQUIRE_BASELINE_MISSES:
        return []
    protocol = json.loads(PROTOCOL.read_text())
    evidence = {
        int(row["task_id"]): row for row in protocol["selection_evidence"]
    }
    if set(evidence) != set(selected):
        raise ValueError("baseline miss evidence does not match selected tasks")
    verified = []
    for task_id in selected:
        row = evidence[task_id]
        source = args.base_output / row["relative_path"]
        if r48.sha256(source) != row["sha256"]:
            raise ValueError(f"R48 selection evidence changed: {source}")
        result = json.loads(source.read_text())
        if int(result["task_id"]) != task_id:
            raise ValueError(f"R48 selection evidence has wrong task: {source}")
        if result["first_contact_episode"] is not None:
            raise ValueError(f"selected R48 task was acquired: {source}")
        verified.append(str(source))
    return verified


def matched_passages(geometry, start_xy, endpoint_xy):
    start = np.asarray(start_xy, np.float32).reshape(-1, 2)
    endpoint = np.asarray(endpoint_xy, np.float32).reshape(-1, 2)
    query = np.concatenate((start, endpoint - start), axis=1) / float(r48.SIGMA)
    _distance, edge = geometry.tree.query(query, k=1, workers=-1)
    middle = start[:, None] + geometry.relative[np.asarray(edge, np.int32)]
    return np.concatenate((start[:, None], middle, endpoint[:, None]), axis=1)


class FactorialAntPlanner(r48.RuleFreeOnlinePlanner):
    method_revision = METHOD_REVISION
    planner_name = "r49_factorial_ant_adapter"

    def configure_factorial(self, observation_level, passage_level, learned):
        self.factorial = FactorialAcquisition(
            recorded_goal_xy=self.graph.nodes[:, :2],
            sigma=r48.SIGMA,
            observation_level=observation_level,
            passage_level=passage_level,
            learned_passage=learned,
            matched_passage=lambda start, endpoint: matched_passages(
                self.passage_geometry, start, endpoint
            ),
        )

    def reset_online_correction_task(self):
        super().reset_online_correction_task()
        if hasattr(self, "factorial"):
            self.factorial.reset()

    def observe_detection_consequence(self, xy, reward):
        self.factorial.observe(xy, reward)

    def _set_goal(self, fields):
        if fields.contacted:
            return super()._set_goal(fields)
        self.factorial.posterior.assert_not_exhausted("AntMaze acquisition decision")
        self.acquisition_active = True
        self.current_fields = fields
        probability = self.factorial.posterior.goal_probability
        positive = float(np.mean(probability > 0.0))
        self.coverage = [{
            "positive_fraction": positive,
            "meaningful_fraction": positive,
        }]
        self.goal_key = ("r49_factorial_acquisition", len(fields.coords))

    def _factorial_gain(self, state, commands, target_index):
        commands = np.asarray(commands, np.float32).reshape(-1, 2)
        states = np.repeat(
            np.asarray(state, np.float32).reshape(1, r48.STATE_DIM),
            len(commands), axis=0,
        )
        endpoint = self.graph.nodes[np.asarray(target_index, np.int32), :2]
        return self.factorial.gains(states, commands, endpoint)

    def _candidate_value(
        self, state, commands, target_index, remaining, reliability, seed,
        return_members=False,
    ):
        if not self.acquisition_active:
            return super()._candidate_value(
                state, commands, target_index, remaining, reliability, seed,
                return_members=return_members,
            )
        del remaining, seed
        member_connection = self._exact_connections(state, commands, target_index)
        offline_connection = member_connection.min(axis=1)
        support, outcome = self._online_mass(state, commands)
        corrected_connection = r48.bayesian_local_connection(
            member_connection, support, outcome
        )
        connection = corrected_connection.min(axis=1)
        reliability = np.asarray(reliability, np.float32).reshape(-1)
        gain, passage_member_gain = self._factorial_gain(
            state, commands, target_index
        )
        member_q = corrected_connection * reliability[:, None] * gain[:, None]
        q = member_q.min(axis=1)
        empirical = np.divide(
            outcome, support, out=np.zeros_like(outcome), where=support > 0.0
        )
        self._online_candidate_batches.append({
            "commands": np.asarray(commands, np.float32).copy(),
            "offline_connection": offline_connection,
            "corrected_connection": connection,
            "support": support,
            "empirical": empirical,
        })
        self._last_live_gain = gain
        self._last_passage_member_gain = passage_member_gain
        if return_members:
            return q, connection, member_q, corrected_connection
        return q, connection

    def _greedy_outgoing(self, node_index, remaining):
        if not self.acquisition_active:
            return super()._greedy_outgoing(node_index, remaining)
        del remaining
        node_index = np.asarray(node_index, np.int32).reshape(-1)
        model_target = self.targets[node_index]
        model_weight = self.weight[node_index]
        model_choice = np.argmax(model_weight, axis=1)
        row = np.arange(len(node_index))
        target = model_target[row, model_choice].astype(np.int32)
        command = self.graph.nodes[target, :2].copy()
        reliability = np.ones(len(node_index), np.float32)
        offline = model_weight[row, model_choice].astype(np.float32)
        kind = np.full(
            len(node_index), "alias_next_model_bridge_reliability_prefilter", object
        )
        for output, node in enumerate(node_index):
            begin = self.outgoing_offsets[node]
            stop = self.outgoing_offsets[node + 1]
            if begin == stop:
                continue
            edges = self.outgoing_order[begin:stop]
            local = int(np.argmax(self.graph.edge_reliability[edges]))
            edge = int(edges[local])
            value = float(self.graph.edge_reliability[edge])
            if value <= float(offline[output]):
                continue
            target[output] = self.graph.endpoint_index[edge]
            command[output] = self.graph.commands[edge]
            reliability[output] = value
            offline[output] = value
            kind[output] = "alias_next_recorded_edge_reliability_prefilter"
        return command, target, reliability, offline, kind

    def choose(
        self, state, command_index, fields, online_states=(), online_trajectories=()
    ):
        if fields.contacted:
            return super().choose(
                state, command_index, fields, online_states, online_trajectories
            )
        if not hasattr(self, "online_starts"):
            self.reset_online_correction_task()
        new_outcome = self._ingest_pending(online_states)
        self._online_candidate_batches = []
        command, diagnostic = r48.lazy.LazyAliasAdvancePlanner.choose(
            self, state, command_index, fields, online_states, online_trajectories
        )
        diagnostic.update(self._selected_online_diagnostic(
            np.asarray(command, np.float32),
            float(diagnostic["selected_connection_weight"]),
        ))
        diagnostic.update(self.factorial.diagnostic())
        diagnostic.update({
            "planner": self.planner_name,
            "method_revision": self.method_revision,
            "phase": "acquisition",
            "acquisition_backup": "R39_immediate_adaptive_greedy",
            "advance_prefilter": "observation_neutral_maximum_reliability",
            "selected_expected_contact_score": float(diagnostic["selected_score"]),
            "online_consequence_observations": len(self.online_starts),
            "new_online_outcome_from_previous_command": new_outcome,
            "online_offline_prior_effective_count_global": (
                r48.RECORDED_EVIDENCE_PRIOR_WEIGHT
            ),
        })
        self.pending_start = np.asarray(state, np.float32).copy()
        self.pending_command = np.asarray(command, np.float32).copy()
        self.pending_online_length = len(online_states)
        return command, diagnostic


def load_runtime(args):
    if r48.CANDIDATE_BUDGET != R12_CANDIDATE_BUDGET:
        raise ValueError("R48 candidate budget does not match the R12 manifest")
    r48.configure_result_directory(args.base_output)
    args.graph = args.base_output / "recorded_transition_graph.npz"
    graph, artifact, landmark, targets, weight = r48._load_bridges(args)
    consequence = r48.AntConsequenceAdapter(r48.R30_CHECKPOINT, args.device)
    endpoint_mobility = r48.mobility.MobilityModel(artifact["mobility"], args.device)
    planner = FactorialAntPlanner(
        consequence, endpoint_mobility, graph, landmark, targets, weight,
        np.random.default_rng(args.seed), args.seed,
    )
    geometry = r48.r31.r28.BoundedPassageGeometry(
        args.root, graph, targets, weight
    )
    r48.verify_passage_metadata(args, geometry.metadata)
    planner.configure_hidden_fields(r48.SIGMA)
    planner.configure_empirical_passage(geometry)
    learned = (
        LearnedPassageEnsemble(args.passage_artifact, args.device)
        if args.passage_level == "shared_learned" else None
    )
    if learned is not None and learned.state_dim != r48.STATE_DIM:
        raise ValueError("learned passage artifact is not the AntMaze adapter artifact")
    planner.configure_factorial(
        args.observation_level, args.passage_level, learned
    )
    planner.artifact_hashes = {
        "base_graph_sha256": r48.sha256(args.graph),
        "base_factorized_consequence_sha256": r48.sha256(r48.R30_CHECKPOINT),
        "base_mobility_sha256": r48.sha256(artifact["mobility"]),
        "base_bridges_sha256": r48.sha256(artifact["bridges"]),
        "base_empirical_passage_sha256": r48.sha256(r48.passage_paths()["metadata"]),
        "learned_passage_sha256": (
            r48.sha256(args.passage_artifact) if learned is not None else None
        ),
    }
    executor = FaramaSACExecutor(paths(args.root)["executor"], args.device)
    env = make_env(max_episode_steps=10_000_000)
    return planner, executor, env


def execute_command(env, executor, state, command, hidden_goal, success_radius):
    states, rewards = [state.copy()], []
    for _ in range(r48.COMMAND_STEPS):
        observation, _reward, terminated, truncated, _info = env.step(
            executor.action(state, command)
        )
        state = full_state(observation)
        states.append(state.copy())
        rewards.append(
            np.linalg.norm(state[:2] - hidden_goal) <= float(success_radius)
        )
        if terminated or truncated:
            raise RuntimeError("environment ended inside a fixed command")
    return np.asarray(states), np.asarray(rewards, np.uint8)


def evaluate_task(args, planner, executor, env, task_id):
    task = r48.r31.r22._task(args.root, task_id)
    arm = f"{args.observation_level}__{args.passage_level}"
    output = args.output / arm
    output.mkdir(parents=True, exist_ok=True)
    log_path = output / f"task_{task_id:03d}.jsonl"
    result_path = output / f"task_{task_id:03d}.json"
    fields = TaskFields(r48.SIGMA, search_mode=planner.acquisition_search_mode)
    planner.reset_online_correction_task()
    planner.rng = np.random.default_rng(args.seed + 1_000_003 * task_id)
    planner.seed = args.seed + 1_000_003 * task_id
    online, completed, logs, successes, episode_diagnostics = [], [], [], [], []
    first_contact, elapsed, first_step = None, 0, None
    goal = np.asarray(task["goal"], np.float32)
    radius = float(r48.PROTOCOL_VALUE["evaluator"]["success_radius_m_evaluator_only"])
    print(
        f"[{RUN_LABEL}][evaluate] arm={arm} task={task_id} "
        f"episodes={args.episodes} start",
        flush=True,
    )
    try:
        for episode in range(args.episodes):
            print(
                f"[{RUN_LABEL}][evaluate] arm={arm} task={task_id} "
                f"episode={episode + 1}/{args.episodes} reset",
                flush=True,
            )
            observation, _ = env.reset(seed=int(task["reset_seed"]))
            state = full_state(observation)
            episode_start_xy = state[:2].copy()
            phase_at_episode_start = (
                "return" if fields.contacted else "acquisition"
            )
            farthest_endpoint_from_start = 0.0
            final_endpoint_from_start = 0.0
            commands_executed = 0
            trajectory_memory = [state.copy()]
            reset_reward = np.uint8(np.linalg.norm(state[:2] - goal) <= radius)
            fields.add(state[:2], reset_reward)
            planner.observe_detection_consequence(state[None, :2], [reset_reward])
            episode_success = bool(reset_reward)
            for command_index in range(r48.HORIZON):
                if episode_success:
                    break
                fields.rebuild()
                phase = "return" if fields.contacted else "acquisition"
                command, diagnostic = planner.choose(
                    state, command_index, fields, online,
                    completed + [trajectory_memory],
                )
                trajectory, rewards = execute_command(
                    env, executor, state, command, goal, radius
                )
                planner.observe_detection_consequence(trajectory[1:, :2], rewards)
                for next_state, reward in zip(trajectory[1:], rewards):
                    fields.add(next_state[:2], reward)
                    online.append(next_state.copy())
                    trajectory_memory.append(next_state.copy())
                if first_step is None and rewards.any():
                    first_step = elapsed + int(np.flatnonzero(rewards)[0]) + 1
                elapsed += r48.COMMAND_STEPS
                episode_success = bool(rewards.any())
                state = trajectory[-1]
                commands_executed = command_index + 1
                final_endpoint_from_start = float(np.linalg.norm(
                    state[:2] - episode_start_xy
                ))
                farthest_endpoint_from_start = max(
                    farthest_endpoint_from_start, final_endpoint_from_start
                )
                diagnostic.update({
                    "episode": episode,
                    "command_index": command_index,
                    "phase_before_command": phase,
                    "command": np.asarray(command).tolist(),
                    "actual_endpoint": trajectory[-1, :2].tolist(),
                    "reward": episode_success,
                    "endpoint_from_episode_start_m": final_endpoint_from_start,
                    "farthest_endpoint_from_episode_start_m": (
                        farthest_endpoint_from_start
                    ),
                })
                logs.append(diagnostic)
                print(
                    f"[{RUN_LABEL}][evaluate] arm={arm} task={task_id} "
                    f"episode={episode + 1}/{args.episodes} "
                    f"command={command_index + 1}/{r48.HORIZON} phase={phase} "
                    f"reward={int(episode_success)} elapsed_steps={elapsed} "
                    f"endpoint_from_start_m={final_endpoint_from_start:.3f} "
                    f"farthest_endpoint_from_start_m="
                    f"{farthest_endpoint_from_start:.3f}",
                    flush=True,
                )
            completed.append(np.asarray(trajectory_memory))
            successes.append(episode_success)
            episode_diagnostics.append({
                "episode": episode,
                "episode_ordinal": episode + 1,
                "phase_at_episode_start": phase_at_episode_start,
                "success": bool(episode_success),
                "commands_executed": commands_executed,
                "farthest_endpoint_from_start_m": (
                    farthest_endpoint_from_start
                ),
                "final_endpoint_from_start_m": final_endpoint_from_start,
            })
            print(
                f"[{RUN_LABEL}][evaluate] arm={arm} task={task_id} "
                f"episode={episode + 1}/{args.episodes} "
                f"phase_at_start={phase_at_episode_start} "
                f"success={int(episode_success)} "
                f"farthest_endpoint_from_start_m="
                f"{farthest_endpoint_from_start:.3f}",
                flush=True,
            )
            if episode_success and first_contact is None:
                first_contact = episode
    except PosteriorFalsifiedError as error:
        failure = {
            "method_revision": METHOD_REVISION,
            "task_id": task_id,
            "arm": arm,
            "status": "model_falsified",
            "error": str(error),
            **planner.factorial.diagnostic(),
        }
        dump(output / f"task_{task_id:03d}_MODEL_FALSIFIED.json", failure)
        raise
    log_path.write_text("\n".join(json.dumps(row) for row in logs) + "\n")
    row = {
        **task,
        "method_revision": METHOD_REVISION,
        "method_reference": str(REFERENCE),
        "method_reference_sha256": r48.sha256(REFERENCE),
        "shared_method_revision": json.loads(MANIFEST.read_text())["method_revision"],
        "shared_method_manifest_sha256": r48.sha256(MANIFEST),
        "protocol_sha256": r48.sha256(PROTOCOL),
        "claim_level": CLAIM_LEVEL,
        "declared_deviations": json.loads(PROTOCOL.read_text())["declared_deviations"],
        "oracle_inputs": [],
        "task_id": task_id,
        "arm": arm,
        "successes": successes,
        "episode_diagnostics": episode_diagnostics,
        "first_contact_episode": first_contact,
        "first_contact_episode_ordinal": (
            first_contact + 1 if first_contact is not None else None
        ),
        "contact_latency": first_step,
        "mean_e2e": float(np.mean(successes)),
        "success_radius_read_by_planner": False,
        "episodes": args.episodes,
        "artifact_hashes": planner.artifact_hashes,
        **planner.factorial.diagnostic(),
    }
    dump(result_path, row)
    print(json.dumps(row), flush=True)
    print(
        f"[{RUN_LABEL}][evaluate] arm={arm} task={task_id} done "
        f"first_contact_episode={first_contact}",
        flush=True,
    )
    return row


def command_evaluate(args):
    selected = TASK_IDS if args.task_id is None else tuple(args.task_id)
    if selected != TASK_IDS:
        raise ValueError(
            f"{METHOD_REVISION} requires Ant tasks {list(TASK_IDS)}"
        )
    if args.episodes != REQUIRED_EPISODES:
        raise ValueError(
            f"{METHOD_REVISION} requires {REQUIRED_EPISODES} episodes per task"
        )
    selection_evidence = verify_baseline_miss_selection(args, selected)
    print(
        f"[{RUN_LABEL}][evaluate] "
        f"loading arm={args.observation_level}__{args.passage_level}",
        flush=True,
    )
    planner, executor, env = load_runtime(args)
    try:
        rows = [evaluate_task(args, planner, executor, env, task) for task in selected]
    finally:
        env.close()
    output = args.output / f"{args.observation_level}__{args.passage_level}"
    summary = {
        "method_revision": METHOD_REVISION,
        "method_reference": str(REFERENCE),
        "method_reference_sha256": r48.sha256(REFERENCE),
        "shared_method_revision": json.loads(MANIFEST.read_text())["method_revision"],
        "claim_level": CLAIM_LEVEL,
        "declared_deviations": json.loads(PROTOCOL.read_text())["declared_deviations"],
        "oracle_inputs": [],
        "shared_method_manifest_sha256": r48.sha256(MANIFEST),
        "protocol_sha256": r48.sha256(PROTOCOL),
        "arm": f"{args.observation_level}__{args.passage_level}",
        "tasks": len(rows),
        "task_ids": list(selected),
        "selection_evidence_verified": selection_evidence,
        "acquired_tasks": int(sum(row["first_contact_episode"] is not None for row in rows)),
        "mean_e2e": float(np.mean([row["mean_e2e"] for row in rows])),
        "rows": rows,
    }
    dump(output / "summary.json", summary)
    print(json.dumps(summary, indent=2))


def command_audit(args):
    protocol = json.loads(PROTOCOL.read_text())
    selection_evidence = verify_baseline_miss_selection(args, TASK_IDS)
    report = {
        "method_revision": METHOD_REVISION,
        "manifest": str(MANIFEST),
        "manifest_sha256": r48.sha256(MANIFEST),
        "candidate_goal_source": "unique_graph_nodes_xy",
        "uniform_candidate_prior": True,
        "posterior_exhaustion": "fatal_model_falsification_no_fallback",
        "shared_passage_model_class": "shared.learned_passage.PassageMLP",
        "domain_local_passage_model_class": False,
        "task_ids": list(TASK_IDS),
        "selection_evidence_verified": selection_evidence,
        "conforms": all(row["status"] == "pass" for row in protocol["conformance_audit"]),
    }
    print(json.dumps(report, indent=2))


def parser():
    value = argparse.ArgumentParser()
    value.add_argument("--root", type=Path, default=Path("/home/ai_diagnosis/SG/e377"))
    value.add_argument("--base-output", type=Path, default=DEFAULT_BASE)
    value.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    value.add_argument("--device", default="cuda")
    value.add_argument("--seed", type=int, default=12049)
    commands = value.add_subparsers(dest="command", required=True)
    commands.add_parser("audit").set_defaults(func=command_audit)
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--observation-level", choices=OBSERVATION_LEVELS, required=True)
    evaluate.add_argument("--passage-level", choices=PASSAGE_LEVELS, required=True)
    evaluate.add_argument("--passage-artifact", type=Path)
    evaluate.add_argument("--task-id", type=int, action="append")
    evaluate.add_argument("--episodes", type=int, default=3)
    evaluate.set_defaults(func=command_evaluate)
    return value


def main():
    args = parser().parse_args()
    if getattr(args, "passage_level", None) == "shared_learned" and args.passage_artifact is None:
        raise ValueError("shared_learned requires --passage-artifact")
    args.func(args)


if __name__ == "__main__":
    main()
