"""Replay R56 task 13 and record primitive-step E549 posture."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from projects.tfsn_consequence_value_iteration.antmaze import (
    r41_stall_posture_replay as r41,
)
from projects.tfsn_consequence_value_iteration.antmaze import (
    r52_e549_preparation as r52,
)
from projects.tfsn_consequence_value_iteration.antmaze.e549_executor import (
    E549_CHECKPOINT,
    E549_CHECKPOINT_SHA256,
    E549SACExecutor,
)
from scripts.antmaze.current_tfsn.executor import full_state, make_env


HERE = Path(__file__).resolve().parent
MANAGEMENT_ROOT = HERE.parents[2]
REFERENCE = HERE / "R57_E549_TASK13_POSTURE_REFERENCE.md"
PROTOCOL = HERE / "r57_e549_task13_posture_protocol.json"
METHOD_REVISION = "antmaze_e549_r56_task13_posture_replay_r57"
SOURCE = (
    MANAGEMENT_ROOT
    / "projects/tfsn_consequence_value_iteration/results/antmaze"
    / "e549_5m_supplied_goal_r56/stalled_pair"
)
DEFAULT_OUTPUT = SOURCE / "task13_posture_replay_r57"


def contract() -> dict:
    value = json.loads(PROTOCOL.read_text())
    expected = {
        "method_revision": METHOD_REVISION,
        "claim_level": "consumed_development_posture_diagnostic_only",
        "source_task_id": 13,
        "commands_replayed": 10,
        "decisive_command_index_zero_based": 7,
        "primitive_steps_per_command": 100,
        "executor_checkpoint_sha256": E549_CHECKPOINT_SHA256,
        "task_reward_used": False,
        "hidden_goal_used": False,
        "confirmation_panel_accessed": False,
    }
    for key, expected_value in expected.items():
        if value.get(key) != expected_value:
            raise ValueError(f"R57 protocol has another {key}")
    reference = Path(value["method_reference"])
    if not reference.is_absolute():
        reference = MANAGEMENT_ROOT / reference
    if reference.resolve() != REFERENCE.resolve():
        raise ValueError("R57 reference changed")
    return value


def source():
    summary_path = SOURCE / "summary.json"
    log_path = SOURCE / "task_013.jsonl"
    summary = json.loads(summary_path.read_text())
    task = next(row for row in summary["rows"] if int(row["task_id"]) == 13)
    rows = [
        json.loads(line) for line in log_path.read_text().splitlines() if line
    ]
    if len(rows) != 10 or any(int(row["command_index"]) != index for index, row in enumerate(rows)):
        raise ValueError("R56 task 13 command trace is incomplete")
    if summary.get("executor_checkpoint_sha256") != E549_CHECKPOINT_SHA256:
        raise ValueError("R56 source used another executor")
    return task, rows, summary_path, log_path


def command_audit(_args):
    value = contract()
    task, rows, summary_path, log_path = source()
    print(json.dumps({
        **value,
        "method_reference_sha256": r52.sha256(REFERENCE),
        "source_summary_sha256": r52.sha256(summary_path),
        "source_log_sha256": r52.sha256(log_path),
        "reset_seed": task["reset_seed"],
        "commands": [row["command"] for row in rows],
        "selected_graph_destinations": [
            row["selected_link_destination_xy"] for row in rows
        ],
    }, indent=2))


def command_replay(args):
    value = contract()
    task, source_rows, summary_path, log_path = source()
    executor = E549SACExecutor(E549_CHECKPOINT, args.device)
    env = make_env(max_episode_steps=10_000_000)
    commands = [np.asarray(row["command"], np.float32) for row in source_rows]
    command_rows, primitive_rows = [], []
    try:
        observation, _ = env.reset(seed=int(task["reset_seed"]))
        state = full_state(observation)
        for command_index, (command, expected) in enumerate(
            zip(commands, source_rows)
        ):
            start_xy = state[:2].copy()
            start_height, start_upright = r41.posture(state)
            heights, uprights, displacements = [], [], []
            for primitive in range(int(value["primitive_steps_per_command"])):
                action = executor.action(state, command)
                observation, _reward, terminated, truncated, _info = env.step(
                    action
                )
                state = full_state(observation)
                height, upright = r41.posture(state)
                heights.append(height)
                uprights.append(upright)
                displacements.append(float(np.linalg.norm(state[:2] - start_xy)))
                primitive_rows.append({
                    "command_index": int(command_index),
                    "primitive_step": int(primitive + 1),
                    "x": float(state[0]),
                    "y": float(state[1]),
                    "torso_height": float(height),
                    "torso_upright": float(upright),
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                })
                if terminated or truncated:
                    raise RuntimeError("environment ended during R57 replay")

            expected_endpoint = np.asarray(
                expected["actual_endpoint"], np.float32
            )
            destination = np.asarray(
                expected["selected_link_destination_xy"], np.float32
            )
            endpoint_error = float(
                np.linalg.norm(state[:2] - expected_endpoint)
            )
            destination_error = float(np.linalg.norm(state[:2] - destination))
            command_rows.append({
                "command_index": int(command_index),
                "command": command.tolist(),
                "selected_graph_destination_xy": destination.tolist(),
                "start_xy": start_xy.tolist(),
                "endpoint_xy": state[:2].tolist(),
                "expected_endpoint_xy": expected_endpoint.tolist(),
                "replay_endpoint_error_m": endpoint_error,
                "graph_destination_error_m": destination_error,
                "movement_m": float(np.linalg.norm(state[:2] - start_xy)),
                "maximum_displacement_m": float(np.max(displacements)),
                "start_torso_height": float(start_height),
                "endpoint_torso_height": float(heights[-1]),
                "minimum_torso_height": float(np.min(heights)),
                "start_torso_upright": float(start_upright),
                "endpoint_torso_upright": float(uprights[-1]),
                "minimum_torso_upright": float(np.min(uprights)),
                "inverted_at_endpoint_diagnostic": bool(uprights[-1] < 0.0),
            })
            print(
                f"R57 command={command_index + 1}/10 "
                f"move={command_rows[-1]['movement_m']:.3f} "
                f"destination_error={destination_error:.3f} "
                f"height={heights[-1]:.3f} upright={uprights[-1]:.3f} "
                f"replay_error={endpoint_error:.6f}",
                flush=True,
            )
    finally:
        env.close()

    decisive_index = int(value["decisive_command_index_zero_based"])
    decisive = command_rows[decisive_index]
    aftermath = command_rows[decisive_index + 1:]
    report = {
        "schema_version": 1,
        "experiment_id": value["experiment_id"],
        "method_reference": str(REFERENCE),
        "method_reference_sha256": r52.sha256(REFERENCE),
        "method_revision": METHOD_REVISION,
        "claim_level": value["claim_level"],
        "declared_deviations": value["declared_deviations"],
        "oracle_inputs": value["oracle_inputs"],
        "source_summary_sha256": r52.sha256(summary_path),
        "source_log_sha256": r52.sha256(log_path),
        "executor_checkpoint_sha256": executor.audit["sha256"],
        "reset_seed": int(task["reset_seed"]),
        "commands_replayed": len(command_rows),
        "primitive_steps_per_command": int(value["primitive_steps_per_command"]),
        "replay_endpoints_exactly_equal": bool(all(
            np.array_equal(
                np.asarray(row["endpoint_xy"], np.float32),
                np.asarray(row["expected_endpoint_xy"], np.float32),
            )
            for row in command_rows
        )),
        "maximum_replay_endpoint_error_m": float(max(
            row["replay_endpoint_error_m"] for row in command_rows
        )),
        "decisive_command_index_zero_based": decisive_index,
        "decisive_graph_destination_error_m": decisive[
            "graph_destination_error_m"
        ],
        "decisive_endpoint_torso_height": decisive["endpoint_torso_height"],
        "decisive_endpoint_torso_upright": decisive[
            "endpoint_torso_upright"
        ],
        "decisive_endpoint_inverted": decisive[
            "inverted_at_endpoint_diagnostic"
        ],
        "post_decisive_command_movements_m": [
            row["movement_m"] for row in aftermath
        ],
        "commands": command_rows,
        "task_reward_used": False,
        "hidden_goal_used": False,
        "confirmation_panel_accessed": False,
    }
    output = Path(args.output).resolve()
    r41.dump(output / "summary.json", report)
    r41.dump(output / "primitive_posture.json", {"rows": primitive_rows})
    print(json.dumps(report, indent=2))


def parser():
    value = argparse.ArgumentParser()
    value.add_argument(
        "--root", type=Path, default=Path("/home/ai_diagnosis/SG/e377")
    )
    value.add_argument("--device", default="cuda")
    value.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    commands = value.add_subparsers(dest="command", required=True)
    commands.add_parser("audit").set_defaults(func=command_audit)
    commands.add_parser("replay").set_defaults(func=command_replay)
    return value


def main():
    args = parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
