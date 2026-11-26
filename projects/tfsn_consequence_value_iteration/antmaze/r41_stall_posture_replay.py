"""Replay R40 task 3 through its first lasting stall and record posture."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from scripts.antmaze.current_tfsn.executor import (
    FaramaSACExecutor,
    full_state,
    make_env,
)
from scripts.antmaze.current_tfsn.main import paths


HERE = Path(__file__).resolve().parent
MANAGEMENT_ROOT = HERE.parents[2]
REFERENCE = HERE / "R41_STALL_POSTURE_REPLAY_REFERENCE.md"
PROTOCOL = HERE / "r41_stall_posture_protocol.json"
METHOD_REVISION = "antmaze_r40_task3_stall_posture_replay_r41"
R40 = (
    MANAGEMENT_ROOT
    / "projects/tfsn_consequence_value_iteration/results/antmaze"
    / "live_posterior_passage_r39/diagnostic_gate2_continuous50"
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def upright_score(quaternion):
    w, x, y, z = np.asarray(quaternion, np.float64).reshape(4)
    return float(1.0 - 2.0 * (x * x + y * y))


def posture(state):
    raw = np.asarray(state, np.float32).reshape(29)[2:]
    return float(raw[0]), upright_score(raw[1:5])


def contract():
    value = json.loads(PROTOCOL.read_text())
    expected = {
        "method_revision": METHOD_REVISION,
        "claim_level": "consumed_development_posture_diagnostic_only",
        "source_task_id": 3,
        "commands_replayed": 4,
        "primitive_steps_per_command": 100,
        "task_reward_used": False,
        "hidden_goal_used": False,
        "confirmation_panel_accessed": False,
    }
    for key, expected_value in expected.items():
        if value.get(key) != expected_value:
            raise ValueError(f"R41 protocol has another {key}")
    reference = Path(value["method_reference"])
    if not reference.is_absolute():
        reference = MANAGEMENT_ROOT / reference
    if reference.resolve() != REFERENCE:
        raise ValueError("R41 reference changed")
    return value


def source():
    summary = json.loads((R40 / "summary.json").read_text())
    task = next(row for row in summary["rows"] if int(row["task_id"]) == 3)
    rows = [
        json.loads(line)
        for line in (R40 / "task_003.jsonl").read_text().splitlines()
        if line
    ]
    if len(rows) != 50:
        raise ValueError("R40 task 3 is incomplete")
    return task, rows[:4]


def command_audit(_args):
    value = contract()
    task, rows = source()
    print(json.dumps({
        **value,
        "method_reference_sha256": sha256(REFERENCE),
        "source_summary_sha256": sha256(R40 / "summary.json"),
        "source_log_sha256": sha256(R40 / "task_003.jsonl"),
        "reset_seed": task["reset_seed"],
        "commands": [row["command"] for row in rows],
    }, indent=2))


def command_replay(args):
    value = contract()
    task, source_rows = source()
    executor_path = paths(args.root)["executor"]
    executor = FaramaSACExecutor(executor_path, args.device)
    env = make_env(max_episode_steps=1000)
    commands = [np.asarray(row["command"], np.float32) for row in source_rows]
    command_rows, primitive_rows = [], []
    try:
        observation, _ = env.reset(seed=int(task["reset_seed"]))
        state = full_state(observation)
        for command_index, (command, expected) in enumerate(zip(commands, source_rows)):
            start_xy = state[:2].copy()
            start_height, start_upright = posture(state)
            heights, uprights, displacements = [], [], []
            for primitive in range(100):
                action = executor.action(state, command)
                observation, _reward, terminated, truncated, _info = env.step(action)
                state = full_state(observation)
                height, upright = posture(state)
                heights.append(height)
                uprights.append(upright)
                displacements.append(float(np.linalg.norm(state[:2] - start_xy)))
                primitive_rows.append({
                    "command_index": command_index,
                    "primitive_step": primitive + 1,
                    "x": float(state[0]),
                    "y": float(state[1]),
                    "torso_height": height,
                    "torso_upright": upright,
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                })
                if terminated or truncated:
                    raise RuntimeError("environment ended during the four-command replay")
            expected_endpoint = np.asarray(expected["actual_endpoint"], np.float32)
            endpoint_error = float(np.linalg.norm(state[:2] - expected_endpoint))
            command_rows.append({
                "command_index": command_index,
                "command": command.tolist(),
                "start_xy": start_xy.tolist(),
                "endpoint_xy": state[:2].tolist(),
                "expected_endpoint_xy": expected_endpoint.tolist(),
                "replay_endpoint_error_m": endpoint_error,
                "movement_m": float(np.linalg.norm(state[:2] - start_xy)),
                "maximum_displacement_m": float(np.max(displacements)),
                "start_torso_height": start_height,
                "endpoint_torso_height": float(heights[-1]),
                "minimum_torso_height": float(np.min(heights)),
                "start_torso_upright": start_upright,
                "endpoint_torso_upright": float(uprights[-1]),
                "minimum_torso_upright": float(np.min(uprights)),
                "inverted_at_endpoint_diagnostic": bool(uprights[-1] < 0.0),
            })
            print(
                f"R41 command={command_index + 1}/4 "
                f"move={command_rows[-1]['movement_m']:.3f} "
                f"height={heights[-1]:.3f} upright={uprights[-1]:.3f} "
                f"replay_error={endpoint_error:.6f}",
                flush=True,
            )
    finally:
        env.close()

    reproducible = bool(max(row["replay_endpoint_error_m"] for row in command_rows) < 0.05)
    stall = command_rows[-1]
    report = {
        "schema_version": 1,
        "experiment_id": value["experiment_id"],
        "method_reference": str(REFERENCE),
        "method_reference_sha256": sha256(REFERENCE),
        "method_revision": METHOD_REVISION,
        "claim_level": value["claim_level"],
        "declared_deviations": value["declared_deviations"],
        "oracle_inputs": value["oracle_inputs"],
        "source_summary_sha256": sha256(R40 / "summary.json"),
        "source_log_sha256": sha256(R40 / "task_003.jsonl"),
        "executor_sha256": executor.audit["sha256"],
        "reset_seed": int(task["reset_seed"]),
        "commands_replayed": len(command_rows),
        "primitive_steps_per_command": 100,
        "replay_matches_recorded_endpoints": reproducible,
        "maximum_replay_endpoint_error_m": float(max(
            row["replay_endpoint_error_m"] for row in command_rows
        )),
        "stall_endpoint_torso_height": stall["endpoint_torso_height"],
        "stall_endpoint_torso_upright": stall["endpoint_torso_upright"],
        "stall_endpoint_inverted": stall["inverted_at_endpoint_diagnostic"],
        "commands": command_rows,
        "task_reward_used": False,
        "hidden_goal_used": False,
        "confirmation_panel_accessed": False,
    }
    output = Path(args.output).resolve()
    dump(output / "summary.json", report)
    dump(output / "primitive_posture.json", {"rows": primitive_rows})
    print(json.dumps(report, indent=2))


def parser():
    value = argparse.ArgumentParser()
    value.add_argument("--root", type=Path, default=Path("/home/ai_diagnosis/SG/e377"))
    value.add_argument("--device", default="cuda")
    value.add_argument(
        "--output",
        type=Path,
        default=R40 / "task3_stall_posture_replay_r41",
    )
    commands = value.add_subparsers(dest="command", required=True)
    commands.add_parser("audit").set_defaults(func=command_audit)
    commands.add_parser("replay").set_defaults(func=command_replay)
    return value


def main():
    args = parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

