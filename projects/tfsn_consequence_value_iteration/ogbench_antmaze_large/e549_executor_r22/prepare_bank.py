#!/usr/bin/env python3
"""Freeze R22 command distances and reset support from reward-free OGBench data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .. import audit_executor as common


def derive_bank(
    observations: np.ndarray, terminals: np.ndarray, command_steps: int
) -> dict[str, np.ndarray]:
    observations = np.asarray(observations, dtype=np.float32)
    distances: list[np.ndarray] = []
    reset_xy: list[np.ndarray] = []
    for start, end in common.trajectory_ranges(terminals):
        if end - start < command_steps:
            raise RuntimeError("training trajectory is shorter than one command")
        reset_xy.append(observations[start, :2])
        source = observations[start : end - command_steps + 1, :2]
        target = observations[start + command_steps : end + 1, :2]
        distances.append(np.linalg.norm(target - source, axis=1))
    requested = np.concatenate(distances).astype(np.float32)
    resets = np.asarray(reset_xy, dtype=np.float32)
    if np.any(requested <= 0.0) or not np.isfinite(requested).all():
        raise RuntimeError("OGBench 100-step distance bank is invalid")
    return {"requested_distance_m": requested, "reset_xy": resets}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--dataset-audit", type=Path, required=True)
    parser.add_argument("--training-dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    datasets = common.audited_datasets(args.dataset_audit)
    common.verify_audited_dataset(
        args.training_dataset, datasets["train"], "training"
    )
    command_steps = int(protocol["training"]["fixed_command_horizon"])
    with np.load(args.training_dataset, allow_pickle=False) as data:
        bank = derive_bank(data["observations"], data["terminals"], command_steps)
    if args.output.is_file():
        with np.load(args.output, allow_pickle=False) as stored:
            for key, expected in bank.items():
                if key not in stored or not np.array_equal(stored[key], expected):
                    raise RuntimeError("frozen R22 training bank changed")
    else:
        common.atomic_npz(args.output, **bank)
    distance = bank["requested_distance_m"]
    report = {
        "schema_version": 1,
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": common.sha256(args.protocol),
        "artifact": str(args.output.resolve()),
        "artifact_sha256": common.sha256(args.output),
        "training_dataset": str(args.training_dataset.resolve()),
        "training_dataset_sha256": common.sha256(args.training_dataset),
        "command_duration_primitive_steps": command_steps,
        "distance_rows": int(len(distance)),
        "reset_rows": int(len(bank["reset_xy"])),
        "requested_distance_m": {
            "mean": float(distance.mean()),
            "rms": float(np.sqrt(np.mean(np.square(distance)))),
            "minimum": float(distance.min()),
            "median": float(np.median(distance)),
            "p90": float(np.quantile(distance, 0.9)),
            "maximum": float(distance.max()),
        },
        "distance_cutoffs_m": [],
        "native_task_rows_accessed": False,
        "native_evaluation_panel_accessed": False,
        "task_reward_used": False,
        "passed": True,
    }
    common.atomic_json(args.output.with_suffix(".json"), report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
