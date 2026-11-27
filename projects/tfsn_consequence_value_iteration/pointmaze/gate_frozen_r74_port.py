"""Acceptance gate for the 24 consumed PointMaze development tasks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .core import atomic_json, sha256
from .frozen_r74_planner import METHOD_REVISION, PROTOCOL, load_protocol


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
BASE = ROOT / "projects/tfsn_consequence_value_iteration/results/pointmaze"
R11_ORIGINAL = BASE / "continuous_reference_development_r11/summary.json"
R11_DISJOINT = (
    BASE / "continuous_reference_hidden_goal_r11_disjoint15_e120/summary.json"
)


def acquired(summary):
    rows = summary.get("task_metrics", [])
    return int(sum(float(row["acquisition_at_horizon"]) > 0.0 for row in rows))


def names(summary):
    return [str(row["goal_name"]) for row in summary.get("task_metrics", [])]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--original9", type=Path, required=True)
    parser.add_argument("--disjoint15", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    protocol = load_protocol()
    current = [
        json.loads(args.original9.read_text()),
        json.loads(args.disjoint15.read_text()),
    ]
    baseline = [
        json.loads(R11_ORIGINAL.read_text()),
        json.loads(R11_DISJOINT.read_text()),
    ]
    expected_tasks = [9, 15]
    panel_rows = []
    for label, now, old, count, path in zip(
        ("original9", "disjoint15"), current, baseline, expected_tasks,
        (args.original9, args.disjoint15),
    ):
        if now.get("method_revision") != METHOD_REVISION:
            raise ValueError(f"{label} was not evaluated with the frozen port")
        if int(now.get("tasks", -1)) != count:
            raise ValueError(f"{label} task count changed")
        if int(now.get("episodes_per_task", -1)) != 8:
            raise ValueError(f"{label} episode contract changed")
        if names(now) != names(old):
            raise ValueError(f"{label} goal rows differ from R11")
        if now.get("confirmation_panel_accessed") is not False:
            raise ValueError(f"{label} accessed confirmation evidence")
        panel_rows.append({
            "panel": label,
            "tasks": count,
            "r11_acquired": acquired(old),
            "frozen_r74_port_acquired": acquired(now),
            "summary": str(path.resolve()),
            "summary_sha256": sha256(path),
        })

    r11_total = sum(row["r11_acquired"] for row in panel_rows)
    current_total = sum(
        row["frozen_r74_port_acquired"] for row in panel_rows
    )
    floor = int(
        protocol["development_panels"]["acceptance_floor_acquired_of_24"]
    )
    if r11_total != 22:
        raise ValueError("frozen R11 baseline is no longer 22/24")
    report = {
        "schema_version": 1,
        "method_reference": protocol["method_reference"],
        "method_revision": METHOD_REVISION,
        "claim_level": protocol["claim_level"],
        "declared_deviations": protocol["declared_deviations"],
        "oracle_inputs": [],
        "protocol": str(PROTOCOL),
        "protocol_sha256": sha256(PROTOCOL),
        "panels": panel_rows,
        "tasks": 24,
        "episodes_per_task": 8,
        "r11_baseline_acquired": r11_total,
        "acceptance_floor_acquired": floor,
        "frozen_r74_port_acquired": current_total,
        "confirmation_panel_accessed": False,
        "passed": bool(current_total >= floor),
    }
    atomic_json(args.output, report)
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise RuntimeError(
            f"PointMaze frozen-R74 port acquired {current_total}/24; "
            f"required at least {floor}/24"
        )


if __name__ == "__main__":
    main()
