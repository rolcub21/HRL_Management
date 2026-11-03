#!/usr/bin/env python3
"""Prepare, train, and compare E215 with matched E191 seed 0."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
from pathlib import Path
from statistics import fmean
from unittest.mock import patch

import torch

from projects.temporal_consequence_realization.method_conformance import (
    render_validated_conformance_audit,
    validate_protocol,
)
from experiments.conditioned_vcg.E187_wait_corrected_e78_246k_v1 import (
    run as e187_run,
)
from experiments.conditioned_vcg.E191_fresh_long_stay_crowding_250k_v1 import (
    run as e191_run,
    stage as e191_stage,
)

from . import protocol as p


class E215Error(RuntimeError):
    pass


BUDGET_RULE = "fixed_300_episodes_e191_long_exploration_drift_diagnostic"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise E215Error(message)


def _conformance() -> dict:
    return validate_protocol(p.contract(), source=p.PROTOCOL)


def _conformance_summary(value: dict) -> dict:
    return {
        "validated": True,
        "method_revision": value["method_revision"],
        "claim_level": value["claim_level"],
        "historical_vcg_diagnostic": True,
    }


@contextmanager
def implementation():
    """Install the E215 protocol at both E191 protocol extension seams."""

    with (
        patch.object(e191_run, "p", p),
        patch.object(e191_run, "BUDGET_RULE", BUDGET_RULE),
        patch.object(e191_stage, "p", p),
    ):
        yield


def prepare(output: Path) -> dict:
    conformance = _conformance()
    with implementation():
        value = e191_run.prepare(output)
    return {
        **value,
        "protocol": p.PROTOCOL,
        "model_seeds": list(p.MODEL_SEEDS),
        "epsilon_decay_decisions": p.EPSILON_DECAY_DECISIONS,
        "target_floor_episode_approximate": p.EPSILON_TARGET_FLOOR_EPISODE,
        "implied_floor_episode": p.implied_floor_episode(),
        "only_training_change_from_e191": True,
        "method_conformance": _conformance_summary(conformance),
    }


def authenticate(output: Path) -> dict:
    _conformance()
    with implementation():
        return e191_run.authenticate(output)


def self_check(output: Path) -> dict:
    conformance = _conformance()
    with implementation():
        require(e191_run.p is p and e191_stage.p is p,
                "E215 did not install both E191 protocol seams")
        value = e191_run.self_check(output)
    contract = authenticate(output)
    require(p.MODEL_SEEDS == (0,), "E215 is not the declared one-seed pilot")
    require(
        p.EPSILON_DECAY_DECISIONS == 57_300
        and 219.9 < p.implied_floor_episode() < 220.1,
        "E215 epsilon calibration differs",
    )
    require(
        contract["drift_diagnostic"]["replay_capacity_unchanged"] == 20_000,
        "E215 changed replay capacity",
    )
    return {
        **value,
        "protocol": p.PROTOCOL,
        "model_seeds": [0],
        "epsilon_decay_decisions": p.EPSILON_DECAY_DECISIONS,
        "implied_floor_episode": p.implied_floor_episode(),
        "method_conformance": _conformance_summary(conformance),
    }


def train_seed(output: Path, through_episode: int, device: str) -> dict:
    _conformance()
    require(1 <= through_episode <= p.TOTAL_EPISODES,
            "E215 through-episode must be between 1 and 300")
    with implementation():
        result = e191_run.train_seed(output, 0, through_episode, device)
    result.pop("e191_protocol", None)
    result.pop("e191_contract_sha256", None)
    return {
        **result,
        "e215_protocol": p.PROTOCOL,
        "e215_contract_sha256": p.contract()["contract_sha256"],
        "epsilon_decay_decisions": p.EPSILON_DECAY_DECISIONS,
        "target_floor_episode_approximate": p.EPSILON_TARGET_FLOOR_EPISODE,
        "only_training_change_from_e191": True,
        "paper_evidence": False,
    }


def status(output: Path) -> dict:
    authenticate(output)
    with implementation():
        value = e191_run.status(output)
    return {
        **value,
        "protocol": p.PROTOCOL,
        "target_floor_episode_approximate": p.EPSILON_TARGET_FLOOR_EPISODE,
    }


def _read(path: Path) -> dict:
    require(path.is_file(), f"required summary is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"invalid JSON object: {path}")
    return value


def _summary_path(output: Path) -> Path:
    return (
        output.resolve()
        / "training/operational"
        / p.SCENARIO
        / "seed-0/training-summary.json"
    )


def _validation_rows(summary: dict) -> list[dict]:
    training = list(summary["training"]["runs"])
    rows = []
    for record in summary["validation_history"]:
        row = e187_run._validation_row(record, training)
        row["by_workload"] = record["summary"].get("e191_by_workload", {})
        rows.append(row)
    return rows


def _curve_diagnostic(rows: list[dict]) -> dict:
    if not rows:
        return {"status": "not_available"}
    best = min(rows, key=lambda row: float(row["J"]))
    final = rows[-1]
    return {
        "status": "available",
        "best_checkpoint": int(best["checkpoint_episode"]),
        "best_J": float(best["J"]),
        "latest_checkpoint": int(final["checkpoint_episode"]),
        "latest_J": float(final["J"]),
        "latest_minus_best_J": float(final["J"]) - float(best["J"]),
        "relative_late_regression": (
            (float(final["J"]) - float(best["J"])) / float(best["J"])
            if float(best["J"]) else None
        ),
    }


def analyze(output: Path) -> dict:
    contract = authenticate(output)
    current_path = _summary_path(output)
    original_path = _summary_path(p.ORIGINAL_E191_OUTPUT)
    current = _read(current_path)
    original = _read(original_path)
    require(current.get("training_protocol") == p.PROTOCOL,
            "E215 training summary identity differs")
    require(original.get("training_protocol") == p.e191.PROTOCOL,
            "original E191 seed-0 summary identity differs")

    corrected_rows = _validation_rows(current)
    original_rows = _validation_rows(original)
    corrected_by_ep = {row["checkpoint_episode"]: row for row in corrected_rows}
    original_by_ep = {row["checkpoint_episode"]: row for row in original_rows}
    common = sorted(set(corrected_by_ep) & set(original_by_ep))
    paired = []
    for episode in common:
        corrected = corrected_by_ep[episode]
        baseline = original_by_ep[episode]
        paired.append({
            "checkpoint_episode": episode,
            "corrected_epsilon": corrected["training_epsilon"],
            "original_epsilon": baseline["training_epsilon"],
            "corrected_J": corrected["J"],
            "original_J": baseline["J"],
            "corrected_minus_original_J": corrected["J"] - baseline["J"],
            "corrected_storage_clock_mae": corrected["storage_clock_mae"],
            "original_storage_clock_mae": baseline["storage_clock_mae"],
            "corrected_rehandles_per_100": corrected["rehandles_per_100"],
            "original_rehandles_per_100": baseline["rehandles_per_100"],
        })

    conformance = _conformance()
    report = {
        "method_conformance": conformance,
        "status": "complete" if len(corrected_rows) == 15 else "partial",
        "protocol": p.PROTOCOL,
        "contract_sha256": contract["contract_sha256"],
        "completed_training_episodes": int(current["completed_training_episodes"]),
        "only_training_change_from_e191": "epsilon decay 18500 -> 57300 decisions",
        "corrected_curve": corrected_rows,
        "original_e191_seed0_curve": original_rows,
        "paired_common_checkpoints": paired,
        "corrected_curve_diagnostic": _curve_diagnostic(corrected_rows),
        "original_curve_diagnostic": _curve_diagnostic(original_rows),
        "paired_mean_J_difference": (
            fmean(row["corrected_minus_original_J"] for row in paired)
            if paired else None
        ),
        "causal_scope": (
            "Tests the short-epsilon explanation only; replay capacity remains "
            "20,000 and still overwrites around episode 75."
        ),
        "test_panel_opened": False,
        "paper_evidence": False,
    }
    report_path = output.resolve() / "e215-report.json"
    e187_run._write(report_path, report)

    text = render_validated_conformance_audit(conformance)
    text += "\nRESULTS\n" + "=" * 72 + "\n"
    text += json.dumps({
        "status": report["status"],
        "completed_training_episodes": report["completed_training_episodes"],
        "corrected_curve_diagnostic": report["corrected_curve_diagnostic"],
        "original_curve_diagnostic": report["original_curve_diagnostic"],
        "paired_common_checkpoints": paired,
        "causal_scope": report["causal_scope"],
    }, indent=2, sort_keys=True) + "\n"
    report_text_path = output.resolve() / "REPORT.txt"
    temporary = report_text_path.with_suffix(".txt.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(report_text_path)
    return {
        "status": report["status"],
        "completed_training_episodes": report["completed_training_episodes"],
        "common_checkpoints": common,
        "report": str(report_path),
        "report_text": str(report_text_path),
        "test_panel_opened": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("prepare", "self-check", "train-seed", "status", "analyze"),
    )
    parser.add_argument("--output", type=Path, default=p.DEFAULT_OUTPUT)
    parser.add_argument("--model-seed", type=int, choices=(0,), default=0)
    parser.add_argument("--through-episode", type=int, default=120)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.command == "prepare":
        result = prepare(args.output)
    elif args.command == "self-check":
        result = self_check(args.output)
    elif args.command == "train-seed":
        require(args.model_seed == 0, "E215 is predeclared for seed 0 only")
        result = train_seed(args.output, args.through_episode, args.device)
    elif args.command == "status":
        result = status(args.output)
    else:
        result = analyze(args.output)
    print(json.dumps(result, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
