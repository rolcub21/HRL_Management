#!/usr/bin/env python3
"""Prepare, train, and analyze E255's temperature-scaled E195 seed 0."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
from pathlib import Path
from statistics import fmean
from unittest.mock import patch

import torch

import run_vcg_objective_gamma_audit as audit
import train_vcg_dense_proper as dense
from experiments.conditioned_vcg.E187_wait_corrected_e78_246k_v1 import (
    run as e187_run,
)
from experiments.conditioned_vcg.E195_e191_method_5x5_longstay_254k_v1 import (
    run as e195_run,
    stage as e195_stage,
)
from projects.temporal_consequence_realization.method_conformance import (
    render_validated_conformance_audit,
    validate_protocol,
)

from . import protocol as p


class E255Error(RuntimeError):
    pass


BUDGET_RULE = "fixed_300_episodes_e195_temperature_scaled_5x5_seed0"
TEMPERATURE_KEYS = frozenset((
    "tau_accept", "tau_recover", "tau_defer", "tau_mode"
))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise E255Error(message)


def _conformance() -> dict:
    return validate_protocol(p.contract(), source=p.PROTOCOL)


def corrected_defaults() -> dict:
    original = dict(dense.FROZEN_AUDIT_DEFAULTS)
    corrected = dict(original)
    corrected.update(p.temperature_defaults())
    changed = {
        key for key in set(original) | set(corrected)
        if original.get(key) != corrected.get(key)
    }
    require(changed == TEMPERATURE_KEYS, "E255 changes more than temperatures")
    require(original["reward_scale"] == corrected["reward_scale"] == 0.01,
            "E255 changed reward scaling")
    return corrected


ORIGINAL_AUDIT_BUILD_PARSER = audit.build_parser


def temperature_scaled_parser():
    """Return the shared parser with only E255's temperatures replaced."""

    parser = ORIGINAL_AUDIT_BUILD_PARSER()
    parser.set_defaults(**p.temperature_defaults())
    return parser


@contextmanager
def implementation():
    """Install E255 at E195's protocol and frozen-default extension seams."""

    defaults = corrected_defaults()
    with (
        patch.object(e195_run, "p", p),
        patch.object(e195_run, "BUDGET_RULE", BUDGET_RULE),
        patch.object(e195_stage, "p", p),
        patch.object(dense, "FROZEN_AUDIT_DEFAULTS", defaults),
        patch.object(audit, "build_parser", temperature_scaled_parser),
    ):
        yield


def prepare(output: Path) -> dict:
    conformance = _conformance()
    with implementation():
        value = e195_run.prepare(output)
    return {
        **value,
        "protocol": p.PROTOCOL,
        "model_seeds": [0],
        "temperature_divisor": p.TEMPERATURE_DIVISOR,
        "temperature_defaults": p.temperature_defaults(),
        "only_training_change_from_e195": "four Boltzmann temperatures",
        "method_conformance": conformance,
    }


def authenticate(output: Path) -> dict:
    _conformance()
    with implementation():
        return e195_run.authenticate(output)


def self_check(output: Path) -> dict:
    conformance = _conformance()
    original = dict(dense.FROZEN_AUDIT_DEFAULTS)
    with implementation():
        require(e195_run.p is p and e195_stage.p is p,
                "E255 did not install both E195 protocol seams")
        require(
            all(dense.FROZEN_AUDIT_DEFAULTS[key] == value
                for key, value in p.temperature_defaults().items()),
            "E255 temperatures did not reach the frozen trainer defaults",
        )
        parsed = audit.build_parser().parse_args((
            "--output-dir", str(output.resolve() / "parser-probe"),
        ))
        require(
            all(getattr(parsed, key) == value
                for key, value in p.temperature_defaults().items()),
            "E255 temperatures did not reach the executable parser path",
        )
        value = e195_run.self_check(output)
    contract = authenticate(output)
    require(p.MODEL_SEEDS == (0,), "E255 must remain a one-seed diagnostic")
    require(p.TOTAL_EPISODES == p.e195.TOTAL_EPISODES == 300,
            "E255 changed the training budget")
    require(p.EPSILON_DECAY_DECISIONS == p.e195.EPSILON_DECAY_DECISIONS,
            "E255 changed E195 exploration")
    require(abs(p.TEMPERATURE_DIVISOR - 32.511208503349955) < 1.0e-12,
            "E255 temperature divisor differs")
    require(dense.FROZEN_AUDIT_DEFAULTS == original,
            "E255 leaked patched defaults after self-check")
    diagnostic = contract["temperature_scale_diagnostic"]
    invariants = (
        "reward_scale_unchanged", "architecture_unchanged",
        "objective_unchanged", "optimizer_unchanged",
        "replay_capacity_unchanged", "epsilon_schedule_unchanged",
        "training_budget_unchanged", "certification_unchanged",
        "guard_unchanged", "training_instances_unchanged",
        "validation_panel_unchanged", "checkpoint_rule_unchanged",
    )
    require(all(diagnostic[key] is True for key in invariants),
            "E255 invariant declaration differs")
    return {
        **value,
        "protocol": p.PROTOCOL,
        "temperature_divisor": p.TEMPERATURE_DIVISOR,
        "original_temperatures": diagnostic["original"],
        "corrected_temperatures": diagnostic["corrected"],
        "development_scale_source": diagnostic["derivation"],
        "method_conformance": conformance,
        "test_panel_opened": False,
    }


def train_seed(output: Path, through_episode: int, device: str) -> dict:
    _conformance()
    require(1 <= through_episode <= p.TOTAL_EPISODES,
            "E255 through-episode must be between 1 and 300")
    with implementation():
        result = e195_run.train_seed(output, 0, through_episode, device)
    result.pop("e195_protocol", None)
    result.pop("e195_contract_sha256", None)
    return {
        **result,
        "e255_protocol": p.PROTOCOL,
        "e255_contract_sha256": p.contract()["contract_sha256"],
        "temperature_divisor": p.TEMPERATURE_DIVISOR,
        "temperature_defaults": p.temperature_defaults(),
        "only_training_change_from_e195": "four Boltzmann temperatures",
        "paper_evidence": False,
    }


def status(output: Path) -> dict:
    authenticate(output)
    with implementation():
        value = e195_run.status(output)
    return {**value, "protocol": p.PROTOCOL}


def _read(path: Path) -> dict:
    require(path.is_file(), f"required E255 artifact is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"invalid E255 JSON object: {path}")
    return value


def _summary_path(output: Path) -> Path:
    return (
        output.resolve() / "training/operational" / p.SCENARIO
        / "seed-0/training-summary.json"
    )


def _validation_rows(summary: dict) -> list[dict]:
    training = list(summary["training"]["runs"])
    rows = []
    for record in summary.get("validation_history", ()):
        row = e187_run._validation_row(record, training)
        row["by_workload"] = record["summary"].get("e195_by_workload", {})
        rows.append(row)
    return rows


def analyze(output: Path) -> dict:
    contract = authenticate(output)
    current = _read(_summary_path(output))
    original = _read(p.DEVELOPMENT_SUMMARIES["5x5_seed0"])
    require(current.get("training_protocol") == p.PROTOCOL,
            "E255 training summary identity differs")
    corrected_rows = _validation_rows(current)
    original_rows = _validation_rows(original)
    corrected = {int(row["checkpoint_episode"]): row for row in corrected_rows}
    baseline = {int(row["checkpoint_episode"]): row for row in original_rows}
    common = sorted(set(corrected) & set(baseline))
    paired = [{
        "checkpoint_episode": episode,
        "corrected_J": corrected[episode]["J"],
        "original_J": baseline[episode]["J"],
        "corrected_minus_original_J": (
            corrected[episode]["J"] - baseline[episode]["J"]
        ),
    } for episode in common]
    complete = int(current["completed_training_episodes"]) == p.TOTAL_EPISODES
    report = {
        "method_conformance": _conformance(),
        "status": "complete" if complete and len(corrected_rows) == 15 else "partial",
        "protocol": p.PROTOCOL,
        "contract_sha256": contract["contract_sha256"],
        "completed_training_episodes": int(current["completed_training_episodes"]),
        "temperature_divisor": p.TEMPERATURE_DIVISOR,
        "temperature_defaults": p.temperature_defaults(),
        "corrected_curve": corrected_rows,
        "original_e195_seed0_curve": original_rows,
        "paired_common_checkpoints": paired,
        "paired_mean_J_difference": (
            fmean(row["corrected_minus_original_J"] for row in paired)
            if paired else None
        ),
        "test_panel_opened": False,
        "paper_evidence": False,
    }
    path = output.resolve() / "e255-report.json"
    e187_run._write(path, report)
    text = render_validated_conformance_audit(report["method_conformance"])
    text += "\nRESULTS\n" + "=" * 72 + "\n"
    text += json.dumps({
        "status": report["status"],
        "completed_training_episodes": report["completed_training_episodes"],
        "temperature_divisor": report["temperature_divisor"],
        "temperature_defaults": report["temperature_defaults"],
        "paired_common_checkpoints": paired,
    }, indent=2, sort_keys=True) + "\n"
    report_text = output.resolve() / "REPORT.txt"
    temporary = report_text.with_suffix(".txt.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(report_text)
    return {
        "status": report["status"],
        "completed_training_episodes": report["completed_training_episodes"],
        "common_checkpoints": common,
        "report": str(path),
        "report_text": str(report_text),
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
    parser.add_argument("--through-episode", type=int, default=300)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.command == "prepare":
        result = prepare(args.output)
    elif args.command == "self-check":
        result = self_check(args.output)
    elif args.command == "train-seed":
        require(args.model_seed == 0, "E255 is predeclared for seed 0 only")
        result = train_seed(args.output, args.through_episode, args.device)
    elif args.command == "status":
        result = status(args.output)
    else:
        result = analyze(args.output)
    print(json.dumps(result, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
