#!/usr/bin/env python3
"""Prepare, train, inspect, and compare the paired E188 guard arm."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
from pathlib import Path
from statistics import fmean
from unittest.mock import patch

import torch

from experiments.conditioned_vcg.E187_wait_corrected_e78_246k_v1 import (
    run as e187,
)

from . import instrumentation, protocol as p, representation


class E188Error(RuntimeError):
    pass


BUDGET_RULE = "fixed_300_episodes_e187_paired_choice_preserving_guard"
ChoicePreservingQueueAgent = representation.ChoicePreservingQueueAgent


def require(condition: bool, message: str) -> None:
    if not condition:
        raise E188Error(message)


@contextmanager
def implementation():
    """Install E188 only at E187's explicit extension seams.

    E187 remains byte-for-byte unchanged.  Its trainer therefore supplies the
    exact same environment, objective, instances, exploration schedule, graph
    architecture, replay, TD backup, and accelerations.  The agent subclass is
    the sole behavioral change and owns the replacement progress guard.
    """

    with (
        patch.object(e187, "p", p),
        patch.object(
            e187, "WaitCorrectedQueueAgent", ChoicePreservingQueueAgent
        ),
        patch.object(e187, "instrumentation", instrumentation),
        patch.object(e187, "BUDGET_RULE", BUDGET_RULE),
    ):
        yield


def authenticate(output: Path) -> dict:
    with implementation():
        return e187.authenticate(output)


def prepare(output: Path) -> dict:
    with implementation():
        result = e187.prepare(output)
    return {
        **result,
        "protocol": p.PROTOCOL,
        "paired_parent_protocol": p.e187.PROTOCOL,
        "only_intended_method_delta": "progress_guard",
    }


def _optional_self_check(module, name: str) -> dict:
    function = getattr(module, "self_check", None)
    if not callable(function):
        return {"status": "not_exposed"}
    value = function()
    require(isinstance(value, dict), f"E188 {name} self-check returned no audit")
    require(value.get("status", "passed") in ("passed", "ok"),
            f"E188 {name} self-check failed: {value!r}")
    return value


def self_check(output: Path) -> dict:
    contract = p.contract()
    require(contract["matched_parent_fields_sha256"] == p.digest({
        name: p.e187.contract()[name] for name in p.MATCHED_PARENT_FIELDS
    }), "E188 matched-field assertion failed")
    with implementation():
        inherited = e187.self_check(output)
    config = e187.EpisodicViabilityGraphConfig(
        gamma=p.GAMMA, episode_horizon_steps=p.MAX_STEPS
    )
    with e187.warm_start_disabled():
        probe = ChoicePreservingQueueAgent(
            config=config, seed=0, device="cpu", epsilon=p.EPSILON_START
        )
    guard = probe.recovery_witness_guard
    require(
        isinstance(guard, representation.guard_module.ChoicePreservingWitnessGuard),
        "E188 did not install the choice-preserving progress guard",
    )
    guard_state = guard.audit()
    for name in (
        "masked_frontier_calls", "progress_frontier_calls",
        "fallback_frontier_calls", "reconfigure_candidates_removed",
        "witness_interruptions", "masked_decisions", "fallback_decisions",
    ):
        require(name in guard_state, f"E188 guard audit omits {name}")
    metadata = probe.checkpoint_metadata()
    require(
        metadata.get("e188_guard_contract")
        == representation.guard_module.GUARD_CONTRACT,
        "E188 checkpoint metadata omits the guard identity",
    )
    guard_module = getattr(representation, "guard_module", None)
    guard_audit = (
        _optional_self_check(guard_module, "guard")
        if guard_module is not None else {"status": "checked_by_agent"}
    )
    representation_audit = _optional_self_check(
        representation, "representation"
    )
    return {
        **inherited,
        "status": "passed",
        "protocol": p.PROTOCOL,
        "contract_sha256": contract["contract_sha256"],
        "matched_parent_fields": list(p.MATCHED_PARENT_FIELDS),
        "guard_self_check": guard_audit,
        "representation_self_check": representation_audit,
        "guard_contract": representation.guard_module.GUARD_CONTRACT,
        "guard_audit_fields": sorted(guard_state),
        "choice_preserving_guard_installed": True,
        "only_intended_method_delta": "progress_guard",
    }


def train_seed(
    output: Path, seed: int, through_episode: int, device: str
) -> dict:
    with implementation():
        result = e187.train_seed(output, seed, through_episode, device)
    # E187 compatibility keys inside trainer artifacts intentionally remain:
    # they authenticate the reused objective and trainer plumbing.  These
    # E188 keys authenticate the experiment identity at the public boundary.
    result.pop("e187_protocol", None)
    return {
        **result,
        "e188_protocol": p.PROTOCOL,
        "e188_contract_sha256": p.contract()["contract_sha256"],
        "paired_parent_protocol": p.e187.PROTOCOL,
        "only_intended_method_delta": "progress_guard",
    }


def status(output: Path) -> dict:
    with implementation():
        result = e187.status(output)
    return {
        **result,
        "protocol": p.PROTOCOL,
        "paired_parent_protocol": p.e187.PROTOCOL,
    }


def _validation_row(record: dict, training_history: list[dict]) -> dict:
    with implementation():
        row = e187._validation_row(record, training_history)
    summary = record["summary"]
    row["guard"] = {
        "forced_deliveries": summary.get("e188_forced_deliveries", 0),
        "voluntary_deliveries": summary.get("e188_voluntary_deliveries", 0),
        "forced_delivery_share": summary.get(
            "e188_guard_forced_delivery_share"
        ),
        "mean_case_forced_delivery_share": summary.get(
            "e188_mean_case_guard_forced_delivery_share"
        ),
        "totals": summary.get("e188_guard_totals", {}),
    }
    return row


def analyze(output: Path) -> dict:
    output = output.resolve()
    contract = authenticate(output)
    seeds = []
    for seed in p.MODEL_SEEDS:
        with implementation():
            path = e187._summary_path(output, seed)
        if not path.is_file():
            seeds.append({
                "model_seed": seed,
                "status": "not_started",
                "completed_training_episodes": 0,
            })
            continue
        with implementation():
            summary = e187._seed_summary(output, seed)
        completed = int(summary["completed_training_episodes"])
        training_history = list(summary["training"]["runs"])
        rows = [
            _validation_row(record, training_history)
            for record in summary.get("validation_history", ())
        ]
        expected = list(range(
            p.VALIDATION_EVERY, completed + 1, p.VALIDATION_EVERY
        ))
        require(
            [row["checkpoint_episode"] for row in rows] == expected,
            f"E188 seed {seed} validation coverage differs",
        )
        seeds.append({
            "model_seed": seed,
            "status": (
                "complete" if completed == p.TOTAL_EPISODES else "partial"
            ),
            "completed_training_episodes": completed,
            "training_summary": str(path),
            "validation_history": rows,
            "terminal": (
                rows[-1] if completed == p.TOTAL_EPISODES else None
            ),
            "trainer_best_checkpoint_descriptive_only": summary.get(
                "best_checkpoint"
            ),
        })
    complete = all(row["status"] == "complete" for row in seeds)
    terminal_j = [
        float(row["terminal"]["J"])
        for row in seeds if row.get("terminal") is not None
    ]
    report = {
        "status": "complete" if complete else "partial",
        "protocol": p.PROTOCOL,
        "paired_parent_protocol": p.e187.PROTOCOL,
        "contract_sha256": contract["contract_sha256"],
        "objective": contract["objective"],
        "scope": "paired E187 progress-guard intervention; development only",
        "seeds": seeds,
        "terminal_J_mean_across_seeds": (
            float(fmean(terminal_j))
            if len(terminal_j) == len(p.MODEL_SEEDS) else None
        ),
        "test_panel_opened": False,
        "paper_evidence": False,
        "episode_runs": 0,
        "training_runs": 0,
    }
    path = output / "e188-report.json"
    e187._write(path, report)
    return {
        "status": report["status"],
        "report": str(path),
        "episode_runs": 0,
        "training_runs": 0,
    }


def _case_metrics(run: dict) -> dict[str, float]:
    metrics = run["e187_metrics"]
    audit = run.get("audit", {})
    selected = run.get("selected_action_counts", {})
    deliveries = int(run.get("delivery_count", 0))
    forced_deliveries = int(
        audit.get("liveness_forced_delivery_decisions", 0)
    )
    result = {
        "J": float(metrics["J"]),
        "admission_wait": float(metrics["mean_admission_wait"]),
        "storage_clock_mae": float(metrics["storage_clock_mae"]),
        "storage_within_20_rate": float(metrics["storage_within_20_rate"]),
        "release_clock_mae": float(metrics["release_clock_mae"]),
        "release_within_20_rate": float(metrics["release_within_20_rate"]),
        "peak_stored": float(metrics["peak_stored"]),
        "steps": float(run["steps"]),
        "physical_rehandles": float(metrics["components"]["physical_rehandles"]),
        "accept_decisions": float(selected.get("accept", 0)),
        "deliver_decisions": float(selected.get("deliver", 0)),
        "reconfigure_decisions": float(selected.get("reconfigure", 0)),
        "liveness_forced_decisions": float(
            audit.get("liveness_forced_decisions", 0)
        ),
        "liveness_guard_activations": float(
            audit.get("liveness_guard_activations", 0)
        ),
        "forced_deliveries": float(forced_deliveries),
        "guard_forced_delivery_share": (
            0.0 if deliveries == 0 else forced_deliveries / deliveries
        ),
        "voluntary_deliveries": float(
            max(deliveries - forced_deliveries, 0)
        ),
    }
    for name, value in metrics["components"].items():
        result[f"component_{name}"] = float(value)
    return result


def _summary_by_seed(output: Path, *, e188: bool) -> dict[int, dict]:
    result = {}
    for seed in p.MODEL_SEEDS:
        if e188:
            with implementation():
                path = e187._summary_path(output, seed)
                if path.is_file():
                    result[seed] = e187._seed_summary(output, seed)
        else:
            path = e187._summary_path(output, seed)
            if path.is_file():
                result[seed] = e187._seed_summary(output, seed)
    return result


def compare(output: Path, parent_output: Path) -> dict:
    """Write exact common-random-number deltas, E188 minus E187."""

    output = output.resolve()
    parent_output = parent_output.resolve()
    authenticate(output)
    e187.authenticate(parent_output)
    current = _summary_by_seed(output, e188=True)
    parent = _summary_by_seed(parent_output, e188=False)
    rows = []
    for seed in sorted(set(current) & set(parent)):
        old_records = {
            int(record["checkpoint_episode"]): record
            for record in parent[seed].get("validation_history", ())
        }
        new_records = {
            int(record["checkpoint_episode"]): record
            for record in current[seed].get("validation_history", ())
        }
        for episode in sorted(set(old_records) & set(new_records)):
            old_runs = {
                int(run["instance_seed"]): run
                for run in old_records[episode]["runs"]
            }
            new_runs = {
                int(run["instance_seed"]): run
                for run in new_records[episode]["runs"]
            }
            require(
                set(old_runs) == set(new_runs) == set(p.VALIDATION_SEEDS[p.SCENARIO]),
                f"paired validation identities differ at seed {seed}, episode {episode}",
            )
            cases = []
            for instance_seed in sorted(old_runs):
                old = _case_metrics(old_runs[instance_seed])
                new = _case_metrics(new_runs[instance_seed])
                require(set(old) == set(new), "paired metric schema differs")
                delta = {name: new[name] - old[name] for name in old}
                cases.append({
                    "instance_seed": instance_seed,
                    "e187": old,
                    "e188": new,
                    "delta_e188_minus_e187": delta,
                })
            metric_names = tuple(cases[0]["delta_e188_minus_e187"])
            mean_delta = {
                name: float(fmean(
                    case["delta_e188_minus_e187"][name] for case in cases
                ))
                for name in metric_names
            }
            rows.append({
                "model_seed": seed,
                "checkpoint_episode": episode,
                "cases": cases,
                "mean_delta_e188_minus_e187": mean_delta,
                "e188_better_on_mean_J": mean_delta["J"] < 0.0,
            })
    report = {
        "status": "complete" if rows else "no_shared_validation_checkpoint",
        "protocol": p.PROTOCOL,
        "parent_protocol": p.e187.PROTOCOL,
        "e188_contract_sha256": p.contract()["contract_sha256"],
        "e187_contract_sha256": p.e187.contract()["contract_sha256"],
        "pairing_key": "model_seed,checkpoint_episode,instance_seed",
        "delta_sign": "E188 minus E187; negative is better for cost metrics",
        "rows": rows,
        "test_panel_opened": False,
        "paper_evidence": False,
    }
    path = output / "e188-vs-e187-paired-report.json"
    e187._write(path, report)
    return {"status": report["status"], "report": str(path), "rows": len(rows)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=(
            "prepare", "self-check", "train-seed", "status", "analyze",
            "compare",
        ),
    )
    parser.add_argument("--output", type=Path, default=p.DEFAULT_OUTPUT)
    parser.add_argument("--parent-output", type=Path, default=p.PARENT_OUTPUT)
    parser.add_argument("--model-seed", type=int, choices=p.MODEL_SEEDS)
    parser.add_argument("--through-episode", type=int, default=p.TOTAL_EPISODES)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.command == "prepare":
        result = prepare(args.output)
    elif args.command == "self-check":
        result = self_check(args.output)
    elif args.command == "train-seed":
        require(args.model_seed is not None, "train-seed requires --model-seed")
        result = train_seed(
            args.output, args.model_seed, args.through_episode, args.device
        )
    elif args.command == "status":
        result = status(args.output)
    elif args.command == "analyze":
        result = analyze(args.output)
    else:
        result = compare(args.output, args.parent_output)
    print(json.dumps(result, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
