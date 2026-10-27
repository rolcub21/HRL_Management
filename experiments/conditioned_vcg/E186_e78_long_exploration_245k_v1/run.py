#!/usr/bin/env python3
"""Fresh 300-episode E78 replication with an E78-calibrated epsilon decay."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
import os
from pathlib import Path
from statistics import mean, pstdev
from types import SimpleNamespace
from unittest.mock import patch

import torch

from experiments.conditioned_vcg.target_size_dense_stage_a_v2 import run as v2run
from experiments.conditioned_vcg.target_size_dense_stage_a_v4 import acceleration
import train_vcg_dense_proper as dense
from train_viability_graph_smdp import epsilon_at
from viability_graph_episodic_audit import EpisodicViabilityGraphHierarchyAgent

from . import protocol as p


class E186Error(RuntimeError):
    pass


BUDGET_RULE = "fixed_300_episodes_long_exploration"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise E186Error(message)


def _read(path: Path) -> dict:
    require(path.is_file(), f"E186 artifact is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"E186 artifact is invalid: {path}")
    return value


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def authenticate(output: Path) -> dict:
    observed = _read(output.resolve() / "contract.json")
    expected = p.contract()
    require(observed == expected, "E186 contract, E78 evidence, or sources changed")
    return expected


def prepare(output: Path) -> dict:
    output = output.resolve()
    expected = p.contract()
    path = output / "contract.json"
    if path.exists():
        authenticate(output)
    else:
        require(
            not output.exists() or not any(output.iterdir()),
            "E186 refuses a nonempty uncontracted output",
        )
        _write(path, expected)
    return {
        "status": "prepared",
        "contract_sha256": expected["contract_sha256"],
        "fresh_start": True,
        "model_seeds": list(p.MODEL_SEEDS),
        "episodes_per_seed": p.TOTAL_EPISODES,
        "epsilon_decay_decisions": p.EPSILON_DECAY_DECISIONS,
        "approximate_floor_episode": expected["epsilon"]["derivation"][
            "implied_floor_episode_at_e78_mean"
        ],
        "validation_every": p.VALIDATION_EVERY,
        "training_runs": 0,
        "episode_runs": 0,
    }


def self_check(output: Path) -> dict:
    contract = authenticate(output)
    require(p.TOTAL_EPISODES == 300, "E186 episode budget changed")
    require(p.MODEL_SEEDS == (0, 1, 2), "E186 seed set changed")
    require(p.SPEC.to_dict()["dense_b"] == 80.0, "E186 is not base-80")
    require(p.SCENARIO == "empty_8x8_n36", "E186 scenario changed")
    require(contract["fresh_start"] and contract["warm_start_checkpoint"] is None,
            "E186 is not a fresh run")
    evidence = contract["epsilon"]["derivation"]
    require(evidence["total_seed_episodes"] == 336
            and evidence["total_macro_decisions"] == 28_336,
            "E78 epsilon calibration evidence changed")
    schedule = (
        epsilon_at(0, start=p.EPSILON_START, end=p.EPSILON_END,
                   warmup_decisions=p.EPSILON_WARMUP_DECISIONS,
                   decay_decisions=p.EPSILON_DECAY_DECISIONS),
        epsilon_at(p.EPSILON_DECAY_DECISIONS // 2,
                   start=p.EPSILON_START, end=p.EPSILON_END,
                   warmup_decisions=p.EPSILON_WARMUP_DECISIONS,
                   decay_decisions=p.EPSILON_DECAY_DECISIONS),
        epsilon_at(p.EPSILON_DECAY_DECISIONS,
                   start=p.EPSILON_START, end=p.EPSILON_END,
                   warmup_decisions=p.EPSILON_WARMUP_DECISIONS,
                   decay_decisions=p.EPSILON_DECAY_DECISIONS),
    )
    require(all(abs(a - b) < 1e-12 for a, b in zip(schedule, (0.90, 0.475, 0.05))),
            "E186 epsilon schedule arithmetic changed")
    with ExitStack() as stack:
        stack.enter_context(patch.object(v2run, "p", p))
        stack.enter_context(patch.object(dense, "FROZEN_OBJECTIVE_SPEC", p.SPEC))
        environment = v2run.self_check()
    require(environment["status"] == "passed"
            and len(environment["rows"]) == 1
            and environment["rows"][0]["jobs"] == 36,
            "E186 environment self-check failed")
    return {
        "status": "passed",
        "contract_sha256": contract["contract_sha256"],
        "scenario": p.SCENARIO,
        "model_seeds": list(p.MODEL_SEEDS),
        "episodes_per_seed": p.TOTAL_EPISODES,
        "epsilon": {
            "at_decision_0": schedule[0],
            "at_decision_9250": schedule[1],
            "at_decision_18500": schedule[2],
            "implied_floor_episode": evidence["implied_floor_episode_at_e78_mean"],
        },
        "validation_every": p.VALIDATION_EVERY,
        "training_runs": 0,
        "episode_runs": 0,
    }


def _train_stage_a(args: SimpleNamespace) -> dict:
    """Retain Stage-A-v4 acceleration without its obsolete 4,000-job label."""

    original_result = dense._result_payload
    original_checkpoint = EpisodicViabilityGraphHierarchyAgent.checkpoint
    original_validate = dense._validate_resume

    def result_payload(*positional, **keywords):
        value = original_result(*positional, **keywords)
        value["td_backup_acceleration"] = acceleration.audit_payload(
            keywords["agent"]
        )
        value["training_budget_rule"] = {
            "rule": BUDGET_RULE,
            "episodes_per_seed": p.TOTAL_EPISODES,
            "training_jobs_per_seed": p.TRAINING_JOB_BUDGET,
        }
        value["e186_epsilon_schedule"] = {
            "start": p.EPSILON_START,
            "end": p.EPSILON_END,
            "warmup_decisions": p.EPSILON_WARMUP_DECISIONS,
            "decay_decisions": p.EPSILON_DECAY_DECISIONS,
        }
        return value

    def checkpoint(agent, *positional, **keywords):
        value = original_checkpoint(agent, *positional, **keywords)
        value["td_backup_implementation"] = p.TD_BACKUP_IMPLEMENTATION
        value["td_backup_acceleration_audit"] = acceleration.audit_payload(agent)
        value["e186_training_budget_rule"] = BUDGET_RULE
        value["e186_epsilon_decay_decisions"] = p.EPSILON_DECAY_DECISIONS
        return value

    def validate_resume(payload, contract):
        original_validate(payload, contract)
        if payload.get("td_backup_implementation") != p.TD_BACKUP_IMPLEMENTATION:
            raise ValueError("checkpoint was not produced by accelerated Stage A")
        if payload.get("e186_training_budget_rule") != BUDGET_RULE:
            raise ValueError("checkpoint has another training-budget rule")
        if payload.get("e186_epsilon_decay_decisions") != p.EPSILON_DECAY_DECISIONS:
            raise ValueError("checkpoint has another epsilon schedule")

    with ExitStack() as stack:
        stack.enter_context(patch.object(v2run, "p", p))
        stack.enter_context(patch.object(dense, "FROZEN_OBJECTIVE_SPEC", p.SPEC))
        stack.enter_context(patch.object(dense, "FROZEN_EPSILON_START", p.EPSILON_START))
        stack.enter_context(patch.object(dense, "FROZEN_EPSILON_END", p.EPSILON_END))
        stack.enter_context(patch.object(
            dense, "FROZEN_EPSILON_WARMUP_DECISIONS", p.EPSILON_WARMUP_DECISIONS
        ))
        stack.enter_context(patch.object(
            dense, "FROZEN_EPSILON_DECAY_DECISIONS", p.EPSILON_DECAY_DECISIONS
        ))
        stack.enter_context(patch.object(
            EpisodicViabilityGraphHierarchyAgent,
            "_td_batch",
            acceleration.accelerated_cached_td_batch,
        ))
        stack.enter_context(patch.object(
            EpisodicViabilityGraphHierarchyAgent, "checkpoint", checkpoint
        ))
        stack.enter_context(patch.object(dense, "_result_payload", result_payload))
        stack.enter_context(patch.object(dense, "_validate_resume", validate_resume))
        return v2run.train_stage_a(
            args.output,
            args.scenario,
            args.model_seed,
            device_name=args.device,
            stop_after_episode=args.stop_after_episode,
            allow_cpu_pilot=False,
        )


def train_seed(output: Path, seed: int, *, through_episode: int) -> dict:
    contract = authenticate(output)
    require(seed in p.MODEL_SEEDS, "E186 model seed must be 0, 1, or 2")
    require(1 <= through_episode <= p.TOTAL_EPISODES,
            "E186 through-episode must be between 1 and 300")
    require(torch.cuda.is_available(), "E186 full training requires CUDA")
    args = SimpleNamespace(
        output=output.resolve(),
        scenario=p.SCENARIO,
        model_seed=seed,
        device="cuda",
        stop_after_episode=through_episode,
    )
    result = _train_stage_a(args)
    return {
        **result,
        "e186_protocol": p.PROTOCOL,
        "e186_contract_sha256": contract["contract_sha256"],
        "through_episode": through_episode,
        "fresh_start_protocol": True,
        "paper_evidence": False,
    }


def _summary_path(output: Path, seed: int) -> Path:
    return (output / "training/operational" / p.SCENARIO
            / f"seed-{seed}/training-summary.json")


def _seed_summary(output: Path, seed: int) -> dict:
    summary = _read(_summary_path(output, seed))
    require(summary.get("model_seed") == seed
            and summary.get("scenario") == p.SCENARIO
            and summary.get("training_protocol") == p.PROTOCOL
            and summary.get("target_contract_sha256") == p.contract()["contract_sha256"]
            and summary.get("objective_spec") == p.SPEC.to_dict()
            and summary.get("gamma") == 0.99
            and summary.get("total_training_episodes") == p.TOTAL_EPISODES
            and summary.get("resume_contract", {}).get("epsilon_schedule", {}).get(
                "decay_decisions") == p.EPSILON_DECAY_DECISIONS
            and not summary.get("test_panel_opened", False),
            f"E186 seed {seed} summary identity differs")
    completed = int(summary.get("completed_training_episodes", -1))
    require(0 <= completed <= p.TOTAL_EPISODES,
            f"E186 seed {seed} episode count is invalid")
    return summary


def _validation_row(record: dict, cumulative_decisions: int) -> dict:
    runs = tuple(record["runs"])
    complete = all(
        bool(run["strict_method_success"])
        and int(run["delivery_count"]) == 36 for run in runs
    )
    deliveries = sum(int(run["delivery_count"]) for run in runs)
    relocations = sum(int(run["audit"]["relocations"]) for run in runs)

    def average(key: str):
        return mean(float(run[key]) for run in runs) if complete else None

    dense_return = average("dense_rescored_return")
    return {
        "checkpoint_episode": int(record["checkpoint_episode"]),
        "cumulative_training_decisions": cumulative_decisions,
        "epsilon": epsilon_at(
            cumulative_decisions,
            start=p.EPSILON_START,
            end=p.EPSILON_END,
            warmup_decisions=p.EPSILON_WARMUP_DECISIONS,
            decay_decisions=p.EPSILON_DECAY_DECISIONS,
        ),
        "strict_complete_rows": sum(
            bool(run["strict_method_success"])
            and int(run["delivery_count"]) == 36 for run in runs
        ),
        "validation_rows": len(runs),
        "base80_return": dense_return,
        "placement_mae": average("mean_absolute_error"),
        "within_20_rate": average("within_target_window_rate"),
        "steps": mean(int(run["steps"]) for run in runs),
        "rehandles_per_100": (
            100.0 * relocations / deliveries if complete and deliveries else None
        ),
    }


def _late_window(rows: list[dict]) -> dict:
    indexed = {row["checkpoint_episode"]: row for row in rows}
    present = [indexed[number] for number in p.REPORT_CHECKPOINTS if number in indexed]
    coverage = len(present) == len(p.REPORT_CHECKPOINTS)
    all_complete = coverage and all(
        row["strict_complete_rows"] == row["validation_rows"] for row in present
    )
    value = {
        "prespecified_checkpoints": list(p.REPORT_CHECKPOINTS),
        "observed_checkpoints": [row["checkpoint_episode"] for row in present],
        "checkpoints_present": coverage,
        "all_strict_complete": all_complete,
    }
    if all_complete:
        mae = [float(row["placement_mae"]) for row in present]
        value.update({
            "mae_mean": mean(mae),
            "mae_worst": max(mae),
            "mae_checkpoint_sd": pstdev(mae),
            "within_20_rate_mean": mean(float(row["within_20_rate"]) for row in present),
            "rehandles_per_100_mean": mean(
                float(row["rehandles_per_100"]) for row in present
            ),
        })
    return value


def analyze(output: Path) -> dict:
    output = output.resolve()
    contract = authenticate(output)
    seeds = []
    for seed in p.MODEL_SEEDS:
        path = _summary_path(output, seed)
        if not path.is_file():
            seeds.append({"model_seed": seed, "status": "not_started",
                          "completed_training_episodes": 0})
            continue
        summary = _seed_summary(output, seed)
        completed = int(summary["completed_training_episodes"])
        training_runs = list(summary["training"]["runs"])
        cumulative = []
        running = 0
        for item in training_runs:
            running += int(item["macro_decisions"])
            cumulative.append(running)
        rows = []
        for record in summary["validation_history"]:
            number = int(record["checkpoint_episode"])
            require(1 <= number <= len(cumulative),
                    f"E186 seed {seed} validation exceeds training history")
            rows.append(_validation_row(record, cumulative[number - 1]))
        expected = list(range(p.VALIDATION_EVERY, completed + 1,
                              p.VALIDATION_EVERY))
        require([row["checkpoint_episode"] for row in rows] == expected,
                f"E186 seed {seed} validation coverage differs")
        terminal = rows[-1] if completed == p.TOTAL_EPISODES else None
        seeds.append({
            "model_seed": seed,
            "status": "complete" if completed == p.TOTAL_EPISODES else "partial",
            "completed_training_episodes": completed,
            "training_summary": str(path),
            "training_summary_sha256": p.sha256(path),
            "terminal": terminal,
            "late_window": _late_window(rows),
            "validation_history": rows,
            "trainer_best_checkpoint_descriptive_only": summary.get("best_checkpoint"),
        })
    complete = all(row["status"] == "complete" for row in seeds)
    strict = complete and all(
        row["terminal"]["strict_complete_rows"] == row["terminal"]["validation_rows"]
        and row["late_window"]["all_strict_complete"] for row in seeds
    )
    aggregate = None
    if strict:
        terminal_mae = [float(row["terminal"]["placement_mae"]) for row in seeds]
        late_mae = [float(row["late_window"]["mae_mean"]) for row in seeds]
        aggregate = {
            "terminal_mae_mean_across_seeds": mean(terminal_mae),
            "terminal_mae_sd_across_seeds": pstdev(terminal_mae),
            "late_window_mae_mean_across_seeds": mean(late_mae),
            "late_window_mae_sd_across_seeds": pstdev(late_mae),
        }
    report = {
        "status": "complete" if complete else "partial",
        "protocol": p.PROTOCOL,
        "contract_sha256": contract["contract_sha256"],
        "scope": "fresh 300-episode original E78 architecture, three seeds",
        "epsilon_decay_decisions": p.EPSILON_DECAY_DECISIONS,
        "target_floor_episode_approximate": p.EPSILON_TARGET_FLOOR_EPISODE,
        "checkpoint_selection": "none; terminal and prespecified late curve reported",
        "seeds": seeds,
        "aggregate": aggregate,
        "all_late_and_terminal_rows_strict_complete": strict,
        "stage_b_trained": False,
        "test_panel_opened": False,
        "paper_evidence": False,
        "training_runs": 0,
        "episode_runs": 0,
    }
    destination = output / "e186-report.json"
    _write(destination, report)
    return {"status": report["status"], "report": str(destination),
            "training_runs": 0, "episode_runs": 0}


def status(output: Path) -> dict:
    authenticate(output)
    rows = []
    for seed in p.MODEL_SEEDS:
        path = _summary_path(output, seed)
        if path.is_file():
            value = _seed_summary(output, seed)
            rows.append({
                "model_seed": seed,
                "completed_training_episodes": value["completed_training_episodes"],
                "epsilon": value["agent_audit"]["epsilon"],
                "decision_count": value["agent_audit"]["decision_count"],
                "status": value["status"],
            })
        else:
            rows.append({"model_seed": seed, "completed_training_episodes": 0,
                         "status": "not_started"})
    return {"status": "ok", "seeds": rows, "training_runs": 0,
            "episode_runs": 0}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("prepare", "self-check", "train-seed", "analyze", "status")
    )
    parser.add_argument("--output", type=Path, default=p.DEFAULT_OUTPUT)
    parser.add_argument("--model-seed", type=int, choices=p.MODEL_SEEDS)
    parser.add_argument("--through-episode", type=int, default=p.TOTAL_EPISODES)
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare(args.output)
    elif args.command == "self-check":
        result = self_check(args.output)
    elif args.command == "train-seed":
        require(args.model_seed is not None, "train-seed requires --model-seed")
        result = train_seed(
            args.output, args.model_seed, through_episode=args.through_episode
        )
    elif args.command == "analyze":
        result = analyze(args.output)
    else:
        result = status(args.output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
