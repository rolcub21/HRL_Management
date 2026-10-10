"""Three-seed full-budget original-VCG base-80 Stage-A orchestration.

Seed 0 resumes the existing E70 run. Seeds 1 and 2 start independently under
E78. Analysis reports terminal and the entire prespecified late validation
window; it never promotes the best development checkpoint to a primary result.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
import os
from pathlib import Path
from statistics import mean, median, pstdev
from types import SimpleNamespace
from unittest.mock import patch

import torch

from experiments.conditioned_vcg.E70_original_vcg_reward_base_135k_v1 import (
    protocol as e70p,
    run as e70run,
)
from experiments.conditioned_vcg.target_size_dense_stage_a_v2 import run as v2run
from experiments.conditioned_vcg.target_size_dense_stage_a_v4 import run as v4run
import train_vcg_dense_proper as dense

from . import protocol as p


class E78Error(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise E78Error(message)


def _read(path: Path) -> dict:
    require(path.is_file(), f"E78 artifact is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"E78 artifact is invalid: {path}")
    return value


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n",
                         encoding="utf-8")
    os.replace(temporary, path)


def authenticate(output: Path) -> dict:
    observed = _read(output.resolve() / "contract.json")
    expected = p.contract()
    require(observed == expected, "E78 contract, E70 prefix, or sources changed")
    require(_read(p.E70_OUTPUT / "contract.json") == e70p.contract(),
            "E70 seed-0 contract changed")
    return expected


def prepare(output: Path) -> dict:
    output = output.resolve()
    expected = p.contract()
    path = output / "contract.json"
    if path.exists():
        authenticate(output)
    else:
        require(not output.exists() or not any(output.iterdir()),
                "E78 refuses a nonempty uncontracted output")
        _write(path, expected)
    return {"status": "prepared", "contract_sha256": expected["contract_sha256"],
            "model_seeds": list(p.AGGREGATE_SEEDS),
            "episodes_per_seed": p.EPISODES_BY_SCENARIO[p.SCENARIO],
            "training_runs": 0, "episode_runs": 0}


def self_check(output: Path) -> dict:
    contract = authenticate(output)
    require(p.SPEC.to_dict() == e70p.SPEC.to_dict()
            and p.SPEC.to_dict()["dense_b"] == 80.0,
            "E78 reward differs from the E70 base-80 pilot")
    require(p.SCENARIO == e70p.SCENARIO
            and p.VALIDATION_SEEDS == e70p.VALIDATION_SEEDS
            and p.TRAIN_BASE == e70p.TRAIN_BASE
            and p.VALIDATION_EVERY == e70p.VALIDATION_EVERY
            and p.CHECKPOINT_EVERY == e70p.CHECKPOINT_EVERY
            and p.TD_BACKUP_IMPLEMENTATION == e70p.TD_BACKUP_IMPLEMENTATION,
            "E78 target-size training recipe differs from E70")
    require(set(p.MODEL_SEEDS) == {1, 2}
            and set(p.AGGREGATE_SEEDS) == {0, 1, 2},
            "E78 model-seed allocation changed")
    with ExitStack() as stack:
        stack.enter_context(patch.object(v2run, "p", p))
        stack.enter_context(patch.object(dense, "FROZEN_OBJECTIVE_SPEC", p.SPEC))
        environment = v2run.self_check()
    require(environment["status"] == "passed"
            and len(environment["rows"]) == 1
            and environment["rows"][0]["jobs"] == 36,
            "E78 8x8 workload self-check failed")
    return {"status": "passed", "contract_sha256": contract["contract_sha256"],
            "scenario": p.SCENARIO, "seeds": list(p.AGGREGATE_SEEDS),
            "seed0_existing_episodes": _seed_summary(p.E70_OUTPUT, 0)[
                "completed_training_episodes"],
            "episode_runs": 0, "training_runs": 0}


def _summary_path(output: Path, seed: int) -> Path:
    base = p.E70_OUTPUT if seed == 0 else output
    return (base / "training/operational" / p.SCENARIO
            / f"seed-{seed}/training-summary.json")


def _seed_summary(output: Path, seed: int) -> dict:
    summary = _read(_summary_path(output, seed))
    expected_protocol = e70p.PROTOCOL if seed == 0 else p.PROTOCOL
    expected_contract = (e70p.contract()["contract_sha256"] if seed == 0
                         else p.contract()["contract_sha256"])
    require(summary.get("model_seed") == seed
            and summary.get("scenario") == p.SCENARIO
            and summary.get("training_protocol") == expected_protocol
            and summary.get("target_contract_sha256") == expected_contract
            and summary.get("objective_spec") == p.SPEC.to_dict()
            and summary.get("gamma") == 0.99
            and summary.get("total_training_episodes") == 112
            and not summary.get("test_panel_opened", False),
            f"E78 seed {seed} summary identity differs")
    completed = int(summary.get("completed_training_episodes", -1))
    require(0 <= completed <= 112, f"E78 seed {seed} episode count is invalid")
    if seed == 0:
        rows = [e70p._row(record) for record in summary["validation_history"]
                if int(record["checkpoint_episode"]) <= 30]
        require(p.digest(rows) == p.contract()["seed0_source"][
                    "first_30_validation_sha256"],
                "E70 seed-0 first-30 evidence changed")
    return summary


def train_seed(output: Path, seed: int, *, through_episode: int) -> dict:
    contract = authenticate(output)
    require(seed in p.AGGREGATE_SEEDS, "E78 model seed must be 0, 1, or 2")
    require(1 <= through_episode <= 112,
            "E78 through-episode must be between 1 and 112")
    require(torch.cuda.is_available(),
            "E78 full-budget training requires an available CUDA device")
    if seed == 0:
        result = e70run.train(p.E70_OUTPUT, through_episode=through_episode,
                             device="cuda")
    else:
        args = SimpleNamespace(output=output.resolve(), scenario=p.SCENARIO,
                               model_seed=seed, device="cuda",
                               stop_after_episode=through_episode,
                               allow_cpu_pilot=False)
        with ExitStack() as stack:
            stack.enter_context(patch.object(v4run, "p", p))
            stack.enter_context(patch.object(dense, "FROZEN_OBJECTIVE_SPEC", p.SPEC))
            result = v4run.train(args)
    return {**result, "e78_protocol": p.PROTOCOL,
            "e78_contract_sha256": contract["contract_sha256"],
            "aggregate_model_seed": seed,
            "through_episode": through_episode,
            "requested_device": "cuda", "paper_evidence": False}


def _validation_row(record: dict, seed: int) -> dict:
    runs = tuple(record["runs"])
    require(len(runs) == len(p.VALIDATION_SEEDS[p.SCENARIO])
            and [int(run["instance_seed"]) for run in runs]
            == list(p.VALIDATION_SEEDS[p.SCENARIO]),
            f"E78 seed {seed} validation instances differ")
    ids = [run["episode_instance_id"] for run in runs]
    complete = all(bool(run["strict_method_success"])
                   and int(run["delivery_count"]) == 36 for run in runs)
    deliveries = sum(int(run["delivery_count"]) for run in runs)
    relocations = sum(int(run["audit"]["relocations"]) for run in runs)
    def average(key: str) -> float | None:
        return mean(float(run[key]) for run in runs) if complete else None
    dense_return = average("dense_rescored_return")
    return {
        "checkpoint_episode": int(record["checkpoint_episode"]),
        "validation_instances": ids,
        "strict_complete_rows": sum(bool(run["strict_method_success"])
                                    and int(run["delivery_count"]) == 36
                                    for run in runs),
        "validation_rows": len(runs),
        "base80_return": dense_return,
        "base40_equivalent_return": (dense_return - 36 * 40.0
                                     if dense_return is not None else None),
        "placement_mae": average("mean_absolute_error"),
        "within_20_rate": average("within_target_window_rate"),
        "steps": mean(int(run["steps"]) for run in runs),
        "rehandles_per_100": (100.0 * relocations / deliveries
                              if complete and deliveries else None),
    }


def _late_window(records: list[dict]) -> dict:
    indexed = {row["checkpoint_episode"]: row for row in records}
    present = [indexed[episode] for episode in p.LATE_CHECKPOINTS
               if episode in indexed]
    checkpoints_present = len(present) == len(p.LATE_CHECKPOINTS)
    all_strict_complete = (
        checkpoints_present
        and all(row["strict_complete_rows"] == row["validation_rows"]
                for row in present)
    )
    output = {
        "prespecified_checkpoints": list(p.LATE_CHECKPOINTS),
        "observed_checkpoints": [row["checkpoint_episode"] for row in present],
        "checkpoints_present": checkpoints_present,
        "all_strict_complete": all_strict_complete,
        "strict_complete_rows": sum(row["strict_complete_rows"] for row in present),
        "total_rows": sum(row["validation_rows"] for row in present),
    }
    if all_strict_complete:
        mae = [float(row["placement_mae"]) for row in present]
        normalized_return = [float(row["base40_equivalent_return"])
                             for row in present]
        output.update({
            "mae_mean": mean(mae), "mae_median": median(mae),
            "mae_worst": max(mae), "mae_range": max(mae) - min(mae),
            "mae_checkpoint_sd": pstdev(mae),
            "base40_equivalent_return_mean": mean(normalized_return),
            "base40_equivalent_return_worst": min(normalized_return),
            "within_20_rate_mean": mean(float(row["within_20_rate"])
                                        for row in present),
            "steps_mean": mean(float(row["steps"]) for row in present),
            "rehandles_per_100_mean": mean(float(row["rehandles_per_100"])
                                            for row in present),
        })
    return output


def analyze(output: Path) -> dict:
    output = output.resolve()
    contract = authenticate(output)
    rows = []
    for seed in p.AGGREGATE_SEEDS:
        summary_path = _summary_path(output, seed)
        if not summary_path.is_file():
            rows.append({"model_seed": seed, "status": "not_started",
                         "completed_training_episodes": 0})
            continue
        summary = _seed_summary(output, seed)
        history = [_validation_row(record, seed)
                   for record in summary["validation_history"]]
        episodes = [row["checkpoint_episode"] for row in history]
        completed = int(summary["completed_training_episodes"])
        expected = list(range(p.VALIDATION_EVERY, completed + 1,
                              p.VALIDATION_EVERY))
        if completed == 112 and 112 not in expected:
            expected.append(112)
        require(episodes == expected,
                f"E78 seed {seed} validation checkpoint coverage changed")
        late = _late_window(history)
        terminal = history[-1] if completed == 112 else None
        require(terminal is None or terminal["checkpoint_episode"] == 112,
                f"E78 seed {seed} terminal validation is missing")
        rows.append({
            "model_seed": seed,
            "status": "complete" if completed == 112 else "partial",
            "source_protocol": e70p.PROTOCOL if seed == 0 else p.PROTOCOL,
            "training_summary": str(summary_path),
            "training_summary_sha256": p.sha256(summary_path),
            "completed_training_episodes": completed,
            "late_window": late,
            "terminal": terminal,
            "validation_history": history,
            "trainer_best_checkpoint_descriptive_only":
                summary.get("best_checkpoint"),
        })
    complete = (all(row["status"] == "complete" for row in rows)
                and all(row["late_window"]["checkpoints_present"]
                        for row in rows))
    all_strict = (complete and all(row["late_window"]["all_strict_complete"]
                                   and row["terminal"]["strict_complete_rows"]
                                   == row["terminal"]["validation_rows"]
                                   for row in rows))
    aggregate = None
    if all_strict:
        late_mae = [float(row["late_window"]["mae_mean"]) for row in rows]
        terminal_mae = [float(row["terminal"]["placement_mae"])
                        for row in rows]
        aggregate = {
            "late_window_mae_mean_across_seeds": mean(late_mae),
            "late_window_mae_sd_across_seeds": pstdev(late_mae),
            "late_window_mae_worst_seed": max(late_mae),
            "terminal_mae_mean_across_seeds": mean(terminal_mae),
            "terminal_mae_sd_across_seeds": pstdev(terminal_mae),
            "all_terminal_strict_rows": sum(row["terminal"][
                "strict_complete_rows"] for row in rows),
            "all_terminal_rows": sum(row["terminal"]["validation_rows"]
                                     for row in rows),
        }
    report = {
        "status": "complete" if complete else "partial",
        "protocol": p.PROTOCOL,
        "contract_sha256": contract["contract_sha256"],
        "scope": "8x8 original VCG Stage A, base 80, three model seeds",
        "seed0_reused_from_e70": True,
        "checkpoint_selection": "none; terminal and all late-window checkpoints reported",
        "late_window_checkpoint_set": list(p.LATE_CHECKPOINTS),
        "seeds": rows,
        "aggregate": aggregate,
        "all_late_and_terminal_rows_strict_complete": all_strict,
        "stage_b_trained": False,
        "test_panel_opened": False,
        "paper_evidence": False,
        "training_runs": 0,
        "episode_runs": 0,
    }
    path = output / "e78-report.json"
    _write(path, report)
    return {"status": report["status"], "report": str(path),
            "completed_training_episodes": {
                str(row["model_seed"]): row["completed_training_episodes"]
                for row in rows},
            "late_window_complete": {
                str(row["model_seed"]): bool(
                    row.get("late_window", {}).get("checkpoints_present"))
                for row in rows},
            "training_runs": 0, "episode_runs": 0}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "self-check", "train-seed", "analyze"))
    parser.add_argument("--output", type=Path, default=p.DEFAULT_OUTPUT)
    parser.add_argument("--model-seed", type=int, choices=p.AGGREGATE_SEEDS)
    parser.add_argument("--through-episode", type=int, default=112)
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare(args.output)
    elif args.command == "self-check":
        result = self_check(args.output)
    elif args.command == "train-seed":
        require(args.model_seed is not None, "train-seed requires --model-seed")
        result = train_seed(args.output, args.model_seed,
                            through_episode=args.through_episode)
    else:
        result = analyze(args.output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
