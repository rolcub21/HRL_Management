#!/usr/bin/env python3
"""Run Dynamic PSLAP and GA on E115's exact validation instances."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
from statistics import fmean, stdev
from unittest.mock import patch

import torch
from scipy.stats import t as student_t

from example.episode_instance import EpisodeInstance
from experiments.conditioned_vcg.E28_empty_start_scale_baselines_100k import adapter
from experiments.conditioned_vcg.E91_matched_target_size_benchmark_156k_v1 import (
    run as e91,
)
from experiments.conditioned_vcg.E115_release_et_cost_vcg_8x8_179k_v1 import (
    protocol as e115p,
    rewards as objective,
)
from . import protocol as p


class E116Error(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise E116Error(message)


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def read_json(path: Path) -> dict:
    require(path.is_file(), f"missing E116 artifact: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"invalid E116 JSON: {path}")
    return value


def make_instance(seed: int) -> EpisodeInstance:
    require(seed in p.INSTANCE_SEEDS, "seed is outside the frozen E116 panel")
    env = objective.ReleaseETCostEnv(
        e115p.SCENARIO, e115p.e107.e105.e78.SPEC
    )
    instance = env.sample_episode_instance(seed=seed)
    instance.validate_for(e91.p.SCENARIO.make_base_env())
    return instance


def manifest(contract: dict) -> dict:
    records = []
    for seed in p.INSTANCE_SEEDS:
        instance = make_instance(seed)
        records.append({
            "seed": seed,
            "instance_id": instance.instance_id,
            "schedule_id": instance.schedule_id,
            "instance_json": instance.to_json(),
        })
    payload = {
        "protocol": p.PROTOCOL,
        "contract_sha256": contract["contract_sha256"],
        "records": records,
    }
    return {**payload, "manifest_sha256": p.digest(payload)}


def prepare(output: Path) -> dict:
    expected = p.contract()
    contract_path = output / "contract.json"
    manifest_path = output / "instances.json"
    if contract_path.is_file():
        require(read_json(contract_path) == expected, "E116 contract changed")
        require(
            read_json(manifest_path) == manifest(expected),
            "E116 instance manifest changed",
        )
    else:
        require(
            not output.exists() or not any(output.iterdir()),
            "E116 refuses a nonempty uncontracted output",
        )
        write_json(contract_path, expected)
        write_json(manifest_path, manifest(expected))
    return {
        "status": "prepared",
        "instances": len(p.INSTANCE_SEEDS),
        "dynamic_pslap_episodes": len(p.INSTANCE_SEEDS),
        "ga_episodes": len(p.INSTANCE_SEEDS) * len(p.GA_REPLICATES),
        "contract_sha256": expected["contract_sha256"],
        "training_runs": 0,
        "episode_runs": 0,
    }


def authenticate(output: Path) -> tuple[dict, dict]:
    prepared = prepare(output)
    contract = read_json(output / "contract.json")
    frozen = read_json(output / "instances.json")
    require(prepared["contract_sha256"] == contract["contract_sha256"], "bad contract")
    return contract, frozen


def frozen_instance(frozen: dict, seed: int) -> EpisodeInstance:
    matches = [row for row in frozen["records"] if row["seed"] == seed]
    require(len(matches) == 1, "E116 instance is not uniquely frozen")
    row = matches[0]
    instance = EpisodeInstance.from_json(row["instance_json"])
    require(
        instance.instance_id == row["instance_id"]
        and instance.schedule_id == row["schedule_id"],
        "E116 frozen instance identity changed",
    )
    instance.validate_for(e91.p.SCENARIO.make_base_env())
    return instance


def replicates(method: str) -> tuple[int, ...]:
    require(method in p.METHODS, "unknown E116 method")
    return p.GA_REPLICATES if method == "ga" else (0,)


def row_path(output: Path, method: str, seed: int, replicate: int) -> Path:
    return output / "rows" / method / f"instance-{seed}-replicate-{replicate}.json"


class TracedEmptyStartEnv(adapter.AuditedEmptyStartEnv):
    """Observe primitive work without changing the baseline controller."""

    created: list["TracedEmptyStartEnv"] = []

    def __init__(self, scenario):
        super().__init__(scenario)
        self.__class__.created.append(self)

    def reset(self, instance=None):
        state = super().reset(instance=instance)
        self.e116_action_counts = Counter()
        self.e116_rehandle_events = 0
        return state

    def step(self, action):
        action_name = str(self.ACTION_NAMES[int(action)])
        result = super().step(action)
        self.e116_action_counts[action_name] += 1
        self.e116_rehandle_events += int(bool(result[3].get("relocated_block")))
        return result


def _average(values) -> float | None:
    values = tuple(float(value) for value in values)
    return float(fmean(values)) if values else None


def objective_metrics(env, strict: bool, reported_rehandles) -> dict:
    blocks = tuple(env.blocks)
    stored = tuple(
        block for block in blocks
        if getattr(block, "stored_time_step", None) is not None
    )
    delivered = tuple(
        block for block in blocks
        if getattr(block, "delivered_time_step", None) is not None
    )
    waits = tuple(
        int(block.stored_time_step) - int(block.arrival_step) for block in stored
    )
    errors = tuple(objective.release_error(block) for block in delivered)
    storage_errors = tuple(
        int(block.delivered_time_step)
        - int(block.stored_time_step)
        - int(block.storage_steps_needed)
        for block in delivered
    )
    tardiness = tuple(objective.earliness_tardiness(value)[0] for value in errors)
    earliness = tuple(objective.earliness_tardiness(value)[1] for value in errors)
    action_counts = dict(sorted(env.e116_action_counts.items()))
    wait_steps = int(action_counts.get("WAIT", 0))
    equipment_work = int(sum(action_counts.values()) - wait_steps)
    rehandles = int(env.e116_rehandle_events)
    if reported_rehandles is not None:
        require(
            rehandles == int(reported_rehandles),
            "primitive rehandle trace differs from baseline report",
        )
    require(
        sum(action_counts.values()) == int(env.time_steps),
        "primitive action trace differs from environment time",
    )
    components = {
        "release_tardiness": float(sum(tardiness)),
        "release_earliness": float(sum(earliness)),
        "equipment_work": float(equipment_work),
        "physical_rehandles": float(rehandles),
    }
    comparable = bool(strict and len(delivered) == len(blocks))
    cost = (
        objective.objective_cost(tuple(components[name] for name in (
            "release_tardiness",
            "release_earliness",
            "equipment_work",
            "physical_rehandles",
        )))
        if comparable else None
    )
    return {
        "required_jobs": len(blocks),
        "stored_jobs": len(stored),
        "delivered_jobs": len(delivered),
        "strict_complete": comparable,
        "objective_cost": cost,
        "components": components,
        "mean_release_tardiness_beyond_tolerance": _average(tardiness),
        "mean_release_earliness_beyond_tolerance": _average(earliness),
        "release_clock_mae": _average(abs(value) for value in errors),
        "release_clock_mean_signed_error": _average(errors),
        "within_tolerance_rate": (
            sum(abs(value) <= objective.TOLERANCE for value in errors) / len(errors)
            if errors else None
        ),
        "mean_wait_before_storage": _average(waits),
        "max_wait_before_storage": max(waits) if waits else None,
        "storage_clock_mae": _average(abs(value) for value in storage_errors),
        "steps": int(env.time_steps),
        "steps_per_delivery": (
            int(env.time_steps) / len(delivered) if delivered else None
        ),
        "equipment_work_per_delivery": (
            equipment_work / len(delivered) if delivered else None
        ),
        "rehandles_per_100": (
            100.0 * rehandles / len(blocks) if comparable else None
        ),
        "primitive_action_counts": action_counts,
        "wait_primitive_steps": wait_steps,
        "objective_comparable": comparable,
    }


def run_one(
    output: Path, method: str, seed: int, replicate: int
) -> dict:
    contract, frozen = authenticate(output)
    require(seed in p.INSTANCE_SEEDS, "seed is outside E116")
    require(replicate in replicates(method), "replicate is outside E116")
    path = row_path(output, method, seed, replicate)
    identity = {
        "protocol": p.PROTOCOL,
        "contract_sha256": contract["contract_sha256"],
        "method": method,
        "instance_seed": seed,
        "replicate": replicate,
    }
    if path.is_file():
        row = read_json(path)
        require(
            all(row.get(key) == value for key, value in identity.items()),
            f"saved E116 coordinate differs: {path}",
        )
        return {**row, "reused": True, "episode_runs": 0}

    instance = frozen_instance(frozen, seed)
    TracedEmptyStartEnv.created = []
    print(
        f"E116 start | {method} seed={seed} replicate={replicate} "
        f"instance={instance.instance_id}",
        flush=True,
    )
    with patch.object(adapter, "AuditedEmptyStartEnv", TracedEmptyStartEnv):
        baseline = e91._baseline_row(
            method,
            instance,
            seed,
            replicate,
            p.INSTANCE_SEEDS.index(seed),
        )
    require(
        len(TracedEmptyStartEnv.created) == 1,
        "E116 baseline did not create exactly one traced environment",
    )
    env = TracedEmptyStartEnv.created[0]
    require(
        env.current_episode_instance.instance_id == instance.instance_id,
        "E116 baseline executed another instance",
    )
    metrics = objective_metrics(
        env, bool(baseline["strict_complete"]), baseline["rehandles_at_stop"]
    )
    require(
        metrics["strict_complete"] == bool(baseline["strict_complete"]),
        "E116 objective and baseline completion flags differ",
    )
    row = {
        **identity,
        "instance_id": instance.instance_id,
        "schedule_id": instance.schedule_id,
        "status": "complete",
        "rng_seed": baseline["rng_seed"],
        "failure_class": baseline["failure_class"],
        "method_failure_reason": baseline["method_failure_reason"],
        "decision_count": baseline["decision_count"],
        "episode_wall_seconds": baseline["episode_wall_seconds"],
        "wall_seconds_per_decision": baseline["amortized_wall_seconds_per_decision"],
        "baseline_reported_mae": baseline["baseline_reported_mae"],
        "metrics": metrics,
        "training_runs": 0,
        "episode_runs": 1,
    }
    write_json(path, row)
    print(
        f"E116 done  | {method} seed={seed} replicate={replicate} "
        f"strict={int(metrics['strict_complete'])} "
        f"J={metrics['objective_cost']} steps={metrics['steps']} "
        f"wait={metrics['mean_wait_before_storage']:.1f} "
        f"wall={baseline['episode_wall_seconds']:.1f}s",
        flush=True,
    )
    return row


def run_method(output: Path, method: str) -> dict:
    episode_runs = 0
    for seed in p.INSTANCE_SEEDS:
        for replicate in replicates(method):
            row = run_one(output, method, seed, replicate)
            episode_runs += int(not row.get("reused", False))
    return {
        "status": "complete",
        "method": method,
        "rows": len(p.INSTANCE_SEEDS) * len(replicates(method)),
        "episode_runs": episode_runs,
        "training_runs": 0,
    }


def _metric(row: dict, name: str):
    metrics = row["metrics"]
    if name.startswith("components."):
        return metrics["components"][name.split(".", 1)[1]]
    return metrics[name]


METRICS = (
    "objective_cost",
    "components.release_tardiness",
    "components.release_earliness",
    "equipment_work_per_delivery",
    "rehandles_per_100",
    "release_clock_mae",
    "within_tolerance_rate",
    "mean_wait_before_storage",
    "storage_clock_mae",
    "steps_per_delivery",
)


def _baseline_points(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    points = []
    summaries = []
    for method in p.METHODS:
        for seed in p.INSTANCE_SEEDS:
            group = [
                row for row in rows
                if row["method"] == method and row["instance_seed"] == seed
            ]
            expected = len(replicates(method))
            complete = len(group) == expected and all(
                row["metrics"]["strict_complete"] for row in group
            )
            points.append({
                "method": method,
                "instance_seed": seed,
                "replicates": len(group),
                "expected_replicates": expected,
                "strict_completion_rate": (
                    _average(row["metrics"]["strict_complete"] for row in group)
                    if group else None
                ),
                "all_replicates_strict_complete": complete,
                "metrics": (
                    {name: _average(_metric(row, name) for row in group)
                     for name in METRICS}
                    if complete else None
                ),
                "mean_episode_wall_seconds": (
                    _average(row["episode_wall_seconds"] for row in group)
                    if group else None
                ),
                "mean_wall_seconds_per_decision": (
                    _average(row["wall_seconds_per_decision"] for row in group)
                    if group and all(row["wall_seconds_per_decision"] is not None
                                     for row in group) else None
                ),
            })
        selected = [point for point in points if point["method"] == method]
        complete_points = [
            point for point in selected if point["all_replicates_strict_complete"]
        ]
        raw = [row for row in rows if row["method"] == method]
        summaries.append({
            "method": method,
            "observed_rows": len(raw),
            "expected_rows": len(p.INSTANCE_SEEDS) * len(replicates(method)),
            "strict_complete_rows": sum(
                row["metrics"]["strict_complete"] for row in raw
            ),
            "fully_complete_instances": len(complete_points),
            "metrics": (
                {name: _average(point["metrics"][name] for point in complete_points)
                 for name in METRICS}
                if complete_points else None
            ),
            "mean_episode_wall_seconds": _average(
                point["mean_episode_wall_seconds"] for point in selected
                if point["mean_episode_wall_seconds"] is not None
            ),
            "mean_wall_seconds_per_decision": _average(
                point["mean_wall_seconds_per_decision"] for point in selected
                if point["mean_wall_seconds_per_decision"] is not None
            ),
        })
    return points, summaries


def _e115_checkpoint_points(record: dict) -> list[dict]:
    points = []
    for run in record["runs"]:
        seed = int(run["instance_seed"])
        require(seed in p.INSTANCE_SEEDS, "E115 validation contains another seed")
        service = run["service_metrics"]
        components = run["factorized_component_totals"]
        complete = bool(run["strict_method_success"] and service["strict_complete"])
        points.append({
            "method": "e115",
            "instance_seed": seed,
            "strict_complete": complete,
            "metrics": ({
                "objective_cost": float(service["objective_cost"]),
                "components.release_tardiness": float(
                    components["release_tardiness_accrual"]
                ),
                "components.release_earliness": float(
                    components["release_earliness_at_delivery"]
                ),
                "equipment_work_per_delivery": float(
                    service["equipment_work_per_delivery"]
                ),
                "rehandles_per_100": (
                    100.0 * float(service["physical_rehandles"])
                    / float(service["required_jobs"])
                ),
                "release_clock_mae": float(service["release_clock_mae"]),
                "within_tolerance_rate": float(service["within_tolerance_rate"]),
                "mean_wait_before_storage": float(
                    service["mean_wait_before_storage"]
                ),
                "storage_clock_mae": float(service["storage_clock_mae"]),
                "steps_per_delivery": float(run["steps"]) / float(
                    service["delivered_jobs"]
                ),
            } if complete else None),
        })
    require(
        sorted(point["instance_seed"] for point in points) == sorted(p.INSTANCE_SEEDS),
        "E115 validation does not contain the five E116 instances",
    )
    return points


def _paired(values: list[float]) -> dict:
    require(len(values) == len(p.INSTANCE_SEEDS), "paired comparison needs five seeds")
    center = float(fmean(values))
    if len(values) < 2:
        interval = [center, center]
    else:
        standard_error = stdev(values) / math.sqrt(len(values))
        critical = float(student_t.ppf(0.975, len(values) - 1))
        interval = [center - critical * standard_error,
                    center + critical * standard_error]
    return {
        "n_instances": len(values),
        "mean_difference": center,
        "nominal_95_ci": interval,
    }


def _comparison(
    checkpoint: int, e115_points: list[dict], baseline_points: list[dict]
) -> dict:
    e115_by_seed = {point["instance_seed"]: point for point in e115_points}
    all_e115_complete = all(point["strict_complete"] for point in e115_points)
    e115_summary = {
        "method": "e115",
        "checkpoint_episode": checkpoint,
        "strict_complete_instances": sum(
            point["strict_complete"] for point in e115_points
        ),
        "metrics": (
            {name: _average(point["metrics"][name] for point in e115_points)
             for name in METRICS}
            if all_e115_complete else None
        ),
    }
    contrasts = []
    for method in p.METHODS:
        selected = [point for point in baseline_points if point["method"] == method]
        comparable = (
            all_e115_complete
            and len(selected) == len(p.INSTANCE_SEEDS)
            and all(point["all_replicates_strict_complete"] for point in selected)
        )
        baseline_by_seed = {point["instance_seed"]: point for point in selected}
        differences = None
        if comparable:
            differences = {
                name: _paired([
                    e115_by_seed[seed]["metrics"][name]
                    - baseline_by_seed[seed]["metrics"][name]
                    for seed in p.INSTANCE_SEEDS
                ])
                for name in METRICS
            }
        contrasts.append({
            "checkpoint_episode": checkpoint,
            "baseline": method,
            "difference_direction": "E115 minus baseline",
            "all_five_instances_comparable": comparable,
            "metrics": differences,
        })
    return {"e115": e115_summary, "contrasts": contrasts}


def _latest_e115_summary() -> tuple[Path, dict]:
    path = (
        p.E115_OUTPUT / "training/operational" / p.SCENARIO_ID
        / "seed-0/training-summary.json"
    )
    summary = read_json(path)
    require(
        summary.get("e115_contract_sha256") == p.contract()["e115_contract_sha256"],
        "E115 training summary belongs to another contract",
    )
    return path, summary


def analyze(output: Path) -> dict:
    contract, _ = authenticate(output)
    expected = [
        (method, seed, replicate)
        for method in p.METHODS
        for seed in p.INSTANCE_SEEDS
        for replicate in replicates(method)
    ]
    rows = []
    missing = []
    for method, seed, replicate in expected:
        path = row_path(output, method, seed, replicate)
        if not path.is_file():
            missing.append(str(path))
            continue
        row = read_json(path)
        require(
            row.get("contract_sha256") == contract["contract_sha256"]
            and row.get("method") == method
            and row.get("instance_seed") == seed
            and row.get("replicate") == replicate,
            f"E116 row identity differs: {path}",
        )
        rows.append(row)
    baseline_points, baseline_summaries = _baseline_points(rows)
    summary_path, e115_summary = _latest_e115_summary()
    validation = sorted(
        e115_summary.get("validation_history", []),
        key=lambda record: int(record["checkpoint_episode"]),
    )
    checkpoint_comparisons = []
    if not missing:
        for record in validation:
            checkpoint = int(record["checkpoint_episode"])
            points = _e115_checkpoint_points(record)
            checkpoint_comparisons.append(
                _comparison(checkpoint, points, baseline_points)
            )
    report = {
        "protocol": p.PROTOCOL,
        "contract_sha256": contract["contract_sha256"],
        "status": "complete" if not missing else "partial",
        "expected_baseline_rows": len(expected),
        "observed_baseline_rows": len(rows),
        "missing_rows": missing,
        "all_baseline_rows_strict_complete": (
            not missing and all(row["metrics"]["strict_complete"] for row in rows)
        ),
        "baseline_instance_points": baseline_points,
        "baseline_summaries": baseline_summaries,
        "e115_training_summary": str(summary_path),
        "e115_completed_training_episodes": e115_summary.get(
            "completed_training_episodes"
        ),
        "e115_validation_checkpoints": [
            int(record["checkpoint_episode"]) for record in validation
        ],
        "checkpoint_comparisons": checkpoint_comparisons,
        "claim_status": (
            "development comparison only; superiority not established on a fresh test"
        ),
        "interval_caveat": (
            "nominal paired t intervals over five reused development instances; "
            "not multiplicity-adjusted and not paper evidence"
        ),
        "training_runs": 0,
        "episode_runs": 0,
    }
    report_path = output / "e116-report.json"
    table_path = output / "e116-table.md"
    write_json(report_path, report)
    table_path.write_text(render_table(report), encoding="utf-8")
    return {
        "status": report["status"],
        "observed_rows": len(rows),
        "expected_rows": len(expected),
        "e115_validation_checkpoints": report["e115_validation_checkpoints"],
        "report": str(report_path),
        "table": str(table_path),
        "training_runs": 0,
        "episode_runs": 0,
    }


def _fmt(value, digits=2) -> str:
    return "—" if value is None else f"{float(value):.{digits}f}"


def render_table(report: dict) -> str:
    lines = [
        "# E116 matched E115 operator-cost comparison",
        "",
        "Five reused E115 development instances. GA is averaged over four "
        "replicates within each instance. Lower is better except completion and "
        "within ±20. This is development evidence, not an independent test.",
        "",
        "## Baselines",
        "",
        "| Method | Strict rows | Complete instances | J | T total | E total | Work/delivery | Rehandles/100 | Release MAE | Within ±20 | Outside wait | Storage MAE | Steps/delivery |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in report["baseline_summaries"]:
        metrics = item["metrics"] or {}
        lines.append(
            f"| {item['method']} | {item['strict_complete_rows']}/{item['expected_rows']} | "
            f"{item['fully_complete_instances']}/5 | "
            f"{_fmt(metrics.get('objective_cost'))} | "
            f"{_fmt(metrics.get('components.release_tardiness'))} | "
            f"{_fmt(metrics.get('components.release_earliness'))} | "
            f"{_fmt(metrics.get('equipment_work_per_delivery'))} | "
            f"{_fmt(metrics.get('rehandles_per_100'))} | "
            f"{_fmt(metrics.get('release_clock_mae'))} | "
            f"{_fmt(metrics.get('within_tolerance_rate'), 3)} | "
            f"{_fmt(metrics.get('mean_wait_before_storage'))} | "
            f"{_fmt(metrics.get('storage_clock_mae'))} | "
            f"{_fmt(metrics.get('steps_per_delivery'))} |"
        )
    lines.extend([
        "",
        "## E115 trajectory on the same five instances",
        "",
        "| Episode | Strict | J | T total | E total | Work/delivery | Rehandles/100 | Release MAE | Within ±20 | Outside wait | Storage MAE | Steps/delivery |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for comparison in report["checkpoint_comparisons"]:
        item = comparison["e115"]
        metrics = item["metrics"] or {}
        lines.append(
            f"| {item['checkpoint_episode']} | {item['strict_complete_instances']}/5 | "
            f"{_fmt(metrics.get('objective_cost'))} | "
            f"{_fmt(metrics.get('components.release_tardiness'))} | "
            f"{_fmt(metrics.get('components.release_earliness'))} | "
            f"{_fmt(metrics.get('equipment_work_per_delivery'))} | "
            f"{_fmt(metrics.get('rehandles_per_100'))} | "
            f"{_fmt(metrics.get('release_clock_mae'))} | "
            f"{_fmt(metrics.get('within_tolerance_rate'), 3)} | "
            f"{_fmt(metrics.get('mean_wait_before_storage'))} | "
            f"{_fmt(metrics.get('storage_clock_mae'))} | "
            f"{_fmt(metrics.get('steps_per_delivery'))} |"
        )
    lines.extend([
        "",
        "`J = 2T + E + 0.1L + 10R`, using release-fixed due dates and a ±20-step tolerance. Completion remains a hard requirement. The JSON report contains per-instance points and E115-minus-baseline paired differences for every validation checkpoint.",
        "",
    ])
    return "\n".join(lines)


def status(output: Path) -> dict:
    prepare(output)
    completed = {
        method: sum(
            row_path(output, method, seed, replicate).is_file()
            for seed in p.INSTANCE_SEEDS
            for replicate in replicates(method)
        )
        for method in p.METHODS
    }
    expected = {
        method: len(p.INSTANCE_SEEDS) * len(replicates(method))
        for method in p.METHODS
    }
    return {
        "status": "complete" if completed == expected else "partial",
        "completed_rows": completed,
        "expected_rows": expected,
        "training_runs": 0,
        "episode_runs": 0,
    }


def self_check(output: Path) -> dict:
    prepared = prepare(output)
    require(len(p.INSTANCE_SEEDS) == 5, "E116 requires five E115 seeds")
    for seed in p.INSTANCE_SEEDS:
        instance = make_instance(seed)
        require(
            tuple(instance.arrival_steps) == tuple(30 * i for i in range(36)),
            "E116 arrival schedule changed",
        )
    probe = objective.objective_cost((3.0, 4.0, 5.0, 2.0))
    require(math.isclose(probe, 30.5, abs_tol=1e-12), "E116 J formula changed")
    return {
        "status": "passed",
        "instances": len(p.INSTANCE_SEEDS),
        "ga_replicates": len(p.GA_REPLICATES),
        "contract_sha256": prepared["contract_sha256"],
        "training_runs": 0,
        "episode_runs": 0,
    }


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("prepare", "self-check", "run-one", "run-method", "run-all", "analyze", "status"),
    )
    parser.add_argument("--output", type=Path, default=p.DEFAULT_OUTPUT)
    parser.add_argument("--method", choices=p.METHODS)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--replicate", type=int, default=0)
    args = parser.parse_args(argv)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    output = args.output.resolve()
    if args.command == "prepare":
        result = prepare(output)
    elif args.command == "self-check":
        result = self_check(output)
    elif args.command == "run-one":
        if args.method is None or args.seed is None:
            parser.error("run-one requires --method and --seed")
        result = run_one(output, args.method, args.seed, args.replicate)
    elif args.command == "run-method":
        if args.method is None:
            parser.error("run-method requires --method")
        result = run_method(output, args.method)
    elif args.command == "run-all":
        episode_runs = 0
        for method in p.METHODS:
            value = run_method(output, method)
            episode_runs += value["episode_runs"]
        result = {
            **analyze(output),
            "episode_runs": episode_runs,
        }
    elif args.command == "analyze":
        result = analyze(output)
    else:
        result = status(output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

