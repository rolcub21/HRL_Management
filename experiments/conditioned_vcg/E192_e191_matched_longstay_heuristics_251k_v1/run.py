#!/usr/bin/env python3
"""Run matched Dynamic PSLAP and GA on every E191 development day."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
import os
from pathlib import Path
from statistics import fmean, stdev
from types import SimpleNamespace
from unittest.mock import patch

import torch
from scipy.stats import t as student_t

from example.episode_instance import EpisodeInstance
from experiments.conditioned_vcg.E28_empty_start_scale_baselines_100k import (
    adapter as empty_adapter,
)
from experiments.conditioned_vcg.E91_matched_target_size_benchmark_156k_v1 import (
    run as e91,
)
from experiments.conditioned_vcg.E187_wait_corrected_e78_246k_v1 import objective
from experiments.conditioned_vcg.E190_e189_matched_heuristic_curve_249k_v1 import (
    run as e190,
)
from experiments.conditioned_vcg.E191_fresh_long_stay_crowding_250k_v1 import (
    protocol as e191,
    scenario as workload,
)

from . import protocol as p


class E192Error(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise E192Error(message)


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def read_json(path: Path) -> dict:
    require(path.is_file(), f"missing E192 artifact: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"invalid E192 JSON object: {path}")
    return value


def _mean(values) -> float | None:
    values = tuple(float(value) for value in values if value is not None)
    return float(fmean(values)) if values else None


def make_instance(seed: int) -> EpisodeInstance:
    require(seed in p.INSTANCE_SEEDS, "seed is outside the E192 development panel")
    env = e191.TargetDenseEnv(e191.SCENARIO, e191.SPEC)
    instance = env.sample_episode_instance(seed=int(seed))
    instance.validate_for(env)
    return instance


def instance_manifest(contract: dict) -> dict:
    records = []
    for seed in p.INSTANCE_SEEDS:
        instance = make_instance(seed)
        meta = workload.workload_metadata(instance)
        records.append({
            "seed": int(seed),
            "instance_id": instance.instance_id,
            "schedule_id": instance.schedule_id,
            "target_label": meta["target_label"],
            "target_nominal_occupancy": meta["target_nominal_occupancy"],
            "arrival_gap_steps": meta["arrival_gap_steps"],
            "instance_json": instance.to_json(),
        })
    payload = {
        "protocol": p.PROTOCOL,
        "contract_sha256": contract["contract_sha256"],
        "records": records,
    }
    return {**payload, "manifest_sha256": p.digest(payload)}


def _source_summary() -> dict:
    summary = read_json(p.E191_SUMMARY)
    source_contract = e191.contract()
    require(
        summary.get("training_protocol") == e191.PROTOCOL
        and summary.get("target_contract_sha256")
        == source_contract["contract_sha256"]
        and summary.get("scenario") == e191.SCENARIO
        and int(summary.get("model_seed", -1)) == p.MODEL_SEED,
        "E191 source summary identity differs",
    )
    require(
        tuple(summary.get("validation_instance_seeds", ())) == p.INSTANCE_SEEDS,
        "E191 validation seeds differ from E192",
    )
    return summary


def _assert_validation_instance_parity(manifest: dict) -> None:
    expected = {
        int(record["seed"]): str(record["instance_id"])
        for record in manifest["records"]
    }
    for validation in _source_summary().get("validation_history", ()):
        runs = tuple(validation.get("runs", ()))
        require(len(runs) == len(p.INSTANCE_SEEDS), "E191 validation panel is incomplete")
        require(
            Counter(int(run["instance_seed"]) for run in runs)
            == Counter(p.INSTANCE_SEEDS),
            "E191 validation panel has missing or duplicate seeds",
        )
        for run in runs:
            seed = int(run["instance_seed"])
            require(
                str(run["episode_instance_id"]) == expected[seed],
                "E191 validation EpisodeInstance differs from E192",
            )


def prepare(output: Path) -> dict:
    output = Path(output).resolve()
    expected_contract = p.contract()
    expected_manifest = instance_manifest(expected_contract)
    contract_path = output / "contract.json"
    manifest_path = output / "instances.json"
    if contract_path.is_file():
        require(read_json(contract_path) == expected_contract, "E192 contract changed")
        require(read_json(manifest_path) == expected_manifest, "E192 manifest changed")
    else:
        require(
            not output.exists() or not any(output.iterdir()),
            "E192 refuses a nonempty uncontracted output",
        )
        write_json(contract_path, expected_contract)
        write_json(manifest_path, expected_manifest)
    _assert_validation_instance_parity(expected_manifest)
    return {
        "status": "prepared",
        "instances": len(p.INSTANCE_SEEDS),
        "workload_instances": {
            label: sum(row["target_label"] == label for row in expected_manifest["records"])
            for label in ("20%", "50%", "80%")
        },
        "dynamic_pslap_episodes": len(p.INSTANCE_SEEDS),
        "ga_episodes": len(p.INSTANCE_SEEDS) * len(p.GA_REPLICATES),
        "expected_rows": p.EXPECTED_ROWS,
        "training_runs": 0,
        "test_panel_opened": False,
        "contract_sha256": expected_contract["contract_sha256"],
    }


def authenticate(output: Path) -> tuple[dict, dict]:
    prepared = prepare(output)
    contract = read_json(Path(output) / "contract.json")
    manifest = read_json(Path(output) / "instances.json")
    require(
        prepared["contract_sha256"] == contract["contract_sha256"],
        "E192 authentication failed",
    )
    return contract, manifest


def frozen_instance(manifest: dict, seed: int) -> EpisodeInstance:
    matches = [row for row in manifest["records"] if int(row["seed"]) == int(seed)]
    require(len(matches) == 1, "E192 instance is not uniquely frozen")
    row = matches[0]
    instance = EpisodeInstance.from_json(row["instance_json"])
    require(
        instance.instance_id == row["instance_id"]
        and instance.schedule_id == row["schedule_id"],
        "E192 frozen instance identity changed",
    )
    validation_env = e191.TargetDenseEnv(e191.SCENARIO, e191.SPEC)
    # E191's live generator updates these instance-specific fields before
    # validation.  A deserialized frozen instance must do the same explicitly.
    validation_env.arrival_rate = float(instance.arrival_rate)
    validation_env.proc_mean = float(instance.proc_mean)
    instance.validate_for(validation_env)
    return instance


def replicates(method: str) -> tuple[int, ...]:
    return p.replicates(method)


def row_path(output: Path, method: str, seed: int, replicate: int) -> Path:
    return (
        Path(output) / "rows" / method
        / f"instance-{int(seed)}-replicate-{int(replicate)}.json"
    )


class TracedLongStayEnv(workload.LongStayCrowdingEnv):
    """Run unchanged heuristics in the exact E191 physics and objective."""

    created: list["TracedLongStayEnv"] = []

    def __init__(self, scenario):
        require(
            scenario.public_dict() == p.SCENARIO_SPEC.public_dict(),
            "baseline requested another scenario",
        )
        super().__init__(e191.SCENARIO, e191.SPEC)
        self.e192_scenario = scenario
        self.warm_start_audit = None
        self.e192_action_counts = Counter()
        self.e192_rehandle_events = 0
        self.e192_requested_defaults = None
        self.occupancy_trace: list[dict] = []
        self.__class__.created.append(self)

    def reset(self, instance=None):
        require(instance is not None, "E192 requires a frozen EpisodeInstance")
        # The three workload levels deliberately have different arrival rates.
        # Bind the environment to the authenticated instance before the native
        # compatibility check performed by reset.
        self.arrival_rate = float(instance.arrival_rate)
        self.proc_mean = float(instance.proc_mean)
        state = super().reset(instance=instance)
        self.e192_action_counts = Counter()
        self.e192_rehandle_events = 0
        self.occupancy_trace = [e190._measurement(self)]
        require(
            self.time_steps == 0
            and int(self.blocks[0].arrival_step) == 0
            and self.blocks[0].position == self.pickup_cell
            and not any(
                block.stored
                or block.delivered
                or block.carrying
                or block.stored_time_step is not None
                for block in self.blocks
            )
            and not any(
                block.position is not None or int(block.arrival_step) <= 0
                for block in self.blocks[1:]
            ),
            "E192 baseline reset was not an empty yard",
        )
        self.warm_start_audit = {
            "passed": True,
            "episode_instance_id": instance.instance_id,
            "time_step": 0,
            "initially_stored_jobs": 0,
            "occupied_storage_slots": 0,
            "future_arrivals": len(self.blocks),
            "target_label": self.e191_workload["target_label"],
        }
        return state

    def step(self, action):
        action_name = str(self.ACTION_NAMES[int(action)])
        result = super().step(action)
        self.e192_action_counts[action_name] += 1
        self.e192_rehandle_events += int(bool(result[3].get("relocated_block")))
        self.occupancy_trace.append(e190._measurement(self))
        return result

    def occupancy_summary(self) -> dict:
        require(bool(self.occupancy_trace), "E192 occupancy trace is empty")
        samples = self.occupancy_trace[:-1] or self.occupancy_trace
        capacity = len(self.storage_positions)
        return {
            "usable_storage_capacity": capacity,
            "initial_occupied_storage_slots": self.occupancy_trace[0]["occupied_storage_slots"],
            "initial_storage_occupancy_ratio": self.occupancy_trace[0]["storage_occupancy_ratio"],
            "time_weighted_mean_occupied_storage_slots": fmean(
                row["occupied_storage_slots"] for row in samples
            ),
            "time_weighted_mean_storage_occupancy_ratio": fmean(
                row["storage_occupancy_ratio"] for row in samples
            ),
            "peak_occupied_storage_slots": max(
                row["occupied_storage_slots"] for row in self.occupancy_trace
            ),
            "peak_storage_occupancy_ratio": max(
                row["storage_occupancy_ratio"] for row in self.occupancy_trace
            ),
            "time_weighted_mean_admitted_workload": fmean(
                row["concurrent_admitted_workload"] for row in samples
            ),
            "peak_concurrent_admitted_workload": max(
                row["concurrent_admitted_workload"] for row in self.occupancy_trace
            ),
            "peak_physical_blocks_present": max(
                row["physical_blocks_present"] for row in self.occupancy_trace
            ),
            "peak_arrived_unadmitted": max(
                row["arrived_unadmitted"] for row in self.occupancy_trace
            ),
            "primitive_step_intervals": len(samples),
        }


def _environment_factory(
    scenario, *, arrival_rate, proc_mean, grid_rows, grid_cols,
    exit_width, number_blocks,
):
    expected = p.SCENARIO_SPEC
    require(
        scenario.public_dict() == expected.public_dict()
        and int(grid_rows) == expected.rows
        and int(grid_cols) == expected.cols
        and int(number_blocks) == expected.total_jobs
        and int(exit_width) == 3,
        "baseline environment request differs from E191 geometry",
    )
    env = TracedLongStayEnv(scenario)
    env.e192_requested_defaults = {
        "arrival_rate": float(arrival_rate),
        "proc_mean": float(proc_mean),
    }
    return env


def score_completed_blocks(blocks, equipment_work: float, rehandles: float) -> dict:
    return e190.score_completed_blocks(blocks, equipment_work, rehandles)


def _completed_metrics(env: TracedLongStayEnv, strict: bool) -> dict:
    metrics = dict(env.e187_metrics())
    accrued = float(metrics["J"])
    comparable = bool(
        strict and metrics["strict_complete"]
        and metrics["stored_jobs"] == metrics["required_jobs"]
        and metrics["delivered_jobs"] == metrics["required_jobs"]
    )
    metrics["accrued_J_at_stop"] = accrued
    metrics["cost_ranking_eligible"] = comparable
    if not comparable:
        metrics["J"] = None
    action_counts = dict(sorted(env.e192_action_counts.items()))
    wait_steps = int(action_counts.get("WAIT", 0))
    equipment = int(sum(action_counts.values()) - wait_steps)
    rehandles = int(env.e192_rehandle_events)
    if comparable:
        require(metrics["objective_accounting_exact"], "baseline J did not telescope")
        require(metrics["reward_sum_equals_negative_J"], "baseline return differs from J")
        require(
            math.isclose(float(metrics["components"]["equipment_work"]), equipment),
            "baseline equipment trace differs from E191 objective",
        )
        require(
            math.isclose(float(metrics["components"]["physical_rehandles"]), rehandles),
            "baseline rehandle trace differs from E191 objective",
        )
    metrics.update({
        "steps": int(env.time_steps),
        "steps_per_delivery": (
            int(env.time_steps) / metrics["delivered_jobs"]
            if metrics["delivered_jobs"] else None
        ),
        "physical_rehandles": rehandles,
        "rehandles_per_100": (
            100.0 * rehandles / metrics["required_jobs"] if comparable else None
        ),
        "primitive_action_counts": action_counts,
        "wait_primitive_steps": wait_steps,
    })
    return metrics


def run_one(output: Path, method: str, seed: int, replicate: int) -> dict:
    contract, manifest = authenticate(output)
    require(seed in p.INSTANCE_SEEDS, "seed is outside E192")
    require(replicate in replicates(method), "replicate is outside E192")
    path = row_path(output, method, seed, replicate)
    identity = {
        "protocol": p.PROTOCOL,
        "contract_sha256": contract["contract_sha256"],
        "method": method,
        "instance_seed": int(seed),
        "replicate": int(replicate),
    }
    if path.is_file():
        row = read_json(path)
        require(
            all(row.get(key) == value for key, value in identity.items()),
            f"saved E192 coordinate differs: {path}",
        )
        return {**row, "reused": True, "episode_runs": 0}

    instance = frozen_instance(manifest, seed)
    meta = workload.workload_metadata(instance)
    TracedLongStayEnv.created = []
    print(
        f"E192 start method={method} target={meta['target_label']} "
        f"seed={seed} replicate={replicate}",
        flush=True,
    )

    def instance_bound_factory(*args, **kwargs):
        env = _environment_factory(*args, **kwargs)
        # The inherited evaluator authenticates the EpisodeInstance before it
        # calls reset, so bind the workload-specific generator parameters at
        # construction time as well as in reset.
        env.arrival_rate = float(instance.arrival_rate)
        env.proc_mean = float(instance.proc_mean)
        return env

    with (
        patch.object(e91.p, "SCENARIO", p.SCENARIO_SPEC),
        patch.object(
            empty_adapter, "empty_start_factory", side_effect=instance_bound_factory
        ),
    ):
        baseline = e91._baseline_row(
            method, instance, int(seed), int(replicate), p.INSTANCE_SEEDS.index(seed)
        )
    require(
        len(TracedLongStayEnv.created) == 1,
        "baseline did not create exactly one E191 objective environment",
    )
    env = TracedLongStayEnv.created[0]
    require(
        env.current_episode_instance.instance_id == instance.instance_id,
        "baseline executed another EpisodeInstance",
    )
    metrics = _completed_metrics(env, bool(baseline["strict_complete"]))
    require(
        bool(metrics["strict_complete"]) == bool(baseline["strict_complete"]),
        "baseline and E191 completion flags differ",
    )
    if baseline["rehandles_at_stop"] is not None:
        require(
            int(baseline["rehandles_at_stop"]) == metrics["physical_rehandles"],
            "baseline rehandle report differs from primitive trace",
        )
    row = {
        **identity,
        "instance_id": instance.instance_id,
        "schedule_id": instance.schedule_id,
        "target_label": meta["target_label"],
        "target_nominal_occupancy": meta["target_nominal_occupancy"],
        "arrival_gap_steps": meta["arrival_gap_steps"],
        "status": "complete",
        "rng_seed": baseline["rng_seed"],
        "failure_class": baseline["failure_class"],
        "method_failure_reason": baseline["method_failure_reason"],
        "decision_count": baseline["decision_count"],
        "episode_wall_seconds": baseline["episode_wall_seconds"],
        "wall_seconds_per_decision": baseline["amortized_wall_seconds_per_decision"],
        "baseline_reported_mae": baseline["baseline_reported_mae"],
        "requested_parser_defaults": env.e192_requested_defaults,
        "metrics": metrics,
        "training_runs": 0,
        "episode_runs": 1,
        "test_panel_opened": False,
    }
    write_json(path, row)
    print(
        f"E192 done method={method} target={meta['target_label']} seed={seed} "
        f"replicate={replicate} strict={int(metrics['strict_complete'])} "
        f"J={metrics['J']} peak={metrics['peak_stored']} "
        f"meanStored={metrics['mean_stored']:.2f} "
        f"util={metrics['transporter_utilization']:.3f} "
        f"wall={baseline['episode_wall_seconds']:.2f}s",
        flush=True,
    )
    return row


FLAT_METRICS = e190.FLAT_METRICS + (
    "mean_stored", "mean_stored_fraction", "arrival_window_mean_stored",
    "nominal_steady_window_mean_stored", "mean_external_queue",
    "peak_external_queue", "transporter_utilization",
)


def _flat_metrics(metrics: dict, *, steps: int | None = None) -> dict:
    result = {
        name: metrics.get(name)
        for name in FLAT_METRICS if not name.startswith("component_")
    }
    if steps is not None:
        result["steps"] = int(steps)
    for name in objective.COMPONENT_NAMES:
        result[f"component_{name}"] = metrics.get("components", {}).get(name)
    required = int(metrics.get("required_jobs", 0))
    rehandles = result.get("physical_rehandles")
    if rehandles is None:
        rehandles = metrics.get("components", {}).get("physical_rehandles")
        result["physical_rehandles"] = rehandles
    if result.get("rehandles_per_100") is None and required and rehandles is not None:
        result["rehandles_per_100"] = 100.0 * float(rehandles) / required
    delivered = int(metrics.get("delivered_jobs", 0))
    if result.get("steps_per_delivery") is None and delivered and result.get("steps") is not None:
        result["steps_per_delivery"] = float(result["steps"]) / delivered
    return result


def _average_metrics(metrics: list[dict]) -> dict:
    require(bool(metrics), "cannot average an empty metric group")
    result = {}
    for name in FLAT_METRICS:
        values = [row.get(name) for row in metrics]
        result[name] = _mean(values) if all(value is not None for value in values) else None
    return result


def _saved_rows(output: Path) -> list[dict]:
    contract = p.contract()
    rows = []
    for method in p.METHODS:
        for seed in p.INSTANCE_SEEDS:
            for replicate in replicates(method):
                path = row_path(output, method, seed, replicate)
                if not path.is_file():
                    continue
                row = read_json(path)
                require(
                    row.get("protocol") == p.PROTOCOL
                    and row.get("contract_sha256") == contract["contract_sha256"]
                    and row.get("method") == method
                    and int(row.get("instance_seed", -1)) == seed
                    and int(row.get("replicate", -1)) == replicate,
                    f"saved E192 row identity differs: {path}",
                )
                rows.append(row)
    return rows


def baseline_points(rows: list[dict]) -> dict[str, list[dict]]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["method"], int(row["instance_seed"]))].append(row)
    result = {method: [] for method in p.METHODS}
    for method in p.METHODS:
        expected_replicates = set(replicates(method))
        for seed in p.INSTANCE_SEEDS:
            group = grouped[(method, seed)]
            observed = {int(row["replicate"]) for row in group}
            complete_rows = [
                row for row in group
                if bool(row["metrics"].get(
                    "cost_ranking_eligible", row["metrics"].get("strict_complete")
                ))
            ]
            fully_observed = observed == expected_replicates
            strict = fully_observed and len(complete_rows) == len(expected_replicates)
            metrics = (
                _average_metrics([_flat_metrics(row["metrics"]) for row in complete_rows])
                if strict else None
            )
            result[method].append({
                "method": method,
                "instance_seed": seed,
                "target_label": workload._level_key(workload.target_level(seed)),
                "observed_replicates": sorted(observed),
                "required_replicates": sorted(expected_replicates),
                "replicate_count": len(group),
                "strict_complete": bool(strict),
                "metrics": metrics,
            })
    return result


def validation_checkpoints() -> list[dict]:
    summary = _source_summary()
    manifest = instance_manifest(p.contract())
    identities = {
        int(row["seed"]): str(row["instance_id"]) for row in manifest["records"]
    }
    checkpoints = []
    seen = set()
    for validation in summary.get("validation_history", ()):
        episode = int(validation["checkpoint_episode"])
        require(episode not in seen, "duplicate E191 validation checkpoint")
        seen.add(episode)
        runs = tuple(validation.get("runs", ()))
        require(
            Counter(int(run["instance_seed"]) for run in runs)
            == Counter(p.INSTANCE_SEEDS),
            "E191 checkpoint does not contain the exact nine-instance panel",
        )
        points = []
        for run in sorted(runs, key=lambda item: int(item["instance_seed"])):
            seed = int(run["instance_seed"])
            require(
                run.get("protocol") == e191.PROTOCOL
                and run.get("split") == "development_validation"
                and float(run.get("scheduled_epsilon_after", -1.0)) == 0.0
                and str(run.get("episode_instance_id")) == identities[seed],
                "E191 checkpoint is not a frozen greedy matched validation",
            )
            metrics = run.get("e187_metrics", {})
            strict = bool(run.get("strict_method_success") and metrics.get("strict_complete"))
            points.append({
                "method": "e191",
                "checkpoint_episode": episode,
                "instance_seed": seed,
                "target_label": workload._level_key(workload.target_level(seed)),
                "strict_complete": strict,
                "metrics": _flat_metrics(metrics, steps=int(run["steps"])) if strict else None,
            })
        checkpoints.append({
            "checkpoint_episode": episode,
            "runs": list(runs),
            "points": points,
            "source_selection_score": validation.get("selection_score"),
        })
    return sorted(checkpoints, key=lambda row: row["checkpoint_episode"])


def _panel_summary(points: list[dict]) -> dict:
    complete = [point for point in points if point["strict_complete"]]
    metrics = (
        _average_metrics([point["metrics"] for point in complete])
        if len(complete) == len(points) and points else None
    )
    return {
        "strict_complete_instances": len(complete),
        "required_instances": len(points),
        "all_instances_complete": bool(points) and len(complete) == len(points),
        "metrics": metrics,
    }


def _by_workload(points: list[dict]) -> dict:
    result = {}
    for label in ("20%", "50%", "80%"):
        selected = [point for point in points if point["target_label"] == label]
        require(len(selected) == workload.VALIDATION_CASES_PER_LEVEL,
                f"E192 workload {label} is incomplete")
        result[label] = _panel_summary(selected)
    return result


def _paired_delta(values: list[float]) -> dict:
    mean = float(fmean(values))
    if len(values) < 2:
        return {"n": len(values), "mean": mean, "ci95": None}
    standard_error = stdev(values) / math.sqrt(len(values))
    critical = float(student_t.ppf(0.975, df=len(values) - 1))
    return {
        "n": len(values),
        "mean": mean,
        "ci95": [mean - critical * standard_error, mean + critical * standard_error],
    }


def _comparison(checkpoint: dict, method: str, baseline: list[dict]) -> dict:
    vcg = {point["instance_seed"]: point for point in checkpoint["points"]}
    control = {point["instance_seed"]: point for point in baseline}
    require(set(vcg) == set(control) == set(p.INSTANCE_SEEDS), "paired panel differs")
    pairs = []
    for seed in p.INSTANCE_SEEDS:
        left, right = vcg[seed], control[seed]
        eligible = bool(left["strict_complete"] and right["strict_complete"])
        delta = None
        if eligible:
            delta = {
                name: float(left["metrics"][name]) - float(right["metrics"][name])
                for name in FLAT_METRICS
                if left["metrics"].get(name) is not None
                and right["metrics"].get(name) is not None
            }
        pairs.append({
            "instance_seed": seed,
            "target_label": workload._level_key(workload.target_level(seed)),
            "e191_complete": bool(left["strict_complete"]),
            "baseline_complete": bool(right["strict_complete"]),
            "cost_pair_eligible": eligible,
            "e191_metrics": left["metrics"],
            "baseline_metrics": right["metrics"],
            "delta_e191_minus_baseline": delta,
        })
    complete_pairs = [pair for pair in pairs if pair["cost_pair_eligible"]]

    def summaries(selected):
        deltas = {}
        if selected:
            for name in FLAT_METRICS:
                values = [pair["delta_e191_minus_baseline"].get(name) for pair in selected]
                if all(value is not None for value in values):
                    deltas[name] = _paired_delta([float(value) for value in values])
        return deltas

    return {
        "checkpoint_episode": checkpoint["checkpoint_episode"],
        "baseline_method": method,
        "difference_direction": "E191 minus baseline",
        "paired_instance_count": len(complete_pairs),
        "pairs": pairs,
        "mean_paired_deltas": (
            summaries(complete_pairs)
            if len(complete_pairs) == len(p.INSTANCE_SEEDS) else {}
        ),
        "mean_paired_deltas_by_workload": {
            label: summaries([
                pair for pair in complete_pairs if pair["target_label"] == label
            ])
            for label in ("20%", "50%", "80%")
        },
    }


def _format(value) -> str:
    return "—" if value is None else f"{float(value):.3f}"


def analyze(output: Path) -> dict:
    output = Path(output).resolve()
    prepare(output)
    rows = _saved_rows(output)
    points = baseline_points(rows)
    baseline_summaries = {
        method: {
            "overall": _panel_summary(method_points),
            "by_workload": _by_workload(method_points),
        }
        for method, method_points in points.items()
    }
    checkpoints = validation_checkpoints()
    checkpoint_summaries = {
        str(item["checkpoint_episode"]): {
            "overall": _panel_summary(item["points"]),
            "by_workload": _by_workload(item["points"]),
        }
        for item in checkpoints
    }
    comparisons = [
        _comparison(checkpoint, method, points[method])
        for checkpoint in checkpoints for method in p.METHODS
    ]
    report = {
        "protocol": p.PROTOCOL,
        "contract_sha256": p.contract()["contract_sha256"],
        "status": "complete" if len(rows) == p.EXPECTED_ROWS else "partial",
        "expected_rows": p.EXPECTED_ROWS,
        "observed_rows": len(rows),
        "strict_complete_rows": sum(
            bool(row["metrics"].get("cost_ranking_eligible")) for row in rows
        ),
        "completion_reported_before_quality": True,
        "baseline_summaries": baseline_summaries,
        "baseline_instance_points": points,
        "e191_checkpoint_summaries": checkpoint_summaries,
        "comparisons": comparisons,
        "available_e191_checkpoints": [item["checkpoint_episode"] for item in checkpoints],
        "ga_aggregation_unit": "four replicates within instance, then nine instances",
        "paired_inference_unit": "nine EpisodeInstances overall; three per workload",
        "training_runs": 0,
        "episode_runs": 0,
        "development_only": True,
        "test_panel_opened": False,
        "paper_evidence": False,
    }
    write_json(output / p.REPORT_NAME, report)

    lines = [
        "# E192 — E191 learning curve versus matched long-stay heuristics", "",
        "Completion is reported before cost. GA replicates are averaged within each instance.", "",
        "| Method/checkpoint | Workload | Complete | J | Storage MAE | Admission wait | Mean stored | Steady stored | Peak | Utilization | Rehandles/100 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    def add_rows(label, summary):
        groups = [("all", summary["overall"])] + list(summary["by_workload"].items())
        for workload_label, item in groups:
            metrics = item["metrics"] or {}
            lines.append(
                f"| {label} | {workload_label} | "
                f"{item['strict_complete_instances']}/{item['required_instances']} | "
                + " | ".join(_format(metrics.get(name)) for name in (
                    "J", "storage_clock_mae", "mean_admission_wait",
                    "mean_stored", "nominal_steady_window_mean_stored",
                    "peak_stored", "transporter_utilization", "rehandles_per_100",
                )) + " |"
            )

    for method in p.METHODS:
        add_rows(method, baseline_summaries[method])
    for checkpoint in checkpoints:
        add_rows(
            f"E191 ep{checkpoint['checkpoint_episode']}",
            checkpoint_summaries[str(checkpoint["checkpoint_episode"])],
        )
    lines += [
        "", "J = 2T_storage + E_storage + 0.5W + 0.1L + 10R.",
        "All rows use the exact nine E191 development EpisodeInstances; this is not the sealed test panel.",
        "Negative paired cost deltas (E191 minus baseline) favor E191.",
        "Certified Dynamic is intentionally excluded and remains a separate slower arm.",
    ]
    (output / p.TABLE_NAME).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "status": report["status"],
        "observed_rows": len(rows),
        "expected_rows": p.EXPECTED_ROWS,
        "strict_complete_rows": report["strict_complete_rows"],
        "available_e191_checkpoints": report["available_e191_checkpoints"],
        "report": str(output / p.REPORT_NAME),
        "table": str(output / p.TABLE_NAME),
        "training_runs": 0,
        "episode_runs": 0,
    }


def self_check(output: Path) -> dict:
    prepared = prepare(output)
    probe = SimpleNamespace(
        arrival_step=0, stored_time_step=30,
        storage_steps_needed=700, delivered_time_step=730,
    )
    scored = score_completed_blocks((probe,), equipment_work=4, rehandles=1)
    require(math.isclose(scored["J"], 25.4), "storage-clock/W discriminator failed")
    require(
        scored["storage_clock_mae"] == 0.0
        and scored["release_clock_mae"] == 30.0
        and scored["components"]["admission_wait_accrual"] == 30.0,
        "E192 scorer substituted release-clock timing",
    )
    return {
        **prepared,
        "status": "self_check_passed",
        "storage_clock_wait_discriminator": scored,
        "episode_runs": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("prepare", "self-check", "run-one", "run-method", "run-all",
                 "analyze", "status"),
    )
    parser.add_argument("--output", type=Path, default=p.DEFAULT_OUTPUT)
    parser.add_argument("--method", choices=p.METHODS)
    parser.add_argument("--seed", type=int, choices=p.INSTANCE_SEEDS)
    parser.add_argument("--replicate", type=int, default=0)
    parser.add_argument("--device", choices=("cpu",), default="cpu")
    args = parser.parse_args()
    torch.set_num_threads(1)
    output = args.output.resolve()
    if args.command == "prepare":
        result = prepare(output)
    elif args.command == "self-check":
        result = self_check(output)
    elif args.command in ("analyze", "status"):
        result = analyze(output)
    else:
        prepare(output)
        if args.command == "run-one":
            require(args.method is not None and args.seed is not None,
                    "run-one requires --method and --seed")
            coordinates = ((args.method, args.seed, args.replicate),)
        elif args.command == "run-method":
            require(args.method is not None, "run-method requires --method")
            coordinates = tuple(
                (args.method, seed, replicate)
                for seed in p.INSTANCE_SEEDS for replicate in replicates(args.method)
            )
        else:
            coordinates = tuple(
                (method, seed, replicate)
                for method in p.METHODS for seed in p.INSTANCE_SEEDS
                for replicate in replicates(method)
            )
        rows = [run_one(output, *coordinate) for coordinate in coordinates]
        result = analyze(output)
        result["episode_runs"] = sum(not row.get("reused", False) for row in rows)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
