#!/usr/bin/env python3
"""Run and analyze frozen, matched 8x8 VCG/GA/Dynamic-PSLAP episodes."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
import math
from pathlib import Path
from statistics import mean, stdev
from time import perf_counter
from unittest.mock import patch

import torch
from scipy.stats import t as student_t

import benchmark_viability_critic_priority as benchmark
from example.episode_instance import EpisodeInstance
from example.helper.timing_metrics import summarize_block_storage_flow
from experiments.conditioned_vcg.E13_operational_scalability_95k import program as e13
from experiments.conditioned_vcg.E14_certification_scalability_95k import reuse
from experiments.conditioned_vcg.E28_empty_start_scale_baselines_100k import adapter, run as e28
from experiments.conditioned_vcg.E82_original_base80_stage_b_profile_147k_v1 import run as e82
from PSLAP.relocation_family_certification import RELOCATION_FAMILY_CERTIFICATION
from train_vcg_preference_conditioned import _atomic_json_save

from . import protocol as p


class E91Error(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise E91Error(message)


def read_json(path: Path) -> dict:
    require(path.is_file(), f"missing E91 file: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"invalid E91 JSON: {path}")
    return value


def manifest(contract: dict) -> dict:
    records = []
    for panel, seeds in p.PANEL_SEEDS.items():
        for seed in seeds:
            instance = e28._instance(p.SCENARIO, seed)
            records.append({
                "panel": panel, "seed": seed,
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


def authenticate(output: Path) -> tuple[dict, dict]:
    observed = read_json(output / "contract.json")
    expected = p.contract()
    require(observed == expected, "E91 contract, checkpoints, or source changed")
    saved_manifest = read_json(output / "instances.json")
    require(saved_manifest == manifest(observed), "E91 instance manifest changed")
    return observed, saved_manifest


def prepare(output: Path) -> dict:
    expected = p.contract()
    if (output / "contract.json").is_file():
        authenticate(output)
    else:
        require(not output.exists() or not any(output.iterdir()),
                "E91 refuses a nonempty, uncontracted output")
        _atomic_json_save(expected, output / "contract.json")
        _atomic_json_save(manifest(expected), output / "instances.json")
    return {
        "status": "prepared", "test_instances": 30,
        "test_vcg_episodes": 30 * len(p.MODEL_SEEDS) * len(p.LAMBDAS),
        "test_baseline_episodes": 30 * (1 + len(p.GA_REPLICATES)),
        "contract_sha256": expected["contract_sha256"],
        "training_runs": 0, "episode_runs": 0,
    }


def coordinates(panel: str, method: str):
    require(panel in p.PANEL_SEEDS and method in p.METHODS, "unknown E91 coordinate")
    for seed in p.PANEL_SEEDS[panel]:
        if method == "vcg":
            model_seeds = p.MODEL_SEEDS if panel == "test" else (0,)
            for model_seed in model_seeds:
                for value in p.LAMBDAS:
                    yield (seed, model_seed, value, None)
        elif method == "ga":
            replicates = p.GA_REPLICATES if panel == "test" else (0,)
            for replicate in replicates:
                yield (seed, None, None, replicate)
        else:
            yield (seed, None, None, 0)


def row_path(output: Path, panel: str, method: str, seed: int,
             model_seed: int | None, value: float | None,
             replicate: int | None) -> Path:
    if method == "vcg":
        name = f"model-{model_seed}-lambda-{value:.2f}".replace(".", "p")
    else:
        name = f"replicate-{replicate}"
    return output / "rows" / panel / method / f"instance-{seed}-{name}.json"


def _instance(frozen: dict, panel: str, seed: int) -> EpisodeInstance:
    matching = [record for record in frozen["records"]
                if record["panel"] == panel and record["seed"] == seed]
    require(len(matching) == 1, "E91 instance is not uniquely frozen")
    instance = EpisodeInstance.from_json(matching[0]["instance_json"])
    require(instance.instance_id == matching[0]["instance_id"]
            and instance.schedule_id == matching[0]["schedule_id"],
            "E91 frozen instance identity changed")
    instance.validate_for(p.SCENARIO.make_base_env())
    return instance


def _metrics(env, strict: bool) -> dict:
    blocks = tuple(env.blocks)
    flow = summarize_block_storage_flow(blocks, int(env.time_steps))
    require(flow["storage_flow_manifest_count"] == p.SCENARIO.total_jobs,
            "E91 storage-flow manifest differs")
    result = {
        "delivered_at_stop": sum(bool(block.delivered) for block in blocks),
        "storage_flow_completed": flow["storage_flow_completed_count"],
        "storage_flow_censored": flow["storage_flow_right_censored_count"],
        "mean_wait_before_storage": None,
        "storage_clock": None,
        "release_clock": None,
        "occupancy": env.occupancy_summary(),
    }
    if strict:
        require(all(block.stored_time_step is not None
                    and block.delivered_time_step is not None for block in blocks),
                "strict episode lacks storage or delivery clock")
        storage_errors = [float(block.delivered_time_step - block.stored_time_step
                                - block.storage_steps_needed) for block in blocks]
        release_errors = [float(block.delivered_time_step - block.arrival_step
                                - block.storage_steps_needed) for block in blocks]
        result.update({
            "mean_wait_before_storage": flow["mean_storage_flow_time"],
            "storage_clock": e82.timing(storage_errors),
            "release_clock": e82.timing(release_errors),
        })
    return result


def _vcg_row(parent: dict, instance: EpisodeInstance, seed: int,
             model_seed: int, value: float) -> dict:
    agent, search, guard, base_digest = e82.load_model(parent, model_seed)
    base_entry = parent["stage_a"][model_seed]
    payload = torch.load(p.ROOT / base_entry["terminal_checkpoint"],
                         map_location="cpu", weights_only=False, mmap=True)
    envs = []

    def make_env(_payload):
        env = adapter.AuditedEmptyStartEnv(p.SCENARIO)
        envs.append(env)
        return env

    def freeze_agent(_payload, *, device, seed):
        require(device.type == "cpu" and seed == model_seed,
                "VCG evaluation changed model seed or device")
        return e82.FixedLambdaAgent(agent, value)

    started = perf_counter()
    raw = None
    failure = None
    with ExitStack() as stack:
        stack.enter_context(reuse.path_cleanup_active())
        stack.enter_context(patch.object(benchmark, "_make_env", make_env))
        stack.enter_context(patch.object(benchmark, "_freeze_agent", freeze_agent))
        stack.enter_context(patch.object(
            benchmark, "ViabilityCertificateCache", reuse.TimingInvariantCertificateCache,
        ))
        try:
            with e13._WallClock(e28.WALL_SECONDS):
                raw = benchmark.run_arm(
                    arm=benchmark.EXACT_FULL, controller_payload=payload,
                    instance=instance, instance_seed=seed,
                    search_config=search, liveness_rule=guard, prioritizer=None,
                    max_steps=p.SCENARIO.max_steps, device=torch.device("cpu"),
                    recovery_certification_strategy=RELOCATION_FAMILY_CERTIFICATION,
                )
        except e13.EpisodeWallLimit as error:
            failure = str(error)
    elapsed = perf_counter() - started
    require(len(envs) == 1 and envs[0].warm_start_audit is not None
            and envs[0].warm_start_audit["passed"]
            and envs[0].current_episode_instance.instance_id == instance.instance_id,
            "VCG did not execute the authenticated empty-start instance")
    require(e82.e80.state_digest(agent.base_agent.Q_local.state_dict()) == base_digest,
            "VCG evaluation mutated Stage A")
    strict = bool(raw is not None and raw["strict_method_success"]
                  and raw["terminal"] and raw["method_failure_reason"] is None
                  and raw["complete_frontier_exactly_verified"]
                  and raw["illegal_drops"] == 0 and raw["macro_failures"] == 0
                  and sum(block.delivered for block in envs[0].blocks) == p.SCENARIO.total_jobs)
    rehandles = None if raw is None else int(raw.get(
        "physical_storage_relocations", raw["relocations"]))
    decisions = () if raw is None else raw["decisions"]
    decision_count = len(decisions)
    return {
        "strict_complete": strict,
        "failure_class": ("completed" if strict else "censored_wall_clock" if failure
                          else "incomplete_or_certificate_limited"),
        "method_failure_reason": failure if raw is None else raw["method_failure_reason"],
        "steps_at_stop": int(envs[0].time_steps),
        "rehandles_at_stop": rehandles,
        "decision_count": decision_count,
        "guard_forced_decisions": (None if raw is None else sum(
            bool(item["liveness_forced"]) for item in decisions)),
        "rehandles_per_100": (100 * rehandles / p.SCENARIO.total_jobs if strict else None),
        "steps_per_delivery": (int(raw["steps"]) / p.SCENARIO.total_jobs if strict else None),
        "episode_wall_seconds": elapsed,
        "amortized_wall_seconds_per_decision": elapsed / decision_count if decision_count else None,
        "mean_selector_seconds": (mean(item["selection_seconds"] for item in decisions)
                                  if decisions else None),
        "mean_frontier_seconds": (raw["total_frontier_seconds"] / decision_count
                                  if decision_count else None),
        "behavior_digest": None if raw is None else raw["behavior_digest"],
        "base_checkpoint_sha256": base_entry["terminal_checkpoint_sha256"],
        "handling_checkpoint_sha256": parent["terminal_heads"][model_seed]["checkpoint_sha256"],
        **_metrics(envs[0], strict),
    }


def _baseline_row(method: str, instance: EpisodeInstance, seed: int,
                  replicate: int, instance_index: int) -> dict:
    baseline_method = e28.DYNAMIC if method == "dynamic_pslap" else e28.GA
    identity = (
        {"replicate": 0, "model_seed": None, "rng_index": None, "rng_seed": None}
        if method == "dynamic_pslap" else
        {"replicate": replicate, "model_seed": None, "rng_index": replicate,
         "rng_seed": e28.e25.GA_RNG_BASE + e28.e25.GA_RNG_STRIDE
                     * (4 * (1000 + instance_index) + replicate)}
    )
    created = []
    captured = []
    original_factory = adapter.empty_start_factory
    original_evaluate = e28.v4.evaluate_one

    def capture_factory(*args, **kwargs):
        env = original_factory(*args, **kwargs)
        created.append(env)
        return env

    def capture_evaluate(*args, **kwargs):
        raw = original_evaluate(*args, **kwargs)
        captured.append(raw)
        return raw

    with (patch.object(adapter, "empty_start_factory", side_effect=capture_factory),
          patch.object(e28.v4, "evaluate_one", side_effect=capture_evaluate)):
        old = e28._run_baseline(baseline_method, p.SCENARIO, instance, seed, identity)
    require(len(created) == 1 and created[0].warm_start_audit is not None
            and created[0].warm_start_audit["passed"]
            and created[0].current_episode_instance.instance_id == instance.instance_id,
            "baseline did not execute the authenticated empty-start instance")
    strict = bool(old["strict_safe_complete"])
    raw = captured[0] if captured else None
    decision_count = (len(raw["urgency_scheduler_audit"]["decisions"])
                      if raw is not None else None)
    result = {
        "strict_complete": strict,
        "failure_class": old["failure_class"],
        "method_failure_reason": old["method_failure_reason"],
        "steps_at_stop": old["steps_at_stop"],
        "rehandles_at_stop": old["physical_rehandles_at_stop"],
        "decision_count": decision_count,
        "rehandles_per_100": old["physical_rehandles_per_100_required_deliveries"],
        "steps_per_delivery": old["steps_per_delivery"],
        "episode_wall_seconds": old["episode_wall_seconds"],
        "amortized_wall_seconds_per_decision": (
            old["episode_wall_seconds"] / decision_count if decision_count else None),
        "mean_selector_seconds": None,
        "mean_frontier_seconds": None,
        "baseline_reported_mae": old["mean_absolute_error"],
        "rng_seed": identity["rng_seed"],
        **_metrics(created[0], strict),
    }
    require(not strict or result["delivered_at_stop"] == p.SCENARIO.total_jobs,
            "baseline strict flag disagrees with physical deliveries")
    return result


def run_one(output: Path, panel: str, method: str, seed: int,
            model_seed: int | None, value: float | None,
            replicate: int | None) -> dict:
    contract, frozen = authenticate(output)
    require((seed, model_seed, value, replicate) in set(coordinates(panel, method)),
            "E91 coordinate is outside the frozen panel")
    path = row_path(output, panel, method, seed, model_seed, value, replicate)
    instance = _instance(frozen, panel, seed)
    identity = {
        "protocol": p.PROTOCOL, "contract_sha256": contract["contract_sha256"],
        "panel": panel, "method": method, "scenario": p.SCENARIO_ID,
        "instance_seed": seed, "instance_id": instance.instance_id,
        "model_seed": model_seed, "preference_lambda": value, "replicate": replicate,
    }
    if path.is_file():
        row = read_json(path)
        require(all(row.get(key) == expected for key, expected in identity.items()),
                f"E91 saved coordinate differs: {path}")
        return {**row, "reused": True}
    print(f"E91 {panel} start | {method} instance={seed} model={model_seed} "
          f"lambda={value} replicate={replicate}", flush=True)
    if method == "vcg":
        parent = e82.p.contract()
        result = _vcg_row(parent, instance, seed, int(model_seed), float(value))
    else:
        result = _baseline_row(method, instance, seed, int(replicate),
                               p.PANEL_SEEDS[panel].index(seed))
    row = {**identity, **result, "required_deliveries": p.SCENARIO.total_jobs}
    _atomic_json_save(row, path)
    print(f"E91 {panel} done | {method} instance={seed} model={model_seed} "
          f"lambda={value} replicate={replicate} strict={int(row['strict_complete'])} "
          f"delivered={row['delivered_at_stop']}/{p.SCENARIO.total_jobs} "
          f"steps={row['steps_at_stop']} wall={row['episode_wall_seconds']:.1f}s",
          flush=True)
    return row


METRICS = (
    "release_mae", "storage_mae", "rehandles_per_100", "within_20",
    "steps_per_delivery", "mean_wait_before_storage",
)


def metric(row: dict, name: str) -> float | None:
    if not row["strict_complete"]:
        return None
    if name == "release_mae":
        return row["release_clock"]["mae"]
    if name == "storage_mae":
        return row["storage_clock"]["mae"]
    if name == "within_20":
        return row["storage_clock"]["within_20_rate"]
    return row[name]


def holm(p_values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(p_values, key=lambda key: (p_values[key], key))
    adjusted = {}
    running = 0.0
    for index, key in enumerate(ordered):
        running = max(running, min(1.0, (len(ordered) - index) * p_values[key]))
        adjusted[key] = running
    return adjusted


def paired(values: list[float]) -> dict:
    require(len(values) == 30 and all(math.isfinite(value) for value in values),
            "E91 paired interval requires 30 finite instance differences")
    point = mean(values)
    se = stdev(values) / math.sqrt(len(values))
    if se == 0:
        p_value = 1.0 if point == 0 else 0.0
    else:
        p_value = float(2 * student_t.sf(abs(point / se), len(values) - 1))
    critical = float(student_t.ppf(0.975, len(values) - 1))
    return {
        "n_instances": len(values), "mean_difference": point,
        "nominal_95_ci": [point - critical * se, point + critical * se],
        "nominal_two_sided_p": p_value,
    }


def analyze(output: Path, panel: str) -> dict:
    contract, _ = authenticate(output)
    expected = [
        (method, *coordinate)
        for method in p.METHODS for coordinate in coordinates(panel, method)
    ]
    rows = []
    missing = []
    for method, seed, model_seed, value, replicate in expected:
        path = row_path(output, panel, method, seed, model_seed, value, replicate)
        if not path.is_file():
            missing.append(str(path))
            continue
        row = read_json(path)
        require(row.get("contract_sha256") == contract["contract_sha256"]
                and row.get("method") == method and row.get("instance_seed") == seed
                and row.get("model_seed") == model_seed
                and row.get("preference_lambda") == value
                and row.get("replicate") == replicate,
                f"E91 row identity differs: {path}")
        rows.append(row)
    labels = [f"vcg_{value:.2f}" for value in p.LAMBDAS] + ["dynamic_pslap", "ga"]
    def label(row):
        return f"vcg_{row['preference_lambda']:.2f}" if row["method"] == "vcg" else row["method"]
    by_label_seed = {(name, seed): [row for row in rows if label(row) == name
                                   and row["instance_seed"] == seed]
                     for name in labels for seed in p.PANEL_SEEDS[panel]}
    points = {}
    summaries = []
    for name in labels:
        expected_replicates = (len(p.MODEL_SEEDS) if name.startswith("vcg_") else
                               len(p.GA_REPLICATES) if name == "ga" else 1)
        if panel == "pilot":
            expected_replicates = 1
        selected = [row for row in rows if label(row) == name]
        for seed in p.PANEL_SEEDS[panel]:
            group = by_label_seed[name, seed]
            complete_group = len(group) == expected_replicates and all(
                row["strict_complete"] for row in group)
            points[name, seed] = {
                "complete": complete_group,
                "strict_completion_fraction": (mean(bool(row["strict_complete"]) for row in group)
                                               if len(group) == expected_replicates else None),
                "metrics": ({metric_name: mean(metric(row, metric_name) for row in group)
                             for metric_name in METRICS} if complete_group else None),
                "episode_wall_seconds": (mean(row["episode_wall_seconds"] for row in group)
                                         if len(group) == expected_replicates else None),
                "amortized_wall_seconds_per_decision": (
                    mean(row["amortized_wall_seconds_per_decision"] for row in group)
                    if len(group) == expected_replicates and all(
                        row["amortized_wall_seconds_per_decision"] is not None for row in group)
                    else None),
            }
        fully_complete = [points[name, seed] for seed in p.PANEL_SEEDS[panel]
                          if points[name, seed]["complete"]]
        summaries.append({
            "method": name, "observed_rows": len(selected),
            "expected_rows": len(p.PANEL_SEEDS[panel]) * expected_replicates,
            "strict_complete_rows": sum(bool(row["strict_complete"]) for row in selected),
            "strict_completion_rate": (mean(bool(row["strict_complete"]) for row in selected)
                                       if selected else None),
            "fully_complete_instances": len(fully_complete),
            "mean_metrics_fully_complete_instances": (
                {metric_name: mean(item["metrics"][metric_name] for item in fully_complete)
                 for metric_name in METRICS} if fully_complete else None),
            "mean_episode_wall_seconds": (mean(item["episode_wall_seconds"]
                                               for item in fully_complete)
                                          if fully_complete else None),
            "mean_amortized_wall_seconds_per_decision": (
                mean(item["amortized_wall_seconds_per_decision"] for item in fully_complete)
                if fully_complete and all(item["amortized_wall_seconds_per_decision"]
                                          is not None for item in fully_complete) else None),
        })
    all_present = not missing
    all_complete = all_present and all(row["strict_complete"] for row in rows)
    contrasts = []
    if panel == "test" and all_complete:
        for value in p.LAMBDAS:
            vcg_label = f"vcg_{value:.2f}"
            for baseline in ("dynamic_pslap", "ga"):
                results = {}
                for name in METRICS:
                    differences = [points[vcg_label, seed]["metrics"][name]
                                   - points[baseline, seed]["metrics"][name]
                                   for seed in p.PANEL_SEEDS[panel]]
                    results[name] = paired(differences)
                contrasts.append({
                    "vcg_lambda": value, "baseline": baseline,
                    "difference": "VCG minus baseline", "metrics": results,
                })
        keyed = {f"{item['vcg_lambda']:.2f}/{item['baseline']}/{name}":
                 item["metrics"][name]["nominal_two_sided_p"]
                 for item in contrasts for name in METRICS}
        adjusted = holm(keyed)
        for item in contrasts:
            for name in METRICS:
                key = f"{item['vcg_lambda']:.2f}/{item['baseline']}/{name}"
                item["metrics"][name]["holm_adjusted_p_across_48"] = adjusted[key]
        baseline_means = {
            baseline: mean(points[baseline, seed]["metrics"]["release_mae"]
                           for seed in p.PANEL_SEEDS[panel])
            for baseline in ("dynamic_pslap", "ga")
        }
        strongest = min(baseline_means, key=lambda key: (baseline_means[key], key))
    else:
        strongest = None
    result = {
        "protocol": p.PROTOCOL, "contract_sha256": contract["contract_sha256"],
        "panel": panel,
        "status": "complete" if all_present else "partial",
        "expected_rows": len(expected), "observed_rows": len(rows),
        "missing_rows": missing,
        "all_rows_strict_complete": all_complete,
        "primary_quality_inference": ("available" if panel == "test" and all_complete
                                      else "refused_until_all_test_rows_strict_complete"),
        "summaries": summaries,
        "instance_points": [
            {"method": name, "instance_seed": seed, **points[name, seed]}
            for name in labels for seed in p.PANEL_SEEDS[panel]
        ],
        "strongest_baseline_by_test_release_mae": strongest,
        "baseline_selection_caveat": "descriptive post-test label only; Holm family includes both baselines",
        "contrasts": contrasts,
        "interval_caveat": "95% paired t intervals are nominal; Holm correction applies to the 48 reported two-sided p-values, not to intervals",
        "quality_clock_caveat": "within ±20 is storage-referenced, matching the historical 5x5 benchmark; release MAE is separately primary",
        "a3c_disposition": contract["a3c_disposition"],
        "training_runs": 0,
    }
    report_path = output / f"e91-{panel}-report.json"
    _atomic_json_save(result, report_path)
    table_path = output / f"e91-{panel}-table.md"
    table_path.write_text(render_table(result), encoding="utf-8")
    return {"status": result["status"], "panel": panel,
            "observed_rows": len(rows), "expected_rows": len(expected),
            "all_rows_strict_complete": all_complete,
            "primary_quality_inference": result["primary_quality_inference"],
            "report": str(report_path), "table": str(table_path),
            "training_runs": 0}


def render_table(report: dict) -> str:
    def number(value, digits=2):
        return "—" if value is None else f"{value:.{digits}f}"

    lines = [
        f"# E91 {report['panel']} matched 8×8 panel",
        "",
        f"Observed {report['observed_rows']}/{report['expected_rows']} episodes. "
        f"Primary quality inference: **{report['primary_quality_inference']}**.",
        "",
        "| Method | Strict | Complete instances | Release MAE | Storage MAE | "
        "Within ±20 | Rehandles/100 | Steps/delivery | Wait before storage | "
        "Wall s/decision |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in report["summaries"]:
        metrics = row["mean_metrics_fully_complete_instances"] or {}
        lines.append(
            f"| {row['method']} | {row['strict_complete_rows']}/{row['expected_rows']} | "
            f"{row['fully_complete_instances']}/{len(p.PANEL_SEEDS[report['panel']])} | "
            f"{number(metrics.get('release_mae'))} | {number(metrics.get('storage_mae'))} | "
            f"{number(metrics.get('within_20'), 3)} | "
            f"{number(metrics.get('rehandles_per_100'))} | "
            f"{number(metrics.get('steps_per_delivery'))} | "
            f"{number(metrics.get('mean_wait_before_storage'))} | "
            f"{number(row['mean_amortized_wall_seconds_per_decision'], 3)} |"
        )
    lines.extend([
        "",
        "Quality means use only instances where every replicate/model seed for that method "
        "completed. Strict completion always uses all observed episodes; missing episodes "
        "remain visible in the denominator. Within ±20 is storage-referenced. "
        "Wall seconds/decision includes environment execution and search.",
    ])
    if report["contrasts"]:
        strongest = report["strongest_baseline_by_test_release_mae"]
        lines.extend([
            "",
            f"Descriptively strongest baseline by release MAE: **{strongest}**. "
            "Both baselines remain in the Holm family.",
            "",
            "| VCG λ | Metric | VCG − strongest baseline [nominal 95% CI] | "
            "Holm p (48 tests) |",
            "| ---: | --- | ---: | ---: |",
        ])
        for contrast in report["contrasts"]:
            if contrast["baseline"] != strongest:
                continue
            for metric_name in METRICS:
                item = contrast["metrics"][metric_name]
                low, high = item["nominal_95_ci"]
                lines.append(
                    f"| {contrast['vcg_lambda']:.2f} | {metric_name} | "
                    f"{item['mean_difference']:+.3f} [{low:+.3f}, {high:+.3f}] | "
                    f"{item['holm_adjusted_p_across_48']:.4g} |"
                )
        lines.extend(["", "CIs are nominal; Holm correction applies to p-values only."])
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "verify-models", "run-one", "run-panel", "analyze"))
    parser.add_argument("--output", type=Path, default=p.DEFAULT_OUTPUT)
    parser.add_argument("--panel", choices=tuple(p.PANEL_SEEDS), default="pilot")
    parser.add_argument("--method", choices=p.METHODS)
    parser.add_argument("--instance-seed", type=int)
    parser.add_argument("--model-seed", type=int, choices=p.MODEL_SEEDS)
    parser.add_argument("--lambda", dest="preference_lambda", type=float, choices=p.LAMBDAS)
    parser.add_argument("--replicate", type=int)
    parser.add_argument("--from-instance", type=int, default=0)
    parser.add_argument("--through-instance", type=int)
    args = parser.parse_args()
    output = args.output.resolve()
    if args.command == "prepare":
        result = prepare(output)
    elif args.command == "verify-models":
        authenticate(output)
        result = e82.verify_models(e82.p.DEFAULT_OUTPUT)
    elif args.command == "run-one":
        require(args.method is not None and args.instance_seed is not None,
                "run-one needs method and instance-seed")
        result = run_one(output, args.panel, args.method, args.instance_seed,
                         args.model_seed, args.preference_lambda, args.replicate)
    elif args.command == "run-panel":
        authenticate(output)
        seeds = p.PANEL_SEEDS[args.panel]
        last = len(seeds) if args.through_instance is None else args.through_instance
        require(0 <= args.from_instance < last <= len(seeds), "invalid instance chunk")
        selected = set(seeds[args.from_instance:last])
        methods = (args.method,) if args.method is not None else p.METHODS
        count = 0
        for method in methods:
            for seed, model_seed, value, replicate in coordinates(args.panel, method):
                if seed not in selected:
                    continue
                row = run_one(output, args.panel, method, seed,
                              model_seed, value, replicate)
                count += int(not row.get("reused", False))
        result = {"status": "chunk_complete", "panel": args.panel,
                  "new_episode_runs": count, "training_runs": 0}
    else:
        result = analyze(output, args.panel)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
