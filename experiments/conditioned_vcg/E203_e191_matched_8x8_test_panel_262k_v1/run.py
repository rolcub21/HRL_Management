#!/usr/bin/env python3
"""Run the locked E203 matched 8x8 paper test panel."""

from __future__ import annotations

import argparse
import builtins
from collections import defaultdict
from contextlib import ExitStack
import json
import math
import os
from pathlib import Path
from statistics import fmean
from unittest.mock import patch

import torch

import run_vcg_objective_gamma_audit as audit
from example.episode_instance import EpisodeInstance
from experiments.conditioned_vcg.E14_certification_scalability_95k import reuse
from experiments.conditioned_vcg.E28_empty_start_scale_baselines_100k import (
    adapter as empty_adapter,
)
from experiments.conditioned_vcg.E91_matched_target_size_benchmark_156k_v1 import (
    run as e91,
)
from experiments.conditioned_vcg.E100_accelerated_release_clock_163k_v1 import (
    fast_inputs,
)
from experiments.conditioned_vcg.E107_queue_context_stage_a_8x8_171k_v1 import (
    representation as e107_representation,
)
from experiments.conditioned_vcg.E189_e188_roundoff_repair_248k_v1.roundoff import (
    repaired_audit,
)
from experiments.conditioned_vcg.E191_fresh_long_stay_crowding_250k_v1 import (
    instrumentation as e191_instrumentation,
    protocol as e191,
    representation as e191_representation,
    scenario as e191_scenario,
    stage as e191_stage,
)
from experiments.conditioned_vcg.E202_e191_multiseed_development_freeze_261k_v1 import (
    run as e202,
)
from experiments.conditioned_vcg.E192_e191_matched_longstay_heuristics_251k_v1 import (
    run as e192,
)

from . import protocol as p, scenario


class E203Error(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise E203Error(message)


def _read(path: Path) -> dict:
    require(Path(path).is_file(), f"missing E203 artifact: {path}")
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"invalid E203 JSON object: {path}")
    return value


def _jsonable(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(
        _jsonable(value), indent=2, sort_keys=True, allow_nan=False, default=str
    ) + "\n"
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(encoded, encoding="utf-8")
    os.replace(temporary, path)


def make_instance(seed: int) -> EpisodeInstance:
    require(seed in scenario.TEST_SEEDS, "seed is outside the E203 test panel")
    env = scenario.TestPanel8x8Env(scenario.SCENARIO, e191.SPEC)
    instance = env.sample_episode_instance(seed=int(seed))
    instance.validate_for(env)
    return instance


def instance_manifest(contract: dict) -> dict:
    records = []
    for seed in scenario.TEST_SEEDS:
        instance = make_instance(seed)
        records.append({
            "seed": int(seed),
            "workload": scenario.level_label(scenario.level_for_seed(seed)),
            "replicate": scenario.replicate_for_seed(seed),
            "instance_id": instance.instance_id,
            "schedule_id": instance.schedule_id,
            "instance_json": instance.to_json(),
        })
    core = {
        "protocol": p.PROTOCOL,
        "contract_sha256": contract["contract_sha256"],
        "records": records,
    }
    return {**core, "manifest_sha256": p.digest(core)}


def prepare(output: Path) -> dict:
    output = output.resolve()
    contract = p.contract()
    manifest = instance_manifest(contract)
    contract_path = output / "contract.json"
    manifest_path = output / "instances.json"
    if contract_path.is_file():
        require(_read(contract_path) == contract, "E203 contract changed")
        require(_read(manifest_path) == manifest, "E203 instance manifest changed")
    else:
        require(
            not output.exists() or not any(output.iterdir()),
            "E203 refuses a nonempty uncontracted output",
        )
        _write(contract_path, contract)
        _write(manifest_path, manifest)
    return {
        "status": "prepared",
        "contract_sha256": contract["contract_sha256"],
        "instances": len(scenario.TEST_SEEDS),
        "expected_rows": p.EXPECTED_ROWS,
        "training_runs": 0,
        "episode_runs": 0,
        "test_panel_opened": False,
    }


def authenticate(output: Path) -> tuple[dict, dict]:
    prepared = prepare(output)
    contract = _read(output.resolve() / "contract.json")
    manifest = _read(output.resolve() / "instances.json")
    require(
        contract["contract_sha256"] == prepared["contract_sha256"],
        "E203 authentication failed",
    )
    return contract, manifest


def frozen_instance(manifest: dict, seed: int) -> EpisodeInstance:
    rows = [row for row in manifest["records"] if int(row["seed"]) == int(seed)]
    require(len(rows) == 1, "E203 instance is not uniquely frozen")
    row = rows[0]
    instance = EpisodeInstance.from_json(row["instance_json"])
    expected = make_instance(seed)
    require(
        instance.instance_id == row["instance_id"] == expected.instance_id
        and instance.schedule_id == row["schedule_id"] == expected.schedule_id,
        "E203 frozen instance identity changed",
    )
    return instance


def _freeze(model_seed: int) -> dict:
    path = p.FREEZE_OUTPUT / f"seed-{model_seed}" / e202.MANIFEST_NAME
    value = _read(path)
    require(
        value.get("status") == "complete"
        and int(value.get("model_seed", -1)) == int(model_seed)
        and value.get("source_contract_sha256")
        == e191.contract()["contract_sha256"],
        f"E203 frozen E191 seed {model_seed} differs",
    )
    checkpoint = Path(value["frozen_checkpoint"])
    require(
        checkpoint.is_file()
        and p.sha256(checkpoint) == value["frozen_checkpoint_sha256"],
        f"E203 frozen E191 seed {model_seed} hash differs",
    )
    return value


def _load_agent(model_seed: int, device: str):
    freeze = _freeze(model_seed)
    payload = torch.load(
        freeze["frozen_checkpoint"], map_location="cpu", weights_only=False
    )
    agent = e191_representation.FreshChoicePreservingQueueAgent.from_checkpoint(
        payload, device=device, resumable=False, seed=model_seed
    )
    agent.set_epsilon(0.0)
    agent.Q_local.eval()
    agent.Q_target.eval()
    return agent, freeze


def _accelerated_enumerator(original):
    return e191_stage._accelerated_enumerator(original)


def _vcg_row_path(output: Path, model_seed: int, seed: int) -> Path:
    return (
        output.resolve() / "rows/e191" / f"model-seed-{model_seed}"
        / f"instance-{int(seed)}.json"
    )


def _baseline_row_path(output: Path, method: str, seed: int, replicate: int) -> Path:
    return (
        output.resolve() / "rows" / method
        / f"instance-{int(seed)}-replicate-{int(replicate)}.json"
    )


def run_vcg_one(
    output: Path, model_seed: int, seed: int, device: str
) -> dict:
    contract, manifest = authenticate(output)
    require(model_seed in p.MODEL_SEEDS, "invalid E203 E191 model seed")
    require(seed in scenario.TEST_SEEDS, "invalid E203 instance seed")
    path = _vcg_row_path(output, model_seed, seed)
    identity = {
        "protocol": p.PROTOCOL,
        "contract_sha256": contract["contract_sha256"],
        "method": "e191",
        "model_seed": int(model_seed),
        "instance_seed": int(seed),
        "replicate": 0,
    }
    if path.is_file():
        row = _read(path)
        require(
            all(row.get(key) == value for key, value in identity.items()),
            f"saved E203 E191 row differs: {path}",
        )
        return {**row, "reused": True, "episode_runs": 0}
    require(device in ("cpu", "cuda"), "E203 device must be cpu or cuda")
    if device == "cuda":
        require(torch.cuda.is_available(), "CUDA was requested but is unavailable")
    expected = frozen_instance(manifest, seed)
    agent, freeze = _load_agent(model_seed, device)
    env = scenario.TestPanel8x8Env(scenario.SCENARIO, e191.SPEC)
    original_print = builtins.print

    def e203_print(*values, **kwargs):
        if values and isinstance(values[0], str) and values[0].startswith("E191 "):
            values = ("E203 " + values[0][len("E191 "):], *values[1:])
        return original_print(*values, **kwargs)

    print(
        f"E203 start method=e191 model_seed={model_seed} seed={seed}",
        flush=True,
    )
    with ExitStack() as stack:
        stack.enter_context(patch.object(
            audit,
            "enumerate_viability_candidates",
            _accelerated_enumerator(audit.enumerate_viability_candidates),
        ))
        stack.enter_context(patch.object(
            audit, "ViabilityCertificateCache", reuse.TimingInvariantCertificateCache
        ))
        stack.enter_context(patch.object(
            audit, "_rescore_run", e191_instrumentation.single_objective_rescore
        ))
        stack.enter_context(patch.object(builtins, "print", e203_print))
        stack.enter_context(reuse.path_cleanup_active())
        stack.enter_context(fast_inputs.enabled())
        stack.enter_context(e107_representation.acceleration_enabled())
        stack.enter_context(repaired_audit())
        run = e191_instrumentation.run_episode(
            agent,
            env,
            objective_spec=e191.SPEC,
            instance_seed=int(seed),
            training=False,
            max_steps=scenario.MAX_STEPS,
            search_config=payload_search(freeze),
            liveness_rule=payload_liveness(freeze),
            epsilon_start=0.0,
            epsilon_end=0.0,
            epsilon_decay_decisions=1,
            epsilon_warmup_decisions=0,
            updates_per_macro=1,
            run_protocol=p.PROTOCOL,
        )
    require(
        env.current_episode_instance.instance_id == expected.instance_id,
        "E203 E191 executed another instance",
    )
    row = {
        **identity,
        "instance_id": expected.instance_id,
        "schedule_id": expected.schedule_id,
        "status": "complete",
        "selected_episode": int(freeze["selected_episode"]),
        "checkpoint_sha256": freeze["frozen_checkpoint_sha256"],
        "strict_complete": bool(run["strict_method_success"]),
        "method_failure_reason": run["method_failure_reason"],
        "run": run,
        "metrics": dict(run["e187_metrics"]),
        "training_runs": 0,
        "episode_runs": 1,
        "test_panel_opened": True,
        "paper_evidence": True,
    }
    _write(path, row)
    return row


def payload_search(freeze: dict):
    payload = torch.load(
        freeze["frozen_checkpoint"], map_location="cpu", weights_only=False
    )
    from PSLAP.viability_filter import ViabilitySearchConfig

    value = payload["viability_search"]
    return ViabilitySearchConfig(
        max_depth=value["max_depth"],
        max_nodes=int(value["max_nodes"]),
        max_primitive_steps=value["max_primitive_steps"],
        reserve_queue_cells=bool(value["reserve_queue_cells"]),
        search_order=str(value["search_order"]),
    )


def payload_liveness(freeze: dict):
    payload = torch.load(
        freeze["frozen_checkpoint"], map_location="cpu", weights_only=False
    )
    from PSLAP.viability_candidates import BoundedEventDeferRule

    value = payload["liveness_rule"]
    return BoundedEventDeferRule(
        max_option_steps=int(value["max_option_steps"]),
        max_consecutive_defer_decisions=int(
            value["max_consecutive_defer_decisions"]
        ),
    )


def _environment_factory(
    baseline_scenario,
    *,
    arrival_rate,
    proc_mean,
    grid_rows,
    grid_cols,
    exit_width,
    number_blocks,
):
    expected = e191.scenario_for(scenario.SCENARIO)
    require(
        baseline_scenario.public_dict()
        == expected.public_dict()
        and int(grid_rows) == expected.rows
        and int(grid_cols) == expected.cols
        and int(number_blocks) == scenario.TOTAL_JOBS
        and int(exit_width) == 3,
        "E203 baseline requested another geometry",
    )
    environment = e192.TracedLongStayEnv(baseline_scenario)
    environment.arrival_rate = float(arrival_rate)
    environment.proc_mean = float(proc_mean)
    environment.e192_requested_defaults = {
        "arrival_rate": float(arrival_rate),
        "proc_mean": float(proc_mean),
    }
    return environment


def run_baseline_one(
    output: Path, method: str, seed: int, replicate: int
) -> dict:
    contract, manifest = authenticate(output)
    require(method in ("dynamic_pslap", "ga"), "invalid E203 baseline")
    require(seed in scenario.TEST_SEEDS, "invalid E203 instance seed")
    require(replicate in p.replicates(method), "invalid E203 replicate")
    path = _baseline_row_path(output, method, seed, replicate)
    identity = {
        "protocol": p.PROTOCOL,
        "contract_sha256": contract["contract_sha256"],
        "method": method,
        "instance_seed": int(seed),
        "replicate": int(replicate),
    }
    if path.is_file():
        row = _read(path)
        require(
            all(row.get(key) == value for key, value in identity.items()),
            f"saved E203 baseline row differs: {path}",
        )
        return {**row, "reused": True, "episode_runs": 0}
    instance = frozen_instance(manifest, seed)
    e192.TracedLongStayEnv.created = []
    print(
        f"E203 start method={method} seed={seed} replicate={replicate}",
        flush=True,
    )
    def instance_bound_factory(*args, **kwargs):
        environment = _environment_factory(*args, **kwargs)
        environment.arrival_rate = float(instance.arrival_rate)
        environment.proc_mean = float(instance.proc_mean)
        return environment

    with (
        patch.object(
            empty_adapter, "empty_start_factory", side_effect=instance_bound_factory
        ),
        patch.object(e91.p, "SCENARIO", e191.scenario_for(scenario.SCENARIO)),
    ):
        baseline = e91._baseline_row(
            method,
            instance,
            int(seed),
            int(replicate),
            scenario.TEST_SEEDS.index(seed),
        )
    require(
        len(e192.TracedLongStayEnv.created) == 1,
        "E203 baseline did not create exactly one environment",
    )
    env = e192.TracedLongStayEnv.created[0]
    require(
        env.current_episode_instance.instance_id == instance.instance_id,
        "E203 baseline executed another instance",
    )
    metrics = e192._completed_metrics(env, bool(baseline["strict_complete"]))
    require(
        bool(metrics["strict_complete"]) == bool(baseline["strict_complete"]),
        "E203 baseline completion flags differ",
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
        "wall_seconds_per_decision": baseline[
            "amortized_wall_seconds_per_decision"
        ],
        "strict_complete": bool(metrics["strict_complete"]),
        "metrics": metrics,
        "training_runs": 0,
        "episode_runs": 1,
        "test_panel_opened": True,
        "paper_evidence": True,
    }
    _write(path, row)
    return row


def _saved_rows(output: Path) -> list[dict]:
    contract = p.contract()
    rows = []
    for method in ("dynamic_pslap", "ga"):
        for seed in scenario.TEST_SEEDS:
            for replicate in p.replicates(method):
                path = _baseline_row_path(output, method, seed, replicate)
                if path.is_file():
                    row = _read(path)
                    require(
                        row.get("protocol") == p.PROTOCOL
                        and row.get("contract_sha256")
                        == contract["contract_sha256"]
                        and row.get("method") == method
                        and int(row.get("instance_seed", -1)) == seed
                        and int(row.get("replicate", -1)) == replicate,
                        f"saved E203 baseline identity differs: {path}",
                    )
                    rows.append(row)
    for model_seed in p.MODEL_SEEDS:
        for seed in scenario.TEST_SEEDS:
            path = _vcg_row_path(output, model_seed, seed)
            if path.is_file():
                row = _read(path)
                require(
                    row.get("protocol") == p.PROTOCOL
                    and row.get("contract_sha256")
                    == contract["contract_sha256"]
                    and row.get("method") == "e191"
                    and int(row.get("model_seed", -1)) == model_seed
                    and int(row.get("instance_seed", -1)) == seed
                    and int(row.get("replicate", -1)) == 0,
                    f"saved E203 E191 identity differs: {path}",
                )
                rows.append(row)
    return rows


def _flat(row: dict) -> dict | None:
    if not row.get("strict_complete"):
        return None
    metrics = row["metrics"]
    steps = int(row.get("run", {}).get("steps", metrics.get("steps", 0)))
    return e192._flat_metrics(metrics, steps=steps)


def _baseline_points(rows: list[dict]) -> dict[str, list[dict]]:
    grouped = defaultdict(list)
    for row in rows:
        if row["method"] in ("dynamic_pslap", "ga"):
            grouped[(row["method"], int(row["instance_seed"]))].append(row)
    result = {method: [] for method in ("dynamic_pslap", "ga")}
    for method in result:
        expected = set(p.replicates(method))
        for seed in scenario.TEST_SEEDS:
            group = grouped[(method, seed)]
            observed = {int(row["replicate"]) for row in group}
            complete = [row for row in group if _flat(row) is not None]
            strict = observed == expected and len(complete) == len(expected)
            result[method].append({
                "method": method,
                "instance_seed": seed,
                "strict_complete": strict,
                "metrics": (
                    e192._average_metrics([_flat(row) for row in complete])
                    if strict else None
                ),
            })
    return result


def _vcg_points(rows: list[dict]) -> dict[int, list[dict]]:
    result = {seed: [] for seed in p.MODEL_SEEDS}
    for model_seed in p.MODEL_SEEDS:
        by_instance = {
            int(row["instance_seed"]): row
            for row in rows
            if row["method"] == "e191"
            and int(row["model_seed"]) == model_seed
        }
        for seed in scenario.TEST_SEEDS:
            row = by_instance.get(seed)
            result[model_seed].append({
                "method": "e191",
                "model_seed": model_seed,
                "instance_seed": seed,
                "strict_complete": bool(row and row["strict_complete"]),
                "metrics": None if row is None else _flat(row),
            })
    return result


def _summary(points: list[dict]) -> dict:
    complete = [row for row in points if row["strict_complete"]]
    return {
        "strict_complete_instances": len(complete),
        "required_instances": len(points),
        "strict_completion_rate": len(complete) / len(points) if points else None,
        "metrics": (
            e192._average_metrics([row["metrics"] for row in complete])
            if complete and len(complete) == len(points) else None
        ),
    }


def _by_workload(points: list[dict]) -> dict:
    return {
        scenario.level_label(level): _summary([
            row for row in points
            if scenario.level_for_seed(row["instance_seed"]) == level
        ])
        for level in scenario.LEVELS
    }


def _comparison(vcg: list[dict], baseline: list[dict]) -> dict:
    left = {row["instance_seed"]: row for row in vcg}
    right = {row["instance_seed"]: row for row in baseline}
    pairs = []
    for seed in scenario.TEST_SEEDS:
        eligible = bool(left[seed]["strict_complete"] and right[seed]["strict_complete"])
        delta = None
        if eligible:
            delta = {
                name: float(left[seed]["metrics"][name])
                - float(right[seed]["metrics"][name])
                for name in e192.FLAT_METRICS
                if left[seed]["metrics"].get(name) is not None
                and right[seed]["metrics"].get(name) is not None
            }
        pairs.append({
            "instance_seed": seed,
            "workload": scenario.level_label(scenario.level_for_seed(seed)),
            "cost_pair_eligible": eligible,
            "delta_e191_minus_baseline": delta,
        })
    eligible = [row for row in pairs if row["cost_pair_eligible"]]
    means = {}
    if len(eligible) == len(scenario.TEST_SEEDS):
        for name in e192.FLAT_METRICS:
            values = [row["delta_e191_minus_baseline"].get(name) for row in eligible]
            if all(value is not None for value in values):
                means[name] = float(fmean(values))
    return {
        "difference_direction": "E191 minus baseline; negative cost favors E191",
        "paired_instance_count": len(eligible),
        "mean_paired_deltas": means,
        "pairs": pairs,
    }


def _format(value) -> str:
    return "—" if value is None else f"{float(value):.3f}"


def analyze(output: Path) -> dict:
    output = output.resolve()
    contract, _ = authenticate(output)
    rows = _saved_rows(output)
    baselines = _baseline_points(rows)
    vcg = _vcg_points(rows)
    baseline_summaries = {
        method: _summary(points) for method, points in baselines.items()
    }
    vcg_summaries = {str(seed): _summary(points) for seed, points in vcg.items()}
    pooled_vcg = [row for seed in p.MODEL_SEEDS for row in vcg[seed]]
    complete = len(rows) == p.EXPECTED_ROWS
    comparisons = {
        method: {
            str(seed): _comparison(vcg[seed], baselines[method])
            for seed in p.MODEL_SEEDS
        }
        for method in baselines
    }
    report = {
        "protocol": p.PROTOCOL,
        "contract_sha256": contract["contract_sha256"],
        "status": "complete" if complete else "partial",
        "expected_rows": p.EXPECTED_ROWS,
        "observed_rows": len(rows),
        "baseline_summaries": baseline_summaries,
        "baseline_by_workload": {
            method: _by_workload(points) for method, points in baselines.items()
        },
        "e191_selected_summaries": vcg_summaries,
        "e191_selected_by_workload": {
            str(seed): _by_workload(vcg[seed]) for seed in p.MODEL_SEEDS
        },
        "e191_selected_pooled": _summary(pooled_vcg),
        "comparisons": comparisons,
        "ga_aggregation_unit": "four nuisance replicates within instance",
        "training_runs": 0,
        "development_only": False,
        "test_panel_opened": bool(rows),
        "paper_evidence": complete,
    }
    _write(output / p.REPORT_NAME, report)
    lines = [
        "# E203 — E191 versus matched 8x8 heuristics on the test panel",
        "",
        "Completion precedes cost. GA is averaged within each instance.",
        "",
        "| Method | Complete | J | Storage MAE | Storage ±20 | Wait | Rehandles/100 | Peak |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    table_rows = [
        (method, baseline_summaries[method]) for method in ("dynamic_pslap", "ga")
    ]
    table_rows += [
        (f"E191 seed {seed}", vcg_summaries[str(seed)]) for seed in p.MODEL_SEEDS
    ]
    table_rows.append(("E191 pooled", _summary(pooled_vcg)))
    for label, summary in table_rows:
        metrics = summary["metrics"] or {}
        lines.append(
            f"| {label} | {summary['strict_complete_instances']}/"
            f"{summary['required_instances']} | "
            + " | ".join(
                _format(metrics.get(name))
                for name in (
                    "J",
                    "storage_clock_mae",
                    "storage_within_20_rate",
                    "mean_admission_wait",
                    "rehandles_per_100",
                    "peak_stored",
                )
            )
            + " |"
        )
    lines += [
        "",
        "J = 2T_storage + E_storage + 0.5W + 0.1L + 10R.",
        "Thirty previously unused instances: ten at each workload.",
        "All E191 checkpoints were selected and frozen before this panel opened.",
    ]
    (output / p.TABLE_NAME).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "status": report["status"],
        "observed_rows": len(rows),
        "expected_rows": p.EXPECTED_ROWS,
        "report": str(output / p.REPORT_NAME),
        "table": str(output / p.TABLE_NAME),
        "training_runs": 0,
        "episode_runs": 0,
        "test_panel_opened": bool(rows),
        "paper_evidence": complete,
    }


def self_check(output: Path) -> dict:
    prepared = prepare(output)
    contract, manifest = authenticate(output)
    all_prior = set(e191.VALIDATION_SEEDS[scenario.SCENARIO])
    all_prior.update(
        e191.TRAIN_BASE[scenario.SCENARIO] + index
        for index in range(e191.TOTAL_EPISODES)
    )
    all_prior.update(e191_scenario.PILOT_SEEDS)
    require(
        not set(scenario.TEST_SEEDS) & all_prior,
        "E203 test seeds overlap E191 training or development",
    )
    by_replicate = {}
    for record in manifest["records"]:
        seed = int(record["seed"])
        instance = EpisodeInstance.from_json(record["instance_json"])
        replicate = scenario.replicate_for_seed(seed)
        durations = tuple(instance.storage_steps_needed)
        if replicate in by_replicate:
            require(by_replicate[replicate] == durations, "test durations differ")
        else:
            by_replicate[replicate] = durations
        level = scenario.level_for_seed(seed)
        require(
            instance.arrival_steps
            == tuple(
                int(scenario.GAPS[level]) * index
                for index in range(scenario.TOTAL_JOBS)
            ),
            "E203 test arrival schedule differs",
        )
    freezes = {
        str(seed): {
            "selected_episode": int(_freeze(seed)["selected_episode"]),
            "checkpoint_sha256": _freeze(seed)["frozen_checkpoint_sha256"],
        }
        for seed in p.MODEL_SEEDS
    }
    return {
        **prepared,
        "status": "self_check_passed",
        "contract_sha256": contract["contract_sha256"],
        "duration_stream_pairing": True,
        "training_development_seed_disjoint": True,
        "frozen_models": freezes,
        "test_panel_opened": False,
    }


def _coordinates(command: str, method: str | None, model_seed: int | None):
    if command == "run-one":
        raise RuntimeError("run-one coordinates are handled directly")
    if command == "run-method":
        require(method is not None, "run-method requires --method")
        if method == "e191":
            seeds = p.MODEL_SEEDS if model_seed is None else (model_seed,)
            return tuple(
                ("e191", seed, instance_seed, 0)
                for seed in seeds for instance_seed in scenario.TEST_SEEDS
            )
        return tuple(
            (method, None, seed, replicate)
            for seed in scenario.TEST_SEEDS for replicate in p.replicates(method)
        )
    return tuple(
        (method, None, seed, replicate)
        for method in ("dynamic_pslap", "ga")
        for seed in scenario.TEST_SEEDS for replicate in p.replicates(method)
    ) + tuple(
        ("e191", model_seed, seed, 0)
        for model_seed in p.MODEL_SEEDS for seed in scenario.TEST_SEEDS
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=(
            "prepare",
            "self-check",
            "run-one",
            "run-method",
            "run-all",
            "analyze",
            "status",
        ),
    )
    parser.add_argument("--output", type=Path, default=p.DEFAULT_OUTPUT)
    parser.add_argument("--method", choices=p.METHODS)
    parser.add_argument("--model-seed", type=int, choices=p.MODEL_SEEDS)
    parser.add_argument("--seed", type=int, choices=scenario.TEST_SEEDS)
    parser.add_argument("--replicate", type=int, default=0)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
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
            coordinate = (
                args.method,
                args.model_seed if args.method == "e191" else None,
                args.seed,
                args.replicate,
            )
            if args.method == "e191":
                require(args.model_seed is not None, "E191 requires --model-seed")
            coordinates = (coordinate,)
        else:
            coordinates = _coordinates(args.command, args.method, args.model_seed)
        rows = []
        for method, model_seed, seed, replicate in coordinates:
            if method == "e191":
                rows.append(run_vcg_one(output, model_seed, seed, args.device))
            else:
                rows.append(run_baseline_one(output, method, seed, replicate))
        result = {
            "status": "complete",
            "rows": len(rows),
            "fresh_episode_runs": sum(int(row.get("episode_runs", 0)) for row in rows),
            "training_runs": 0,
            "analysis": analyze(output),
        }
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False, default=str))


if __name__ == "__main__":
    main()

