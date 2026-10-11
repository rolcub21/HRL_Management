#!/usr/bin/env python3
"""Held-out E28-style comparison for target-size-trained VCG checkpoints.

The test gate requires every planned checkpoint to be frozen first. No command
in this module trains a network or changes an existing E28 result.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import ExitStack
import json
from pathlib import Path
from statistics import fmean
from time import perf_counter
from unittest.mock import patch

import torch

import benchmark_viability_critic_priority as benchmark
from experiments.conditioned_vcg.target_size_vcg_training import protocol as p
from experiments.conditioned_vcg.target_size_vcg_training import run as training
from experiments.conditioned_vcg.E13_operational_scalability_95k import program as e13
from experiments.conditioned_vcg.E14_certification_scalability_95k import reuse
from experiments.conditioned_vcg.E25_scale_baseline_transfer_95k import run as e25
from experiments.conditioned_vcg.E27_certified_dynamic_sustained_99k import run as e27
from experiments.conditioned_vcg.E28_empty_start_scale_baselines_100k import adapter
from experiments.conditioned_vcg.E28_empty_start_scale_baselines_100k import run as e28
from PSLAP.relocation_family_certification import RELOCATION_FAMILY_CERTIFICATION
import run_vcg_v11_conditioned_handling_seed0_85k as conditioned_seed0
import run_vcg_v11_nested_handling_pilot as pilot
from vcg_v11_conditioned_handling import ConditionedHandlingAgent

METHODS = (e28.VCG, e28.DYNAMIC, e28.GA, e28.A3C)
BASELINE_REPLICATES = e25.REPLICATES
METRICS = (
    "mean_absolute_error", "within_target_window_rate",
    "release_mean_absolute_error", "release_within_target_window_rate",
    "mean_pre_storage_flow_steps",
    "physical_rehandles_per_100_required_deliveries", "steps_per_delivery",
)


def _two_clock_metrics(env, *, strict: bool, required: int) -> dict:
    empty = {
        "release_mean_absolute_error": None,
        "release_within_target_window_rate": None,
        "mean_pre_storage_flow_steps": None,
        "p95_pre_storage_flow_steps": None,
        "timing_identity_verified": False,
    }
    if not strict:
        return empty
    if env is None or len(env.blocks) != required:
        raise RuntimeError("strict episode has no complete timing state")
    waits, release_errors = [], []
    for block in env.blocks:
        if (block.stored_time_step is None or block.delivered_time_step is None
                or not block.delivered):
            raise RuntimeError("strict episode omitted a block timing event")
        arrival = int(block.arrival_step)
        stored = int(block.stored_time_step)
        delivered = int(block.delivered_time_step)
        duration = int(block.storage_steps_needed)
        wait = stored - arrival
        placement_error = delivered - stored - duration
        release_error = delivered - arrival - duration
        if (wait < 0 or release_error != placement_error + wait
                or float(block.delivery_error_time) != float(placement_error)):
            raise RuntimeError("two-clock timing identity failed")
        waits.append(wait)
        release_errors.append(release_error)
    sorted_waits = sorted(waits)
    rank = 0.95 * (len(sorted_waits) - 1)
    lower = int(rank)
    fraction = rank - lower
    p95 = sorted_waits[lower] * (1 - fraction) + sorted_waits[min(lower + 1, len(sorted_waits) - 1)] * fraction
    return {
        "release_mean_absolute_error": fmean(abs(value) for value in release_errors),
        "release_within_target_window_rate": fmean(abs(value) <= 20 for value in release_errors),
        "mean_pre_storage_flow_steps": fmean(waits),
        "p95_pre_storage_flow_steps": p95,
        "timing_identity_verified": True,
    }


def _gate_path(output: Path) -> Path:
    return output / "test-gate.json"


def _terminal_hashes(output: Path) -> dict:
    result = {}
    contract = training._read_contract(output)
    for name in p.SCENARIOS:
        for model_seed in p.MODEL_SEEDS:
            _, _, base_sha, base_digest = training._load_base(
                output, name, model_seed, torch.device("cpu")
            )
            path = training._stage_b_path(output, name, model_seed) / "terminal.pth"
            if not path.is_file():
                raise RuntimeError(f"missing terminal handling checkpoint: {path}")
            payload = torch.load(path, map_location="cpu", weights_only=False)
            required = {
                "training_protocol": training.STAGE_B_PROTOCOL,
                "role": "terminal", "target_contract_sha256": contract["contract_sha256"],
                "scenario": name, "model_seed": model_seed,
                "base_checkpoint_sha256": base_sha,
                "base_policy_digest": base_digest,
                "completed_rounds": p.HANDLING_ROUNDS,
            }
            if any(payload.get(key) != value for key, value in required.items()):
                raise RuntimeError(f"handling terminal is not eligible: {path}")
            key = f"{name}/seed-{model_seed}"
            result[key] = {"base_sha256": base_sha, "handling_sha256": p.file_sha256(path)}
    return result


def freeze_test(output: Path) -> dict:
    contract = training._read_contract(output)
    hashes = _terminal_hashes(output)
    payload = {
        "protocol": p.PROTOCOL, "contract_sha256": contract["contract_sha256"],
        "checkpoint_sha256": hashes,
        "test_seeds": {key: list(value) for key, value in p.TEST_SEEDS.items()},
        "test_opened_after_all_checkpoints_frozen": True,
    }
    gate = {**payload, "gate_sha256": p.digest(payload)}
    path = _gate_path(output)
    if path.is_file():
        if json.loads(path.read_text(encoding="utf-8")) != gate:
            raise RuntimeError("existing test gate or frozen checkpoints changed")
    else:
        e13._atomic_json(path, gate)
    return {"status": "frozen", "gate_sha256": gate["gate_sha256"],
            "target_models": len(hashes), "test_instances": sum(map(len, p.TEST_SEEDS.values()))}


def _read_gate(output: Path) -> dict:
    path = _gate_path(output)
    if not path.is_file():
        raise RuntimeError("freeze-test is required before a test-panel run")
    saved = json.loads(path.read_text(encoding="utf-8"))
    payload = {key: value for key, value in saved.items() if key != "gate_sha256"}
    if saved.get("gate_sha256") != p.digest(payload):
        raise RuntimeError("test gate digest mismatch")
    contract = training._read_contract(output)
    if saved["contract_sha256"] != contract["contract_sha256"]:
        raise RuntimeError("test gate uses another contract")
    if saved["checkpoint_sha256"] != _terminal_hashes(output):
        raise RuntimeError("a frozen target-size checkpoint changed")
    return saved


def _identity(method: str, name: str, seed: int, replicate: int, panel: str) -> dict:
    seeds = p.PILOT_SEEDS[name] if panel == "pilot" else p.TEST_SEEDS[name]
    coordinate = tuple(p.SCENARIOS).index(name) * len(seeds) + seeds.index(seed)
    if method == e28.DYNAMIC:
        return {"replicate": 0, "model_seed": None, "rng_index": None, "rng_seed": None}
    if method == e28.GA:
        rng = e25.GA_RNG_BASE + e25.GA_RNG_STRIDE * (4 * coordinate + replicate)
        return {"replicate": replicate, "model_seed": None,
                "rng_index": replicate, "rng_seed": rng}
    if method == e28.A3C:
        model_seed, stream = divmod(replicate, 5)
        rng = e25.A3C_POLICY_RNG_BASE + 1000 * model_seed + 10 * coordinate + stream
        return {"replicate": replicate, "model_seed": model_seed,
                "rng_index": stream, "rng_seed": rng}
    raise ValueError(method)


def _vcg_row(output: Path, name: str, seed: int, model_seed: int,
             preference_lambda: float) -> dict:
    scenario = p.scenario_for(name)
    instance = p.instance_for(name, seed)
    base_payload, _, base_sha, base_digest = training._load_base(
        output, name, model_seed, torch.device("cpu")
    )
    path = training._stage_b_path(output, name, model_seed) / "terminal.pth"
    terminal = torch.load(path, map_location="cpu", weights_only=False)
    if (terminal.get("base_checkpoint_sha256") != base_sha
            or terminal.get("base_policy_digest") != base_digest):
        raise RuntimeError("Stage B terminal is not bound to Stage A")
    agents, envs = [], []

    def agent_factory(base):
        agent = ConditionedHandlingAgent.from_checkpoint(
            terminal["agent_checkpoint"], base_agent=base,
            expected_base_checkpoint_sha256=base_sha,
            expected_base_policy_digest=base_digest,
            expected_source_cost_sha256=training.NO_SOURCE_COST,
            seed=model_seed,
        )
        agent.set_epsilon(0.0)
        wrapped = e27.FrontierRecordAgent(
            conditioned_seed0._FixedLambdaAgent(agent, preference_lambda)
        )
        agents.append(wrapped)
        return wrapped

    def env_factory(_payload):
        env = adapter.AuditedEmptyStartEnv(scenario)
        envs.append(env)
        return env

    raw, error, failure_class = None, None, None
    started = perf_counter()
    try:
        cache = reuse.TimingInvariantCertificateCache()
        with ExitStack() as stack:
            stack.enter_context(e13._WallClock(e28.WALL_SECONDS))
            stack.enter_context(reuse.path_cleanup_active())
            stack.enter_context(patch.object(benchmark, "_make_env", env_factory))
            stack.enter_context(pilot._agent_factory(agent_factory))
            stack.enter_context(patch.object(benchmark, "ViabilityCertificateCache", lambda: cache))
            raw = benchmark.run_arm(
                arm=benchmark.EXACT_FULL, controller_payload=base_payload,
                instance=instance, instance_seed=seed,
                search_config=benchmark._search_config(base_payload),
                liveness_rule=benchmark._liveness_rule(base_payload),
                prioritizer=None, max_steps=scenario.max_steps,
                device=torch.device("cpu"),
                recovery_certification_strategy=RELOCATION_FAMILY_CERTIFICATION,
            )
    except e13.EpisodeWallLimit as exc:
        failure_class, error = "censored_wall_clock", str(exc)
    except Exception as exc:
        failure_class, error = "unsupported_or_implementation_error", f"{type(exc).__name__}: {exc}"
    elapsed = perf_counter() - started
    strict = bool(
        raw is not None and raw["strict_method_success"] and raw["terminal"]
        and raw["method_failure_reason"] is None
        and raw["complete_frontier_exactly_verified"]
        and raw["illegal_drops"] == 0 and raw["macro_failures"] == 0
        and len(raw["delivery_deviations"]) == scenario.total_jobs
    )
    frontiers = () if raw is None else raw["frontiers"]
    if failure_class is None:
        if strict:
            failure_class = "completed"
        elif any(item.get("unknown_candidates_rejected", 0) > 0
                 or item.get("safe_candidate_count", 0) == 0 for item in frontiers):
            failure_class = "certificate_budget_limited"
        else:
            failure_class = "valid_but_operationally_incomplete"
    timing = (
        e13.d10._timing(raw["delivery_deviations"], expected_deliveries=scenario.total_jobs)
        if strict else {"mean_absolute_error": None, "within_target_window_rate": None}
    )
    rehandles = None if raw is None else int(raw.get("physical_storage_relocations", raw.get("relocations", 0)))
    env = envs[0] if len(envs) == 1 else None
    if raw is not None and (env is None or env.warm_start_audit is None):
        raise RuntimeError("target VCG reset was not authenticated as empty")
    decisions = () if raw is None else raw["decisions"]
    result = {
        "method": e28.VCG, "model_seed": model_seed,
        "preference_lambda": preference_lambda, "strict_safe_complete": strict,
        "failure_class": failure_class,
        "method_failure_reason": None if raw is None else raw["method_failure_reason"],
        "error": error, "deliveries_at_stop": None if raw is None else len(raw["delivery_deviations"]),
        "required_deliveries": scenario.total_jobs,
        "steps_at_stop": None if raw is None else int(raw["steps"]),
        "physical_rehandles_at_stop": rehandles,
        "mean_absolute_error": timing["mean_absolute_error"],
        "within_target_window_rate": timing["within_target_window_rate"],
        "physical_rehandles_per_100_required_deliveries": (
            100.0 * rehandles / scenario.total_jobs if strict else None
        ),
        "steps_per_delivery": raw["steps"] / scenario.total_jobs if strict else None,
        "initial_state_audit": None if env is None else env.warm_start_audit,
        "occupancy": None if env is None or not env.occupancy_trace else env.occupancy_summary(),
        "first_frontier_sha256": None if not agents else agents[0].first_frontier_sha256,
        "guard_forced_decisions": sum(bool(item["liveness_forced"]) for item in decisions),
        "decision_count": len(decisions),
        "unknown_candidate_rejections": sum(item.get("unknown_candidates_rejected", 0) for item in frontiers),
        "admission_successes": sum(item["selected_action_type"] == "accept" and item["option_success"] for item in decisions),
        "episode_wall_seconds": elapsed,
    }
    result.update(_two_clock_metrics(env, strict=strict, required=scenario.total_jobs))
    return result


def _ledger_path(output: Path, panel: str, method: str, name: str,
                 seed: int, replicate: int, model_seed: int, value: float) -> Path:
    label = (f"model-{model_seed}-lambda-{value:.3f}" if method == e28.VCG
             else f"replicate-{replicate}")
    return output / "run-ledger" / panel / method / name / f"seed-{seed}-{label}.json"


def run_one(output: Path, panel: str, method: str, name: str, seed: int,
            replicate: int, model_seed: int, value: float) -> dict:
    contract = training._read_contract(output)
    if method not in METHODS or name not in p.SCENARIOS or panel not in ("pilot", "test"):
        raise ValueError("unsupported evaluation coordinate")
    permitted = p.PILOT_SEEDS[name] if panel == "pilot" else p.TEST_SEEDS[name]
    if seed not in permitted:
        raise ValueError("instance seed is outside the declared panel")
    if method == e28.VCG:
        if model_seed not in p.MODEL_SEEDS or value not in p.PREFERENCES:
            raise ValueError("unsupported target VCG model seed or preference")
        if panel == "pilot" and (model_seed, value) != (0, 0.10):
            raise ValueError("pilot freezes a single VCG coordinate")
    elif replicate not in BASELINE_REPLICATES[method]:
        raise ValueError("unsupported baseline replicate")
    if panel == "pilot" and method != e28.VCG and replicate != 0:
        raise ValueError("pilot freezes one baseline replicate")
    gate = _read_gate(output) if panel == "test" else None
    path = _ledger_path(output, panel, method, name, seed, replicate, model_seed, value)
    if path.is_file():
        ledger = json.loads(path.read_text(encoding="utf-8"))
        payload = {k: v for k, v in ledger.items() if k != "ledger_sha256"}
        if (ledger.get("ledger_sha256") != p.digest(payload)
                or ledger.get("contract_sha256") != contract["contract_sha256"]
                or ledger.get("gate_sha256") != (None if gate is None else gate["gate_sha256"])):
            raise RuntimeError("existing evaluation ledger is not authenticated")
        return ledger["row"]
    instance = p.instance_for(name, seed)
    if method == e28.VCG:
        row = _vcg_row(output, name, seed, model_seed, value)
    else:
        created = []
        original_factory = adapter.empty_start_factory

        def captured_factory(*args, **kwargs):
            env = original_factory(*args, **kwargs)
            created.append(env)
            return env

        with patch.object(adapter, "empty_start_factory", captured_factory):
            row = e28._run_baseline(
                method, p.scenario_for(name), instance, seed,
                _identity(method, name, seed, replicate, panel),
            )
        if len(created) > 1:
            raise RuntimeError("baseline created more than one yard environment")
        row.update(_two_clock_metrics(
            created[0] if created else None,
            strict=bool(row["strict_safe_complete"]),
            required=p.scenario_for(name).total_jobs,
        ))
    row.update({"protocol": p.PROTOCOL, "panel": panel, "scenario_id": name,
                "instance_seed": seed, "instance_id": instance.instance_id,
                "schedule_id": instance.schedule_id})
    ledger_payload = {
        "contract_sha256": contract["contract_sha256"],
        "gate_sha256": None if gate is None else gate["gate_sha256"],
        "row": row,
    }
    e13._atomic_json(path, {**ledger_payload, "ledger_sha256": p.digest(ledger_payload)})
    return row


def analyze(output: Path, panel: str) -> dict:
    contract = training._read_contract(output)
    gate = _read_gate(output) if panel == "test" else None
    seeds_by_name = p.PILOT_SEEDS if panel == "pilot" else p.TEST_SEEDS
    observed = []
    missing = []
    for name, seeds in seeds_by_name.items():
        for seed in seeds:
            coordinates = [(e28.VCG, 0, model_seed, value)
                           for model_seed in (p.MODEL_SEEDS if panel == "test" else (0,))
                           for value in (p.PREFERENCES if panel == "test" else (0.10,))]
            coordinates += [(method, rep, 0, 0.0)
                            for method in METHODS[1:]
                            for rep in (BASELINE_REPLICATES[method] if panel == "test" else (0,))]
            for method, replicate, model_seed, value in coordinates:
                path = _ledger_path(output, panel, method, name, seed, replicate, model_seed, value)
                if not path.is_file():
                    missing.append(str(path.relative_to(output)))
                    continue
                ledger = json.loads(path.read_text(encoding="utf-8"))
                payload = {k: v for k, v in ledger.items() if k != "ledger_sha256"}
                if (ledger.get("ledger_sha256") != p.digest(payload)
                        or ledger.get("contract_sha256") != contract["contract_sha256"]
                        or ledger.get("gate_sha256") != (None if gate is None else gate["gate_sha256"])):
                    raise RuntimeError(f"unauthenticated evaluation ledger: {path}")
                observed.append(ledger["row"])
    groups = defaultdict(list)
    for row in observed:
        key = (row["scenario_id"], row["method"], row.get("preference_lambda"))
        groups[key].append(row)
    summaries = []
    for (name, method, value), rows in sorted(groups.items(), key=lambda item: str(item[0])):
        per_instance = defaultdict(list)
        for row in rows:
            per_instance[row["instance_seed"]].append(row)
        quality = {}
        for metric in METRICS:
            means = [fmean(float(row[metric]) for row in reps if row[metric] is not None)
                     for reps in per_instance.values()
                     if any(row[metric] is not None for row in reps)]
            quality[metric] = None if not means else fmean(means)
        summaries.append({
            "scenario": name, "method": method, "preference_lambda": value,
            "observed_runs": len(rows),
            "strict_complete_runs": sum(bool(row["strict_safe_complete"]) for row in rows),
            "instances": len(per_instance), "quality_conditional_on_completion": quality,
            "mean_wall_seconds": fmean(float(row["episode_wall_seconds"]) for row in rows),
            "failures": {failure: sum(row["failure_class"] == failure for row in rows)
                         for failure in sorted({row["failure_class"] for row in rows})},
        })
    report = {
        "status": "complete" if not missing else "partial",
        "panel": panel, "contract_sha256": contract["contract_sha256"],
        "gate_sha256": None if gate is None else gate["gate_sha256"],
        "observed_rows": len(observed), "missing_rows": len(missing),
        "missing_coordinates": missing, "summaries": summaries,
        "qualification": "Target-size VCG training is in-domain; A3C is a frozen 5x5-trained control. Both placement- and release-clock quality are conditional on strict completion.",
    }
    e13._atomic_json(output / f"{panel}-report.json", report)
    return {"status": report["status"], "panel": panel,
            "observed_rows": len(observed), "missing_rows": len(missing),
            "report": str(output / f"{panel}-report.json")}


def run_panel(output: Path, panel: str) -> dict:
    training._read_contract(output)
    if panel == "test":
        _read_gate(output)
    completed = 0
    for name in p.SCENARIOS:
        seeds = p.PILOT_SEEDS[name] if panel == "pilot" else p.TEST_SEEDS[name]
        for seed in seeds:
            model_seeds = (0,) if panel == "pilot" else p.MODEL_SEEDS
            preferences = (0.10,) if panel == "pilot" else p.PREFERENCES
            for model_seed in model_seeds:
                for value in preferences:
                    row = run_one(output, panel, e28.VCG, name, seed, 0, model_seed, value)
                    completed += 1
                    print(f"{panel} {completed}: VCG {name} seed={seed} model={model_seed} lambda={value} strict={int(row['strict_safe_complete'])}", flush=True)
                    if row["failure_class"] == "unsupported_or_implementation_error":
                        raise RuntimeError("implementation error; panel stopped after retaining the failing ledger")
            for method in METHODS[1:]:
                replicates = (0,) if panel == "pilot" else BASELINE_REPLICATES[method]
                for replicate in replicates:
                    row = run_one(output, panel, method, name, seed, replicate, 0, 0.0)
                    completed += 1
                    print(f"{panel} {completed}: {method} {name} seed={seed} replicate={replicate} strict={int(row['strict_safe_complete'])}", flush=True)
                    if row["failure_class"] == "unsupported_or_implementation_error":
                        raise RuntimeError("implementation error; panel stopped after retaining the failing ledger")
    return analyze(output, panel)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze-test", "run-one", "run-panel", "analyze"))
    parser.add_argument("--output", type=Path, default=p.DEFAULT_OUTPUT)
    parser.add_argument("--panel", choices=("pilot", "test"), default="pilot")
    parser.add_argument("--scenario", choices=tuple(p.SCENARIOS))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--method", choices=METHODS, default=e28.VCG)
    parser.add_argument("--replicate", type=int, default=0)
    parser.add_argument("--model-seed", type=int, default=0)
    parser.add_argument("--lambda", dest="preference_lambda", type=float, default=0.10)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output = args.output.resolve()
    if args.command == "freeze-test":
        result = freeze_test(output)
    elif args.command == "run-one":
        if args.scenario is None or args.seed is None:
            raise SystemExit("run-one requires --scenario and --seed")
        result = run_one(output, args.panel, args.method, args.scenario,
                         args.seed, args.replicate, args.model_seed,
                         args.preference_lambda)
    elif args.command == "run-panel":
        result = run_panel(output, args.panel)
    else:
        result = analyze(output, args.panel)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
