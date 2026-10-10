#!/usr/bin/env python3
"""E28: empty-start, own-placement scale comparison against frozen baselines."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import replace
import json
from pathlib import Path
from statistics import fmean
from time import perf_counter
from unittest.mock import patch

import torch

import benchmark_viability_critic_priority as benchmark
from example.episode_instance import EpisodeInstance
from experiments.conditioned_vcg.E13_operational_scalability_95k import program as e13
from experiments.conditioned_vcg.E14_certification_scalability_95k import reuse
from experiments.conditioned_vcg.E25_scale_baseline_transfer_95k import run as e25
from experiments.conditioned_vcg.E25_v2_full_vcg_family_scale_95k import run as e25v2
from experiments.conditioned_vcg.E25_v3_matched_warm_start_95k import run as v3
from experiments.conditioned_vcg.E25_v4_reconfiguration_baselines_95k import controller as v4
from experiments.conditioned_vcg.E27_certified_dynamic_sustained_99k import run as e27
from experiments.conditioned_vcg.E28_empty_start_scale_baselines_100k import adapter, render
from experiments.conditioned_vcg.development.D10_scalability_support_screen import (
    occupancy_extension as occupancy,
)
import run_vcg_conditioned_final_comparison_90k as final90
import run_vcg_v11_conditioned_handling_seed0_85k as conditioned_seed0
import run_vcg_v11_nested_handling_pilot as pilot
from PSLAP.relocation_family_certification import RELOCATION_FAMILY_CERTIFICATION


ROOT = Path(__file__).resolve().parents[3]
OUTPUT = ROOT / "results/vcg-conditioned-e28-empty-start-scale-baselines-100k-v2"
PROTOCOL = "vcg_conditioned_e28_empty_start_scale_baselines_100k_v2"
CONTRACT_NAME = "e28-contract.json"
DEV_MANIFEST = "e28-development-manifest.json"
CONFIRMATION_MANIFEST = "e28-confirmation-manifest.json"
GATE_NAME = "e28-pilot-gate.json"
REPORT_NAME = "e28-report.json"
TABLE_NAME = "e28-results-table.md"
DEV_SEEDS = (79_200,)
SEEDS = (100_000, 100_001, 100_002)
LAMBDAS = e25v2.LAMBDAS
PILOT_LAMBDA = 0.10
PILOT_VCG_REPLICATE = LAMBDAS.index(PILOT_LAMBDA)
VCG = "vcg"
DYNAMIC = e25.METHOD_DYNAMIC
GA = e25.METHOD_GA
A3C = e25.METHOD_A3C
METHODS = (VCG, DYNAMIC, GA, A3C)
REPLICATES = {VCG: tuple(range(len(LAMBDAS))), **e25.REPLICATES}
SCENARIOS = (
    occupancy.Scenario("empty_5x5_n8", "empty_start_sustained", "emergent", 5, 5, 0, 8),
    occupancy.Scenario("empty_8x8_n36", "empty_start_sustained", "emergent", 8, 8, 0, 36),
    occupancy.Scenario("empty_10x10_n64", "empty_start_sustained", "emergent", 10, 10, 0, 64),
)
SCENARIO_BY_ID = {item.scenario_id: item for item in SCENARIOS}
PILOT_SCENARIO = SCENARIOS[0].scenario_id
WALL_SECONDS = 1800


class E28Error(RuntimeError):
    pass


def _hashed(value, field):
    return e27._hash_dict(value, field)


def _read(path, field):
    return e27._read_hashed(path, field)


def _write(path, value):
    e13._atomic_json(path, value)


def _inputs():
    parent_contract, conditioned_inputs = e27._parent()
    a3c_completions, _ = e25._a3c_inputs()
    return parent_contract, conditioned_inputs, a3c_completions


def _contract_payload(parent_contract, conditioned_inputs, a3c_completions):
    paths = (
        Path(__file__).resolve(), Path(adapter.__file__).resolve(),
        Path(render.__file__).resolve(),
        Path(v4.__file__).resolve(), Path(v3.__file__).resolve(),
        Path(e25.__file__).resolve(), Path(e25v2.__file__).resolve(),
        Path(e27.__file__).resolve(),
        Path(occupancy.__file__).resolve(),
        ROOT / "benchmark_viability_critic_priority.py",
        ROOT / "example/urgency_scheduler.py",
        ROOT / "example/Options/RetrieveDeliverOption.py",
        ROOT / "example/Options/StrategicDeferOption.py",
        ROOT / "example/small_rooms_env.py",
        ROOT / "PSLAP/retrieval_executor.py",
    )
    return {
        "schema_version": 1, "protocol": PROTOCOL, "post_hoc": True,
        "question": "integrated quality of frozen online methods on yards they populate themselves",
        "methods": list(METHODS),
        "replicates": {method: list(REPLICATES[method]) for method in METHODS},
        "development_seeds": list(DEV_SEEDS),
        "confirmation_seeds": list(SEEDS),
        "scenarios": [item.public_dict() for item in SCENARIOS],
        "all_jobs_arrive": True,
        "arrival_steps": "10*j, j=0,...,N-1; first block at pickup at reset, storage empty",
        "storage_durations": "Poisson(80), seeded by EpisodeInstance",
        "initial_storage_occupancy": 0,
        "occupancy_definition": "occupied usable storage slots / usable storage capacity",
        "baseline_adapter": adapter.WaitEnabledReconfigurationScheduler.POLICY,
        "baseline_retrieval": "shared strict retrieval with legal blocker relocation; optional standalone reconfigure; bounded defer if blocked",
        "baseline_certification": "none; physical legality only",
        "vcg_certification": {
            "max_nodes": e27.SEARCH_MAX_NODES,
            "search_order": "goal_directed",
            "cache": "E14 timing-invariant",
            "path_cleanup": True,
            "strategy": RELOCATION_FAMILY_CERTIFICATION,
        },
        "vcg_model_seeds": [0], "vcg_preference_lambdas": list(LAMBDAS),
        "vcg_parent_contract_sha256": parent_contract["contract_sha256"],
        "vcg_terminal_sha256": {
            str(seed): value for seed, value
            in conditioned_inputs["conditioned"]["terminal_sha256"].items()
        },
        "a3c_checkpoint_sha256": {
            str(seed): item["selected_checkpoint_raw_sha256"]
            for seed, item in a3c_completions.items()
        },
        "wall_limit_seconds_per_episode": WALL_SECONDS,
        "max_steps_rule": "max(2000,100*N)",
        "quality_scope": "only strictly complete episodes; failures and censoring in completion denominator",
        "aggregation": "replications within instance, then equal-weight instances within yard size",
        "no_reused_control_rows": True,
        "training_runs": 0,
        "source_sha256": {str(path.relative_to(ROOT)): e27._sha(path) for path in paths},
    }


def _contract(output):
    parent, conditioned, a3c = _inputs()
    expected = _hashed(_contract_payload(parent, conditioned, a3c), "contract_sha256")
    observed = _read(output / CONTRACT_NAME, "contract_sha256")
    if observed != expected:
        raise E28Error("contract, parent checkpoint, or source changed")
    return observed, conditioned


def _manifest_payload(panel, seeds, contract):
    records = []
    for scenario in SCENARIOS:
        for seed in seeds:
            instance = _instance(scenario, seed)
            records.append({
                "scenario_id": scenario.scenario_id,
                "seed": seed,
                "instance_id": instance.instance_id,
                "schedule_id": instance.schedule_id,
                "instance_json": instance.to_json(),
            })
    return _hashed({
        "schema_version": 1, "protocol": PROTOCOL,
        "panel": panel, "contract_sha256": contract["contract_sha256"],
        "records": records,
    }, "manifest_sha256")


def _instance(scenario, seed):
    # EpisodeInstance requires the first release at t=0. Generate durations
    # using D10's frozen process, but do not warm-start any storage block.
    source = replace(scenario, initial_occupied_slots=1)
    sampled = occupancy._instance(source, seed)
    instance = replace(
        sampled,
        arrival_steps=tuple(occupancy.EXTRA_ARRIVAL_GAP * index
                            for index in range(scenario.total_jobs)),
    )
    instance.validate_for(scenario.make_base_env())
    return instance


def _manifest(output, panel, contract):
    filename = DEV_MANIFEST if panel == "development" else CONFIRMATION_MANIFEST
    observed = _read(output / filename, "manifest_sha256")
    expected = _manifest_payload(
        panel, DEV_SEEDS if panel == "development" else SEEDS, contract
    )
    if observed != expected:
        raise E28Error(f"{panel} instance manifest changed")
    return observed


def prepare(output):
    output = output.resolve()
    parent, conditioned, a3c = _inputs()
    expected = _hashed(_contract_payload(parent, conditioned, a3c), "contract_sha256")
    path = output / CONTRACT_NAME
    if path.is_file():
        if _read(path, "contract_sha256") != expected:
            raise E28Error("existing contract differs")
    else:
        _write(path, expected)
    development = _manifest_payload("development", DEV_SEEDS, expected)
    path = output / DEV_MANIFEST
    if path.is_file():
        if _read(path, "manifest_sha256") != development:
            raise E28Error("existing development manifest differs")
    else:
        _write(path, development)
    return {"status": "prepared", "development_instances": len(development["records"]),
            "confirmation_opened": (output / CONFIRMATION_MANIFEST).is_file(),
            "training_runs": 0}


def _coordinate(output, panel, method, scenario_id, seed, replicate):
    if (panel not in ("development", "confirmation")
        or method not in METHODS or scenario_id not in SCENARIO_BY_ID
        or seed not in (DEV_SEEDS if panel == "development" else SEEDS)
        or replicate not in REPLICATES[method]):
        raise E28Error("unsupported run coordinate")
    contract, inputs = _contract(output)
    manifest = _manifest(output, panel, contract)
    record = next(item for item in manifest["records"]
                  if item["scenario_id"] == scenario_id and item["seed"] == seed)
    instance = EpisodeInstance.from_json(record["instance_json"])
    scenario = SCENARIO_BY_ID[scenario_id]
    instance.validate_for(scenario.make_base_env())
    if (scenario.initial_occupied_slots != 0
        or instance.arrival_steps != tuple(occupancy.EXTRA_ARRIVAL_GAP * index
                                           for index in range(scenario.total_jobs))):
        raise E28Error("manifest no longer encodes an empty yard")
    return contract, inputs, scenario, instance


def _ledger_path(output, panel, method, scenario_id, seed, replicate):
    return output / "run-ledger" / panel / method / scenario_id / f"seed-{seed}-rep-{replicate}.json"


def _replicate_identity(method, scenario_id, seed, replicate, panel):
    coordinate = SCENARIOS.index(SCENARIO_BY_ID[scenario_id]) * (
        len(DEV_SEEDS) if panel == "development" else len(SEEDS)
    ) + (DEV_SEEDS if panel == "development" else SEEDS).index(seed)
    if method == DYNAMIC:
        return {"replicate": 0, "model_seed": None, "rng_index": None, "rng_seed": None}
    if method == GA:
        rng = e25.GA_RNG_BASE + e25.GA_RNG_STRIDE * (4 * coordinate + replicate)
        return {"replicate": replicate, "model_seed": None, "rng_index": replicate,
                "rng_seed": rng}
    if method == A3C:
        model_seed, rollout = divmod(replicate, 5)
        rng = e25.A3C_POLICY_RNG_BASE + 1_000 * model_seed + 10 * coordinate + rollout
        return {"replicate": replicate, "model_seed": model_seed, "rng_index": rollout,
                "rng_seed": rng}
    return {"replicate": replicate, "model_seed": 0, "rng_index": None,
            "rng_seed": None}


def _baseline_source(method, identity):
    if method == DYNAMIC:
        return None, e25.matched.DETERMINISTIC_METHOD_TO_SOURCE[
            e25.matched.DYNAMIC_METHOD
        ], None
    if method == GA:
        implementation = e25.ga_repair_v2._impl
        return None, implementation.REPAIRED_METHOD_TO_SOURCE[
            implementation.REPAIRED_2009_METHOD
        ], None
    _, checkpoints = e25._a3c_inputs()
    return checkpoints[int(identity["model_seed"])], e25.kim85.TRACK_A_KIM2020_A3C_SPATIAL, int(identity["rng_seed"])


def _run_baseline(method, scenario, instance, seed, identity):
    args = e25._runtime_args(
        scenario,
        ga_seed_base=(int(identity["rng_seed"]) - seed
                      if method == GA else e25.GA_RNG_BASE),
    )
    payload, assignment_source, policy_seed = _baseline_source(method, identity)
    created = []

    def factory(actual_scenario, **kwargs):
        if actual_scenario != scenario:
            raise E28Error("baseline received another yard scenario")
        env = adapter.empty_start_factory(actual_scenario, **kwargs)
        created.append(env)
        return env

    raw = None
    failure_class = None
    error = None
    started = perf_counter()
    try:
        with (e13._WallClock(WALL_SECONDS),
              patch.object(v3, "_warm_start_factory", side_effect=factory),
              patch.object(v4, "ReconfigurationScheduler",
                           adapter.WaitEnabledReconfigurationScheduler)):
            raw = v4.evaluate_one(
                args, seed, payload, instance=instance, scenario=scenario,
                assignment_source=assignment_source, policy_seed=policy_seed,
                deployment_mode=e25.kim85.DEPLOYMENT_STOCHASTIC,
            )
    except e13.EpisodeWallLimit as exc:
        failure_class, error = "censored_wall_clock", str(exc)
    except Exception as exc:
        failure_class, error = "unsupported_or_implementation_error", f"{type(exc).__name__}: {exc}"
    elapsed = perf_counter() - started
    if raw is not None:
        row = e25._normalize(
            raw, method=method, scenario=scenario, identity=identity, elapsed=elapsed
        )
        failure_class = row["failure_class"]
        if row["episode_instance_id"] != instance.instance_id or row["schedule_id"] != instance.schedule_id:
            raise E28Error("baseline lost paired EpisodeInstance")
    else:
        row = {
            "method": method, **identity,
            "strict_safe_complete": False, "failure_class": failure_class,
            "method_failure_reason": error,
            "deliveries_at_stop": None, "required_deliveries": scenario.total_jobs,
            "steps_at_stop": None, "physical_rehandles_at_stop": None,
            "mean_absolute_error": None, "within_target_window_rate": None,
            "physical_rehandles_per_100_required_deliveries": None,
            "steps_per_delivery": None, "episode_wall_seconds": elapsed,
        }
    env = created[0] if len(created) == 1 else None
    audit = None if env is None else env.warm_start_audit
    if raw is not None and (audit is None or audit["passed"] is not True):
        raise E28Error("baseline reset was not authenticated as empty")
    scheduler = {} if raw is None else raw["urgency_scheduler_audit"]
    decisions = scheduler.get("decisions", ())
    row.update({
        "error": error,
        "initial_state_audit": audit,
        "occupancy": None if env is None or not env.occupancy_trace
        else env.occupancy_summary(),
        "adapter_blocked_wait_decisions": sum(
            item.get("reason") == "blocked_inbound_defer" for item in decisions
        ),
        "adapter_standalone_reconfigurations": (
            None if raw is None else int(raw["standalone_reconfigurations"])
        ),
        "retrieval_blocker_relocations": (
            None if raw is None else int(raw["target_bound_obstruction_clearances"])
        ),
        "admission_successes": (
            None if raw is None else int(raw["scheduler_audit"]["inbound_successes"])
        ),
        "invalid_assignment_count": (
            None if raw is None else int(raw["selector_audit"]["invalid_assignment_count"])
        ),
        "fallback_count": (
            None if raw is None else int(raw["selector_audit"].get("fallback_count", 0))
        ),
        "illegal_drops": None if raw is None else int(raw["illegal_drops"]),
        "episode_wall_seconds": elapsed,
    })
    return row


def _run_vcg(scenario, instance, seed, preference_lambda, conditioned):
    model_seed = 0
    arm = conditioned["inputs"]["arms"][model_seed]
    agents = []
    envs = []

    def agent_factory(base):
        agent = final90._load_conditioned_agent(
            ROOT, {"conditioned": conditioned}, model_seed=model_seed,
            base=base, device=torch.device("cpu"),
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

    raw = None
    failure_class = None
    error = None
    started = perf_counter()
    try:
        cache = reuse.TimingInvariantCertificateCache()
        with (
            e13._WallClock(WALL_SECONDS),
            reuse.path_cleanup_active(),
            patch.object(benchmark, "_make_env", env_factory),
            pilot._agent_factory(agent_factory),
            patch.object(benchmark, "ViabilityCertificateCache", lambda: cache),
        ):
            raw = benchmark.run_arm(
                arm=benchmark.EXACT_FULL, controller_payload=arm.payload,
                instance=instance, instance_seed=seed,
                search_config=e13._search_config(arm, e27.SEARCH_MAX_NODES),
                liveness_rule=benchmark._liveness_rule(arm.payload),
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
        elif any(
            item.get("unknown_candidates_rejected", 0) > 0
            or item.get("safe_candidate_count", 0) == 0
            for item in frontiers
        ):
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
        raise E28Error("VCG reset was not authenticated as empty")
    decisions = () if raw is None else raw["decisions"]
    return {
        "method": VCG, "model_seed": model_seed,
        "preference_lambda": preference_lambda,
        "rng_index": None, "rng_seed": None,
        "strict_safe_complete": strict, "failure_class": failure_class,
        "method_failure_reason": None if raw is None else raw["method_failure_reason"],
        "error": error,
        "deliveries_at_stop": None if raw is None else len(raw["delivery_deviations"]),
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
        "occupancy": None if env is None or not env.occupancy_trace
        else env.occupancy_summary(),
        "first_frontier_sha256": (
            None if not agents else agents[0].first_frontier_sha256
        ),
        "guard_forced_decisions": sum(
            bool(item["liveness_forced"]) for item in decisions
        ),
        "decision_count": len(decisions),
        "unknown_candidate_rejections": sum(
            item.get("unknown_candidates_rejected", 0) for item in frontiers
        ),
        "admission_successes": sum(
            item["selected_action_type"] == "accept" and item["option_success"]
            for item in decisions
        ),
        "retrieval_blocker_relocations": 0,
        "adapter_blocked_wait_decisions": 0,
        "adapter_standalone_reconfigurations": (
            None if raw is None else int(raw["standalone_reconfigurations"])
        ),
        "illegal_drops": None if raw is None else int(raw["illegal_drops"]),
        "episode_wall_seconds": elapsed,
    }


def run_one(output, panel, method, scenario_id, seed, replicate):
    output = output.resolve()
    contract, inputs, scenario, instance = _coordinate(
        output, panel, method, scenario_id, seed, replicate
    )
    path = _ledger_path(output, panel, method, scenario_id, seed, replicate)
    if path.is_file():
        ledger = _read(path, "ledger_sha256")
        if ledger["contract_sha256"] != contract["contract_sha256"]:
            raise E28Error("saved ledger contract differs")
        return ledger["row"]
    identity = _replicate_identity(method, scenario_id, seed, replicate, panel)
    if method == VCG:
        row = _run_vcg(scenario, instance, seed, LAMBDAS[replicate],
                       inputs["conditioned"])
    else:
        row = _run_baseline(method, scenario, instance, seed, identity)
    row.update({
        "schema_version": 1, "protocol": PROTOCOL, "panel": panel,
        "method": method, "scenario_id": scenario_id,
        "preference_lambda": LAMBDAS[replicate] if method == VCG else None,
        "scenario": scenario.public_dict(), "instance_seed": seed,
        "instance_id": instance.instance_id, "schedule_id": instance.schedule_id,
        "initially_stored_jobs": 0, "arriving_jobs": scenario.total_jobs,
        "training_runs": 0,
    })
    ledger = _hashed({
        "schema_version": 1, "protocol": PROTOCOL,
        "contract_sha256": contract["contract_sha256"], "row": row,
    }, "ledger_sha256")
    _write(path, ledger)
    return row


def run_pilot(output):
    prepare(output)
    rows = [
        run_one(output, "development", method, PILOT_SCENARIO, DEV_SEEDS[0],
                PILOT_VCG_REPLICATE if method == VCG else 0)
        for method in METHODS
    ]
    starts_empty = all(
        row.get("initial_state_audit", {}).get("passed") is True
        and row["initial_state_audit"]["occupied_storage_slots"] == 0
        for row in rows
    )
    no_implementation_error = all(
        row["failure_class"] != "unsupported_or_implementation_error"
        for row in rows
    )
    baseline_reached_admission = all(
        (row.get("admission_successes") or 0) > 0 for row in rows if row["method"] != VCG
    )
    passed = starts_empty and no_implementation_error and baseline_reached_admission
    gate = _hashed({
        "schema_version": 1, "protocol": PROTOCOL,
        "status": "passed" if passed else "failed",
        "scope": "integration_only_no_quality_based_selection",
        "paper_evidence": False,
        "empty_start_verified": starts_empty,
        "baseline_reached_admission": baseline_reached_admission,
        "arms": [{
            "method": row["method"],
            "failure_class": row["failure_class"],
            "strict_complete": row["strict_safe_complete"],
            "admission_successes": row.get("admission_successes"),
            "steps_at_stop": row["steps_at_stop"],
            "error": row.get("error"),
        } for row in rows],
        "training_runs": 0,
    }, "gate_sha256")
    _write(output / GATE_NAME, gate)
    return gate


def freeze_confirmation(output):
    output = output.resolve()
    contract, _ = _contract(output)
    gate = _read(output / GATE_NAME, "gate_sha256")
    if gate["status"] != "passed":
        raise E28Error("integration pilot has not passed")
    expected = _manifest_payload("confirmation", SEEDS, contract)
    path = output / CONFIRMATION_MANIFEST
    if path.is_file():
        if _read(path, "manifest_sha256") != expected:
            raise E28Error("confirmation manifest differs")
    else:
        _write(path, expected)
    count = len(SCENARIOS) * len(SEEDS) * sum(len(value) for value in REPLICATES.values())
    return {"status": "frozen", "confirmation_instances": len(expected["records"]),
            "expected_runs": count, "training_runs": 0}


def _run_method(output, method):
    if method not in METHODS:
        raise E28Error("unknown method")
    freeze_confirmation(output)
    coordinates = [
        (scenario.scenario_id, seed, replicate)
        for scenario in SCENARIOS for seed in SEEDS
        for replicate in REPLICATES[method]
    ]
    complete = 0
    for index, (scenario_id, seed, replicate) in enumerate(coordinates, 1):
        row = run_one(output, "confirmation", method, scenario_id, seed, replicate)
        complete += int(row["strict_safe_complete"])
        print(
            f"E28 {method} {index}/{len(coordinates)} | {scenario_id} "
            f"seed={seed} rep={replicate} strict={int(row['strict_safe_complete'])} "
            f"failure={row['failure_class']} wall={row['episode_wall_seconds']:.1f}s",
            flush=True,
        )
    return {"status": "complete", "method": method, "rows": len(coordinates),
            "strict_complete_rows": complete, "training_runs": 0}


METRICS = (
    "mean_absolute_error", "within_target_window_rate",
    "physical_rehandles_per_100_required_deliveries", "steps_per_delivery",
)


def analyze(output):
    output = output.resolve()
    contract, _ = _contract(output)
    _manifest(output, "confirmation", contract)
    rows = []
    for scenario in SCENARIOS:
        for seed in SEEDS:
            for method in METHODS:
                for replicate in REPLICATES[method]:
                    path = _ledger_path(
                        output, "confirmation", method, scenario.scenario_id,
                        seed, replicate
                    )
                    ledger = _read(path, "ledger_sha256")
                    if ledger["contract_sha256"] != contract["contract_sha256"]:
                        raise E28Error(f"ledger contract mismatch: {path}")
                    row = ledger["row"]
                    if (row["scenario_id"], row["instance_seed"], row["method"],
                        row["replicate"]) != (scenario.scenario_id, seed, method, replicate):
                        raise E28Error(f"ledger coordinate mismatch: {path}")
                    rows.append(row)
    by_instance = defaultdict(list)
    for row in rows:
        by_instance[(
            row["scenario_id"], row["instance_seed"], row["method"],
            row["preference_lambda"],
        )].append(row)
    instance_means = []
    for (scenario_id, seed, method, preference_lambda), group in sorted(
        by_instance.items(), key=lambda item: (
            item[0][0], item[0][1], item[0][2],
            -1.0 if item[0][3] is None else item[0][3]
        )
    ):
        complete = [row for row in group if row["strict_safe_complete"]]
        item = {
            "scenario_id": scenario_id, "instance_seed": seed,
            "method": method, "preference_lambda": preference_lambda,
            "complete": len(complete), "runs": len(group),
            "completion_rate": len(complete) / len(group),
        }
        for name in METRICS:
            values = [row[name] for row in complete if row[name] is not None]
            item[name] = fmean(values) if values else None
        instance_means.append(item)
    summaries = []
    for scenario in SCENARIOS:
        for method in METHODS:
            for preference_lambda in (LAMBDAS if method == VCG else (None,)):
                group = [
                    item for item in instance_means
                    if item["scenario_id"] == scenario.scenario_id
                    and item["method"] == method
                    and item["preference_lambda"] == preference_lambda
                ]
                raw = [
                    row for row in rows
                    if row["scenario_id"] == scenario.scenario_id
                    and row["method"] == method
                    and row["preference_lambda"] == preference_lambda
                ]
                complete_rows = [row for row in raw if row["strict_safe_complete"]]
                item = {
                    "scenario_id": scenario.scenario_id, "method": method,
                    "preference_lambda": preference_lambda,
                    "strict_complete": sum(row["strict_safe_complete"] for row in raw),
                    "runs": len(raw),
                    "mean_instance_completion_rate": fmean(
                        value["completion_rate"] for value in group
                    ),
                    "failure_counts": dict(Counter(row["failure_class"] for row in raw)),
                    "adapter_blocked_wait_decisions": sum(
                        row.get("adapter_blocked_wait_decisions") or 0 for row in raw
                    ),
                    "mean_peak_occupancy": fmean(
                        row["occupancy"]["peak_storage_occupancy_ratio"]
                        for row in complete_rows if row.get("occupancy")
                    ) if any(row.get("occupancy") for row in complete_rows) else None,
                    "mean_time_weighted_occupancy": fmean(
                        row["occupancy"]["time_weighted_mean_storage_occupancy_ratio"]
                        for row in complete_rows if row.get("occupancy")
                    ) if any(row.get("occupancy") for row in complete_rows) else None,
                }
                for name in METRICS:
                    values = [value[name] for value in group if value[name] is not None]
                    item[name] = fmean(values) if values else None
                    item[f"{name}_contributing_instances"] = len(values)
                summaries.append(item)
    report = _hashed({
        "schema_version": 1, "protocol": PROTOCOL,
        "contract_sha256": contract["contract_sha256"],
        "status": "complete", "post_hoc": True,
        "rows": len(rows), "distinct_instances": len(SCENARIOS) * len(SEEDS),
        "new_episode_runs": len(rows),
        "vcg_preference_lambdas": list(LAMBDAS),
        "reused_controls_are_independent_replications": False,
        "instance_means": instance_means, "summaries": summaries,
        "interpretation": (
            "integrated controllers on own placements; baseline retrieval, dispatch, "
            "reconfiguration and waiting are explicitly declared common adapters, "
            "not algorithms supplied by the cited placement papers"
        ),
        "training_runs": 0,
    }, "report_sha256")
    _write(output / REPORT_NAME, report)
    lines = [
        "# E28 empty-start scale comparison", "",
        "| Yard | Method | λ | Strict | Placement MAE | Within ±20 | Rehandles/100 | Steps/delivery | Mean occupancy | Peak occupancy |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    def fmt(value, factor=1):
        return "—" if value is None else f"{factor*value:.2f}"
    def fmt_lambda(value):
        return "—" if value is None else f"{value:.4f}".rstrip("0").rstrip(".")
    for item in summaries:
        lines.append(
            f"| {item['scenario_id']} | {item['method']} | "
            f"{fmt_lambda(item['preference_lambda'])} | "
            f"{item['strict_complete']}/{item['runs']} | "
            f"{fmt(item['mean_absolute_error'])} | "
            f"{fmt(item['within_target_window_rate'],100)}% | "
            f"{fmt(item['physical_rehandles_per_100_required_deliveries'])} | "
            f"{fmt(item['steps_per_delivery'])} | "
            f"{fmt(item['mean_time_weighted_occupancy'],100)}% | "
            f"{fmt(item['mean_peak_occupancy'],100)}% |"
        )
    lines += [
        "",
        "Quality is conditional on strict completion. For each scenario, replications "
        "are averaged within each of three paired instances, then instances are "
        "weighted equally; a metric can have fewer than three contributing "
        "instances when a method fails. All rows are new executions, with no "
        "reused VCG control rows. Methods other than VCG are versioned online "
        "adaptations with a common retrieval/reconfiguration/wait scheduler.",
        "",
    ]
    e13._atomic_text(output / TABLE_NAME, "\n".join(lines))
    figure = render.render(report, output)
    return {"status": "complete", "rows": len(rows),
            "report": str(output / REPORT_NAME), "table": str(output / TABLE_NAME),
            "figure": figure,
            "training_runs": 0}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("prepare", "run-pilot", "freeze-confirmation",
                            "run-one", "run-method", "run-all", "analyze")
    )
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--panel", choices=("development", "confirmation"),
                        default="confirmation")
    parser.add_argument("--method", choices=METHODS)
    parser.add_argument("--scenario", choices=tuple(SCENARIO_BY_ID))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--replicate", type=int, default=0)
    args = parser.parse_args(argv)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    output = args.output_dir.resolve()
    if args.command == "prepare":
        result = prepare(output)
    elif args.command == "run-pilot":
        result = run_pilot(output)
    elif args.command == "freeze-confirmation":
        result = freeze_confirmation(output)
    elif args.command == "run-one":
        if args.method is None or args.scenario is None or args.seed is None:
            parser.error("run-one needs --method, --scenario, and --seed")
        result = run_one(
            output, args.panel, args.method, args.scenario, args.seed,
            args.replicate
        )
    elif args.command == "run-method":
        if args.method is None:
            parser.error("run-method needs --method")
        result = _run_method(output, args.method)
    elif args.command == "run-all":
        result = [_run_method(output, method) for method in METHODS]
    else:
        result = analyze(output)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False), flush=True)
    return result


if __name__ == "__main__":
    main()
