#!/usr/bin/env python3
"""E27: certified Dynamic PSLAP versus frozen VCG under sustained arrivals."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
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
from experiments.conditioned_vcg.E27_certified_dynamic_sustained_99k.selector import CertifiedDynamicAgent
from experiments.conditioned_vcg.development.D10_scalability_support_screen import occupancy_extension as occupancy
import run_vcg_conditioned_final_comparison_90k as final90
import run_vcg_v11_conditioned_handling_seed0_85k as conditioned_seed0
import run_vcg_v11_nested_handling_pilot as pilot
from PSLAP.relocation_family_certification import RELOCATION_FAMILY_CERTIFICATION


ROOT = Path(__file__).resolve().parents[3]
OUTPUT = ROOT / "results/vcg-conditioned-e27-certified-dynamic-sustained-99k-v2"
PROTOCOL = "vcg_conditioned_e27_certified_dynamic_sustained_99k_v2"
CONTRACT_NAME = "e27-contract.json"
DEV_MANIFEST = "e27-development-manifest.json"
GATE_NAME = "e27-pilot-gate.json"
CONFIRMATION_MANIFEST = "e27-confirmation-manifest.json"
REPORT_NAME = "e27-report.json"
TABLE_NAME = "e27-results-table.md"
DEV_SEEDS = (79_100,)
CONFIRMATION_SEEDS = (99_000, 99_001, 99_002)
MODEL_SEED = 0
LAMBDA = 0.10
SEARCH_MAX_NODES = e13.SEARCH_MAX_NODES
WALL_LIMIT_SECONDS = e13.WALL_LIMIT_SECONDS
METHODS = ("vcg", "certified_dynamic_pslap")
SCENARIOS = (
    occupancy.Scenario("sustained_5x5_medium_n8", "sustained_arrivals", "medium", 5, 5, 4, 8),
    occupancy.Scenario("sustained_8x8_medium_n36", "sustained_arrivals", "medium", 8, 8, 18, 36),
    occupancy.Scenario("sustained_10x10_medium_n64", "sustained_arrivals", "medium", 10, 10, 32, 64),
)
SCENARIO_BY_ID = {scenario.scenario_id: scenario for scenario in SCENARIOS}
PILOT_SCENARIO = SCENARIOS[0].scenario_id


class E27Error(RuntimeError):
    pass


def _sha(path: Path) -> str:
    if not path.is_file() or path.is_symlink():
        raise E27Error(f"missing regular source: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _hash_dict(value, field):
    payload = dict(value)
    payload.pop(field, None)
    result = dict(payload)
    result[field] = hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()).hexdigest()
    return result


def _read_hashed(path, field):
    if not path.is_file() or path.is_symlink():
        raise E27Error(f"missing authenticated artifact: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if value != _hash_dict(value, field):
        raise E27Error(f"artifact hash mismatch: {path}")
    return value


class FrontierRecordAgent:
    def __init__(self, agent):
        self.agent = agent
        self.config = agent.config
        self.first_frontier_sha256 = None

    def reset_episode_state(self):
        self.first_frontier_sha256 = None
        return self.agent.reset_episode_state()

    on_episode_reset = reset_episode_state

    def select(self, snapshot, *, training=False, epsilon=0.0):
        if self.first_frontier_sha256 is None:
            keys = sorted(candidate.key for candidate in snapshot.candidates)
            self.first_frontier_sha256 = hashlib.sha256(
                json.dumps(keys, separators=(",", ":")).encode()
            ).hexdigest()
        return self.agent.select(snapshot, training=training, epsilon=epsilon)

    def observe_outcome(self, decision, *, next_snapshot, done):
        return self.agent.observe_outcome(decision, next_snapshot=next_snapshot, done=done)


def _parent():
    # E13's historical contract is bound to its original executor source.  A
    # later corrected executor makes live re-authentication intentionally fail.
    # Authenticate those historical artifacts as frozen inputs, then bind the
    # *current* executor by E27's own source hashes (as E25-v2 does).
    parent_contract, _manifest, _scope = e13.authenticate_frozen(e13.DEFAULT_OUTPUT)
    inputs = e13._parent_inputs()
    if (inputs["conditioned"]["terminal_sha256"][MODEL_SEED]
        != parent_contract["conditioned_terminal_sha256"]):
        raise E27Error("frozen conditioned checkpoint identity changed")
    return parent_contract, inputs


def _contract_payload(parent_contract, inputs):
    paths = (
        Path(__file__).resolve(), Path(__file__).with_name("selector.py"),
        Path(occupancy.__file__).resolve(), Path(e13.__file__).resolve(),
        ROOT / "benchmark_viability_critic_priority.py",
        ROOT / "PSLAP/dynamic_yard.py", ROOT / "PSLAP/retrieval_dispatch.py",
        ROOT / "PSLAP/viability_candidates.py", ROOT / "viability_graph_hierarchy.py",
    )
    return {
        "schema_version": 1, "protocol": PROTOCOL, "post_hoc": True,
        "question": "ranking value under shared certification and guard with sustained arrivals",
        "methods": list(METHODS), "model_seed": MODEL_SEED,
        "preference_lambda": LAMBDA,
        "development_seeds": list(DEV_SEEDS),
        "confirmation_seeds": list(CONFIRMATION_SEEDS),
        "scenarios": [scenario.public_dict() for scenario in SCENARIOS],
        "arrivals": "20 + 10*j for each post-initial job",
        "storage_durations": "Poisson(80), independent per instance seed",
        "initial_occupancy": "50%-51% of usable storage; initial blocks stored at t=0",
        "certification": {
            "max_nodes": SEARCH_MAX_NODES, "search_order": "goal_directed",
            "cache": "E14 timing-invariant", "path_cleanup": True,
            "strategy": RELOCATION_FAMILY_CERTIFICATION,
        },
        "shared": ["candidate generation", "exact positive certificate",
                   "recovery-witness guard", "macro executor", "episode instance",
                   "mode aggregation", "CPU and thread settings"],
        "dynamic_policy": CertifiedDynamicAgent.POLICY,
        "projection_audit_required": True,
        "wall_limit_seconds": WALL_LIMIT_SECONDS,
        "quality_scope": "strict-complete only; censored and incomplete separately",
        "parent_e13_contract_sha256": parent_contract["contract_sha256"],
        "conditioned_terminal_sha256": inputs["conditioned"]["terminal_sha256"][MODEL_SEED],
        "source_sha256": {str(path.relative_to(ROOT)): _sha(path) for path in paths},
        "training_runs": 0,
    }


def _contract(output):
    parent_contract, inputs = _parent()
    expected = _hash_dict(_contract_payload(parent_contract, inputs), "contract_sha256")
    observed = _read_hashed(output / CONTRACT_NAME, "contract_sha256")
    if expected != observed:
        raise E27Error("contract, E13 parent, checkpoint, or source changed")
    return observed, inputs


def _manifest_payload(panel, seeds, contract):
    records = []
    for scenario in SCENARIOS:
        for seed in seeds:
            instance = occupancy._instance(scenario, seed)
            records.append({
                "scenario_id": scenario.scenario_id, "seed": seed,
                "instance_id": instance.instance_id, "schedule_id": instance.schedule_id,
                "instance_json": instance.to_json(),
            })
    return _hash_dict({
        "schema_version": 1, "protocol": PROTOCOL, "panel": panel,
        "contract_sha256": contract["contract_sha256"],
        "records": records,
    }, "manifest_sha256")


def _manifest(output, panel, contract):
    filename = DEV_MANIFEST if panel == "development" else CONFIRMATION_MANIFEST
    observed = _read_hashed(output / filename, "manifest_sha256")
    seeds = DEV_SEEDS if panel == "development" else CONFIRMATION_SEEDS
    expected = _manifest_payload(panel, seeds, contract)
    if observed != expected:
        raise E27Error(f"{panel} manifest changed")
    return observed


def prepare(output):
    output = output.resolve()
    parent_contract, inputs = _parent()
    expected = _hash_dict(_contract_payload(parent_contract, inputs), "contract_sha256")
    path = output / CONTRACT_NAME
    if path.is_file():
        if _read_hashed(path, "contract_sha256") != expected:
            raise E27Error("existing E27 contract differs from current sources")
    else:
        e13._atomic_json(path, expected)
    development = _manifest_payload("development", DEV_SEEDS, expected)
    dev_path = output / DEV_MANIFEST
    if dev_path.is_file():
        if _read_hashed(dev_path, "manifest_sha256") != development:
            raise E27Error("existing development manifest differs")
    else:
        e13._atomic_json(dev_path, development)
    return {"status": "prepared", "development_instances": len(development["records"]),
            "confirmation_panel_opened": (output / CONFIRMATION_MANIFEST).is_file(),
            "training_runs": 0}


def _coordinate(output, panel, method, scenario_id, seed):
    if panel not in ("development", "confirmation") or method not in METHODS:
        raise E27Error("unsupported panel or method")
    if scenario_id not in SCENARIO_BY_ID:
        raise E27Error("unsupported scenario")
    if seed not in (DEV_SEEDS if panel == "development" else CONFIRMATION_SEEDS):
        raise E27Error("unsupported seed")
    contract, inputs = _contract(output)
    manifest = _manifest(output, panel, contract)
    record = next(record for record in manifest["records"]
                  if record["scenario_id"] == scenario_id and record["seed"] == seed)
    instance = EpisodeInstance.from_json(record["instance_json"])
    instance.validate_for(SCENARIO_BY_ID[scenario_id].make_base_env())
    return contract, inputs, instance


def _ledger_path(output, panel, method, scenario_id, seed):
    return output / "run-ledger" / panel / method / scenario_id / f"seed-{seed}.json"


def run_one(output, panel, method, scenario_id, seed):
    output = output.resolve()
    contract, inputs, instance = _coordinate(output, panel, method, scenario_id, seed)
    path = _ledger_path(output, panel, method, scenario_id, seed)
    if path.is_file():
        ledger = _read_hashed(path, "ledger_sha256")
        if ledger.get("contract_sha256") != contract["contract_sha256"]:
            raise E27Error("saved ledger has a different contract")
        return ledger["row"]
    scenario = SCENARIO_BY_ID[scenario_id]
    conditioned = inputs["conditioned"]
    arm = conditioned["inputs"]["arms"][MODEL_SEED]
    agents = []
    envs = []
    def agent_factory(base):
        if method == "vcg":
            agent = final90._load_conditioned_agent(
                ROOT, {"conditioned": conditioned}, model_seed=MODEL_SEED,
                base=base, device=torch.device("cpu")
            )
            agent.set_epsilon(0.0)
            selected = FrontierRecordAgent(conditioned_seed0._FixedLambdaAgent(agent, LAMBDA))
        else:
            # E4's nonlearned rankers wrap the bare VCG graph controller.
            # Only the VCG arm needs the conditioned handling head.
            base.set_epsilon(0.0)
            selected = CertifiedDynamicAgent(base)
        agents.append(selected)
        return selected
    def env_factory(_payload):
        env = occupancy.OccupancyTrackingEnv(scenario)
        envs.append(env)
        return env
    raw = None
    failure_class = None
    error = None
    started = perf_counter()
    try:
        cache = reuse.TimingInvariantCertificateCache()
        with (
            e13._WallClock(WALL_LIMIT_SECONDS),
            reuse.path_cleanup_active(),
            patch.object(benchmark, "_make_env", env_factory),
            pilot._agent_factory(agent_factory),
            patch.object(benchmark, "ViabilityCertificateCache", lambda: cache),
        ):
            raw = benchmark.run_arm(
                arm=benchmark.EXACT_FULL, controller_payload=arm.payload,
                instance=instance, instance_seed=seed,
                search_config=e13._search_config(arm, SEARCH_MAX_NODES),
                liveness_rule=benchmark._liveness_rule(arm.payload),
                prioritizer=None, max_steps=scenario.max_steps,
                device=torch.device("cpu"),
                recovery_certification_strategy=RELOCATION_FAMILY_CERTIFICATION,
            )
    except e13.EpisodeWallLimit as exc:
        failure_class, error = "censored_wall_clock", str(exc)
    except Exception as exc:
        failure_class, error = "unsupported_or_implementation_error", f"{type(exc).__name__}: {exc}"
    wall_seconds = perf_counter() - started
    strict = bool(raw is not None and raw["strict_method_success"] and raw["terminal"]
                  and raw["method_failure_reason"] is None
                  and raw["complete_frontier_exactly_verified"]
                  and raw["illegal_drops"] == 0 and raw["macro_failures"] == 0
                  and len(raw["delivery_deviations"]) == scenario.total_jobs)
    if failure_class is None:
        failure_class = "completed" if strict else "incomplete_or_certificate_limited"
    timing = (e13.d10._timing(raw["delivery_deviations"], expected_deliveries=scenario.total_jobs)
              if strict else {"mean_absolute_error": None, "within_target_window_rate": None})
    rehandles = (None if raw is None else int(raw.get("physical_storage_relocations", raw.get("relocations", 0))))
    selected = agents[0] if agents else None
    choice_audit = selected.choice_audit if method != "vcg" and selected is not None else []
    row = {
        "schema_version": 1, "protocol": PROTOCOL,
        "panel": panel, "method": method,
        "scenario_id": scenario_id, "scenario": scenario.public_dict(),
        "seed": seed, "instance_id": instance.instance_id,
        "schedule_id": instance.schedule_id,
        "strict_safe_complete": strict, "failure_class": failure_class,
        "error": error, "method_failure_reason": None if raw is None else raw["method_failure_reason"],
        "required_deliveries": scenario.total_jobs,
        "deliveries_at_stop": None if raw is None else len(raw["delivery_deviations"]),
        "steps_at_stop": None if raw is None else int(raw["steps"]),
        "dense_return": float(raw["return"]) if strict else None,
        "mean_absolute_error": timing["mean_absolute_error"],
        "within_target_window_rate": timing["within_target_window_rate"],
        "physical_rehandles": rehandles if strict else None,
        "physical_rehandles_per_100_required_deliveries": (
            100.0*rehandles/scenario.total_jobs if strict else None
        ),
        "steps_per_delivery": raw["steps"]/scenario.total_jobs if strict else None,
        "first_frontier_sha256": None if selected is None else selected.first_frontier_sha256,
        "dynamic_choice_audit": choice_audit,
        "dynamic_projection_count": sum(int(item["dynamic_placement_projected"])
                                        for item in choice_audit),
        "guard_forced_decisions": sum(int(item["guard_forced"]) for item in choice_audit)
        if method != "vcg" else None,
        "occupancy": envs[0].occupancy_summary() if envs else None,
        "episode_wall_seconds": wall_seconds,
        "training_runs": 0,
    }
    ledger = _hash_dict({"schema_version": 1, "protocol": PROTOCOL,
                         "contract_sha256": contract["contract_sha256"], "row": row},
                        "ledger_sha256")
    e13._atomic_json(path, ledger)
    return row


def run_pilot(output):
    output = output.resolve()
    prepare(output)
    rows = [run_one(output, "development", method, PILOT_SCENARIO, DEV_SEEDS[0])
            for method in METHODS]
    same_frontier = bool(rows[0]["first_frontier_sha256"] and
                         rows[0]["first_frontier_sha256"] == rows[1]["first_frontier_sha256"])
    passed = bool(same_frontier and all(row["failure_class"] != "unsupported_or_implementation_error"
                                        for row in rows))
    gate = _hash_dict({
        "schema_version": 1, "protocol": PROTOCOL,
        "status": "passed" if passed else "failed",
        "scope": "integration_only_no_quality_based_selection",
        "paper_evidence": False,
        "same_first_certified_frontier": same_frontier,
        "arms": [{"method": row["method"], "strict_safe_complete": row["strict_safe_complete"],
                  "failure_class": row["failure_class"],
                  "dynamic_projection_count": row["dynamic_projection_count"]} for row in rows],
        "training_runs": 0,
    }, "gate_sha256")
    e13._atomic_json(output / GATE_NAME, gate)
    return gate


def freeze_confirmation(output):
    output = output.resolve()
    contract, _inputs = _contract(output)
    gate = _read_hashed(output / GATE_NAME, "gate_sha256")
    if gate["status"] != "passed" or not gate["same_first_certified_frontier"]:
        raise E27Error("pilot integration gate has not passed")
    expected = _manifest_payload("confirmation", CONFIRMATION_SEEDS, contract)
    path = output / CONFIRMATION_MANIFEST
    if path.is_file():
        if _read_hashed(path, "manifest_sha256") != expected:
            raise E27Error("confirmation manifest changed")
    else:
        e13._atomic_json(path, expected)
    return {"status": "frozen", "confirmation_instances": len(expected["records"]),
            "expected_runs": len(expected["records"])*len(METHODS), "training_runs": 0}


def analyze(output):
    output = output.resolve()
    contract, _inputs = _contract(output)
    _manifest(output, "confirmation", contract)
    rows = []
    for scenario in SCENARIOS:
        for seed in CONFIRMATION_SEEDS:
            for method in METHODS:
                ledger = _read_hashed(_ledger_path(output, "confirmation", method,
                                                   scenario.scenario_id, seed), "ledger_sha256")
                if ledger["contract_sha256"] != contract["contract_sha256"]:
                    raise E27Error("ledger contract mismatch")
                rows.append(ledger["row"])
    summaries = []
    for scenario in SCENARIOS:
        for method in METHODS:
            subset = [row for row in rows if row["scenario_id"] == scenario.scenario_id
                      and row["method"] == method]
            complete = [row for row in subset if row["strict_safe_complete"]]
            def avg(name):
                values = [row[name] for row in complete if row[name] is not None]
                return fmean(values) if values else None
            summaries.append({
                "scenario_id": scenario.scenario_id, "method": method,
                "strict_complete": len(complete), "runs": len(subset),
                "mean_absolute_error": avg("mean_absolute_error"),
                "within_target_window_rate": avg("within_target_window_rate"),
                "physical_rehandles_per_100_required_deliveries": avg(
                    "physical_rehandles_per_100_required_deliveries"),
                "steps_per_delivery": avg("steps_per_delivery"),
                "failure_counts": dict(Counter(row["failure_class"] for row in subset)),
                "dynamic_projection_count": sum(row["dynamic_projection_count"] for row in subset),
            })
    report = _hash_dict({
        "schema_version": 1, "protocol": PROTOCOL,
        "contract_sha256": contract["contract_sha256"],
        "status": "complete", "post_hoc": True, "rows": len(rows),
        "reused_controls_are_independent_replications": False,
        "summaries": summaries,
        "interpretation": "only the certified selector differs; trajectories and guard activation may diverge",
        "training_runs": 0,
    }, "report_sha256")
    e13._atomic_json(output / REPORT_NAME, report)
    lines = ["# E27 certified Dynamic PSLAP on sustained-arrival scale panel", "",
             "| Yard | Method | Strict | Placement MAE | Within ±20 | Rehandles/100 | Steps/delivery | Projections |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    def fmt(value, factor=1.0):
        return "—" if value is None else f"{factor*value:.2f}"
    for item in summaries:
        lines.append(
            f"| {item['scenario_id']} | {item['method']} | {item['strict_complete']}/{item['runs']} | "
            f"{fmt(item['mean_absolute_error'])} | "
            f"{fmt(item['within_target_window_rate'], 100)}% | "
            f"{fmt(item['physical_rehandles_per_100_required_deliveries'])} | "
            f"{fmt(item['steps_per_delivery'])} | {item['dynamic_projection_count']} |"
        )
    lines.extend(["", "All metrics except completion are conditional on strict completion. "
                  "Each arm shares the same certified frontier construction, guard, and executor. "
                  "Dynamic's original placement proposal is projected only when absent from the SAFE frontier; "
                  "those events are counted, not silently attributed to the original algorithm.", ""])
    e13._atomic_text(output / TABLE_NAME, "\n".join(lines))
    return {"status": "complete", "rows": len(rows), "report": str(output / REPORT_NAME),
            "table": str(output / TABLE_NAME), "training_runs": 0}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run-pilot", "freeze-confirmation",
                                            "run-one", "run-all", "analyze"))
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--panel", choices=("development", "confirmation"), default="confirmation")
    parser.add_argument("--method", choices=METHODS)
    parser.add_argument("--scenario", choices=tuple(SCENARIO_BY_ID))
    parser.add_argument("--seed", type=int)
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
        result = run_one(output, args.panel, args.method, args.scenario, args.seed)
    elif args.command == "run-all":
        freeze_confirmation(output)
        rows = []
        for scenario in SCENARIOS:
            for seed in CONFIRMATION_SEEDS:
                for method in METHODS:
                    row = run_one(output, "confirmation", method, scenario.scenario_id, seed)
                    rows.append(row)
                    print(f"E27 {len(rows)}/{len(SCENARIOS)*len(CONFIRMATION_SEEDS)*len(METHODS)} | "
                          f"{scenario.scenario_id} seed={seed} {method} "
                          f"strict={int(row['strict_safe_complete'])} wall={row['episode_wall_seconds']:.1f}s",
                          flush=True)
        result = {"status": "complete", "rows": len(rows), "training_runs": 0}
    else:
        result = analyze(output)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False), flush=True)
    return result


if __name__ == "__main__":
    main()
