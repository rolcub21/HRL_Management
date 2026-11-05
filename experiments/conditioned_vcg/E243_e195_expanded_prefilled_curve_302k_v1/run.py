#!/usr/bin/env python3
"""Run E243's matched-ranker 5x5 prefilled certification curve."""

from __future__ import annotations

import argparse
import builtins
from collections import Counter
from contextlib import ExitStack
import json
import math
import os
from pathlib import Path
from statistics import fmean
from unittest.mock import patch

import torch

import run_vcg_objective_gamma_audit as audit
from PSLAP.viability import ViabilityStatus
from PSLAP.viability_candidates import BoundedEventDeferRule
from PSLAP.viability_filter import ViabilitySearchConfig
from experiments.conditioned_vcg.E14_certification_scalability_95k import reuse
from experiments.conditioned_vcg.E100_accelerated_release_clock_163k_v1 import (
    fast_inputs,
)
from experiments.conditioned_vcg.E107_queue_context_stage_a_8x8_171k_v1 import (
    representation as e107_representation,
)
from experiments.conditioned_vcg.E189_e188_roundoff_repair_248k_v1.roundoff import (
    repaired_audit,
)
from experiments.conditioned_vcg.E195_e191_method_5x5_longstay_254k_v1 import (
    instrumentation as e195_instrumentation,
    protocol as e195,
    stage as e195_stage,
)
from experiments.conditioned_vcg.E198_e195_shadow_certification_ablation_257k_v1 import (
    frontier as e198_frontier,
)
from projects.temporal_consequence_realization.method_conformance import (
    render_validated_conformance_audit,
)

from . import protocol as p, scenario
from .tracker import MatchedDynamicE195Agent, PrefillTracker, TrackedE195Agent


class E243Error(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise E243Error(message)


def _read(path: Path) -> dict:
    require(Path(path).is_file(), f"missing E243 artifact: {path}")
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"invalid E243 artifact: {path}")
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
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(
        _jsonable(value), indent=2, sort_keys=True, allow_nan=False, default=str
    ) + "\n"
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(encoded, encoding="utf-8")
    os.replace(temporary, path)


def _payload() -> dict:
    identity = p.checkpoint_identity()
    return torch.load(
        identity["source_checkpoint"], map_location="cpu", weights_only=False
    )


def _search_and_liveness():
    payload = _payload()
    search = payload["viability_search"]
    liveness = payload["liveness_rule"]
    return (
        ViabilitySearchConfig(
            max_depth=search["max_depth"],
            max_nodes=int(search["max_nodes"]),
            max_primitive_steps=search["max_primitive_steps"],
            reserve_queue_cells=bool(search["reserve_queue_cells"]),
            search_order=str(search["search_order"]),
        ),
        BoundedEventDeferRule(
            max_option_steps=int(liveness["max_option_steps"]),
            max_consecutive_defer_decisions=int(
                liveness["max_consecutive_defer_decisions"]
            ),
        ),
    )


def _load_agent(device: str, arm: str, tracker: PrefillTracker):
    cls = MatchedDynamicE195Agent if arm in p.DYNAMIC_ARMS else TrackedE195Agent
    agent = cls.from_checkpoint(
        _payload(), device=device, resumable=False, seed=p.MODEL_SEED
    )
    agent.set_epsilon(0.0)
    agent.Q_local.eval()
    agent.Q_target.eval()
    if arm in p.DYNAMIC_ARMS:
        agent.configure_e243(tracker)
    else:
        agent.configure_e233(tracker)
    return agent


def _frontier_wrapper(original, tracker: PrefillTracker, arm: str):
    accelerated = e195_stage._accelerated_enumerator(original)

    def wrapped(env, **kwargs):
        certified = accelerated(env, **kwargs)
        cache = kwargs.get("cache")
        require(cache is not None, "E243 requires a per-episode certificate cache")
        physical = e198_frontier.expose_physical_snapshot(
            env,
            certified,
            search_config=kwargs["search_config"],
            liveness_rule=kwargs["liveness_rule"],
            consecutive_defer_decisions=kwargs["consecutive_defer_decisions"],
            cache=cache,
            reserved_cells=kwargs.get("reserved_cells", ()),
            max_replans=kwargs.get("max_replans", 8),
        )
        tracker.record_frontier(certified, physical)
        return physical if arm in p.UNFILTERED_ARMS else certified

    return wrapped


def _failure_class(strict: bool, reason, steps: int, monitor: dict) -> str:
    if strict:
        return "completed"
    text = "" if reason is None else str(reason).lower()
    if monitor.get("guard_witness_unavailable", 0):
        return "guard_witness_unavailable"
    if int(steps) >= int(scenario.MAX_STEPS):
        return "step_limit"
    if monitor.get("first_exhaustive_unsafe_current") is not None:
        return "stranded_block"
    if any(word in text for word in ("no viable", "empty", "infeasible", "deadlock")):
        return "deadlock"
    return "other_method_failure"


def _run_one(*, arm: str, seed: int, device: str, protocol: str) -> dict:
    require(arm in p.ARMS, "invalid E243 arm")
    tracker = PrefillTracker(arm)
    agent = _load_agent(device, arm, tracker)
    env = scenario.Prefilled5x5Env(scenario.SCENARIO, e195.SPEC)
    env.current_episode = scenario.SEEDS.index(int(seed)) + 1
    search, liveness = _search_and_liveness()
    original_print = builtins.print

    def e243_print(*values, **kwargs):
        if values and isinstance(values[0], str) and values[0].startswith("E195 "):
            values = ("E243 " + values[0][len("E195 "):], *values[1:])
        return original_print(*values, **kwargs)

    with ExitStack() as stack:
        stack.enter_context(patch.object(
            audit,
            "enumerate_viability_candidates",
            _frontier_wrapper(audit.enumerate_viability_candidates, tracker, arm),
        ))
        stack.enter_context(patch.object(
            audit, "ViabilityCertificateCache", reuse.TimingInvariantCertificateCache
        ))
        stack.enter_context(patch.object(
            audit, "_rescore_run", e195_instrumentation.single_objective_rescore
        ))
        stack.enter_context(patch.object(builtins, "print", e243_print))
        stack.enter_context(reuse.path_cleanup_active())
        stack.enter_context(fast_inputs.enabled())
        stack.enter_context(e107_representation.acceleration_enabled())
        stack.enter_context(repaired_audit())
        run = e195_instrumentation.run_episode(
            agent,
            env,
            objective_spec=e195.SPEC,
            instance_seed=int(seed),
            training=False,
            max_steps=scenario.MAX_STEPS,
            search_config=search,
            liveness_rule=liveness,
            epsilon_start=0.0,
            epsilon_end=0.0,
            epsilon_decay_decisions=1,
            epsilon_warmup_decisions=0,
            updates_per_macro=1,
            run_protocol=protocol,
        )
    monitor = tracker.audit()
    initial = monitor.get("initial_current_certificate") or {}
    require(
        initial.get("status") == ViabilityStatus.SAFE.value
        and int(initial.get("witness_length", 0)) > 0,
        f"E243 seed {seed} did not begin with a concrete completion witness",
    )
    strict = bool(run.get("strict_method_success"))
    metrics = dict(run.get("e187_metrics", {}))
    if metrics:
        metrics["strict_complete"] = strict
    dynamic_audit = None
    if arm in p.DYNAMIC_ARMS:
        dynamic_audit = {
            "matched_policy": agent.MATCHED_POLICY,
            "ranker_sha256": p.dynamic_ranker_sha256(),
            "choices": list(agent.e243_dynamic.choice_audit),
        }
    steps = int(run.get("steps", env.time_steps))
    return {
        "status": "complete",
        "protocol": protocol,
        "arm": arm,
        "model_seed": p.MODEL_SEED,
        "instance_seed": int(seed),
        "initial_fill": scenario.initial_fill(seed),
        "initial_fill_fraction": scenario.initial_fill(seed) / scenario.CAPACITY,
        "initial_layout": [list(cell) for cell in scenario.placement_cells(seed)],
        "initial_completion_witness": initial,
        "strict_complete": strict,
        "failure_class": _failure_class(
            strict, run.get("method_failure_reason"), steps, monitor
        ),
        "method_failure_reason": run.get("method_failure_reason"),
        "steps_at_stop": steps,
        "delivered_at_stop": int(run.get("delivery_count", 0)),
        "metrics": metrics,
        "run": run,
        "certification_monitor": monitor,
        "dynamic_ranker_audit": dynamic_audit,
        "test_panel_opened": False,
    }


def _row_path(output: Path, arm: str, seed: int) -> Path:
    return output.resolve() / "rows" / arm / f"instance-{int(seed)}.json"


def _manifest_records() -> list[dict]:
    env = scenario.Prefilled5x5Env(scenario.SCENARIO, e195.SPEC)
    records = []
    for seed in scenario.SEEDS:
        instance = env.sample_episode_instance(seed)
        records.append({
            "seed": int(seed),
            "instance_id": instance.instance_id,
            "schedule_id": instance.schedule_id,
            "initial_fill": scenario.initial_fill(seed),
            "initial_layout": [list(cell) for cell in scenario.placement_cells(seed)],
            "transporter_cell": list(scenario.transporter_cell(seed)),
            "arrival_steps": list(instance.arrival_steps),
            "storage_steps_needed": list(instance.storage_steps_needed),
        })
    return records


def _validate_initial_states() -> list[dict]:
    search, liveness = _search_and_liveness()
    rows = []
    for seed in scenario.SEEDS:
        env = scenario.Prefilled5x5Env(scenario.SCENARIO, e195.SPEC)
        instance = env.sample_episode_instance(seed)
        env.reset(instance=instance)
        cache = reuse.TimingInvariantCertificateCache()
        with reuse.path_cleanup_active():
            snapshot = audit.enumerate_viability_candidates(
                env,
                consecutive_defer_decisions=0,
                search_config=search,
                liveness_rule=liveness,
                cache=cache,
            )
        certificate = snapshot.current_certificate
        require(
            certificate.status is ViabilityStatus.SAFE and certificate.witness,
            f"E243 initial seed {seed} is not certified retrievable",
        )
        rows.append({
            "seed": int(seed),
            "initial_fill": scenario.initial_fill(seed),
            "status": certificate.status.value,
            "exhaustive": bool(certificate.exhaustive),
            "reason": str(certificate.reason),
            "witness_length": len(certificate.witness),
        })
    return rows


def gate(device: str) -> dict:
    contract = p.gate_contract()
    output = p.DEFAULT_GATE_OUTPUT.resolve()
    path = output / p.GATE_NAME
    if path.is_file():
        existing = _read(path)
        core = dict(existing)
        recorded = core.pop("manifest_sha256", None)
        require(recorded == p.digest(core), "E243 gate manifest hash is invalid")
        require(
            existing.get("gate_contract_sha256") == contract["contract_sha256"],
            "E243 gate source identity changed",
        )
        return existing
    seed = scenario.SEEDS[0]
    routes = {}
    for arm in p.ARMS:
        row = _run_one(
            arm=arm, seed=seed, device=device, protocol=p.GATE_PROTOCOL
        )
        routes[arm] = {
            "route_executed": True,
            "strict_complete": bool(row["strict_complete"]),
            "monitor_present": isinstance(row.get("certification_monitor"), dict),
            "initial_witness": row["initial_completion_witness"],
            "dynamic_ranker_sha256": (
                None if arm not in p.DYNAMIC_ARMS
                else row["dynamic_ranker_audit"]["ranker_sha256"]
            ),
        }
    passed = bool(
        all(item["strict_complete"] and item["monitor_present"] for item in routes.values())
        and routes["dynamic_certified"]["dynamic_ranker_sha256"]
        == routes["dynamic_physical_unfiltered"]["dynamic_ranker_sha256"]
        == contract["dynamic_ranker_sha256"]
    )
    result = {
        "status": "pass" if passed else "fail",
        "protocol": p.GATE_PROTOCOL,
        "gate_contract_sha256": contract["contract_sha256"],
        "development_seed": seed,
        "routes": routes,
        "episode_runs": len(routes),
        "test_panel_opened": False,
    }
    result["manifest_sha256"] = p.digest(result)
    _write(path, result)
    require(passed, "E243 development integration gate failed")
    return result


def prepare(output: Path) -> dict:
    output = output.resolve()
    expected = p.contract()
    render_validated_conformance_audit(expected)
    initial = _validate_initial_states()
    contract_path = output / "contract.json"
    manifest_path = output / "instances.json"
    manifest = {
        "protocol": p.PROTOCOL,
        "contract_sha256": expected["contract_sha256"],
        "records": _manifest_records(),
        "initial_certificates": initial,
        "all_initial_states_have_completion_witness": True,
        "test_panel_opened": False,
    }
    if contract_path.is_file():
        require(_read(contract_path) == expected, "E243 contract or sources changed")
        require(_read(manifest_path) == manifest, "E243 instance manifest changed")
    else:
        require(
            not output.exists() or not any(output.iterdir()),
            "E243 refuses a nonempty uncontracted output",
        )
        _write(contract_path, expected)
        _write(manifest_path, manifest)
    return {
        "status": "prepared",
        "contract_sha256": expected["contract_sha256"],
        "initial_states_verified": len(initial),
        "expected_rows": p.EXPECTED_ROWS,
        "episode_runs": 0,
        "test_panel_opened": False,
    }


def authenticate(output: Path) -> dict:
    expected = p.contract()
    require(
        _read(output.resolve() / "contract.json") == expected,
        "E243 contract, checkpoint, ranker, or sources changed",
    )
    manifest = _read(output.resolve() / "instances.json")
    require(
        manifest.get("contract_sha256") == expected["contract_sha256"]
        and manifest.get("all_initial_states_have_completion_witness") is True
        and len(manifest.get("records", ())) == len(scenario.SEEDS),
        "E243 instance manifest is invalid",
    )
    return expected


def _validate_row(row: dict, contract: dict, arm: str, seed: int) -> None:
    expected = {
        "status": "complete",
        "protocol": p.PROTOCOL,
        "arm": arm,
        "model_seed": p.MODEL_SEED,
        "instance_seed": int(seed),
        "initial_fill": scenario.initial_fill(seed),
        "test_panel_opened": False,
    }
    mismatches = {
        key: (row.get(key), value) for key, value in expected.items()
        if row.get(key) != value
    }
    require(not mismatches, f"E243 saved row differs: {mismatches!r}")
    require(
        row.get("initial_completion_witness", {}).get("status") == "SAFE",
        "E243 saved row lacks its initial completion witness",
    )
    if arm in p.DYNAMIC_ARMS:
        require(
            row.get("dynamic_ranker_audit", {}).get("ranker_sha256")
            == contract["dynamic_ranker_sha256"],
            "E243 Dynamic row used another ranker",
        )


def evaluate(output: Path, device: str) -> dict:
    prepare(output)
    contract = authenticate(output)
    completed = reused = 0
    for arm in p.ARMS:
        for seed in scenario.SEEDS:
            path = _row_path(output, arm, seed)
            if path.is_file():
                _validate_row(_read(path), contract, arm, seed)
                reused += 1
                continue
            print(
                f"E243 start arm={arm} fill={scenario.initial_fill(seed)}/8 "
                f"instance={seed}", flush=True,
            )
            row = _run_one(
                arm=arm, seed=seed, device=device, protocol=p.PROTOCOL
            )
            _validate_row(row, contract, arm, seed)
            _write(path, row)
            completed += 1
            print(
                f"E243 done arm={arm} fill={scenario.initial_fill(seed)}/8 "
                f"instance={seed} strict={int(row['strict_complete'])} "
                f"failure={row['failure_class']}", flush=True,
            )
    result = analyze(output)
    return {
        "status": "complete",
        "new_episode_runs": completed,
        "reused_rows": reused,
        "rows": completed + reused,
        "analysis": result,
        "test_panel_opened": False,
    }


def _metric(row: dict, name: str):
    value = row.get("metrics", {}).get(name)
    return None if value is None else float(value)


def analyze(output: Path) -> dict:
    contract = authenticate(output)
    rows = []
    for arm in p.ARMS:
        for seed in scenario.SEEDS:
            path = _row_path(output, arm, seed)
            if path.is_file():
                row = _read(path)
                _validate_row(row, contract, arm, seed)
                rows.append(row)
    grouped = {}
    for arm in p.ARMS:
        grouped[arm] = {}
        for fill in scenario.INITIAL_FILLS:
            subset = [
                row for row in rows
                if row["arm"] == arm and int(row["initial_fill"]) == fill
            ]
            completed = [row for row in subset if row["strict_complete"]]
            j_values = [
                _metric(row, "J") for row in completed if _metric(row, "J") is not None
            ]
            grouped[arm][str(fill)] = {
                "cases": len(subset),
                "strict_completion_rate": (
                    None if not subset else sum(row["strict_complete"] for row in subset) / len(subset)
                ),
                "failure_rate": (
                    None if not subset else sum(not row["strict_complete"] for row in subset) / len(subset)
                ),
                "failure_classes": dict(Counter(
                    row["failure_class"] for row in subset if not row["strict_complete"]
                )),
                "unsafe_selected": sum(
                    int(row["certification_monitor"]["selected_successor_status_counts"].get("UNSAFE", 0))
                    for row in subset
                ),
                "mean_J_joint_definition_completed_only": (
                    None if not j_values else fmean(j_values)
                ),
                "mean_peak_stored": (
                    None if not completed else fmean(
                        _metric(row, "peak_stored") or 0.0 for row in completed
                    )
                ),
            }
    dynamic_hashes = {
        row["dynamic_ranker_audit"]["ranker_sha256"]
        for row in rows if row["arm"] in p.DYNAMIC_ARMS
    }
    report = {
        "status": "complete" if len(rows) == p.EXPECTED_ROWS else "partial",
        "protocol": p.PROTOCOL,
        "rows": len(rows),
        "expected_rows": p.EXPECTED_ROWS,
        "all_initial_states_witnessed": all(
            row["initial_completion_witness"]["status"] == "SAFE" for row in rows
        ) if rows else None,
        "dynamic_pair_one_ranker": dynamic_hashes in (
            set(), {contract["dynamic_ranker_sha256"]}
        ),
        "by_arm_and_initial_fill": grouped,
        "test_panel_opened": False,
        "paper_evidence": len(rows) == p.EXPECTED_ROWS,
    }
    _write(output.resolve() / p.REPORT_JSON, report)
    lines = [
        "# E243 prefilled certification curve", "",
        "| arm | fill | cases | strict completion | failures | unsafe selected | mean J (completers) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in p.ARMS:
        for fill in scenario.INITIAL_FILLS:
            item = grouped[arm][str(fill)]
            rate = item["strict_completion_rate"]
            mean_j = item["mean_J_joint_definition_completed_only"]
            lines.append(
                f"| {arm} | {fill}/8 | {item['cases']} | "
                f"{'n/a' if rate is None else f'{rate:.3f}'} | "
                f"{sum(item['failure_classes'].values())} | {item['unsafe_selected']} | "
                f"{'n/a' if mean_j is None else f'{mean_j:.1f}'} |"
            )
    table = "\n".join(lines) + "\n"
    (output.resolve() / p.TABLE_NAME).write_text(table, encoding="utf-8")
    conformance = render_validated_conformance_audit(contract)
    (output.resolve() / p.REPORT_NAME).write_text(
        conformance + "\n\n" + table,
        encoding="utf-8",
    )
    return report


def status(output: Path) -> dict:
    rows = list((output.resolve() / "rows").glob("*/*.json"))
    return {
        "status": "complete" if len(rows) == p.EXPECTED_ROWS else "partial",
        "rows": len(rows),
        "expected_rows": p.EXPECTED_ROWS,
        "percent_complete": 100.0 * len(rows) / p.EXPECTED_ROWS,
        "test_panel_opened": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("gate", "prepare", "evaluate", "analyze", "status")
    )
    parser.add_argument("--output", type=Path, default=p.DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if args.command == "gate":
        value = gate(args.device)
    elif args.command == "prepare":
        value = prepare(args.output)
    elif args.command == "evaluate":
        value = evaluate(args.output, args.device)
    elif args.command == "analyze":
        value = analyze(args.output)
    else:
        value = status(args.output)
    print(json.dumps(value, indent=2, sort_keys=True, allow_nan=False, default=str))


if __name__ == "__main__":
    main()
