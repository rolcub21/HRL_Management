#!/usr/bin/env python3
"""Run and compare E204's exact immediate-only control ablation."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import csv
import json
import math
import os
from pathlib import Path
from statistics import fmean
from time import perf_counter

import torch

import train_vcg_preference_conditioned as common
from experiments.conditioned_vcg.E191_fresh_long_stay_crowding_250k_v1 import protocol as e191
from experiments.conditioned_vcg.E204_e191_centered_rehandle_stage_b_263k_v1 import run as e204
from experiments.conditioned_vcg.E205_e204_paired_tradeoff_development_264k_v1 import (
    protocol as e205,
    run as e205_run,
)
from projects.temporal_consequence_realization.method_conformance import (
    render_validated_conformance_audit,
    validate_protocol,
)

from . import protocol as p
from .immediate import from_stage_b_agent


class E207Error(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise E207Error(message)


def _read(path: Path) -> dict:
    require(Path(path).is_file(), f"missing E207 artifact: {path}")
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"invalid E207 JSON: {path}")
    return value


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path)


def prepare(output: Path) -> dict:
    output = output.resolve()
    contract = p.contract()
    validate_protocol(contract, source=p.PROTOCOL)
    source_manifest = _read(p.SOURCE_OUTPUT / "instances.json")
    contract_path = output / "contract.json"
    instances_path = output / "instances.json"
    if contract_path.is_file():
        require(_read(contract_path) == contract, "E207 contract changed")
        require(_read(instances_path) == source_manifest, "E207 instances changed")
    else:
        require(not output.exists() or not any(output.iterdir()), "E207 refuses nonempty output")
        _write(contract_path, contract)
        _write(instances_path, source_manifest)
    return {
        "status": "prepared", "contract_sha256": contract["contract_sha256"],
        "episode_runs": 0, "training_runs": 0, "expected_rows": 6,
        "test_panel_opened": False,
    }


def _label(value: float) -> str:
    return f"{float(value):g}".replace(".", "p")


def row_path(output: Path, seed: int, value: float) -> Path:
    rho = int(round(100 * e205.workload_for_seed(seed)))
    return output.resolve() / "rows" / f"rho-{rho}" / f"lambda-{_label(value)}.json"


def load_agent(device_name: str):
    stage_b_agent, search, liveness, base_digest, stage_b, device = e205_run.load_agent(device_name)
    agent = from_stage_b_agent(stage_b_agent, seed=266_207_700)
    with torch.no_grad():
        probe = torch.zeros((3, agent.config.feature_dim), device=device)
        predicted = agent.handling_network(probe, torch.tensor([10.0, 20.0, 40.0], device=device))
    require(bool((predicted == 0).all()), "E207 future term is not exact zero")
    return agent, search, liveness, base_digest, stage_b


def run_one_loaded(output, contract, agent, search, liveness, base_digest, stage_b, seed, value):
    path = row_path(output, seed, value)
    instance = e205_run.frozen_instance(_read(output / "instances.json"), seed)
    identity = {
        "protocol": p.PROTOCOL, "contract_sha256": contract["contract_sha256"],
        "instance_seed": int(seed), "instance_id": instance.instance_id,
        "schedule_id": instance.schedule_id, "workload": e205.workload_for_seed(seed),
        "lambda": float(value), "arm": "immediate_only",
        "stage_b_terminal_sha256": stage_b["terminal_checkpoint_sha256"],
    }
    if path.is_file():
        row = _read(path)
        require(all(row.get(k) == v for k, v in identity.items()), f"E207 row differs: {path}")
        return {**row, "episode_runs": 0, "reused": True}
    env = e191.TargetDenseEnv(p.SCENARIO, e191.SPEC)
    agent.base_agent.bind_context_environment(env, required_jobs=e191.scenario_for(p.SCENARIO).total_jobs, max_steps=e191.MAX_STEPS)
    before = e204.state_digest(agent.base_agent.Q_local.state_dict())
    label = f"rho={100*identity['workload']:.0f}% lambda={value:g}"
    print(f"E207 start {label}", flush=True)
    started = perf_counter()
    with e205_run.heartbeat(f"E207 {label}"), ExitStack() as stack:
        e204._runtime_stack(stack)
        result = common.run_preference_episode(
            agent, env, instance_seed=int(seed), preference_lambda=float(value),
            preference_schedule_record={
                "behavior_lambda": float(value), "lambda_center": p.REFERENCE_LAMBDA,
                "arm": "immediate_only", "fixed_for_complete_episode": True,
            },
            max_steps=e191.MAX_STEPS, search_config=search, liveness_rule=liveness,
            epsilon_start=0.0, epsilon_end=0.0, epsilon_decay_decisions=1,
            epsilon_warmup_decisions=0, updates_per_macro=1,
        )
    agent.pop_completed_samples()
    require(e204.state_digest(agent.base_agent.Q_local.state_dict()) == before == base_digest, "E207 mutated E191")
    metrics = env.e187_metrics()
    components = {key: float(item) for key, item in metrics["components"].items()}
    rehandles = int(result["physical_rehandles"])
    j10 = float(metrics["J"])
    row = {
        **identity, "status": "complete", "strict_complete": bool(result["strict_method_success"]),
        "J10": j10, "J_lambda": j10 + (float(value) - 10.0) * rehandles,
        "physical_rehandles": rehandles,
        "timing_cost_2T_plus_E": 2*components["storage_tardiness_accrual"] + components["storage_earliness_at_delivery"],
        "admission_wait_cost_0p5W": 0.5*components["admission_wait_accrual"],
        "equipment_cost_0p1L": 0.1*components["equipment_work"],
        "steps": int(result["steps"]), "peak_stored": int(metrics["peak_stored"]),
        "mean_stored": float(metrics["mean_stored"]),
        "wall_seconds": perf_counter()-started, "episode_runs": 1,
        "training_runs": 0, "test_panel_opened": False, "paper_evidence": False,
    }
    require(row["strict_complete"], f"E207 episode failed: {row}")
    _write(path, row)
    print(f"E207 done {label} Jlambda={row['J_lambda']:.1f} R={rehandles} timing={row['timing_cost_2T_plus_E']:.1f} wall={row['wall_seconds']:.1f}s", flush=True)
    return row


def run_all(output: Path, device_name: str) -> dict:
    prepared = prepare(output)
    contract = _read(output.resolve() / "contract.json")
    agent, search, liveness, base_digest, stage_b = load_agent(device_name)
    rows, new = [], 0
    for seed in p.INSTANCE_SEEDS:
        for value in p.LAMBDAS:
            row = run_one_loaded(output.resolve(), contract, agent, search, liveness, base_digest, stage_b, seed, value)
            rows.append(row); new += int(row["episode_runs"])
    return {"status": "complete", "rows": len(rows), "new_episode_runs": new, "reused_rows": len(rows)-new, "test_panel_opened": False}


def _dominates(left: dict, right: dict) -> bool:
    return (
        left["physical_rehandles"] <= right["physical_rehandles"]
        and left["timing_cost_2T_plus_E"] <= right["timing_cost_2T_plus_E"]
        and (
            left["physical_rehandles"] < right["physical_rehandles"]
            or left["timing_cost_2T_plus_E"] < right["timing_cost_2T_plus_E"]
        )
    )


def analyze(output: Path) -> dict:
    prepared = prepare(output)
    contract = _read(output.resolve() / "contract.json")
    paired = []
    for seed in p.INSTANCE_SEEDS:
        center_path = e205_run.row_path(p.SOURCE_OUTPUT, seed, p.REFERENCE_LAMBDA)
        require(center_path.is_file(), f"missing E205 center row: {center_path}")
        center = _read(center_path)
        for value in p.LAMBDAS:
            learned_path = e205_run.row_path(p.SOURCE_OUTPUT, seed, value)
            immediate_path = row_path(output, seed, value)
            require(learned_path.is_file(), f"missing E205 learned row: {learned_path}")
            require(immediate_path.is_file(), f"missing E207 row: {immediate_path}")
            learned, immediate = _read(learned_path), _read(immediate_path)
            require(learned["instance_id"] == immediate["instance_id"] == center["instance_id"], "E207 pairing differs")
            for arm, row in (("learned_future", learned), ("immediate_only", immediate)):
                avoided = int(center["physical_rehandles"]) - int(row["physical_rehandles"])
                non_r = float(row["J10"]) - 10.0*float(row["physical_rehandles"])
                center_non_r = float(center["J10"]) - 10.0*float(center["physical_rehandles"])
                paired.append({
                    "instance_seed": int(seed), "workload": float(row["workload"]),
                    "lambda": float(value), "arm": arm,
                    "physical_rehandles": int(row["physical_rehandles"]),
                    "timing_cost_2T_plus_E": float(row["timing_cost_2T_plus_E"]),
                    "J_lambda": float(row["J_lambda"]), "J10": float(row["J10"]),
                    "steps": int(row["steps"]), "peak_stored": int(row["peak_stored"]),
                    "rehandles_avoided_from_lambda10": avoided,
                    "non_rehandle_cost_change_from_lambda10": non_r-center_non_r,
                    "non_rehandle_cost_per_rehandle_avoided": ((non_r-center_non_r)/avoided if avoided > 0 else None),
                })
    comparisons = []
    for seed in p.INSTANCE_SEEDS:
        for value in p.LAMBDAS:
            learned = next(row for row in paired if row["instance_seed"] == seed and row["lambda"] == value and row["arm"] == "learned_future")
            immediate = next(row for row in paired if row["instance_seed"] == seed and row["lambda"] == value and row["arm"] == "immediate_only")
            comparisons.append({
                "instance_seed": seed, "workload": learned["workload"], "lambda": value,
                "delta_learned_minus_immediate_rehandles": learned["physical_rehandles"]-immediate["physical_rehandles"],
                "delta_learned_minus_immediate_timing": learned["timing_cost_2T_plus_E"]-immediate["timing_cost_2T_plus_E"],
                "delta_learned_minus_immediate_J_lambda": learned["J_lambda"]-immediate["J_lambda"],
                "learned_dominates_immediate": _dominates(learned, immediate),
                "immediate_dominates_learned": _dominates(immediate, learned),
            })
    result = {
        "status": "complete", "protocol": p.PROTOCOL,
        "contract_sha256": prepared["contract_sha256"], "paired_rows": paired,
        "paired_comparisons": comparisons,
        "summary": {
            "cases": len(comparisons),
            "learned_lower_J_lambda_rate": fmean(float(row["delta_learned_minus_immediate_J_lambda"] < 0) for row in comparisons),
            "learned_dominance_rate": fmean(float(row["learned_dominates_immediate"]) for row in comparisons),
            "immediate_dominance_rate": fmean(float(row["immediate_dominates_learned"]) for row in comparisons),
            "mean_delta_rehandles": fmean(row["delta_learned_minus_immediate_rehandles"] for row in comparisons),
            "mean_delta_timing": fmean(row["delta_learned_minus_immediate_timing"] for row in comparisons),
            "mean_delta_J_lambda": fmean(row["delta_learned_minus_immediate_J_lambda"] for row in comparisons),
        },
        "scope": contract["scope"], "test_panel_opened": False, "paper_evidence": False,
    }
    _write(output.resolve()/p.REPORT_JSON, result)
    declaration = {key: contract[key] for key in ("method_reference", "method_revision", "claim_level", "declared_deviations", "oracle_inputs", "nonconforming_shortcuts", "conformance_audit")}
    summary = result["summary"]
    text = render_validated_conformance_audit(declaration) + "\nRESULTS\n" + "="*72 + "\n\n" + (
        f"Paired comparisons: {summary['cases']}\n"
        f"Learned future head lower J_lambda: {100*summary['learned_lower_J_lambda_rate']:.1f}%\n"
        f"Learned head Pareto-dominates immediate-only: {100*summary['learned_dominance_rate']:.1f}%\n"
        f"Immediate-only Pareto-dominates learned head: {100*summary['immediate_dominance_rate']:.1f}%\n"
        f"Mean learned-minus-immediate rehandles: {summary['mean_delta_rehandles']:+.2f}\n"
        f"Mean learned-minus-immediate timing cost: {summary['mean_delta_timing']:+.2f}\n"
        f"Mean learned-minus-immediate J_lambda: {summary['mean_delta_J_lambda']:+.2f}\n"
        "\nThis is a three-instance development diagnostic, not final paper evidence.\n"
    )
    (output.resolve()/p.REPORT_NAME).write_text(text, encoding="utf-8")
    fields = tuple(paired[0].keys())
    with (output.resolve()/p.CSV_NAME).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(paired)
    from .plot import create_plot
    create_plot(paired, output.resolve()/p.PLOT_NAME)
    return {"status": "complete", "report": str(output.resolve()/p.REPORT_NAME), "plot": str(output.resolve()/p.PLOT_NAME), **summary, "test_panel_opened": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run-all", "analyze"))
    parser.add_argument("--output", type=Path, default=p.DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cuda")
    args = parser.parse_args()
    result = prepare(args.output) if args.command == "prepare" else run_all(args.output, args.device) if args.command == "run-all" else analyze(args.output)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
