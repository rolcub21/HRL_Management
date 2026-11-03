#!/usr/bin/env python3
"""Train and evaluate E208's self-consistent future-rehandle head."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import copy
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
from experiments.conditioned_vcg.E204_e191_centered_rehandle_stage_b_263k_v1.centered import (
    CenteredFutureHandlingNetwork,
    CenteredHandlingAgent,
    CenteredHandlingConfig,
)
from experiments.conditioned_vcg.E205_e204_paired_tradeoff_development_264k_v1 import (
    protocol as e205,
    run as e205_run,
)
from projects.temporal_consequence_realization.method_conformance import (
    render_validated_conformance_audit,
    validate_protocol,
)
from train_viability_graph_smdp import resolve_device, seed_everything

from . import protocol as p
from .td import diagnostics, fit_episode_td, transitions_from_episode


class E208Error(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise E208Error(message)


def _read(path: Path) -> dict:
    require(Path(path).is_file(), f"missing E208 artifact: {path}")
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"invalid E208 JSON: {path}")
    return value


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False, default=str) + "\n")
    os.replace(temporary, path)


def _declaration(contract: dict) -> dict:
    return {key: contract[key] for key in (
        "method_reference", "method_revision", "claim_level", "declared_deviations",
        "oracle_inputs", "nonconforming_shortcuts", "conformance_audit",
    )}


def prepare(output: Path) -> dict:
    output = output.resolve()
    contract = p.contract()
    validate_protocol(contract, source=p.PROTOCOL)
    path = output / "contract.json"
    if path.is_file():
        require(_read(path) == contract, "E208 prepared contract changed")
    else:
        require(not output.exists() or not any(output.iterdir()), "E208 refuses nonempty uncontracted output")
        _write(path, contract)
        _write(output / "schedule.json", {
            "protocol": p.PROTOCOL, "contract_sha256": contract["contract_sha256"],
            "training_schedule": contract["training_schedule"],
        })
    return {
        "status": "prepared", "contract_sha256": contract["contract_sha256"],
        "training_episodes": p.TRAIN_EPISODES, "endpoint_evaluation_episodes": 9,
        "test_panel_opened": False, "paper_evidence": False,
    }


def authenticate(output: Path) -> dict:
    output = output.resolve()
    if not (output / "contract.json").is_file():
        prepare(output)
    contract = _read(output / "contract.json")
    require(contract.get("protocol") == p.PROTOCOL, "E208 protocol identity differs")
    payload = {key: value for key, value in contract.items() if key != "contract_sha256"}
    require(p.digest(payload) == contract.get("contract_sha256"), "E208 contract digest differs")
    render_validated_conformance_audit(_declaration(contract))
    for relative, expected in contract["source_sha256"].items():
        require(p.sha256(p.ROOT / relative) == expected, f"E208 source changed: {relative}")
    frozen = contract["frozen_stage_a"][0]
    require(p.sha256(Path(frozen["checkpoint"])) == frozen["checkpoint_sha256"], "E208 E191 checkpoint changed")
    require(
        p.sha256(e205.DEFAULT_OUTPUT / "instances.json")
        == contract["endpoint_evaluation"]["instances_sha256"],
        "E208 paired instance manifest changed",
    )
    return contract


def new_agent(base) -> CenteredHandlingAgent:
    config = CenteredHandlingConfig(
        feature_dim=int(base.Q_local.q_head[0].in_features),
        hidden_dim=base.config.head_hidden_dim,
        lambda_min=0.0, lambda_center=10.0, lambda_max=40.0,
        gamma_op=base.config.gamma, reward_scale=base.config.reward_scale,
    )
    seed_everything(p.HEAD_SEED)
    network = CenteredFutureHandlingNetwork(config, seed=p.HEAD_SEED).to(base.device)
    network.initialize_near_zero(0.01)
    return CenteredHandlingAgent(base, network, config=config, seed=p.HEAD_SEED + 1, epsilon=0.0)


def _target_network(agent, state=None):
    target = CenteredFutureHandlingNetwork(agent.config, seed=p.HEAD_SEED + 2).to(agent.device)
    target.load_state_dict(agent.handling_network.state_dict() if state is None else state)
    target.requires_grad_(False).eval()
    return target


def _checkpoint(agent, target, optimizer, contract, base_digest, records, role):
    frozen = contract["frozen_stage_a"][0]
    return {
        "training_protocol": p.PROTOCOL, "role": role,
        "contract_sha256": contract["contract_sha256"], "scenario": p.SCENARIO,
        "model_seed": p.MODEL_SEED, "completed_training_episodes": len(records),
        "base_checkpoint_sha256": frozen["checkpoint_sha256"],
        "base_policy_digest": base_digest, "episode_records": records,
        "agent_checkpoint": agent.checkpoint(
            base_checkpoint_sha256=frozen["checkpoint_sha256"],
            base_policy_digest=base_digest,
        ),
        "target_network_state": target.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "test_panel_opened": False, "paper_evidence": False,
    }


def _load_training(output, contract, device_name):
    device = resolve_device(device_name)
    base, search, liveness, base_digest, frozen = e204.load_base(
        contract, p.MODEL_SEED, device
    )
    destination = output.resolve() / "training" / f"seed-{p.MODEL_SEED}"
    latest = destination / "latest.pth"
    if latest.is_file():
        saved = torch.load(latest, map_location="cpu", weights_only=False)
        expected = {
            "training_protocol": p.PROTOCOL, "role": "latest",
            "contract_sha256": contract["contract_sha256"],
            "model_seed": p.MODEL_SEED,
            "base_checkpoint_sha256": frozen["checkpoint_sha256"],
            "base_policy_digest": base_digest,
        }
        mismatch = {key: (saved.get(key), value) for key, value in expected.items() if saved.get(key) != value}
        require(not mismatch, f"E208 latest identity differs: {mismatch}")
        agent = CenteredHandlingAgent.from_checkpoint(
            saved["agent_checkpoint"], base_agent=base,
            expected_base_checkpoint_sha256=frozen["checkpoint_sha256"],
            expected_base_policy_digest=base_digest, seed=p.HEAD_SEED + 1,
        )
        target = _target_network(agent, saved["target_network_state"])
        optimizer = torch.optim.Adam(agent.handling_network.parameters(), lr=p.OPTIMIZER_LR)
        optimizer.load_state_dict(saved["optimizer_state"])
        records = list(saved["episode_records"])
    else:
        require(not destination.exists() or not any(destination.iterdir()), "E208 nonempty training directory lacks latest")
        agent = new_agent(base)
        target = _target_network(agent)
        optimizer = torch.optim.Adam(agent.handling_network.parameters(), lr=p.OPTIMIZER_LR)
        records = []
    return agent, target, optimizer, records, search, liveness, base_digest, destination


def _episode_artifact(destination: Path, number: int) -> Path:
    return destination / "episodes" / f"episode-{number:03d}.pth"


def collect(agent, env, coordinate, contract, destination, search, liveness):
    number = int(coordinate["episode"])
    path = _episode_artifact(destination, number)
    head_digest = e204.state_digest(agent.handling_network.state_dict())
    identity = {
        "contract_sha256": contract["contract_sha256"], "episode": number,
        "instance_seed": int(coordinate["instance_seed"]),
        "workload": float(coordinate["workload"]), "alpha": float(coordinate["alpha"]),
        "behavior_head_sha256": head_digest,
    }
    if path.is_file():
        saved = torch.load(path, map_location="cpu", weights_only=False)
        require(all(saved.get(key) == value for key, value in identity.items()), f"E208 episode artifact differs: {path}")
        return saved["row"], tuple(saved["samples"]), False
    agent.base_agent.bind_context_environment(
        env, required_jobs=e191.scenario_for(p.SCENARIO).total_jobs,
        max_steps=e191.MAX_STEPS,
    )
    label = f"episode={number}/{p.TRAIN_EPISODES} rho={100*coordinate['workload']:.0f}% alpha={coordinate['alpha']:.3f}"
    print(f"E208 train start {label}", flush=True)
    started = perf_counter()
    with e205_run.heartbeat(f"E208 {label}"), ExitStack() as stack:
        e204._runtime_stack(stack)
        result = common.run_preference_episode(
            agent, env, instance_seed=int(coordinate["instance_seed"]),
            preference_lambda=float(coordinate["alpha"]),
            preference_schedule_record={
                "behavior_alpha": float(coordinate["alpha"]),
                "alpha_fixed_for_complete_episode": True,
                "target_contract": contract["bellman_target"],
                "episode": number, "workload": float(coordinate["workload"]),
            },
            max_steps=e191.MAX_STEPS, search_config=search, liveness_rule=liveness,
            epsilon_start=0.0, epsilon_end=0.0, epsilon_decay_decisions=1,
            epsilon_warmup_decisions=0, updates_per_macro=1,
        )
    samples = tuple(item.numeric_copy() for item in agent.pop_completed_samples())
    # The frozen E191 audit log is useful within an episode but is not part of
    # the Stage-B state.  Keeping it across long episodes only consumes memory.
    agent.base_agent.decision_log.clear()
    metrics = env.e187_metrics()
    row = {
        "episode": number, "instance_seed": int(coordinate["instance_seed"]),
        "workload": float(coordinate["workload"]), "alpha": float(coordinate["alpha"]),
        "strict_complete": bool(result["strict_method_success"]),
        "failure": result["method_failure_reason"], "J10": float(metrics["J"]),
        "rehandles": int(result["physical_rehandles"]), "steps": int(result["steps"]),
        "macro_decisions": int(result["macro_decisions"]),
        "peak_stored": int(metrics["peak_stored"]), "wall_seconds": perf_counter()-started,
    }
    require(row["strict_complete"] and len(samples) == row["macro_decisions"], "E208 collection did not complete")
    common._atomic_torch_save({**identity, "row": row, "samples": samples}, path)
    print(f"E208 train done {label} J10={row['J10']:.1f} R={row['rehandles']} wall={row['wall_seconds']:.1f}s", flush=True)
    return row, samples, True


def train(output: Path, through_episode: int, device_name: str) -> dict:
    contract = authenticate(output)
    require(1 <= int(through_episode) <= p.TRAIN_EPISODES, "E208 through-episode must be 1..36")
    agent, target, optimizer, records, search, liveness, base_digest, destination = _load_training(
        output, contract, device_name
    )
    require(len(records) <= through_episode, "E208 cannot train backward")
    env = e191.TargetDenseEnv(p.SCENARIO, e191.SPEC)
    new_runs = 0
    schedule = contract["training_schedule"]
    for coordinate in schedule[len(records):through_episode]:
        before = e204.state_digest(agent.handling_network.state_dict())
        row, samples, executed = collect(
            agent, env, coordinate, contract, destination, search, liveness
        )
        transitions = transitions_from_episode(samples)
        fit = fit_episode_td(
            agent.handling_network, target, optimizer, transitions,
            device=agent.device, epochs=p.TD_EPOCHS_PER_EPISODE,
            batch_size=p.TD_BATCH_SIZE,
            seed=p.HEAD_SEED + int(coordinate["episode"]),
            target_tau=p.TARGET_TAU,
        )
        after = e204.state_digest(agent.handling_network.state_dict())
        require(after != before, "E208 TD update did not change the head")
        require(e204.state_digest(agent.base_agent.Q_local.state_dict()) == base_digest, "E208 mutated E191")
        records.append({**row, "head_before_sha256": before, "head_after_sha256": after, "td_fit": fit})
        payload = _checkpoint(agent, target, optimizer, contract, base_digest, records, "latest")
        common._atomic_torch_save(payload, destination / "latest.pth")
        if len(records) in p.CHECKPOINT_EPISODES:
            endpoint = {**payload, "role": "endpoint", "endpoint_episode": len(records)}
            common._atomic_torch_save(endpoint, destination / "checkpoints" / f"episode-{len(records):03d}.pth")
        _write(destination / "training-summary.json", {
            "status": "complete" if len(records) == p.TRAIN_EPISODES else "paused",
            "training_protocol": p.PROTOCOL, "contract_sha256": contract["contract_sha256"],
            "model_seed": p.MODEL_SEED, "completed_training_episodes": len(records),
            "base_checkpoint_sha256": contract["frozen_stage_a"][0]["checkpoint_sha256"],
            "base_policy_digest": base_digest, "episode_records": records,
            "test_panel_opened": False, "paper_evidence": False,
        })
        new_runs += int(executed)
        print(
            f"E208 update episode={len(records)} tdMAE={fit['after']['td_mae']:.3f} "
            f"mcMAE={fit['after']['mc_mae']:.3f} predMean={fit['after']['prediction_mean']:.2f}",
            flush=True,
        )
    return {
        "status": "complete" if len(records) == p.TRAIN_EPISODES else "paused",
        "completed_training_episodes": len(records), "new_episode_runs": new_runs,
        "latest": str(destination / "latest.pth"), "test_panel_opened": False,
    }


def _load_endpoint(output: Path, contract: dict, device_name: str):
    device = resolve_device(device_name)
    base, search, liveness, base_digest, frozen = e204.load_base(contract, p.MODEL_SEED, device)
    path = output.resolve() / "training" / f"seed-{p.MODEL_SEED}" / "checkpoints" / "episode-036.pth"
    require(path.is_file(), "E208 requires the episode-36 endpoint")
    saved = torch.load(path, map_location="cpu", weights_only=False)
    require(saved.get("role") == "endpoint" and saved.get("endpoint_episode") == 36, "E208 endpoint identity differs")
    agent = CenteredHandlingAgent.from_checkpoint(
        saved["agent_checkpoint"], base_agent=base,
        expected_base_checkpoint_sha256=frozen["checkpoint_sha256"],
        expected_base_policy_digest=base_digest, seed=p.HEAD_SEED + 1,
    )
    return agent, search, liveness, base_digest, path


def _eval_path(output, seed, value):
    rho = int(round(100*e205.workload_for_seed(seed)))
    return output.resolve() / "evaluation" / f"rho-{rho}" / f"lambda-{value:g}.json"


def evaluate(output: Path, device_name: str) -> dict:
    contract = authenticate(output)
    agent, search, liveness, base_digest, endpoint = _load_endpoint(output, contract, device_name)
    manifest = _read(e205.DEFAULT_OUTPUT / "instances.json")
    env = e191.TargetDenseEnv(p.SCENARIO, e191.SPEC)
    new_runs = 0
    for seed in e205.INSTANCE_SEEDS:
        instance = e205_run.frozen_instance(manifest, seed)
        for value in p.EVAL_LAMBDAS:
            path = _eval_path(output, seed, value)
            if path.is_file():
                continue
            agent.base_agent.bind_context_environment(env, required_jobs=e191.scenario_for(p.SCENARIO).total_jobs, max_steps=e191.MAX_STEPS)
            before = e204.state_digest(agent.handling_network.state_dict())
            label = f"rho={100*e205.workload_for_seed(seed):.0f}% lambda={value:g}"
            print(f"E208 eval start {label}", flush=True)
            started = perf_counter()
            with e205_run.heartbeat(f"E208 eval {label}"), ExitStack() as stack:
                e204._runtime_stack(stack)
                result = common.run_preference_episode(
                    agent, env, instance_seed=int(seed), preference_lambda=float(value),
                    preference_schedule_record={"behavior_lambda": value, "endpoint_episode": 36, "fixed_for_complete_episode": True},
                    max_steps=e191.MAX_STEPS, search_config=search, liveness_rule=liveness,
                    epsilon_start=0.0, epsilon_end=0.0, epsilon_decay_decisions=1,
                    epsilon_warmup_decisions=0, updates_per_macro=1,
                )
            samples = tuple(agent.pop_completed_samples())
            agent.base_agent.decision_log.clear()
            require(e204.state_digest(agent.handling_network.state_dict()) == before, "E208 evaluation mutated head")
            metrics = env.e187_metrics(); components = metrics["components"]
            transitions = transitions_from_episode(samples)
            calibration = diagnostics(agent.handling_network, transitions, device=agent.device)
            rehandles = int(result["physical_rehandles"]); j10 = float(metrics["J"])
            row = {
                "protocol": p.PROTOCOL, "contract_sha256": contract["contract_sha256"],
                "endpoint_sha256": p.sha256(endpoint), "instance_seed": int(seed),
                "instance_id": instance.instance_id, "workload": e205.workload_for_seed(seed),
                "lambda": float(value), "strict_complete": bool(result["strict_method_success"]),
                "J10": j10, "J_lambda": j10 + (value-10.0)*rehandles,
                "physical_rehandles": rehandles,
                "timing_cost_2T_plus_E": 2*float(components["storage_tardiness_accrual"])+float(components["storage_earliness_at_delivery"]),
                "admission_wait_cost_0p5W": 0.5*float(components["admission_wait_accrual"]),
                "equipment_cost_0p1L": 0.1*float(components["equipment_work"]),
                "steps": int(result["steps"]), "peak_stored": int(metrics["peak_stored"]),
                "calibration": calibration, "wall_seconds": perf_counter()-started,
                "test_panel_opened": False, "paper_evidence": False,
            }
            require(row["strict_complete"], f"E208 evaluation failed: {row}")
            _write(path, row); new_runs += 1
            print(f"E208 eval done {label} Jlambda={row['J_lambda']:.1f} R={rehandles} timing={row['timing_cost_2T_plus_E']:.1f}", flush=True)
    return {"status": "complete", "rows": 9, "new_episode_runs": new_runs, "test_panel_opened": False}


def analyze(output: Path) -> dict:
    contract = authenticate(output)
    rows = [_read(_eval_path(output, seed, value)) for seed in e205.INSTANCE_SEEDS for value in p.EVAL_LAMBDAS]
    comparisons = []
    for row in rows:
        if row["lambda"] == 10.0:
            continue
        immediate_path = (
            p.ROOT / "results/vcg-conditioned-e207-e204-immediate-only-ablation-266k-v1" /
            "rows" / f"rho-{int(round(100*row['workload']))}" / f"lambda-{row['lambda']:g}.json"
        )
        immediate = _read(immediate_path)
        comparisons.append({
            "workload": row["workload"], "lambda": row["lambda"],
            "new_R": row["physical_rehandles"], "immediate_R": immediate["physical_rehandles"],
            "new_timing": row["timing_cost_2T_plus_E"], "immediate_timing": immediate["timing_cost_2T_plus_E"],
            "new_J_lambda": row["J_lambda"], "immediate_J_lambda": immediate["J_lambda"],
            "delta_new_minus_immediate_J_lambda": row["J_lambda"]-immediate["J_lambda"],
        })
    price_rows = []
    for workload in sorted({row["workload"] for row in rows}):
        selected = [row for row in rows if row["workload"] == workload]
        for price in p.EVAL_LAMBDAS:
            scores = {row["lambda"]: row["J10"]+(price-10.0)*row["physical_rehandles"] for row in selected}
            best = min(scores.values()); diagonal = scores[price]
            price_rows.append({
                "workload": workload, "price": price,
                "best_behavior_lambdas": [key for key, score in scores.items() if math.isclose(score, best, abs_tol=1e-9)],
                "diagonal_regret": diagonal-best,
                "diagonal_relative_regret": (diagonal-best)/max(abs(best), 1.0),
            })
    report = {
        "status": "complete", "protocol": p.PROTOCOL,
        "contract_sha256": contract["contract_sha256"], "evaluation_rows": rows,
        "paired_immediate_comparisons": comparisons, "price_regret": price_rows,
        "summary": {
            "strict_completion_rate": fmean(float(row["strict_complete"]) for row in rows),
            "lower_J_than_immediate_rate": fmean(float(row["delta_new_minus_immediate_J_lambda"] < 0) for row in comparisons),
            "price_exact_rate": fmean(float(math.isclose(row["diagonal_regret"], 0.0, abs_tol=1e-9)) for row in price_rows),
            "price_near_5pct_rate": fmean(float(row["diagonal_relative_regret"] <= 0.05) for row in price_rows),
            "mean_heldout_td_mae": fmean(row["calibration"]["td_mae"] for row in rows),
            "mean_heldout_mc_mae": fmean(row["calibration"]["mc_mae"] for row in rows),
        },
        "scope": contract["scope"], "test_panel_opened": False, "paper_evidence": False,
    }
    _write(output.resolve()/"e208-report.json", report)
    declaration = render_validated_conformance_audit(_declaration(contract))
    summary = report["summary"]
    text = declaration + "\nRESULTS\n" + "="*72 + "\n\n" + "\n".join([
        f"Strict completion: {100*summary['strict_completion_rate']:.1f}%",
        f"Lower J_lambda than immediate-only: {100*summary['lower_J_than_immediate_rate']:.1f}%",
        f"Exact lambda=w rate: {100*summary['price_exact_rate']:.1f}%",
        f"Within-5% lambda=w rate: {100*summary['price_near_5pct_rate']:.1f}%",
        f"Mean held-out TD MAE: {summary['mean_heldout_td_mae']:.3f}",
        f"Mean held-out MC MAE: {summary['mean_heldout_mc_mae']:.3f}",
        "", "Development pilot only; multiseed evidence is not authorized.", "",
    ])
    (output.resolve()/"REPORT.txt").write_text(text, encoding="utf-8")
    from .plot import create_plot
    create_plot(rows, output.resolve()/"e208-endpoint.png")
    return {"status": "complete", **summary, "report": str(output.resolve()/"REPORT.txt"), "plot": str(output.resolve()/"e208-endpoint.png"), "test_panel_opened": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "train-seed", "evaluate", "analyze"))
    parser.add_argument("--output", type=Path, default=p.DEFAULT_OUTPUT)
    parser.add_argument("--through-episode", type=int, default=p.TRAIN_EPISODES)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cuda")
    args = parser.parse_args()
    if args.command == "prepare": result = prepare(args.output)
    elif args.command == "train-seed": result = train(args.output, args.through_episode, args.device)
    elif args.command == "evaluate": result = evaluate(args.output, args.device)
    else: result = analyze(args.output)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
