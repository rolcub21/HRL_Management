#!/usr/bin/env python3
"""Corrected target-size Stage-A wrapper around the deployed VCG-Dense trainer."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

import torch

from experiments.conditioned_vcg.target_size_dense_stage_a_v2 import protocol as p
from experiments.conditioned_vcg.E13_operational_scalability_95k import program as e13
from experiments.conditioned_vcg.E14_certification_scalability_95k import reuse
from PSLAP.relocation_family_certification import RELOCATION_FAMILY_CERTIFICATION
import run_vcg_objective_gamma_audit as audit
import train_vcg_dense_proper as dense


def _read_contract(output: Path) -> dict:
    path = output / "contract.json"
    if not path.is_file():
        raise RuntimeError("run prepare before training")
    observed = json.loads(path.read_text(encoding="utf-8"))
    expected = p.contract()
    if observed != expected:
        raise RuntimeError("contract or authenticated source changed; use a new output version")
    return observed


def prepare(output: Path) -> dict:
    expected = p.contract()
    path = output / "contract.json"
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != expected:
        raise RuntimeError("existing Stage-A-v2 contract differs")
    if not path.exists():
        e13._atomic_json(path, expected)
    return {
        "status": "prepared",
        "contract_sha256": expected["contract_sha256"],
        "scenarios": list(p.SCENARIOS),
        "training_runs": 0,
        "test_panel_opened": False,
    }


def self_check() -> dict:
    rows = []
    for name in p.SCENARIOS:
        objective = dense.FROZEN_OBJECTIVE_SPEC
        env = p.TargetDenseEnv(name, objective)
        seed = p.VALIDATION_SEEDS[name][0]
        instance = env.sample_episode_instance(seed)
        env.reset(instance=instance)
        scenario = p.scenario_for(name)
        if env.timing_objective_spec.to_dict() != objective.to_dict():
            raise RuntimeError("dense timing objective was not installed")
        if env.time_steps != 0 or any(block.stored or block.delivered for block in env.blocks):
            raise RuntimeError("target environment is not empty-start")
        rows.append({
            "scenario": name,
            "seed": seed,
            "grid": [scenario.rows, scenario.cols],
            "jobs": scenario.total_jobs,
            "max_steps": scenario.max_steps,
            "arrival_first_last": [instance.arrival_steps[0], instance.arrival_steps[-1]],
            "objective": objective.objective,
        })
    return {"status": "passed", "rows": rows, "episodes_executed": 0}


def hardware() -> dict:
    cuda = bool(torch.cuda.is_available())
    return {
        "status": "available" if cuda else "cuda_unavailable",
        "torch": torch.__version__,
        "cuda_available": cuda,
        "cuda_device_count": int(torch.cuda.device_count()) if cuda else 0,
        "cuda_devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if cuda else [],
        "recommendation": "use --device cuda; do not launch the full target-size run on CPU",
    }


def _accelerated_enumerator(original):
    def wrapped(*args, **kwargs):
        existing = kwargs.get("recovery_certification_strategy")
        if existing is not None and existing != RELOCATION_FAMILY_CERTIFICATION:
            raise RuntimeError("conflicting recovery certification strategy")
        kwargs["recovery_certification_strategy"] = RELOCATION_FAMILY_CERTIFICATION
        return original(*args, **kwargs)
    return wrapped


def _destination(output: Path, scenario: str, model_seed: int) -> Path:
    return output / "training" / "operational" / scenario / f"seed-{model_seed}"


def train_stage_a(
    output: Path,
    scenario_name: str,
    model_seed: int,
    *,
    device_name: str,
    stop_after_episode: int | None,
    allow_cpu_pilot: bool,
) -> dict:
    contract = _read_contract(output)
    if model_seed not in p.MODEL_SEEDS:
        raise ValueError(f"model seed must be one of {p.MODEL_SEEDS}")
    scenario = p.scenario_for(scenario_name)
    resolved_cuda = torch.cuda.is_available()
    if device_name == "cuda" and not resolved_cuda:
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    if device_name == "auto" and not resolved_cuda:
        raise RuntimeError("auto resolved to CPU; target-size training requires CUDA")
    if device_name == "cpu" and not (allow_cpu_pilot and stop_after_episode is not None and stop_after_episode <= 2):
        raise RuntimeError("CPU is allowed only for an explicit <=2-episode diagnostic pilot")

    destination = _destination(output, scenario_name, model_seed)
    argv = [
        "--output-dir", str(destination),
        "--model-seed", str(model_seed),
        "--device", device_name,
        "--log-every", "1",
    ]
    if (destination / "latest.pth").is_file():
        argv.append("--resume-existing")
    if stop_after_episode is not None:
        argv.extend(("--stop-after-episode", str(stop_after_episode)))

    original_args = dense._frozen_audit_args
    original_contract = dense._resume_contract
    original_result = dense._result_payload

    def frozen_args(runtime):
        value = original_args(runtime)
        value.grid_rows = scenario.rows
        value.grid_cols = scenario.cols
        value.number_blocks = scenario.total_jobs
        value.arrival_rate = 0.1
        value.proc_mean = 80
        value.max_steps = scenario.max_steps
        value.validation_seeds = list(p.VALIDATION_SEEDS[scenario_name])
        value.eval_every = p.VALIDATION_EVERY
        value.checkpoint_every = p.CHECKPOINT_EVERY
        return value

    def resume_contract(*args, **kwargs):
        value = original_contract(*args, **kwargs)
        value.update({
            "training_protocol": p.PROTOCOL,
            "method_version": p.METHOD_VERSION,
            "target_contract_sha256": contract["contract_sha256"],
            "scenario": scenario_name,
            "instance_rule": "E28._instance: empty start, arrivals 10*j, Poisson(80)",
            "recovery_strategy": RELOCATION_FAMILY_CERTIFICATION,
            "certificate_cache": "E14 timing-invariant",
            "test_panel_opened": False,
        })
        return value

    def result_payload(*args, **kwargs):
        value = original_result(*args, **kwargs)
        value.update({
            "training_protocol": p.PROTOCOL,
            "method_version": p.METHOD_VERSION,
            "target_contract_sha256": contract["contract_sha256"],
            "scenario": scenario_name,
            "test_panel_opened": False,
        })
        return value

    frozen_defaults = dict(dense.FROZEN_AUDIT_DEFAULTS)
    frozen_defaults.update({
        "grid_rows": scenario.rows,
        "grid_cols": scenario.cols,
        "number_blocks": scenario.total_jobs,
        "arrival_rate": 0.1,
        "proc_mean": 80,
    })
    started = perf_counter()
    with ExitStack() as stack:
        stack.enter_context(patch.object(dense, "TRAINING_PROTOCOL", p.PROTOCOL))
        stack.enter_context(patch.object(dense, "METHOD_VERSION", p.METHOD_VERSION))
        stack.enter_context(patch.object(dense, "FROZEN_RECIPE_SOURCE", "deployed_vcg_dense_v1_1_plus_e28_target_geometry"))
        stack.enter_context(patch.object(dense, "FROZEN_TOTAL_EPISODES", p.TOTAL_EPISODES))
        stack.enter_context(patch.object(dense, "FROZEN_VALIDATION_SEEDS", tuple(p.VALIDATION_SEEDS[scenario_name])))
        stack.enter_context(patch.object(dense, "FROZEN_TRAIN_SEED_ORIGIN", p.TRAIN_BASE[scenario_name]))
        stack.enter_context(patch.object(dense, "FROZEN_EVAL_EVERY", p.VALIDATION_EVERY))
        stack.enter_context(patch.object(dense, "FROZEN_CHECKPOINT_EVERY", p.CHECKPOINT_EVERY))
        stack.enter_context(patch.object(dense, "FROZEN_MAX_STEPS", scenario.max_steps))
        stack.enter_context(patch.object(dense, "FROZEN_AUDIT_DEFAULTS", frozen_defaults))
        stack.enter_context(patch.object(dense, "_frozen_audit_args", frozen_args))
        stack.enter_context(patch.object(dense, "_resume_contract", resume_contract))
        stack.enter_context(patch.object(dense, "_result_payload", result_payload))
        stack.enter_context(patch.object(audit, "_new_env", lambda _args, objective: p.TargetDenseEnv(scenario_name, objective)))
        stack.enter_context(patch.object(audit, "ViabilityCertificateCache", reuse.TimingInvariantCertificateCache))
        stack.enter_context(patch.object(audit, "enumerate_viability_candidates", _accelerated_enumerator(audit.enumerate_viability_candidates)))
        stack.enter_context(reuse.path_cleanup_active())
        result = dense.main(argv)

    return {
        "status": result["status"],
        "scenario": scenario_name,
        "model_seed": model_seed,
        "completed_training_episodes": result["completed_training_episodes"],
        "wall_seconds_this_invocation": perf_counter() - started,
        "latest_checkpoint": result["latest_checkpoint"],
        "best_checkpoint": result["best_checkpoint"],
        "deployment_checkpoint_eligible": result["deployment_checkpoint_eligible"],
        "stage_b_authorized": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("hardware", "self-check", "prepare", "train-stage-a"))
    parser.add_argument("--output", type=Path, default=p.DEFAULT_OUTPUT)
    parser.add_argument("--scenario", choices=tuple(p.SCENARIOS))
    parser.add_argument("--model-seed", type=int, choices=p.MODEL_SEEDS)
    parser.add_argument("--device", choices=("cuda", "auto", "cpu"), default="cuda")
    parser.add_argument("--stop-after-episode", type=int)
    parser.add_argument("--allow-cpu-pilot", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "hardware":
        result = hardware()
    elif args.command == "self-check":
        result = self_check()
    elif args.command == "prepare":
        result = prepare(args.output)
    else:
        if args.scenario is None or args.model_seed is None:
            raise ValueError("train-stage-a requires --scenario and --model-seed")
        result = train_stage_a(
            args.output,
            args.scenario,
            args.model_seed,
            device_name=args.device,
            stop_after_episode=args.stop_after_episode,
            allow_cpu_pilot=args.allow_cpu_pilot,
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
