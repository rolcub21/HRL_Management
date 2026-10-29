"""Stage-A adapter for E209's scale-matched 10x10 workload."""

from __future__ import annotations

from contextlib import ExitStack
import json
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

import torch

import run_vcg_objective_gamma_audit as audit
import train_vcg_dense_proper as dense
from PSLAP.relocation_family_certification import RELOCATION_FAMILY_CERTIFICATION
from experiments.conditioned_vcg.E14_certification_scalability_95k import reuse
from projects.temporal_consequence_realization.method_conformance import (
    validate_protocol,
)

from . import protocol as p
from . import scenario as workload


REFERENCE_ARRIVAL_RATE = 1.0 / workload.GAPS[0.50]


def _read_contract(output: Path) -> dict:
    path = Path(output) / "contract.json"
    if not path.is_file():
        raise RuntimeError("run E209 prepare before training")
    observed = json.loads(path.read_text(encoding="utf-8"))
    expected = p.contract()
    if observed != expected:
        raise RuntimeError("E209 contract or authenticated source changed")
    validate_protocol(observed, source=p.PROTOCOL)
    return observed


def _accelerated_enumerator(original):
    def wrapped(*args, **kwargs):
        existing = kwargs.get("recovery_certification_strategy")
        if existing is not None and existing != RELOCATION_FAMILY_CERTIFICATION:
            raise RuntimeError("conflicting recovery certification strategy")
        kwargs["recovery_certification_strategy"] = RELOCATION_FAMILY_CERTIFICATION
        return original(*args, **kwargs)

    return wrapped


def _destination(output: Path, scenario: str, model_seed: int) -> Path:
    return (
        Path(output) / "training" / "operational" / scenario / f"seed-{model_seed}"
    )


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
    cuda = torch.cuda.is_available()
    if device_name == "cuda" and not cuda:
        raise RuntimeError("CUDA was requested but is unavailable")
    if device_name == "auto" and not cuda:
        raise RuntimeError("auto resolved to CPU; E209 training requires CUDA")
    if device_name == "cpu" and not (
        allow_cpu_pilot
        and stop_after_episode is not None
        and stop_after_episode <= 2
    ):
        raise RuntimeError("CPU is allowed only for an explicit <=2 episode pilot")

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
        value.arrival_rate = REFERENCE_ARRIVAL_RATE
        value.proc_mean = workload.MEAN_DURATION
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
            "instance_rule": (
                "E209 empty 10x10 n64; seed-balanced rho in {20,50,80}%; "
                "gap in {100,40,25}; max(1,Poisson(1260)) durations"
            ),
            "instance_specific_arrival_rate": True,
            "reference_parser_arrival_rate": REFERENCE_ARRIVAL_RATE,
            "proc_mean": workload.MEAN_DURATION,
            "recovery_strategy": RELOCATION_FAMILY_CERTIFICATION,
            "certificate_cache": "E14 timing-invariant",
            "fresh_network_initialization": True,
            "scale_matched_to_e191": True,
            "scale": "10x10",
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
            "fresh_network_initialization": True,
            "instance_specific_arrival_rate": True,
            "scale_matched_to_e191": True,
            "scale": "10x10",
            "test_panel_opened": False,
        })
        return value

    frozen_defaults = dict(dense.FROZEN_AUDIT_DEFAULTS)
    frozen_defaults.update({
        "grid_rows": scenario.rows,
        "grid_cols": scenario.cols,
        "number_blocks": scenario.total_jobs,
        "arrival_rate": REFERENCE_ARRIVAL_RATE,
        "proc_mean": workload.MEAN_DURATION,
    })
    started = perf_counter()
    with ExitStack() as stack:
        stack.enter_context(patch.object(dense, "TRAINING_PROTOCOL", p.PROTOCOL))
        stack.enter_context(patch.object(dense, "METHOD_VERSION", p.METHOD_VERSION))
        stack.enter_context(patch.object(
            dense,
            "FROZEN_RECIPE_SOURCE",
            "e191_exact_method_fresh_scale_matched_10x10",
        ))
        stack.enter_context(patch.object(
            dense, "FROZEN_TOTAL_EPISODES", p.TOTAL_EPISODES
        ))
        stack.enter_context(patch.object(
            dense, "FROZEN_VALIDATION_SEEDS",
            tuple(p.VALIDATION_SEEDS[scenario_name]),
        ))
        stack.enter_context(patch.object(
            dense, "FROZEN_TRAIN_SEED_ORIGIN", p.TRAIN_BASE[scenario_name]
        ))
        stack.enter_context(patch.object(
            dense, "FROZEN_EVAL_EVERY", p.VALIDATION_EVERY
        ))
        stack.enter_context(patch.object(
            dense, "FROZEN_CHECKPOINT_EVERY", p.CHECKPOINT_EVERY
        ))
        stack.enter_context(patch.object(
            dense, "FROZEN_MAX_STEPS", scenario.max_steps
        ))
        stack.enter_context(patch.object(
            dense, "FROZEN_AUDIT_DEFAULTS", frozen_defaults
        ))
        stack.enter_context(patch.object(dense, "_frozen_audit_args", frozen_args))
        stack.enter_context(patch.object(dense, "_resume_contract", resume_contract))
        stack.enter_context(patch.object(dense, "_result_payload", result_payload))
        stack.enter_context(patch.object(
            audit,
            "_new_env",
            lambda _args, objective: p.TargetDenseEnv(scenario_name, objective),
        ))
        stack.enter_context(patch.object(
            audit, "ViabilityCertificateCache", reuse.TimingInvariantCertificateCache
        ))
        stack.enter_context(patch.object(
            audit,
            "enumerate_viability_candidates",
            _accelerated_enumerator(audit.enumerate_viability_candidates),
        ))
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


__all__ = ["REFERENCE_ARRIVAL_RATE", "train_stage_a"]
