#!/usr/bin/env python3
"""Target-yard VCG training, kept separate from the zero-shot investigation.

Only explicitly requested commands launch training or evaluation. Preparation
and self-check do not run an episode or update a weight.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import random
from typing import Mapping
from unittest.mock import patch

import torch

from experiments.conditioned_vcg.target_size_vcg_training import protocol as p

import benchmark_viability_critic_priority as benchmark
from experiments.conditioned_vcg.E13_operational_scalability_95k import program as e13
from experiments.conditioned_vcg.E14_certification_scalability_95k import reuse
from PSLAP.relocation_family_certification import RELOCATION_FAMILY_CERTIFICATION
import train_viability_graph_smdp as smdp
import train_viability_graph_smdp_proper as proper
import train_vcg_preference_conditioned as common
from train_viability_graph_smdp import resolve_device, seed_everything
from vcg_v11_conditioned_handling import (
    ConditionedFutureHandlingNetwork,
    ConditionedHandlingAgent,
    ConditionedHandlingConfig,
    fit_conditioned_future_handling,
    handling_calibration,
)
from viability_graph_hierarchy import ViabilityGraphHierarchyAgent

STAGE_A_PROTOCOL = p.PROTOCOL + "_stage_a"
STAGE_B_PROTOCOL = p.PROTOCOL + "_stage_b"
NO_SOURCE_COST = "random_initialization_no_source_cost_head_v1"


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
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != expected:
            raise RuntimeError("existing target-size contract differs")
    else:
        e13._atomic_json(path, expected)
    return {
        "status": "prepared", "contract_sha256": expected["contract_sha256"],
        "scenarios": list(p.SCENARIOS), "training_runs": 0,
        "test_panel_opened": False,
    }


def self_check() -> dict:
    """Check exact E28 instances and reset invariants without model inference."""
    p.check_splits()
    rows = []
    for name in p.SCENARIOS:
        env = p.TargetSizeEnv(name)
        seed = p.PILOT_SEEDS[name][0]
        instance = env.sample_episode_instance(seed)
        env.reset(instance=instance)
        first = env.blocks[0]
        if (
            env.time_steps != 0
            or first.position != env.pickup_cell
            or any(block.stored or block.delivered for block in env.blocks)
        ):
            raise RuntimeError("target-size reset is not empty-start")
        rows.append({
            "scenario": name, "seed": seed,
            "instance_id": instance.instance_id,
            "jobs": instance.number_blocks,
            "arrivals_first_last": [instance.arrival_steps[0], instance.arrival_steps[-1]],
            "storage_capacity": len(env.storage_positions),
            "exit_cells": [list(cell) for cell in env.exit_cells],
        })
    return {"status": "passed", "rows": rows, "episodes_executed": 0}


def _accelerated_enumerator(original):
    def wrapped(*args, **kwargs):
        existing = kwargs.get("recovery_certification_strategy")
        if existing is not None and existing != RELOCATION_FAMILY_CERTIFICATION:
            raise RuntimeError("conflicting recovery certification strategy")
        kwargs["recovery_certification_strategy"] = RELOCATION_FAMILY_CERTIFICATION
        return original(*args, **kwargs)
    return wrapped


def _stage_a_path(output: Path, name: str, model_seed: int) -> Path:
    return output / "training" / "operational" / name / f"seed-{model_seed}"


def _stage_b_path(output: Path, name: str, model_seed: int) -> Path:
    return output / "training" / "handling" / name / f"seed-{model_seed}"


def train_stage_a(
    output: Path, name: str, model_seed: int, *,
    device_name: str, stop_after_episode: int | None,
) -> dict:
    contract = _read_contract(output)
    scenario = p.scenario_for(name)
    base = p.training_seed(name, model_seed)
    destination = _stage_a_path(output, name, model_seed)
    arguments = [
        "--output-dir", str(destination),
        "--model-seed", str(model_seed),
        "--total-episodes", str(p.OPERATIONAL_EPISODES),
        "--train-instance-seed-base", str(base),
        "--validation-seeds", *(str(seed) for seed in p.VALIDATION_SEEDS[name]),
        "--eval-every", str(p.VALIDATION_EVERY),
        "--checkpoint-every", "10",
        "--grid-rows", str(scenario.rows),
        "--grid-cols", str(scenario.cols),
        "--number-blocks", str(scenario.total_jobs),
        "--arrival-rate", "0.1", "--proc-mean", "80",
        "--max-steps", str(scenario.max_steps),
        "--search-max-nodes", "20000",
        "--search-order", "goal_directed",
        "--device", device_name,
    ]
    if (destination / "latest.pth").is_file():
        arguments.extend(("--resume", str(destination / "latest.pth")))
    if stop_after_episode is not None:
        arguments.extend(("--stop-after-episode", str(stop_after_episode)))

    original_resume = proper._resume_contract
    original_result = proper._result_payload

    def resume_contract(*args, **kwargs):
        value = original_resume(*args, **kwargs)
        value.update({
            "training_protocol": STAGE_A_PROTOCOL,
            "target_contract_sha256": contract["contract_sha256"],
            "scenario": name,
            "instance_rule": "E28._instance, empty start, arrivals 10*j",
            "recovery_strategy": RELOCATION_FAMILY_CERTIFICATION,
            "certificate_cache": "E14 timing-invariant",
        })
        return value

    def result_payload(*args, **kwargs):
        value = original_result(*args, **kwargs)
        value.update({
            "training_protocol": STAGE_A_PROTOCOL,
            "target_contract_sha256": contract["contract_sha256"],
            "scenario": name, "test_panel_opened": False,
            "canonical_500_episode_replication_complete": False,
        })
        return value

    with ExitStack() as stack:
        stack.enter_context(patch.object(proper, "TRAINING_PROTOCOL", STAGE_A_PROTOCOL))
        stack.enter_context(patch.object(proper, "_new_env", lambda _args: p.TargetSizeEnv(name)))
        stack.enter_context(patch.object(proper, "_resume_contract", resume_contract))
        stack.enter_context(patch.object(proper, "_result_payload", result_payload))
        stack.enter_context(patch.object(proper, "ViabilityCertificateCache", reuse.TimingInvariantCertificateCache))
        stack.enter_context(patch.object(smdp, "enumerate_viability_candidates", _accelerated_enumerator(smdp.enumerate_viability_candidates)))
        stack.enter_context(reuse.path_cleanup_active())
        result = proper.main(arguments)
    return {
        "status": result["status"], "scenario": name, "model_seed": model_seed,
        "completed_training_episodes": result["completed_training_episodes"],
        "best_checkpoint": result["best_checkpoint"],
        "deployment_checkpoint_eligible": result["deployment_checkpoint_eligible"],
    }


def _state_digest(state: Mapping[str, torch.Tensor]) -> str:
    import hashlib
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _load_base(output: Path, name: str, model_seed: int, device: torch.device):
    summary_path = _stage_a_path(output, name, model_seed) / "training-summary.json"
    if not summary_path.is_file():
        raise RuntimeError("Stage A has no training summary")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if (
        summary.get("status") != "complete"
        or summary.get("completed_training_episodes") != p.OPERATIONAL_EPISODES
        or summary.get("deployment_checkpoint_eligible") is not True
        or summary.get("target_contract_sha256") != _read_contract(output)["contract_sha256"]
    ):
        raise RuntimeError("Stage A has not completed its frozen training budget")
    path = _stage_a_path(output, name, model_seed) / "best.pth"
    sha = p.file_sha256(path)
    if summary.get("best_checkpoint_sha256") != sha:
        raise RuntimeError("Stage A best checkpoint hash changed")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("resume_contract", {}).get("scenario") != name:
        raise RuntimeError("Stage A checkpoint has another target yard")
    base = ViabilityGraphHierarchyAgent.from_checkpoint(
        payload, device=device, resumable=False, seed=model_seed,
    )
    base.set_epsilon(0.0)
    for network in (base.Q_local, base.Q_target):
        network.requires_grad_(False).eval()
    return payload, base, sha, _state_digest(base.Q_local.state_dict())


def _lambda_schedule(name: str, model_seed: int, round_index: int) -> list[float]:
    rng = random.Random(p.handling_seed(name, model_seed, round_index, 0) + 500_000)
    values = [
        0.2 * (index + rng.random()) / p.HANDLING_EPISODES_PER_ROUND
        for index in range(p.HANDLING_EPISODES_PER_ROUND)
    ]
    rng.shuffle(values)
    return values


def _stage_b_checkpoint(
    agent: ConditionedHandlingAgent, *, role: str,
    contract: dict, name: str, model_seed: int,
    base_sha: str, base_digest: str, round_records: list[dict],
) -> dict:
    return {
        "training_protocol": STAGE_B_PROTOCOL,
        "role": role,
        "target_contract_sha256": contract["contract_sha256"],
        "scenario": name, "model_seed": model_seed,
        "base_checkpoint_sha256": base_sha,
        "base_policy_digest": base_digest,
        "completed_rounds": len(round_records),
        "round_records": round_records,
        "agent_checkpoint": agent.checkpoint(
            base_checkpoint_sha256=base_sha,
            base_policy_digest=base_digest,
            source_cost_sha256=NO_SOURCE_COST,
        ),
    }


def train_stage_b(
    output: Path, name: str, model_seed: int, *,
    device_name: str, stop_after_round: int | None,
) -> dict:
    contract = _read_contract(output)
    scenario = p.scenario_for(name)
    device = resolve_device(device_name)
    base_payload, base, base_sha, base_digest = _load_base(output, name, model_seed, device)
    search = benchmark._search_config(base_payload)
    liveness = benchmark._liveness_rule(base_payload)
    config = ConditionedHandlingConfig(
        feature_dim=3 * base.config.graph_embedding_dim + base.config.action_embedding_dim,
        hidden_dim=base.config.head_hidden_dim,
        lambda_max=0.2, gamma_op=base.config.gamma,
        reward_scale=base.config.reward_scale,
    )
    destination = _stage_b_path(output, name, model_seed)
    latest_path = destination / "latest.pth"
    terminal_path = destination / "terminal.pth"
    if latest_path.is_file():
        saved = torch.load(latest_path, map_location="cpu", weights_only=False)
        expected = {
            "training_protocol": STAGE_B_PROTOCOL,
            "role": "latest", "target_contract_sha256": contract["contract_sha256"],
            "scenario": name, "model_seed": model_seed,
            "base_checkpoint_sha256": base_sha, "base_policy_digest": base_digest,
        }
        if any(saved.get(key) != value for key, value in expected.items()):
            raise RuntimeError("handling resume checkpoint contract changed")
        round_records = list(saved["round_records"])
        if len(round_records) != saved["completed_rounds"]:
            raise RuntimeError("handling round count is inconsistent")
        agent = ConditionedHandlingAgent.from_checkpoint(
            saved["agent_checkpoint"], base_agent=base,
            expected_base_checkpoint_sha256=base_sha,
            expected_base_policy_digest=base_digest,
            expected_source_cost_sha256=NO_SOURCE_COST,
            seed=model_seed,
        )
    else:
        if destination.exists() and any(destination.iterdir()):
            raise RuntimeError("nonempty handling output has no latest checkpoint")
        destination.mkdir(parents=True, exist_ok=True)
        seed_everything(model_seed)
        network = ConditionedFutureHandlingNetwork(config, seed=model_seed).to(device)
        agent = ConditionedHandlingAgent(base, network, config=config, seed=model_seed)
        round_records = []

    stop_at = stop_after_round or p.HANDLING_ROUNDS
    if not 1 <= stop_at <= p.HANDLING_ROUNDS or len(round_records) > stop_at:
        raise ValueError("invalid handling stop round")
    env = p.TargetSizeEnv(name)
    with ExitStack() as stack:
        stack.enter_context(patch.object(common, "enumerate_viability_candidates", _accelerated_enumerator(common.enumerate_viability_candidates)))
        stack.enter_context(patch.object(common, "ViabilityCertificateCache", reuse.TimingInvariantCertificateCache))
        stack.enter_context(reuse.path_cleanup_active())
        for round_index in range(len(round_records), stop_at):
            values = _lambda_schedule(name, model_seed, round_index)
            episode_samples = []
            rows = []
            before = _state_digest(agent.handling_network.state_dict())
            for offset, value in enumerate(values):
                seed = p.handling_seed(name, model_seed, round_index, offset)
                run = common.run_preference_episode(
                    agent, env, instance_seed=seed,
                    preference_lambda=value,
                    preference_schedule_record={
                        "behavior_lambda": value, "round": round_index + 1,
                        "episode": offset + 1, "fixed_for_complete_episode": True,
                    },
                    max_steps=scenario.max_steps,
                    search_config=search, liveness_rule=liveness,
                    epsilon_start=0.0, epsilon_end=0.0,
                    epsilon_decay_decisions=1, epsilon_warmup_decisions=0,
                    updates_per_macro=1,
                )
                row = {
                    "seed": seed, "lambda": value,
                    "strict_safe_complete": bool(run["strict_method_success"]),
                    "failure": run["method_failure_reason"],
                    "return": run["return"], "steps": run["steps"],
                    "rehandles": run["physical_rehandles"],
                    "macro_decisions": run["macro_decisions"],
                    "instance_id": run["episode_instance_id"],
                }
                if not row["strict_safe_complete"]:
                    e13._atomic_json(destination / "failed-collection.json", {
                        "round": round_index + 1, "episode": offset + 1,
                        "row": row, "completed_prior_rounds": len(round_records),
                    })
                    raise RuntimeError(
                        "Stage B requires completed MC continuations; "
                        "a collection failed, so this version stops without fitting"
                    )
                samples = agent.pop_completed_samples()
                if len(samples) != run["macro_decisions"]:
                    raise RuntimeError("handling sample count differs from decisions")
                episode_samples.append(samples)
                rows.append(row)
                agent.base_agent.decision_log.clear()
                if offset == 0 or (offset + 1) % 10 == 0:
                    print(
                        f"Stage B {name} seed={model_seed} round={round_index + 1} "
                        f"episode={offset + 1}/{len(values)} strict=1",
                        flush=True,
                    )
            order = list(range(len(rows)))
            random.Random(115_000_000 + model_seed * 1000 + round_index).shuffle(order)
            validation_indices = set(order[:10])
            training = tuple(
                sample for i, episode in enumerate(episode_samples)
                if i not in validation_indices for sample in episode
            )
            validation = tuple(
                sample for i, episode in enumerate(episode_samples)
                if i in validation_indices for sample in episode
            )
            prior_state = {
                key: value.detach().clone()
                for key, value in agent.handling_network.state_dict().items()
            }
            fit = fit_conditioned_future_handling(
                agent.handling_network, training, validation,
                device=device, epochs=8, batch_size=256,
                learning_rate=5.0e-4,
                seed=116_000_000 + model_seed * 1000 + round_index,
            )
            update_fraction = (
                1.0 if round_index < 4 else p.HANDLING_LATE_UPDATE_FRACTION
            )
            if update_fraction != 1.0:
                fitted = agent.handling_network.state_dict()
                agent.handling_network.load_state_dict({
                    key: (1.0 - update_fraction) * prior_state[key]
                    + update_fraction * fitted[key]
                    for key in fitted
                })
            fit["applied_update_fraction"] = update_fraction
            fit["applied_validation"] = handling_calibration(
                agent.handling_network, validation, device=device
            )
            if _state_digest(base.Q_local.state_dict()) != base_digest:
                raise RuntimeError("Stage B mutated the frozen operational critic")
            after = _state_digest(agent.handling_network.state_dict())
            if after == before:
                raise RuntimeError("handling fit did not update weights")
            round_records.append({
                "round": round_index + 1, "episodes": rows,
                "training_samples": len(training),
                "validation_samples": len(validation),
                "validation_episode_indices": sorted(validation_indices),
                "network_before_sha256": before,
                "network_after_sha256": after,
                "fit": fit,
            })
            common._atomic_torch_save(
                _stage_b_checkpoint(
                    agent, role="latest", contract=contract,
                    name=name, model_seed=model_seed, base_sha=base_sha,
                    base_digest=base_digest, round_records=round_records,
                ),
                latest_path,
            )
            e13._atomic_json(destination / "training-summary.json", {
                "status": "complete" if len(round_records) == p.HANDLING_ROUNDS else "paused",
                "target_contract_sha256": contract["contract_sha256"],
                "scenario": name, "model_seed": model_seed,
                "completed_rounds": len(round_records),
                "base_checkpoint_sha256": base_sha,
                "round_records": round_records,
            })
    if len(round_records) == p.HANDLING_ROUNDS:
        common._atomic_torch_save(
            _stage_b_checkpoint(
                agent, role="terminal", contract=contract,
                name=name, model_seed=model_seed, base_sha=base_sha,
                base_digest=base_digest, round_records=round_records,
            ),
            terminal_path,
        )
    return {
        "status": "complete" if len(round_records) == p.HANDLING_ROUNDS else "paused",
        "scenario": name, "model_seed": model_seed,
        "completed_rounds": len(round_records),
        "terminal_checkpoint": str(terminal_path) if terminal_path.is_file() else None,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=(
        "self-check", "prepare", "train-stage-a", "train-stage-b",
    ))
    parser.add_argument("--output", type=Path, default=p.DEFAULT_OUTPUT)
    parser.add_argument("--scenario", choices=tuple(p.SCENARIOS))
    parser.add_argument("--model-seed", type=int, choices=p.MODEL_SEEDS)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    parser.add_argument("--stop-after-episode", type=int)
    parser.add_argument("--stop-after-round", type=int)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "self-check":
        result = self_check()
    elif args.command == "prepare":
        result = prepare(args.output.resolve())
    else:
        if args.scenario is None or args.model_seed is None:
            raise SystemExit("training requires --scenario and --model-seed")
        if args.command == "train-stage-a":
            result = train_stage_a(
                args.output.resolve(), args.scenario, args.model_seed,
                device_name=args.device,
                stop_after_episode=args.stop_after_episode,
            )
        else:
            result = train_stage_b(
                args.output.resolve(), args.scenario, args.model_seed,
                device_name=args.device,
                stop_after_round=args.stop_after_round,
            )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
