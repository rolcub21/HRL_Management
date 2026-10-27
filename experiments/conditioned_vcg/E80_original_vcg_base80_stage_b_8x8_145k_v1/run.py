#!/usr/bin/env python3
"""Freeze E78's terminal Stage A and fit a three-seed conditioned handling head."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import gc
import hashlib
import json
from pathlib import Path
import random
from statistics import mean
from unittest.mock import patch

import torch

import benchmark_viability_critic_priority as benchmark
from experiments.conditioned_vcg.E14_certification_scalability_95k import reuse
from experiments.conditioned_vcg.target_size_vcg_training import run as prior
from PSLAP.relocation_family_certification import RELOCATION_FAMILY_CERTIFICATION
import train_vcg_preference_conditioned as common
from train_viability_graph_smdp import resolve_device, seed_everything
from vcg_v11_conditioned_handling import (
    ConditionedFutureHandlingNetwork, ConditionedHandlingAgent,
    ConditionedHandlingConfig, fit_conditioned_future_handling,
    handling_calibration,
)
from viability_graph_episodic_audit import EpisodicViabilityGraphHierarchyAgent

from . import protocol as p


class E80Error(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise E80Error(message)


def read_json(path: Path) -> dict:
    require(path.is_file(), f"missing E80 file: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"invalid E80 JSON: {path}")
    return value


def authenticate(output: Path) -> dict:
    observed = read_json(output / "contract.json")
    expected = p.contract()
    require(observed == expected,
            "E80 contract, E78 terminal sources, or implementation changed")
    return expected


def prepare(output: Path) -> dict:
    expected = p.contract()
    file = output / "contract.json"
    if file.is_file():
        authenticate(output)
    else:
        require(not output.exists() or not any(output.iterdir()),
                "E80 refuses a nonempty, uncontracted output")
        common._atomic_json_save(expected, file)
    return {"status": "prepared", "contract_sha256": expected["contract_sha256"],
            "stage_a_terminal_seeds": list(p.MODEL_SEEDS),
            "stage_b_episodes_per_seed": p.ROUNDS * p.EPISODES_PER_ROUND,
            "episode_runs": 0, "training_runs": 0, "test_panel_opened": False}


def state_digest(state: dict) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


def load_base(contract: dict, seed: int, device: torch.device):
    source = contract["stage_a"][seed]
    file = p.ROOT / source["terminal_checkpoint"]
    require(p.sha256(file) == source["terminal_checkpoint_sha256"],
            "E80 terminal Stage-A checkpoint changed")
    payload = torch.load(file, map_location="cpu", weights_only=False, mmap=True)
    expected_protocol = source["source_protocol"]
    require(payload.get("training_protocol") == expected_protocol
            and payload.get("model_seed") == seed
            and payload.get("checkpoint_role") == "latest_resumable"
            and payload.get("completed_training_episodes") == 112
            and payload.get("total_training_episodes") == 112
            and payload.get("timing_objective_spec") == p.e78.SPEC.to_dict()
            and payload.get("resume_contract", {}).get("scenario") == p.SCENARIO
            and payload.get("resume_contract", {}).get("target_contract_sha256")
                == source["source_contract_sha256"],
            "E80 Stage-A checkpoint identity or objective differs")
    search = benchmark._search_config(payload)
    liveness = benchmark._liveness_rule(payload)
    base = EpisodicViabilityGraphHierarchyAgent.from_checkpoint(
        payload, device=device, resumable=False, seed=seed,
    )
    del payload
    gc.collect()
    base.set_epsilon(0.0)
    for network in (base.Q_local, base.Q_target):
        network.requires_grad_(False).eval()
    return base, search, liveness, state_digest(base.Q_local.state_dict())


def verify_base(output: Path, seed: int, device_name: str) -> dict:
    contract = authenticate(output)
    device = resolve_device(device_name)
    base, search, liveness, digest = load_base(contract, seed, device)
    require(base.config.gamma == 0.99 and base.config.episode_horizon_steps == 3600,
            "E80 Stage-A configuration differs")
    agent = new_agent(base, seed)
    with torch.no_grad():
        features = torch.zeros((2, agent.config.feature_dim), device=device)
        values = agent.handling_network(features, torch.tensor([0.05, 0.20], device=device))
    require(values.shape == (2,) and bool(torch.isfinite(values).all()),
            "E80 handling head is incompatible with the frozen Stage A")
    return {"status": "passed", "model_seed": seed, "device": str(device),
            "base_policy_digest": digest,
            "base_checkpoint_sha256": contract["stage_a"][seed]["terminal_checkpoint_sha256"],
            "search": str(search.search_order),
            "max_consecutive_defer": liveness.max_consecutive_defer_decisions,
            "episode_runs": 0, "training_runs": 0}


def lambda_schedule(seed: int, round_index: int) -> list[float]:
    rng = random.Random(p.collection_seed(seed, round_index, 0) + 500_000)
    values = [
        p.LAMBDA_MAX * (i + rng.random()) / p.EPISODES_PER_ROUND
        for i in range(p.EPISODES_PER_ROUND)
    ]
    rng.shuffle(values)
    return values


def stage_b_dir(output: Path, seed: int) -> Path:
    return output / "training/handling" / p.SCENARIO / f"seed-{seed}"


def checkpoint(agent: ConditionedHandlingAgent, contract: dict, seed: int,
               base_digest: str, rounds: list[dict], role: str) -> dict:
    base_sha = contract["stage_a"][seed]["terminal_checkpoint_sha256"]
    return {
        "training_protocol": p.PROTOCOL,
        "role": role,
        "contract_sha256": contract["contract_sha256"],
        "scenario": p.SCENARIO,
        "model_seed": seed,
        "base_checkpoint_sha256": base_sha,
        "base_policy_digest": base_digest,
        "completed_rounds": len(rounds),
        "round_records": rounds,
        "agent_checkpoint": agent.checkpoint(
            base_checkpoint_sha256=base_sha,
            base_policy_digest=base_digest,
            source_cost_sha256=prior.NO_SOURCE_COST,
        ),
    }


def new_agent(base, seed: int) -> ConditionedHandlingAgent:
    config = ConditionedHandlingConfig(
        feature_dim=(3 * base.config.graph_embedding_dim
                     + base.config.action_embedding_dim),
        hidden_dim=base.config.head_hidden_dim,
        lambda_max=p.LAMBDA_MAX,
        gamma_op=base.config.gamma,
        reward_scale=base.config.reward_scale,
    )
    seed_everything(seed)
    network = ConditionedFutureHandlingNetwork(config, seed=seed).to(base.device)
    return ConditionedHandlingAgent(base, network, config=config, seed=seed)


def episode_file(destination: Path, round_index: int, offset: int) -> Path:
    return destination / "collection" / f"round-{round_index + 1:02d}" / f"episode-{offset + 1:02d}.pth"


def collect_episode(agent, env, *, contract, destination, seed, round_index,
                    offset, preference_lambda, start_digest, search, liveness):
    file = episode_file(destination, round_index, offset)
    instance_seed = p.collection_seed(seed, round_index, offset)
    expected = {
        "contract_sha256": contract["contract_sha256"],
        "model_seed": seed, "round": round_index + 1, "offset": offset + 1,
        "instance_seed": instance_seed, "lambda": preference_lambda,
        "round_start_network_digest": start_digest,
    }
    if file.is_file():
        saved = torch.load(file, map_location="cpu", weights_only=False)
        require(all(saved.get(key) == value for key, value in expected.items()),
                f"E80 saved collection coordinate differs: {file}")
        require(saved["row"]["strict_safe_complete"]
                and len(saved["samples"]) == saved["row"]["macro_decisions"],
                f"E80 saved collection is incomplete: {file}")
        return saved["row"], tuple(saved["samples"]), False

    run = common.run_preference_episode(
        agent, env, instance_seed=instance_seed,
        preference_lambda=preference_lambda,
        preference_schedule_record={
            "behavior_lambda": preference_lambda, "round": round_index + 1,
            "episode": offset + 1, "fixed_for_complete_episode": True,
        },
        max_steps=p.e78.SCENARIOS[p.SCENARIO].max_steps,
        search_config=search, liveness_rule=liveness,
        epsilon_start=0.0, epsilon_end=0.0,
        epsilon_decay_decisions=1, epsilon_warmup_decisions=0,
        updates_per_macro=1,
    )
    row = {
        "seed": instance_seed, "lambda": preference_lambda,
        "strict_safe_complete": bool(run["strict_method_success"]),
        "failure": run["method_failure_reason"],
        "return": float(run["return"]), "steps": int(run["steps"]),
        "rehandles": int(run["physical_rehandles"]),
        "macro_decisions": int(run["macro_decisions"]),
        "instance_id": run["episode_instance_id"],
    }
    if not row["strict_safe_complete"]:
        common._atomic_json_save({**expected, "row": row},
                                 destination / "failed-collection.json")
        raise E80Error("Stage B needs a completed Monte Carlo continuation; collection failed")
    samples = tuple(item.numeric_copy() for item in agent.pop_completed_samples())
    require(len(samples) == row["macro_decisions"],
            "E80 sample count differs from executed macro decisions")
    agent.base_agent.decision_log.clear()
    common._atomic_torch_save({**expected, "row": row, "samples": samples}, file)
    return row, samples, True


def train_seed(output: Path, seed: int, *, through_round: int, device_name: str) -> dict:
    contract = authenticate(output)
    require(seed in p.MODEL_SEEDS and 1 <= through_round <= p.ROUNDS,
            "E80 requires seed 0/1/2 and through-round 1..8")
    device = resolve_device(device_name)
    base, search, liveness, base_digest = load_base(contract, seed, device)
    destination = stage_b_dir(output, seed)
    latest = destination / "latest.pth"
    terminal = destination / "terminal.pth"
    base_sha = contract["stage_a"][seed]["terminal_checkpoint_sha256"]
    if latest.is_file():
        saved = torch.load(latest, map_location="cpu", weights_only=False)
        require(saved.get("training_protocol") == p.PROTOCOL
                and saved.get("role") == "latest"
                and saved.get("contract_sha256") == contract["contract_sha256"]
                and saved.get("scenario") == p.SCENARIO
                and saved.get("model_seed") == seed
                and saved.get("base_checkpoint_sha256") == base_sha
                and saved.get("base_policy_digest") == base_digest,
                "E80 latest checkpoint identity differs")
        rounds = list(saved["round_records"])
        require(len(rounds) == saved["completed_rounds"],
                "E80 latest checkpoint round count differs")
        agent = ConditionedHandlingAgent.from_checkpoint(
            saved["agent_checkpoint"], base_agent=base,
            expected_base_checkpoint_sha256=base_sha,
            expected_base_policy_digest=base_digest,
            expected_source_cost_sha256=prior.NO_SOURCE_COST,
            seed=seed,
        )
    else:
        require(not destination.exists() or all(
            child.name == "collection" for child in destination.iterdir()
        ), "E80 nonempty handling output lacks latest checkpoint")
        rounds = []
        agent = new_agent(base, seed)
    require(len(rounds) <= through_round, "E80 cannot train backward")
    new_episode_runs = 0
    env = p.e78.TargetDenseEnv(p.SCENARIO, p.e78.SPEC)
    with ExitStack() as stack:
        stack.enter_context(patch.object(
            common, "enumerate_viability_candidates",
            prior._accelerated_enumerator(common.enumerate_viability_candidates),
        ))
        stack.enter_context(patch.object(
            common, "ViabilityCertificateCache", reuse.TimingInvariantCertificateCache,
        ))
        stack.enter_context(patch.object(common, "OBJECTIVE_SPEC", p.e78.SPEC))
        stack.enter_context(reuse.path_cleanup_active())
        for round_index in range(len(rounds), through_round):
            start_digest = state_digest(agent.handling_network.state_dict())
            episode_samples = []
            rows = []
            for offset, value in enumerate(lambda_schedule(seed, round_index)):
                row, samples, executed = collect_episode(
                    agent, env, contract=contract, destination=destination,
                    seed=seed, round_index=round_index, offset=offset,
                    preference_lambda=value, start_digest=start_digest,
                    search=search, liveness=liveness,
                )
                new_episode_runs += int(executed)
                episode_samples.append(samples)
                rows.append(row)
                print(f"E80 seed={seed} round={round_index + 1}/{p.ROUNDS} "
                      f"episode={offset + 1}/{p.EPISODES_PER_ROUND} "
                      f"{'run' if executed else 'reused'} strict=1", flush=True)
            order = list(range(len(rows)))
            random.Random(145_900_000 + seed * 1000 + round_index).shuffle(order)
            validation_indices = set(order[:p.VALIDATION_EPISODES_PER_ROUND])
            training = tuple(sample for i, episode in enumerate(episode_samples)
                             if i not in validation_indices for sample in episode)
            validation = tuple(sample for i, episode in enumerate(episode_samples)
                               if i in validation_indices for sample in episode)
            prior_state = {key: value.detach().clone()
                           for key, value in agent.handling_network.state_dict().items()}
            fit = fit_conditioned_future_handling(
                agent.handling_network, training, validation,
                device=device, epochs=8, batch_size=256,
                learning_rate=5.0e-4,
                seed=145_910_000 + seed * 1000 + round_index,
            )
            fraction = 1.0 if round_index < 4 else p.LATE_UPDATE_FRACTION
            if fraction != 1.0:
                fitted = agent.handling_network.state_dict()
                agent.handling_network.load_state_dict({
                    key: (1.0 - fraction) * prior_state[key] + fraction * fitted[key]
                    for key in fitted
                })
            fit["applied_update_fraction"] = fraction
            fit["applied_validation"] = handling_calibration(
                agent.handling_network, validation, device=device,
            )
            require(state_digest(base.Q_local.state_dict()) == base_digest,
                    "E80 Stage B mutated the frozen Stage-A critic")
            end_digest = state_digest(agent.handling_network.state_dict())
            require(end_digest != start_digest, "E80 handling fit did not update weights")
            rounds.append({
                "round": round_index + 1, "episodes": rows,
                "training_samples": len(training),
                "validation_samples": len(validation),
                "validation_episode_indices": sorted(validation_indices),
                "network_before_sha256": start_digest,
                "network_after_sha256": end_digest,
                "fit": fit,
            })
            common._atomic_torch_save(
                checkpoint(agent, contract, seed, base_digest, rounds, "latest"), latest,
            )
            common._atomic_json_save({
                "status": "complete" if len(rounds) == p.ROUNDS else "paused",
                "training_protocol": p.PROTOCOL,
                "contract_sha256": contract["contract_sha256"],
                "scenario": p.SCENARIO, "model_seed": seed,
                "completed_rounds": len(rounds),
                "completed_collection_episodes": len(rounds) * p.EPISODES_PER_ROUND,
                "base_checkpoint_sha256": base_sha,
                "base_policy_digest": base_digest,
                "round_records": rounds,
                "test_panel_opened": False,
            }, destination / "training-summary.json")
    if len(rounds) == p.ROUNDS:
        if not terminal.is_file():
            common._atomic_torch_save(
                checkpoint(agent, contract, seed, base_digest, rounds, "terminal"), terminal,
            )
    return {
        "status": "complete" if len(rounds) == p.ROUNDS else "paused",
        "model_seed": seed, "completed_rounds": len(rounds),
        "completed_collection_episodes": len(rounds) * p.EPISODES_PER_ROUND,
        "episode_runs_this_invocation": new_episode_runs,
        "terminal_checkpoint": str(terminal) if terminal.is_file() else None,
        "test_panel_opened": False,
    }


def pilot_collect(output: Path, seed: int, device_name: str) -> dict:
    """Collect exactly one first-round episode; the full run will reuse it."""
    contract = authenticate(output)
    device = resolve_device(device_name)
    base, search, liveness, base_digest = load_base(contract, seed, device)
    destination = stage_b_dir(output, seed)
    require(not (destination / "latest.pth").exists(),
            "E80 pilot collection is only for a seed before round 1 is fitted")
    agent = new_agent(base, seed)
    value = lambda_schedule(seed, 0)[0]
    env = p.e78.TargetDenseEnv(p.SCENARIO, p.e78.SPEC)
    with ExitStack() as stack:
        stack.enter_context(patch.object(
            common, "enumerate_viability_candidates",
            prior._accelerated_enumerator(common.enumerate_viability_candidates),
        ))
        stack.enter_context(patch.object(
            common, "ViabilityCertificateCache", reuse.TimingInvariantCertificateCache,
        ))
        stack.enter_context(patch.object(common, "OBJECTIVE_SPEC", p.e78.SPEC))
        stack.enter_context(reuse.path_cleanup_active())
        row, samples, executed = collect_episode(
            agent, env, contract=contract, destination=destination,
            seed=seed, round_index=0, offset=0, preference_lambda=value,
            start_digest=state_digest(agent.handling_network.state_dict()),
            search=search, liveness=liveness,
        )
    require(state_digest(base.Q_local.state_dict()) == base_digest,
            "E80 pilot mutated the frozen Stage-A critic")
    return {"status": "passed", "model_seed": seed, "episode_runs": int(executed),
            "cached_for_full_run": True, "row": row, "samples": len(samples),
            "training_runs": 0}


def analyze(output: Path) -> dict:
    contract = authenticate(output)
    rows = []
    for seed in p.MODEL_SEEDS:
        directory = stage_b_dir(output, seed)
        summary_file = directory / "training-summary.json"
        if not summary_file.is_file():
            rows.append({"model_seed": seed, "status": "not_started"})
            continue
        summary = read_json(summary_file)
        rounds = summary["round_records"]
        require(summary.get("training_protocol") == p.PROTOCOL
                and summary.get("contract_sha256") == contract["contract_sha256"]
                and summary.get("model_seed") == seed
                and summary.get("base_checkpoint_sha256")
                    == contract["stage_a"][seed]["terminal_checkpoint_sha256"]
                and summary.get("completed_rounds") == len(rounds),
                f"E80 seed {seed} summary identity differs")
        require([r["round"] for r in rounds] == list(range(1, len(rounds) + 1)),
                f"E80 seed {seed} round coverage differs")
        episode_rows = [episode for round_record in rounds
                        for episode in round_record["episodes"]]
        require(len(episode_rows) == len(rounds) * p.EPISODES_PER_ROUND
                and all(episode["strict_safe_complete"] for episode in episode_rows),
                f"E80 seed {seed} collection coverage or completion differs")
        terminal = directory / "terminal.pth"
        if len(rounds) == p.ROUNDS:
            require(terminal.is_file() and summary["status"] == "complete",
                    f"E80 seed {seed} terminal artifact is missing")
        rows.append({
            "model_seed": seed, "status": summary["status"],
            "completed_rounds": len(rounds),
            "collection_episodes": len(episode_rows),
            "strict_complete_episodes": len(episode_rows),
            "mean_collection_return": mean(float(row["return"]) for row in episode_rows),
            "mean_collection_rehandles": mean(float(row["rehandles"]) for row in episode_rows),
            "terminal_checkpoint": str(terminal) if terminal.is_file() else None,
            "terminal_checkpoint_sha256": p.sha256(terminal) if terminal.is_file() else None,
            "training_summary_sha256": p.sha256(summary_file),
        })
    result = {
        "protocol": p.PROTOCOL, "contract_sha256": contract["contract_sha256"],
        "status": "complete" if all(row["status"] == "complete" for row in rows)
                  else "partial",
        "scope": "development Stage-B collection and fit; no fresh quality panel",
        "checkpoint_selection": "terminal_round_8_only",
        "model_seeds": rows,
        "episode_runs": 0, "training_runs": 0,
        "test_panel_opened": False, "paper_evidence": False,
    }
    common._atomic_json_save(result, output / "e80-report.json")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=(
        "prepare", "verify-base", "pilot-collect", "train-seed", "analyze",
    ))
    parser.add_argument("--output", type=Path, default=p.DEFAULT_OUTPUT)
    parser.add_argument("--model-seed", type=int, choices=p.MODEL_SEEDS)
    parser.add_argument("--through-round", type=int, default=p.ROUNDS)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cuda")
    args = parser.parse_args()
    output = args.output.resolve()
    if args.command == "prepare":
        result = prepare(output)
    elif args.command == "analyze":
        result = analyze(output)
    else:
        require(args.model_seed is not None, "E80 command requires --model-seed")
        if args.command == "verify-base":
            result = verify_base(output, args.model_seed, args.device)
        elif args.command == "pilot-collect":
            result = pilot_collect(output, args.model_seed, args.device)
        else:
            result = train_seed(output, args.model_seed,
                                through_round=args.through_round,
                                device_name=args.device)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
