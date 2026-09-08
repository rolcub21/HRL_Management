#!/usr/bin/env python3
"""Matched policy-consistent handling training for one E12 representation."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from statistics import fmean
import sys
from typing import Mapping, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

import benchmark_viability_critic_priority as benchmark
from experiments.conditioned_vcg.E12_representation_ablation_94k import program
from experiments.conditioned_vcg.E12_representation_ablation_94k import train_operational
from methods.conditioned_vcg.controller import (
    ConditionedFutureHandlingNetwork,
    ConditionedHandlingConfig,
    fit_conditioned_future_handling,
)
from methods.conditioned_vcg.representation_ablation import (
    REPRESENTATION_VARIANTS,
    RepresentationConditionedHandlingAgent,
    agent_class_for_variant,
)
from train_viability_graph_smdp import resolve_device, seed_everything
import train_vcg_preference_conditioned as common
import train_vcg_v11_conditioned_handling_iterative as iterative


TRAINING_PROTOCOL = "vcg_e12_matched_conditioned_handling_training_v1"
LATEST_ROLE = "e12_round_boundary_resumable_handling"
TERMINAL_ROLE = "e12_fixed_round8_handling"


def output_path(output_dir: Path, variant: str, model_seed: int) -> Path:
    return output_dir / "training" / "handling" / variant / f"seed-{model_seed}"


def _state_digest(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _load_operational(output_dir: Path, variant: str, model_seed: int, device):
    directory = train_operational.output_path(output_dir, variant, model_seed)
    summary = program.load_json(
        directory / "training-summary.json", label="E12 operational summary"
    )
    expected = {
        "status": "complete",
        "completed_training_episodes": program.OPERATIONAL_EPISODES,
        "model_seed": model_seed,
        "representation_variant": variant,
        "e12_contract_sha256": program.authenticate(output_dir)[0][
            "contract_sha256"
        ],
    }
    mismatch = {
        key: (summary.get(key), value)
        for key, value in expected.items()
        if summary.get(key) != value
    }
    if mismatch:
        raise program.E12Error(f"operational training is not complete: {mismatch}")
    checkpoint_path = directory / "best.pth"
    checkpoint_sha = program.sha256(checkpoint_path)
    if summary.get("best_checkpoint_sha256") != checkpoint_sha:
        raise program.E12Error("operational best checkpoint hash changed")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    agent_class = agent_class_for_variant(variant)
    agent = agent_class.from_checkpoint(
        payload, device=device, resumable=False, seed=model_seed
    )
    for network in (agent.Q_local, agent.Q_target):
        network.requires_grad_(False).eval()
    agent.set_epsilon(0.0)
    return payload, agent, checkpoint_sha, _state_digest(agent.Q_local.state_dict())


def lambda_schedule(round_index: int, model_seed: int) -> tuple[dict, ...]:
    rng_seed = 105_000_000 + model_seed * 1_000_000 + round_index
    rng = random.Random(rng_seed)
    count = program.HANDLING_EPISODES_PER_ROUND
    values = [0.2 * (index + rng.random()) / count for index in range(count)]
    rng.shuffle(values)
    return tuple(
        {
            "episode_number": round_index * count + offset + 1,
            "round_number": round_index + 1,
            "position_in_round": offset + 1,
            "behavior_lambda": float(value),
            "schedule_rng_seed": rng_seed,
            "stratified_uniform_over_[0,0.2]": True,
            "fixed_for_complete_episode": True,
        }
        for offset, value in enumerate(values)
    )


def _split(episodes, *, round_index: int, model_seed: int):
    if len(episodes) != program.HANDLING_EPISODES_PER_ROUND:
        raise ValueError("handling round has the wrong episode count")
    order = list(range(len(episodes)))
    random.Random(106_000_000 + model_seed * 1_000 + round_index).shuffle(order)
    validation_indices = set(order[: program.HANDLING_VALIDATION_EPISODES])
    training = tuple(
        sample
        for index, episode in enumerate(episodes)
        if index not in validation_indices
        for sample in episode
    )
    validation = tuple(
        sample
        for index, episode in enumerate(episodes)
        if index in validation_indices
        for sample in episode
    )
    return training, validation, sorted(validation_indices)


def _checkpoint(
    agent,
    *,
    contract: Mapping,
    variant: str,
    model_seed: int,
    completed_rounds: int,
    round_records: Sequence[Mapping],
    base_checkpoint_sha256: str,
    base_policy_digest: str,
    role: str,
) -> dict:
    return {
        "training_protocol": TRAINING_PROTOCOL,
        "checkpoint_role": role,
        "completed_rounds": int(completed_rounds),
        "completed_training_episodes": int(
            completed_rounds * program.HANDLING_EPISODES_PER_ROUND
        ),
        "e12_contract_sha256": contract["contract_sha256"],
        "representation_variant": variant,
        "model_seed": int(model_seed),
        "base_checkpoint_sha256": base_checkpoint_sha256,
        "base_policy_digest": base_policy_digest,
        "agent_checkpoint": agent.checkpoint(
            base_checkpoint_sha256=base_checkpoint_sha256,
            base_policy_digest=base_policy_digest,
        ),
        "round_records": tuple(round_records),
        "fixed_terminal_checkpoint": role == TERMINAL_ROLE,
        "checkpoint_selection_used": False,
        "development_pilot_opened": False,
        "e11_confirmation_opened": False,
    }


def _validate_resume(
    payload: Mapping,
    *,
    contract: Mapping,
    variant: str,
    model_seed: int,
    base_checkpoint_sha256: str,
    base_policy_digest: str,
) -> None:
    expected = {
        "training_protocol": TRAINING_PROTOCOL,
        "checkpoint_role": LATEST_ROLE,
        "e12_contract_sha256": contract["contract_sha256"],
        "representation_variant": variant,
        "model_seed": model_seed,
        "base_checkpoint_sha256": base_checkpoint_sha256,
        "base_policy_digest": base_policy_digest,
        "fixed_terminal_checkpoint": False,
        "checkpoint_selection_used": False,
        "development_pilot_opened": False,
        "e11_confirmation_opened": False,
    }
    mismatch = {
        key: (payload.get(key), value)
        for key, value in expected.items()
        if payload.get(key) != value
    }
    if mismatch:
        raise program.E12Error(f"handling resume checkpoint changed: {mismatch}")
    completed = int(payload.get("completed_rounds", -1))
    if completed < 0 or len(tuple(payload.get("round_records", ()))) != completed:
        raise program.E12Error("handling resume clock is inconsistent")


def run(
    output_dir: Path,
    *,
    variant: str,
    model_seed: int,
    device_name: str,
    stop_after_round: Optional[int],
    log_every: int,
) -> dict:
    contract, _manifest = program.authenticate(output_dir)
    if variant not in REPRESENTATION_VARIANTS:
        raise program.E12Error(f"unknown representation: {variant}")
    if model_seed not in program.MODEL_SEEDS:
        raise program.E12Error("model seed must be 0, 1, or 2")
    if stop_after_round is not None and not 1 <= stop_after_round <= program.HANDLING_ROUNDS:
        raise ValueError("stop-after-round is outside the declared eight rounds")
    if log_every <= 0:
        raise ValueError("log-every must be positive")
    device = resolve_device(device_name)
    operational_payload, base, base_sha, policy_digest = _load_operational(
        output_dir, variant, model_seed, device
    )
    search = benchmark._search_config(operational_payload)
    liveness = benchmark._liveness_rule(operational_payload)
    config = ConditionedHandlingConfig(
        feature_dim=(
            3 * base.config.graph_embedding_dim
            + base.config.action_embedding_dim
        ),
        hidden_dim=base.config.head_hidden_dim,
        lambda_max=0.2,
        gamma_op=base.config.gamma,
        reward_scale=base.config.reward_scale,
    )
    destination = output_path(output_dir, variant, model_seed)
    latest_path = destination / "latest.pth"
    terminal_path = destination / "terminal.pth"
    summary_path = destination / "training-summary.json"

    if latest_path.is_file():
        payload = torch.load(latest_path, map_location="cpu", weights_only=False)
        _validate_resume(
            payload,
            contract=contract,
            variant=variant,
            model_seed=model_seed,
            base_checkpoint_sha256=base_sha,
            base_policy_digest=policy_digest,
        )
        agent = RepresentationConditionedHandlingAgent.from_checkpoint(
            payload["agent_checkpoint"],
            base_agent=base,
            expected_base_checkpoint_sha256=base_sha,
            expected_base_policy_digest=policy_digest,
            seed=model_seed,
        )
        completed_rounds = int(payload["completed_rounds"])
        round_records = list(payload["round_records"])
        print(f"Resumed {variant}/seed-{model_seed} after round {completed_rounds}", flush=True)
    else:
        if destination.exists() and any(destination.iterdir()):
            raise FileExistsError("nonempty handling directory has no latest checkpoint")
        destination.mkdir(parents=True, exist_ok=True)
        seed_everything(model_seed)
        network = ConditionedFutureHandlingNetwork(config, seed=model_seed).to(device)
        network.requires_grad_(False).eval()
        agent = RepresentationConditionedHandlingAgent(
            base, network, config=config, seed=model_seed, epsilon=0.0
        )
        completed_rounds = 0
        round_records = []

    stop_at = stop_after_round or program.HANDLING_ROUNDS
    if completed_rounds > stop_at:
        raise ValueError("resume checkpoint is beyond the requested round")
    env = benchmark._make_env(operational_payload)
    print(
        f"E12 handling | {variant} | seed={model_seed} | device={device} | "
        f"rounds={program.HANDLING_ROUNDS}x{program.HANDLING_EPISODES_PER_ROUND} | "
        "random_init=true | frozen_Qop=true",
        flush=True,
    )
    for round_index in range(completed_rounds, stop_at):
        episode_samples = []
        rows = []
        before_digest = _state_digest(agent.handling_network.state_dict())
        schedule = lambda_schedule(round_index, model_seed)
        for offset, schedule_record in enumerate(schedule):
            episode_number = (
                round_index * program.HANDLING_EPISODES_PER_ROUND + offset + 1
            )
            instance_seed = (
                program.HANDLING_TRAIN_SEED_BASE
                + model_seed * program.HANDLING_MODEL_SEED_STRIDE
                + round_index * program.HANDLING_ROUND_SEED_STRIDE
                + offset
            )
            env.current_episode = episode_number
            episode, samples = iterative._run_episode(
                agent,
                env,
                instance_seed=instance_seed,
                value=float(schedule_record["behavior_lambda"]),
                schedule=schedule_record,
                max_steps=program.MAX_STEPS,
                search=search,
                liveness=liveness,
            )
            episode["episode_number"] = episode_number
            rows.append(iterative._compact_run(episode))
            episode_samples.append(samples)
            agent.base_agent.decision_log.clear()
            if offset == 0 or (offset + 1) % log_every == 0:
                recent = rows[-min(10, len(rows)):]
                print(
                    f"Round {round_index + 1} Ep {offset + 1:2d}/"
                    f"{program.HANDLING_EPISODES_PER_ROUND} | "
                    f"lambda {schedule_record['behavior_lambda']:.4f} | "
                    f"R {fmean(row['return'] for row in recent):7.2f} | "
                    f"samples {sum(len(items) for items in episode_samples):4d}",
                    flush=True,
                )
        training, validation, validation_indices = _split(
            episode_samples, round_index=round_index, model_seed=model_seed
        )
        fit = fit_conditioned_future_handling(
            agent.handling_network,
            training,
            validation,
            device=device,
            epochs=program.HANDLING_FIT_EPOCHS,
            batch_size=program.HANDLING_FIT_BATCH_SIZE,
            learning_rate=program.HANDLING_FIT_LEARNING_RATE,
            seed=107_000_000 + model_seed * 1_000 + round_index,
        )
        after_digest = _state_digest(agent.handling_network.state_dict())
        if after_digest == before_digest:
            raise RuntimeError("handling fit did not update the network")
        round_records.append(
            {
                "round_number": round_index + 1,
                "policy_state_before_sha256": before_digest,
                "policy_state_after_sha256": after_digest,
                "collection_rows": rows,
                "collection_summary": iterative._round_summary(rows),
                "training_sample_count": len(training),
                "validation_sample_count": len(validation),
                "validation_episode_indices": validation_indices,
                "fit": fit,
            }
        )
        completed_rounds = round_index + 1
        latest = _checkpoint(
            agent,
            contract=contract,
            variant=variant,
            model_seed=model_seed,
            completed_rounds=completed_rounds,
            round_records=round_records,
            base_checkpoint_sha256=base_sha,
            base_policy_digest=policy_digest,
            role=LATEST_ROLE,
        )
        common._atomic_torch_save(latest, latest_path)
        program.atomic_json(
            summary_path,
            {
                "status": (
                    "complete"
                    if completed_rounds == program.HANDLING_ROUNDS
                    else "paused"
                ),
                "training_protocol": TRAINING_PROTOCOL,
                "e12_contract_sha256": contract["contract_sha256"],
                "representation_variant": variant,
                "model_seed": model_seed,
                "completed_rounds": completed_rounds,
                "completed_training_episodes": (
                    completed_rounds * program.HANDLING_EPISODES_PER_ROUND
                ),
                "base_checkpoint_sha256": base_sha,
                "base_policy_digest": policy_digest,
                "round_records": round_records,
                "latest_checkpoint": str(latest_path),
            },
        )
        print(
            f"Round {completed_rounds} fit | val_MAE "
            f"{fit['final_validation']['mae']:.4f} | model {after_digest[:12]}",
            flush=True,
        )

    terminal_sha = None
    if completed_rounds == program.HANDLING_ROUNDS:
        terminal = _checkpoint(
            agent,
            contract=contract,
            variant=variant,
            model_seed=model_seed,
            completed_rounds=completed_rounds,
            round_records=round_records,
            base_checkpoint_sha256=base_sha,
            base_policy_digest=policy_digest,
            role=TERMINAL_ROLE,
        )
        if terminal_path.is_file():
            observed = torch.load(terminal_path, map_location="cpu", weights_only=False)
            # Tensor-valued payloads do not support direct dictionary
            # equality. Authenticate the immutable binding and load the
            # enclosed network through its strict schema instead.
            expected_binding = (
                observed.get("training_protocol"),
                observed.get("checkpoint_role"),
                observed.get("e12_contract_sha256"),
                observed.get("representation_variant"),
                observed.get("model_seed"),
                observed.get("completed_rounds"),
                observed.get("base_checkpoint_sha256"),
                observed.get("base_policy_digest"),
            )
            binding = (
                TRAINING_PROTOCOL,
                TERMINAL_ROLE,
                contract["contract_sha256"],
                variant,
                model_seed,
                program.HANDLING_ROUNDS,
                base_sha,
                policy_digest,
            )
            if expected_binding != binding:
                raise program.E12Error("existing terminal checkpoint changed")
            RepresentationConditionedHandlingAgent.from_checkpoint(
                observed["agent_checkpoint"],
                base_agent=base,
                expected_base_checkpoint_sha256=base_sha,
                expected_base_policy_digest=policy_digest,
                seed=model_seed,
            )
        else:
            common._atomic_torch_save(terminal, terminal_path)
        terminal_sha = program.sha256(terminal_path)

    result = {
        "status": (
            "complete"
            if completed_rounds == program.HANDLING_ROUNDS
            else "paused"
        ),
        "training_protocol": TRAINING_PROTOCOL,
        "e12_contract_sha256": contract["contract_sha256"],
        "representation_variant": variant,
        "model_seed": model_seed,
        "completed_rounds": completed_rounds,
        "completed_training_episodes": (
            completed_rounds * program.HANDLING_EPISODES_PER_ROUND
        ),
        "base_checkpoint_sha256": base_sha,
        "base_policy_digest": policy_digest,
        "round_records": round_records,
        "latest_checkpoint": str(latest_path),
        "terminal_checkpoint": str(terminal_path) if terminal_sha else None,
        "terminal_checkpoint_sha256": terminal_sha,
        "checkpoint_selection_used": False,
        "development_pilot_opened": False,
        "e11_confirmation_opened": False,
    }
    program.atomic_json(summary_path, result)
    return result


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=program.DEFAULT_OUTPUT)
    parser.add_argument("--variant", choices=REPRESENTATION_VARIANTS, required=True)
    parser.add_argument("--model-seed", type=int, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--stop-after-round", type=int)
    parser.add_argument("--log-every", type=int, default=5)
    args = parser.parse_args(argv)
    result = run(
        args.output_dir.resolve(),
        variant=args.variant,
        model_seed=args.model_seed,
        device_name=args.device,
        stop_after_round=args.stop_after_round,
        log_every=args.log_every,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "representation_variant": result["representation_variant"],
                "model_seed": result["model_seed"],
                "completed_rounds": result["completed_rounds"],
                "terminal_checkpoint": result["terminal_checkpoint"],
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
