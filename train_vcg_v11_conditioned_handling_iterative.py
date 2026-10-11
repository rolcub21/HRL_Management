#!/usr/bin/env python3
"""Iterative policy-consistent MC fitting for conditioned VCG handling cost."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import random
from statistics import fmean
from typing import Mapping, Optional, Sequence

import torch

import benchmark_viability_critic_priority as benchmark
import run_vcg_v11_nested_handling_pilot as pilot
import train_vcg_preference_conditioned as common
from train_viability_graph_smdp import resolve_device, seed_everything
from vcg_v11_conditioned_handling import (
    ConditionedFutureHandlingNetwork,
    ConditionedHandlingAgent,
    ConditionedHandlingConfig,
    fit_conditioned_future_handling,
)


TRAINING_PROTOCOL = "vcg_v1_1_conditioned_handling_iterative_mc_seed0_dev_v1"
TRAINER_SCHEMA_VERSION = 1
MODEL_SEED = 0
ROUNDS = 4
EPISODES_PER_ROUND = 50
TOTAL_EPISODES = ROUNDS * EPISODES_PER_ROUND
TRAIN_SEED_BASE = 58_000_000
ROUND_SEED_STRIDE = 10_000
LAMBDA_SCHEDULE_SEED = 581_000_000
SPLIT_SEED = 582_000_000
FIT_SEED = 583_000_000
PROBE_SEED_BASE = 58_900_000
PROBE_LAMBDAS = (0.025, 0.0375, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2)
FIT_EPOCHS = 8
FIT_BATCH_SIZE = 256
FIT_LEARNING_RATE = 5.0e-4
VALIDATION_EPISODES = 10
LATEST_CHECKPOINT_ROLE = "round_boundary_resumable_conditioned_handling"
TERMINAL_CHECKPOINT_ROLE = "fixed_round4_conditioned_handling_model"
COST_RELATIVE_PATH = Path(
    "results/vcg-v1-1-nested-handling-seed0-85k-development/handling-cost-head.pth"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_hash(value: Mapping) -> str:
    return hashlib.sha256(
        json.dumps(
            common._json_safe(dict(value)),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _state_digest(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


def stratified_lambda_schedule(round_index: int) -> tuple[dict, ...]:
    """Continuous stratified-uniform preferences, fixed within each episode."""

    if round_index not in range(ROUNDS):
        raise ValueError("round_index is outside the declared training rounds")
    rng_seed = LAMBDA_SCHEDULE_SEED + round_index
    rng = random.Random(rng_seed)
    values = [
        0.2 * (index + rng.random()) / EPISODES_PER_ROUND
        for index in range(EPISODES_PER_ROUND)
    ]
    rng.shuffle(values)
    return tuple(
        {
            "episode_number": round_index * EPISODES_PER_ROUND + offset + 1,
            "round_number": round_index + 1,
            "position_in_round": offset + 1,
            "behavior_lambda": float(value),
            "schedule_rng_seed": rng_seed,
            "stratified_uniform_over_[0,0.2]": True,
            "fixed_for_complete_episode": True,
        }
        for offset, value in enumerate(values)
    )


def _inputs(project_root: Path, device):
    arm, _latest, _sources, q_digest, _records = pilot._authenticate_inputs(project_root)
    base = pilot._fresh_base(arm, device)
    cost_path = project_root / COST_RELATIVE_PATH
    old_cost = pilot._load_bound_cost(
        cost_path, arm, device=device, config=base.config
    )
    config = ConditionedHandlingConfig(
        feature_dim=(
            3 * base.config.graph_embedding_dim + base.config.action_embedding_dim
        ),
        hidden_dim=base.config.head_hidden_dim,
        lambda_max=0.2,
        gamma_op=base.config.gamma,
        reward_scale=base.config.reward_scale,
    )
    return arm, base, old_cost, config, q_digest, cost_path


def _contract(args, *, arm, config, q_digest, cost_path, search, liveness):
    schedules = [
        list(stratified_lambda_schedule(round_index))
        for round_index in range(ROUNDS)
    ]
    semantic = {
        "training_protocol": TRAINING_PROTOCOL,
        "trainer_schema_version": TRAINER_SCHEMA_VERSION,
        "development_only": True,
        "model_seed": MODEL_SEED,
        "rounds": ROUNDS,
        "episodes_per_round": EPISODES_PER_ROUND,
        "total_episodes": TOTAL_EPISODES,
        "train_seed_base": TRAIN_SEED_BASE,
        "round_seed_stride": ROUND_SEED_STRIDE,
        "continuous_lambda_sampling": "stratified_uniform_[0,0.2]",
        "lambda_schedules": schedules,
        "policy_frozen_during_each_collection_round": True,
        "fit_only_after_complete_round": True,
        "current_round_only_fitting": True,
        "whole_episode_validation_split": True,
        "validation_episodes_per_round": VALIDATION_EPISODES,
        "fit_epochs": FIT_EPOCHS,
        "fit_batch_size": FIT_BATCH_SIZE,
        "fit_learning_rate": FIT_LEARNING_RATE,
        "probe_lambdas": list(PROBE_LAMBDAS),
        "probe_seed_base": PROBE_SEED_BASE,
        "environment": {
            "grid_rows": 5,
            "grid_cols": 5,
            "number_blocks": 8,
            "max_steps": int(args.max_steps),
        },
        "config": config.to_dict(),
        "search_config": asdict(search),
        "liveness_rule": asdict(liveness),
        "base_checkpoint_sha256": arm.checkpoint_sha256,
        "base_policy_digest": arm.deployment_policy_digest,
        "base_q_state_sha256": q_digest,
        "source_cost_sha256": _sha256(cost_path),
        "source_sha256": {
            "trainer": _sha256(Path(__file__).resolve()),
            "controller": _sha256(
                Path(__file__).with_name("vcg_v11_conditioned_handling.py")
            ),
        },
        "warm_start_exactly_nested_handling_head": True,
        "operational_critic_frozen": True,
        "lambda_zero_direct_vcg_v1_1_delegation": True,
        "exact_immediate_rehandle_term": True,
        "future_target_excludes_current_macro": True,
        "future_target_undiscounted": True,
        "target_network_or_bootstrap": False,
        "collection_epsilon": 0.0,
        "teacher_policy_queries": False,
        "exact_verifier_authoritative": True,
        "unsafe_unknown_fail_closed": True,
        "checkpoint_selection": False,
        "evaluation_panels_opened": False,
    }
    return {**semantic, "contract_sha256": _canonical_hash(semantic)}


def _split_episode_samples(episodes, *, round_index: int):
    if len(episodes) != EPISODES_PER_ROUND:
        raise ValueError("collection round has the wrong episode count")
    order = list(range(len(episodes)))
    random.Random(SPLIT_SEED + round_index).shuffle(order)
    validation_indices = set(order[:VALIDATION_EPISODES])
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


def _behavior_digest(run: Mapping) -> str:
    value = {
        "instance_seed": run["instance_seed"],
        "behavior_lambda": run["behavior_lambda"],
        "actions": [
            {
                "candidate_key": item["candidate_key"],
                "action_type": item["action_type"],
                "mode": item["mode"],
                "duration": item["duration"],
                "raw_rehandles": item["raw_rehandles"],
            }
            for item in run["macro_outcomes"]
        ],
        "return": run["return"],
        "steps": run["steps"],
        "delivery_deviations": run["delivery_deviations"],
    }
    return _canonical_hash(value)


def _compact_run(run: Mapping) -> dict:
    return {
        "episode_number": run.get("episode_number"),
        "instance_seed": int(run["instance_seed"]),
        "episode_instance_id": run["episode_instance_id"],
        "behavior_lambda": float(run["behavior_lambda"]),
        "return": float(run["return"]),
        "mean_absolute_error": (
            float(run["mean_absolute_error"])
            if run.get("mean_absolute_error") is not None
            and math.isfinite(float(run["mean_absolute_error"]))
            else None
        ),
        "physical_rehandles": int(run["physical_rehandles"]),
        "steps": int(run["steps"]),
        "macro_decisions": int(run["macro_decisions"]),
        "strict_method_success": bool(run["strict_method_success"]),
        "method_failure_reason": run["method_failure_reason"],
        "behavior_digest": _behavior_digest(run),
    }


def _run_episode(agent, env, *, instance_seed, value, schedule, max_steps, search, liveness):
    run = common.run_preference_episode(
        agent,
        env,
        instance_seed=instance_seed,
        preference_lambda=value,
        preference_schedule_record=schedule,
        max_steps=max_steps,
        search_config=search,
        liveness_rule=liveness,
        epsilon_start=0.0,
        epsilon_end=0.0,
        epsilon_decay_decisions=1,
        epsilon_warmup_decisions=0,
        updates_per_macro=1,
    )
    if not run["strict_method_success"]:
        raise RuntimeError(
            f"policy-frozen collection failed on {instance_seed}: "
            f"{run['method_failure_reason']}"
        )
    samples = agent.pop_completed_samples()
    if len(samples) != run["macro_decisions"]:
        raise RuntimeError("MC sample count does not match macro decisions")
    return run, samples


def _probe(agent, env, *, max_steps, search, liveness):
    rows = []
    for index, value in enumerate(PROBE_LAMBDAS):
        seed = PROBE_SEED_BASE + index
        schedule = {
            "probe": True,
            "behavior_lambda": value,
            "fixed_for_complete_episode": True,
        }
        run, _samples = _run_episode(
            agent,
            env,
            instance_seed=seed,
            value=value,
            schedule=schedule,
            max_steps=max_steps,
            search=search,
            liveness=liveness,
        )
        rows.append(_compact_run(run))
    return rows


def _round_summary(rows):
    return {
        "episodes": len(rows),
        "mean_return": float(fmean(row["return"] for row in rows)),
        "mean_absolute_error": float(
            fmean(
                row["mean_absolute_error"]
                for row in rows
                if row["mean_absolute_error"] is not None
            )
        ),
        "total_physical_rehandles": sum(row["physical_rehandles"] for row in rows),
        "physical_rehandles_per_100": float(
            100.0
            * sum(row["physical_rehandles"] for row in rows)
            / (len(rows) * 8)
        ),
        "all_strict_safe_complete": all(row["strict_method_success"] for row in rows),
    }


def _checkpoint(agent, *, arm, cost_path, contract, completed_rounds, records, role):
    return {
        "training_protocol": TRAINING_PROTOCOL,
        "trainer_schema_version": TRAINER_SCHEMA_VERSION,
        "checkpoint_role": role,
        "completed_rounds": int(completed_rounds),
        "completed_training_episodes": int(completed_rounds * EPISODES_PER_ROUND),
        "resume_contract": dict(contract),
        "resume_contract_sha256": contract["contract_sha256"],
        "agent_checkpoint": agent.checkpoint(
            base_checkpoint_sha256=arm.checkpoint_sha256,
            base_policy_digest=arm.deployment_policy_digest,
            source_cost_sha256=_sha256(cost_path),
        ),
        "round_records": tuple(records),
        "fixed_terminal_checkpoint": role == TERMINAL_CHECKPOINT_ROLE,
        "development_only": True,
        "evaluation_panels_opened": False,
    }


def _validate_checkpoint(payload: Mapping, contract: Mapping, *, terminal: bool):
    expected = {
        "training_protocol": TRAINING_PROTOCOL,
        "trainer_schema_version": TRAINER_SCHEMA_VERSION,
        "checkpoint_role": (
            TERMINAL_CHECKPOINT_ROLE if terminal else LATEST_CHECKPOINT_ROLE
        ),
        "fixed_terminal_checkpoint": terminal,
        "development_only": True,
        "evaluation_panels_opened": False,
    }
    mismatch = {
        key: (payload.get(key), value)
        for key, value in expected.items()
        if payload.get(key) != value
    }
    if mismatch:
        raise ValueError(f"incompatible conditioned handling checkpoint: {mismatch}")
    if payload.get("resume_contract") != contract:
        raise ValueError("conditioned handling resume contract changed")
    completed = int(payload.get("completed_rounds", -1))
    if len(tuple(payload.get("round_records", ()))) != completed:
        raise ValueError("conditioned handling round clock changed")
    if terminal and completed != ROUNDS:
        raise ValueError("terminal model is not the fixed fourth round")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume-existing", action="store_true")
    parser.add_argument("--stop-after-round", type=int)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--max-steps", type=int, default=2_000)
    parser.add_argument("--log-every", type=int, default=5)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> dict:
    args = build_parser().parse_args(argv)
    if args.max_steps <= 0 or args.log_every <= 0:
        raise ValueError("max steps and log interval must be positive")
    if args.stop_after_round is not None and not 1 <= args.stop_after_round <= ROUNDS:
        raise ValueError("stop-after-round must lie in [1,4]")
    args.output_dir = args.output_dir.resolve()
    project_root = Path(__file__).resolve().parent
    device = resolve_device(args.device)
    arm, base, old_cost, config, q_digest, cost_path = _inputs(project_root, device)
    search = benchmark._search_config(arm.payload)
    liveness = benchmark._liveness_rule(arm.payload)
    contract = _contract(
        args,
        arm=arm,
        config=config,
        q_digest=q_digest,
        cost_path=cost_path,
        search=search,
        liveness=liveness,
    )
    latest_path = args.output_dir / "latest.pth"
    terminal_path = args.output_dir / "terminal.pth"
    summary_path = args.output_dir / "training-summary.json"

    if args.resume_existing:
        payload = torch.load(latest_path, map_location="cpu", weights_only=False)
        _validate_checkpoint(payload, contract, terminal=False)
        fresh_base = pilot._fresh_base(arm, device)
        agent = ConditionedHandlingAgent.from_checkpoint(
            payload["agent_checkpoint"],
            base_agent=fresh_base,
            expected_base_checkpoint_sha256=arm.checkpoint_sha256,
            expected_base_policy_digest=arm.deployment_policy_digest,
            expected_source_cost_sha256=_sha256(cost_path),
            seed=MODEL_SEED,
        )
        completed_rounds = int(payload["completed_rounds"])
        round_records = list(payload["round_records"])
        print(f"Resumed conditioned handling after round {completed_rounds}", flush=True)
    else:
        if args.output_dir.exists() and any(args.output_dir.iterdir()):
            raise FileExistsError(
                "fresh conditioned handling training refuses a nonempty directory; "
                "use --resume-existing or choose another directory"
            )
        args.output_dir.mkdir(parents=True, exist_ok=True)
        seed_everything(MODEL_SEED)
        network = ConditionedFutureHandlingNetwork(config, seed=MODEL_SEED).to(device)
        network.initialize_from_nested_head(old_cost)
        initial_digest = _state_digest(network.state_dict())
        agent = ConditionedHandlingAgent(
            base, network, config=config, seed=MODEL_SEED, epsilon=0.0
        )
        completed_rounds = 0
        round_records = []
        print(f"Warm-start handling state: {initial_digest}", flush=True)

    stop_at = args.stop_after_round or ROUNDS
    if completed_rounds > stop_at:
        raise ValueError("resume checkpoint is already beyond requested round")
    print(
        "Conditioned handling VCG | seed=0 | "
        f"device={device} | rounds={ROUNDS}x{EPISODES_PER_ROUND} | "
        "continuous_lambda=true | frozen_Qop=true | bootstrap=false",
        flush=True,
    )
    env = benchmark._make_env(arm.payload)
    previous_probe = (
        round_records[-1]["probe_rows"] if round_records else None
    )
    for round_index in range(completed_rounds, stop_at):
        round_number = round_index + 1
        before_digest = _state_digest(agent.handling_network.state_dict())
        episode_samples = []
        rows = []
        schedule = stratified_lambda_schedule(round_index)
        for offset, schedule_record in enumerate(schedule):
            episode_number = round_index * EPISODES_PER_ROUND + offset + 1
            instance_seed = (
                TRAIN_SEED_BASE + round_index * ROUND_SEED_STRIDE + offset
            )
            env.current_episode = episode_number
            run, samples = _run_episode(
                agent,
                env,
                instance_seed=instance_seed,
                value=float(schedule_record["behavior_lambda"]),
                schedule=schedule_record,
                max_steps=args.max_steps,
                search=search,
                liveness=liveness,
            )
            run["episode_number"] = episode_number
            rows.append(_compact_run(run))
            episode_samples.append(samples)
            agent.base_agent.decision_log.clear()
            if offset == 0 or (offset + 1) % args.log_every == 0:
                recent = rows[-min(10, len(rows)):]
                print(
                    f"Round {round_number} Ep {offset + 1:2d}/{EPISODES_PER_ROUND} | "
                    f"Lambda {schedule_record['behavior_lambda']:.4f} | "
                    f"R {fmean(row['return'] for row in recent):7.2f} | "
                    f"Reh {run['physical_rehandles']:2d} | "
                    f"Samples {sum(len(x) for x in episode_samples):4d}",
                    flush=True,
                )
        training, validation, validation_indices = _split_episode_samples(
            episode_samples, round_index=round_index
        )
        fit = fit_conditioned_future_handling(
            agent.handling_network,
            training,
            validation,
            device=device,
            epochs=FIT_EPOCHS,
            batch_size=FIT_BATCH_SIZE,
            learning_rate=FIT_LEARNING_RATE,
            seed=FIT_SEED + round_index,
        )
        after_digest = _state_digest(agent.handling_network.state_dict())
        if after_digest == before_digest:
            raise RuntimeError("conditioned handling fit did not update the model")
        probe_rows = _probe(
            agent,
            env,
            max_steps=args.max_steps,
            search=search,
            liveness=liveness,
        )
        changed = None
        if previous_probe is not None:
            changed = sum(
                left["behavior_digest"] != right["behavior_digest"]
                for left, right in zip(previous_probe, probe_rows)
            )
        record = {
            "round_number": round_number,
            "policy_state_before_sha256": before_digest,
            "policy_state_after_sha256": after_digest,
            "collection_policy_frozen": True,
            "collection_rows": rows,
            "collection_summary": _round_summary(rows),
            "training_sample_count": len(training),
            "validation_sample_count": len(validation),
            "validation_episode_indices": validation_indices,
            "fit": fit,
            "probe_rows": probe_rows,
            "probe_behavior_changes_from_previous_round": changed,
        }
        round_records.append(record)
        previous_probe = probe_rows
        completed_rounds = round_number
        latest = _checkpoint(
            agent,
            arm=arm,
            cost_path=cost_path,
            contract=contract,
            completed_rounds=completed_rounds,
            records=round_records,
            role=LATEST_CHECKPOINT_ROLE,
        )
        common._atomic_torch_save(latest, latest_path)
        common._atomic_json_save(
            {
                "status": "complete" if completed_rounds == ROUNDS else "paused",
                "completed_rounds": completed_rounds,
                "completed_training_episodes": completed_rounds * EPISODES_PER_ROUND,
                "round_records": round_records,
                "agent_audit": agent.audit(),
                "latest_checkpoint": str(latest_path),
            },
            summary_path,
        )
        print(
            f"Round {round_number} fit | val_MAE "
            f"{fit['final_validation']['mae']:.4f} | "
            f"probe_changes {changed} | model {after_digest[:12]}",
            flush=True,
        )

    terminal_sha = None
    if completed_rounds == ROUNDS:
        if terminal_path.is_file():
            terminal = torch.load(terminal_path, map_location="cpu", weights_only=False)
            _validate_checkpoint(terminal, contract, terminal=True)
        else:
            terminal = _checkpoint(
                agent,
                arm=arm,
                cost_path=cost_path,
                contract=contract,
                completed_rounds=completed_rounds,
                records=round_records,
                role=TERMINAL_CHECKPOINT_ROLE,
            )
            common._atomic_torch_save(terminal, terminal_path)
        terminal_sha = _sha256(terminal_path)
    result = {
        "status": "complete" if completed_rounds == ROUNDS else "paused",
        "training_protocol": TRAINING_PROTOCOL,
        "completed_rounds": completed_rounds,
        "completed_training_episodes": completed_rounds * EPISODES_PER_ROUND,
        "round_records": round_records,
        "agent_audit": agent.audit(),
        "resume_contract": contract,
        "latest_checkpoint": str(latest_path),
        "terminal_checkpoint": str(terminal_path) if terminal_sha else None,
        "terminal_checkpoint_sha256": terminal_sha,
        "checkpoint_selection_used": False,
        "evaluation_panels_opened": False,
    }
    common._atomic_json_save(result, summary_path)
    print(
        json.dumps(
            {
                "status": result["status"],
                "completed_rounds": completed_rounds,
                "completed_training_episodes": result["completed_training_episodes"],
                "terminal_checkpoint": result["terminal_checkpoint"],
                "summary": str(summary_path),
            },
            indent=2,
        ),
        flush=True,
    )
    return result


if __name__ == "__main__":
    main()
