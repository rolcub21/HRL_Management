#!/usr/bin/env python3
"""Conservative fitted-policy evaluation for conditioned-handling VCG seed 0."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
from typing import Mapping, Optional, Sequence

import torch

import benchmark_viability_critic_priority as benchmark
import run_vcg_v11_nested_handling_pilot as pilot
import train_vcg_preference_conditioned as common
import train_vcg_v11_conditioned_handling_convergence_extension as extension
import train_vcg_v11_conditioned_handling_iterative as original
from vcg_v11_conditioned_handling import (
    ConditionedHandlingAgent,
    fit_conditioned_future_handling,
    handling_calibration,
)


PROTOCOL = "vcg_v1_1_conditioned_handling_seed0_damped_convergence_v1"
SCHEMA_VERSION = 1
POLYAK_RHO = 0.25
DAMPED_ROUNDS = 4
EPISODES_PER_ROUND = original.EPISODES_PER_ROUND
PARENT_GLOBAL_ROUNDS = original.ROUNDS
TOTAL_COLLECTION_EPISODES = DAMPED_ROUNDS * EPISODES_PER_ROUND
CONTRACT_NAME = "damped-convergence-contract.json"
LATEST_NAME = "latest.pth"
TERMINAL_NAME = "terminal.pth"
SUMMARY_NAME = "damped-convergence-summary.json"
LATEST_ROLE = "resumable_damped_convergence_round_boundary"
TERMINAL_ROLE = "fixed_four_round_damped_convergence_terminal"


class DampedConvergenceError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    if not path.is_file():
        raise DampedConvergenceError(f"missing required artifact: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_hash(value: Mapping) -> str:
    return hashlib.sha256(
        json.dumps(
            common._json_safe(dict(value)),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def damped_lambda_schedule(round_index: int) -> tuple[dict, ...]:
    if round_index not in range(DAMPED_ROUNDS):
        raise ValueError("damped round index is out of range")
    global_round_index = PARENT_GLOBAL_ROUNDS + round_index
    rng_seed = original.LAMBDA_SCHEDULE_SEED + global_round_index
    rng = random.Random(rng_seed)
    values = [
        0.2 * (index + rng.random()) / EPISODES_PER_ROUND
        for index in range(EPISODES_PER_ROUND)
    ]
    rng.shuffle(values)
    return tuple(
        {
            "damped_episode_number": round_index * EPISODES_PER_ROUND + offset + 1,
            "global_episode_number": (
                original.TOTAL_EPISODES
                + round_index * EPISODES_PER_ROUND
                + offset
                + 1
            ),
            "damped_round_number": round_index + 1,
            "global_round_number": global_round_index + 1,
            "position_in_round": offset + 1,
            "behavior_lambda": float(value),
            "schedule_rng_seed": rng_seed,
            "stratified_uniform_over_[0,0.2]": True,
            "fixed_for_complete_episode": True,
        }
        for offset, value in enumerate(values)
    )


def _clone_state(network) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().clone()
        for name, tensor in network.state_dict().items()
    }


def polyak_blend_network(network, previous_state: Mapping[str, torch.Tensor]) -> None:
    """Blend a provisionally fitted model toward its pre-fit parameters."""

    current = network.state_dict()
    if tuple(current) != tuple(previous_state):
        raise ValueError("Polyak state dictionaries do not align")
    blended = {}
    for name, provisional in current.items():
        previous = previous_state[name].to(
            device=provisional.device, dtype=provisional.dtype
        )
        if not provisional.is_floating_point():
            if not torch.equal(previous, provisional):
                raise ValueError(f"nonfloating Polyak state changed: {name}")
            blended[name] = provisional
        else:
            blended[name] = previous.mul(1.0 - POLYAK_RHO).add(
                provisional, alpha=POLYAK_RHO
            )
    network.load_state_dict(blended)
    network.requires_grad_(False).eval()


def _parent_artifacts(project_root: Path) -> dict:
    return extension._parent_artifacts(project_root)


def _contract(project_root: Path, output_dir: Path, *, max_steps: int) -> dict:
    artifacts = _parent_artifacts(project_root)
    semantic = {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "scope": "post_evaluation_seed0_training_only_conservative_fit_diagnostic",
        "parent_contract_sha256": artifacts["contract"]["contract_sha256"],
        "parent_terminal_sha256": artifacts["terminal_sha256"],
        "parent_training_summary_sha256": artifacts["summary_sha256"],
        "parent_completed_rounds": PARENT_GLOBAL_ROUNDS,
        "fixed_damped_rounds": DAMPED_ROUNDS,
        "episodes_per_round": EPISODES_PER_ROUND,
        "total_collection_episodes": TOTAL_COLLECTION_EPISODES,
        "probe_episodes_per_round": len(original.PROBE_LAMBDAS),
        "total_probe_episodes": DAMPED_ROUNDS * len(original.PROBE_LAMBDAS),
        "total_additional_simulator_episodes": (
            TOTAL_COLLECTION_EPISODES
            + DAMPED_ROUNDS * len(original.PROBE_LAMBDAS)
        ),
        "global_rounds_after_experiment": PARENT_GLOBAL_ROUNDS + DAMPED_ROUNDS,
        "polyak_parameter_blend_rho": POLYAK_RHO,
        "polyak_rule": "theta_next=(1-rho)*theta_previous+rho*theta_provisional",
        "lambda_schedules": [
            damped_lambda_schedule(index) for index in range(DAMPED_ROUNDS)
        ],
        "paired_first_two_streams_with_full_update_extension": True,
        "max_steps": int(max_steps),
        "fit_epochs": original.FIT_EPOCHS,
        "fit_batch_size": original.FIT_BATCH_SIZE,
        "fit_learning_rate": original.FIT_LEARNING_RATE,
        "whole_episode_validation_split": True,
        "validation_episodes_per_round": original.VALIDATION_EPISODES,
        "policy_frozen_within_collection_round": True,
        "current_round_only_mc_fitting": True,
        "operational_critic_frozen": True,
        "provisional_fit_then_fixed_parameter_damping": True,
        "evaluation_panel_reopened": False,
        "checkpoint_selection": False,
        "fixed_four_round_terminal": True,
        "convergence_rule": {
            "assessed_on_final_two_round_transitions_only": True,
            "maximum_changed_probe_behaviors_per_transition": (
                extension.MAX_CHANGED_PROBES_PER_TRANSITION
            ),
            "aggregate_probe_rehandles_must_be_unchanged": True,
            "all_probe_episodes_strict_safe_complete": True,
            "maximum_deployed_validation_mae": extension.MAX_VALIDATION_MAE,
            "maximum_absolute_deployed_validation_bias": (
                extension.MAX_ABSOLUTE_VALIDATION_BIAS
            ),
            "evaluation_return_and_mae_not_used": True,
        },
        "source_sha256": {
            "damped_trainer": _sha256(Path(__file__).resolve()),
            "conditioned_controller": _sha256(
                project_root / "vcg_v11_conditioned_handling.py"
            ),
            "parent_trainer": _sha256(
                project_root / "train_vcg_v11_conditioned_handling_iterative.py"
            ),
            "parent_runner": _sha256(
                project_root / "run_vcg_v11_conditioned_handling_seed0_85k.py"
            ),
            "stability_definition": _sha256(
                project_root
                / "train_vcg_v11_conditioned_handling_convergence_extension.py"
            ),
        },
        "output_dir": str(output_dir.resolve()),
    }
    return {**semantic, "contract_sha256": _canonical_hash(semantic)}


def _ensure_contract(project_root: Path, output_dir: Path, *, max_steps: int) -> dict:
    expected = _contract(project_root, output_dir, max_steps=max_steps)
    path = output_dir / CONTRACT_NAME
    if path.is_file():
        if json.loads(path.read_text(encoding="utf-8")) != expected:
            raise DampedConvergenceError(
                "damped contract, parent artifacts, or sources changed"
            )
    else:
        if output_dir.exists() and any(output_dir.iterdir()):
            raise DampedConvergenceError(
                "nonempty damped output has no authenticated contract"
            )
        pilot._atomic_json(path, expected)
    return expected


def _checkpoint(
    agent,
    *,
    arm,
    cost_path: Path,
    contract: Mapping,
    artifacts: Mapping,
    completed_rounds: int,
    records,
    role: str,
) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "checkpoint_role": role,
        "completed_damped_rounds": int(completed_rounds),
        "completed_collection_episodes": int(
            completed_rounds * EPISODES_PER_ROUND
        ),
        "global_completed_rounds": int(PARENT_GLOBAL_ROUNDS + completed_rounds),
        "polyak_rho": POLYAK_RHO,
        "parent_terminal_sha256": artifacts["terminal_sha256"],
        "contract": dict(contract),
        "contract_sha256": contract["contract_sha256"],
        "agent_checkpoint": agent.checkpoint(
            base_checkpoint_sha256=arm.checkpoint_sha256,
            base_policy_digest=arm.deployment_policy_digest,
            source_cost_sha256=_sha256(cost_path),
        ),
        "round_records": tuple(records),
        "fixed_terminal_checkpoint": role == TERMINAL_ROLE,
        "checkpoint_selection_used": False,
        "evaluation_panel_reopened": False,
    }


def _validate_checkpoint(payload: Mapping, contract: Mapping, *, terminal: bool):
    expected = {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "checkpoint_role": TERMINAL_ROLE if terminal else LATEST_ROLE,
        "polyak_rho": POLYAK_RHO,
        "contract_sha256": contract["contract_sha256"],
        "fixed_terminal_checkpoint": terminal,
        "checkpoint_selection_used": False,
        "evaluation_panel_reopened": False,
    }
    mismatch = {
        key: (payload.get(key), value)
        for key, value in expected.items()
        if payload.get(key) != value
    }
    if mismatch:
        raise DampedConvergenceError(f"damped checkpoint mismatch: {mismatch}")
    if payload.get("contract") != contract:
        raise DampedConvergenceError("damped checkpoint contract changed")
    completed = int(payload.get("completed_damped_rounds", -1))
    if len(tuple(payload.get("round_records", ()))) != completed:
        raise DampedConvergenceError("damped checkpoint round clock changed")
    if terminal and completed != DAMPED_ROUNDS:
        raise DampedConvergenceError("damped terminal is incomplete")


def _load_agent_from_checkpoint(payload, *, arm, cost_path, device):
    return ConditionedHandlingAgent.from_checkpoint(
        payload["agent_checkpoint"],
        base_agent=pilot._fresh_base(arm, device),
        expected_base_checkpoint_sha256=arm.checkpoint_sha256,
        expected_base_policy_digest=arm.deployment_policy_digest,
        expected_source_cost_sha256=_sha256(cost_path),
        seed=original.MODEL_SEED,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume-existing", action="store_true")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--max-steps", type=int, default=2_000)
    parser.add_argument("--log-every", type=int, default=5)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> dict:
    args = build_parser().parse_args(argv)
    if args.max_steps <= 0 or args.log_every <= 0:
        raise ValueError("max steps and log interval must be positive")
    args.output_dir = args.output_dir.resolve()
    project_root = Path(__file__).resolve().parent
    torch.set_num_threads(1)
    device = original.resolve_device(args.device)
    artifacts = _parent_artifacts(project_root)
    contract = _ensure_contract(
        project_root, args.output_dir, max_steps=args.max_steps
    )
    arm, _base, _old_cost, _config, _q_digest, cost_path = original._inputs(
        project_root, device
    )
    search = benchmark._search_config(arm.payload)
    liveness = benchmark._liveness_rule(arm.payload)
    latest_path = args.output_dir / LATEST_NAME
    terminal_path = args.output_dir / TERMINAL_NAME
    summary_path = args.output_dir / SUMMARY_NAME

    if args.resume_existing:
        payload = torch.load(latest_path, map_location="cpu", weights_only=False)
        _validate_checkpoint(payload, contract, terminal=False)
        agent = _load_agent_from_checkpoint(
            payload, arm=arm, cost_path=cost_path, device=device
        )
        completed_rounds = int(payload["completed_damped_rounds"])
        records = list(payload["round_records"])
        previous_probe = (
            records[-1]["probe_rows"] if records else artifacts["previous_probe"]
        )
        print(f"Resumed damped fitting after round {completed_rounds}", flush=True)
    else:
        if latest_path.exists() or terminal_path.exists() or summary_path.exists():
            raise FileExistsError("damped artifacts exist; use --resume-existing")
        agent = extension.parent._new_terminal_agent(
            project_root, arm, artifacts["terminal"], device
        )
        completed_rounds = 0
        records = []
        previous_probe = artifacts["previous_probe"]

    env = benchmark._make_env(arm.payload)
    print(
        "Conditioned handling damped fitting | seed=0 | "
        f"device={device} | rho={POLYAK_RHO} | "
        f"rounds={DAMPED_ROUNDS}x{EPISODES_PER_ROUND} | "
        "evaluation_panel_reopened=false",
        flush=True,
    )
    for round_index in range(completed_rounds, DAMPED_ROUNDS):
        global_round_index = PARENT_GLOBAL_ROUNDS + round_index
        schedule = damped_lambda_schedule(round_index)
        before_state = _clone_state(agent.handling_network)
        before_digest = original._state_digest(before_state)
        episode_samples = []
        rows = []
        for offset, schedule_record in enumerate(schedule):
            instance_seed = (
                original.TRAIN_SEED_BASE
                + global_round_index * original.ROUND_SEED_STRIDE
                + offset
            )
            env.current_episode = int(schedule_record["global_episode_number"])
            run, samples = original._run_episode(
                agent,
                env,
                instance_seed=instance_seed,
                value=float(schedule_record["behavior_lambda"]),
                schedule=schedule_record,
                max_steps=args.max_steps,
                search=search,
                liveness=liveness,
            )
            run["episode_number"] = int(schedule_record["global_episode_number"])
            rows.append(original._compact_run(run))
            episode_samples.append(samples)
            agent.base_agent.decision_log.clear()
            if offset == 0 or (offset + 1) % args.log_every == 0:
                print(
                    f"Damped round {round_index + 1}/{DAMPED_ROUNDS} "
                    f"episode {offset + 1:2d}/{EPISODES_PER_ROUND} | "
                    f"lambda={schedule_record['behavior_lambda']:.4f} | "
                    f"rehandles={run['physical_rehandles']}",
                    flush=True,
                )
        training, validation, validation_indices = original._split_episode_samples(
            episode_samples, round_index=global_round_index
        )
        fit = fit_conditioned_future_handling(
            agent.handling_network,
            training,
            validation,
            device=device,
            epochs=original.FIT_EPOCHS,
            batch_size=original.FIT_BATCH_SIZE,
            learning_rate=original.FIT_LEARNING_RATE,
            seed=original.FIT_SEED + global_round_index,
        )
        provisional_digest = original._state_digest(
            agent.handling_network.state_dict()
        )
        provisional_validation = fit["final_validation"]
        polyak_blend_network(agent.handling_network, before_state)
        deployed_digest = original._state_digest(agent.handling_network.state_dict())
        if deployed_digest in (before_digest, provisional_digest):
            raise DampedConvergenceError(
                "damped model must differ from both previous and provisional states"
            )
        deployed_validation = handling_calibration(
            agent.handling_network, validation, device=device
        )
        probe_rows = original._probe(
            agent,
            env,
            max_steps=args.max_steps,
            search=search,
            liveness=liveness,
        )
        stability = extension._stability(
            previous_probe,
            probe_rows,
            {"final_validation": deployed_validation},
        )
        record = {
            "damped_round_number": round_index + 1,
            "global_round_number": global_round_index + 1,
            "policy_state_before_sha256": before_digest,
            "provisional_policy_state_sha256": provisional_digest,
            "deployed_policy_state_sha256": deployed_digest,
            "polyak_rho": POLYAK_RHO,
            "collection_rows": rows,
            "collection_summary": original._round_summary(rows),
            "training_sample_count": len(training),
            "validation_sample_count": len(validation),
            "validation_episode_indices": validation_indices,
            "fit": fit,
            "provisional_validation": provisional_validation,
            "deployed_after_damping_validation": deployed_validation,
            "probe_rows": probe_rows,
            "stability_against_previous_round": stability,
        }
        records.append(record)
        previous_probe = probe_rows
        completed_rounds = round_index + 1
        latest = _checkpoint(
            agent,
            arm=arm,
            cost_path=cost_path,
            contract=contract,
            artifacts=artifacts,
            completed_rounds=completed_rounds,
            records=records,
            role=LATEST_ROLE,
        )
        common._atomic_torch_save(latest, latest_path)
        print(
            f"Damped round {completed_rounds} | "
            f"changed_probes={stability['changed_probe_behavior_count']} | "
            f"probe_rehandles={stability['current_aggregate_probe_rehandles']} | "
            f"deployed_val_MAE={stability['final_validation_mae']:.4f} | "
            f"stable={stability['stable']}",
            flush=True,
        )

    final_records = records[-2:] if len(records) >= 2 else []
    convergence = {
        "assessed_transition_count": len(final_records),
        "required_final_stable_transitions": 2,
        "transitions": [
            {
                "from_global_round": record["global_round_number"] - 1,
                "to_global_round": record["global_round_number"],
                **record["stability_against_previous_round"],
            }
            for record in final_records
        ],
    }
    convergence["passed"] = bool(
        len(final_records) == 2
        and all(
            record["stability_against_previous_round"]["stable"]
            for record in final_records
        )
    )
    convergence["decision"] = (
        "authorize_frozen_85k_diagnostic_evaluation"
        if convergence["passed"]
        else "do_not_evaluate_or_train_more_seeds_reconsider_fitting"
    )

    terminal_sha = None
    if completed_rounds == DAMPED_ROUNDS:
        if terminal_path.is_file():
            terminal = torch.load(
                terminal_path, map_location="cpu", weights_only=False
            )
            _validate_checkpoint(terminal, contract, terminal=True)
        else:
            terminal = _checkpoint(
                agent,
                arm=arm,
                cost_path=cost_path,
                contract=contract,
                artifacts=artifacts,
                completed_rounds=completed_rounds,
                records=records,
                role=TERMINAL_ROLE,
            )
            terminal["convergence_assessment"] = convergence
            common._atomic_torch_save(terminal, terminal_path)
        terminal_sha = _sha256(terminal_path)
    result = {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "status": "complete" if terminal_sha else "paused",
        "parent_terminal_sha256": artifacts["terminal_sha256"],
        "contract_sha256": contract["contract_sha256"],
        "polyak_rho": POLYAK_RHO,
        "completed_damped_rounds": completed_rounds,
        "completed_collection_episodes": completed_rounds * EPISODES_PER_ROUND,
        "completed_probe_episodes": completed_rounds * len(original.PROBE_LAMBDAS),
        "round_records": records,
        "convergence_assessment": convergence,
        "latest_checkpoint": str(latest_path),
        "terminal_checkpoint": str(terminal_path) if terminal_sha else None,
        "terminal_checkpoint_sha256": terminal_sha,
        "evaluation_panel_reopened": False,
        "checkpoint_selection_used": False,
    }
    common._atomic_json_save(result, summary_path)
    print(
        json.dumps(
            {
                "status": result["status"],
                "polyak_rho": POLYAK_RHO,
                "completed_collection_episodes": result[
                    "completed_collection_episodes"
                ],
                "completed_probe_episodes": result["completed_probe_episodes"],
                "convergence_assessment": convergence,
                "summary": str(summary_path),
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    return result


if __name__ == "__main__":
    main()
