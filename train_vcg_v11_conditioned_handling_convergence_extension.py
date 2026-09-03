#!/usr/bin/env python3
"""Fixed two-round convergence extension for conditioned-handling VCG seed 0."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
from typing import Mapping, Optional, Sequence

import torch

import benchmark_viability_critic_priority as benchmark
import run_vcg_v11_conditioned_handling_seed0_85k as parent
import run_vcg_v11_nested_handling_pilot as pilot
import train_vcg_preference_conditioned as common
import train_vcg_v11_conditioned_handling_iterative as original
from vcg_v11_conditioned_handling import (
    ConditionedHandlingAgent,
    fit_conditioned_future_handling,
)


PROTOCOL = "vcg_v1_1_conditioned_handling_seed0_convergence_extension_v1"
SCHEMA_VERSION = 1
EXTENSION_ROUNDS = 2
EPISODES_PER_ROUND = original.EPISODES_PER_ROUND
TOTAL_COLLECTION_EPISODES = EXTENSION_ROUNDS * EPISODES_PER_ROUND
PARENT_GLOBAL_ROUNDS = original.ROUNDS
CONTRACT_NAME = "convergence-extension-contract.json"
LATEST_NAME = "latest.pth"
TERMINAL_NAME = "terminal.pth"
SUMMARY_NAME = "convergence-extension-summary.json"
MAX_CHANGED_PROBES_PER_TRANSITION = 1
MAX_VALIDATION_MAE = 0.35
MAX_ABSOLUTE_VALIDATION_BIAS = 0.10
LATEST_ROLE = "resumable_convergence_extension_round_boundary"
TERMINAL_ROLE = "fixed_two_round_convergence_extension_terminal"


class ConvergenceExtensionError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    if not path.is_file():
        raise ConvergenceExtensionError(f"missing required artifact: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_hash(value: Mapping) -> str:
    return hashlib.sha256(
        json.dumps(
            common._json_safe(dict(value)),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def extension_lambda_schedule(extension_round_index: int) -> tuple[dict, ...]:
    if extension_round_index not in range(EXTENSION_ROUNDS):
        raise ValueError("extension round index is out of range")
    global_round_index = PARENT_GLOBAL_ROUNDS + extension_round_index
    rng_seed = original.LAMBDA_SCHEDULE_SEED + global_round_index
    rng = random.Random(rng_seed)
    values = [
        0.2 * (index + rng.random()) / EPISODES_PER_ROUND
        for index in range(EPISODES_PER_ROUND)
    ]
    rng.shuffle(values)
    return tuple(
        {
            "extension_episode_number": (
                extension_round_index * EPISODES_PER_ROUND + offset + 1
            ),
            "global_episode_number": (
                original.TOTAL_EPISODES
                + extension_round_index * EPISODES_PER_ROUND
                + offset
                + 1
            ),
            "extension_round_number": extension_round_index + 1,
            "global_round_number": global_round_index + 1,
            "position_in_round": offset + 1,
            "behavior_lambda": float(value),
            "schedule_rng_seed": rng_seed,
            "stratified_uniform_over_[0,0.2]": True,
            "fixed_for_complete_episode": True,
        }
        for offset, value in enumerate(values)
    )


def _parent_artifacts(project_root: Path):
    parent_root = (
        project_root
        / "results/vcg-v1-1-conditioned-handling-seed0-85k-development"
    )
    parent_contract = parent._require_contract(project_root, parent_root)
    terminal, terminal_sha = parent._terminal(project_root, parent_root)
    summary_path = (
        parent_root / parent.TRAINING_RELATIVE_PATH / "training-summary.json"
    )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("status") != "complete":
        raise ConvergenceExtensionError("parent four-round training is incomplete")
    records = summary.get("round_records")
    if not isinstance(records, list) or len(records) != PARENT_GLOBAL_ROUNDS:
        raise ConvergenceExtensionError("parent round records are incomplete")
    previous_probe = records[-1].get("probe_rows")
    if not isinstance(previous_probe, list) or len(previous_probe) != len(
        original.PROBE_LAMBDAS
    ):
        raise ConvergenceExtensionError("parent round-4 probes are incomplete")
    return {
        "root": parent_root,
        "contract": parent_contract,
        "terminal": terminal,
        "terminal_sha256": terminal_sha,
        "summary": summary,
        "summary_sha256": _sha256(summary_path),
        "previous_probe": previous_probe,
    }


def _contract(project_root: Path, output_dir: Path, *, max_steps: int) -> dict:
    artifacts = _parent_artifacts(project_root)
    schedules = [
        extension_lambda_schedule(index) for index in range(EXTENSION_ROUNDS)
    ]
    semantic = {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "scope": "post_evaluation_seed0_training_only_convergence_diagnostic",
        "parent_contract_sha256": artifacts["contract"]["contract_sha256"],
        "parent_terminal_sha256": artifacts["terminal_sha256"],
        "parent_training_summary_sha256": artifacts["summary_sha256"],
        "parent_completed_rounds": PARENT_GLOBAL_ROUNDS,
        "fixed_extension_rounds": EXTENSION_ROUNDS,
        "episodes_per_round": EPISODES_PER_ROUND,
        "total_collection_episodes": TOTAL_COLLECTION_EPISODES,
        "probe_episodes_per_round": len(original.PROBE_LAMBDAS),
        "total_probe_episodes": EXTENSION_ROUNDS * len(original.PROBE_LAMBDAS),
        "total_additional_simulator_episodes": (
            TOTAL_COLLECTION_EPISODES
            + EXTENSION_ROUNDS * len(original.PROBE_LAMBDAS)
        ),
        "global_rounds_after_extension": (
            PARENT_GLOBAL_ROUNDS + EXTENSION_ROUNDS
        ),
        "lambda_schedules": schedules,
        "instance_seed_formula": (
            "58000000 + global_zero_based_round*10000 + position"
        ),
        "max_steps": int(max_steps),
        "fit_epochs": original.FIT_EPOCHS,
        "fit_batch_size": original.FIT_BATCH_SIZE,
        "fit_learning_rate": original.FIT_LEARNING_RATE,
        "whole_episode_validation_split": True,
        "validation_episodes_per_round": original.VALIDATION_EPISODES,
        "policy_frozen_within_collection_round": True,
        "current_round_only_fitting": True,
        "operational_critic_frozen": True,
        "complete_episode_future_handling_mc": True,
        "evaluation_panel_reopened": False,
        "checkpoint_selection": False,
        "fixed_two_round_terminal": True,
        "probe_stability_rule": {
            "applied_to_both_parent4_to_extension1_and_extension1_to_extension2": True,
            "maximum_changed_behavior_digests": (
                MAX_CHANGED_PROBES_PER_TRANSITION
            ),
            "aggregate_physical_rehandles_must_be_unchanged": True,
            "all_probe_episodes_strict_safe_complete": True,
            "maximum_validation_mae": MAX_VALIDATION_MAE,
            "maximum_absolute_validation_bias": (
                MAX_ABSOLUTE_VALIDATION_BIAS
            ),
            "performance_return_or_mae_not_used_for_checkpoint_selection": True,
        },
        "source_sha256": {
            "extension_trainer": _sha256(Path(__file__).resolve()),
            "conditioned_controller": _sha256(
                project_root / "vcg_v11_conditioned_handling.py"
            ),
            "parent_trainer": _sha256(
                project_root / "train_vcg_v11_conditioned_handling_iterative.py"
            ),
            "parent_runner": _sha256(
                project_root / "run_vcg_v11_conditioned_handling_seed0_85k.py"
            ),
        },
        "output_dir": str(output_dir.resolve()),
    }
    return {**semantic, "contract_sha256": _canonical_hash(semantic)}


def _ensure_contract(project_root: Path, output_dir: Path, *, max_steps: int) -> dict:
    expected = _contract(project_root, output_dir, max_steps=max_steps)
    path = output_dir / CONTRACT_NAME
    if path.is_file():
        observed = json.loads(path.read_text(encoding="utf-8"))
        if observed != expected:
            raise ConvergenceExtensionError(
                "extension contract, parent artifacts, or sources changed"
            )
    else:
        if output_dir.exists() and any(output_dir.iterdir()):
            raise ConvergenceExtensionError(
                "nonempty extension output has no authenticated contract"
            )
        pilot._atomic_json(path, expected)
    return expected


def _stability(previous_rows, current_rows, fit: Mapping) -> dict:
    if len(previous_rows) != len(current_rows):
        raise ConvergenceExtensionError("probe grids do not align")
    changed = []
    for previous, current in zip(previous_rows, current_rows):
        if float(previous["behavior_lambda"]) != float(current["behavior_lambda"]):
            raise ConvergenceExtensionError("probe lambda order changed")
        if previous["behavior_digest"] != current["behavior_digest"]:
            changed.append(
                {
                    "lambda": float(current["behavior_lambda"]),
                    "previous_rehandles": int(previous["physical_rehandles"]),
                    "current_rehandles": int(current["physical_rehandles"]),
                    "previous_return": float(previous["return"]),
                    "current_return": float(current["return"]),
                    "previous_mae": previous["mean_absolute_error"],
                    "current_mae": current["mean_absolute_error"],
                }
            )
    previous_rehandles = sum(int(row["physical_rehandles"]) for row in previous_rows)
    current_rehandles = sum(int(row["physical_rehandles"]) for row in current_rows)
    validation = fit["final_validation"]
    criteria = {
        "changed_probe_behaviors_at_most_one": len(changed)
        <= MAX_CHANGED_PROBES_PER_TRANSITION,
        "aggregate_probe_rehandles_unchanged": previous_rehandles
        == current_rehandles,
        "all_current_probes_strict_safe_complete": all(
            bool(row["strict_method_success"]) for row in current_rows
        ),
        "validation_mae_within_bound": float(validation["mae"])
        <= MAX_VALIDATION_MAE,
        "absolute_validation_bias_within_bound": abs(float(validation["bias"]))
        <= MAX_ABSOLUTE_VALIDATION_BIAS,
    }
    return {
        "changed_probe_behavior_count": len(changed),
        "changed_probes": changed,
        "previous_aggregate_probe_rehandles": previous_rehandles,
        "current_aggregate_probe_rehandles": current_rehandles,
        "final_validation_mae": float(validation["mae"]),
        "final_validation_bias": float(validation["bias"]),
        "criteria": criteria,
        "stable": all(criteria.values()),
    }


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
        "completed_extension_rounds": int(completed_rounds),
        "completed_extension_collection_episodes": int(
            completed_rounds * EPISODES_PER_ROUND
        ),
        "global_completed_rounds": int(PARENT_GLOBAL_ROUNDS + completed_rounds),
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
        raise ConvergenceExtensionError(f"extension checkpoint mismatch: {mismatch}")
    if payload.get("contract") != contract:
        raise ConvergenceExtensionError("extension checkpoint contract changed")
    completed = int(payload.get("completed_extension_rounds", -1))
    if len(tuple(payload.get("round_records", ()))) != completed:
        raise ConvergenceExtensionError("extension checkpoint round clock changed")
    if terminal and completed != EXTENSION_ROUNDS:
        raise ConvergenceExtensionError("extension terminal is incomplete")


def _build_parent_agent(project_root: Path, artifacts: Mapping, arm, device):
    return parent._new_terminal_agent(
        project_root, arm, artifacts["terminal"], device
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
        base = pilot._fresh_base(arm, device)
        agent = ConditionedHandlingAgent.from_checkpoint(
            payload["agent_checkpoint"],
            base_agent=base,
            expected_base_checkpoint_sha256=arm.checkpoint_sha256,
            expected_base_policy_digest=arm.deployment_policy_digest,
            expected_source_cost_sha256=_sha256(cost_path),
            seed=original.MODEL_SEED,
        )
        completed_rounds = int(payload["completed_extension_rounds"])
        records = list(payload["round_records"])
        previous_probe = (
            records[-1]["probe_rows"] if records else artifacts["previous_probe"]
        )
        print(f"Resumed extension after round {completed_rounds}", flush=True)
    else:
        if latest_path.exists() or terminal_path.exists() or summary_path.exists():
            raise FileExistsError(
                "extension artifacts already exist; use --resume-existing"
            )
        agent = _build_parent_agent(project_root, artifacts, arm, device)
        completed_rounds = 0
        records = []
        previous_probe = artifacts["previous_probe"]

    env = benchmark._make_env(arm.payload)
    print(
        "Conditioned handling convergence extension | seed=0 | "
        f"device={device} | rounds={EXTENSION_ROUNDS}x{EPISODES_PER_ROUND} | "
        "evaluation_panel_reopened=false",
        flush=True,
    )
    for extension_index in range(completed_rounds, EXTENSION_ROUNDS):
        global_round_index = PARENT_GLOBAL_ROUNDS + extension_index
        schedule = extension_lambda_schedule(extension_index)
        before_digest = original._state_digest(agent.handling_network.state_dict())
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
                    f"Extension round {extension_index + 1}/{EXTENSION_ROUNDS} "
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
        after_digest = original._state_digest(agent.handling_network.state_dict())
        if before_digest == after_digest:
            raise ConvergenceExtensionError("extension fit did not update the model")
        probe_rows = original._probe(
            agent,
            env,
            max_steps=args.max_steps,
            search=search,
            liveness=liveness,
        )
        stability = _stability(previous_probe, probe_rows, fit)
        record = {
            "extension_round_number": extension_index + 1,
            "global_round_number": global_round_index + 1,
            "policy_state_before_sha256": before_digest,
            "policy_state_after_sha256": after_digest,
            "collection_rows": rows,
            "collection_summary": original._round_summary(rows),
            "training_sample_count": len(training),
            "validation_sample_count": len(validation),
            "validation_episode_indices": validation_indices,
            "fit": fit,
            "probe_rows": probe_rows,
            "stability_against_previous_round": stability,
        }
        records.append(record)
        previous_probe = probe_rows
        completed_rounds = extension_index + 1
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
            f"Extension round {completed_rounds} | "
            f"changed_probes={stability['changed_probe_behavior_count']} | "
            f"probe_rehandles={stability['current_aggregate_probe_rehandles']} | "
            f"val_MAE={stability['final_validation_mae']:.4f} | "
            f"stable={stability['stable']}",
            flush=True,
        )

    convergence = {
        "required_stable_transitions": EXTENSION_ROUNDS,
        "observed_stable_transitions": sum(
            bool(record["stability_against_previous_round"]["stable"])
            for record in records
        ),
        "each_transition": [
            {
                "from_global_round": record["global_round_number"] - 1,
                "to_global_round": record["global_round_number"],
                **record["stability_against_previous_round"],
            }
            for record in records
        ],
    }
    convergence["passed"] = bool(
        len(records) == EXTENSION_ROUNDS
        and all(
            record["stability_against_previous_round"]["stable"]
            for record in records
        )
    )
    convergence["decision"] = (
        "authorize_matched_seeds_1_and_2_preparation"
        if convergence["passed"]
        else "do_not_train_more_seeds_reconsider_fitting_stability"
    )

    terminal_sha = None
    if completed_rounds == EXTENSION_ROUNDS:
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
        "completed_extension_rounds": completed_rounds,
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
