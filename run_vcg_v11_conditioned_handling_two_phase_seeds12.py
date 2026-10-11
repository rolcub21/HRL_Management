#!/usr/bin/env python3
"""Matched seed-1/2 training for the finalized two-phase handling controller."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Mapping, Optional, Sequence

import torch

import benchmark_viability_critic_priority as benchmark
import run_vcg_v11_nested_handling_pilot as pilot
import run_vcg_v11_nested_handling_seed_stability as nested_stability
import train_vcg_preference_conditioned as common
import train_vcg_v11_conditioned_handling_convergence_extension as stability_rule
import train_vcg_v11_conditioned_handling_damped_convergence as damped
import train_vcg_v11_conditioned_handling_iterative as full
from vcg_v11_conditioned_handling import (
    ConditionedFutureHandlingNetwork,
    ConditionedHandlingAgent,
    ConditionedHandlingConfig,
    fit_conditioned_future_handling,
    handling_calibration,
)


PROTOCOL = "vcg_v1_1_conditioned_handling_two_phase_seeds12_development_v1"
SCHEMA_VERSION = 1
MODEL_SEEDS = (1, 2)
FULL_ROUNDS = 4
DAMPED_ROUNDS = 4
TOTAL_ROUNDS = FULL_ROUNDS + DAMPED_ROUNDS
EPISODES_PER_ROUND = full.EPISODES_PER_ROUND
COLLECTION_EPISODES_PER_SEED = TOTAL_ROUNDS * EPISODES_PER_ROUND
PROBE_EPISODES_PER_SEED = TOTAL_ROUNDS * len(full.PROBE_LAMBDAS)
SIMULATOR_EPISODES_PER_SEED = (
    COLLECTION_EPISODES_PER_SEED + PROBE_EPISODES_PER_SEED
)
NESTED_ROOT_RELATIVE = Path(
    "results/vcg-v1-1-nested-handling-seed-stability-85k-development"
)
SEED0_DAMPED_ROOT_RELATIVE = Path(
    "results/vcg-v1-1-conditioned-handling-seed0-damped-convergence"
)
CONTRACT_NAME = "two-phase-seeds12-contract.json"
REPORT_NAME = "two-phase-seeds12-training-report.json"
LATEST_ROLE = "two_phase_round_boundary_resumable"
TERMINAL_ROLE = "fixed_eight_round_two_phase_terminal"


class TwoPhaseSeedError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    if not path.is_file():
        raise TwoPhaseSeedError(f"missing required artifact: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_form(value):
    return json.loads(
        json.dumps(
            common._json_safe(value),
            sort_keys=True,
            separators=(",", ":"),
        )
    )


def _canonical_hash(value: Mapping) -> str:
    return hashlib.sha256(
        json.dumps(
            _json_form(dict(value)), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()


def recipe_schedule(round_index: int) -> tuple[dict, ...]:
    if round_index not in range(TOTAL_ROUNDS):
        raise ValueError("recipe round index is out of range")
    if round_index < FULL_ROUNDS:
        return tuple(full.stratified_lambda_schedule(round_index))
    return tuple(damped.damped_lambda_schedule(round_index - FULL_ROUNDS))


def recipe_phase(round_index: int) -> str:
    if round_index not in range(TOTAL_ROUNDS):
        raise ValueError("recipe round index is out of range")
    return "full_update" if round_index < FULL_ROUNDS else "rho_0p25_damped"


def recipe_instance_seed(round_index: int, position: int) -> int:
    if round_index not in range(TOTAL_ROUNDS):
        raise ValueError("recipe round index is out of range")
    if position not in range(EPISODES_PER_ROUND):
        raise ValueError("recipe position is out of range")
    return full.TRAIN_SEED_BASE + round_index * full.ROUND_SEED_STRIDE + position


def _seed_paths(output_root: Path, seed: int) -> dict:
    root = output_root / f"seed-{seed}"
    return {
        "root": root,
        "latest": root / "latest.pth",
        "terminal": root / "terminal.pth",
        "summary": root / "training-summary.json",
    }


def _authenticate_inputs(project_root: Path):
    nested_root = project_root / NESTED_ROOT_RELATIVE
    arms, sources, _latest, _datasets, _parent, _records = (
        nested_stability._authenticate_inputs(project_root)
    )
    nested_contract_path = nested_root / nested_stability.CONTRACT_NAME
    nested_contract = json.loads(nested_contract_path.read_text(encoding="utf-8"))
    if nested_contract.get("protocol") != nested_stability.PROTOCOL:
        raise TwoPhaseSeedError("nested seed-stability protocol changed")
    if nested_contract.get("source_sha256") != nested_stability._source_hashes():
        raise TwoPhaseSeedError("nested seed-stability sources changed")
    nested_report_path = nested_root / nested_stability.REPORT_NAME
    nested_report = json.loads(nested_report_path.read_text(encoding="utf-8"))
    if nested_report.get("status") != "passed":
        raise TwoPhaseSeedError("nested seed-stability parent did not pass")
    seed_inputs = {}
    for seed in MODEL_SEEDS:
        arm = arms[seed]
        cost_path = nested_stability._cost_path(nested_root, seed)
        cost_report_path = nested_stability._cost_report_path(nested_root, seed)
        cost_report = json.loads(cost_report_path.read_text(encoding="utf-8"))
        cost_sha = _sha256(cost_path)
        if cost_report.get("status") != "complete":
            raise TwoPhaseSeedError(f"seed-{seed} nested handling head is incomplete")
        if cost_report.get("cost_checkpoint_sha256") != cost_sha:
            raise TwoPhaseSeedError(f"seed-{seed} nested handling identity changed")
        seed_inputs[seed] = {
            "arm": arm,
            "cost_path": cost_path,
            "cost_sha256": cost_sha,
            "cost_report_path": cost_report_path,
            "cost_report_sha256": _sha256(cost_report_path),
            "q_state_sha256": pilot._state_digest(
                arm.payload["agent_state"]["Q_local"]
            ),
        }
    seed0_root = project_root / SEED0_DAMPED_ROOT_RELATIVE
    seed0_contract_path = seed0_root / damped.CONTRACT_NAME
    seed0_contract = json.loads(seed0_contract_path.read_text(encoding="utf-8"))
    expected_seed0_contract = damped._contract(
        project_root, seed0_root, max_steps=2_000
    )
    if seed0_contract != _json_form(expected_seed0_contract):
        raise TwoPhaseSeedError("seed-0 damped recipe contract changed")
    seed0_summary_path = seed0_root / damped.SUMMARY_NAME
    seed0_summary = json.loads(seed0_summary_path.read_text(encoding="utf-8"))
    if (
        seed0_summary.get("status") != "complete"
        or seed0_summary.get("convergence_assessment", {}).get("passed") is not True
    ):
        raise TwoPhaseSeedError("seed-0 damped recipe did not pass convergence")
    return {
        "arms": arms,
        "sources": sources,
        "seed_inputs": seed_inputs,
        "nested_contract": nested_contract,
        "nested_contract_sha256": _sha256(nested_contract_path),
        "nested_report_sha256": _sha256(nested_report_path),
        "seed0_contract": seed0_contract,
        "seed0_summary_sha256": _sha256(seed0_summary_path),
        "seed0_terminal_sha256": seed0_summary["terminal_checkpoint_sha256"],
    }


def _contract(project_root: Path, output_root: Path, *, max_steps: int) -> dict:
    inputs = _authenticate_inputs(project_root)
    schedules = [recipe_schedule(index) for index in range(TOTAL_ROUNDS)]
    identities = [
        {
            "instance_seed": seed,
            "episode_instance_id": inputs["sources"].instances[seed].instance_id,
            "schedule_id": inputs["sources"].instances[seed].schedule_id,
        }
        for seed in nested_stability.INSTANCE_SEEDS
    ]
    semantic = {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "scope": "matched_model_seed_training_on_development_streams_only",
        "model_seeds": list(MODEL_SEEDS),
        "training_recipe": {
            "full_update_rounds": FULL_ROUNDS,
            "rho_0p25_damped_rounds": DAMPED_ROUNDS,
            "total_rounds": TOTAL_ROUNDS,
            "episodes_per_round": EPISODES_PER_ROUND,
            "collection_episodes_per_seed": COLLECTION_EPISODES_PER_SEED,
            "probe_episodes_per_seed": PROBE_EPISODES_PER_SEED,
            "simulator_episodes_per_seed": SIMULATOR_EPISODES_PER_SEED,
            "polyak_rho": damped.POLYAK_RHO,
            "same_environment_and_lambda_streams_across_model_seeds": True,
            "same_streams_as_finalized_seed0_recipe": True,
            "lambda_schedules": schedules,
            "instance_seed_formula": (
                "58000000 + zero_based_recipe_round*10000 + position"
            ),
            "policy_frozen_within_collection_round": True,
            "complete_episode_future_rehandle_mc": True,
            "current_round_only_fitting": True,
            "fit_epochs": full.FIT_EPOCHS,
            "fit_batch_size": full.FIT_BATCH_SIZE,
            "fit_learning_rate": full.FIT_LEARNING_RATE,
            "max_steps": int(max_steps),
        },
        "stability_rule": {
            "final_two_damped_transitions_must_pass": True,
            "maximum_changed_probe_behaviors": (
                stability_rule.MAX_CHANGED_PROBES_PER_TRANSITION
            ),
            "aggregate_probe_rehandles_unchanged": True,
            "maximum_validation_mae": stability_rule.MAX_VALIDATION_MAE,
            "maximum_absolute_validation_bias": (
                stability_rule.MAX_ABSOLUTE_VALIDATION_BIAS
            ),
        },
        "operational_critic_frozen": True,
        "lambda_zero_direct_vcg_v1_1_delegation": True,
        "independent_seed_specific_nested_warm_starts": True,
        "evaluation_panels_opened": False,
        "checkpoint_selection": False,
        "fixed_round8_terminals": True,
        "seed_inputs": {
            str(seed): {
                "selected_checkpoint_sha256": inputs["seed_inputs"][seed][
                    "arm"
                ].checkpoint_sha256,
                "selected_policy_digest": inputs["seed_inputs"][seed][
                    "arm"
                ].deployment_policy_digest,
                "selected_episode": nested_stability.SELECTED_EPISODES[seed],
                "q_state_sha256": inputs["seed_inputs"][seed]["q_state_sha256"],
                "nested_cost_sha256": inputs["seed_inputs"][seed]["cost_sha256"],
                "nested_cost_report_sha256": inputs["seed_inputs"][seed][
                    "cost_report_sha256"
                ],
            }
            for seed in MODEL_SEEDS
        },
        "nested_seed_stability_contract_sha256": inputs[
            "nested_contract_sha256"
        ],
        "nested_seed_stability_report_sha256": inputs["nested_report_sha256"],
        "seed0_finalized_recipe": {
            "damped_contract_sha256": inputs["seed0_contract"]["contract_sha256"],
            "summary_sha256": inputs["seed0_summary_sha256"],
            "terminal_sha256": inputs["seed0_terminal_sha256"],
        },
        "opened_85k_instance_identities_for_future_evaluation": identities,
        "source_sha256": {
            "two_phase_runner": _sha256(Path(__file__).resolve()),
            "conditioned_controller": _sha256(
                project_root / "vcg_v11_conditioned_handling.py"
            ),
            "full_update_trainer": _sha256(
                project_root / "train_vcg_v11_conditioned_handling_iterative.py"
            ),
            "damped_trainer": _sha256(
                project_root / "train_vcg_v11_conditioned_handling_damped_convergence.py"
            ),
            "nested_seed_stability": _sha256(
                project_root / "run_vcg_v11_nested_handling_seed_stability.py"
            ),
        },
        "output_root": str(output_root.resolve()),
    }
    return {**semantic, "contract_sha256": _canonical_hash(semantic)}


def prepare(project_root: Path, output_root: Path, *, max_steps: int) -> dict:
    expected = _json_form(_contract(project_root, output_root, max_steps=max_steps))
    path = output_root / CONTRACT_NAME
    if path.is_file():
        if json.loads(path.read_text(encoding="utf-8")) != expected:
            raise TwoPhaseSeedError("two-phase contract, inputs, or sources changed")
    else:
        if output_root.exists() and any(output_root.iterdir()):
            raise TwoPhaseSeedError("nonempty output root has no two-phase contract")
        pilot._atomic_json(path, expected)
    return {
        "status": "prepared",
        "model_seeds": list(MODEL_SEEDS),
        "simulator_episodes_per_seed": SIMULATOR_EPISODES_PER_SEED,
        "total_simulator_episodes": len(MODEL_SEEDS) * SIMULATOR_EPISODES_PER_SEED,
        "contract": str(path.resolve()),
    }


def _require_contract(project_root: Path, output_root: Path, *, max_steps: int):
    path = output_root / CONTRACT_NAME
    if not path.is_file():
        raise TwoPhaseSeedError("run prepare first")
    observed = json.loads(path.read_text(encoding="utf-8"))
    expected = _json_form(_contract(project_root, output_root, max_steps=max_steps))
    if observed != expected:
        raise TwoPhaseSeedError("two-phase contract, inputs, or sources changed")
    return observed


def _new_agent(project_root: Path, inputs: Mapping, *, seed: int, device):
    seed_input = inputs["seed_inputs"][seed]
    arm = seed_input["arm"]
    base = pilot._fresh_base(arm, device)
    nested_root = project_root / NESTED_ROOT_RELATIVE
    old_cost = nested_stability._load_cost(
        nested_root,
        arm,
        seed=seed,
        device=device,
        config=base.config,
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
    network = ConditionedFutureHandlingNetwork(config, seed=seed).to(device)
    network.initialize_from_nested_head(old_cost)
    return ConditionedHandlingAgent(
        base, network, config=config, seed=seed, epsilon=0.0
    )


def _load_agent(payload, inputs, *, seed: int, device):
    seed_input = inputs["seed_inputs"][seed]
    arm = seed_input["arm"]
    return ConditionedHandlingAgent.from_checkpoint(
        payload["agent_checkpoint"],
        base_agent=pilot._fresh_base(arm, device),
        expected_base_checkpoint_sha256=arm.checkpoint_sha256,
        expected_base_policy_digest=arm.deployment_policy_digest,
        expected_source_cost_sha256=seed_input["cost_sha256"],
        seed=seed,
    )


def _checkpoint(
    agent,
    *,
    inputs: Mapping,
    seed: int,
    contract: Mapping,
    completed_rounds: int,
    records,
    role: str,
) -> dict:
    seed_input = inputs["seed_inputs"][seed]
    arm = seed_input["arm"]
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "checkpoint_role": role,
        "model_seed": seed,
        "completed_recipe_rounds": int(completed_rounds),
        "completed_collection_episodes": int(
            completed_rounds * EPISODES_PER_ROUND
        ),
        "contract_sha256": contract["contract_sha256"],
        "agent_checkpoint": agent.checkpoint(
            base_checkpoint_sha256=arm.checkpoint_sha256,
            base_policy_digest=arm.deployment_policy_digest,
            source_cost_sha256=seed_input["cost_sha256"],
        ),
        "round_records": tuple(records),
        "fixed_terminal_checkpoint": role == TERMINAL_ROLE,
        "checkpoint_selection_used": False,
        "evaluation_panels_opened": False,
    }


def _validate_checkpoint(
    payload: Mapping, contract: Mapping, *, seed: int, terminal: bool
):
    expected = {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "checkpoint_role": TERMINAL_ROLE if terminal else LATEST_ROLE,
        "model_seed": seed,
        "contract_sha256": contract["contract_sha256"],
        "fixed_terminal_checkpoint": terminal,
        "checkpoint_selection_used": False,
        "evaluation_panels_opened": False,
    }
    mismatch = {
        key: (payload.get(key), value)
        for key, value in expected.items()
        if payload.get(key) != value
    }
    if mismatch:
        raise TwoPhaseSeedError(f"seed-{seed} checkpoint mismatch: {mismatch}")
    completed = int(payload.get("completed_recipe_rounds", -1))
    if len(tuple(payload.get("round_records", ()))) != completed:
        raise TwoPhaseSeedError(f"seed-{seed} checkpoint round clock changed")
    if terminal and completed != TOTAL_ROUNDS:
        raise TwoPhaseSeedError(f"seed-{seed} terminal is incomplete")


def train_seed(
    project_root: Path,
    output_root: Path,
    *,
    seed: int,
    device_name: str,
    max_steps: int,
    log_every: int,
    resume_existing: bool,
) -> dict:
    if seed not in MODEL_SEEDS:
        raise TwoPhaseSeedError(f"model seed must be one of {MODEL_SEEDS}")
    contract = _require_contract(
        project_root, output_root, max_steps=max_steps
    )
    inputs = _authenticate_inputs(project_root)
    paths = _seed_paths(output_root, seed)
    device = full.resolve_device(device_name)
    arm = inputs["seed_inputs"][seed]["arm"]
    search = benchmark._search_config(arm.payload)
    liveness = benchmark._liveness_rule(arm.payload)
    if resume_existing:
        payload = torch.load(
            paths["latest"], map_location="cpu", weights_only=False
        )
        _validate_checkpoint(payload, contract, seed=seed, terminal=False)
        agent = _load_agent(payload, inputs, seed=seed, device=device)
        completed_rounds = int(payload["completed_recipe_rounds"])
        records = list(payload["round_records"])
        previous_probe = records[-1]["probe_rows"] if records else None
        print(f"Resumed seed {seed} after recipe round {completed_rounds}", flush=True)
    else:
        if any(paths[name].exists() for name in ("latest", "terminal", "summary")):
            raise FileExistsError(f"seed-{seed} artifacts exist; resume the run")
        paths["root"].mkdir(parents=True, exist_ok=True)
        agent = _new_agent(project_root, inputs, seed=seed, device=device)
        completed_rounds = 0
        records = []
        previous_probe = None
    env = benchmark._make_env(arm.payload)
    print(
        f"Two-phase conditioned handling | seed={seed} | device={device} | "
        f"rounds={FULL_ROUNDS} full + {DAMPED_ROUNDS} damped | "
        "evaluation=false",
        flush=True,
    )
    for round_index in range(completed_rounds, TOTAL_ROUNDS):
        phase = recipe_phase(round_index)
        before_state = damped._clone_state(agent.handling_network)
        before_digest = full._state_digest(before_state)
        episode_samples = []
        rows = []
        schedule = recipe_schedule(round_index)
        for position, schedule_record in enumerate(schedule):
            instance_seed = recipe_instance_seed(round_index, position)
            env.current_episode = round_index * EPISODES_PER_ROUND + position + 1
            run, samples = full._run_episode(
                agent,
                env,
                instance_seed=instance_seed,
                value=float(schedule_record["behavior_lambda"]),
                schedule=schedule_record,
                max_steps=max_steps,
                search=search,
                liveness=liveness,
            )
            run["episode_number"] = env.current_episode
            rows.append(full._compact_run(run))
            episode_samples.append(samples)
            agent.base_agent.decision_log.clear()
            if position == 0 or (position + 1) % log_every == 0:
                print(
                    f"Seed {seed} round {round_index + 1}/{TOTAL_ROUNDS} "
                    f"({phase}) episode {position + 1:2d}/{EPISODES_PER_ROUND} | "
                    f"lambda={schedule_record['behavior_lambda']:.4f} | "
                    f"rehandles={run['physical_rehandles']}",
                    flush=True,
                )
        training, validation, validation_indices = full._split_episode_samples(
            episode_samples, round_index=round_index
        )
        fit = fit_conditioned_future_handling(
            agent.handling_network,
            training,
            validation,
            device=device,
            epochs=full.FIT_EPOCHS,
            batch_size=full.FIT_BATCH_SIZE,
            learning_rate=full.FIT_LEARNING_RATE,
            seed=full.FIT_SEED + round_index,
        )
        provisional_digest = full._state_digest(agent.handling_network.state_dict())
        provisional_validation = fit["final_validation"]
        if phase == "rho_0p25_damped":
            damped.polyak_blend_network(agent.handling_network, before_state)
        deployed_digest = full._state_digest(agent.handling_network.state_dict())
        if deployed_digest == before_digest:
            raise TwoPhaseSeedError(f"seed-{seed} round {round_index + 1} did not update")
        if phase == "rho_0p25_damped" and deployed_digest == provisional_digest:
            raise TwoPhaseSeedError("damped deployment equals provisional fit")
        deployed_validation = handling_calibration(
            agent.handling_network, validation, device=device
        )
        probe_rows = full._probe(
            agent,
            env,
            max_steps=max_steps,
            search=search,
            liveness=liveness,
        )
        stability = (
            None
            if previous_probe is None
            else stability_rule._stability(
                previous_probe,
                probe_rows,
                {"final_validation": deployed_validation},
            )
        )
        record = {
            "recipe_round_number": round_index + 1,
            "phase": phase,
            "policy_state_before_sha256": before_digest,
            "provisional_policy_state_sha256": provisional_digest,
            "deployed_policy_state_sha256": deployed_digest,
            "polyak_rho": damped.POLYAK_RHO if phase == "rho_0p25_damped" else 1.0,
            "collection_rows": rows,
            "collection_summary": full._round_summary(rows),
            "training_sample_count": len(training),
            "validation_sample_count": len(validation),
            "validation_episode_indices": validation_indices,
            "fit": fit,
            "provisional_validation": provisional_validation,
            "deployed_validation": deployed_validation,
            "probe_rows": probe_rows,
            "stability_against_previous_round": stability,
        }
        records.append(record)
        previous_probe = probe_rows
        completed_rounds = round_index + 1
        latest = _checkpoint(
            agent,
            inputs=inputs,
            seed=seed,
            contract=contract,
            completed_rounds=completed_rounds,
            records=records,
            role=LATEST_ROLE,
        )
        common._atomic_torch_save(latest, paths["latest"])
        print(
            f"Seed {seed} round {completed_rounds} | phase={phase} | "
            f"changed_probes={None if stability is None else stability['changed_probe_behavior_count']} | "
            f"deployed_val_MAE={deployed_validation['mae']:.4f}",
            flush=True,
        )
    assessed = records[-2:] if len(records) >= 2 else []
    stability_gate = {
        "assessed_final_transition_count": len(assessed),
        "required_stable_transitions": 2,
        "transitions": [
            record["stability_against_previous_round"] for record in assessed
        ],
    }
    stability_gate["passed"] = bool(
        len(assessed) == 2
        and all(
            record["phase"] == "rho_0p25_damped"
            and record["stability_against_previous_round"] is not None
            and record["stability_against_previous_round"]["stable"]
            for record in assessed
        )
    )
    stability_gate["decision"] = (
        "authorize_frozen_seed_evaluation"
        if stability_gate["passed"]
        else "do_not_evaluate_seed"
    )
    terminal_sha = None
    if completed_rounds == TOTAL_ROUNDS:
        if paths["terminal"].is_file():
            terminal = torch.load(
                paths["terminal"], map_location="cpu", weights_only=False
            )
            _validate_checkpoint(terminal, contract, seed=seed, terminal=True)
        else:
            terminal = _checkpoint(
                agent,
                inputs=inputs,
                seed=seed,
                contract=contract,
                completed_rounds=completed_rounds,
                records=records,
                role=TERMINAL_ROLE,
            )
            terminal["stability_gate"] = stability_gate
            common._atomic_torch_save(terminal, paths["terminal"])
        terminal_sha = _sha256(paths["terminal"])
    result = {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "status": "complete" if terminal_sha else "paused",
        "model_seed": seed,
        "contract_sha256": contract["contract_sha256"],
        "completed_recipe_rounds": completed_rounds,
        "completed_collection_episodes": completed_rounds * EPISODES_PER_ROUND,
        "completed_probe_episodes": completed_rounds * len(full.PROBE_LAMBDAS),
        "round_records": records,
        "stability_gate": stability_gate,
        "latest_checkpoint": str(paths["latest"]),
        "terminal_checkpoint": str(paths["terminal"]) if terminal_sha else None,
        "terminal_checkpoint_sha256": terminal_sha,
        "evaluation_panels_opened": False,
        "checkpoint_selection_used": False,
    }
    common._atomic_json_save(result, paths["summary"])
    return result


def analyze_training(project_root: Path, output_root: Path, *, max_steps: int):
    contract = _require_contract(project_root, output_root, max_steps=max_steps)
    seeds = {}
    for seed in MODEL_SEEDS:
        paths = _seed_paths(output_root, seed)
        if not paths["summary"].is_file() or not paths["terminal"].is_file():
            raise TwoPhaseSeedError(f"seed-{seed} training is incomplete")
        summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
        terminal = torch.load(paths["terminal"], map_location="cpu", weights_only=False)
        _validate_checkpoint(terminal, contract, seed=seed, terminal=True)
        if terminal.get("stability_gate") != summary.get("stability_gate"):
            raise TwoPhaseSeedError(f"seed-{seed} terminal stability changed")
        seeds[str(seed)] = {
            "status": summary["status"],
            "terminal_checkpoint_sha256": _sha256(paths["terminal"]),
            "stability_gate": summary["stability_gate"],
        }
    passed = all(item["stability_gate"]["passed"] for item in seeds.values())
    report = {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "status": "passed" if passed else "failed",
        "contract_sha256": contract["contract_sha256"],
        "seed_training": seeds,
        "both_seeds_stable": passed,
        "decision": (
            "prepare_matched_three_seed_frozen_evaluation"
            if passed
            else "do_not_open_evaluation_inspect_unstable_seed"
        ),
        "evaluation_panels_opened": False,
    }
    path = output_root / REPORT_NAME
    if path.is_file() and json.loads(path.read_text(encoding="utf-8")) != report:
        raise TwoPhaseSeedError("existing two-phase training report changed")
    if not path.is_file():
        pilot._atomic_json(path, report)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("prepare", "train-seed1", "train-seed2", "analyze")
    )
    parser.add_argument(
        "--project-root", type=Path, default=Path(__file__).resolve().parent
    )
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cuda")
    parser.add_argument("--max-steps", type=int, default=2_000)
    parser.add_argument("--log-every", type=int, default=5)
    parser.add_argument("--resume-existing", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _parser().parse_args(argv)
    if args.max_steps <= 0 or args.log_every <= 0:
        raise ValueError("max steps and log interval must be positive")
    project_root = args.project_root.resolve()
    output_root = (
        args.output_root.resolve()
        if args.output_root is not None
        else project_root
        / "results/vcg-v1-1-conditioned-handling-seeds12-two-phase-development"
    )
    torch.set_num_threads(1)
    if args.command == "prepare":
        result = prepare(project_root, output_root, max_steps=args.max_steps)
    elif args.command.startswith("train-seed"):
        seed = int(args.command[-1])
        result = train_seed(
            project_root,
            output_root,
            seed=seed,
            device_name=args.device,
            max_steps=args.max_steps,
            log_every=args.log_every,
            resume_existing=args.resume_existing,
        )
    else:
        result = analyze_training(
            project_root, output_root, max_steps=args.max_steps
        )
    if "round_records" in result:
        result = {
            "status": result["status"],
            "model_seed": result["model_seed"],
            "completed_collection_episodes": result[
                "completed_collection_episodes"
            ],
            "completed_probe_episodes": result["completed_probe_episodes"],
            "stability_gate": result["stability_gate"],
            "summary": str(_seed_paths(output_root, result["model_seed"])["summary"]),
        }
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
