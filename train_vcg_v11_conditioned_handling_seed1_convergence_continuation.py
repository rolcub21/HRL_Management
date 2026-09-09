#!/usr/bin/env python3
"""Convergence-controlled continuation for the unresolved conditioned VCG seed 1.

The continuation starts from the immutable two-phase round-8 terminal.  It
uses the existing rho=.25 fitted-policy damping rule and stops at the first
predeclared point with two consecutive stable training-probe transitions, or
after four additional rounds.  A common fixed bank is decomposed before and
after every update.  No evaluation panel or checkpoint selection is used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
from typing import Mapping, Optional, Sequence

import torch

import benchmark_viability_critic_priority as benchmark
import diagnose_vcg_v11_conditioned_handling_fixed_bank as fixed_bank
import run_vcg_v11_conditioned_handling_two_phase_seeds12 as parent
import run_vcg_v11_nested_handling_pilot as pilot
import train_vcg_preference_conditioned as common
import train_vcg_v11_conditioned_handling_convergence_extension as stability_rule
import train_vcg_v11_conditioned_handling_damped_convergence as damped
import train_vcg_v11_conditioned_handling_iterative as full
from vcg_v11_conditioned_handling import (
    fit_conditioned_future_handling,
    handling_calibration,
)


PROTOCOL = "vcg_v1_1_conditioned_handling_seed1_convergence_continuation_v1"
SCHEMA_VERSION = 1
MODEL_SEED = 1
PARENT_GLOBAL_ROUNDS = parent.TOTAL_ROUNDS
MAX_ADDITIONAL_ROUNDS = 4
REQUIRED_CONSECUTIVE_STABLE_TRANSITIONS = 2
EPISODES_PER_ROUND = full.EPISODES_PER_ROUND
POLYAK_RHO = damped.POLYAK_RHO
PARENT_RELATIVE = parent.SEED0_DAMPED_ROOT_RELATIVE.parent / (
    "vcg-v1-1-conditioned-handling-seeds12-two-phase-development"
)
OUTPUT_RELATIVE = Path(
    "results/vcg-v1-1-conditioned-handling-seed1-convergence-continuation"
)
CONTRACT_NAME = "seed1-continuation-contract.json"
BANK_NAME = "fixed-bank.pth"
LATEST_NAME = "latest.pth"
TERMINAL_NAME = "terminal.pth"
SUMMARY_NAME = "seed1-continuation-summary.json"
LATEST_ROLE = "resumable_convergence_controlled_round_boundary"
TERMINAL_ROLE = "convergence_controlled_stopping_terminal"


class Seed1ContinuationError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    if not path.is_file():
        raise Seed1ContinuationError(f"missing required artifact: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_form(value):
    return json.loads(
        json.dumps(
            common._json_safe(value), sort_keys=True, separators=(",", ":")
        )
    )


def _canonical_hash(value: Mapping) -> str:
    return hashlib.sha256(
        json.dumps(
            _json_form(dict(value)), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()


def continuation_schedule(round_index: int) -> tuple[dict, ...]:
    if round_index not in range(MAX_ADDITIONAL_ROUNDS):
        raise ValueError("continuation round index is out of range")
    global_round_index = PARENT_GLOBAL_ROUNDS + round_index
    rng_seed = full.LAMBDA_SCHEDULE_SEED + global_round_index
    rng = random.Random(rng_seed)
    values = [
        0.2 * (index + rng.random()) / EPISODES_PER_ROUND
        for index in range(EPISODES_PER_ROUND)
    ]
    rng.shuffle(values)
    return tuple(
        {
            "continuation_episode_number": (
                round_index * EPISODES_PER_ROUND + position + 1
            ),
            "global_episode_number": (
                global_round_index * EPISODES_PER_ROUND + position + 1
            ),
            "continuation_round_number": round_index + 1,
            "global_round_number": global_round_index + 1,
            "position_in_round": position + 1,
            "behavior_lambda": float(value),
            "schedule_rng_seed": rng_seed,
            "stratified_uniform_over_[0,0.2]": True,
            "fixed_for_complete_episode": True,
        }
        for position, value in enumerate(values)
    )


def continuation_instance_seed(round_index: int, position: int) -> int:
    if round_index not in range(MAX_ADDITIONAL_ROUNDS):
        raise ValueError("continuation round index is out of range")
    if position not in range(EPISODES_PER_ROUND):
        raise ValueError("continuation episode position is out of range")
    global_round_index = PARENT_GLOBAL_ROUNDS + round_index
    return (
        full.TRAIN_SEED_BASE
        + global_round_index * full.ROUND_SEED_STRIDE
        + position
    )


def stable_streak(records: Sequence[Mapping]) -> int:
    count = 0
    for record in reversed(tuple(records)):
        stability = record.get("stability_against_previous_round")
        if not isinstance(stability, Mapping) or stability.get("stable") is not True:
            break
        count += 1
    return count


def stopping_assessment(records: Sequence[Mapping]) -> dict:
    streak = stable_streak(records)
    reached_cap = len(records) >= MAX_ADDITIONAL_ROUNDS
    passed = streak >= REQUIRED_CONSECUTIVE_STABLE_TRANSITIONS
    return {
        "completed_additional_rounds": len(records),
        "maximum_additional_rounds": MAX_ADDITIONAL_ROUNDS,
        "required_consecutive_stable_transitions": (
            REQUIRED_CONSECUTIVE_STABLE_TRANSITIONS
        ),
        "current_consecutive_stable_transitions": streak,
        "passed": passed,
        "reached_round_cap": reached_cap,
        "stop": passed or reached_cap,
        "decision": (
            "authorize_matched_frozen_evaluation"
            if passed
            else (
                "do_not_evaluate_continuation_cap_reached"
                if reached_cap
                else "continue_training_only"
            )
        ),
    }


def _parent_artifacts(project_root: Path, *, max_steps: int) -> dict:
    output_root = project_root / PARENT_RELATIVE
    contract = parent._require_contract(
        project_root, output_root, max_steps=max_steps
    )
    inputs = parent._authenticate_inputs(project_root)
    paths = parent._seed_paths(output_root, MODEL_SEED)
    summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
    terminal = torch.load(
        paths["terminal"], map_location="cpu", weights_only=False
    )
    parent._validate_checkpoint(
        terminal, contract, seed=MODEL_SEED, terminal=True
    )
    if terminal.get("stability_gate") != summary.get("stability_gate"):
        raise Seed1ContinuationError("parent seed-1 stability record changed")
    if summary.get("status") != "complete":
        raise Seed1ContinuationError("parent seed-1 training is incomplete")
    if summary.get("stability_gate", {}).get("passed") is not False:
        raise Seed1ContinuationError(
            "continuation is only defined for the unresolved seed-1 terminal"
        )
    records = summary.get("round_records")
    if not isinstance(records, list) or len(records) != PARENT_GLOBAL_ROUNDS:
        raise Seed1ContinuationError("parent seed-1 rounds are incomplete")
    previous_probe = records[-1].get("probe_rows")
    if not isinstance(previous_probe, list) or len(previous_probe) != len(
        full.PROBE_LAMBDAS
    ):
        raise Seed1ContinuationError("parent seed-1 probe bank is incomplete")
    report_path = output_root / parent.REPORT_NAME
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("decision") != "do_not_open_evaluation_inspect_unstable_seed":
        raise Seed1ContinuationError("parent report did not retain evaluation closure")
    return {
        "root": output_root,
        "contract": contract,
        "inputs": inputs,
        "terminal": terminal,
        "terminal_path": paths["terminal"],
        "terminal_sha256": _sha256(paths["terminal"]),
        "summary": summary,
        "summary_path": paths["summary"],
        "summary_sha256": _sha256(paths["summary"]),
        "report_path": report_path,
        "report_sha256": _sha256(report_path),
        "previous_probe": previous_probe,
    }


def _contract(project_root: Path, output_dir: Path, *, max_steps: int) -> dict:
    artifacts = _parent_artifacts(project_root, max_steps=max_steps)
    semantic = {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "scope": "unresolved_seed_training_only_convergence_control",
        "model_seed": MODEL_SEED,
        "parent_protocol": parent.PROTOCOL,
        "parent_contract_sha256": artifacts["contract"]["contract_sha256"],
        "parent_terminal_sha256": artifacts["terminal_sha256"],
        "parent_training_summary_sha256": artifacts["summary_sha256"],
        "parent_report_sha256": artifacts["report_sha256"],
        "parent_global_rounds": PARENT_GLOBAL_ROUNDS,
        "maximum_additional_rounds": MAX_ADDITIONAL_ROUNDS,
        "minimum_additional_rounds": REQUIRED_CONSECUTIVE_STABLE_TRANSITIONS,
        "episodes_per_round": EPISODES_PER_ROUND,
        "maximum_collection_episodes": (
            MAX_ADDITIONAL_ROUNDS * EPISODES_PER_ROUND
        ),
        "probe_episodes_per_round": len(full.PROBE_LAMBDAS),
        "maximum_probe_episodes": (
            MAX_ADDITIONAL_ROUNDS * len(full.PROBE_LAMBDAS)
        ),
        "stopping_rule": {
            "first_two_consecutive_stable_training_probe_transitions": True,
            "hard_maximum_additional_rounds": MAX_ADDITIONAL_ROUNDS,
            "performance_metrics_not_used": True,
            "evaluation_metrics_not_used": True,
        },
        "stability_rule": {
            "maximum_changed_probe_behaviors": (
                stability_rule.MAX_CHANGED_PROBES_PER_TRANSITION
            ),
            "aggregate_probe_rehandles_unchanged": True,
            "all_probe_episodes_strict_safe_complete": True,
            "maximum_validation_mae": stability_rule.MAX_VALIDATION_MAE,
            "maximum_absolute_validation_bias": (
                stability_rule.MAX_ABSOLUTE_VALIDATION_BIAS
            ),
        },
        "polyak_parameter_blend_rho": POLYAK_RHO,
        "polyak_rule": "theta_next=(1-rho)*theta_previous+rho*theta_provisional",
        "lambda_schedules": [
            continuation_schedule(index)
            for index in range(MAX_ADDITIONAL_ROUNDS)
        ],
        "instance_seed_formula": (
            "58000000 + global_zero_based_round*10000 + position"
        ),
        "max_steps": int(max_steps),
        "fit_epochs": full.FIT_EPOCHS,
        "fit_batch_size": full.FIT_BATCH_SIZE,
        "fit_learning_rate": full.FIT_LEARNING_RATE,
        "whole_episode_validation_split": True,
        "validation_episodes_per_round": full.VALIDATION_EPISODES,
        "policy_frozen_within_collection_round": True,
        "current_round_only_mc_fitting": True,
        "operational_critic_frozen": True,
        "evaluation_panels_opened": False,
        "checkpoint_selection": False,
        "fixed_bank_diagnostic": {
            "source": "predeclared_training_probe_states",
            "lambda_grid": list(fixed_bank.LAMBDA_GRID),
            "maximum_frontiers_per_probe": fixed_bank.MAX_FRONTIERS_PER_PROBE,
            "measured_at_parent_and_after_every_continuation_round": True,
            "used_for_stopping_or_fitting": False,
        },
        "source_sha256": {
            "continuation_trainer": _sha256(Path(__file__).resolve()),
            "conditioned_controller": _sha256(
                project_root / "vcg_v11_conditioned_handling.py"
            ),
            "parent_runner": _sha256(
                project_root
                / "run_vcg_v11_conditioned_handling_two_phase_seeds12.py"
            ),
            "damped_rule": _sha256(
                project_root
                / "train_vcg_v11_conditioned_handling_damped_convergence.py"
            ),
            "stability_rule": _sha256(
                project_root
                / "train_vcg_v11_conditioned_handling_convergence_extension.py"
            ),
            "fixed_bank_diagnostic": _sha256(
                project_root
                / "diagnose_vcg_v11_conditioned_handling_fixed_bank.py"
            ),
        },
        "output_dir": str(output_dir.resolve()),
    }
    return {**semantic, "contract_sha256": _canonical_hash(semantic)}


def prepare(project_root: Path, output_dir: Path, *, max_steps: int) -> dict:
    expected = _json_form(_contract(project_root, output_dir, max_steps=max_steps))
    path = output_dir / CONTRACT_NAME
    if path.is_file():
        if json.loads(path.read_text(encoding="utf-8")) != expected:
            raise Seed1ContinuationError(
                "continuation contract, parent artifacts, or sources changed"
            )
    else:
        if output_dir.exists() and any(output_dir.iterdir()):
            raise Seed1ContinuationError(
                "nonempty continuation output has no authenticated contract"
            )
        pilot._atomic_json(path, expected)
    return {
        "status": "prepared",
        "model_seed": MODEL_SEED,
        "minimum_additional_rounds": REQUIRED_CONSECUTIVE_STABLE_TRANSITIONS,
        "maximum_additional_rounds": MAX_ADDITIONAL_ROUNDS,
        "simulator_episodes_per_round": (
            EPISODES_PER_ROUND + len(full.PROBE_LAMBDAS)
        ),
        "fixed_bank_probe_rollouts": len(full.PROBE_LAMBDAS),
        "contract": str(path.resolve()),
    }


def _require_contract(project_root: Path, output_dir: Path, *, max_steps: int):
    path = output_dir / CONTRACT_NAME
    if not path.is_file():
        raise Seed1ContinuationError("run prepare first")
    observed = json.loads(path.read_text(encoding="utf-8"))
    expected = _json_form(_contract(project_root, output_dir, max_steps=max_steps))
    if observed != expected:
        raise Seed1ContinuationError(
            "continuation contract, parent artifacts, or sources changed"
        )
    return observed


def _bank_semantic(bank) -> dict:
    return {
        "state_count": len(bank),
        "candidate_count": sum(len(item["records"]) for item in bank),
        "states": [
            {
                "state_id": item["state_id"],
                "frontier_digest": item["digest"],
                "candidate_count": len(item["records"]),
                "sources": item["sources"],
            }
            for item in bank
        ],
    }


def _load_or_build_bank(
    project_root: Path,
    output_dir: Path,
    artifacts,
    *,
    device: torch.device,
    max_steps: int,
):
    path = output_dir / BANK_NAME
    if path.is_file():
        payload = torch.load(path, map_location="cpu", weights_only=False)
        expected = {
            "protocol": PROTOCOL,
            "parent_terminal_sha256": artifacts["terminal_sha256"],
            "lambda_grid": fixed_bank.LAMBDA_GRID,
            "maximum_frontiers_per_probe": fixed_bank.MAX_FRONTIERS_PER_PROBE,
        }
        mismatch = {
            key: (payload.get(key), value)
            for key, value in expected.items()
            if payload.get(key) != value
        }
        if mismatch:
            raise Seed1ContinuationError(f"fixed bank mismatch: {mismatch}")
        if payload.get("semantic") != _bank_semantic(payload["bank"]):
            raise Seed1ContinuationError("fixed bank semantic identity changed")
        return payload["bank"], payload["probe_runs"], _sha256(path)
    diagnostic_auth = fixed_bank._authenticate(project_root, max_steps=max_steps)
    if (
        diagnostic_auth["terminals"][MODEL_SEED]
        ["agent_checkpoint"]["base_checkpoint_sha256"]
        != artifacts["terminal"]["agent_checkpoint"]["base_checkpoint_sha256"]
    ):
        raise Seed1ContinuationError("fixed-bank and parent base identities differ")
    bank, probe_runs = fixed_bank._collect_bank(
        project_root, diagnostic_auth, device=device
    )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "parent_terminal_sha256": artifacts["terminal_sha256"],
        "lambda_grid": fixed_bank.LAMBDA_GRID,
        "maximum_frontiers_per_probe": fixed_bank.MAX_FRONTIERS_PER_PROBE,
        "semantic": _bank_semantic(bank),
        "probe_runs": probe_runs,
        "bank": bank,
    }
    common._atomic_torch_save(payload, path)
    return bank, probe_runs, _sha256(path)


def _diagnose_checkpoint(agent, bank, output_dir: Path, *, label: str) -> dict:
    directory = output_dir / "fixed-bank-diagnostics"
    directory.mkdir(parents=True, exist_ok=True)
    values_path = directory / f"{label}-values.csv"
    selections_path = directory / f"{label}-selections.csv"
    values, selections, sequences, winners = fixed_bank._decompose(agent, bank)
    tagged_values = [{"checkpoint": label, **row} for row in values]
    tagged_selections = [{"checkpoint": label, **row} for row in selections]
    fixed_bank._atomic_csv(values_path, tuple(tagged_values[0]), tagged_values)
    fixed_bank._atomic_csv(
        selections_path, tuple(tagged_selections[0]), tagged_selections
    )
    return {
        "label": label,
        "summary": fixed_bank.summarize_sequences(sequences, winners),
        "values_csv": str(values_path.resolve()),
        "values_sha256": _sha256(values_path),
        "selections_csv": str(selections_path.resolve()),
        "selections_sha256": _sha256(selections_path),
    }


def _checkpoint(
    agent,
    *,
    artifacts,
    contract,
    bank_sha256: str,
    completed_rounds: int,
    records,
    role: str,
    stopping=None,
):
    seed_input = artifacts["inputs"]["seed_inputs"][MODEL_SEED]
    payload = {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "checkpoint_role": role,
        "model_seed": MODEL_SEED,
        "completed_additional_rounds": int(completed_rounds),
        "global_completed_rounds": PARENT_GLOBAL_ROUNDS + int(completed_rounds),
        "polyak_rho": POLYAK_RHO,
        "parent_terminal_sha256": artifacts["terminal_sha256"],
        "contract_sha256": contract["contract_sha256"],
        "fixed_bank_sha256": bank_sha256,
        "agent_checkpoint": agent.checkpoint(
            base_checkpoint_sha256=seed_input["arm"].checkpoint_sha256,
            base_policy_digest=seed_input["arm"].deployment_policy_digest,
            source_cost_sha256=seed_input["cost_sha256"],
        ),
        "round_records": tuple(records),
        "fixed_terminal_checkpoint": role == TERMINAL_ROLE,
        "checkpoint_selection_used": False,
        "evaluation_panels_opened": False,
    }
    if stopping is not None:
        payload["stopping_assessment"] = dict(stopping)
    return payload


def _validate_checkpoint(payload, contract, *, artifacts, bank_sha256, terminal):
    expected = {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "checkpoint_role": TERMINAL_ROLE if terminal else LATEST_ROLE,
        "model_seed": MODEL_SEED,
        "polyak_rho": POLYAK_RHO,
        "parent_terminal_sha256": artifacts["terminal_sha256"],
        "contract_sha256": contract["contract_sha256"],
        "fixed_bank_sha256": bank_sha256,
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
        raise Seed1ContinuationError(f"continuation checkpoint mismatch: {mismatch}")
    completed = int(payload.get("completed_additional_rounds", -1))
    records = tuple(payload.get("round_records", ()))
    if completed != len(records) or completed not in range(
        1, MAX_ADDITIONAL_ROUNDS + 1
    ):
        raise Seed1ContinuationError("continuation checkpoint clock changed")
    if terminal:
        expected_stopping = stopping_assessment(records)
        if not expected_stopping["stop"]:
            raise Seed1ContinuationError("terminal precedes the stopping rule")
        if payload.get("stopping_assessment") != expected_stopping:
            raise Seed1ContinuationError("terminal stopping assessment changed")


def _load_agent(payload, artifacts, *, device):
    return parent._load_agent(
        payload,
        artifacts["inputs"],
        seed=MODEL_SEED,
        device=device,
    )


def train(
    project_root: Path,
    output_dir: Path,
    *,
    device_name: str,
    max_steps: int,
    log_every: int,
):
    contract = _require_contract(project_root, output_dir, max_steps=max_steps)
    artifacts = _parent_artifacts(project_root, max_steps=max_steps)
    device = full.resolve_device(device_name)
    bank, probe_runs, bank_sha = _load_or_build_bank(
        project_root,
        output_dir,
        artifacts,
        device=device,
        max_steps=max_steps,
    )
    latest_path = output_dir / LATEST_NAME
    terminal_path = output_dir / TERMINAL_NAME
    summary_path = output_dir / SUMMARY_NAME
    if terminal_path.is_file():
        terminal = torch.load(
            terminal_path, map_location="cpu", weights_only=False
        )
        _validate_checkpoint(
            terminal,
            contract,
            artifacts=artifacts,
            bank_sha256=bank_sha,
            terminal=True,
        )
        if not summary_path.is_file():
            raise Seed1ContinuationError("terminal exists without its summary")
        return json.loads(summary_path.read_text(encoding="utf-8"))
    if latest_path.is_file():
        latest = torch.load(latest_path, map_location="cpu", weights_only=False)
        _validate_checkpoint(
            latest,
            contract,
            artifacts=artifacts,
            bank_sha256=bank_sha,
            terminal=False,
        )
        agent = _load_agent(latest, artifacts, device=device)
        records = list(latest["round_records"])
        previous_probe = records[-1]["probe_rows"]
        print(
            f"Resumed seed 1 after {len(records)} continuation rounds",
            flush=True,
        )
    else:
        if summary_path.exists():
            raise Seed1ContinuationError("summary exists without a checkpoint")
        agent = _load_agent(artifacts["terminal"], artifacts, device=device)
        records = []
        previous_probe = artifacts["previous_probe"]
    agent.set_epsilon(0.0)
    parent_diagnostic = _diagnose_checkpoint(
        _load_agent(artifacts["terminal"], artifacts, device=device),
        bank,
        output_dir,
        label="parent-global-round-8",
    )
    seed_input = artifacts["inputs"]["seed_inputs"][MODEL_SEED]
    arm = seed_input["arm"]
    search = benchmark._search_config(arm.payload)
    liveness = benchmark._liveness_rule(arm.payload)
    env = benchmark._make_env(arm.payload)
    print(
        "Seed-1 convergence continuation | "
        f"device={device} | rho={POLYAK_RHO} | "
        f"stop=first {REQUIRED_CONSECUTIVE_STABLE_TRANSITIONS} stable transitions | "
        f"cap={MAX_ADDITIONAL_ROUNDS} rounds | evaluation=false",
        flush=True,
    )
    while not stopping_assessment(records)["stop"]:
        round_index = len(records)
        global_round_index = PARENT_GLOBAL_ROUNDS + round_index
        schedule = continuation_schedule(round_index)
        before_state = damped._clone_state(agent.handling_network)
        before_digest = full._state_digest(before_state)
        episode_samples = []
        rows = []
        for position, schedule_record in enumerate(schedule):
            instance_seed = continuation_instance_seed(round_index, position)
            env.current_episode = int(schedule_record["global_episode_number"])
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
                    f"Continuation round {round_index + 1}/{MAX_ADDITIONAL_ROUNDS} "
                    f"episode {position + 1:2d}/{EPISODES_PER_ROUND} | "
                    f"lambda={schedule_record['behavior_lambda']:.4f} | "
                    f"rehandles={run['physical_rehandles']}",
                    flush=True,
                )
        training, validation, validation_indices = full._split_episode_samples(
            episode_samples, round_index=global_round_index
        )
        fit = fit_conditioned_future_handling(
            agent.handling_network,
            training,
            validation,
            device=device,
            epochs=full.FIT_EPOCHS,
            batch_size=full.FIT_BATCH_SIZE,
            learning_rate=full.FIT_LEARNING_RATE,
            seed=full.FIT_SEED + global_round_index,
        )
        provisional_digest = full._state_digest(
            agent.handling_network.state_dict()
        )
        provisional_validation = fit["final_validation"]
        damped.polyak_blend_network(agent.handling_network, before_state)
        deployed_digest = full._state_digest(agent.handling_network.state_dict())
        if deployed_digest in (before_digest, provisional_digest):
            raise Seed1ContinuationError(
                "damped deployment must differ from previous and provisional states"
            )
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
        stability = stability_rule._stability(
            previous_probe,
            probe_rows,
            {"final_validation": deployed_validation},
        )
        diagnostic = _diagnose_checkpoint(
            agent,
            bank,
            output_dir,
            label=f"continuation-round-{round_index + 1}",
        )
        record = {
            "continuation_round_number": round_index + 1,
            "global_round_number": global_round_index + 1,
            "policy_state_before_sha256": before_digest,
            "provisional_policy_state_sha256": provisional_digest,
            "deployed_policy_state_sha256": deployed_digest,
            "polyak_rho": POLYAK_RHO,
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
            "fixed_bank_diagnostic": diagnostic,
        }
        records.append(record)
        previous_probe = probe_rows
        latest = _checkpoint(
            agent,
            artifacts=artifacts,
            contract=contract,
            bank_sha256=bank_sha,
            completed_rounds=len(records),
            records=records,
            role=LATEST_ROLE,
        )
        common._atomic_torch_save(latest, latest_path)
        print(
            f"Continuation round {len(records)} | "
            f"changed_probes={stability['changed_probe_behavior_count']} | "
            f"probe_rehandles={stability['current_aggregate_probe_rehandles']} | "
            f"val_MAE={stability['final_validation_mae']:.4f} | "
            f"val_bias={stability['final_validation_bias']:.4f} | "
            f"stable={stability['stable']} | "
            f"stable_streak={stable_streak(records)}",
            flush=True,
        )
    stopping = stopping_assessment(records)
    terminal = _checkpoint(
        agent,
        artifacts=artifacts,
        contract=contract,
        bank_sha256=bank_sha,
        completed_rounds=len(records),
        records=records,
        role=TERMINAL_ROLE,
        stopping=stopping,
    )
    common._atomic_torch_save(terminal, terminal_path)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "status": "complete",
        "model_seed": MODEL_SEED,
        "contract_sha256": contract["contract_sha256"],
        "parent_terminal_sha256": artifacts["terminal_sha256"],
        "fixed_bank_sha256": bank_sha,
        "fixed_bank_probe_runs": probe_runs,
        "parent_fixed_bank_diagnostic": parent_diagnostic,
        "completed_additional_rounds": len(records),
        "completed_collection_episodes": len(records) * EPISODES_PER_ROUND,
        "completed_probe_episodes": len(records) * len(full.PROBE_LAMBDAS),
        "round_records": records,
        "stopping_assessment": stopping,
        "latest_checkpoint": str(latest_path.resolve()),
        "terminal_checkpoint": str(terminal_path.resolve()),
        "terminal_checkpoint_sha256": _sha256(terminal_path),
        "evaluation_panels_opened": False,
        "checkpoint_selection_used": False,
    }
    common._atomic_json_save(summary, summary_path)
    return summary


def analyze(project_root: Path, output_dir: Path, *, max_steps: int) -> dict:
    contract = _require_contract(project_root, output_dir, max_steps=max_steps)
    artifacts = _parent_artifacts(project_root, max_steps=max_steps)
    bank_sha = _sha256(output_dir / BANK_NAME)
    terminal_path = output_dir / TERMINAL_NAME
    summary_path = output_dir / SUMMARY_NAME
    terminal = torch.load(
        terminal_path, map_location="cpu", weights_only=False
    )
    _validate_checkpoint(
        terminal,
        contract,
        artifacts=artifacts,
        bank_sha256=bank_sha,
        terminal=True,
    )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("terminal_checkpoint_sha256") != _sha256(terminal_path):
        raise Seed1ContinuationError("terminal and summary identities differ")
    return {
        "status": "passed" if terminal["stopping_assessment"]["passed"] else "failed",
        "protocol": PROTOCOL,
        "model_seed": MODEL_SEED,
        "completed_additional_rounds": terminal["completed_additional_rounds"],
        "stopping_assessment": terminal["stopping_assessment"],
        "evaluation_panels_opened": False,
        "checkpoint_selection_used": False,
        "summary": str(summary_path.resolve()),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run", "analyze"))
    parser.add_argument(
        "--project-root", type=Path, default=Path(__file__).resolve().parent
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cuda")
    parser.add_argument("--max-steps", type=int, default=2_000)
    parser.add_argument("--log-every", type=int, default=5)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    if args.max_steps <= 0 or args.log_every <= 0:
        raise ValueError("max steps and log interval must be positive")
    torch.set_num_threads(1)
    project_root = args.project_root.resolve()
    output_dir = (
        project_root / OUTPUT_RELATIVE
        if args.output_dir is None
        else args.output_dir.resolve()
    )
    if args.command == "prepare":
        result = prepare(project_root, output_dir, max_steps=args.max_steps)
    elif args.command == "run":
        result = train(
            project_root,
            output_dir,
            device_name=args.device,
            max_steps=args.max_steps,
            log_every=args.log_every,
        )
    else:
        result = analyze(project_root, output_dir, max_steps=args.max_steps)
    compact = result
    if args.command == "run":
        compact = {
            key: result[key]
            for key in (
                "status",
                "model_seed",
                "completed_additional_rounds",
                "completed_collection_episodes",
                "completed_probe_episodes",
                "stopping_assessment",
                "terminal_checkpoint",
                "terminal_checkpoint_sha256",
                "evaluation_panels_opened",
                "checkpoint_selection_used",
            )
        }
    print(json.dumps(compact, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
