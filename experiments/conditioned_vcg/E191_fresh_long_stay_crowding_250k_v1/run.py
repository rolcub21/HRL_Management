#!/usr/bin/env python3
"""Prepare, verify, train, and summarize E191."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from viability_graph_episodic_audit import EpisodicViabilityGraphConfig
from experiments.conditioned_vcg.E187_wait_corrected_e78_246k_v1 import (
    run as e187_run,
    representation as e187_representation,
)
from experiments.conditioned_vcg.E188_mask_only_progress_guard_247k_v1 import (
    representation as e188_representation,
)

from . import instrumentation, protocol as p, representation, scenario, stage
from .roundoff import repaired_audit


class E191Error(RuntimeError):
    pass


BUDGET_RULE = "fixed_300_episodes_fresh_e189_arch_longstay_three_workloads"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise E191Error(message)


@contextmanager
def implementation():
    """Install E191 only at E187's explicit trainer extension seams."""

    with (
        patch.object(e187_run, "p", p),
        patch.object(
            e187_run,
            "WaitCorrectedQueueAgent",
            representation.FreshChoicePreservingQueueAgent,
        ),
        patch.object(e187_run, "instrumentation", instrumentation),
        patch.object(e187_run, "BUDGET_RULE", BUDGET_RULE),
        patch.object(e187_run, "authenticate_sources", lambda: {}),
        patch.object(e187_run.v2run, "train_stage_a", stage.train_stage_a),
    ):
        yield


def prepare(output: Path) -> dict:
    with implementation():
        value = e187_run.prepare(output)
    return {
        **value,
        "protocol": p.PROTOCOL,
        "architecture_identical_to_e189": True,
        "fresh_random_initialization": True,
        "workload_levels": list(scenario.LEVELS),
    }


def authenticate(output: Path) -> dict:
    with implementation():
        return e187_run.authenticate(output)


def _state_equal(first: dict, second: dict) -> bool:
    return set(first) == set(second) and all(
        torch.equal(first[name], second[name]) for name in first
    )


def self_check(output: Path) -> dict:
    prepared = prepare(output)
    contract = authenticate(output)
    parent = p.e189.contract()
    require(
        contract["unchanged_method_fields_sha256"] == p.digest({
            name: parent[name] for name in p.UNCHANGED_METHOD_FIELDS
        }),
        "E191 method fields no longer match E189",
    )
    require(
        representation.NETWORK_ARCHITECTURE
        == e188_representation.NETWORK_ARCHITECTURE
        and representation.QUEUE_CONTEXT_NAMES
        == e188_representation.QUEUE_CONTEXT_NAMES
        and representation.QUEUE_CONTEXT_DIM
        == e188_representation.QUEUE_CONTEXT_DIM == 8,
        "E191 neural input architecture differs from E189",
    )

    config = EpisodicViabilityGraphConfig(
        gamma=p.GAMMA, episode_horizon_steps=p.MAX_STEPS
    )
    fresh = representation.FreshChoicePreservingQueueAgent(
        config=config, seed=0, device="cpu", epsilon=p.EPSILON_START
    )
    with e187_representation.warm_start_disabled():
        reference = e188_representation.ChoicePreservingQueueAgent(
            config=config, seed=0, device="cpu", epsilon=p.EPSILON_START
        )
    require(
        representation.architecture_signature(fresh)
        == representation.architecture_signature(reference),
        "E191 parameter names or shapes differ from E189",
    )
    require(
        fresh.e187_warm_start_manifest is None
        and len(fresh.replay) == 0
        and fresh.decision_count == 0
        and fresh.gradient_steps == 0
        and not fresh.optimizer.state,
        "E191 did not start with completely fresh training state",
    )

    payload = fresh.checkpoint(
        include_replay=True,
        training_protocol=p.PROTOCOL,
        future_schedule_visible_to_policy=False,
    )
    restored = representation.FreshChoicePreservingQueueAgent.from_checkpoint(
        payload, device="cpu", resumable=True, seed=0
    )
    require(
        _state_equal(fresh.Q_local.state_dict(), restored.Q_local.state_dict())
        and restored.e187_warm_start_manifest is None,
        "E191 fresh checkpoint did not round-trip exactly",
    )

    env = p.TargetDenseEnv(p.SCENARIO, p.SPEC)
    cases = []
    by_replicate = {}
    for seed in scenario.VALIDATION_SEEDS:
        instance = env.sample_episode_instance(seed)
        instance.validate_for(env)
        metadata = scenario.workload_metadata(instance)
        level = metadata["target_nominal_occupancy"]
        gap = scenario.gap_for_level(level)
        require(
            instance.number_blocks == scenario.TOTAL_JOBS
            and instance.arrival_steps
            == tuple(gap * index for index in range(scenario.TOTAL_JOBS))
            and abs(metadata["nominal_occupancy_from_declared_mean"] - level)
            < 1.0e-12,
            "E191 validation workload arithmetic differs",
        )
        replicate = seed % 100
        previous = by_replicate.setdefault(
            replicate, tuple(instance.storage_steps_needed)
        )
        require(
            tuple(instance.storage_steps_needed) == previous,
            "E191 paired validation durations differ across workloads",
        )
        env.reset(instance=instance)
        require(
            not any(block.stored or block.delivered for block in env.blocks),
            "E191 validation case does not start empty",
        )
        snapshot = SimpleNamespace(decision_epoch=0, inbound_label="B1")
        context = representation.public_queue_context(
            env,
            snapshot,
            required_jobs=scenario.TOTAL_JOBS,
            max_steps=p.MAX_STEPS,
        )
        cases.append({
            "seed": seed,
            "target": level,
            "gap": gap,
            "first_last_release": [
                instance.arrival_steps[0], instance.arrival_steps[-1]
            ],
            "duration_mean": metadata["realized_duration_mean"],
            "initial_context": list(context.values),
        })
    initial_contexts = {tuple(case["initial_context"]) for case in cases}
    require(
        len(initial_contexts) == 1,
        "E191 leaked the future workload through the initial observation",
    )
    require(
        max(case["first_last_release"][1] for case in cases)
        + 2 * scenario.MEAN_DURATION
        + 100 * scenario.TOTAL_JOBS
        <= p.MAX_STEPS,
        "E191 error horizon lacks its declared conservative margin",
    )

    probe = p.TargetDenseEnv(p.SCENARIO, p.SPEC)
    instance = probe.sample_episode_instance(scenario.PILOT_SEEDS[0])
    probe.reset(instance=instance)
    _state, reward, _done, info = probe.step(probe.ACTION_IDS["WAIT"])
    components = info["timing_objective_components"]["e187_components"]
    require(
        abs(float(reward) + 0.5) < 1.0e-12
        and components["admission_wait_accrual"] == 1.0
        and components["equipment_work"] == 0.0,
        "E191 changed the E189 primitive objective",
    )
    return {
        "status": "passed",
        "contract_sha256": prepared["contract_sha256"],
        "architecture": contract["architecture"],
        "fresh_initialization": contract["initialization"],
        "validation_cases": cases,
        "checkpoint_roundtrip": True,
        "initial_observation_hides_workload": True,
        "primitive_wait_probe_reward": reward,
        "accelerations": contract["acceleration_assertion"],
        "episode_runs": 0,
        "training_runs": 0,
    }


def train_seed(output: Path, seed: int, through_episode: int, device: str) -> dict:
    prepare(output)
    contract = authenticate(output)
    require(seed in p.MODEL_SEEDS, "E191 model seed must be 0, 1, or 2")
    require(
        1 <= through_episode <= p.TOTAL_EPISODES,
        "E191 through-episode must be between 1 and 300",
    )
    require(
        device == "cuda" and torch.cuda.is_available(),
        "E191 training requires CUDA",
    )
    with implementation(), repaired_audit():
        result = e187_run.train_seed(output, seed, through_episode, device)
    result.pop("e187_protocol", None)
    return {
        **result,
        "e191_protocol": p.PROTOCOL,
        "e191_contract_sha256": contract["contract_sha256"],
        "fresh_random_initialization": True,
        "architecture_identical_to_e189": True,
    }


def status(output: Path) -> dict:
    with implementation():
        result = e187_run.status(output)
    return {**result, "protocol": p.PROTOCOL}


def analyze(output: Path) -> dict:
    output = output.resolve()
    contract = authenticate(output)
    seeds = []
    for seed in p.MODEL_SEEDS:
        with implementation():
            path = e187_run._summary_path(output, seed)
        if not path.is_file():
            seeds.append({
                "model_seed": seed,
                "status": "not_started",
                "completed_training_episodes": 0,
            })
            continue
        with implementation():
            summary = e187_run._seed_summary(output, seed)
        completed = int(summary["completed_training_episodes"])
        training_history = list(summary["training"]["runs"])
        rows = []
        for record in summary.get("validation_history", ()):
            with implementation():
                row = e187_run._validation_row(record, training_history)
            row["by_workload"] = record["summary"].get(
                "e191_by_workload", {}
            )
            row["mean_stored"] = record["summary"].get(
                "mean_e191_stored"
            )
            row["mean_external_queue"] = record["summary"].get(
                "mean_e191_external_queue"
            )
            row["mean_transporter_utilization"] = record["summary"].get(
                "mean_e191_transporter_utilization"
            )
            rows.append(row)
        expected = list(range(
            p.VALIDATION_EVERY, completed + 1, p.VALIDATION_EVERY
        ))
        require(
            [row["checkpoint_episode"] for row in rows] == expected,
            f"E191 seed {seed} validation coverage differs",
        )
        seeds.append({
            "model_seed": seed,
            "status": "complete" if completed == p.TOTAL_EPISODES else "partial",
            "completed_training_episodes": completed,
            "training_summary": str(path),
            "validation_history": rows,
            "terminal": rows[-1] if completed == p.TOTAL_EPISODES else None,
        })
    report = {
        "status": (
            "complete"
            if all(item["status"] == "complete" for item in seeds)
            else "partial"
        ),
        "protocol": p.PROTOCOL,
        "contract_sha256": contract["contract_sha256"],
        "architecture": contract["architecture"],
        "initialization": contract["initialization"],
        "workload": contract["workload"],
        "seeds": seeds,
        "test_panel_opened": False,
        "paper_evidence": False,
    }
    path = output / "e191-report.json"
    e187_run._write(path, report)
    return {"status": report["status"], "report": str(path)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("prepare", "self-check", "train-seed", "status", "analyze")
    )
    parser.add_argument("--output", type=Path, default=p.DEFAULT_OUTPUT)
    parser.add_argument("--model-seed", type=int, choices=p.MODEL_SEEDS)
    parser.add_argument("--through-episode", type=int, default=p.TOTAL_EPISODES)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.command == "prepare":
        result = prepare(args.output)
    elif args.command == "self-check":
        result = self_check(args.output)
    elif args.command == "train-seed":
        require(args.model_seed is not None, "train-seed requires --model-seed")
        result = train_seed(
            args.output, args.model_seed, args.through_episode, args.device
        )
    elif args.command == "status":
        result = status(args.output)
    else:
        result = analyze(args.output)
    print(json.dumps(result, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
