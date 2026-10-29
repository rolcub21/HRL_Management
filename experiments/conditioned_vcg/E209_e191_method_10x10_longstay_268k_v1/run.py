#!/usr/bin/env python3
"""Prepare, verify, train, and summarize scale-matched 10x10 E191 Stage A."""

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
)
from experiments.conditioned_vcg.E191_fresh_long_stay_crowding_250k_v1 import (
    protocol as e191_protocol,
    representation as e191_representation,
)
from experiments.conditioned_vcg.E191_fresh_long_stay_crowding_250k_v1.roundoff import (
    repaired_audit,
)
from projects.temporal_consequence_realization.method_conformance import (
    render_validated_conformance_audit,
    validate_protocol,
)

from . import instrumentation, protocol as p, representation, scenario, stage


class E209Error(RuntimeError):
    pass


BUDGET_RULE = "fixed_300_episodes_fresh_e191_method_scale_matched_10x10"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise E209Error(message)


def _declaration(contract: dict) -> dict:
    return {
        key: contract[key]
        for key in (
            "method_reference", "method_revision", "claim_level",
            "declared_deviations", "oracle_inputs", "nonconforming_shortcuts",
            "conformance_audit",
        )
    }


@contextmanager
def implementation():
    """Install E209 only at E187's explicit extension seams."""

    with (
        patch.object(e187_run, "p", p),
        patch.object(
            e187_run,
            "WaitCorrectedQueueAgent",
            representation.FreshScaleMatched10x10Agent,
        ),
        patch.object(e187_run, "instrumentation", instrumentation),
        patch.object(e187_run, "BUDGET_RULE", BUDGET_RULE),
        patch.object(e187_run, "authenticate_sources", lambda: {}),
        patch.object(e187_run.v2run, "train_stage_a", stage.train_stage_a),
    ):
        yield


def prepare(output: Path) -> dict:
    contract = p.contract()
    validate_protocol(contract, source=p.PROTOCOL)
    with implementation():
        value = e187_run.prepare(output)
    return {
        **value,
        "protocol": p.PROTOCOL,
        "architecture_identical_to_e191": True,
        "controller_learning_rule_identical_to_e191": True,
        "exploration_schedule_job_budget_normalized": True,
        "fresh_random_initialization": True,
        "scale_matched_10x10": True,
        "workload_levels": list(scenario.LEVELS),
        "initial_remote_target_episode": 120,
    }


def authenticate(output: Path) -> dict:
    with implementation():
        contract = e187_run.authenticate(output)
    validate_protocol(contract, source=p.PROTOCOL)
    return contract


def _state_equal(first: dict, second: dict) -> bool:
    return set(first) == set(second) and all(
        torch.equal(first[name], second[name]) for name in first
    )


def self_check(output: Path) -> dict:
    prepared = prepare(output)
    contract = authenticate(output)
    parent = e191_protocol.contract()
    require(
        contract["unchanged_method_fields_sha256"] == p.digest({
            name: parent[name] for name in p.UNCHANGED_METHOD_FIELDS
        }),
        "E209 method fields no longer match E191",
    )
    require(
        representation.NETWORK_ARCHITECTURE
        == e191_representation.NETWORK_ARCHITECTURE
        and representation.QUEUE_CONTEXT_NAMES
        == e191_representation.QUEUE_CONTEXT_NAMES
        and representation.QUEUE_CONTEXT_DIM
        == e191_representation.QUEUE_CONTEXT_DIM == 8,
        "E209 neural input architecture differs from E191",
    )

    config = EpisodicViabilityGraphConfig(
        gamma=p.GAMMA, episode_horizon_steps=p.MAX_STEPS
    )
    fresh = representation.FreshScaleMatched10x10Agent(
        config=config, seed=0, device="cpu", epsilon=p.EPSILON_START
    )
    reference = e191_representation.FreshChoicePreservingQueueAgent(
        config=config, seed=0, device="cpu", epsilon=p.EPSILON_START
    )
    require(
        representation.architecture_signature(fresh)
        == e191_representation.architecture_signature(reference),
        "E209 parameter names or shapes differ from E191",
    )
    require(
        fresh.e187_warm_start_manifest is None
        and len(fresh.replay) == 0
        and fresh.decision_count == 0
        and fresh.gradient_steps == 0
        and not fresh.optimizer.state,
        "E209 did not start with completely fresh training state",
    )

    payload = fresh.checkpoint(
        include_replay=True,
        training_protocol=p.PROTOCOL,
        future_schedule_visible_to_policy=False,
    )
    restored = representation.FreshScaleMatched10x10Agent.from_checkpoint(
        payload, device="cpu", resumable=True, seed=0
    )
    require(
        _state_equal(fresh.Q_local.state_dict(), restored.Q_local.state_dict())
        and restored.e187_warm_start_manifest is None,
        "E209 fresh checkpoint did not round-trip exactly",
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
            "E209 validation workload arithmetic differs",
        )
        replicate = seed % 100
        previous = by_replicate.setdefault(
            replicate, tuple(instance.storage_steps_needed)
        )
        require(
            tuple(instance.storage_steps_needed) == previous,
            "E209 paired validation durations differ across workloads",
        )
        env.reset(instance=instance)
        require(
            not any(block.stored or block.delivered for block in env.blocks),
            "E209 validation case does not start empty",
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
    require(
        len({tuple(case["initial_context"]) for case in cases}) == 1,
        "E209 leaked the future workload through the initial observation",
    )
    require(
        max(case["first_last_release"][1] for case in cases)
        + 2 * scenario.MEAN_DURATION
        + 100 * scenario.TOTAL_JOBS
        <= p.MAX_STEPS,
        "E209 error horizon lacks its declared conservative margin",
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
        "E209 changed E191's primitive objective",
    )
    return {
        "status": "passed",
        "contract_sha256": prepared["contract_sha256"],
        "architecture": contract["architecture"],
        "fresh_initialization": contract["initialization"],
        "scale_match_to_e191": contract["scale_match_to_e191"],
        "validation_cases": cases,
        "checkpoint_roundtrip": True,
        "initial_observation_hides_workload": True,
        "primitive_wait_probe_reward": reward,
        "accelerations": contract["acceleration_assertion"],
        "episode_runs": 0,
        "training_runs": 0,
        "test_panel_opened": False,
    }


def train_seed(output: Path, seed: int, through_episode: int, device: str) -> dict:
    prepare(output)
    contract = authenticate(output)
    require(seed in p.MODEL_SEEDS, "E209 model seed must be 0, 1, or 2")
    require(
        1 <= through_episode <= p.TOTAL_EPISODES,
        "E209 through-episode must be between 1 and 300",
    )
    require(
        device == "cuda" and torch.cuda.is_available(),
        "E209 training requires CUDA",
    )
    with implementation(), repaired_audit():
        result = e187_run.train_seed(output, seed, through_episode, device)
    result.pop("e187_protocol", None)
    return {
        **result,
        "e209_protocol": p.PROTOCOL,
        "e209_contract_sha256": contract["contract_sha256"],
        "fresh_random_initialization": True,
        "architecture_identical_to_e191": True,
        "controller_learning_rule_identical_to_e191": True,
        "exploration_schedule_job_budget_normalized": True,
        "scale_matched_10x10": True,
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
            row["by_workload"] = record["summary"].get("e209_by_workload", {})
            row["mean_stored"] = record["summary"].get("mean_e209_stored")
            row["mean_external_queue"] = record["summary"].get(
                "mean_e209_external_queue"
            )
            row["mean_transporter_utilization"] = record["summary"].get(
                "mean_e209_transporter_utilization"
            )
            rows.append(row)
        expected = list(range(
            p.VALIDATION_EVERY, completed + 1, p.VALIDATION_EVERY
        ))
        require(
            [row["checkpoint_episode"] for row in rows] == expected,
            f"E209 seed {seed} validation coverage differs",
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
        "scale_match_to_e191": contract["scale_match_to_e191"],
        "seeds": seeds,
        "test_panel_opened": False,
        "paper_evidence": False,
    }
    path = output / "e209-report.json"
    e187_run._write(path, report)
    audit = render_validated_conformance_audit(_declaration(contract))
    (output / "REPORT.txt").write_text(
        audit
        + "\nRESULTS\n"
        + "=" * 72
        + "\n\n"
        + f"Status: {report['status']}\n"
        + "This is development Stage-A evidence; the sealed test panel remains closed.\n",
        encoding="utf-8",
    )
    return {
        "status": report["status"],
        "report": str(path),
        "text_report": str(output / "REPORT.txt"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("prepare", "self-check", "train-seed", "status", "analyze")
    )
    parser.add_argument("--output", type=Path, default=p.DEFAULT_OUTPUT)
    parser.add_argument("--model-seed", type=int, choices=p.MODEL_SEEDS)
    parser.add_argument("--through-episode", type=int, default=120)
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
