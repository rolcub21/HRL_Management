#!/usr/bin/env python3
"""Prepare, verify, train, resume, and summarize E244 Stage A."""

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
    representation as e191_representation,
)
from experiments.conditioned_vcg.E191_fresh_long_stay_crowding_250k_v1.roundoff import (
    repaired_audit,
)
from experiments.conditioned_vcg.E209_e191_method_10x10_longstay_268k_v1 import (
    run as e209_run,
    stage as e209_stage,
    representation,
    scenario,
)
from projects.temporal_consequence_realization.method_conformance import (
    render_validated_conformance_audit,
    validate_protocol,
)

from . import protocol as p


class E244Error(RuntimeError):
    pass


BUDGET_RULE = "fixed_300_episodes_final_e215_method_scale_matched_10x10"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise E244Error(message)


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
    """Install E244 at E209's two explicit protocol seams."""

    with (
        patch.object(e209_run, "p", p),
        patch.object(e209_run, "BUDGET_RULE", BUDGET_RULE),
        patch.object(e209_stage, "p", p),
    ):
        yield


def prepare(output: Path) -> dict:
    contract = p.contract()
    validate_protocol(contract, source=p.PROTOCOL)
    with implementation():
        value = e209_run.prepare(output)
    return {
        **value,
        "protocol": p.PROTOCOL,
        "model_seeds": list(p.MODEL_SEEDS),
        "episodes_per_seed": p.TOTAL_EPISODES,
        "epsilon_decay_decisions": p.EPSILON_DECAY_DECISIONS,
        "target_floor_episode_approximate": p.EPSILON_TARGET_FLOOR_EPISODE,
        "architecture_identical_to_e215_e218": True,
        "controller_learning_rule_identical_to_e215_e218": True,
        "fresh_random_initialization": True,
        "scale_matched_10x10": True,
        "test_panel_opened": False,
    }


def authenticate(output: Path) -> dict:
    with implementation():
        contract = e209_run.authenticate(output)
    validate_protocol(contract, source=p.PROTOCOL)
    return contract


def _state_equal(first: dict, second: dict) -> bool:
    return set(first) == set(second) and all(
        torch.equal(first[name], second[name]) for name in first
    )


def self_check(output: Path) -> dict:
    prepared = prepare(output)
    contract = authenticate(output)
    require(
        p.EPSILON_DECAY_DECISIONS == 101_867,
        "E244 job-normalized E215 exploration schedule differs",
    )
    require(
        p.MODEL_SEEDS == (0, 1, 2)
        and p.TOTAL_EPISODES == 300
        and p.VALIDATION_EVERY == 20,
        "E244 training or validation budget differs",
    )
    require(
        representation.NETWORK_ARCHITECTURE
        == e191_representation.NETWORK_ARCHITECTURE
        and representation.QUEUE_CONTEXT_NAMES
        == e191_representation.QUEUE_CONTEXT_NAMES
        and representation.QUEUE_CONTEXT_DIM
        == e191_representation.QUEUE_CONTEXT_DIM == 8,
        "E244 neural input architecture differs from E215/E218",
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
        "E244 parameter names or shapes differ from E215/E218",
    )
    require(
        fresh.e187_warm_start_manifest is None
        and len(fresh.replay) == 0
        and fresh.decision_count == 0
        and fresh.gradient_steps == 0
        and not fresh.optimizer.state,
        "E244 did not start with a fresh training state",
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
        _state_equal(fresh.Q_local.state_dict(), restored.Q_local.state_dict()),
        "E244 checkpoint did not round-trip",
    )

    env = p.TargetDenseEnv(p.SCENARIO, p.SPEC)
    validation_cases = []
    paired_durations = {}
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
            "E244 validation workload arithmetic differs",
        )
        replicate = seed % 100
        previous = paired_durations.setdefault(
            replicate, tuple(instance.storage_steps_needed)
        )
        require(
            tuple(instance.storage_steps_needed) == previous,
            "E244 paired validation durations differ across workloads",
        )
        env.reset(instance=instance)
        require(
            not any(block.stored or block.delivered for block in env.blocks),
            "E244 validation case does not start empty",
        )
        context = representation.public_queue_context(
            env,
            SimpleNamespace(decision_epoch=0, inbound_label="B1"),
            required_jobs=scenario.TOTAL_JOBS,
            max_steps=p.MAX_STEPS,
        )
        validation_cases.append({
            "seed": seed,
            "target": level,
            "gap": gap,
            "initial_context": list(context.values),
        })
    require(
        len({tuple(case["initial_context"]) for case in validation_cases}) == 1,
        "E244 leaked future workload through the initial observation",
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
        "E244 changed the Stage-A primitive objective",
    )
    return {
        "status": "passed",
        "contract_sha256": prepared["contract_sha256"],
        "epsilon_decay_decisions": p.EPSILON_DECAY_DECISIONS,
        "architecture": contract["architecture"],
        "fresh_initialization": contract["initialization"],
        "scale_match": contract["scale_match"],
        "validation_cases": validation_cases,
        "checkpoint_roundtrip": True,
        "initial_observation_hides_workload": True,
        "primitive_wait_probe_reward": reward,
        "episode_runs": 0,
        "training_runs": 0,
        "test_panel_opened": False,
    }


def train_seed(output: Path, seed: int, through_episode: int, device: str) -> dict:
    prepare(output)
    contract = authenticate(output)
    require(seed in p.MODEL_SEEDS, "E244 model seed must be 0, 1, or 2")
    require(
        1 <= through_episode <= p.TOTAL_EPISODES,
        "E244 through-episode must be between 1 and 300",
    )
    require(
        device == "cuda" and torch.cuda.is_available(),
        "E244 training requires CUDA",
    )
    with implementation(), repaired_audit():
        result = e209_run.train_seed(output, seed, through_episode, device)
    for name in ("e187_protocol", "e209_protocol", "e209_contract_sha256"):
        result.pop(name, None)
    return {
        **result,
        "e244_protocol": p.PROTOCOL,
        "e244_contract_sha256": contract["contract_sha256"],
        "epsilon_decay_decisions": p.EPSILON_DECAY_DECISIONS,
        "fresh_random_initialization": True,
        "architecture_identical_to_e215_e218": True,
        "controller_learning_rule_identical_to_e215_e218": True,
        "scale_matched_10x10": True,
        "paper_evidence": False,
    }


def status(output: Path) -> dict:
    authenticate(output)
    with implementation():
        result = e209_run.status(output)
    return {
        **result,
        "protocol": p.PROTOCOL,
        "epsilon_decay_decisions": p.EPSILON_DECAY_DECISIONS,
    }


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
        history = list(summary["training"]["runs"])
        rows = []
        for record in summary.get("validation_history", ()):
            with implementation():
                row = e187_run._validation_row(record, history)
            row["by_workload"] = record["summary"].get("e209_by_workload", {})
            rows.append(row)
        expected = list(range(
            p.VALIDATION_EVERY, completed + 1, p.VALIDATION_EVERY
        ))
        require(
            [row["checkpoint_episode"] for row in rows] == expected,
            f"E244 seed {seed} validation coverage differs",
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
        "epsilon": contract["epsilon"],
        "workload": contract["workload"],
        "scale_match": contract["scale_match"],
        "seeds": seeds,
        "test_panel_opened": False,
        "paper_evidence": False,
    }
    report_path = output / "e244-report.json"
    e187_run._write(report_path, report)
    text_path = output / "REPORT.txt"
    text_path.write_text(
        render_validated_conformance_audit(_declaration(contract))
        + "\nRESULTS\n"
        + "=" * 72
        + "\n\n"
        + f"Status: {report['status']}\n"
        + "Development Stage A only; the sealed test panel remains closed.\n",
        encoding="utf-8",
    )
    return {
        "status": report["status"],
        "report": str(report_path),
        "text_report": str(text_path),
        "test_panel_opened": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("prepare", "self-check", "train-seed", "status", "analyze"),
    )
    parser.add_argument("--output", type=Path, default=p.DEFAULT_OUTPUT)
    parser.add_argument("--model-seed", type=int, choices=p.MODEL_SEEDS)
    parser.add_argument("--through-episode", type=int, default=300)
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

