#!/usr/bin/env python3
"""Prepare, train, inspect, and analyze E187."""

from __future__ import annotations

import argparse
import builtins
from contextlib import ExitStack
import json
import os
from pathlib import Path
from statistics import fmean
from types import SimpleNamespace
from unittest.mock import patch

import torch

import run_vcg_objective_gamma_audit as audit
import train_vcg_dense_proper as dense
from PSLAP.viability import RecoveryState
from train_viability_graph_smdp import epsilon_at
from viability_graph_hierarchy import (
    ACTION_FEATURE_DIM,
    ViabilityGraphCandidateRecord,
    ViabilityGraphTransition,
)
from viability_graph_episodic_audit import (
    EpisodicViabilityGraphConfig,
    EpisodicViabilityGraphHierarchyAgent,
)
from experiments.conditioned_vcg.E100_accelerated_release_clock_163k_v1 import (
    fast_inputs,
)
from experiments.conditioned_vcg.E107_queue_context_stage_a_8x8_171k_v1 import (
    representation as e107_representation,
)
from experiments.conditioned_vcg.target_size_dense_stage_a_v2 import run as v2run
from experiments.conditioned_vcg.target_size_dense_stage_a_v4 import acceleration

from . import instrumentation, protocol as p
from .objective import COMPONENT_NAMES, scalar_cost
from .representation import (
    QUEUE_CONTEXT_DIM,
    QUEUE_CONTEXT_NAMES,
    WaitCorrectedQueueAgent,
    attach_queue_context,
    authenticate_sources,
    deduplicated_queue_scores,
    public_queue_context,
    warm_start_disabled,
)


class E187Error(RuntimeError):
    pass


BUDGET_RULE = "fixed_300_episodes_wait_corrected_e78"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise E187Error(message)


def _read(path: Path) -> dict:
    require(path.is_file(), f"E187 artifact is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"E187 artifact is invalid: {path}")
    return value


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def authenticate(output: Path) -> dict:
    observed = _read(output.resolve() / "contract.json")
    expected = p.contract()
    require(observed == expected, "E187 contract, sources, or objective changed")
    authenticate_sources()
    return expected


def prepare(output: Path) -> dict:
    output = output.resolve()
    authenticate_sources()
    expected = p.contract()
    path = output / "contract.json"
    if path.is_file():
        authenticate(output)
    else:
        require(
            not output.exists() or not any(output.iterdir()),
            "E187 refuses a nonempty uncontracted output",
        )
        _write(path, expected)
    return {
        "status": "prepared",
        "contract_sha256": expected["contract_sha256"],
        "model_seeds": list(p.MODEL_SEEDS),
        "episodes_per_seed": p.TOTAL_EPISODES,
        "objective": expected["objective"],
        "warm_start": expected["warm_start"],
        "epsilon_decay_decisions": p.EPSILON_DECAY_DECISIONS,
        "validation_every": p.VALIDATION_EVERY,
        "episode_runs": 0,
        "training_runs": 0,
    }


def destination(output: Path, seed: int) -> Path:
    return (
        output.resolve()
        / "training/operational"
        / p.SCENARIO
        / f"seed-{seed}"
    )


_BASE_VALIDATION_RECORD = audit._validation_record


def _development_validation_record(*args, **kwargs):
    value = _BASE_VALIDATION_RECORD(*args, **kwargs)
    value["deployment_eligible"] = False
    value["development_selection_only"] = True
    value["e187_wait_corrected_development"] = True
    return value


def self_check(output: Path) -> dict:
    prepared = prepare(output)
    contract = authenticate(output)
    require(contract["objective"]["wait_grace_steps"] == 0,
            "E187 accidentally introduced a wait grace period")
    require(contract["gamma"] == 1.0, "E187 J is no longer undiscounted")
    require(not contract["frontier"]["accept_first_mask"],
            "E187 introduced an Accept-first preference rule")
    require(not contract["frontier"]["inbound_defer_mask"],
            "E187 introduced an inbound-Defer preference rule")

    # The real environment must charge one released, unstored block exactly
    # 0.5 on a primitive WAIT at t=0, with no equipment charge.
    env = p.TargetDenseEnv(p.SCENARIO, p.SPEC)
    instance = env.sample_episode_instance(p.VALIDATION_SEEDS[p.SCENARIO][0])
    env.reset(instance=instance)
    _state, reward, _done, info = env.step(env.ACTION_IDS["WAIT"])
    components = info["timing_objective_components"]["e187_components"]
    require(abs(reward + 0.5) < 1.0e-12,
            "E187 full waiting charge is not active from release")
    require(components["admission_wait_accrual"] == 1.0
            and components["equipment_work"] == 0.0,
            "E187 primitive wait/equipment components differ")
    require(abs(scalar_cost((1.0, 2.0, 3.0, 4.0, 5.0)) - 55.9) < 1e-12,
            "E187 scalar objective arithmetic changed")

    class Block:
        def __init__(self, label, arrival, stored=None, delivered=False):
            self.label = label
            self.arrival_step = arrival
            self.stored_time_step = stored
            self.delivered = delivered

    class FutureBlock:
        label = "future"
        arrival_step = 500

        @property
        def stored_time_step(self):
            raise AssertionError("E187 inspected future storage state")

        @property
        def delivered(self):
            raise AssertionError("E187 inspected future delivery state")

    fake_env = SimpleNamespace(
        time_steps=120,
        blocks=(Block("B0", 0), Block("B1", 20, 30), FutureBlock()),
    )
    snapshot = SimpleNamespace(decision_epoch=120, inbound_label="B0")
    context = public_queue_context(
        fake_env, snapshot, required_jobs=3, max_steps=1000
    )
    require(context.waiting_jobs == 1 and context.oldest_wait_age == 120,
            "E187 public queue context differs")
    require(all(0.0 <= value <= 1.0 for value in context.values),
            "E187 queue context is not bounded")

    state = RecoveryState(
        rows=2,
        cols=2,
        traversable=frozenset({(0, 0), (0, 1), (1, 0), (1, 1)}),
        storage_cells=frozenset({(1, 0), (1, 1)}),
        exits=((0, 1),),
        blocks=(),
        agent_position=(0, 0),
        pickup_cells=frozenset({(0, 0)}),
        wait_cells=frozenset({(0, 0)}),
    )
    observed = attach_queue_context(state, context.values)
    record = ViabilityGraphCandidateRecord(
        key="accept:B0:1:0",
        mode_id=0,
        action_type="accept",
        current_state=observed,
        successor_state=state,
        action_features=(0.0,) * ACTION_FEATURE_DIM,
    )
    config = EpisodicViabilityGraphConfig(
        gamma=p.GAMMA, episode_horizon_steps=p.MAX_STEPS
    )
    with warm_start_disabled():
        fresh = WaitCorrectedQueueAgent(
            config=config, seed=0, device="cpu", epsilon=p.EPSILON_START
        )
    warm = WaitCorrectedQueueAgent(
        config=config, seed=0, device="cpu", epsilon=p.EPSILON_START
    )
    require(warm.e187_warm_start_manifest is not None,
            "E187 did not apply its graph warm start")
    for module_name in ("action_encoder", "context_encoder", "q_head"):
        fresh_state = getattr(fresh.Q_local, module_name).state_dict()
        warm_state = getattr(warm.Q_local, module_name).state_dict()
        require(all(torch.equal(fresh_state[name], warm_state[name])
                    for name in fresh_state),
                f"E187 transferred the reset module {module_name}")
    require(any(
        not torch.equal(
            fresh.Q_local.graph_encoder.state_dict()[name],
            warm.Q_local.graph_encoder.state_dict()[name],
        )
        for name in fresh.Q_local.graph_encoder.state_dict()
    ), "E187 graph warm start had no effect")
    require(all(
        torch.equal(
            warm.Q_local.graph_encoder.state_dict()[name],
            warm.Q_target.graph_encoder.state_dict()[name],
        )
        for name in warm.Q_local.graph_encoder.state_dict()
    ), "E187 local and target warm encoders differ")
    require(len(warm.replay) == 0 and warm.decision_count == 0
            and warm.gradient_steps == 0 and not warm.optimizer.state,
            "E187 inherited training state from E78")

    with torch.no_grad():
        direct = warm.Q_local(
            (record.current_state,),
            (record.successor_state,),
            (record.action_features,),
        )
        accelerated = deduplicated_queue_scores((record,), warm.Q_local)
    require(torch.equal(direct, accelerated),
            "E187 accelerated score differs from direct score")

    warm.replay.add(ViabilityGraphTransition(
        chosen=record,
        reward=0.0,
        duration=1,
        next_candidates=(),
        done=True,
    ))
    payload = warm.checkpoint(
        include_replay=True, future_schedule_visible_to_policy=False
    )
    restored = WaitCorrectedQueueAgent.from_checkpoint(
        payload, device="cpu", resumable=True, seed=0
    )
    replayed = restored.replay.memory[0].chosen.current_state
    require(replayed.queue_context == observed.queue_context,
            "E187 replay did not retain queue context")

    schedule = {
        str(decisions): epsilon_at(
            decisions,
            start=p.EPSILON_START,
            end=p.EPSILON_END,
            warmup_decisions=p.EPSILON_WARMUP_DECISIONS,
            decay_decisions=p.EPSILON_DECAY_DECISIONS,
        )
        for decisions in (0, p.EPSILON_DECAY_DECISIONS // 2,
                          p.EPSILON_DECAY_DECISIONS)
    }
    require(abs(schedule[str(p.EPSILON_DECAY_DECISIONS)] - p.EPSILON_END) < 1e-12,
            "E187 epsilon schedule does not reach its floor")
    return {
        "status": "passed",
        "contract_sha256": prepared["contract_sha256"],
        "objective": contract["objective"],
        "primitive_wait_probe_reward": reward,
        "queue_context_names": list(QUEUE_CONTEXT_NAMES),
        "queue_context_values": list(context.values),
        "future_job_excluded": True,
        "warm_start_manifest": warm.e187_warm_start_manifest,
        "reset_modules_verified": ["action_encoder", "context_encoder", "q_head"],
        "optimizer_replay_counters_reset": True,
        "direct_accelerated_max_abs_difference": float(
            (direct - accelerated).abs().max().item()
        ),
        "replay_context_roundtrip": True,
        "epsilon_schedule_probe": schedule,
        "episode_runs": 0,
        "training_runs": 0,
    }


def _train_with_context(args, trainer_protocol) -> dict:
    original_resume_contract = dense._resume_contract
    original_result = dense._result_payload
    original_checkpoint = EpisodicViabilityGraphHierarchyAgent.checkpoint
    original_validate = dense._validate_resume
    original_print = builtins.print

    def e187_print(*values, **keywords):
        if values and isinstance(values[0], str):
            first = values[0]
            if first.startswith("VCG-Dense v1.1 |"):
                first = first.replace(
                    "objective=40-1.5|e|-0.5(|e|-20)+",
                    "objective=2T+E+0.5W+0.1L+10R",
                )
                values = (first, *values[1:])
        return original_print(*values, **keywords)

    def resume_contract(*positional, **keywords):
        value = original_resume_contract(*positional, **keywords)
        value.update({
            "e187_contract_sha256": p.contract()["contract_sha256"],
            "e187_objective": p.contract()["objective"],
            "agent_architecture": "e187_wait_corrected_queue_observed",
            "queue_context_names": list(QUEUE_CONTEXT_NAMES),
            "queue_context_stored_in_replay": True,
            "future_schedule_visible_to_policy": False,
            "warm_start": p.contract()["warm_start"],
            "training_budget_rule": BUDGET_RULE,
        })
        return value

    def result_payload(*positional, **keywords):
        value = original_result(*positional, **keywords)
        value.update({
            "e187_contract_sha256": p.contract()["contract_sha256"],
            "e187_objective": p.contract()["objective"],
            "e187_epsilon_schedule": {
                "start": p.EPSILON_START,
                "end": p.EPSILON_END,
                "warmup_decisions": p.EPSILON_WARMUP_DECISIONS,
                "decay_decisions": p.EPSILON_DECAY_DECISIONS,
            },
            "training_budget_rule": BUDGET_RULE,
            "td_backup_acceleration": acceleration.audit_payload(
                keywords["agent"]
            ),
            "development_only": True,
            "paper_evidence": False,
        })
        return value

    def checkpoint(agent, *positional, **keywords):
        value = original_checkpoint(agent, *positional, **keywords)
        value.update({
            "e187_contract_sha256": p.contract()["contract_sha256"],
            "e187_objective": p.contract()["objective"],
            "e187_training_budget_rule": BUDGET_RULE,
            "e187_epsilon_decay_decisions": p.EPSILON_DECAY_DECISIONS,
            "td_backup_implementation": p.TD_BACKUP_IMPLEMENTATION,
            "td_backup_acceleration_audit": acceleration.audit_payload(agent),
            "performance_claim_authorized": False,
        })
        return value

    def validate_resume(payload, contract):
        original_validate(payload, contract)
        expected = {
            "e187_contract_sha256": p.contract()["contract_sha256"],
            "e187_objective": p.contract()["objective"],
            "e187_training_budget_rule": BUDGET_RULE,
            "e187_epsilon_decay_decisions": p.EPSILON_DECAY_DECISIONS,
            "td_backup_implementation": p.TD_BACKUP_IMPLEMENTATION,
            "queue_context_stored_in_replay": True,
            "future_schedule_visible_to_policy": False,
        }
        mismatches = {
            name: (payload.get(name), wanted)
            for name, wanted in expected.items()
            if payload.get(name) != wanted
        }
        if mismatches:
            raise ValueError(f"E187 resume metadata mismatch: {mismatches!r}")

    with ExitStack() as stack:
        stack.enter_context(patch.object(v2run, "p", trainer_protocol))
        stack.enter_context(patch.object(dense, "FROZEN_OBJECTIVE_SPEC", p.SPEC))
        stack.enter_context(patch.object(dense, "FROZEN_GAMMA", p.GAMMA))
        stack.enter_context(patch.object(dense, "FROZEN_EPSILON_START", p.EPSILON_START))
        stack.enter_context(patch.object(dense, "FROZEN_EPSILON_END", p.EPSILON_END))
        stack.enter_context(patch.object(
            dense, "FROZEN_EPSILON_WARMUP_DECISIONS", p.EPSILON_WARMUP_DECISIONS
        ))
        stack.enter_context(patch.object(
            dense, "FROZEN_EPSILON_DECAY_DECISIONS", p.EPSILON_DECAY_DECISIONS
        ))
        stack.enter_context(patch.object(
            EpisodicViabilityGraphHierarchyAgent,
            "_td_batch",
            acceleration.accelerated_cached_td_batch,
        ))
        stack.enter_context(patch.object(
            EpisodicViabilityGraphHierarchyAgent, "checkpoint", checkpoint
        ))
        for module in (dense, audit):
            stack.enter_context(patch.object(
                module,
                "EpisodicViabilityGraphHierarchyAgent",
                WaitCorrectedQueueAgent,
            ))
        stack.enter_context(patch.object(dense, "_resume_contract", resume_contract))
        stack.enter_context(patch.object(dense, "_result_payload", result_payload))
        stack.enter_context(patch.object(dense, "_validate_resume", validate_resume))
        stack.enter_context(patch.object(
            audit, "run_objective_audit_episode", instrumentation.run_episode
        ))
        stack.enter_context(patch.object(
            audit, "summarize_objective_runs", instrumentation.summarize_runs
        ))
        stack.enter_context(patch.object(
            audit, "_rescore_run", instrumentation.single_objective_rescore
        ))
        stack.enter_context(patch.object(
            audit, "_validation_record", _development_validation_record
        ))
        stack.enter_context(patch.object(builtins, "print", e187_print))
        stack.enter_context(fast_inputs.enabled())
        stack.enter_context(e107_representation.acceleration_enabled())
        return v2run.train_stage_a(
            args.output,
            args.scenario,
            args.model_seed,
            device_name=args.device,
            stop_after_episode=args.stop_after_episode,
            allow_cpu_pilot=False,
        )


def train_seed(output: Path, seed: int, through_episode: int, device: str) -> dict:
    prepare(output)
    contract = authenticate(output)
    require(seed in p.MODEL_SEEDS, "E187 model seed must be 0, 1, or 2")
    require(1 <= through_episode <= p.TOTAL_EPISODES,
            "E187 through-episode must be between 1 and 300")
    require(device == "cuda" and torch.cuda.is_available(),
            "E187 training requires CUDA")
    args = SimpleNamespace(
        output=output.resolve(),
        scenario=p.SCENARIO,
        model_seed=seed,
        device=device,
        stop_after_episode=through_episode,
    )
    result = _train_with_context(args, p.trainer_protocol())
    return {
        **result,
        "e187_protocol": p.PROTOCOL,
        "e187_contract_sha256": contract["contract_sha256"],
        "model_seed": seed,
        "through_episode": through_episode,
        "paper_evidence": False,
    }


def _summary_path(output: Path, seed: int) -> Path:
    return destination(output, seed) / "training-summary.json"


def _seed_summary(output: Path, seed: int) -> dict:
    summary = _read(_summary_path(output, seed))
    require(summary.get("model_seed") == seed
            and summary.get("scenario") == p.SCENARIO
            and summary.get("training_protocol") == p.PROTOCOL
            and summary.get("target_contract_sha256") == p.contract()["contract_sha256"]
            and summary.get("gamma") == p.GAMMA
            and summary.get("total_training_episodes") == p.TOTAL_EPISODES
            and summary.get("e187_objective") == p.contract()["objective"]
            and not summary.get("test_panel_opened", False),
            f"E187 seed {seed} summary identity differs")
    return summary


def _validation_row(record: dict, training_history: list[dict]) -> dict:
    summary = record["summary"]
    episode = int(record["checkpoint_episode"])
    training = training_history[episode - 1] if len(training_history) >= episode else {}
    return {
        "checkpoint_episode": episode,
        "training_decision_count": training.get("training_decision_count_after"),
        "training_epsilon": training.get("scheduled_epsilon_after"),
        "strict_completion_rate": summary["strict_method_success_rate"],
        "J": summary["mean_e187_J"],
        "components": {
            name: summary[f"mean_component_{name}"]
            for name in COMPONENT_NAMES
        },
        "mean_admission_wait": summary["mean_admission_wait"],
        "median_admission_wait": summary["mean_median_admission_wait"],
        "p90_admission_wait": summary["mean_p90_admission_wait"],
        "max_admission_wait": summary["mean_max_admission_wait"],
        "storage_clock_mae": summary["mean_storage_clock_mae"],
        "storage_within_20_rate": summary["mean_storage_within_20_rate"],
        "release_clock_mae": summary["mean_release_clock_mae"],
        "release_within_20_rate": summary["mean_release_within_20_rate"],
        "mean_peak_stored": summary["mean_peak_stored"],
        "max_peak_stored": summary["max_peak_stored"],
        "steps": summary["mean_steps"],
        "rehandles_per_100": summary["relocations_per_100_deliveries"],
        "unknown_candidates_rejected": summary["total_unknown_candidates_rejected"],
        "fail_closed_rejections": summary["total_fail_closed_rejections"],
        "objective_accounting_exact": summary[
            "objective_accounting_exact_for_all_runs"
        ],
        "horizon_empirically_nonbinding": summary[
            "horizon_empirically_nonbinding"
        ],
    }


def analyze(output: Path) -> dict:
    output = output.resolve()
    contract = authenticate(output)
    seeds = []
    for seed in p.MODEL_SEEDS:
        path = _summary_path(output, seed)
        if not path.is_file():
            seeds.append({"model_seed": seed, "status": "not_started",
                          "completed_training_episodes": 0})
            continue
        summary = _seed_summary(output, seed)
        completed = int(summary["completed_training_episodes"])
        training_history = list(summary["training"]["runs"])
        rows = [
            _validation_row(record, training_history)
            for record in summary.get("validation_history", ())
        ]
        expected = list(range(p.VALIDATION_EVERY, completed + 1, p.VALIDATION_EVERY))
        require([row["checkpoint_episode"] for row in rows] == expected,
                f"E187 seed {seed} validation coverage differs")
        seeds.append({
            "model_seed": seed,
            "status": "complete" if completed == p.TOTAL_EPISODES else "partial",
            "completed_training_episodes": completed,
            "training_summary": str(path),
            "validation_history": rows,
            "terminal": rows[-1] if completed == p.TOTAL_EPISODES else None,
            "trainer_best_checkpoint_descriptive_only": summary.get("best_checkpoint"),
        })
    complete = all(row["status"] == "complete" for row in seeds)
    terminal_j = [
        float(row["terminal"]["J"])
        for row in seeds
        if row.get("terminal") is not None
    ]
    report = {
        "status": "complete" if complete else "partial",
        "protocol": p.PROTOCOL,
        "contract_sha256": contract["contract_sha256"],
        "objective": contract["objective"],
        "scope": "wait-corrected E78 workload and certifier; development only",
        "seeds": seeds,
        "terminal_J_mean_across_seeds": (
            float(fmean(terminal_j)) if len(terminal_j) == len(p.MODEL_SEEDS) else None
        ),
        "test_panel_opened": False,
        "paper_evidence": False,
        "episode_runs": 0,
        "training_runs": 0,
    }
    path = output / "e187-report.json"
    _write(path, report)
    return {"status": report["status"], "report": str(path),
            "episode_runs": 0, "training_runs": 0}


def status(output: Path) -> dict:
    authenticate(output)
    rows = []
    for seed in p.MODEL_SEEDS:
        path = _summary_path(output, seed)
        if not path.is_file():
            rows.append({"model_seed": seed, "status": "not_started",
                         "completed_training_episodes": 0})
            continue
        value = _seed_summary(output, seed)
        rows.append({
            "model_seed": seed,
            "status": value["status"],
            "completed_training_episodes": value["completed_training_episodes"],
            "decision_count": value["agent_audit"]["decision_count"],
            "epsilon": value["agent_audit"]["epsilon"],
        })
    return {"status": "ok", "seeds": rows,
            "episode_runs": 0, "training_runs": 0}


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
