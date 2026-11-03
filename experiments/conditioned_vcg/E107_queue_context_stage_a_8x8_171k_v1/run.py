"""Prepare, train, and analyze E107 queue-observed Stage A."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
import os
from pathlib import Path
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
from experiments.conditioned_vcg.target_size_dense_stage_a_v2 import run as v2run
from experiments.conditioned_vcg.target_size_dense_stage_a_v4 import acceleration
from . import instrumentation, protocol as p
from .representation import (
    QUEUE_CONTEXT_DIM,
    QUEUE_CONTEXT_NAMES,
    QueueContextAgent,
    acceleration_enabled,
    attach_queue_context,
    deduplicated_queue_scores,
    public_queue_context,
)


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def read_json(path: Path) -> dict:
    if not path.is_file():
        raise RuntimeError(f"missing E107 artifact: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"invalid E107 artifact: {path}")
    return value


def prepare(output: Path) -> dict:
    output = output.resolve()
    expected = p.contract()
    path = output / "contract.json"
    if path.is_file():
        if read_json(path) != expected:
            raise RuntimeError("E107 contract or authenticated code changed")
    else:
        if output.exists() and any(output.iterdir()):
            raise RuntimeError("E107 refuses a nonempty uncontracted output")
        write_json(path, expected)
    return {
        "status": "prepared",
        "contract_sha256": expected["contract_sha256"],
        "fresh_initialization": True,
        "maximum_episode": p.MAX_EPISODE,
        "epsilon_decay_decisions": p.EPSILON_DECAY_DECISIONS,
        "decision_checkpoints": list(p.DECISION_CHECKPOINTS),
        "episode_runs": 0,
        "training_runs": 0,
    }


def destination(output: Path) -> Path:
    return (
        output.resolve()
        / "training/operational"
        / p.SCENARIO
        / "seed-0"
    )


def _development_validation_record(*args, **kwargs):
    value = _BASE_VALIDATION_RECORD(*args, **kwargs)
    value["deployment_eligible"] = False
    value["development_selection_only"] = True
    value["e107_queue_context_development"] = True
    return value


_BASE_VALIDATION_RECORD = audit._validation_record


def _train_with_context(args, trainer_protocol):
    original_resume_contract = dense._resume_contract
    original_result = dense._result_payload
    original_checkpoint = EpisodicViabilityGraphHierarchyAgent.checkpoint
    original_validate = dense._validate_resume

    def resume_contract(*positional, **keywords):
        value = original_resume_contract(*positional, **keywords)
        value.update({
            "e107_contract_sha256": p.contract()["contract_sha256"],
            "agent_architecture": "e107_queue_observed_stage_a",
            "queue_context_names": list(QUEUE_CONTEXT_NAMES),
            "queue_context_stored_in_replay": True,
            "future_schedule_visible_to_policy": False,
            "training_horizon_rule": p.HORIZON_RULE,
        })
        return value

    def result_payload(*positional, **keywords):
        value = original_result(*positional, **keywords)
        value.update({
            "e107_contract_sha256": p.contract()["contract_sha256"],
            "training_horizon_rule": p.HORIZON_RULE,
            "epsilon_schedule": {
                "start": p.EPSILON_START,
                "end": p.EPSILON_END,
                "warmup_decisions": p.EPSILON_WARMUP_DECISIONS,
                "decay_decisions": p.EPSILON_DECAY_DECISIONS,
            },
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
            "e107_contract_sha256": p.contract()["contract_sha256"],
            "training_horizon_rule": p.HORIZON_RULE,
            "td_backup_implementation": p.TD_BACKUP_IMPLEMENTATION,
            "td_backup_acceleration_audit": acceleration.audit_payload(agent),
            "performance_claim_authorized": False,
        })
        return value

    def validate_resume(payload, contract):
        original_validate(payload, contract)
        expected = {
            "e107_contract_sha256": p.contract()["contract_sha256"],
            "training_horizon_rule": p.HORIZON_RULE,
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
            raise ValueError(f"E107 resume metadata mismatch: {mismatches!r}")

    with ExitStack() as stack:
        stack.enter_context(patch.object(v2run, "p", trainer_protocol))
        stack.enter_context(
            patch.object(
                EpisodicViabilityGraphHierarchyAgent,
                "_td_batch",
                acceleration.accelerated_cached_td_batch,
            )
        )
        stack.enter_context(
            patch.object(
                EpisodicViabilityGraphHierarchyAgent,
                "checkpoint",
                checkpoint,
            )
        )
        for module in (dense, audit):
            stack.enter_context(
                patch.object(
                    module,
                    "EpisodicViabilityGraphHierarchyAgent",
                    QueueContextAgent,
                )
            )
        stack.enter_context(patch.object(dense, "_resume_contract", resume_contract))
        stack.enter_context(patch.object(dense, "_result_payload", result_payload))
        stack.enter_context(patch.object(dense, "_validate_resume", validate_resume))
        stack.enter_context(
            patch.object(dense, "FROZEN_EPSILON_START", p.EPSILON_START)
        )
        stack.enter_context(
            patch.object(dense, "FROZEN_EPSILON_END", p.EPSILON_END)
        )
        stack.enter_context(
            patch.object(
                dense,
                "FROZEN_EPSILON_WARMUP_DECISIONS",
                p.EPSILON_WARMUP_DECISIONS,
            )
        )
        stack.enter_context(
            patch.object(
                dense,
                "FROZEN_EPSILON_DECAY_DECISIONS",
                p.EPSILON_DECAY_DECISIONS,
            )
        )
        return v2run.train_stage_a(
            args.output,
            args.scenario,
            args.model_seed,
            device_name=args.device,
            stop_after_episode=args.stop_after_episode,
            allow_cpu_pilot=False,
        )


def self_check(output: Path) -> dict:
    prepared = prepare(output)

    class Block:
        def __init__(self, label, arrival, stored=None, delivered=False):
            self.label = label
            self.arrival_step = arrival
            self.stored_time_step = stored
            self.delivered = delivered

    class FutureBlock:
        label = "B2"
        arrival_step = 500

        @property
        def stored_time_step(self):
            raise AssertionError("E107 inspected future storage state")

        @property
        def delivered(self):
            raise AssertionError("E107 inspected future delivery state")

    env = SimpleNamespace(
        time_steps=120,
        blocks=(
            Block("B0", 0),
            Block("B1", 20, stored=30),
            FutureBlock(),
        ),
    )
    snapshot = SimpleNamespace(decision_epoch=120, inbound_label="B0")
    context = public_queue_context(
        env, snapshot, wait_cap=80, required_jobs=3, max_steps=1000
    )
    assert context.waiting_jobs == 1
    assert context.oldest_wait_age == 120
    assert context.inbound_wait_age == 120
    assert context.over_cap_jobs == 1
    assert context.accepted_jobs == 1
    assert context.delivered_jobs == 0
    assert all(0.0 <= value <= 1.0 for value in context.values)

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
    agent = QueueContextAgent(
        config=EpisodicViabilityGraphConfig(
            gamma=0.99, episode_horizon_steps=1000
        ),
        seed=7,
        device="cpu",
        epsilon=p.EPSILON_START,
    )
    with torch.no_grad():
        direct = agent.Q_local(
            (record.current_state,),
            (record.successor_state,),
            (record.action_features,),
        )
        accelerated = deduplicated_queue_scores((record,), agent.Q_local)
    if not torch.equal(direct, accelerated):
        raise RuntimeError("E107 accelerated scoring differs from direct scoring")

    alternative = attach_queue_context(
        state, (1.0,) * QUEUE_CONTEXT_DIM
    )
    encoded = agent.Q_local.context_encoder(
        torch.as_tensor(
            (observed.queue_context, alternative.queue_context),
            dtype=torch.float32,
        )
    )
    if torch.equal(encoded[0], encoded[1]):
        raise RuntimeError("E107 context encoder ignores its public context")

    agent.replay.add(
        ViabilityGraphTransition(
            chosen=record,
            reward=0.0,
            duration=1,
            next_candidates=(),
            done=True,
        )
    )
    payload = agent.checkpoint(
        include_replay=True,
        future_schedule_visible_to_policy=False,
    )
    restored = QueueContextAgent.from_checkpoint(
        payload, device="cpu", resumable=True, seed=7
    )
    replayed = restored.replay.memory[0].chosen.current_state
    if replayed.queue_context != observed.queue_context:
        raise RuntimeError("E107 replay did not retain public queue context")

    schedule = {
        str(decisions): epsilon_at(
            decisions,
            start=p.EPSILON_START,
            end=p.EPSILON_END,
            warmup_decisions=p.EPSILON_WARMUP_DECISIONS,
            decay_decisions=p.EPSILON_DECAY_DECISIONS,
        )
        for decisions in (0, 3_000, 9_000, 18_000, 24_000)
    }
    if (
        abs(schedule["18000"] - p.EPSILON_END) > 1.0e-12
        or schedule["9000"] <= p.EPSILON_END
    ):
        raise RuntimeError("E107 epsilon schedule is not the intended slow decay")
    return {
        "status": "passed",
        "contract_sha256": prepared["contract_sha256"],
        "queue_context_names": list(QUEUE_CONTEXT_NAMES),
        "queue_context_values": list(context.values),
        "future_job_excluded": True,
        "direct_accelerated_max_abs_difference": float(
            (direct - accelerated).abs().max().item()
        ),
        "replay_context_roundtrip": True,
        "epsilon_schedule_probe": schedule,
        "episode_runs": 0,
        "training_runs": 0,
    }


def train_seed(output: Path, through_episode: int, device: str) -> dict:
    prepare(output)
    if not 1 <= through_episode <= p.MAX_EPISODE:
        raise ValueError(f"through-episode must be in [1,{p.MAX_EPISODE}]")
    if device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("E107 training requires CUDA")
    trainer_protocol = p.trainer_protocol()
    args = SimpleNamespace(
        output=output.resolve(),
        scenario=p.SCENARIO,
        model_seed=0,
        device=device,
        stop_after_episode=through_episode,
    )
    with ExitStack() as stack:
        stack.enter_context(
            patch.object(dense, "FROZEN_OBJECTIVE_SPEC", p.e105.e78.SPEC)
        )
        stack.enter_context(
            patch.object(audit, "run_objective_audit_episode", instrumentation.run_episode)
        )
        stack.enter_context(
            patch.object(audit, "summarize_objective_runs", instrumentation.summarize_runs)
        )
        stack.enter_context(
            patch.object(audit, "_validation_record", _development_validation_record)
        )
        stack.enter_context(fast_inputs.enabled())
        stack.enter_context(acceleration_enabled())
        result = _train_with_context(args, trainer_protocol)
    return {
        **result,
        "e107_protocol": p.PROTOCOL,
        "fresh_initialization": True,
        "through_episode": through_episode,
        "maximum_episode": p.MAX_EPISODE,
        "epsilon_decay_decisions": p.EPSILON_DECAY_DECISIONS,
        "paper_evidence": False,
    }


def _row(record: dict, training_history: list[dict]) -> dict:
    summary = record["summary"]
    episode = int(record["checkpoint_episode"])
    training = training_history[episode - 1] if len(training_history) >= episode else {}
    return {
        "checkpoint_episode": episode,
        "training_decision_count": training.get("training_decision_count_after"),
        "training_epsilon": training.get("scheduled_epsilon_after"),
        "strict_completion_rate": summary["strict_method_success_rate"],
        "dense_return": summary["mean_dense_rescored_return"],
        "mean_wait_before_storage": summary["mean_wait_before_storage"],
        "mean_jobs_exceeding_wait_cap": summary["mean_jobs_exceeding_wait_cap"],
        "wait_cap_compliance_rate": summary["mean_wait_cap_compliance_rate"],
        "release_clock_mae": summary["mean_release_clock_mae"],
        "storage_clock_mae": summary["mean_storage_clock_mae"],
        "storage_clock_signed_error": summary["mean_storage_clock_signed_error"],
        "within_20_rate": summary["within_target_window_rate"],
        "steps": summary["mean_steps"],
        "rehandles_per_100": summary["relocations_per_100_deliveries"],
        "guard_activations": summary["total_liveness_guard_activations"],
        "mean_outside_wait_penalty": summary["mean_outside_wait_penalty"],
        "wait_accounting_exact": summary[
            "wait_accounting_exact_for_all_complete_runs"
        ],
    }


def _gate(row: dict, thresholds: dict) -> dict:
    checks = {
        "strict_completion": row["strict_completion_rate"]
        >= thresholds["strict_completion_rate"],
        "wait_reduced": row["mean_wait_before_storage"]
        <= thresholds["mean_wait_before_storage_max"],
        "storage_mae_retained": row["storage_clock_mae"]
        <= thresholds["storage_clock_mae_max"],
        "within_window_retained": row["within_20_rate"]
        >= thresholds["within_20_rate_min"],
        "steps_bounded": row["steps"] <= thresholds["mean_steps_max"],
        "wait_accounting_exact": bool(row["wait_accounting_exact"]),
    }
    return {"passed": all(checks.values()), "checks": checks}


def analyze(output: Path) -> dict:
    prepare(output)
    summary_path = destination(output) / "training-summary.json"
    if summary_path.is_file():
        saved = read_json(summary_path)
    else:
        latest = destination(output) / "latest.pth"
        if not latest.is_file():
            raise RuntimeError("train E107 before analyze")
        payload = torch.load(latest, map_location="cpu", weights_only=False)
        saved = {
            "completed_training_episodes": payload["completed_training_episodes"],
            "training_history": payload["training_history"],
            "validation_history": payload["validation_history"],
        }
    training_history = list(saved.get("training_history", ()))
    rows = [
        _row(record, training_history)
        for record in saved.get("validation_history", ())
    ]
    thresholds = p.contract()["joint_mechanism_thresholds"]
    gates = {
        str(episode): _gate(row, thresholds)
        for episode in p.DECISION_CHECKPOINTS
        for row in rows
        if row["checkpoint_episode"] == episode
    }
    completed = int(saved["completed_training_episodes"])
    report = {
        "protocol": p.PROTOCOL,
        "status": "complete" if completed == p.MAX_EPISODE else "partial",
        "completed_training_episodes": completed,
        "maximum_episode": p.MAX_EPISODE,
        "fresh_initialization": True,
        "queue_context_names": list(QUEUE_CONTEXT_NAMES),
        "epsilon_decay_decisions": p.EPSILON_DECAY_DECISIONS,
        "decision_checkpoints": list(p.DECISION_CHECKPOINTS),
        "rows": rows,
        "decision_checkpoint_gates": gates,
        "first_nonnegative_dense_return_checkpoint": next(
            (row["checkpoint_episode"] for row in rows if row["dense_return"] >= 0),
            None,
        ),
        "interpretation": (
            "Fresh queue-observed development training. The reused validation "
            "panel and one model seed do not support a paper or deployment claim."
        ),
        "stage_b_authorized": False,
        "independent_test_opened": False,
        "paper_evidence": False,
        "episode_runs": 0,
        "training_runs": 0,
    }
    report_path = output.resolve() / "e107-report.json"
    write_json(report_path, report)
    return {
        "status": report["status"],
        "completed_training_episodes": completed,
        "maximum_episode": p.MAX_EPISODE,
        "validation_checkpoints": [row["checkpoint_episode"] for row in rows],
        "decision_checkpoint_gates": gates,
        "report": str(report_path),
        "episode_runs": 0,
        "training_runs": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("prepare", "self-check", "train-seed", "analyze")
    )
    parser.add_argument("--output", type=Path, default=p.DEFAULT_OUTPUT)
    parser.add_argument("--through-episode", type=int, default=36)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.command == "prepare":
        result = prepare(args.output)
    elif args.command == "self-check":
        result = self_check(args.output)
    elif args.command == "train-seed":
        result = train_seed(args.output, args.through_episode, args.device)
    else:
        result = analyze(args.output)
    print(json.dumps(result, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
