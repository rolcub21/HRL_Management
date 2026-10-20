"""Prepare, train, and analyze E115 Release-Clock ET Cost VCG."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
import math
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
)
from viability_graph_episodic_audit import (
    EpisodicViabilityGraphConfig,
)
from experiments.conditioned_vcg.E100_accelerated_release_clock_163k_v1 import (
    fast_inputs,
)
from experiments.conditioned_vcg.E107_queue_context_stage_a_8x8_171k_v1.representation import (
    QUEUE_CONTEXT_DIM,
    attach_queue_context,
)
from experiments.conditioned_vcg.target_size_dense_stage_a_v2 import run as v2run
from . import instrumentation, protocol as p
from .model import (
    COMPONENT_LOSS_WEIGHTS,
    FactorizedConservativeAgent,
    FactorizedTransition,
    learning_rate_at,
)
from .rewards import (
    ARRIVAL_GAP,
    COMPONENT_NAMES,
    ReleaseETCostEnv,
    PrimitiveComponents,
    TOLERANCE,
    earliness_tardiness,
    objective_cost,
    scalarized_components,
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
        raise RuntimeError(f"missing E115 artifact: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"invalid E115 artifact: {path}")
    return value


def prepare(output: Path) -> dict:
    output = output.resolve()
    expected = p.contract()
    path = output / "contract.json"
    if path.is_file():
        if read_json(path) != expected:
            raise RuntimeError("E115 contract or authenticated code changed")
    else:
        if output.exists() and any(output.iterdir()):
            raise RuntimeError("E115 refuses a nonempty uncontracted output")
        write_json(path, expected)
    return {
        "status": "prepared",
        "architecture_name": p.ARCHITECTURE_NAME,
        "contract_sha256": expected["contract_sha256"],
        "fresh_initialization": True,
        "maximum_episode": p.MAX_EPISODE,
        "epsilon_decay_decisions": p.EPSILON_DECAY_DECISIONS,
        "decision_checkpoints": list(p.DECISION_CHECKPOINTS),
        "episode_runs": 0,
        "training_runs": 0,
    }


def destination(output: Path) -> Path:
    return output.resolve() / "training/operational" / p.SCENARIO / "seed-0"


_BASE_VALIDATION_RECORD = audit._validation_record


def _development_validation_record(*args, **kwargs):
    value = _BASE_VALIDATION_RECORD(*args, **kwargs)
    value["deployment_eligible"] = False
    value["development_selection_only"] = True
    value["e115_release_et_cost_development"] = True
    return value


def _agent_audit(agent: FactorizedConservativeAgent) -> dict:
    requests = (
        agent._target_component_cache_hits
        + agent._target_component_cache_misses
    )
    return {
        "cache_scope": "current process since start or resume",
        "component_names": list(COMPONENT_NAMES),
        "component_loss_weights": list(COMPONENT_LOSS_WEIGHTS),
        "anchor_pairs_used": int(agent.anchor_pairs_used),
        "anchor_loss_observations": len(agent.anchor_loss_history),
        "target_component_cache_requests": requests,
        "target_component_cache_hits": agent._target_component_cache_hits,
        "target_component_cache_misses": agent._target_component_cache_misses,
        "target_component_cache_hit_rate": (
            None if requests == 0
            else agent._target_component_cache_hits / requests
        ),
        "current_learning_rate": learning_rate_at(agent.decision_count),
    }


def _train_factorized(args, trainer_protocol):
    original_resume_contract = dense._resume_contract
    original_result = dense._result_payload
    original_validate = dense._validate_resume
    original_checkpoint = FactorizedConservativeAgent.checkpoint

    def resume_contract(*positional, **keywords):
        value = original_resume_contract(*positional, **keywords)
        value.update({
            "e115_contract_sha256": p.contract()["contract_sha256"],
            "agent_architecture": "release_et_cost_vcg_e115",
            "component_names": list(COMPONENT_NAMES),
            "queue_context_stored_in_replay": True,
            "future_schedule_visible_to_policy": False,
            "training_horizon_rule": p.HORIZON_RULE,
            "td_backup_implementation": p.TD_BACKUP_IMPLEMENTATION,
        })
        return value

    def result_payload(*positional, **keywords):
        value = original_result(*positional, **keywords)
        agent = keywords["agent"]
        value.update({
            "e115_contract_sha256": p.contract()["contract_sha256"],
            "architecture_name": p.ARCHITECTURE_NAME,
            "training_horizon_rule": p.HORIZON_RULE,
            "epsilon_schedule": {
                "start": p.EPSILON_START,
                "end": p.EPSILON_END,
                "warmup_decisions": p.EPSILON_WARMUP_DECISIONS,
                "decay_decisions": p.EPSILON_DECAY_DECISIONS,
            },
            "e115_agent_audit": _agent_audit(agent),
            "development_only": True,
            "paper_evidence": False,
        })
        return value

    def validate_resume(payload, contract):
        original_validate(payload, contract)
        expected = {
            "e115_contract_sha256": p.contract()["contract_sha256"],
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
            raise ValueError(f"E115 resume metadata mismatch: {mismatches!r}")

    def checkpoint(agent, *positional, **keywords):
        value = original_checkpoint(agent, *positional, **keywords)
        value.update({
            "e115_contract_sha256": p.contract()["contract_sha256"],
            "training_horizon_rule": p.HORIZON_RULE,
            "td_backup_implementation": p.TD_BACKUP_IMPLEMENTATION,
            "queue_context_stored_in_replay": True,
            "future_schedule_visible_to_policy": False,
            "performance_claim_authorized": False,
        })
        return value

    with ExitStack() as stack:
        stack.enter_context(patch.object(v2run, "p", trainer_protocol))
        for module in (dense, audit):
            stack.enter_context(patch.object(
                module,
                "EpisodicViabilityGraphHierarchyAgent",
                FactorizedConservativeAgent,
            ))
        stack.enter_context(patch.object(
            FactorizedConservativeAgent, "checkpoint", checkpoint
        ))
        stack.enter_context(patch.object(dense, "_resume_contract", resume_contract))
        stack.enter_context(patch.object(dense, "_result_payload", result_payload))
        stack.enter_context(patch.object(dense, "_validate_resume", validate_resume))
        stack.enter_context(patch.object(dense, "FROZEN_EPSILON_START", p.EPSILON_START))
        stack.enter_context(patch.object(dense, "FROZEN_EPSILON_END", p.EPSILON_END))
        stack.enter_context(patch.object(
            dense,
            "FROZEN_EPSILON_WARMUP_DECISIONS",
            p.EPSILON_WARMUP_DECISIONS,
        ))
        stack.enter_context(patch.object(
            dense,
            "FROZEN_EPSILON_DECAY_DECISIONS",
            p.EPSILON_DECAY_DECISIONS,
        ))
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
    timing_probe = {
        "late_35": earliness_tardiness(35),
        "early_35": earliness_tardiness(-35),
        "inside_positive_edge": earliness_tardiness(TOLERANCE),
        "inside_negative_edge": earliness_tardiness(-TOLERANCE),
    }
    if timing_probe != {
        "late_35": (15.0, 0.0),
        "early_35": (0.0, 15.0),
        "inside_positive_edge": (0.0, 0.0),
        "inside_negative_edge": (0.0, 0.0),
    }:
        raise RuntimeError("E115 earliness--tardiness formula changed")

    primitive = PrimitiveComponents(2.0, 3.0, 4.0, 1.0).vector()
    if not math.isclose(objective_cost(primitive), 17.4, abs_tol=1.0e-12):
        raise RuntimeError("E115 scalar component reconstruction changed")

    env = ReleaseETCostEnv(p.SCENARIO, p.e107.e105.e78.SPEC)
    instance = env.sample_episode_instance(
        p.e107.e105.e78.VALIDATION_SEEDS[p.SCENARIO][0]
    )
    expected_arrivals = tuple(
        ARRIVAL_GAP * index for index in range(len(instance.arrival_steps))
    )
    if tuple(instance.arrival_steps) != expected_arrivals:
        raise RuntimeError("E115 feasible arrival schedule changed")
    env.reset(instance=instance)
    _, environment_reward, _, info = env.step(env.ACTION_IDS["WAIT"])
    if abs(environment_reward) > 1.0e-10:
        raise RuntimeError("E115 pre-due WAIT must have zero equipment cost")
    observed_components = info["timing_objective_components"][
        "e115_release_et_components"
    ]
    if observed_components != {
        "release_tardiness_accrual": 0.0,
        "release_earliness_at_delivery": 0.0,
        "equipment_work": 0.0,
        "physical_rehandles": 0.0,
    }:
        raise RuntimeError("E115 real environment component record changed")

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
    observed = attach_queue_context(state, (0.0,) * QUEUE_CONTEXT_DIM)
    records = tuple(
        ViabilityGraphCandidateRecord(
            key=key,
            mode_id=mode,
            action_type=action,
            current_state=observed,
            successor_state=state,
            action_features=(float(index),) + (0.0,) * (ACTION_FEATURE_DIM - 1),
        )
        for index, (key, mode, action) in enumerate((
            ("accept:B0:1:0", 0, "accept"),
            ("defer", 2, "defer"),
        ))
    )
    agent = FactorizedConservativeAgent(
        config=EpisodicViabilityGraphConfig(
            gamma=1.0,
            episode_horizon_steps=1000,
            batch_size=2,
        ),
        seed=7,
        device="cpu",
        epsilon=p.EPSILON_START,
    )
    with torch.no_grad(), fast_inputs.enabled():
        vectors = agent._component_scores(records)
        merits = agent._score_records(records)
    if vectors.shape != (2, len(COMPONENT_NAMES)) or merits.shape != (2,):
        raise RuntimeError("E115 network output shape changed")
    if bool((vectors < 0).any()):
        raise RuntimeError("E115 cost head produced a negative value")

    terminal_components = (5.0, 1.0, 2.0, 0.0)
    terminal = FactorizedTransition(
        chosen=records[0],
        reward=scalarized_components(terminal_components) / 0.01,
        duration=2,
        next_candidates=(),
        done=True,
        component_returns=terminal_components,
    )
    agent.replay.add(terminal)
    payload = agent.checkpoint(
        include_replay=True,
        future_schedule_visible_to_policy=False,
    )
    restored = FactorizedConservativeAgent.from_checkpoint(
        payload, device="cpu", resumable=True, seed=7
    )
    replayed = restored.replay.memory[0]
    if not isinstance(replayed, FactorizedTransition):
        raise RuntimeError("E115 replay lost its factorized transition type")
    if replayed.component_returns != terminal_components:
        raise RuntimeError("E115 replay lost component returns")

    schedule = {
        str(decisions): epsilon_at(
            decisions,
            start=p.EPSILON_START,
            end=p.EPSILON_END,
            warmup_decisions=p.EPSILON_WARMUP_DECISIONS,
            decay_decisions=p.EPSILON_DECAY_DECISIONS,
        )
        for decisions in (0, 12_000, 24_000, 30_000)
    }
    if (
        abs(schedule["24000"] - p.EPSILON_END) > 1.0e-12
        or not p.EPSILON_END < schedule["12000"] < p.EPSILON_START
    ):
        raise RuntimeError("E115 epsilon schedule changed")
    learning_rates = {
        str(decisions): learning_rate_at(decisions)
        for decisions in (0, 17_999, 18_000, 23_999, 24_000, 36_000)
    }
    return {
        "status": "passed",
        "contract_sha256": prepared["contract_sha256"],
        "architecture_name": p.ARCHITECTURE_NAME,
        "earliness_tardiness_probe": timing_probe,
        "objective_cost_probe": objective_cost(primitive),
        "arrival_first_last": [instance.arrival_steps[0], instance.arrival_steps[-1]],
        "real_environment_predue_wait_reward": environment_reward,
        "component_names": list(COMPONENT_NAMES),
        "network_output_shape": list(vectors.shape),
        "cost_heads_nonnegative": True,
        "replay_component_roundtrip": True,
        "epsilon_schedule_probe": schedule,
        "learning_rate_schedule_probe": learning_rates,
        "episode_runs": 0,
        "training_runs": 0,
    }


def train_seed(output: Path, through_episode: int, device: str) -> dict:
    prepare(output)
    if not 1 <= through_episode <= p.MAX_EPISODE:
        raise ValueError(f"through-episode must be in [1,{p.MAX_EPISODE}]")
    if device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("E115 training requires CUDA")
    args = SimpleNamespace(
        output=output.resolve(),
        scenario=p.SCENARIO,
        model_seed=0,
        device=device,
        stop_after_episode=through_episode,
    )
    with ExitStack() as stack:
        stack.enter_context(patch.object(
            dense, "FROZEN_OBJECTIVE_SPEC", p.e107.e105.e78.SPEC
        ))
        stack.enter_context(patch.object(dense, "FROZEN_GAMMA", 1.0))
        stack.enter_context(patch.object(
            audit, "run_objective_audit_episode", instrumentation.run_episode
        ))
        stack.enter_context(patch.object(
            audit, "summarize_objective_runs", instrumentation.summarize_runs
        ))
        stack.enter_context(patch.object(
            audit, "_validation_record", _development_validation_record
        ))
        stack.enter_context(fast_inputs.enabled())
        result = _train_factorized(args, p.trainer_protocol())
    return {
        **result,
        "e115_protocol": p.PROTOCOL,
        "architecture_name": p.ARCHITECTURE_NAME,
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
        "training_learning_rate": training.get("learning_rate_after"),
        "strict_completion_rate": summary["strict_method_success_rate"],
        "operator_cost": summary["mean_operator_cost"],
        "return": summary["mean_selected_objective_return"],
        "mean_wait_before_storage": summary["mean_wait_before_storage"],
        "release_clock_mae": summary["mean_release_clock_mae"],
        "release_clock_signed_error": summary["mean_release_clock_signed_error"],
        "release_tardiness": summary["mean_release_tardiness"],
        "release_earliness": summary["mean_release_earliness"],
        "within_tolerance_rate": summary["mean_within_tolerance_rate"],
        "storage_clock_mae": summary["mean_storage_clock_mae"],
        "steps": summary["mean_steps"],
        "equipment_work": summary["mean_equipment_work"],
        "equipment_work_per_delivery": summary[
            "mean_equipment_work_per_delivery"
        ],
        "rehandles_per_100": summary["relocations_per_100_deliveries"],
        "guard_activations": summary["total_liveness_guard_activations"],
        "component_accounting_exact": summary["component_accounting_exact"],
    }


def _gate(row: dict) -> dict:
    checks = {
        "strict_completion": float(row["strict_completion_rate"]) == 1.0,
        "finite_operator_cost": math.isfinite(float(row["operator_cost"])),
        "component_accounting_exact": bool(row["component_accounting_exact"]),
    }
    return {"passed": all(checks.values()), "checks": checks}


def _rolling_window(rows: list[dict], endpoint: int, width: int = 3):
    positions = [
        index for index, row in enumerate(rows)
        if row["checkpoint_episode"] == endpoint
    ]
    if not positions:
        return None
    stop = positions[0] + 1
    window = rows[max(0, stop - width):stop]
    numeric = (
        "strict_completion_rate",
        "operator_cost",
        "return",
        "mean_wait_before_storage",
        "release_clock_mae",
        "release_clock_signed_error",
        "release_tardiness",
        "release_earliness",
        "within_tolerance_rate",
        "storage_clock_mae",
        "steps",
        "equipment_work",
        "equipment_work_per_delivery",
        "rehandles_per_100",
        "guard_activations",
    )

    def mean_or_none(name):
        values = [row.get(name) for row in window]
        if any(value is None for value in values):
            return None
        return sum(float(value) for value in values) / len(values)

    return {
        "checkpoint_episode": endpoint,
        "source_checkpoints": [row["checkpoint_episode"] for row in window],
        "window_width": len(window),
        **{
            name: mean_or_none(name)
            for name in numeric
        },
        "component_accounting_exact": all(
            bool(row["component_accounting_exact"]) for row in window
        ),
    }


def analyze(output: Path) -> dict:
    prepare(output)
    summary_path = destination(output) / "training-summary.json"
    if summary_path.is_file():
        saved = read_json(summary_path)
        training_history = list(saved.get("training", {}).get("runs", ()))
    else:
        latest = destination(output) / "latest.pth"
        if not latest.is_file():
            raise RuntimeError("train E115 before analyze")
        payload = torch.load(latest, map_location="cpu", weights_only=False)
        saved = {
            "completed_training_episodes": payload["completed_training_episodes"],
            "training_history": payload["training_history"],
            "validation_history": payload["validation_history"],
            "e115_agent_audit": _agent_audit(
                FactorizedConservativeAgent.from_checkpoint(
                    payload, device="cpu", resumable=False, seed=0
                )
            ),
        }
        training_history = list(saved.get("training_history", ()))
    rows = [
        _row(record, training_history)
        for record in saved.get("validation_history", ())
    ]
    point_gates = {
        str(episode): _gate(row)
        for episode in p.DECISION_CHECKPOINTS
        for row in rows
        if row["checkpoint_episode"] == episode
    }
    rolling_windows = {
        str(episode): window
        for episode in p.DECISION_CHECKPOINTS
        for window in (_rolling_window(rows, episode),)
        if window is not None
    }
    rolling_gates = {
        episode: _gate(window)
        for episode, window in rolling_windows.items()
    }
    completed = int(saved["completed_training_episodes"])
    report = {
        "protocol": p.PROTOCOL,
        "architecture_name": p.ARCHITECTURE_NAME,
        "status": "complete" if completed == p.MAX_EPISODE else "partial",
        "completed_training_episodes": completed,
        "maximum_episode": p.MAX_EPISODE,
        "fresh_initialization": True,
        "epsilon_decay_decisions": p.EPSILON_DECAY_DECISIONS,
        "decision_checkpoints": list(p.DECISION_CHECKPOINTS),
        "rows": rows,
        "point_checkpoint_gates": point_gates,
        "rolling_three_validation_windows": rolling_windows,
        "decision_checkpoint_gates": rolling_gates,
        "agent_audit": saved.get("e115_agent_audit"),
        "first_joint_gate_checkpoint": next(
            (
                int(key)
                for key, value in rolling_gates.items()
                if value["passed"]
            ),
            None,
        ),
        "interpretation": (
            "Fresh release-clock earliness--tardiness cost training on a feasible "
            "arrival process. The mechanism gate checks completion and exact "
            "accounting only; quality requires matched baselines. Conservative "
            "anchoring is an anti-drift trust region, not an oracle label. One "
            "seed and a reused validation panel are not paper evidence."
        ),
        "stage_b_authorized": False,
        "independent_test_opened": False,
        "paper_evidence": False,
        "episode_runs": 0,
        "training_runs": 0,
    }
    report_path = output.resolve() / "e115-report.json"
    write_json(report_path, report)
    return {
        "status": report["status"],
        "completed_training_episodes": completed,
        "maximum_episode": p.MAX_EPISODE,
        "validation_checkpoints": [row["checkpoint_episode"] for row in rows],
        "point_checkpoint_gates": point_gates,
        "decision_checkpoint_gates": rolling_gates,
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
