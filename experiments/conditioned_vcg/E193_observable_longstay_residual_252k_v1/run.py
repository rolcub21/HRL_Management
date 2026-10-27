#!/usr/bin/env python3
"""Prepare, verify, train, and summarize E193."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import fields
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from PSLAP.viability import RecoveryState, ViabilityStatus
from viability_graph_episodic_audit import EpisodicViabilityGraphConfig
from viability_graph_hierarchy import (
    ACTION_FEATURE_DIM,
    ViabilityGraphCandidateRecord,
)
from experiments.conditioned_vcg.E187_wait_corrected_e78_246k_v1 import (
    run as e187_run,
)
from experiments.conditioned_vcg.E191_fresh_long_stay_crowding_250k_v1 import (
    scenario,
)
from experiments.conditioned_vcg.E191_fresh_long_stay_crowding_250k_v1.roundoff import (
    repaired_audit,
)

from . import defer, instrumentation, protocol as p, representation, residual, stage


class E193Error(RuntimeError):
    pass


BUDGET_RULE = (
    "fixed_300_episodes_fresh_observable_event_residual_longstay_three_workloads"
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise E193Error(message)


@contextmanager
def implementation():
    """Install E193 only at E187's explicit trainer extension seams."""

    # E187's accelerated TD path enters this imported context manager by name.
    # Rebinding it is essential: the E193 scorer caches total Q=K+residual,
    # rather than silently caching the residual network output alone.
    with (
        patch.object(e187_run, "p", p),
        patch.object(
            e187_run,
            "WaitCorrectedQueueAgent",
            residual.ObservableResidualAgent,
        ),
        patch.object(e187_run, "instrumentation", instrumentation),
        patch.object(e187_run, "BUDGET_RULE", BUDGET_RULE),
        patch.object(e187_run, "authenticate_sources", lambda: {}),
        patch.object(e187_run.v2run, "train_stage_a", stage.train_stage_a),
        patch.object(
            e187_run.e107_representation,
            "acceleration_enabled",
            residual.acceleration_enabled,
        ),
    ):
        yield


def prepare(output: Path) -> dict:
    with implementation():
        value = e187_run.prepare(output)
    return {
        **value,
        "protocol": p.PROTOCOL,
        "fresh_random_initialization": True,
        "workload_identical_to_e191": True,
        "workload_levels": list(scenario.LEVELS),
        "learning_interface": {
            "observable_arrival_context": True,
            "longstay_timing_scale": representation.TIMING_SCALE,
            "event_defer": defer.DEFER_VERSION,
            "nominal_cost_residual": residual.RESIDUAL_VALUE_CONTRACT,
        },
    }


def authenticate(output: Path) -> dict:
    with implementation():
        return e187_run.authenticate(output)


def _state_equal(first: dict, second: dict) -> bool:
    return set(first) == set(second) and all(
        torch.equal(first[name], second[name]) for name in first
    )


def _probe_state() -> RecoveryState:
    return RecoveryState(
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


def self_check(output: Path) -> dict:
    """Run only construction, information-boundary, and algebra checks."""

    prepared = prepare(output)
    contract = authenticate(output)
    parent = p.e191.contract()
    require(
        contract["parent_contract_sha256"] == parent["contract_sha256"],
        "E193 no longer authenticates the E191 workload parent",
    )
    require(
        contract["controlled_constants"]["objective"] == parent["objective"]
        and contract["controlled_constants"]["workload"] == parent["workload"],
        "E193 changed the E191 workload or objective",
    )
    require(
        representation.QUEUE_CONTEXT_DIM == 10
        and representation.TIMING_SCALE == scenario.MEAN_DURATION,
        "E193 representation scale or public-context width differs",
    )

    config = representation.with_longstay_timing_scale(
        EpisodicViabilityGraphConfig(
            gamma=p.GAMMA,
            episode_horizon_steps=p.MAX_STEPS,
        )
    )
    fresh = residual.ObservableResidualAgent(
        config=config, seed=0, device="cpu", epsilon=p.EPSILON_START
    )
    require(
        fresh.e187_warm_start_manifest is None
        and len(fresh.replay) == 0
        and fresh.decision_count == 0
        and fresh.gradient_steps == 0
        and not fresh.optimizer.state,
        "E193 did not start with completely fresh training state",
    )
    payload = fresh.checkpoint(
        include_replay=True,
        training_protocol=p.PROTOCOL,
        future_schedule_visible_to_policy=False,
    )
    restored = residual.ObservableResidualAgent.from_checkpoint(
        payload, device="cpu", resumable=True, seed=0
    )
    require(
        _state_equal(fresh.Q_local.state_dict(), restored.Q_local.state_dict())
        and restored.e187_warm_start_manifest is None,
        "E193 fresh residual checkpoint did not round-trip exactly",
    )

    env = p.TargetDenseEnv(p.SCENARIO, p.SPEC)
    cases = []
    contexts_at_zero = set()
    cadence_features = {}
    for seed in scenario.PILOT_SEEDS:
        instance = env.sample_episode_instance(seed)
        metadata = scenario.workload_metadata(instance)
        env.reset(instance=instance)
        snapshot = SimpleNamespace(decision_epoch=0, inbound_label="B1")
        initial = representation.public_longstay_context(
            env,
            snapshot,
            required_jobs=scenario.TOTAL_JOBS,
            max_steps=p.MAX_STEPS,
        )
        contexts_at_zero.add(initial.values)
        for _ in range(representation.ARRIVAL_WINDOW_STEPS):
            env.step(env.ACTION_IDS["WAIT"])
        observed = representation.public_longstay_context(
            env,
            SimpleNamespace(decision_epoch=env.time_steps, inbound_label=None),
            required_jobs=scenario.TOTAL_JOBS,
            max_steps=p.MAX_STEPS,
        )
        label = metadata["target_label"]
        cadence_features[label] = observed.values[-2]
        cases.append({
            "seed": seed,
            "target": metadata["target_nominal_occupancy"],
            "gap": metadata["arrival_gap_steps"],
            "initial_context": list(initial.values),
            "observed_rate_feature_at_step_200": observed.values[-2],
        })
    require(
        len(contexts_at_zero) == 1,
        "E193 leaked the workload through the initial observation",
    )
    require(
        list(cadence_features.values()) == sorted(cadence_features.values())
        and len(set(cadence_features.values())) == len(scenario.LEVELS),
        "E193 observed cadence does not distinguish the workloads after history",
    )

    class Block:
        def __init__(
            self,
            label,
            *,
            position=None,
            stored=False,
            delivered=False,
            stored_time_step=None,
            storage_steps_needed=700,
        ):
            self.label = label
            self.position = position
            self.stored = stored
            self.delivered = delivered
            self.carrying = False
            self.stored_time_step = stored_time_step
            self.storage_steps_needed = storage_steps_needed

    fake_env = SimpleNamespace(
        time_steps=100,
        blocks=(
            Block(
                "seen",
                position=(1, 0),
                stored=True,
                stored_time_step=50,
            ),
            Block("unseen"),
        ),
    )
    rule = defer.ObservableLongStayDeferRule(
        episode_horizon_steps=p.MAX_STEPS,
        delivery_window=20,
    )
    decision = rule.evaluate(
        fake_env,
        None,
        inbound_present=True,
        current_certificate=SimpleNamespace(status=ViabilityStatus.SAFE),
        consecutive_defer_decisions=10_000,
        terminal=False,
    )
    require(
        decision.allowed
        and decision.horizon_steps == 630
        and "observed_release" in decision.reason,
        "E193 Defer is not the next observable release/storage-window event",
    )

    base = _probe_state()
    current = representation.attach_longstay_context(
        base, (0.0,) * representation.QUEUE_CONTEXT_DIM
    )
    base_record = ViabilityGraphCandidateRecord(
        key="accept:B1:1:0",
        mode_id=0,
        action_type="accept",
        current_state=current,
        successor_state=base,
        action_features=(0.0,) * ACTION_FEATURE_DIM,
    )
    record = residual.ObservableCostRecord(
        **{
            item.name: getattr(base_record, item.name)
            for item in fields(ViabilityGraphCandidateRecord)
        },
        nominal_components=(1.0, 2.0, 3.0, 4.0, 5.0),
        nominal_duration=1,
    )
    like = torch.zeros(1)
    known = residual.known_q_tensor(
        (record,), like, reward_scale=config.reward_scale
    )
    require(
        torch.allclose(
            known,
            torch.tensor([-config.reward_scale * record.nominal_J]),
        ),
        "E193 known immediate cost has the wrong sign or scale",
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
        "E193 changed the primitive objective",
    )
    return {
        "status": "passed",
        "contract_sha256": prepared["contract_sha256"],
        "representation": contract["representation"],
        "defer_contract": contract["defer_contract"],
        "value_learning": contract["value_learning"],
        "fresh_initialization": contract["initialization"],
        "validation_cases": cases,
        "checkpoint_roundtrip": True,
        "initial_observation_hides_workload": True,
        "observed_history_distinguishes_cadence": True,
        "event_defer_probe_horizon": decision.horizon_steps,
        "known_cost_probe_q": float(known.item()),
        "primitive_wait_probe_reward": reward,
        "accelerations": contract["acceleration_assertion"],
        "episode_runs": 0,
        "training_runs": 0,
    }


def train_seed(output: Path, seed: int, through_episode: int, device: str) -> dict:
    prepare(output)
    contract = authenticate(output)
    require(seed in p.MODEL_SEEDS, "E193 model seed must be 0, 1, or 2")
    require(
        1 <= through_episode <= p.TOTAL_EPISODES,
        "E193 through-episode must be between 1 and 300",
    )
    require(
        device == "cuda" and torch.cuda.is_available(),
        "E193 training requires CUDA",
    )
    with implementation(), repaired_audit():
        result = e187_run.train_seed(output, seed, through_episode, device)
    result.pop("e187_protocol", None)
    return {
        **result,
        "e193_protocol": p.PROTOCOL,
        "e193_contract_sha256": contract["contract_sha256"],
        "fresh_random_initialization": True,
        "workload_identical_to_e191": True,
        "observable_arrival_context": True,
        "event_defer": True,
        "nominal_cost_residual": True,
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
            row["by_workload"] = record["summary"].get("e191_by_workload", {})
            row["mean_stored"] = record["summary"].get("mean_e191_stored")
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
            f"E193 seed {seed} validation coverage differs",
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
        "representation": contract["representation"],
        "defer_contract": contract["defer_contract"],
        "value_learning": contract["value_learning"],
        "initialization": contract["initialization"],
        "workload": contract["workload"],
        "seeds": seeds,
        "test_panel_opened": False,
        "paper_evidence": False,
    }
    path = output / "e193-report.json"
    e187_run._write(path, report)
    return {"status": report["status"], "report": str(path)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("prepare", "self-check", "train-seed", "status", "analyze"),
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
