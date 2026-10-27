from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from PSLAP.viability import RecoveryState, ViabilityStatus
from viability_graph_episodic_audit import EpisodicViabilityGraphConfig
from viability_graph_hierarchy import (
    ACTION_FEATURE_DIM,
    ViabilityGraphTransition,
)
from experiments.conditioned_vcg.E191_fresh_long_stay_crowding_250k_v1 import (
    protocol as e191,
    scenario,
)
from experiments.conditioned_vcg.E193_observable_longstay_residual_252k_v1 import (
    defer,
    protocol,
    representation,
    residual,
    run,
    stage,
)
from experiments.conditioned_vcg.target_size_dense_stage_a_v3 import (
    acceleration as v3_acceleration,
)


def _config(**updates):
    values = {
        "gamma": protocol.GAMMA,
        "episode_horizon_steps": protocol.MAX_STEPS,
        "timing_scale": representation.TIMING_SCALE,
    }
    values.update(updates)
    return EpisodicViabilityGraphConfig(**values)


def _state(*, context=False):
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
    if context:
        return representation.attach_longstay_context(
            state, (0.0,) * representation.QUEUE_CONTEXT_DIM
        )
    return state


def _cost_record(*, key="accept:B1:1:0", components=(1, 2, 3, 4, 5)):
    return residual.ObservableCostRecord(
        key=key,
        mode_id=0,
        action_type="accept",
        current_state=_state(context=True),
        successor_state=_state(),
        action_features=(0.0,) * ACTION_FEATURE_DIM,
        nominal_components=tuple(float(value) for value in components),
        nominal_duration=1,
    )


class _ContextBlock:
    def __init__(
        self,
        label,
        arrival,
        *,
        position=None,
        stored_time=None,
        delivered=False,
        duration=700,
    ):
        self.label = label
        self.arrival_step = arrival
        self.position = position
        self.stored_time_step = stored_time
        self.stored = stored_time is not None
        self.delivered = delivered
        self.carrying = False
        self.storage_steps_needed = duration


class _DeferEnv:
    ACTION_IDS = {"WAIT": 6}

    def __init__(self, *, now=200, future=True):
        self.time_steps = now
        self.blocks = [
            _ContextBlock(
                "B1", 0, position=(1, 1), stored_time=100, duration=700
            )
        ]
        if future:
            self.blocks.append(_ContextBlock("B2", 2_000))
        self.current_state = object()
        self.current_episode_instance = SimpleNamespace(instance_id="e193-test")

    def is_state_terminal(self, _state):
        return False


def _evaluate(env, *, inbound=True, streak=10_000, terminal=False):
    return defer.ObservableLongStayDeferRule(
        episode_horizon_steps=9_000,
        delivery_window=20,
    ).evaluate(
        env,
        SimpleNamespace(),
        inbound_present=inbound,
        current_certificate=SimpleNamespace(status=ViabilityStatus.SAFE),
        consecutive_defer_decisions=streak,
        terminal=terminal,
    )


def test_contract_changes_learning_interface_not_scenario_or_objective():
    value = protocol.contract()
    parent = e191.contract()
    assert value["controlled_constants"]["objective"] == parent["objective"]
    assert value["controlled_constants"]["workload"] == parent["workload"]
    assert value["controlled_constants"]["progress_guard"] == (
        parent["frontier"]["progress_guard"]
    )
    assert value["representation"]["queue_context_dim"] == 10
    assert value["representation"]["observed_arrival_history"] is True
    assert value["representation"]["future_schedule_visible_to_policy"] is False
    assert value["representation"]["graph_timing_scale"] == 700
    assert value["defer_contract"]["numeric_consecutive_defer_cap"] is None
    assert value["value_learning"]["contract"] == (
        residual.RESIDUAL_VALUE_CONTRACT
    )
    assert all(value["acceleration_assertion"].values())


def test_workload_is_hidden_initially_then_inferred_from_observed_history():
    initial = []
    later = {}
    for seed in scenario.PILOT_SEEDS:
        env = protocol.TargetDenseEnv(protocol.SCENARIO, protocol.SPEC)
        env.reset(instance=env.sample_episode_instance(seed))
        context = representation.public_longstay_context(
            env,
            SimpleNamespace(decision_epoch=0, inbound_label="B1"),
        )
        initial.append(context.values)
        for _ in range(representation.ARRIVAL_WINDOW_STEPS):
            env.step(env.ACTION_IDS["WAIT"])
        context = representation.public_longstay_context(
            env,
            SimpleNamespace(decision_epoch=env.time_steps, inbound_label=None),
        )
        later[scenario.target_level(seed)] = context.values[8]

    assert len(set(initial)) == 1
    assert later == pytest.approx({0.20: 0.25, 0.50: 0.625, 0.80: 1.0})


def test_public_context_is_invariant_to_unobserved_future_timestamps():
    def context(future_arrivals):
        env = SimpleNamespace(
            time_steps=120,
            blocks=(
                _ContextBlock("B1", 0, position=(1, 1), stored_time=10),
                _ContextBlock("B2", 40, position=(1, 0)),
                *(
                    _ContextBlock(f"B{index + 3}", arrival)
                    for index, arrival in enumerate(future_arrivals)
                ),
            ),
        )
        return representation.public_longstay_context(
            env,
            SimpleNamespace(decision_epoch=120, inbound_label="B2"),
            required_jobs=4,
            max_steps=9_000,
        )

    early = context((200, 300))
    late = context((2_000, 3_000))
    assert early == late
    assert all(value <= 120 for value in early.observed_arrival_times)


def test_longstay_timing_scale_is_mandatory_and_checkpointed():
    default = EpisodicViabilityGraphConfig(
        gamma=protocol.GAMMA,
        episode_horizon_steps=protocol.MAX_STEPS,
    )
    corrected = representation.with_longstay_timing_scale(default)
    assert corrected.timing_scale == 700
    with pytest.raises(ValueError, match="timing_scale=700"):
        residual.ObservableResidualAgent(
            config=default, seed=0, device="cpu", epsilon=0.0
        )

    agent = residual.ObservableResidualAgent(
        config=corrected, seed=0, device="cpu", epsilon=0.0
    )
    payload = agent.checkpoint(
        include_replay=True,
        training_protocol=protocol.PROTOCOL,
        future_schedule_visible_to_policy=False,
    )
    assert payload["timing_scale"] == 700
    assert payload["e193_residual_value_contract"] == (
        residual.RESIDUAL_VALUE_CONTRACT
    )


def test_event_defer_allows_inbound_and_ignores_obsolete_streak_cap():
    decision = _evaluate(_DeferEnv(), inbound=True, streak=10_000)
    assert decision.allowed
    # Stored at 100, duration 700, window 20: opens at 780 from now=200.
    assert decision.horizon_steps == 580
    assert decision.reason == "wait_for_storage_window_or_observed_release"


def test_event_defer_uses_error_boundary_only_for_unknown_release():
    env = _DeferEnv(now=200)
    env.blocks[0].delivered = True
    env.blocks[0].stored = False
    env.blocks[0].stored_time_step = None
    decision = _evaluate(env)
    assert decision.allowed
    assert decision.horizon_steps == 8_800
    assert decision.reason == "wait_for_observed_release"

    env.blocks[1].position = (0, 0)
    exhausted = _evaluate(env)
    assert not exhausted.allowed
    assert exhausted.reason == "no_future_public_event"


def test_defer_option_stops_when_a_release_becomes_observable():
    env = _DeferEnv(now=200)
    option = defer.ObservableLongStayDeferOption(env, horizon_steps=580)
    assert option.initiation(env.current_state)
    assert option.policy(env.current_state) == env.ACTION_IDS["WAIT"]
    env.blocks[1].position = (0, 0)
    env.time_steps += 1
    assert option.termination(env.current_state)
    assert option.last_outcome["reason"] == "observed_event"


def test_known_nominal_cost_has_negative_reward_sign_and_reward_scale():
    record = _cost_record(components=(1, 2, 3, 4, 5))
    like = torch.zeros(1)
    known = residual.known_q_tensor([record], like, reward_scale=0.01)
    # 2*1 + 1*2 + .5*3 + .1*4 + 10*5 = 55.9 cost.
    assert known.item() == pytest.approx(-0.559)


def test_native_and_accelerated_total_q_are_identical():
    agent = residual.ObservableResidualAgent(
        config=_config(), seed=0, device="cpu", epsilon=0.0
    )
    records = (
        _cost_record(key="a", components=(1, 2, 3, 4, 5)),
        _cost_record(key="b", components=(0, 1, 0, 2, 0)),
    )
    with torch.no_grad():
        native = agent._score_records(records, network=agent.Q_local)
        accelerated = residual.deduplicated_total_scores(
            records, agent.Q_local
        )
    assert torch.allclose(native, accelerated, rtol=0.0, atol=1.0e-6)

    original = v3_acceleration._deduplicated_no_grad_scores
    with residual.acceleration_enabled():
        assert (
            v3_acceleration._deduplicated_no_grad_scores
            is residual.deduplicated_total_scores
        )
    assert v3_acceleration._deduplicated_no_grad_scores is original


def test_terminal_td_does_not_double_count_known_nominal_cost():
    agent = residual.ObservableResidualAgent(
        config=_config(), seed=0, device="cpu", epsilon=0.0
    )
    transition = ViabilityGraphTransition(
        chosen=_cost_record(),
        reward=-55.9,
        duration=1,
        next_candidates=(),
        done=True,
    )
    prediction, target = agent._td_batch((transition,))
    assert prediction.requires_grad
    assert target.item() == pytest.approx(-0.559)


def test_residual_checkpoint_roundtrip_preserves_replay_cost_records():
    agent = residual.ObservableResidualAgent(
        config=_config(), seed=0, device="cpu", epsilon=0.0
    )
    transition = ViabilityGraphTransition(
        chosen=_cost_record(),
        reward=-1.0,
        duration=1,
        next_candidates=(),
        done=True,
    )
    agent.replay.add(transition)
    payload = agent.checkpoint(
        include_replay=True,
        training_protocol=protocol.PROTOCOL,
        future_schedule_visible_to_policy=False,
    )
    restored = residual.ObservableResidualAgent.from_checkpoint(
        payload, device="cpu", resumable=True, seed=0
    )
    assert isinstance(
        restored.replay.memory[0].chosen, residual.ObservableCostRecord
    )
    for name, value in agent.Q_local.state_dict().items():
        assert torch.equal(value, restored.Q_local.state_dict()[name])


def test_training_adapter_installs_event_defer_and_longstay_encoding(
    tmp_path, monkeypatch
):
    observed = {}

    def fake_main(argv):
        runtime = SimpleNamespace(
            output_dir=tmp_path,
            model_seed=0,
            device="cuda",
            stop_after_episode=0,
            log_every=1,
        )
        args = stage.dense._frozen_audit_args(runtime)
        observed["config"] = stage.audit._graph_config(args, protocol.GAMMA)
        observed["rule"] = stage.audit._liveness_rule(args)
        observed["option"] = stage.frontier.ExplicitDeferOption
        return {
            "status": "partial",
            "completed_training_episodes": 0,
            "latest_checkpoint": None,
            "best_checkpoint": None,
            "deployment_checkpoint_eligible": False,
        }

    monkeypatch.setattr(stage.dense, "main", fake_main)
    monkeypatch.setattr(run.torch.cuda, "is_available", lambda: True)
    result = run.train_seed(tmp_path, 0, 1, "cuda")
    assert result["fresh_random_initialization"] is True
    assert observed["config"].timing_scale == 700
    assert isinstance(observed["rule"], defer.ObservableLongStayDeferRule)
    assert observed["option"] is defer.ObservableLongStayDeferOption


def test_run_implementation_routes_cached_td_through_total_q():
    original = run.e187_run.e107_representation.acceleration_enabled
    with run.implementation():
        assert (
            run.e187_run.e107_representation.acceleration_enabled
            is residual.acceleration_enabled
        )
    assert run.e187_run.e107_representation.acceleration_enabled is original


def test_self_check_is_non_episodic_and_idempotent(tmp_path):
    first = run.self_check(tmp_path)
    second = run.self_check(tmp_path)
    assert first["status"] == second["status"] == "passed"
    assert first["contract_sha256"] == second["contract_sha256"]
    assert first["episode_runs"] == second["episode_runs"] == 0

