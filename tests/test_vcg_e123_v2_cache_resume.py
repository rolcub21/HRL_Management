from dataclasses import replace
from copy import deepcopy

import torch

from experiments.conditioned_vcg.E100_accelerated_release_clock_163k_v1 import fast_inputs
from experiments.conditioned_vcg.E115_release_et_cost_vcg_8x8_179k_v1.model import FactorizedTransition
from experiments.conditioned_vcg.E123_absolute_progress_8x8_187k_v1.self_check import toy_record
from experiments.conditioned_vcg.E123_absolute_progress_8x8_187k_v1.model import ProgressAgent, attach_progress
from experiments.conditioned_vcg.E123_absolute_progress_8x8_187k_v2.acceleration import (
    share_records, compact_transitions, cache_session, SharedProgressAgent, INPUT_CAPACITY)
from experiments.conditioned_vcg.E123_absolute_progress_8x8_187k_v2.run import controller
from experiments.conditioned_vcg.E123_absolute_progress_8x8_187k_v2.verification import equal_values
from viability_graph_episodic_audit import EpisodicViabilityGraphConfig


def records():
    first = toy_record(10.)
    return tuple(replace(first, key=f"candidate-{i}", current_state=attach_progress(
        first.current_state, first.current_state.progress_context)) for i in range(5))


def test_sharing_changes_only_identity_and_preserves_order():
    original = records()
    assert len({id(r.current_state) for r in original}) == 5
    shared = share_records(original)
    assert original == shared
    assert len({id(r.current_state) for r in shared}) == 1
    assert [r.key for r in original] == [r.key for r in shared]


def test_progress_queue_and_block_clocks_cannot_be_merged():
    from PSLAP.dynamic_yard import BlockView
    first = toy_record(10.)
    first = replace(first, current_state=replace(first.current_state, blocks=(BlockView("B", (3, 3), 30.),)))
    other_progress = replace(first, current_state=attach_progress(first.current_state, (11/3600., 1., 0., 1.)))
    other_queue = replace(first, current_state=replace(first.current_state, queue_context=(.5,) + (0.,)*7))
    other_block = replace(first, current_state=replace(first.current_state, blocks=tuple(
        replace(b, remaining_time=b.remaining_time+1) for b in first.current_state.blocks)))
    shared = share_records((first, other_progress, other_queue, other_block))
    assert len({id(r.current_state) for r in shared}) == 4


def test_replay_compaction_preserves_values_terminal_masks_and_components():
    items = records()
    transition = FactorizedTransition(chosen=items[0], reward=-17.4, duration=3, next_candidates=items,
                                      done=False, component_returns=(2., 3., 4., 1.))
    terminal = replace(transition, done=True, next_candidates=())
    compacted, stats = compact_transitions((transition, terminal))
    assert compacted == [transition, terminal]
    assert stats["shared_current_state_objects"] == 1


def test_cache_survives_controller_boundary_but_not_arm_exit():
    agent = SharedProgressAgent(seed=0)
    with cache_session():
        with controller("A_outflow_bounded"):
            cache = fast_inputs._cache_for(agent.Q_local)
            cache.slots_for((toy_record().current_state,))
            count = cache.size
            assert cache.capacity == INPUT_CAPACITY
        with controller("A_outflow_bounded"):
            assert fast_inputs._cache_for(agent.Q_local) is cache and cache.size == count
    assert not fast_inputs._REGISTRY


def test_input_tensors_component_scores_targets_and_gradients_equal():
    config = EpisodicViabilityGraphConfig(gamma=1., episode_horizon_steps=3600, batch_size=2)
    original = ProgressAgent(config=config, seed=3)
    corrected = SharedProgressAgent(config=config, seed=3)
    # Nonconstant learned-style heads, so equality is not a zero-head tautology.
    with torch.no_grad():
        for head in (original.Q_local.service_head, original.Q_local.wait_head,
                     original.Q_local.step_head, original.Q_local.rehandle_head):
            head.weight.fill_(.02)
    original.Q_target.load_state_dict(original.Q_local.state_dict())
    corrected.Q_local.load_state_dict(original.Q_local.state_dict())
    corrected.Q_target.load_state_dict(original.Q_target.state_dict())
    items = records()
    shared = share_records(items)
    with cache_session(), controller("A_outflow_bounded"):
        a = fast_inputs._cache_for(original.Q_local).batch(tuple(r.current_state for r in items))
        b = fast_inputs._cache_for(corrected.Q_local).batch(tuple(r.current_state for r in shared))
        assert all(torch.equal(x, y) for x, y in zip(a, b))
        assert torch.equal(original._component_scores(items), corrected._component_scores(shared))
    transition = FactorizedTransition(chosen=items[0], reward=-17.4, duration=3, next_candidates=items,
                                      done=False, component_returns=(2., 3., 4., 1.))
    comp, _ = compact_transitions((transition, replace(transition, done=True, next_candidates=())))
    for transition in (transition, replace(transition, done=True, next_candidates=())):
        original.replay.add(transition)
    for transition in comp:
        corrected.replay.add(transition)
    with cache_session(), controller("A_outflow_bounded"):
        old_loss = original.learn()
    with cache_session(), controller("A_outflow_bounded"):
        new_loss = corrected.learn()
    assert old_loss == new_loss
    assert equal_values(original.checkpoint_state(include_replay=False), corrected.checkpoint_state(include_replay=False))
    restored = SharedProgressAgent.from_checkpoint(corrected.checkpoint(), resumable=True, seed=3)
    assert equal_values(corrected.checkpoint_state(include_replay=False), restored.checkpoint_state(include_replay=False))


def test_independent_resume_probes_with_nonempty_adam_state():
    config = EpisodicViabilityGraphConfig(gamma=1., episode_horizon_steps=3600, batch_size=2)
    source = ProgressAgent(config=config, seed=3)
    item = FactorizedTransition(chosen=toy_record(), reward=-17.4, duration=3, next_candidates=(),
                               done=True, component_returns=(2., 3., 4., 1.))
    source.replay.add(item)
    source.replay.add(item)
    with cache_session(), controller("A_outflow_bounded"):
        source.learn()
    payload = source.checkpoint()
    baseline = ProgressAgent.from_checkpoint(payload, resumable=True, seed=3)
    corrected = SharedProgressAgent.from_checkpoint(payload, resumable=True, seed=3)
    baseline.optimizer.load_state_dict(deepcopy(payload["agent_state"]["optimizer"]))
    corrected.optimizer.load_state_dict(deepcopy(payload["agent_state"]["optimizer"]))
    with cache_session(), controller("A_outflow_bounded"):
        baseline.learn()
        corrected.learn()
    assert equal_values(baseline.checkpoint_state(include_replay=False), corrected.checkpoint_state(include_replay=False))
