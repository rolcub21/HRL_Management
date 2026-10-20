from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from PSLAP.viability import RecoveryActionKind, ViabilityStatus
from experiments.conditioned_vcg.E123_absolute_progress_8x8_187k_v1 import liveness as live
from experiments.conditioned_vcg.E123_absolute_progress_8x8_187k_v1.model import ProgressAgent, attach_progress
from experiments.conditioned_vcg.E123_absolute_progress_8x8_187k_v1.self_check import candidate, checks, toy_record


def env(now, clock=None, live_work=True):
    block = SimpleNamespace(delivered=False, stored=False, carrying=False,
                            position=(1, 6) if live_work else None)
    return SimpleNamespace(progress_clock=clock or live.ProgressClock(), time_steps=now, blocks=[block])


def defer(environment, streak=0):
    return live.AbsoluteProgressDeferRule().evaluate(environment, SimpleNamespace(blocks=()),
        inbound_present=True, current_certificate=SimpleNamespace(status=ViabilityStatus.SAFE),
        consecutive_defer_decisions=streak, terminal=False)


def test_clock_does_not_reset_when_arrivals_reset_legacy_defer_streak():
    clock = live.ProgressClock()
    for now in range(1, 121):
        clock.observe(now, 0, 0)
    assert clock.age(120) == 120 and clock.events == 0
    assert not defer(env(120, clock), streak=0).allowed


def test_only_monotone_actual_admission_or_delivery_resets_clock():
    clock = live.ProgressClock()
    clock.observe(30, 1, 0)
    clock.observe(60, 1, 0)  # rehandle or arrival
    assert clock.last_progress_time == 30
    clock.observe(68, 1, 1)
    assert clock.last_progress_time == 68 and clock.events == 2
    with pytest.raises(RuntimeError, match="monotone"):
        clock.observe(70, 0, 1)


@pytest.mark.parametrize("now,horizon", [(0, 10), (21, 9), (29, 1)])
def test_wait_horizon_is_capped_by_absolute_remaining_time(now, horizon):
    assert defer(env(now)).horizon_steps == horizon


def test_deadline_has_no_due_date_or_arrival_exception_with_available_work():
    for now in (30, 31, 1000):
        assert not defer(env(now)).allowed
    assert defer(env(30, live_work=False)).reason == "external_idle_no_released_work"


def test_all_nonprogress_choices_removed_at_deadline_not_only_defer():
    items = (candidate("accept", 10), candidate("reconfigure", 1), candidate("defer", 1), candidate("deliver", 10))
    snapshot = SimpleNamespace(candidates=items, decision_epoch=30)
    prepared = SimpleNamespace(source_indices=(0, 1, 2, 3), records=(0, 1, 2, 3))
    assert live.eligible_indices(snapshot, prepared, env(30), False) == (0, 3)


def test_long_nonprogress_macro_cannot_overshoot_deadline():
    items = (candidate("reconfigure", 6), candidate("defer", 5), candidate("accept", 20))
    snapshot = SimpleNamespace(candidates=items, decision_epoch=25)
    prepared = SimpleNamespace(source_indices=(0, 1, 2), records=(0, 1, 2))
    assert live.eligible_indices(snapshot, prepared, env(25), False) == (1, 2)


def test_progress_macro_budget_is_certified_not_observed_average():
    assert live.direct_progress(candidate("deliver", 20))
    assert not live.direct_progress(candidate("deliver", 21))


def test_committed_prefix_may_require_multiple_macros():
    relocation = SimpleNamespace(kind=RecoveryActionKind.RELOCATION, steps=7)
    delivery = SimpleNamespace(kind=RecoveryActionKind.DELIVERY, steps=12)
    certificate = SimpleNamespace(status=ViabilityStatus.SAFE, witness=(relocation, delivery))
    assert live.bounded_prefix(certificate) == (relocation, delivery)
    too_long = SimpleNamespace(status=ViabilityStatus.SAFE, witness=(relocation, replace_step(delivery, 14)))
    assert not live.bounded_prefix(too_long)
    guard = live.AbsoluteProgressGuard()
    guard.environment = env(30)
    option = SimpleNamespace(action_type=SimpleNamespace(value="reconfigure"), recovery_action=relocation,
                             horizon_steps=None, successor_state=SimpleNamespace(blocks=(1,)), certificate=certificate)
    snapshot = SimpleNamespace(candidates=(option,), current_certificate=certificate, decision_epoch=30)
    prepared = SimpleNamespace(records=(0,), source_indices=(0,))
    assert guard.forced_prepared_index(snapshot, prepared) == 0
    assert guard.active_witness == (relocation, delivery)
    restored = live.AbsoluteProgressGuard()
    restored.load_state_dict(guard.state_dict())
    assert restored.active_witness == guard.active_witness and restored.commit_time == 30
    restored.reset()
    assert restored.commit_time is None


def replace_step(action, steps):
    return SimpleNamespace(kind=action.kind, steps=steps)


def test_no_silent_fallback_if_progress_prefix_budget_cannot_be_proved():
    guard = live.AbsoluteProgressGuard()
    guard.environment = env(30)
    certificate = SimpleNamespace(status=ViabilityStatus.SAFE,
        witness=(SimpleNamespace(kind=RecoveryActionKind.DELIVERY, steps=21),))
    snapshot = SimpleNamespace(candidates=(), current_certificate=certificate, decision_epoch=30)
    prepared = SimpleNamespace(records=(), source_indices=())
    with pytest.raises(Exception, match="not certified"):
        guard.forced_prepared_index(snapshot, prepared)


def test_release_and_horizon_assumptions_are_explicit():
    result = live.quantitative_contract(range(0, 1080, 30))
    assert result["fits_horizon"] and result["conditional_completion_bound"] == 3600
    assert not live.quantitative_contract(range(0, 2160, 60))["fits_horizon"]


def test_model_optimizer_and_checkpoint_smoke():
    result = checks()
    assert result["passed"], result


def test_same_bellman_action_for_all_heads_with_clock_observed_records():
    from viability_graph_episodic_audit import EpisodicViabilityGraphConfig
    from experiments.conditioned_vcg.E115_release_et_cost_vcg_8x8_179k_v1.model import FactorizedTransition
    agent = ProgressAgent(config=EpisodicViabilityGraphConfig(gamma=1., episode_horizon_steps=3600), seed=0)
    first = toy_record(29.)
    second = replace(first, key="second", current_state=attach_progress(first.current_state, (30/3600., 0., 0., 1.)))
    online = torch.tensor([[0., 8., 0., 0.], [5., 0., 0., 0.]])
    target = torch.tensor([[1., 10., 100., 1.], [2., 20., 200., 2.]])
    def scores(records, **kwargs):
        return torch.zeros((1, 4), requires_grad=True) if len(records) == 1 else online
    transition = FactorizedTransition(chosen=first, reward=0., duration=1,
        next_candidates=(first, second), done=False, component_returns=(0., 0., 0., 0.))
    with patch.object(agent, "_component_scores", scores), patch.object(agent, "_target_components", return_value=target), \
         patch.object(agent, "_anchor_term", return_value=(torch.tensor(0.), 0)):
        _, result, _, _ = agent._td_batch((transition,))
    assert torch.equal(result[0], target[0])
