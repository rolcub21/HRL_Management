"""Deterministic clock/budget/context checks and an isolated optimizer probe."""

from dataclasses import replace
import math
from types import SimpleNamespace

import torch

from PSLAP.viability import ViabilityStatus
from PSLAP.viability_candidates import ViabilityActionType
from viability_graph_hierarchy import ACTION_FEATURE_DIM, ViabilityGraphCandidateRecord
from experiments.conditioned_vcg.E107_queue_context_stage_a_8x8_171k_v1.representation import attach_queue_context
from experiments.conditioned_vcg.E115_release_et_cost_vcg_8x8_179k_v1.model import FactorizedTransition
from experiments.conditioned_vcg.E122_joint_flow_contract_8x8_186k_v1.preflight import yard_state
from .liveness import ProgressClock, AbsoluteProgressDeferRule, eligible_indices, quantitative_contract
from .model import ProgressAgent, attach_progress


def candidate(kind, duration, *, supported=True):
    certificate = SimpleNamespace(status=ViabilityStatus.SAFE, witness=())
    return SimpleNamespace(action_type=ViabilityActionType(kind),
        recovery_action=SimpleNamespace(steps=duration) if kind in ("deliver", "reconfigure") else None,
        horizon_steps=duration if kind == "defer" else None,
        option=SimpleNamespace(_block=lambda: object(), _strict_actions=lambda _: [0] * duration),
        successor_state=SimpleNamespace(blocks=() if supported else (1,)), certificate=certificate)


def toy_record(age=0.):
    state = yard_state()
    observed = attach_progress(attach_queue_context(state, (0.,) * 8), (age / 3600., 1., 0., 1.))
    return ViabilityGraphCandidateRecord(key="test:deliver", mode_id=1, action_type="deliver",
        current_state=observed, successor_state=state, action_features=(0.,) * ACTION_FEATURE_DIM)


def checks():
    from viability_graph_episodic_audit import EpisodicViabilityGraphConfig
    results = {}
    clock = ProgressClock()
    for now in (10, 20, 30):
        clock.observe(now, 0, 0)  # arrivals/reconfiguration, no actual flow event
    results["arrivals_and_reconfiguration_cannot_reset_absolute_clock"] = clock.age(30) == 30
    clock.observe(38, 1, 0)
    results["admission_resets_only_on_actual_flow_event"] = clock.last_progress_time == 38 and clock.events == 1
    clock.observe(80, 1, 1)
    results["delivery_is_progress"] = clock.last_progress_time == 80 and clock.events == 2
    env = SimpleNamespace(progress_clock=ProgressClock(), time_steps=30,
        blocks=[SimpleNamespace(delivered=False, stored=False, carrying=False, position=(1, 6))])
    rule = AbsoluteProgressDeferRule()
    result = rule.evaluate(env, SimpleNamespace(blocks=()), inbound_present=True,
        current_certificate=SimpleNamespace(status=ViabilityStatus.SAFE),
        consecutive_defer_decisions=0, terminal=False)
    results["arrival_reset_of_legacy_streak_cannot_enable_wait_at_deadline"] = not result.allowed
    items = (candidate("accept", 8), candidate("deliver", 9), candidate("reconfigure", 2), candidate("defer", 10))
    snapshot = SimpleNamespace(candidates=items, decision_epoch=30)
    prepared = SimpleNamespace(records=(0, 1, 2, 3), source_indices=(0, 1, 2, 3))
    results["deadline_excludes_reconfigure_even_with_accept_available"] = eligible_indices(snapshot, prepared, env, False) == (0, 1)
    budget = quantitative_contract(tuple(30 * i for i in range(36)))
    results["quantitative_budget_is_conditional_and_fits"] = budget["fits_horizon"] and budget["conditional_completion_bound"] == 3600
    results["unsupported_release_gaps_are_rejected"] = not quantitative_contract(tuple(60 * i for i in range(36)))["fits_horizon"]
    # When recovery empties the yard, every possible exit landing must still
    # permit a bounded next admission. This checks the external-idle branch.
    from .run import new_env
    from .liveness import direct_progress
    from experiments.conditioned_vcg.E122_joint_flow_contract_8x8_186k_v1.flow import JointVerifier, JointCertificateCache
    from PSLAP.viability_filter import ViabilitySearchConfig
    empty_env = new_env("C_joint_absolute")
    empty_env.reset(instance=empty_env.sample_episode_instance(187990))
    bounded_admissions = []
    verifier = JointVerifier()
    with verifier.enabled():
        for position in (empty_env.start_state, *empty_env.exit_cells):
            empty_env.current_state = tuple(position)
            snapshot = verifier.enumerate(empty_env, consecutive_defer_decisions=0,
                search_config=ViabilitySearchConfig(search_order="goal_directed"),
                liveness_rule=AbsoluteProgressDeferRule(), cache=JointCertificateCache())
            bounded_admissions.append(any(direct_progress(c) for c in snapshot.candidates))
    results["empty_yard_all_exit_landings_have_bounded_next_admission"] = all(bounded_admissions)
    config = EpisodicViabilityGraphConfig(gamma=1., episode_horizon_steps=3600, batch_size=2)
    first = ProgressAgent(config=config, seed=7, device="cpu", epsilon=0.)
    second = ProgressAgent(config=config, seed=7, device="cpu", epsilon=0., absolute_progress=True, arm="C_joint_absolute")
    results["matched_initialization"] = all(torch.equal(v, second.Q_local.state_dict()[key])
                                            for key, v in first.Q_local.state_dict().items())
    record = toy_record(29.)
    changed = replace(record, current_state=attach_progress(record.current_state, (30/3600., 0., 0., 1.)))
    results["clock_is_explicit_in_critic_input"] = not torch.equal(first.Q_local.context_tensor((record.current_state,)),
                                                                  first.Q_local.context_tensor((changed.current_state,)))
    components = (2., 3., 4., 1.)
    transition = FactorizedTransition(chosen=record, reward=-17.4, duration=1,
        next_candidates=(), done=True, component_returns=components)
    first.replay.add(transition)
    first.replay.add(transition)
    with __import__("experiments.conditioned_vcg.E100_accelerated_release_clock_163k_v1.fast_inputs", fromlist=["enabled"]).enabled():
        loss = first.learn()
    restored = ProgressAgent.from_checkpoint(first.checkpoint(include_replay=True), resumable=True, seed=7)
    results["optimizer_and_resumable_replay_smoke"] = (loss is not None and math.isfinite(loss)
        and restored.gradient_steps == 1 and restored.replay.memory[0].chosen.current_state.progress_context == record.current_state.progress_context)
    return {"passed": all(results.values()), "checks": results,
            "synthetic_optimizer_updates": 1, "research_training_runs": 0}
