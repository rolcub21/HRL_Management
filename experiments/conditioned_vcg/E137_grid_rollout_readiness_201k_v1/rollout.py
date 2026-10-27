from collections import Counter
from dataclasses import replace
from time import perf_counter

from experiments.conditioned_vcg.E136_grid_interface_gate_200k_v1.environment import GridEnv
from experiments.conditioned_vcg.E136_grid_interface_gate_200k_v1.controller import GridAgent, interface, HORIZON
from experiments.conditioned_vcg.E123_absolute_progress_8x8_187k_v2 import run as shared
from experiments.conditioned_vcg.E27_certified_dynamic_sustained_99k.selector import CertifiedDynamicAgent
from PSLAP.viability_filter import ViabilitySearchConfig
from PSLAP.relocation_family_certification import RELOCATION_FAMILY_CERTIFICATION
from PSLAP.ga_policy import DEFAULT_ROLLING_GA_CONFIG
from example.Options.selector_v5 import make_track_b_assignment_option
from example.controller_options import build_controller_options
from example.urgency_scheduler import RESERVED_CELL_ACTION_INTERFACE
from experiments.conditioned_vcg.E28_empty_start_scale_baselines_100k import run as old_baselines
from experiments.conditioned_vcg.E28_empty_start_scale_baselines_100k.adapter import WaitEnabledReconfigurationScheduler

METHODS = ('dynamic_pslap', 'ga', 'certified_dynamic', 'scratch_vcg_interface')
GA_CONFIG = replace(DEFAULT_ROLLING_GA_CONFIG, seed=201000)


class RecordedGridEnv(GridEnv):
    """Physical validation independent of selector/certificate, plus recorder."""
    def reset(self, instance=None):
        value = super().reset(instance)
        self.primitive_records = []
        self.action_counts = Counter()
        self.pending_pickup = None
        return value

    def step(self, action):
        action = int(action)
        if action not in range(7):
            raise RuntimeError('invalid primitive action')
        if self.pending_pickup is not None and action != 4:
            raise RuntimeError('occupied source traversed without pickup')
        held = [b for b in self.blocks if b.carrying]
        if len(held) > 1:
            raise RuntimeError('multiple carried blocks')
        pending = None
        if action < 4:
            dr, dc = ((-1,0),(1,0),(0,-1),(0,1))[action]
            cell = (self.current_state[0]+dr, self.current_state[1]+dc)
            if not (0 <= cell[0] < 8 and 0 <= cell[1] < 8) or self.rooms[cell] == '#':
                raise RuntimeError('wall or boundary collision')
            occupied = [b for b in self.blocks if b.position == cell and not b.carrying and not b.delivered]
            if occupied:
                if held or len(occupied) != 1:
                    raise RuntimeError('loaded block collision')
                pending = occupied[0].label
        if action == 4:
            occupants = [b for b in self.blocks if b.position == self.current_state and not b.delivered and not b.carrying]
            if held or len(occupants) != 1 or self.pending_pickup not in (None, occupants[0].label):
                raise RuntimeError('invalid physical pickup')
        if action == 5:
            if len(held) != 1:
                raise RuntimeError('putdown without held block')
            cell = self.current_state
            if cell not in self.storage_positions and cell not in self.exit_cells:
                raise RuntimeError('putdown outside legal storage/exit')
            if not held[0].stored and cell != held[0].storage_location:
                raise RuntimeError('unbound first placement')
            if any(b.position == cell and not b.delivered and not b.carrying for b in self.blocks):
                raise RuntimeError('putdown collision')
        result = super().step(action)
        self.pending_pickup = pending
        info = result[3]
        self.action_counts[self.ACTION_NAMES[action]] += 1
        self.primitive_records.append({'step': self.time_steps, 'action': action,
            'position': self.current_state, 'picked': info.get('picked_block'),
            'stored': info.get('stored_block'), 'delivered': info.get('delivered_block'),
            'relocated': info.get('relocated_block'), 'components': self.factorized_component_records[-1]})
        if info.get('illegal_drop'):
            raise RuntimeError('simulator rejected putdown')
        return result


def plain_controller(env, method):
    original_method = old_baselines.DYNAMIC if method == 'dynamic_pslap' else old_baselines.GA
    _, source, _ = old_baselines._baseline_source(original_method, {})
    selector = make_track_b_assignment_option(env, source, device='cpu', seed=201000,
        rolling_ga_config=GA_CONFIG, rolling_ga_egress_weight=4)
    build_controller_options(env, selector, controller_action_interface=RESERVED_CELL_ACTION_INTERFACE,
                             max_defer_steps=10)
    return WaitEnabledReconfigurationScheduler(env, margin_steps=2.), selector


def collect(env, strict):
    metrics = env.trajectory_metrics()
    if not strict:
        metrics['strict_complete'] = metrics['cost_ranking_eligible'] = False
    incoming = [b for b in env.blocks if b.label not in env.initial_labels]
    delivered = [b for b in env.blocks if b.delivered]
    errors = [b.delivered_time_step-env.public_due_date(b) for b in delivered]
    waits = [b.stored_time_step-b.arrival_step for b in incoming if b.stored_time_step is not None]
    metrics.update(delivered=len(delivered), required_jobs=len(env.blocks),
        release_mae=sum(abs(x) for x in errors)/len(errors) if errors else None,
        within_20=sum(abs(x)<=20 for x in errors)/len(errors) if errors else None,
        mean_observed_admission_wait=sum(waits)/len(waits) if waits else None,
        observed_incoming_admissions=len(waits), action_counts=dict(env.action_counts))
    return metrics


def run(env, method, *, max_decisions, max_nodes, emit):
    """Mutates env and diagnostic status so caller can preserve wall timeouts."""
    state = {'decisions': 0, 'macro_failures': 0, 'reason': None, 'latencies': [],
             'guard_outcome_updates': 0, 'forced_decisions': 0, 'macro_records': []}
    env.rollout_state = state
    certified = method in ('certified_dynamic', 'scratch_vcg_interface')
    base = GridAgent.bound(env) if certified else None
    agent = CertifiedDynamicAgent(base) if method == 'certified_dynamic' else base
    audit = shared.original.replay.audit
    with interface(env) as rule:
        if not certified:
            scheduler, selector = plain_controller(env, method)
        cache = shared.original.replay.reuse.TimingInvariantCertificateCache()
        streak = 0
        def snapshot():
            return audit.enumerate_viability_candidates(env, consecutive_defer_decisions=streak,
                search_config=ViabilitySearchConfig(max_nodes=max_nodes), liveness_rule=rule,
                cache=cache, recovery_certification_strategy=RELOCATION_FAMILY_CERTIFICATION)
        current = None
        pending_frontier_seconds = 0.
        while not all(b.delivered for b in env.blocks) and env.time_steps < HORIZON:
            if state['decisions'] >= max_decisions:
                state['reason'] = 'decision_cap'
                break
            started = perf_counter()
            before_events = env.progress_clock.events
            if certified:
                if current is None:
                    current = snapshot()
                decision = agent.select(current, training=False, epsilon=0.)
                state['latencies'].append(perf_counter()-started+pending_frontier_seconds)
                pending_frontier_seconds = 0.
                state['decisions'] += 1
                state['forced_decisions'] += int(decision.liveness_forced)
                cursor = env.time_steps
                execution = audit._execute_audit_macro(env, decision.candidate, gamma=1.,
                    remaining_steps=HORIZON-env.time_steps, evaluation=True)
                state['macro_records'].append({'key': decision.candidate.key,
                    'start': cursor, 'duration': execution.duration, 'success': execution.option_success})
                if not execution.option_success:
                    state['macro_failures'] += 1
                    state['reason'] = execution.failure_reason
                    break
                streak = (streak+1 if decision.candidate.action_type.value == 'defer'
                          and env.progress_clock.events == before_events else 0)
                done = all(b.delivered for b in env.blocks)
                frontier_started = perf_counter()
                current = None if done else snapshot()
                agent.observe_outcome(decision, next_snapshot=current, done=done)
                pending_frontier_seconds = perf_counter()-frontier_started if not done else 0.
                state['guard_outcome_updates'] += 1
            else:
                option = scheduler.select_action(env.get_current_state())
                state['latencies'].append(perf_counter()-started)
                state['decisions'] += 1
                cursor = env.time_steps
                while env.time_steps < HORIZON:
                    _, reward, _, info = env.step(option.policy(env.get_current_state()))
                    selector.on_step(reward, info)
                    terminated = option.termination(env.get_current_state())
                    if terminated or all(b.delivered for b in env.blocks):
                        break
                outcome = getattr(option, 'last_outcome', None)
                success = not (isinstance(outcome, dict) and outcome.get('success') is False)
                state['macro_records'].append({'key': type(option).__name__, 'start': cursor,
                    'duration': env.time_steps-cursor, 'success': success})
                if not success:
                    state['macro_failures'] += 1
                    state['reason'] = str(outcome)
                    break
            emit(state)
        if state['reason'] is None and not all(b.delivered for b in env.blocks):
            state['reason'] = 'primitive_horizon'
    return state
