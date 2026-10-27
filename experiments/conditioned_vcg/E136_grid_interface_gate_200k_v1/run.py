"""Bounded development integration checks; no training or test entrypoint."""
import argparse
import hashlib
import json
from pathlib import Path
import signal
from time import perf_counter

import torch
from example.episode_instance import EpisodeInstance
from experiments.conditioned_vcg.E135_independent_grid_reference_199k_v1 import run as generator
from experiments.conditioned_vcg.E135_independent_grid_reference_199k_v1.reference import path
from experiments.conditioned_vcg.E123_absolute_progress_8x8_187k_v2 import run as shared
from PSLAP.viability_filter import ViabilitySearchConfig
from PSLAP.relocation_family_certification import RELOCATION_FAMILY_CERTIFICATION
from .environment import GridEnv
from .controller import certified_pair, interface, horizon_audit, predict

ROOT = generator.ROOT
OUTPUT = ROOT/'results/vcg-conditioned-e136-grid-interface-gate-200k-v1'


def read(path):
    return json.loads(path.read_text())


def authenticate(output):
    generator.authenticate(generator.OUTPUT)
    paths = [*Path(__file__).parent.glob('*.py'), Path(__file__).with_name('README.txt'),
             Path(__file__).with_name('run.sh'), ROOT/'tests/test_vcg_e136_grid_interface.py',
             ROOT/'results/vcg-conditioned-e135-independent-grid-reference-199k-v1/e135-report.json']
    paths += [ROOT/name for name in (
        'vcg_objective_audit.py', 'run_vcg_objective_gamma_audit.py', 'viability_graph_hierarchy.py',
        'PSLAP/viability_candidates.py', 'PSLAP/viability_filter.py', 'PSLAP/dynamic_yard.py',
        'experiments/conditioned_vcg/E115_release_et_cost_vcg_8x8_179k_v1/rewards.py',
        'experiments/conditioned_vcg/E115_release_et_cost_vcg_8x8_179k_v1/model.py',
        'experiments/conditioned_vcg/E123_absolute_progress_8x8_187k_v1/model.py',
        'experiments/conditioned_vcg/E123_absolute_progress_8x8_187k_v1/liveness.py',
        'experiments/conditioned_vcg/E123_absolute_progress_8x8_187k_v2/run.py',
        'experiments/conditioned_vcg/E123_absolute_progress_8x8_187k_v2/acceleration.py',
        'experiments/conditioned_vcg/E126_release_clock_informed_baselines_190k_v1/clock.py',
        'experiments/conditioned_vcg/E130_exact_macro_cost_diagnostic_194k_v1/costs.py',
        'experiments/conditioned_vcg/E131_nominal_cost_residual_pilot_195k_v1/model.py',
        'experiments/conditioned_vcg/E27_certified_dynamic_sustained_99k/selector.py')]
    for cell in read(generator.OUTPUT/'e135-report.json')['cells']:
        name = (f'fill-{cell["fill"]:g}-load-{cell["requested_load"]:g}'
                f'-gap-{cell["frozen_gap"]}-seed-199000.json')
        paths.append(generator.OUTPUT/'calibration'/name)
    value = {'protocol': 'vcg_conditioned_e136_grid_interface_gate_200k_v1',
             'scope': 'bounded development integration; no trained policy quality claims',
             'frontier_wall_seconds': 30, 'guard_stress_wall_seconds': 30,
             'guard_stress_forced_macros': 2, 'frontier_max_nodes': 2000,
             'methods_share': 'literal GridAgent preparation; A-outflow/E121 guard and Defer',
             'sources': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}
    output.mkdir(parents=True, exist_ok=True)
    target = output/'contract.json'
    if target.exists() and read(target) != value:
        raise RuntimeError('E136 source changed; use fresh output')
    if not target.exists():
        target.write_text(json.dumps(value, indent=2, sort_keys=True)+'\n')
    return value


def obstruction(env):
    stored = [b for b in env.blocks if b.stored and not b.delivered]
    blocked = []
    for block in stored:
        approach = path(env, env.current_state, block.position, block.position)
        retrieval = [path(env, block.position, gate, block.position) for gate in env.exit_cells]
        if approach is None or all(route is None for route in retrieval):
            blocked.append(block.label)
    return {'stored_jobs': len(stored), 'directly_obstructed_jobs': len(blocked),
            'blocked_labels': blocked,
            'interpretation': 'direct pickup/exit path obstruction, not proof that rehandles are necessary'}


def replay_reference(case):
    env = GridEnv(case)
    original = tuple(b.storage_steps_needed for b in env.blocks)
    peak_blocked = obstruction(env)['directly_obstructed_jobs']
    for record in case['reference']['trace']:
        if record['kind'] == 'accept':
            block = next(b for b in env.blocks if b.label == record['job'])
            block.storage_location = tuple(record['target'])
        _, _, _, info = env.step(record['action'])
        if info.get('stored_block') or info.get('delivered_block'):
            peak_blocked = max(peak_blocked, obstruction(env)['directly_obstructed_jobs'])
    metrics = env.trajectory_metrics()
    if not metrics['strict_complete'] or tuple(b.storage_steps_needed for b in env.blocks) != original:
        raise RuntimeError('physical completion/duration preservation failed')
    if metrics['components']['tardiness'] or metrics['components']['earliness']:
        raise RuntimeError('reference external-date window disagrees with E135')
    return {'dense_offline_J_match': True, 'metrics': metrics,
            'maximum_directly_obstructed_jobs': peak_blocked,
            'nominal_storage_durations_unchanged': True}


def frontier_check(case):
    env = GridEnv(case)
    learned, rule_agent = certified_pair(env)
    audit = shared.original.replay.audit
    started = perf_counter()
    def timeout(*_):
        raise TimeoutError('E136 development frontier exceeded 30 seconds')
    previous = signal.signal(signal.SIGALRM, timeout)
    signal.setitimer(signal.ITIMER_REAL, 30.)
    try:
        with interface(env) as rule:
            snapshot = audit.enumerate_viability_candidates(env, consecutive_defer_decisions=0,
                search_config=ViabilitySearchConfig(max_nodes=2000), liveness_rule=rule,
                cache=shared.original.replay.reuse.TimingInvariantCertificateCache(),
                recovery_certification_strategy=RELOCATION_FAMILY_CERTIFICATION)
            left, left_forced = learned._admissible_prepared(snapshot)
            right, right_forced = rule_agent.base_agent._admissible_prepared(snapshot)
            if left != right or left_forced != right_forced or not left.records:
                raise RuntimeError('certified preparation differs or eligible frontier empty')
            # These are disposable scratch merits, NOT an existing-policy eval.
            with torch.no_grad():
                selected = learned.select(snapshot, training=False, epsilon=0.)
                baseline = rule_agent.select(snapshot, training=False, epsilon=0.)
            allowed = {r.key for r in left.records}
            if selected.record.key not in allowed or baseline.record.key not in allowed:
                raise RuntimeError('selector escaped shared eligible frontier')
            if left_forced and selected.record.key != baseline.record.key:
                raise RuntimeError('guard-forced actions differ')
            checks = []
            for kind in ('accept', 'deliver', 'reconfigure', 'defer'):
                candidate = next((c for c in snapshot.candidates if c.action_type.value == kind), None)
                if candidate is None:
                    continue
                prediction = predict(env, candidate)
                # Copy together to retain option -> environment -> block
                # identity, without executing on or mutating the live boundary.
                import copy
                shadow, option = copy.deepcopy((env, candidate))
                execution = audit._execute_audit_macro(shadow, option, gamma=1.,
                    remaining_steps=6000-shadow.time_steps, evaluation=True)
                totals = tuple(sum(row[i] for row in shadow.factorized_component_records)
                               for i in range(4))
                error = max(abs(a-b) for a,b in zip(prediction['components'], totals))
                if not execution.option_success or kind != 'defer' and error > 1e-7:
                    raise RuntimeError(f'external macro {kind}: success={execution.option_success} '
                                       f'failure={execution.failure_reason} cost_error={error} '
                                       f'nominal={prediction["components"]} observed={totals}')
                checks.append({'kind': kind, 'nominal_duration': prediction['duration'],
                               'actual_duration': execution.duration, 'max_cost_error': error,
                               'event_sensitive': kind == 'defer'})
            return {'status': 'pass', 'candidate_count': len(left.records),
                    'identical_prepared_records': True, 'identical_forced_flag': True,
                    'guard_forced': left_forced, 'learned_selected_key': selected.record.key,
                    'rule_selected_key': baseline.record.key, 'macro_cost_checks': checks,
                    'scope': 'same boundary, fresh identical guards; NOT full trajectory guard-lifecycle coverage',
                    'wall_seconds': perf_counter()-started}
    except (TimeoutError, RuntimeError, ValueError) as error:
        return {'status': 'incomplete_or_failed', 'reason': str(error),
                'wall_seconds': perf_counter()-started}
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.)
        signal.signal(signal.SIGALRM, previous)


def guard_lifecycle_check(case):
    """Two real forced macros from an explicitly injected trigger state.

    Counter injection exercises the retained-witness path; it is not evidence
    that a natural policy trajectory reaches that trigger or finishes a task.
    """
    env = GridEnv(case)
    if not env.initial_labels:
        return {'status': 'not_applicable', 'reason': 'no initial recovery witness'}
    left, right = certified_pair(env)
    right_base = right.base_agent
    for agent in (left, right_base):
        guard = agent.recovery_witness_guard
        guard.nonprogress_recovery_decisions = guard.max_nonprogress_recovery_decisions
    audit = shared.original.replay.audit
    started = perf_counter()
    def timeout(*_):
        raise TimeoutError('guard lifecycle exceeded 30 seconds')
    old_handler = signal.signal(signal.SIGALRM, timeout)
    signal.setitimer(signal.ITIMER_REAL, 30.)
    steps = []
    try:
        with interface(env) as rule:
            cache = shared.original.replay.reuse.TimingInvariantCertificateCache()
            def enumerate_now():
                return audit.enumerate_viability_candidates(env, consecutive_defer_decisions=0,
                    search_config=ViabilitySearchConfig(max_nodes=2000), liveness_rule=rule, cache=cache,
                    recovery_certification_strategy=RELOCATION_FAMILY_CERTIFICATION)
            snapshot = enumerate_now()
            for _ in range(2):
                with torch.no_grad():
                    learned = left.select(snapshot, training=False, epsilon=0.)
                    heuristic = right.select(snapshot, training=False, epsilon=0.)
                if (not learned.liveness_forced or not heuristic.liveness_forced
                        or learned.prepared_snapshot != heuristic.prepared_snapshot
                        or learned.candidate.key != heuristic.candidate.key):
                    raise RuntimeError('forced pair diverged')
                execution = audit._execute_audit_macro(env, heuristic.candidate, gamma=1.,
                    remaining_steps=6000-env.time_steps, evaluation=True)
                if not execution.option_success:
                    raise RuntimeError(f'forced macro failed: {execution.failure_reason}')
                snapshot = enumerate_now()
                left_next = left.observe_outcome(learned, next_snapshot=snapshot, done=False)
                right_next = right.observe_outcome(heuristic, next_snapshot=snapshot, done=False)
                if (left_next != right_next or left.recovery_witness_guard.state_dict()
                        != right_base.recovery_witness_guard.state_dict()):
                    raise RuntimeError('post-execution guard state or backup frontier differs')
                steps.append({'key': heuristic.candidate.key, 'duration': execution.duration,
                              'shared_post_execution_frontier': True})
        return {'status': 'pass', 'forced_macros': steps,
                'scope': 'injected trigger, two real committed-witness macros; not full-episode completion',
                'wall_seconds': perf_counter()-started}
    except (TimeoutError, RuntimeError, ValueError) as error:
        return {'status': 'incomplete_or_failed', 'reason': str(error), 'forced_macros': steps,
                'wall_seconds': perf_counter()-started}
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.)
        signal.signal(signal.SIGALRM, old_handler)


def preflight(output):
    authenticate(output)
    torch.set_num_threads(1)
    rows = []
    calibration = read(generator.OUTPUT/'e135-report.json')
    for cell in calibration['cells']:
        if cell['status'] != 'pass':
            raise RuntimeError('E135 calibration not passed')
        fill, load, gap = cell['fill'], cell['requested_load'], cell['frozen_gap']
        filename = f'fill-{fill:g}-load-{load:g}-gap-{gap}-seed-199000.json'
        case = read(generator.OUTPUT/'calibration'/filename)
        print(f'E136 development fill={fill:g} load={load:g} replay/reference checks', flush=True)
        replay = replay_reference(case)
        frontier = frontier_check(case)
        lifecycle = guard_lifecycle_check(case)
        rows.append({'fill': fill, 'load': load, 'reference_replay': replay,
                     'initial_obstruction': obstruction(GridEnv(case)), 'shared_frontier': frontier,
                     'guard_lifecycle': lifecycle,
                     'horizon': horizon_audit(EpisodeInstance.from_dict(case['instance']), len(case['initial_positions']))})
    result = {'status': 'bounded_integration_checks_passed' if all(r['shared_frontier']['status']=='pass'
              and r['guard_lifecycle']['status'] in ('pass', 'not_applicable') for r in rows)
              else 'integration_gate_incomplete', 'rows': rows, 'training_runs': 0,
              'test_panel_opened': False, 'training_authorized': False,
              'training_settings_frozen': False,
              'remaining_gates': ['full trajectory shared guard lifecycle and plain baseline harness smoke',
                                  'uniform progress proof OR explicitly limited empirical completion claim',
                                  'mixture/seed/checkpoint/statistics/GA budget freeze']}
    (output/'e136-report.json').write_text(json.dumps(result, indent=2, sort_keys=True)+'\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('preflight',))
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args()
    result = preflight(args.output.resolve())
    print(json.dumps({k:v for k,v in result.items() if k != 'rows'}, indent=2))


if __name__ == '__main__':
    main()
