"""Prepare/run/resume/report user-launched development episodes, never training."""
import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import signal
import random
from time import perf_counter

import torch
import numpy as np
from experiments.conditioned_vcg.E136_grid_interface_gate_200k_v1 import run as integration
from experiments.conditioned_vcg.E136_grid_interface_gate_200k_v1.controller import HORIZON
from experiments.conditioned_vcg.E135_independent_grid_reference_199k_v1 import run as generator
from .rollout import RecordedGridEnv, METHODS, GA_CONFIG, run, collect

ROOT = generator.ROOT
OUTPUT = ROOT/'results/vcg-conditioned-e137-grid-rollout-readiness-201k-v1'


class WallTimeout(Exception):
    pass


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n')
    os.replace(temporary, path)


def cells():
    report = read(generator.OUTPUT/'e135-report.json')
    result = []
    for cell in report['cells']:
        if cell['status'] != 'pass':
            raise RuntimeError('calibration not passed')
        name = (f'fill-{cell["fill"]:g}-load-{cell["requested_load"]:g}'
                f'-gap-{cell["frozen_gap"]}-seed-199000.json')
        path = generator.OUTPUT/'calibration'/name
        result.append({'fill': cell['fill'], 'load': cell['requested_load'],
                       'path': str(path.relative_to(ROOT)), 'sha256': sha(path)})
    return result


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(output, wall_seconds=300, max_decisions=1000, max_nodes=2000):
    if wall_seconds <= 0 or max_decisions < 1 or max_nodes < 1:
        raise ValueError('budgets must be positive')
    integration.authenticate(integration.OUTPUT)
    parent_report = integration.OUTPUT/'e136-report.json'
    if read(parent_report)['status'] != 'bounded_integration_checks_passed':
        raise RuntimeError('run/pass E136 preflight first')
    sources = [*Path(__file__).parent.glob('*.py'), Path(__file__).with_name('run.sh'),
               Path(__file__).with_name('README.txt'), ROOT/'tests/test_vcg_e137_grid_rollout.py',
               parent_report, integration.OUTPUT/'contract.json']
    # Hash the native baseline dependency chain too; source differences cannot
    # silently reuse another machine's completed rows.
    sources += [ROOT/name for name in (
        'example/controller_options.py', 'example/urgency_scheduler.py',
        'example/Options/selector_v5.py', 'example/Options/AcceptStoreOption.py',
        'example/Options/RetrieveDeliverOption.py', 'example/Options/StrategicDeferOption.py',
        'PSLAP/track_a.py', 'PSLAP/ga_policy.py', 'PSLAP/ga_capacity_aware.py',
        'PSLAP/ga_optimizer.py',
        'experiments/conditioned_vcg/E25_v4_reconfiguration_baselines_95k/controller.py',
        'experiments/conditioned_vcg/E28_empty_start_scale_baselines_100k/adapter.py')]
    value = {'protocol': 'vcg_conditioned_e137_grid_rollout_readiness_201k_v1',
        'scope': 'development calibration instances; scratch control is not a trained-policy benchmark',
        'methods': METHODS, 'cases': cells(), 'horizon': HORIZON,
        'wall_seconds_per_row': wall_seconds, 'max_decisions': max_decisions,
        'certificate_max_nodes': max_nodes, 'ga': asdict(GA_CONFIG),
        'source_sha256': {str(p.relative_to(ROOT)):sha(p) for p in sources},
        'training_authorized': False, 'test_panel_opened': False,
        'completion_bound_proved': False,
        'macro_alignment': 'certified pair shares exact interface; plain native compound retrieval '
                           'versus certified elementary macros remains a declared alignment limitation'}
    # Canonical JSON also normalizes tuples for comparison after loading.
    value = json.loads(json.dumps(value))
    path = output/'contract.json'
    if path.exists() and read(path) != value:
        raise RuntimeError('E137 contract/source/budget changed; choose a fresh --output')
    if not path.exists():
        write(path, value)
    return value


def row_path(output, method, cell):
    return output/'rows'/method/f'fill-{cell["fill"]:g}-load-{cell["load"]:g}.json'


def coordinate(output, contract, method, cell, *, retry=False):
    path = row_path(output, method, cell)
    if path.exists() and not retry:
        print(f'E137 reuse {method} fill={cell["fill"]:g} load={cell["load"]:g}', flush=True)
        return read(path)
    if path.exists():
        # Preserve prior failures/interruptions rather than erasing them.
        if read(path)['status'] == 'complete':
            raise RuntimeError('refusing to retry/overwrite a completed coordinate')
        index = 1
        while path.with_name(path.stem+f'.attempt-{index}.json').exists():
            index += 1
        path.rename(path.with_name(path.stem+f'.attempt-{index}.json'))
    case_path = ROOT/cell['path']
    if sha(case_path) != cell['sha256']:
        raise RuntimeError('calibration case changed')
    random.seed(201000)
    np.random.seed(201000)
    torch.manual_seed(201000)
    env = RecordedGridEnv(read(case_path))
    state = {'decisions': 0, 'latencies': [], 'macro_failures': 0,
             'forced_decisions': 0, 'macro_records': [], 'reason': None}
    env.rollout_state = state
    started = perf_counter()
    previous_log = started-30
    identity = f'{method} fill={cell["fill"]:g} load={cell["load"]:g}'
    def emit(current, force=False):
        nonlocal previous_log
        now = perf_counter()
        if force or now-previous_log >= 30:
            previous_log = now
            progress = {'method': method, **cell, 'decision': current['decisions'],
                        'step': env.time_steps, 'delivered': sum(b.delivered for b in env.blocks),
                        'required': len(env.blocks), 'wall_seconds': now-started}
            write(output/'progress'/method/f'fill-{cell["fill"]:g}-load-{cell["load"]:g}.json', progress)
            print(f'E137 {identity} | decision={current["decisions"]} step={env.time_steps} '
                  f'delivered={progress["delivered"]}/{len(env.blocks)} elapsed={now-started:.1f}s', flush=True)
    def alarm(*_):
        raise WallTimeout('coordinate wall limit')
    def terminate(*_):
        raise KeyboardInterrupt('termination requested')
    old_alarm = signal.signal(signal.SIGALRM, alarm)
    old_term = signal.signal(signal.SIGTERM, terminate)
    signal.setitimer(signal.ITIMER_REAL, contract['wall_seconds_per_row'])
    status, error, interrupted = 'failed', None, False
    try:
        emit(state, True)
        state = run(env, method, max_decisions=contract['max_decisions'],
                    max_nodes=contract['certificate_max_nodes'], emit=emit)
        status = 'complete' if all(b.delivered for b in env.blocks) and not state['reason'] else 'incomplete'
    except WallTimeout as exc:
        status, error = 'bounded_timeout', str(exc)
    except KeyboardInterrupt as exc:
        status, error, interrupted = 'interrupted', str(exc), True
    except Exception as exc:
        status, error = 'failed', f'{type(exc).__name__}: {exc}'
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.)
        signal.signal(signal.SIGALRM, old_alarm)
        signal.signal(signal.SIGTERM, old_term)
    state = env.rollout_state
    metrics, accounting_error = None, None
    try:
        metrics = collect(env, status == 'complete')
    except Exception as exc:
        accounting_error = f'{type(exc).__name__}: {exc}'
        if status == 'complete':
            status = 'failed'
    row = {'method': method, 'fill': cell['fill'], 'load': cell['load'],
           'case_sha256':cell['sha256'], 'contract_sha256':sha(output/'contract.json'),
           'status':status, 'error':error, 'accounting_error':accounting_error,
           'wall_seconds':perf_counter()-started, 'metrics':metrics, 'rollout':state,
           'mean_decision_latency': sum(state['latencies'])/len(state['latencies']) if state['latencies'] else None,
           'primitive_records':env.primitive_records, 'training_runs':0,
           'test_panel_opened':False, 'weights':'scratch integration control' if method.startswith('scratch') else 'none'}
    write(path, row)
    print(f'E137 {identity} | {status} steps={env.time_steps} '
          f'delivered={sum(b.delivered for b in env.blocks)}/{len(env.blocks)}', flush=True)
    if interrupted:
        raise KeyboardInterrupt
    return row


def analyze(output, contract):
    rows = []
    for method in METHODS:
        for cell in contract['cases']:
            path = row_path(output, method, cell)
            if path.exists():
                row = read(path)
                if row['contract_sha256'] != sha(output/'contract.json'):
                    raise RuntimeError('saved row contract differs')
                rows.append({k:v for k,v in row.items() if k != 'primitive_records'})
    report = {'status':'complete' if len(rows)==24 else 'partial', 'observed_rows':len(rows),
        'expected_rows':24, 'complete_rows':sum(r['status']=='complete' for r in rows),
        'rows':rows, 'training_runs':0, 'test_panel_opened':False, 'training_authorized':False,
        'readiness':'diagnostic only; no automatic training promotion or superiority claims',
        'remaining_gates':['plain/certified macro alignment decision', 'finite-progress/horizon claim',
                           'mixture/seed/budget/statistics freeze']}
    write(output/'e137-report.json', report)
    lines = ['# E137 development rollout readiness', '',
        'Scratch VCG is an untrained integration control. This is not a quality benchmark.', '',
        '| Method | Fill | Load | Status | Delivered | J (complete only) | Seconds |',
        '|---|---:|---:|---|---:|---:|---:|']
    for row in rows:
        m = row['metrics'] or {}
        j = f'{m["J"]:.2f}' if m.get('cost_ranking_eligible') else '—'
        lines.append(f'| {row["method"]} | {row["fill"]:g} | {row["load"]:g} | {row["status"]} '
                     f'| {m.get("delivered", "?")}/{m.get("required_jobs", "?")} | {j} | {row["wall_seconds"]:.1f} |')
    (output/'e137-table.md').write_text('\n'.join(lines)+'\n')
    return {**{k:v for k,v in report.items() if k != 'rows'},
            'report':str(output/'e137-report.json'), 'table':str(output/'e137-table.md')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare','run','status','analyze'))
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--method', choices=('all',*METHODS), default='all')
    parser.add_argument('--fill', type=float, choices=(0.,.4,.7))
    parser.add_argument('--load', type=float, choices=(.6,.85))
    parser.add_argument('--wall-seconds', type=float, default=300.)
    parser.add_argument('--max-decisions', type=int, default=1000)
    parser.add_argument('--max-nodes', type=int, default=2000)
    parser.add_argument('--retry-failed', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(1)
    output = args.output.resolve()
    contract = prepare(output,args.wall_seconds,args.max_decisions,args.max_nodes)
    try:
        if args.command == 'run':
            for method in METHODS if args.method == 'all' else (args.method,):
                for cell in contract['cases']:
                    if args.fill is not None and cell['fill'] != args.fill:
                        continue
                    if args.load is not None and cell['load'] != args.load:
                        continue
                    coordinate(output, contract, method, cell, retry=args.retry_failed)
    except KeyboardInterrupt:
        print(json.dumps(analyze(output, contract),indent=2))
        raise SystemExit(130)
    summary = analyze(output,contract)
    if args.command == 'prepare':
        summary['status'] = 'prepared'
    print(json.dumps(summary,indent=2))


if __name__ == '__main__':
    main()
