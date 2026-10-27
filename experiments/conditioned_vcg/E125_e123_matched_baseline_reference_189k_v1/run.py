"""Evaluate unchanged Dynamic PSLAP/GA on E123 A's exact development instances."""
import argparse
import builtins
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from statistics import fmean
from types import SimpleNamespace
from unittest.mock import patch

import torch

from experiments.conditioned_vcg.E116_e115_matched_operator_baselines_180k_v1 import run as baseline
from experiments.conditioned_vcg.E123_absolute_progress_8x8_187k_v2 import protocol as source

ROOT = source.ROOT
OUTPUT = ROOT / 'results/vcg-conditioned-e125-e123-matched-baseline-reference-189k-v1'
PROTOCOL = 'vcg_conditioned_e125_e123_matched_baseline_reference_189k_v1'
SEEDS = tuple(source.VALIDATION_SEEDS)
WINDOW = (96, 102, 108, 114, 120)
ARM = 'A_outflow_bounded'
METHODS = ('dynamic_pslap', 'ga')
GA_REPLICATES = baseline.p.GA_REPLICATES
sha256, digest = baseline.p.sha256, baseline.p.digest
write, read, require = baseline.write_json, baseline.read_json, baseline.require


@lru_cache(maxsize=1)
def source_data():
    expected = source.contract()
    require(read(source.DEFAULT_OUTPUT / 'contract.json') == expected,
            'E123-v2 source contract differs')
    directory = source.DEFAULT_OUTPUT / ARM / 'seed-0'
    summary = read(directory / 'summary.json')
    require(summary['completed_training_episodes'] == 120 and summary['arm'] == ARM
            and summary['model_seed'] == 0 and summary['contract_sha256'] == expected['contract_sha256'],
            'E125 requires completed E123-v2 A seed 0 at episode 120')
    selected = [v for v in summary['validation_history'] if v['checkpoint_episode'] in WINDOW]
    require([v['checkpoint_episode'] for v in selected] == list(WINDOW), 'Missing fixed late checkpoint')
    for validation in selected:
        require(validation['evaluation_epsilon'] == 0. and not validation['evaluation_learning'],
                'Only frozen greedy validation is comparable')
        require(sorted(r['instance_seed'] for r in validation['runs']) == sorted(SEEDS),
                'Source validation panel differs')
    return summary, selected


@lru_cache(maxsize=1)
def contract():
    summary, selected = source_data()
    directory = source.DEFAULT_OUTPUT / ARM / 'seed-0'
    paths = [Path(__file__), Path(__file__).with_name('run.sh'),
             Path(baseline.__file__), Path(baseline.e91.__file__), Path(baseline.adapter.__file__),
             ROOT / 'experiments/conditioned_vcg/E25_v4_reconfiguration_baselines_95k/controller.py',
             ROOT / 'tests/test_vcg_e125_matched_baselines.py']
    value = dict(protocol=PROTOCOL, scope='reused five-instance development reference; not independent test',
        source_contract_sha256=source.contract()['contract_sha256'], arm=ARM, model_seed=0,
        source_summary_sha256=sha256(directory / 'summary.json'),
        source_candidate120_sha256=sha256(directory / 'candidate-episode-120.pth'),
        late_window=list(WINDOW), instance_seeds=list(SEEDS), methods=list(METHODS),
        ga_replicates=list(GA_REPLICATES), objective='J=2T+E+0.1L+10R; release clock; tolerance +/-20',
        workload='empty 8x8 n36, releases 30*j, frozen Poisson(80) stays',
        source_sha256={str(p.relative_to(ROOT)):sha256(p) for p in paths},
        source_validation_digest=digest(selected),
        aggregation='GA replicates and VCG checkpoints averaged within instance; five paired instances',
        inference='descriptive paired differences with nominal 95% t intervals; no confirmatory claim',
        training_runs=0, independent_test_opened=False, paper_evidence=False)
    return {**value, 'contract_sha256':digest(value)}


@contextmanager
def adapter():
    settings = SimpleNamespace(PROTOCOL=PROTOCOL, INSTANCE_SEEDS=SEEDS, METHODS=METHODS,
        GA_REPLICATES=GA_REPLICATES, contract=contract, digest=digest)
    def readable(message, *args, **kwargs):
        if isinstance(message, str):
            message = message.replace('E116 ', 'E125 ')
        builtins.print(message, *args, **kwargs)
    with patch.object(baseline, 'p', settings), patch.object(baseline, 'print', readable, create=True):
        yield


def prepare(output):
    with adapter():
        result = baseline.prepare(output)
        frozen = read(output / 'instances.json')
        identities = {row['seed']:row['instance_id'] for row in frozen['records']}
        for validation in source_data()[1]:
            require(all(identities[r['instance_seed']] == r['episode_instance_id']
                        for r in validation['runs']), 'Baseline generator differs from realized VCG instances')
    return result


def average_checkpoints(checkpoints, seeds):
    """Average within instance, never treat repeated checkpoint rows as new instances."""
    require(len(checkpoints) == len(WINDOW), 'Fixed window requires five checkpoints')
    for points in checkpoints:
        require(sorted(p['instance_seed'] for p in points) == sorted(seeds), 'Checkpoint seed panel differs')
    result = []
    for seed in seeds:
        group = [next(p for p in points if p['instance_seed'] == seed) for points in checkpoints]
        complete = all(p['strict_complete'] for p in group)
        result.append(dict(method='e123_A_late_window', instance_seed=seed, strict_complete=complete,
            metrics={k:fmean(p['metrics'][k] for p in group) for k in baseline.METRICS} if complete else None))
    return result


def comparison(label, points, baseline_points):
    result = baseline._comparison(label, points, baseline_points)
    result['vcg'] = result.pop('e115')
    result['vcg']['method'] = f'e123_A_{label}'
    for row in result['contrasts']:
        row['difference_direction'] = 'E123 A minus baseline'
    return result


def analyze(output):
    prepare(output)
    with adapter():
        rows = []
        for method in METHODS:
            for seed in SEEDS:
                for rep in baseline.replicates(method):
                    path = baseline.row_path(output, method, seed, rep)
                    if path.exists():
                        row = read(path)
                        require(row['contract_sha256'] == contract()['contract_sha256']
                                and row['protocol'] == PROTOCOL
                                and (row['method'], row['instance_seed'], row['replicate']) == (method, seed, rep),
                                'Saved baseline coordinate differs')
                        rows.append(row)
        points, summaries = baseline._baseline_points(rows)
        validations = source_data()[1]
        checkpoints = [baseline._e115_checkpoint_points(v) for v in validations]
        comparisons = [comparison('late_window', average_checkpoints(checkpoints, SEEDS), points),
                       comparison(120, checkpoints[-1], points)]
        report = dict(protocol=PROTOCOL, contract_sha256=contract()['contract_sha256'],
            status='complete' if len(rows)==25 else 'partial', expected_rows=25, observed_rows=len(rows),
            strict_complete_rows=sum(r['metrics']['strict_complete'] for r in rows),
            baseline_summaries=summaries, baseline_instance_points=points, comparisons=comparisons,
            training_runs=0, episode_runs=0, paper_evidence=False, independent_test_opened=False,
            scope='same five reused E123 validation instances; not independent test; no seed or checkpoint selection')
        write(output / 'e125-report.json', report)
        lines = ['E125 — E123 A versus matched baselines (development only)', '',
                 '| Method | Complete instances | J | Release MAE | Within ±20 | Outside wait | Storage MAE | Rehandles/100 | Steps/delivery |',
                 '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
        entries = [(c['vcg']['method'], c['vcg']['strict_complete_instances'], c['vcg']['metrics']) for c in comparisons]
        entries += [(r['method'],r['fully_complete_instances'], r['metrics'] if r['fully_complete_instances']==5 else None)
                    for r in summaries]
        for name, count, metrics in entries:
            if metrics is None:
                lines.append(f'| {name} | {count}/5 | — | — | — | — | — | — | — |')
                continue
            values = [metrics[k] for k in ('objective_cost','release_clock_mae','within_tolerance_rate',
                'mean_wait_before_storage','storage_clock_mae','rehandles_per_100','steps_per_delivery')]
            lines.append(f'| {name} | {count}/5 | ' + ' | '.join(f'{v:.3f}' for v in values) + ' |')
        lines += ['', 'Within ±20 is a fraction. J=2T+E+0.1L+10R; L excludes WAIT, but total steps includes it.',
                  'VCG checkpoint repeats and GA replicates are averaged within instance; paired n=5, not 25.',
                  'Nominal paired intervals are descriptive, not multiplicity-adjusted confirmation.',
                  'Baseline latency is saved in JSON; cached VCG evaluation timings are not a controlled latency benchmark.',
                  'A3C omitted: no matched retraining. No independent test or model updates.']
        (output / 'e125-table.md').write_text('\n'.join(lines)+'\n')
    return dict(status=report['status'], observed_rows=len(rows), expected_rows=25,
                strict_complete_rows=report['strict_complete_rows'], report=str(output/'e125-report.json'),
                table=str(output/'e125-table.md'), training_runs=0, episode_runs=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare','run-one','run-all','analyze','status'))
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--method', choices=METHODS)
    parser.add_argument('--seed', type=int, choices=SEEDS)
    parser.add_argument('--replicate', type=int, default=0)
    args = parser.parse_args()
    torch.set_num_threads(1)
    output = args.output.resolve()
    if args.command == 'prepare':
        result = prepare(output)
    elif args.command in ('analyze','status'):
        result = analyze(output)
    else:
        prepare(output)
        with adapter():
            if args.command == 'run-one':
                require(args.method is not None and args.seed is not None, 'run-one requires --method and --seed')
                rows = [baseline.run_one(output, args.method, args.seed, args.replicate)]
            else:
                rows = [baseline.run_one(output, m, s, r) for m in METHODS for s in SEEDS
                        for r in baseline.replicates(m)]
        result = analyze(output)
        result['episode_runs'] = sum(not r.get('reused',False) for r in rows)
    import json
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
