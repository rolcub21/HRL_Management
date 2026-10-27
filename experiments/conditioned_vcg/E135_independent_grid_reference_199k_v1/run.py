"""Development calibration only; no contender evaluation or training entrypoint."""
import argparse
import hashlib
import json
from pathlib import Path
import random
from statistics import fmean

from .reference import FILLS, HORIZON, dispatch, generate
from .verify import verify

ROOT = Path(__file__).resolve().parents[3]
OUTPUT = ROOT / 'results/vcg-conditioned-e135-independent-grid-reference-199k-v1'
LOADS = (0.60, 0.85)
CALIBRATION_SEEDS = (199000, 199001, 199002)
MAX_ITERATIONS = 8
LOAD_TOLERANCE = 0.03


def specification():
    return {'protocol': 'vcg_conditioned_e135_independent_grid_reference_199k_v1',
            'scope': 'development generator/reference calibration only',
            'fills': FILLS, 'loads': LOADS, 'incoming_jobs': 36,
            'calibration_seeds': CALIBRATION_SEEDS,
            'max_iterations': MAX_ITERATIONS, 'load_absolute_tolerance': LOAD_TOLERANCE,
            'first_gap_estimate_work_per_job': 30,
            'gap_update': 'nearest integer mean incoming reference work / requested load',
            'cycle_or_stagnation': 'fail, never silently relax tolerance',
            'initial_inventory': 'floor(fill*35); random subset outside reserved left/bottom corridor',
            'nominal_durations': 'independent Poisson(80), minimum 1; initial storage age zero',
            'reference_policy': 'due-ready reachable retrieval first; otherwise FIFO admission deepest reachable cell; otherwise wait',
            'due_construction': 'reference delivery + independent discrete uniform integer [0,20]',
            'reference_construction_horizon': HORIZON,
            'contender_horizon': None,
            'training_authorized': False, 'test_panel_opened': False,
            'remaining_gates': ['external due-date information/critic integration',
                                'identical VCG/certified-Dynamic frontier/guard/Defer audit',
                                'task horizon and liveness proof assumptions for variable job count/gaps',
                                'training/dev/test seed manifests and mixture schedule',
                                'budget/exploration/checkpoint selection; fixed GA budget',
                                'statistical hypotheses and multiplicity families']}


def authenticate(output):
    files = [Path(__file__).with_name(name) for name in
             ('__init__.py', 'reference.py', 'verify.py', 'run.py', 'README.txt', 'run.sh')]
    files += [ROOT/'example/small_rooms_env.py', ROOT/'example/episode_instance.py',
              ROOT/'tests/test_vcg_e135_independent_reference.py',
              ROOT/'experiments/conditioned_vcg/E134_regime_grid_protocol_198k_v1/protocol.py']
    payload = {'design': specification(), 'source_sha256': {
        str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
    output.mkdir(parents=True, exist_ok=True)
    path = output/'contract.json'
    encoded = json.dumps(payload, indent=2, sort_keys=True)+'\n'
    if path.exists() and path.read_text() != encoded:
        raise RuntimeError('E135 source/design changed; use a separately versioned output')
    if not path.exists():
        path.write_text(encoded)
    return payload


def case(seed, fill, gap):
    instance, initial = generate(seed, fill, gap)
    result = dispatch(instance, initial)
    rng = random.Random(seed + 1000000)
    dues = {label: step+rng.randint(0, 20) for label, step in result['deliveries'].items()}
    metrics = verify(instance, initial, result, dues)
    return {'instance': instance.to_dict(), 'initial_positions': initial,
            'external_due_dates': dues, 'reference': result,
            'metrics': metrics, 'seed': seed, 'fill': fill, 'gap': gap,
            'visibility': 'reference trace/full future manifest never exposed to contenders'}


def calibrate(output):
    authenticate(output)
    cells = []
    for fill in FILLS:
        for load in LOADS:
            gap = max(1, int(30/load+0.5))
            seen = set()
            history = []
            passed = False
            for iteration in range(MAX_ITERATIONS):
                if gap in seen:
                    break
                seen.add(gap)
                rows = []
                for seed in CALIBRATION_SEEDS:
                    row = case(seed, fill, gap)
                    name = f'fill-{fill:g}-load-{load:g}-gap-{gap}-seed-{seed}.json'
                    target = output/'calibration'/name
                    target.parent.mkdir(exist_ok=True)
                    encoded = json.dumps(row, indent=2, sort_keys=True)+'\n'
                    if target.exists() and target.read_text() != encoded:
                        raise RuntimeError('refusing to overwrite changed calibration case')
                    if not target.exists():
                        target.write_text(encoded)
                    rows.append(row['metrics'])
                work = fmean(r['work_per_incoming_job'] for r in rows)
                achieved = work/gap
                history.append({'iteration': iteration+1, 'gap': gap,
                                'mean_work_per_incoming_job': work,
                                'achieved_load': achieved, 'reference_metrics': rows})
                print(f'E135 fill={fill:g} load={load:g} iteration={iteration+1} '
                      f'gap={gap} achieved={achieved:.3f}', flush=True)
                if abs(achieved-load) <= LOAD_TOLERANCE:
                    passed = True
                    break
                gap = max(1, int(work/load+0.5))
            cells.append({'fill': fill, 'requested_load': load,
                          'status': 'pass' if passed else 'calibration_failed',
                          'frozen_gap': gap if passed else None, 'history': history})
    report = {'status': 'reference_calibration_passed' if all(c['status']=='pass' for c in cells)
              else 'reference_calibration_failed', 'cells': cells,
              'training_runs': 0, 'contender_episode_runs': 0,
              'test_panel_opened': False, 'training_authorized': False,
              'remaining_gates': specification()['remaining_gates']}
    (output/'e135-report.json').write_text(json.dumps(report, indent=2, sort_keys=True)+'\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'calibrate'))
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args()
    output = args.output.resolve()
    if args.command == 'prepare':
        authenticate(output)
        result = {'status': 'prepared', 'training_authorized': False}
    else:
        result = calibrate(output)
    print(json.dumps({k: v for k, v in result.items() if k != 'cells'}, indent=2))


if __name__ == '__main__':
    main()
