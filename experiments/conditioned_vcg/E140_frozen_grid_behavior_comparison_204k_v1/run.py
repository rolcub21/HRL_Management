import argparse
import hashlib
import json
import os
import random
import signal
from pathlib import Path
import time
from unittest.mock import patch

import torch
import numpy as np
from experiments.conditioned_vcg.E138_residual_grid_mixture_pilot_202k_v1 import run as training
from experiments.conditioned_vcg.E139_grid_mixture_three_seeds_203k_v1 import run as replicates
from experiments.conditioned_vcg.E137_grid_rollout_readiness_201k_v1 import run as evaluation
from experiments.conditioned_vcg.E137_grid_rollout_readiness_201k_v1 import rollout

ROOT = training.p.ROOT
OUTPUT = ROOT/'results/vcg-conditioned-e140-frozen-grid-behavior-comparison-204k-v1'
BASELINES = ('dynamic_pslap','ga','certified_dynamic')


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):
            h.update(chunk)
    return h.hexdigest()


def source(seed):
    return training.p.OUTPUT/'seed-0' if seed == 0 else replicates.OUTPUT/f'seed-{seed}'


def model_class(seed):
    return training.MixtureAgent if seed == 0 else replicates.agent_class(seed)


def parameter_digest(agent):
    h = hashlib.sha256()
    for network in (agent.Q_local,agent.Q_target):
        for name,value in network.state_dict().items():
            h.update(name.encode())
            h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def freeze(output,seed):
    manifest = output/'models'/f'seed-{seed}.json'
    if manifest.exists():
        info = training.read(manifest)
        if sha(output/info['file']) != info['snapshot_sha256']:
            raise RuntimeError('frozen checkpoint changed')
        return info
    directory = source(seed)
    summary = training.read(directory/'summary.json')
    stopped = directory/'stopped.json' if seed else directory.parent/'stopped.json'
    if seed in (0,1) and summary['completed_training_episodes'] != 42:
        raise RuntimeError('initial comparison predeclares episode 42 for seeds 0 and 1')
    if seed == 2 and summary['completed_training_episodes'] < 60 and not stopped.exists():
        raise RuntimeError('seed 2 still running; add it only after completion or a recorded stop')
    original = directory/'latest.pth'
    identity = (original.stat().st_size,original.stat().st_mtime_ns)
    original_sha = sha(original)
    payload = torch.load(original,map_location='cpu',weights_only=False)
    episode = payload['completed_training_episodes']
    if episode != summary['completed_training_episodes']:
        raise RuntimeError('checkpoint/summary endpoint differs')
    if identity != (original.stat().st_size,original.stat().st_mtime_ns) or original_sha != sha(original):
        raise RuntimeError('source checkpoint changed; do not snapshot an active writer')
    agent = model_class(seed).from_checkpoint(payload,device='cpu',resumable=False)
    compact = agent.checkpoint(include_replay=False,e140_source_episode=episode)
    compact['agent_state'].pop('optimizer',None)
    destination = output/'models'/f'seed-{seed}-ep-{episode}.pth'
    destination.parent.mkdir(parents=True,exist_ok=True)
    temporary = destination.with_suffix('.pth.tmp')
    torch.save(compact,temporary)
    os.replace(temporary,destination)
    info = dict(model_seed=seed,episode=episode,file=str(destination.relative_to(output)),
                source=str(original),source_sha256=original_sha,snapshot_sha256=sha(destination),
                weights_sha256=parameter_digest(agent),resumable_training_checkpoint=False)
    training.write(manifest,info)
    return info


def prepare(output):
    parent = training.prepare(training.p.OUTPUT)
    replicates.prepare(replicates.OUTPUT)
    paths = [*Path(__file__).parent.glob('*.py'),Path(__file__).with_name('run.sh'),
             Path(__file__).with_name('README.txt'),ROOT/'tests/test_vcg_e140_behavior_eval.py',
             replicates.OUTPUT/'contract.json',training.p.OUTPUT/'contract.json']
    cases = []
    for index in range(6):
        path = training.p.OUTPUT/'instances'/f'cell-{index}-seed-{training.p.DEV_BASE+index}.json'
        if not path.is_file():
            raise RuntimeError('fixed E138 development instance missing')
        cases.append(dict(index=index,path=str(path.relative_to(ROOT)),sha256=sha(path)))
    value = dict(protocol='vcg_conditioned_e140_frozen_grid_behavior_comparison_204k_v1',
        parent_contract_sha256=parent['contract_sha256'],cases=cases,
        scope='reused development panel; not fresh test evidence',
        epsilon=0,optimizer_updates=0,device='cpu',wall_seconds_per_row=900,
        max_decisions=1000,max_nodes=2000,baselines=list(BASELINES),
        model_policy='seeds0/1 episode42; optional seed2 terminal endpoint explicitly labelled',
        caveat='native baseline compound retrieval differs from elementary certified macro set',
        test_panel_opened=False,source_sha256={str(x.relative_to(ROOT)):sha(x) for x in paths})
    value['contract_sha256'] = hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()
    path = output/'contract.json'
    if path.exists() and training.read(path) != value:
        raise RuntimeError('evaluation source/contract changed; use a new output')
    if not path.exists():
        training.write(path,value)
    return value


def load_case(cell):
    path = ROOT/cell['path']
    if sha(path) != cell['sha256']:
        raise RuntimeError('development instance changed')
    return training.read(path)


def behavior(env):
    jobs = []
    for b in env.blocks:
        error = b.delivered_time_step-env.public_due_date(b) if b.delivered else None
        jobs.append(dict(label=b.label,release=b.arrival_step,due=env._external_dates[b.label],
                         initially_stored=b.label in env.initial_labels,first_storage=b.stored_time_step,
                         delivery=b.delivered_time_step,error=error))
    delivered = [x for x in jobs if x['error'] is not None]
    return dict(jobs=jobs,early_beyond_20=sum(x['error'] < -20 for x in delivered),
                late_beyond_20=sum(x['error'] > 20 for x in delivered),steps=env.time_steps,
                steps_per_delivery=env.time_steps/len(delivered) if delivered else None,
                action_counts=dict(env.action_counts))


def run_row(output,contract,label,cell,info=None):
    if info and sha(output/info['file']) != info['snapshot_sha256']:
        raise RuntimeError('frozen model file changed')
    path = output/'rows'/label/f'cell-{cell["index"]}.json'
    if path.exists():
        old = training.read(path)
        if old['contract_sha256'] != contract['contract_sha256']:
            raise RuntimeError('saved row contract differs')
        return old
    random.seed(201000);np.random.seed(201000);torch.manual_seed(201000)
    env = rollout.RecordedGridEnv(load_case(cell))
    seed = info['model_seed'] if info else None
    agent = None
    if info:
        agent = model_class(seed).from_checkpoint(torch.load(output/info['file'],map_location='cpu',weights_only=False),
                                                 device='cpu',resumable=False)
        agent.Q_local.eval().requires_grad_(False)
        agent.Q_target.eval().requires_grad_(False)
    before = parameter_digest(agent) if agent else None
    updates_before = agent.gradient_steps if agent else None
    last = time.monotonic()-30
    def emit(state):
        nonlocal last
        if time.monotonic()-last >= 30:
            print(f'E140 {label} regime={cell["index"]+1}/6 decision={state["decisions"]} '
                  f'step={env.time_steps} delivered={sum(b.delivered for b in env.blocks)}/{len(env.blocks)}',flush=True)
            last = time.monotonic()
    def bound(live_env):
        agent.bind_context_environment(live_env,wait_cap=80,required_jobs=len(live_env.blocks),max_steps=6000)
        agent.reset_episode_state()
        return agent
    env.rollout_state = {'decisions':0,'latencies':[],'reason':None}
    status,error,interrupted = 'failed',None,False
    started = time.monotonic()
    try:
        with training.wall_limit(900), torch.no_grad():
            if agent:
                with patch.object(rollout.GridAgent,'bound',bound):
                    state = rollout.run(env,'scratch_vcg_interface',max_decisions=1000,max_nodes=2000,emit=emit)
            else:
                state = rollout.run(env,label,max_decisions=1000,max_nodes=2000,emit=emit)
        status = 'complete' if all(b.delivered for b in env.blocks) and not state['reason'] else 'incomplete'
    except training.prior.WallTimeout as exc:
        status,error = 'bounded_timeout',str(exc)
    except KeyboardInterrupt:
        status,error,interrupted = 'interrupted','user interruption',True
    except Exception as exc:
        status,error = 'failed',f'{type(exc).__name__}: {exc}'
    metrics = None
    try:
        metrics = rollout.collect(env,status=='complete')
    except Exception as exc:
        status,error = 'failed',f'accounting: {type(exc).__name__}: {exc}'
    unchanged = not agent or (before==parameter_digest(agent) and len(agent.replay)==0
                             and updates_before==agent.gradient_steps)
    if not unchanged:
        raise RuntimeError('evaluation changed parameters or populated replay')
    row = dict(label=label,regime=cell['index'],model_seed=seed,
        checkpoint_episode=info['episode'] if info else None,checkpoint_sha256=info['snapshot_sha256'] if info else None,
        contract_sha256=contract['contract_sha256'],case_sha256=cell['sha256'],status=status,error=error,
        metrics=metrics,behavior=behavior(env),rollout=env.rollout_state,
        primitive_records=env.primitive_records,wall_seconds=time.monotonic()-started,
        no_learning=True,parameters_unchanged=unchanged,test_panel_opened=False)
    training.write(path,row)
    print(f'E140 {label} regime={cell["index"]+1} {status} '
          f'delivered={sum(b.delivered for b in env.blocks)}/{len(env.blocks)}',flush=True)
    if interrupted:
        raise KeyboardInterrupt
    return row


def analyze(output,contract):
    rows = [training.read(x) for x in sorted((output/'rows').glob('*/*.json'))]
    for r in rows:
        if r['contract_sha256'] != contract['contract_sha256']:
            raise RuntimeError('row contract differs')
    expected = 6*(len(BASELINES)+len(list((output/'models').glob('seed-*.json'))))
    training.write(output/'e140-report.json',dict(rows=rows,observed_rows=len(rows),expected_rows=expected,
        scope='descriptive matched development comparison; one instance per regime',
        test_panel_opened=False,no_training=True))
    lines = ['# Frozen-checkpoint behavior on matched development instances','',
        'Lower J is better. Failures remain in completion counts; incomplete costs are not ranked.','',
        '| Controller | Regime | Status | Delivered | J | Wait* | Rehandles | Early / late >20 |',
        '|---|---:|---|---:|---:|---:|---:|---:|']
    for r in rows:
        m=r['metrics'] or {};b=r['behavior'];c=m.get('components',{})
        j=f'{m["J"]:.1f}' if m.get('cost_ranking_eligible') else '—'
        w=m.get('mean_observed_admission_wait');w=f'{w:.1f}' if w is not None else '—'
        lines.append(f'| {r["label"]} | {r["regime"]+1} | {r["status"]} | '
                     f'{m.get("delivered","?")}/{m.get("required_jobs","?")} | {j} | {w} | '
                     f'{c.get("rehandles","—")} | {b["early_beyond_20"]} / {b["late_beyond_20"]} |')
    lines += ['', '*Wait is release to first completed storage; initial inventory excluded. '
              'For incomplete rows it covers observed admissions only; censored waits remain in JSON.',
              'Regimes 1/2: empty .60/.85; 3/4: 40% .60/.85; 5/6: approximately70% .60/.85.',
              'Native plain-baseline versus certified macro mismatch remains a limitation.']
    (output/'e140-table.md').write_text('\n'.join(lines)+'\n')
    return dict(status='complete' if len(rows)==expected else 'partial',
                observed_rows=len(rows),expected_rows=expected,complete_rows=sum(x['status']=='complete' for x in rows),
                report=str(output/'e140-report.json'),table=str(output/'e140-table.md'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('prepare','run','status','add-seed2'))
    parser.add_argument('--output',type=Path,default=OUTPUT)
    args=parser.parse_args()
    torch.set_num_threads(1)
    def terminate(*_):raise KeyboardInterrupt('termination requested')
    signal.signal(signal.SIGTERM,terminate)
    output=args.output.resolve()
    contract=prepare(output)
    if args.command in ('prepare','run'):
        for seed in (0,1):freeze(output,seed)
    if args.command=='add-seed2':freeze(output,2)
    if args.command in ('run','add-seed2'):
        try:
            for label in BASELINES:
                for cell in contract['cases']:run_row(output,contract,label,cell)
            for manifest in sorted((output/'models').glob('seed-*.json')):
                info=training.read(manifest)
                label=f'vcg_seed{info["model_seed"]}_ep{info["episode"]}'
                for cell in contract['cases']:run_row(output,contract,label,cell,info)
        except KeyboardInterrupt:
            analyze(output,contract)
            raise SystemExit(130)
    print(json.dumps(analyze(output,contract),indent=2))


if __name__=='__main__':main()
