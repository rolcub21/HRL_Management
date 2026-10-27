"""Scratch mixture pilot. Long training/validation runs are user launched."""
import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import signal
import time

import torch
from train_viability_graph_smdp import seed_everything
from experiments.conditioned_vcg.E137_grid_rollout_readiness_201k_v1 import run as prior
from experiments.conditioned_vcg.E135_independent_grid_reference_199k_v1 import run as generator
from experiments.conditioned_vcg.E123_absolute_progress_8x8_187k_v2.acceleration import cache_session
from experiments.conditioned_vcg.E123_absolute_progress_8x8_187k_v2 import run as shared
from . import protocol as p
from .learner import fresh, MixtureAgent, episode, RecordedGridEnv

read, write = prior.read, prior.write
audit = shared.original.replay.audit


@contextmanager
def wall_limit(seconds):
    def expired(*_):
        raise prior.WallTimeout(f'episode exceeded {seconds}s')
    old = signal.signal(signal.SIGALRM,expired)
    signal.setitimer(signal.ITIMER_REAL,seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL,0.)
        signal.signal(signal.SIGALRM,old)


def prepare(output):
    prior.prepare(prior.OUTPUT)
    report = read(prior.OUTPUT/'e137-report.json')
    if report['observed_rows'] != 24:
        raise RuntimeError('E137 readiness panel must have all 24 recorded outcomes')
    paths = [*Path(__file__).parent.glob('*.py'),Path(__file__).with_name('README.txt'),
             Path(__file__).with_name('run.sh'),p.ROOT/'tests/test_vcg_e138_mixture_pilot.py',
             prior.OUTPUT/'contract.json',prior.OUTPUT/'e137-report.json',
             p.ROOT/'experiments/conditioned_vcg/E137_grid_rollout_readiness_201k_v1/rollout.py']
    value = {'specification':p.specification(), 'calibrated_cells':prior.cells(),
             'sources':{str(x.relative_to(p.ROOT)):prior.sha(x) for x in paths}}
    value['contract_sha256'] = hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()
    value = json.loads(json.dumps(value))
    path = output/'contract.json'
    if path.exists() and read(path) != value:
        raise RuntimeError('E138 source/contract changed; choose a new output')
    if not path.exists():
        write(path,value)
    return value


def workload(output, contract, index, seed):
    cell = contract['calibrated_cells'][index]
    source = read(p.ROOT/cell['path'])
    path = output/'instances'/f'cell-{index}-seed-{seed}.json'
    if path.exists():
        case = read(path)
        expected = case.pop('e138_case_sha256',None)
        if expected != hashlib.sha256(json.dumps(case,sort_keys=True).encode()).hexdigest():
            raise RuntimeError('stored workload content changed')
        if case['seed'] != seed or case['fill'] != cell['fill'] or case['gap'] != source['gap']:
            raise RuntimeError('stored workload identity differs')
    else:
        case = generator.case(seed,cell['fill'],source['gap'])
        digest = hashlib.sha256(json.dumps(case,sort_keys=True).encode()).hexdigest()
        write(path,{**case,'e138_case_sha256':digest})
    return case


def save(output,contract,agent,history,validations):
    payload = agent.checkpoint(include_replay=True,e138_contract_sha256=contract['contract_sha256'],
        training_history=history,validation_history=validations,
        completed_training_episodes=len(history),global_rng_state=audit._capture_global_rng_state(),
        test_panel_opened=False)
    if payload.get('deployment_checkpoint_eligible') is not False or payload.get('paper_evidence') is not False:
        raise RuntimeError('pilot checkpoint must remain ineligible for deployment/paper evidence')
    directory = output/'seed-0'
    directory.mkdir(parents=True,exist_ok=True)
    path = directory/'latest.pth'
    temporary = directory/'latest.pth.tmp'
    torch.save(payload,temporary)
    os.replace(temporary,path)
    write(directory/'summary.json',dict(completed_training_episodes=len(history),
        training_decisions=agent.decision_count,gradient_steps=agent.gradient_steps,
        epsilon=p.epsilon(agent.decision_count),validation_endpoints=[v['episode'] for v in validations],
        training_history=history,validation_history=validations,
        deployment_checkpoint_eligible=False,test_panel_opened=False))


def validation(output,contract,agent,endpoint):
    rng = audit._capture_global_rng_state()
    try:
        evaluator = MixtureAgent.from_checkpoint(agent.checkpoint(include_replay=False),device=agent.device)
        evaluator.Q_local.eval().requires_grad_(False)
        evaluator.Q_target.eval().requires_grad_(False)
        rows = []
        with torch.no_grad():
            for index in range(6):
                print(f'E138 greedy checkpoint={endpoint} regime={index+1}/6',flush=True)
                case = workload(output,contract,index,p.DEV_BASE+index)
                env = RecordedGridEnv(case)
                try:
                    with wall_limit(300):
                        row = episode(evaluator,env,training=False)
                except (Exception,KeyboardInterrupt) as exc:
                    failure = dict(strict_complete=False,regime=index,instance_seed=p.DEV_BASE+index,
                                   error=f'{type(exc).__name__}: {exc}',steps=env.time_steps)
                    write(output/'validation'/f'ep-{endpoint}-cell-{index}-failure-{time.time_ns()}.json',failure)
                    raise
                row.update(regime=index,instance_seed=p.DEV_BASE+index)
                write(output/'validation'/f'ep-{endpoint}-cell-{index}.json',row)
                rows.append(row)
                if not row['strict_complete']:
                    raise RuntimeError('greedy validation incomplete; stopped by frozen rule')
        return dict(episode=endpoint,rows=rows)
    finally:
        audit._restore_global_rng_state(rng)


def train(output,contract,through,device,retry_failed):
    if not 1 <= through <= 60:
        raise ValueError('through-episode must be between 1 and 60')
    if device != 'cuda' or not torch.cuda.is_available():
        raise RuntimeError('Long pilot training requires available CUDA; unit smoke only on CPU')
    stopped = output/'stopped.json'
    if stopped.exists():
        if not retry_failed:
            raise RuntimeError('Previous stop recorded; inspect stopped.json before explicit --retry-failed')
        stopped.rename(output/f'acknowledged-stop-{time.time_ns()}.json')
    path = output/'seed-0/latest.pth'
    if path.exists():
        payload = torch.load(path,map_location='cpu',weights_only=False)
        if payload.get('e138_contract_sha256') != contract['contract_sha256']:
            raise RuntimeError('resume contract differs')
        agent = MixtureAgent.from_checkpoint(payload,device=device,resumable=True)
        audit._optimizer_to(agent.optimizer,torch.device(device))
        history, validations = payload['training_history'],payload['validation_history']
        audit._restore_global_rng_state(payload['global_rng_state'])
    else:
        seed_everything(0)
        agent,history,validations = fresh(device),[],[]
        save(output,contract,agent,history,validations)
    if len(history) > through:
        raise ValueError('requested endpoint precedes completed checkpoint')
    number = len(history)
    try:
        with cache_session():
            # Complete a pending validation before any further optimizer steps.
            if number and number%6 == 0 and not any(v['episode']==number for v in validations):
                validations.append(validation(output,contract,agent,number))
                save(output,contract,agent,history,validations)
            for number in range(len(history)+1,through+1):
                index = p.regime(number)
                print(f'E138 train ep={number}/60 regime={index+1}/6 '
                      f'decisions={agent.decision_count} eps={p.epsilon(agent.decision_count):.3f}',flush=True)
                case = workload(output,contract,index,p.TRAIN_BASE+number)
                env = RecordedGridEnv(case)
                last_log = time.monotonic()
                def progress(row):
                    nonlocal last_log
                    if time.monotonic()-last_log >= 30:
                        print(f'E138 ep={number} decision={row["decisions"]} step={row["steps"]} '
                              f'delivered={row["delivered"]}/{row["required"]} eps={row["epsilon"]:.3f}',flush=True)
                        write(output/'progress.json',dict(episode=number,regime=index,**row))
                        last_log = time.monotonic()
                with wall_limit(1200):
                    row = episode(agent,env,training=True,emit=progress)
                row.update(episode=number,regime=index,instance_seed=p.TRAIN_BASE+number)
                if not row['strict_complete']:
                    raise RuntimeError(f'training episode incomplete: {row["reason"]}')
                history.append(row)
                save(output,contract,agent,history,validations)
                print(f'E138 ep={number} complete J={row["metrics"]["J"]:.1f} '
                      f'steps={env.time_steps} elapsed={row["wall_seconds"]:.1f}s',flush=True)
                if number%6 == 0:
                    validations.append(validation(output,contract,agent,number))
                    save(output,contract,agent,history,validations)
    except (Exception,KeyboardInterrupt) as exc:
        write(output/'stopped.json',dict(episode=number,error=f'{type(exc).__name__}: {exc}',
            resume_checkpoint=str(path),completed_training_episodes=len(read(output/'seed-0/summary.json')['training_history']),
            incomplete_in_memory_updates_not_saved=True))
        raise
    return dict(status='paused_at_requested_endpoint' if through<60 else 'pilot_complete',
                completed_training_episodes=len(history),validation_endpoints=[v['episode'] for v in validations],
                latest_checkpoint=str(path),test_panel_opened=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('prepare','train','status'))
    parser.add_argument('--output',type=Path,default=p.OUTPUT)
    parser.add_argument('--through-episode',type=int,default=6)
    parser.add_argument('--device',choices=('cuda',),default='cuda')
    parser.add_argument('--retry-failed',action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(1)
    # SIGTERM follows the same rollback/stop path as Ctrl-C, never continues.
    def terminate(*_):
        raise KeyboardInterrupt('termination requested')
    signal.signal(signal.SIGTERM,terminate)
    output = args.output.resolve()
    contract = prepare(output)
    if args.command == 'train':
        result = train(output,contract,args.through_episode,args.device,args.retry_failed)
    elif args.command == 'status':
        summary = output/'seed-0/summary.json'
        result = read(summary) if summary.exists() else {'status':'prepared_not_started'}
        result = {k:v for k,v in result.items() if k not in ('training_history','validation_history')}
        if (output/'stopped.json').exists():
            result['stop'] = read(output/'stopped.json')
    else:
        result = dict(status='prepared',maximum_episode=60,first_endpoint=6,
                      initialization='scratch',training_runs=0,test_panel_opened=False)
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
