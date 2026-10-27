import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import signal
import time
from unittest.mock import patch

import torch
from experiments.conditioned_vcg.E138_residual_grid_mixture_pilot_202k_v1 import run as base
from experiments.conditioned_vcg.E138_residual_grid_mixture_pilot_202k_v1 import protocol as p
from experiments.conditioned_vcg.E138_residual_grid_mixture_pilot_202k_v1.learner import MixtureAgent
from experiments.conditioned_vcg.E131_nominal_cost_residual_pilot_195k_v1.model import EpisodicViabilityGraphConfig

OUTPUT = p.ROOT/'results/vcg-conditioned-e139-grid-mixture-three-seeds-203k-v1'


def agent_class(seed):
    if seed not in (1,2):
        raise ValueError('E139 adds only seeds 1 and 2; seed 0 stays under E138')
    class ReplicateAgent(MixtureAgent):
        def checkpoint_metadata(self):
            return {**super().checkpoint_metadata(),'e139_model_seed':seed}

        @classmethod
        def from_checkpoint(cls,payload,**kwargs):
            if payload.get('e139_model_seed') != seed:
                raise ValueError('replicate checkpoint seed differs')
            kwargs['seed'] = seed
            return super().from_checkpoint(payload,**kwargs)
    return ReplicateAgent


def fresh(seed,device='cpu',config=None):
    return agent_class(seed)(config=config or EpisodicViabilityGraphConfig(gamma=1.,episode_horizon_steps=6000),
        seed=seed,device=device,epsilon=.9,arm='A_outflow_bounded',absolute_progress=False,variant='residual_q')


def prepare(output):
    parent = base.prepare(p.OUTPUT)
    files = [*Path(__file__).parent.glob('*.py'),Path(__file__).with_name('run.sh'),
             Path(__file__).with_name('night.sh'),Path(__file__).with_name('README.txt'),
             p.ROOT/'tests/test_vcg_e139_replicates.py']
    value = dict(protocol='vcg_conditioned_e139_grid_mixture_three_seeds_203k_v1',
        parent_contract_sha256=parent['contract_sha256'],additional_model_seeds=[1,2],
        unchanged_recipe=parent['specification'],calibrated_cells=parent['calibrated_cells'],
        randomness='different initialization/exploration/replay/global RNG; matched workload seeds and regime order',
        reporting='retain all seeds including failures; never erase/select only successful seeds',
        queue='wait for active seed 0; finish/resume seed 0; run 1 then 2; failed seed stops but next independent seed may run',
        test_panel_opened=False,source_sha256={str(x.relative_to(p.ROOT)):base.prior.sha(x) for x in files})
    value['contract_sha256'] = hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()
    value = json.loads(json.dumps(value))
    path = output/'contract.json'
    if path.exists() and base.read(path) != value:
        raise RuntimeError('E139 source/recipe differs; use new output')
    if not path.exists():
        base.write(path,value)
    return value


def save(directory,contract,agent,history,validations):
    payload = agent.checkpoint(include_replay=True,e139_contract_sha256=contract['contract_sha256'],
        training_history=history,validation_history=validations,completed_training_episodes=len(history),
        global_rng_state=base.audit._capture_global_rng_state(),test_panel_opened=False)
    directory.mkdir(parents=True,exist_ok=True)
    temporary = directory/'latest.pth.tmp'
    torch.save(payload,temporary)
    os.replace(temporary,directory/'latest.pth')
    base.write(directory/'summary.json',dict(model_seed=agent.seed,completed_training_episodes=len(history),
        training_decisions=agent.decision_count,gradient_steps=agent.gradient_steps,
        epsilon=p.epsilon(agent.decision_count),validation_endpoints=[v['episode'] for v in validations],
        training_history=history,validation_history=validations,test_panel_opened=False))


def validate(directory,contract,agent,endpoint,seed):
    # Reuse the frozen evaluator but restore using the correct model-seed class.
    with patch.object(base,'MixtureAgent',agent_class(seed)):
        return base.validation(directory,contract,agent,endpoint)


def train(output,contract,seed,through,device):
    if not 1 <= through <= 60:
        raise ValueError('endpoint outside 1..60')
    if device != 'cuda' or not torch.cuda.is_available():
        raise RuntimeError('Long runs require CUDA')
    cls = agent_class(seed)
    directory = output/f'seed-{seed}'
    if (directory/'stopped.json').exists():
        raise RuntimeError('Previous stop retained; inspect it before a separately authorized retry')
    path = directory/'latest.pth'
    if path.exists():
        payload = torch.load(path,map_location='cpu',weights_only=False)
        if payload.get('e139_contract_sha256') != contract['contract_sha256']:
            raise RuntimeError('resume contract differs')
        agent = cls.from_checkpoint(payload,device=device,resumable=True)
        base.audit._optimizer_to(agent.optimizer,torch.device(device))
        history,validations = payload['training_history'],payload['validation_history']
        base.audit._restore_global_rng_state(payload['global_rng_state'])
    else:
        base.seed_everything(seed)
        agent,history,validations = fresh(seed,device),[],[]
        save(directory,contract,agent,history,validations)
    if len(history)>through:
        raise ValueError('endpoint precedes saved checkpoint')
    number = len(history)
    try:
        with base.cache_session():
            if number and number%6 == 0 and not any(v['episode']==number for v in validations):
                validations.append(validate(directory,contract,agent,number,seed))
                save(directory,contract,agent,history,validations)
            for number in range(len(history)+1,through+1):
                index = p.regime(number)
                print(f'E139 seed={seed} train ep={number}/60 regime={index+1}/6 '
                      f'eps={p.epsilon(agent.decision_count):.3f}',flush=True)
                case = base.workload(directory,contract,index,p.TRAIN_BASE+number)
                env = base.RecordedGridEnv(case)
                last = time.monotonic()
                def progress(row):
                    nonlocal last
                    if time.monotonic()-last >= 30:
                        print(f'E139 seed={seed} ep={number} decision={row["decisions"]} '
                              f'step={row["steps"]} delivered={row["delivered"]}/{row["required"]}',flush=True)
                        base.write(directory/'progress.json',dict(episode=number,model_seed=seed,**row))
                        last = time.monotonic()
                with base.wall_limit(1200):
                    row = base.episode(agent,env,training=True,emit=progress)
                row.update(episode=number,regime=index,instance_seed=p.TRAIN_BASE+number,model_seed=seed)
                if not row['strict_complete']:
                    raise RuntimeError(f'training incomplete: {row["reason"]}')
                history.append(row)
                save(directory,contract,agent,history,validations)
                print(f'E139 seed={seed} ep={number} complete J={row["metrics"]["J"]:.1f} '
                      f'elapsed={row["wall_seconds"]:.1f}s',flush=True)
                if number%6 == 0:
                    validations.append(validate(directory,contract,agent,number,seed))
                    save(directory,contract,agent,history,validations)
    except (Exception,KeyboardInterrupt) as exc:
        base.write(directory/'stopped.json',dict(model_seed=seed,episode=number,error=f'{type(exc).__name__}: {exc}',
                   resume_checkpoint=str(path),incomplete_in_memory_updates_not_saved=True))
        raise
    return dict(model_seed=seed,completed_training_episodes=len(history),
                validation_endpoints=[v['episode'] for v in validations],status='endpoint_complete')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('prepare','train','status'))
    parser.add_argument('--model-seed',type=int,choices=(1,2),default=1)
    parser.add_argument('--through-episode',type=int,default=60)
    parser.add_argument('--device',choices=('cuda',),default='cuda')
    parser.add_argument('--output',type=Path,default=OUTPUT)
    args = parser.parse_args()
    torch.set_num_threads(1)
    def terminate(*_):
        raise KeyboardInterrupt('termination requested')
    signal.signal(signal.SIGTERM,terminate)
    output = args.output.resolve()
    contract = prepare(output)
    if args.command == 'train':
        value = train(output,contract,args.model_seed,args.through_episode,args.device)
    elif args.command == 'status':
        value = {}
        for seed in (1,2):
            path = output/f'seed-{seed}/summary.json'
            row = base.read(path) if path.exists() else {'status':'not_started'}
            value[str(seed)] = {k:v for k,v in row.items() if k not in ('training_history','validation_history')}
            stop = output/f'seed-{seed}/stopped.json'
            if stop.exists(): value[str(seed)]['stop'] = base.read(stop)
    else:
        value = dict(status='prepared',additional_seeds=[1,2],training_runs=0,test_panel_opened=False)
    print(json.dumps(value,indent=2))


if __name__ == '__main__':
    main()
