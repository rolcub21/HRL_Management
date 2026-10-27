"""User-launched warm-start adaptation; original checkpoints remain read-only."""
import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import random
import signal
import time

import torch
from experiments.conditioned_vcg.E138_residual_grid_mixture_pilot_202k_v1 import run as base
from experiments.conditioned_vcg.E141_original_frozen_workload_transfer_205k_v1 import run as original
from experiments.conditioned_vcg.E131_nominal_cost_residual_pilot_195k_v1.model import EpisodicViabilityGraphConfig
from experiments.conditioned_vcg.E142_frozen_regime_gradient_audit.run import file_sha, digest

ROOT = base.p.ROOT
OUTPUT = ROOT/'results/vcg-conditioned-e144-warm-start-mixture'
MAXIMUM = 360


def regime(number):
    if not 1 <= number <= MAXIMUM:
        raise ValueError('episode outside adaptation budget')
    block, index = divmod(number-1, 6)
    order = list(range(6))
    random.Random(138000+block).shuffle(order)
    return order[index]


class WarmAgent(original.FrozenOriginal):
    def _td_batch(self, transitions):
        result = super()._td_batch(transitions)
        if float((result[0].detach()-result[1]).abs().max()) > 1e6:
            raise RuntimeError('TD error exceeded unchanged mixture sanity gate')
        return result

    def bind_context_environment(self, env, **kwargs):
        # Finite-episode accounting permits 6000 steps; network observations
        # retain the source's 3600 units. Never silently re-normalize a warm start.
        kwargs['max_steps'] = 3600
        return super(original.FrozenOriginal, self).bind_context_environment(env, **kwargs)

    def checkpoint_metadata(self):
        return {**super().checkpoint_metadata(), 'e144_warm_start':True,
                'e144_input_normalization':3600, 'deployment_checkpoint_eligible':False,
                'paper_evidence':False}

    @classmethod
    def from_checkpoint(cls, payload, **kwargs):
        if not payload.get('e144_warm_start'):
            raise ValueError('Not an E144 resume checkpoint')
        return super().from_checkpoint(payload, **kwargs)


def initialize(payload, device='cpu'):
    config = replace(EpisodicViabilityGraphConfig.from_dict(payload['config']),
                     episode_horizon_steps=6000)
    agent = WarmAgent(config=config, seed=0, device=device, epsilon=.9,
        variant='residual_q', arm='A_outflow_bounded', absolute_progress=False)
    agent.Q_local.load_state_dict(payload['agent_state']['Q_local'])
    agent.Q_target.load_state_dict(payload['agent_state']['Q_target'])
    return agent


def prepare(output):
    folder = original.OUTPUT
    info = base.read(folder/'models/seed-0.json')
    source = folder/info['file']
    if file_sha(source) != info['snapshot_sha256'] or info['episode'] != 60:
        raise ValueError('Frozen original episode-60 source authentication failed')
    prior = base.read(base.p.OUTPUT/'contract.json')
    cases = base.read(folder/'contract.json')['cases']
    baselines = []
    for index in range(6):
        path = folder/f'rows/certified_dynamic/cell-{index+5}.json'
        row = base.read(path)
        if row['status'] != 'complete' or row['case_sha256'] != cases[index+5]['sha256']:
            raise ValueError('Certified Dynamic reference identity/completion differs')
        baselines.append(dict(regime=index, J=row['metrics']['J'], path=str(path), sha256=file_sha(path)))
    sources = [Path(__file__), Path(__file__).with_name('README.txt'),
        ROOT/'tests/test_vcg_e144_warm_start.py', Path(base.__file__),
        Path(original.__file__), ROOT/'experiments/conditioned_vcg/E138_residual_grid_mixture_pilot_202k_v1/learner.py',
        ROOT/'experiments/conditioned_vcg/E131_nominal_cost_residual_pilot_195k_v1/model.py',
        ROOT/'experiments/conditioned_vcg/E115_release_et_cost_vcg_8x8_179k_v1/model.py',
        ROOT/'experiments/conditioned_vcg/E123_absolute_progress_8x8_187k_v2/acceleration.py']
    value = dict(protocol='e144_warm_start_mixture_v1', source=str(source),
        source_sha256=info['snapshot_sha256'], source_episode=60, model_seed=0,
        maximum_adaptation_episodes=360, episodes_per_regime=60,
        calibrated_cells=prior['calibrated_cells'], development_cases=cases,
        certified_reference=baselines, initialization='online and target weights only; fresh optimizer/replay/counters',
        exploration='unchanged mixture: .9 to .05 over 6000 adaptation decisions',
        objective='unchanged J=2T+E+0.1L+10R; no normalization or head reweighting',
        training_seeds='21000001..21000360; first60 same workload/regime schedule as E138',
        validation_endpoints=[0,6,*range(30,361,30)],
        criterion='all validation rows complete; easy J <=1.10*warm-start endpoint0 easy J; '
                  'each congested regime J <= matched certified Dynamic J; '
                  'both last endpoints330 and360 must pass; otherwise no seed expansion',
        input_normalization=3600, execution_horizon=6000,
        runtime=dict(training_wall_seconds=1200, validation_wall_seconds=900,
                     macros=1000, certificate_nodes=2000),
        validation_failure='record incompletion/timeout and continue training; unexpected errors stop',
        training_failure='stop and roll back unfinished episode; no automatic retries',
        limitation='one changed-workload development instance per regime; five native controls; '
                   'adaptation, not zero-shot generalization; no uniform completion-bound claim',
        test_panel_opened=False, stage_b_authorized=False,
        sources={str(p.relative_to(ROOT)):file_sha(p) for p in sources})
    value['contract_sha256'] = hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()
    value = json.loads(json.dumps(value))
    path = output/'contract.json'
    if path.exists() and base.read(path) != value:
        raise ValueError('Contract changed; use a new output')
    if not path.exists():
        base.write(path,value)
    return value


def save(output, contract, agent, history, validations):
    directory = output/'seed-0'
    directory.mkdir(parents=True,exist_ok=True)
    payload = agent.checkpoint(include_replay=True, e144_contract_sha256=contract['contract_sha256'],
        completed_training_episodes=len(history), training_history=history, validation_history=validations,
        global_rng_state=base.audit._capture_global_rng_state(), test_panel_opened=False)
    torch.save(payload,directory/'latest.pth.tmp')
    os.replace(directory/'latest.pth.tmp',directory/'latest.pth')
    base.write(directory/'summary.json',dict(completed_adaptation_episodes=len(history),
        source_training_episodes=60, maximum_adaptation_episodes=360,
        episodes_per_regime={r:sum(h['regime']==r for h in history) for r in range(6)},
        epsilon=base.p.epsilon(agent.decision_count), gradient_steps=agent.gradient_steps,
        validation_history=validations, test_panel_opened=False))


def validate(output, contract, agent, endpoint):
    rng = base.audit._capture_global_rng_state()
    try:
        evaluator = WarmAgent.from_checkpoint(agent.checkpoint(include_replay=False), device=agent.device)
        evaluator.Q_local.eval().requires_grad_(False)
        rows = []
        with torch.no_grad():
            for case_info in contract['development_cases']:
                index = case_info['index']
                dest = output/f'validation/ep-{endpoint}-case-{index}.json'
                if dest.exists():
                    rows.append(base.read(dest))
                    continue
                path = Path(case_info['path'])
                if file_sha(path) != case_info['sha256']:
                    raise ValueError('Validation case changed')
                env = base.RecordedGridEnv(base.read(path))
                print(f'E144 greedy adaptation_ep={endpoint} case={index+1}/11',flush=True)
                try:
                    with base.wall_limit(900):
                        row = base.episode(evaluator,env,training=False)
                except base.prior.WallTimeout as exc:
                    row = dict(strict_complete=False,reason=str(exc),metrics=None)
                row.update(case=index,panel=case_info['panel'])
                base.write(dest,row)
                rows.append(row)
        easy = next(r for r in rows if r['case']==5)
        origin = next((v for v in base.read(output/'seed-0/summary.json')['validation_history']
                       if v['episode']==0),None) if endpoint else None
        easy_reference = origin['easy_J'] if origin else (easy['metrics']['J'] if easy['metrics'] else None)
        passed = all(r['strict_complete'] for r in rows)
        if easy_reference is None or easy['metrics'] is None:
            passed = False
        else:
            passed = passed and easy['metrics']['J'] <= 1.10*easy_reference
        for baseline in contract['certified_reference'][2:]:
            row = next(r for r in rows if r['case']==baseline['regime']+5)
            passed = passed and row['metrics'] is not None and row['metrics']['J'] <= baseline['J']
        return dict(episode=endpoint, rows=rows, easy_J=easy['metrics']['J'] if easy['metrics'] else None,
                    easy_reference_J=easy_reference, passes_development_gate=bool(passed))
    finally:
        base.audit._restore_global_rng_state(rng)


def train(output, contract, through, device):
    if not 1 <= through <= MAXIMUM:
        raise ValueError('endpoint outside1..360')
    if device != 'cuda' or not torch.cuda.is_available():
        raise RuntimeError('Long training requires CUDA')
    if (output/'stopped.json').exists():
        raise RuntimeError('Previous stop retained; inspect before a separately authorized retry')
    path = output/'seed-0/latest.pth'
    if path.exists():
        payload = torch.load(path,map_location='cpu',weights_only=False)
        if payload.get('e144_contract_sha256') != contract['contract_sha256']:
            raise ValueError('Resume contract differs')
        agent = WarmAgent.from_checkpoint(payload,device=device,resumable=True)
        base.audit._optimizer_to(agent.optimizer,torch.device(device))
        history, validations = payload['training_history'], payload['validation_history']
        base.audit._restore_global_rng_state(payload['global_rng_state'])
    else:
        base.seed_everything(0)
        payload = torch.load(contract['source'],map_location='cpu',weights_only=False)
        agent = initialize(payload,device)
        history, validations = [], []
        save(output,contract,agent,history,validations)
    if len(history)>through:
        raise ValueError('Endpoint precedes checkpoint')
    number = len(history)
    try:
        with base.cache_session():
            def endpoint_validation(number):
                if number in contract['validation_endpoints'] and not any(v['episode']==number for v in validations):
                    validations.append(validate(output,contract,agent,number))
                    save(output,contract,agent,history,validations)
            endpoint_validation(number)
            for number in range(len(history)+1,through+1):
                index = regime(number)
                env = base.RecordedGridEnv(base.workload(output,contract,index,21000000+number))
                print(f'E144 adaptation_ep={number}/360 regime={index} eps={base.p.epsilon(agent.decision_count):.3f}',flush=True)
                last_log = time.monotonic()
                def progress(row):
                    nonlocal last_log
                    if time.monotonic()-last_log >=30:
                        print(f'E144 ep={number} step={row["steps"]} delivered={row["delivered"]}/{row["required"]}',flush=True)
                        base.write(output/'progress.json',dict(episode=number,regime=index,**row))
                        last_log = time.monotonic()
                with base.wall_limit(1200):
                    row = base.episode(agent,env,training=True,emit=progress)
                if not row['strict_complete']:
                    raise RuntimeError(f'Training incomplete: {row["reason"]}')
                row.update(episode=number,regime=index,instance_seed=21000000+number)
                history.append(row)
                save(output,contract,agent,history,validations)
                print(f'E144 ep={number} complete J={row["metrics"]["J"]:.1f} wall={row["wall_seconds"]:.1f}s',flush=True)
                endpoint_validation(number)
    except (Exception,KeyboardInterrupt) as exc:
        base.write(output/'stopped.json',dict(episode=number,error=f'{type(exc).__name__}: {exc}',
            resume_checkpoint=str(path),unfinished_episode_updates_not_saved=True))
        raise
    return dict(status='paused_at_requested_endpoint' if through<360 else 'complete',
                completed_adaptation_episodes=len(history), latest_checkpoint=str(path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('prepare','train','status'))
    parser.add_argument('--through-episode',type=int,default=6)
    parser.add_argument('--output',type=Path,default=OUTPUT)
    parser.add_argument('--device',choices=('cuda',),default='cuda')
    args = parser.parse_args()
    torch.set_num_threads(1)
    def terminate(*_):
        raise KeyboardInterrupt('termination requested')
    signal.signal(signal.SIGTERM,terminate)
    contract = prepare(args.output)
    if args.command=='train':
        result = train(args.output,contract,args.through_episode,args.device)
    elif args.command=='status':
        path = args.output/'seed-0/summary.json'
        result = base.read(path) if path.exists() else dict(status='not_started')
        result.pop('validation_history',None)
    else:
        result = dict(status='prepared_not_launched',maximum_adaptation_episodes=360)
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
