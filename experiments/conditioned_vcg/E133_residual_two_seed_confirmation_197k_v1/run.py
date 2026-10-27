"""Sequential seed-1/2 confirmation; never modifies E131 seed-0 artifacts."""
import argparse
from functools import lru_cache
import json
import math
import os
from pathlib import Path
from statistics import fmean

import torch
from train_viability_graph_smdp import seed_everything
from experiments.conditioned_vcg.E131_nominal_cost_residual_pilot_195k_v1 import run as pilot
from experiments.conditioned_vcg.E131_nominal_cost_residual_pilot_195k_v1.model import PilotAgent,CostRecord,ARMS

ROOT=pilot.ROOT
OUTPUT=ROOT/'results/vcg-conditioned-e133-residual-two-seed-confirmation-197k-v1'
PROTOCOL='vcg_conditioned_e133_residual_two_seed_confirmation_197k_v1'
SEEDS=(1,2)
MAX_EPISODE=60
write,read,require=pilot.write,pilot.read,pilot.require


def train_instance(model_seed,episode):
    require(model_seed in SEEDS and 1<=episode<=MAX_EPISODE,'Invalid confirmation training coordinate')
    return pilot.source.p.train_seed(model_seed,episode)


def schedule():
    return tuple((seed,arm) for seed in SEEDS for arm in ARMS)


@lru_cache(maxsize=1)
def contract():
    pilot.prepare(pilot.OUTPUT)
    inherited=pilot.contract()
    protected=[pilot.OUTPUT/'contract.json',pilot.OUTPUT/'smoke.json',pilot.OUTPUT/'preflight.json']
    protected.extend(pilot.OUTPUT/arm/'seed-0/summary.json' for arm in ARMS)
    sources=[*Path(__file__).parent.glob('*.py'),Path(__file__).with_name('run.sh'),ROOT/'tests/test_vcg_e133_confirmation.py']
    value=dict(protocol=PROTOCOL,pilot_contract_sha256=inherited['contract_sha256'],
        model_seeds=list(SEEDS),arms=list(ARMS),maximum_episode=60,validation_every=6,
        fixed_decision_checkpoints=list(pilot.LATE),validation_seeds=list(pilot.source.p.VALIDATION_SEEDS),
        initialization='fresh paired weights/agent RNG for each seed; no transfer or optimizer inheritance',
        train_seed_formula='187000000 + model_seed*10000 + episode - 1',
        training_instances={str(s):[train_instance(s,e) for e in range(1,61)] for s in SEEDS},
        execution_order=[dict(model_seed=s,arm=a) for s,a in schedule()],
        unchanged='E131 learner, signed heads both arms, K/replay/TD, objective, guard, grouping, optimizer, exploration and budget',
        sanity_gate=inherited['sanity_rule'],minimum_J_improvement=4.,
        primary_confirmation='seeds 1 and 2 only, equal weight per training seed after averaging fixed checkpoints within instance; apply E131 4-J/completion/tail gate',
        discovery_seed0='read-only context; report three-seed aggregate separately, never as independent confirmation',
        independent_units='two new training seeds, five reused development instances; repeated checkpoints are not independent instances',
        test_panel_opened=False,paper_evidence=False,
        protected_pilot_sha256={str(p.relative_to(ROOT)):pilot.sha256(p) for p in protected},
        source_sha256={str(p.relative_to(ROOT)):pilot.sha256(p) for p in sources})
    return {**value,'contract_sha256':pilot.digest(value)}


def directory(output,seed,arm):
    require(seed in SEEDS and arm in ARMS,'Unsupported seed/arm')
    return output/arm/f'seed-{seed}'


def fresh(seed,arm,device='cpu'):
    require(seed in SEEDS and arm in ARMS,'Unsupported scratch identity')
    seed_everything(seed)
    return PilotAgent(config=pilot.source.original.replay.model.EpisodicViabilityGraphConfig(
        gamma=1.,episode_horizon_steps=3600),seed=seed,device=device,epsilon=.9,
        arm=pilot.BASE_ARM,absolute_progress=False,variant=arm)


def prepare(output):
    expected=contract()
    path=output/'contract.json'
    if path.exists(): require(read(path)==expected,'E133 source/contract changed: use fresh version')
    else:
        require(not output.exists() or not any(output.iterdir()),'Need empty contracted output')
        write(path,expected)
    for relative,sha in expected['protected_pilot_sha256'].items():
        require(pilot.sha256(ROOT/relative)==sha,f'Protected pilot artifact changed: {relative}')
    for name in ('preflight','smoke'):
        gate=read(pilot.OUTPUT/f'{name}.json')
        require(gate['passed'] and gate['contract_sha256']==expected['pilot_contract_sha256'],
                f'Frozen E131 {name} gate is not valid')
    if not (output/'preflight.json').exists():
        digests={}
        for seed in SEEDS:
            a,b=fresh(seed,'total_q'),fresh(seed,'residual_q')
            digest_a=pilot.source.original.replay.parameter_digest(a)
            require(digest_a==pilot.source.original.replay.parameter_digest(b),'Within-seed scratch weights differ')
            require(a.rng.getstate()==b.rng.getstate(),'Within-seed sampler RNG differs')
            digests[str(seed)]=digest_a
        require(len(set(digests.values()))==len(SEEDS),'Different model seeds have identical initialization')
        write(output/'preflight.json',dict(contract_sha256=expected['contract_sha256'],passed=True,
            paired_initial_parameter_digests=digests,inherited_physical_and_optimizer_smoke=True,
            training_runs=0,test_panel_opened=False))
    else:
        gate=read(output/'preflight.json')
        require(gate['passed'] and gate['contract_sha256']==expected['contract_sha256'],'E133 preflight differs')


def save(path,agent,seed,history,validations):
    payload=agent.checkpoint(include_replay=True,e133_protocol=PROTOCOL,
        e133_contract_sha256=contract()['contract_sha256'],model_seed=seed,
        completed_training_episodes=len(history),training_history=history,validation_history=validations,
        global_rng_state=pilot.source.original.replay.audit._capture_global_rng_state(),
        deployment_checkpoint_eligible=False,test_panel_opened=False,paper_evidence=False)
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix('.pth.tmp')
    torch.save(payload,temporary)
    os.replace(temporary,path)


def authenticate(payload,seed,arm):
    require(payload.get('e133_protocol')==PROTOCOL
        and payload.get('e133_contract_sha256')==contract()['contract_sha256']
        and payload.get('model_seed')==seed and payload.get('e131_variant')==arm,
        'Resume identity differs; never import seed-0 or another arm/seed checkpoint')


def validate(agent,seed,number):
    audit=pilot.source.original.replay.audit
    rng=audit._capture_global_rng_state()
    digest=pilot.source.original.replay.parameter_digest(agent)
    counters=(agent.decision_count,agent.gradient_steps,len(agent.replay),agent.rng.getstate())
    evaluator=PilotAgent.from_checkpoint(agent.checkpoint(include_replay=False),
        device=agent.device,resumable=False,seed=seed)
    evaluator.Q_local.eval().requires_grad_(False)
    evaluator.Q_target.eval().requires_grad_(False)
    try:
        with torch.no_grad(): runs=[pilot.episode(evaluator,s,False) for s in pilot.source.p.VALIDATION_SEEDS]
    finally: audit._restore_global_rng_state(rng)
    require(digest==pilot.source.original.replay.parameter_digest(agent)
        and counters==(agent.decision_count,agent.gradient_steps,len(agent.replay),agent.rng.getstate()),
        'Greedy validation changed training state')
    return dict(checkpoint_episode=number,runs=runs,model_seed=seed,evaluation_epsilon=0.,evaluation_learning=False)


def train(output,seed,arm,through,device,cpu_smoke=False):
    prepare(output)
    require(device=='cuda' or cpu_smoke and through<=1,'CPU training limited to one explicit smoke episode')
    require(device!='cuda' or torch.cuda.is_available(),'CUDA unavailable')
    audit=pilot.source.original.replay.audit
    folder=directory(output,seed,arm)
    path=folder/'latest.pth'
    if path.exists():
        payload=torch.load(path,map_location='cpu',weights_only=False)
        authenticate(payload,seed,arm)
        agent=PilotAgent.from_checkpoint(payload,device=device,resumable=True,seed=seed)
        audit._optimizer_to(agent.optimizer,torch.device(device))
        history,validations=payload['training_history'],payload['validation_history']
        require(len(history)==payload['completed_training_episodes'],'Checkpoint history length differs')
        audit._restore_global_rng_state(payload['global_rng_state'])
    else:
        agent=fresh(seed,arm,device)
        history,validations=[],[]
        save(path,agent,seed,history,validations)
    require(all(r['strict_method_success'] for r in history),'Saved completion failure: investigate before resuming')
    require(len(history)<=through<=MAX_EPISODE,'Invalid episode endpoint')
    with pilot.source.cache_session():
        for number in range(len(history)+1,through+1):
            if number>36:
                require(pilot.sanity(history,validations,agent.td_diagnostics),'Endpoint-36 sanity gate failed; stop')
            print(f'E133 model_seed={seed} arm={arm} episode={number}/60 '
                  f'decisions={agent.decision_count} epsilon={agent.epsilon:.3f}',flush=True)
            row=pilot.episode(agent,train_instance(seed,number),True)
            row['e133_model_seed']=seed
            history.append(row)
            require(all(isinstance(t.chosen,CostRecord) and all(isinstance(r,CostRecord) for r in t.next_candidates)
                for t in agent.replay.memory),'Nominal costs missing from replay')
            if number%6==0 and row['strict_method_success']: validations.append(validate(agent,seed,number))
            save(path,agent,seed,history,validations)
            write(folder/'summary.json',dict(contract_sha256=contract()['contract_sha256'],model_seed=seed,
                variant=arm,completed_training_episodes=len(history),training_history=history,
                validation_history=validations,training_decisions=agent.decision_count,epsilon=agent.epsilon,
                gradient_steps=agent.gradient_steps,td_diagnostics=agent.td_diagnostics,
                losses=agent.component_loss_history,paper_evidence=False,test_panel_opened=False))
            require(row['strict_method_success'],'Completion failure: endpoint saved; stop and diagnose')
            print(f'E133 saved seed={seed} arm={arm} episode={number} J={row["operator_cost"]:.2f}',flush=True)
    print(f'E133 seed={seed} arm={arm}: saved endpoint={len(history)}',flush=True)


def arm_summary(summary):
    endpoint=summary['completed_training_episodes']
    late=[v for v in summary['validation_history'] if v['checkpoint_episode'] in pilot.LATE]
    require(len({v['checkpoint_episode'] for v in late})==len(late),'Duplicate validation checkpoint')
    expected=set(pilot.source.p.VALIDATION_SEEDS)
    for v in late:
        require(len(v['runs'])==5 and {r['instance_seed'] for r in v['runs']}==expected,
                'Validation instances differ or are missing')
    return dict(episodes=endpoint,epsilon=summary['epsilon'],decisions=summary['training_decisions'],
        sanity=pilot.sanity(summary['training_history'],summary['validation_history'],summary['td_diagnostics'])
            if endpoint>=36 else None,
        late_window=pilot.aggregate([r for v in late for r in v['runs']])
            if endpoint==60 and len(late)==3 else None)


def seed_average(metrics):
    if not metrics or not all(m.get('all_complete') for m in metrics): return dict(all_complete=False)
    return dict(all_complete=True,**{key:fmean(m[key] for m in metrics)
        for key in metrics[0] if key!='all_complete'})


def analyze(output):
    prepare(output)
    seeds={}
    for seed in (0,*SEEDS):
        arms={}
        for arm in ARMS:
            path=pilot.OUTPUT/arm/'seed-0/summary.json' if seed==0 else directory(output,seed,arm)/'summary.json'
            if not path.exists(): continue
            summary=read(path)
            require(summary['contract_sha256']==(contract()['pilot_contract_sha256'] if seed==0 else contract()['contract_sha256']),
                    'Summary source contract differs')
            if seed!=0: require(summary['model_seed']==seed and summary['variant']==arm,'Summary identity differs')
            arms[arm]=arm_summary(summary)
        ready=all(a in arms and arms[a]['late_window'] is not None for a in ARMS)
        item=dict(arms=arms,status='complete' if ready else 'partial',discovery_only=seed==0)
        if ready:
            control,residual=(arms[a]['late_window'] for a in ARMS)
            item.update(decision=pilot.decision(control,residual),
                J_improvement=control['J']-residual['J'] if control['all_complete'] and residual['all_complete'] else None)
        seeds[str(seed)]=item
    ready=all(seeds[str(s)]['status']=='complete' for s in SEEDS)
    report=dict(protocol=PROTOCOL,contract_sha256=contract()['contract_sha256'],status='complete' if ready else 'partial',
        seeds=seeds,confirmation_only=None,three_seed_descriptive=None,
        training_runs=0,paper_evidence=False,test_panel_opened=False)
    def comparison(selected):
        metrics={a:seed_average([seeds[str(s)]['arms'][a]['late_window'] for s in selected]) for a in ARMS}
        c,r=(metrics[a] for a in ARMS)
        return dict(training_seeds=list(selected),metrics=metrics,decision=pilot.decision(c,r),
            J_improvement=c['J']-r['J'] if c['all_complete'] and r['all_complete'] else None)
    if ready:
        report['confirmation_only']=comparison(SEEDS)
        if seeds['0']['status']=='complete': report['three_seed_descriptive']=comparison((0,*SEEDS))
    write(output/'e133-report.json',report)
    lines=['# E133 paired seed confirmation','',
        '| Seed | Arm | Episodes | Late-window J |','|---|---|---:|---:|']
    for seed,item in seeds.items():
        for arm,a in item['arms'].items():
            m=a['late_window']
            value=f'{m["J"]:.3f}' if m and m['all_complete'] else 'pending/failed'
            lines.append(f'| {seed} | {arm} | {a["episodes"]} | {value} |')
    lines.extend(['','Seed 0 is reused discovery evidence; primary confirmation uses only seeds 1 and 2.',
        'Same five development instances; no independent test panel opened.',''])
    (output/'e133-table.md').write_text('\n'.join(lines))
    print(f'E133 status={report["status"]}; report={output/"e133-report.json"}',flush=True)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('prepare','train','train-seeds','analyze'))
    p.add_argument('--output',type=Path,default=OUTPUT)
    p.add_argument('--model-seed',type=int,choices=SEEDS)
    p.add_argument('--arm',choices=ARMS)
    p.add_argument('--through-episode',type=int,default=60)
    p.add_argument('--device',choices=('cpu','cuda'),default='cuda')
    p.add_argument('--cpu-smoke',action='store_true')
    args=p.parse_args()
    torch.set_num_threads(1)
    output=args.output.resolve()
    prepare(output)
    if args.command=='train-seeds':
        # Exceptions propagate: never silently launch the next job after failure.
        for seed,arm in schedule(): train(output,seed,arm,args.through_episode,args.device,args.cpu_smoke)
    elif args.command=='train':
        require(args.model_seed in SEEDS and args.arm in ARMS,'Need --model-seed and --arm')
        train(output,args.model_seed,args.arm,args.through_episode,args.device,args.cpu_smoke)
    analyze(output)


if __name__=='__main__': main()
