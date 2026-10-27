"""Bounded matched nominal-cost residual training, with a predeclared gate."""
import argparse
from copy import deepcopy
from functools import lru_cache
import json
import math
import os
from pathlib import Path
from statistics import fmean
from unittest.mock import patch

import torch
from train_viability_graph_smdp import seed_everything
from experiments.conditioned_vcg.E123_absolute_progress_8x8_187k_v2 import run as source
from experiments.conditioned_vcg.E130_exact_macro_cost_diagnostic_194k_v1 import run as cost_audit
from .model import PilotAgent,ARMS

ROOT=source.p.ROOT
OUTPUT=ROOT/'results/vcg-conditioned-e131-nominal-cost-residual-pilot-195k-v1-repair2'
PROTOCOL='vcg_conditioned_e131_nominal_cost_residual_pilot_195k_v1'
BASE_ARM='A_outflow_bounded'
MAX_EPISODE=60
LATE=(48,54,60)
write,read,require=source.write,source.read,source.require
sha256,digest=cost_audit.ref.sha256,cost_audit.ref.digest


@lru_cache(maxsize=1)
def contract():
    parent=source.p.contract()
    paths=[*Path(__file__).parent.glob('*.py'),Path(__file__).with_name('run.sh'),ROOT/'tests/test_vcg_e131_residual.py']
    value=dict(protocol=PROTOCOL,parent_contract_sha256=parent['contract_sha256'],
        arms=list(ARMS),maximum_episode=60,decision_checkpoints=list(LATE),sanity_endpoint=36,
        minimum_paired_J_improvement=4.,model_seeds=[0],validation_seeds=list(source.p.VALIDATION_SEEDS),
        initialization='scratch, identical trunk/head weights and signed output links in both arms; never load E123 weights',
        control='signed-head total-cost Q; not bitwise old softplus-head E123',
        intervention='full four-component nominal K added only in residual arm; store observed C separately',
        TD='Huber(K+R,C+discount*next_total), algebraically identical to residual target C-K+discount*next_total',
        next_choice='one grouped online combined-cost choice for every target head; unchanged witness guard',
        uncertainty='K is nominal for every macro including Defer; signed residual corrects execution/event discrepancies',
        unchanged=['objective','frontier','guard','grouped selector','exploration by 8000 decisions','learning rate','batch size','target refresh','instances'],
        scaling='raw physical cost heads; merits use inherited -0.01 component weights',
        tails='residual late-window early AND late violation counts and magnitudes must not exceed control to advance',
        mechanism='report equipment and timing-cost changes separately; no mechanism attribution without supporting components',
        sanity_rule='all first 36 training episodes and all five greedy endpoint-36 episodes complete; recorded TD diagnostics finite and max absolute TD error <= 1e6',
        test_panel_opened=False,paper_evidence=False,
        source_sha256={str(p.relative_to(ROOT)):sha256(p) for p in paths})
    return {**value,'contract_sha256':digest(value)}


def prepare(output):
    expected=contract()
    path=output/'contract.json'
    if path.exists(): require(read(path)==expected,'E131 source changed; use fresh output version')
    else:
        require(not output.exists() or not any(output.iterdir()),'Need empty output')
        write(path,expected)


def fresh(variant,device='cpu'):
    seed_everything(0)
    return PilotAgent(config=source.original.replay.model.EpisodicViabilityGraphConfig(gamma=1.,episode_horizon_steps=3600),
        seed=0,device=device,epsilon=.9,arm=BASE_ARM,absolute_progress=False,variant=variant)


def episode(agent,seed,training,updates=1):
    with source.implementation():
        row=source.original.episode(agent,BASE_ARM,seed,training,updates=updates)
    row['e131_variant']=agent.variant
    return row


def physical_preflight():
    """Newly certify a genuinely overdue state; never edit a certified state."""
    from PSLAP.viability_filter import ViabilitySearchConfig
    from PSLAP.relocation_family_certification import RELOCATION_FAMILY_CERTIFICATION
    audit=source.original.replay.audit
    env=source.original.new_env(BASE_ARM)
    env.reset(instance=env.sample_episode_instance(source.p.PILOT_SEED))
    for _ in range(120): env.step(env.ACTION_IDS['WAIT'])
    manifest=read(source.p.parent.SOURCE/'frozen-source.json')
    with source.controller(BASE_ARM) as rule:
        snapshot=audit.enumerate_viability_candidates(env,consecutive_defer_decisions=0,
            search_config=ViabilitySearchConfig(**manifest['search_config']),liveness_rule=rule,
            cache=audit.ViabilityCertificateCache(),recovery_certification_strategy=RELOCATION_FAMILY_CERTIFICATION)
        rows=[]
        for stage in range(2):
            candidates=sorted(snapshot.candidates,key=lambda c:c.key)
            for kind in ('accept','defer','deliver','reconfigure'):
                for c in [c for c in candidates if c.action_type.value==kind][:2]:
                    prediction=cost_audit.predict(env,c,1.)
                    execution,totals=cost_audit.shadow_execution(env,c,1.)
                    rows.append(dict(kind=kind,stage=stage,start_time=env.time_steps,nominal=prediction,
                        observed=totals,duration=execution.duration,success=execution.option_success,
                        cost_error=max(abs(a-b) for a,b in zip(prediction['components'],totals))))
            if stage==0:
                c=next(c for c in candidates if c.action_type.value=='accept')
                execution=audit._execute_audit_macro(env,c,gamma=1.,remaining_steps=3600-env.time_steps,evaluation=True)
                require(execution.option_success,'Overdue preflight could not establish storage')
                # Start the next newly certified frontier one step before an
                # arrival, covering route execution across an exogenous event.
                while env.time_steps%30!=29: env.step(env.ACTION_IDS['WAIT'])
                snapshot=audit.enumerate_viability_candidates(env,consecutive_defer_decisions=0,
                    search_config=ViabilitySearchConfig(**manifest['search_config']),liveness_rule=rule,
                    cache=audit.ViabilityCertificateCache(),recovery_certification_strategy=RELOCATION_FAMILY_CERTIFICATION)
    require(rows and any(r['observed'][0]>0 for r in rows),'No positive tardiness physical coverage')
    require(any(r['kind']=='deliver' and r['observed'][0]>0 for r in rows),'No overdue delivery coverage')
    # Nominal mismatches are permitted, but only signed observed-minus-K
    # corrections are valid. Successful route-bound physical checks are required.
    require(all(r['success'] and r['cost_error']<1e-8 for r in rows if r['kind']!='defer'),
            'Route-bound overdue physical costs disagree')
    return rows


def smoke(output):
    prepare(output)
    require(read(output/'preflight.json')['contract_sha256']==contract()['contract_sha256'],'Run preflight')
    results=[]
    for variant in ARMS:
        agent=fresh(variant)
        before=source.original.replay.parameter_digest(agent)
        with source.cache_session(): row=episode(agent,source.p.train_seed(0,1),True)
        require(row['strict_method_success'],'Smoke episode failed completion')
        # Short episodes may not fill the inherited batch of 128. Exercise one
        # real backward/Adam update with repeated disposable transitions only;
        # these weights/data are never saved as an experimental checkpoint.
        repeated=0
        if agent.gradient_steps==0:
            originals=tuple(agent.replay.memory)
            require(originals,'No smoke replay')
            while len(agent.replay)<agent.config.batch_size:
                agent.replay.add(originals[repeated%len(originals)])
                repeated+=1
            agent.learn()
        require(agent.gradient_steps>0 and before!=source.original.replay.parameter_digest(agent),'No real learning exercised')
        require(all(math.isfinite(v) for loss in agent.component_loss_history for v in loss),'Nonfinite smoke losses')
        restored=PilotAgent.from_checkpoint(agent.checkpoint(include_replay=True),resumable=True)
        require(restored.gradient_steps==agent.gradient_steps and len(restored.replay)==len(agent.replay),
                'Resumable smoke checkpoint roundtrip failed')
        results.append(dict(variant=variant,strict_complete=True,steps=row['steps'],
            gradient_steps=agent.gradient_steps,repeated_disposable_transitions=repeated,
            maximum_abs_TD_error=max(d['max_abs_td_error'] for d in agent.td_diagnostics)))
        print(f'E131 smoke {variant}: strict complete, optimizer steps={agent.gradient_steps}',flush=True)
    write(output/'smoke.json',dict(contract_sha256=contract()['contract_sha256'],passed=True,
        arms=results,scope='disposable one-episode execution/backward/resume checks; no quality comparison',
        experimental_checkpoints_written=False))


def preflight(output):
    prepare(output)
    a,b=fresh('total_q'),fresh('residual_q')
    require(source.original.replay.parameter_digest(a)==source.original.replay.parameter_digest(b),
            'Scratch initialization is not paired')
    rows=physical_preflight()
    write(output/'preflight.json',dict(contract_sha256=contract()['contract_sha256'],passed=True,
          scratch_weights_identical=True,overdue_physical_checks=rows,training_runs=0))
    print('E131 preflight passed: paired scratch weights and positive-tardiness physical costs',flush=True)


def save(path,agent,history,validations):
    payload=agent.checkpoint(include_replay=True,e131_contract_sha256=contract()['contract_sha256'],
        training_history=history,validation_history=validations,completed_training_episodes=len(history),
        global_rng_state=source.original.replay.audit._capture_global_rng_state(),
        model_seed=0,deployment_checkpoint_eligible=False,paper_evidence=False,test_panel_opened=False)
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix('.pth.tmp')
    torch.save(payload,temporary)
    os.replace(temporary,path)


def sanity(history,validations,diagnostics):
    endpoints=[v for v in validations if v['checkpoint_episode']==36]
    return bool(len(history)>=36 and all(r['strict_method_success'] for r in history[:36])
        and len(endpoints)==1 and len(endpoints[0]['runs'])==5
        and all(r['strict_method_success'] for r in endpoints[0]['runs']) and diagnostics
        and all(math.isfinite(float(v)) for d in diagnostics for v in d.values())
        and max(d['max_abs_td_error'] for d in diagnostics)<=1e6)


def validate(agent,number):
    state=source.original.replay.audit._capture_global_rng_state()
    before=source.original.replay.parameter_digest(agent)
    counts=(agent.decision_count,agent.gradient_steps,len(agent.replay))
    evaluator=PilotAgent.from_checkpoint(agent.checkpoint(include_replay=False),device=agent.device)
    evaluator.Q_local.eval().requires_grad_(False)
    evaluator.Q_target.eval().requires_grad_(False)
    try:
        with torch.no_grad(): runs=[episode(evaluator,s,False) for s in source.p.VALIDATION_SEEDS]
    finally: source.original.replay.audit._restore_global_rng_state(state)
    require(before==source.original.replay.parameter_digest(agent)
        and counts==(agent.decision_count,agent.gradient_steps,len(agent.replay)),'Validation changed training')
    return dict(checkpoint_episode=number,runs=runs)


def train(output,variant,through,device,cpu_smoke):
    prepare(output)
    gate=read(output/'preflight.json')
    require(gate['passed'] and gate['contract_sha256']==contract()['contract_sha256'],'Run preflight first')
    learning_gate=read(output/'smoke.json')
    require(learning_gate['passed'] and learning_gate['contract_sha256']==contract()['contract_sha256'],
            'Run disposable smoke learning checks first')
    require(device=='cuda' or cpu_smoke and through<=1,'CPU training capped at one smoke episode')
    require(device!='cuda' or torch.cuda.is_available(),'CUDA unavailable')
    path=output/variant/'seed-0/latest.pth'
    if path.exists():
        payload=torch.load(path,map_location='cpu',weights_only=False)
        require(payload['e131_contract_sha256']==contract()['contract_sha256'] and payload['e131_variant']==variant,
                'Resume contract or arm differs')
        agent=PilotAgent.from_checkpoint(payload,device=device,resumable=True)
        source.original.replay.audit._optimizer_to(agent.optimizer,torch.device(device))
        history,validations=payload['training_history'],payload['validation_history']
        source.original.replay.audit._restore_global_rng_state(payload['global_rng_state'])
    else:
        agent=fresh(variant,device)
        history,validations=[],[]
        save(path,agent,history,validations)
    require(all(r['strict_method_success'] for r in history),'Failed history: investigate before resume')
    require(len(history)<=through<=MAX_EPISODE,'Invalid endpoint')
    from .model import CostRecord
    with source.cache_session():
        for number in range(len(history)+1,through+1):
            if number>36:
                require(sanity(history,validations,agent.td_diagnostics),
                        'Endpoint-36 sanity gate failed: stop and diagnose, do not tune or continue')
            print(f'E131 {variant} episode={number}/60 decisions={agent.decision_count} epsilon={agent.epsilon:.3f}',flush=True)
            row=episode(agent,source.p.train_seed(0,number),True)
            history.append(row)
            require(all(isinstance(t.chosen,CostRecord) and all(isinstance(r,CostRecord) for r in t.next_candidates)
                        for t in agent.replay.memory),'Nominal cost missing from replay')
            if number%6==0 and row['strict_method_success']: validations.append(validate(agent,number))
            save(path,agent,history,validations)
            write(path.parent/'summary.json',dict(contract_sha256=contract()['contract_sha256'],variant=variant,
                completed_training_episodes=len(history),training_history=history,validation_history=validations,
                training_decisions=agent.decision_count,epsilon=agent.epsilon,gradient_steps=agent.gradient_steps,
                td_diagnostics=agent.td_diagnostics,losses=agent.component_loss_history,
                paper_evidence=False,test_panel_opened=False))
            require(row['strict_method_success'],'Completion failure: saved endpoint; stop and diagnose')
    print(f'E131 {variant}: paused at {len(history)}; checkpoint={path}',flush=True)


def aggregate(runs):
    if not runs or not all(r['strict_method_success'] for r in runs): return dict(all_complete=False)
    # Timing cost components, not storage-clock errors.
    values=dict(all_complete=True,J=fmean(r['operator_cost'] for r in runs))
    for key in ('equipment_work','physical_rehandles','mean_release_earliness_beyond_tolerance',
                'mean_release_tardiness_beyond_tolerance','mean_wait_before_storage','release_clock_mae','within_tolerance_rate'):
        values[key]=fmean(r['service_metrics'][key] for r in runs)
    for tail in ('early','late'):
        values[tail+'_violations']=fmean(sum(e< -20 if tail=='early' else e>20
                              for e in r['delivery_deviations']) for r in runs)
    values['equipment_cost']=.1*values['equipment_work']
    values['timing_cost']=36*(2*values['mean_release_tardiness_beyond_tolerance']+
                                values['mean_release_earliness_beyond_tolerance'])
    return values


def decision(control,residual):
    if not control.get('all_complete') or not residual.get('all_complete'):
        return 'do_not_advance_completion_failure'
    improvement=control['J']-residual['J']
    tails=all(residual[k]<=control[k] for k in ('early_violations','late_violations',
        'mean_release_earliness_beyond_tolerance','mean_release_tardiness_beyond_tolerance'))
    return 'advance_to_confirmation' if improvement>=4. and tails else 'inconclusive_do_not_tune'


def analyze(output):
    prepare(output)
    arms={}
    for variant in ARMS:
        path=output/variant/'seed-0/summary.json'
        if not path.exists(): continue
        summary=read(path)
        require(summary['contract_sha256']==contract()['contract_sha256'],'Summary contract changed')
        late=[v for v in summary['validation_history'] if v['checkpoint_episode'] in LATE]
        complete=len(late)==3 and summary['completed_training_episodes']==60
        arms[variant]=dict(episodes=summary['completed_training_episodes'],decisions=summary['training_decisions'],
            epsilon=summary['epsilon'],late_window=aggregate([r for v in late for r in v['runs']]) if complete else None,
            episode_36_sanity_passed=sanity(summary['training_history'],summary['validation_history'],summary['td_diagnostics'])
                if summary['completed_training_episodes']>=36 else None,
            validation=[dict(episode=v['checkpoint_episode'],**aggregate(v['runs'])) for v in summary['validation_history']])
    ready=all(a in arms and arms[a]['late_window'] is not None for a in ARMS)
    report=dict(protocol=PROTOCOL,contract_sha256=contract()['contract_sha256'],arms=arms,
        status='complete' if ready else 'partial',decision=decision(*(arms[a]['late_window'] for a in ARMS)) if ready else None,
        training_runs=0,paper_evidence=False,test_panel_opened=False)
    if ready:
        c,r=(arms[a]['late_window'] for a in ARMS)
        report['J_improvement']=c.get('J',0)-r.get('J',0) if c['all_complete'] and r['all_complete'] else None
        report['mechanism_components']=dict(control=c,residual=r)
    write(output/'e131-report.json',report)
    print(json.dumps(report,indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('preflight','smoke','train','analyze'))
    parser.add_argument('--output',type=Path,default=OUTPUT)
    parser.add_argument('--arm',choices=ARMS)
    parser.add_argument('--through-episode',type=int,default=60)
    parser.add_argument('--device',choices=('cpu','cuda'),default='cuda')
    parser.add_argument('--cpu-smoke',action='store_true')
    args=parser.parse_args()
    torch.set_num_threads(1)
    output=args.output.resolve()
    if args.command=='preflight': preflight(output)
    elif args.command=='smoke': smoke(output)
    elif args.command=='train':
        require(args.arm in ARMS,'Need --arm')
        train(output,args.arm,args.through_episode,args.device,args.cpu_smoke)
    analyze(output)


if __name__=='__main__': main()
