"""Predict macro costs, execute physical clones, and gate future learning work."""
import argparse
from collections import Counter
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
import random
import signal
from time import monotonic
from unittest.mock import patch

import numpy as np
import torch
import run_vcg_objective_gamma_audit as audit
from experiments.conditioned_vcg.E127_same_state_ranking_diagnostic_191k_v1 import run as ref
from .costs import predict,compare

ROOT=ref.ROOT
OUTPUT=ROOT/'results/vcg-conditioned-e130-exact-macro-cost-diagnostic-194k-v1'
PROTOCOL='vcg_conditioned_e130_exact_macro_cost_diagnostic_194k_v1'
BOUNDARIES=32
WALL_SECONDS=180
read,write,require=ref.read,ref.write,ref.require


class BoundedStop(BaseException): pass


@lru_cache(maxsize=1)
def contract():
    parent=ref.contract()
    paths=[Path(__file__),Path(__file__).with_name('costs.py'),Path(__file__).with_name('run.sh'),
           ROOT/'tests/test_vcg_e130_exact_macro_cost.py']
    payload=dict(protocol=PROTOCOL,parent_contract_sha256=parent['contract_sha256'],
        checkpoint_sha256=ref.sha256(ref.CHECKPOINT),checkpoint_episode=120,
        instance_seed=ref.SEED,maximum_boundaries=BOUNDARIES,wall_seconds=WALL_SECONDS,
        sampling='up to three lexically spread guard-admissible candidates per action type at each boundary',
        cost_inputs='released undelivered jobs, current time, bound recovery paths or current Accept paths; Defer horizon',
        scope='development-only cost fidelity diagnostic; no new policy, optimizer or training',
        source_sha256={str(p.relative_to(ROOT)):ref.sha256(p) for p in paths},
        training_runs=0,independent_test_opened=False)
    return {**payload,'contract_sha256':ref.digest(payload)}


def prepare(output):
    expected=contract()
    path=output/'contract.json'
    if path.exists(): require(read(path)==expected,'E130 source changed: use a fresh repair output')
    else:
        require(not output.exists() or not any(output.iterdir()),'Need empty output')
        write(path,expected)


def candidates(decision,snapshot):
    groups={}
    for index in decision.prepared_snapshot.source_indices:
        item=snapshot.candidates[index]
        groups.setdefault(item.action_type.value,[]).append(item)
    result=[]
    for kind,items in sorted(groups.items()):
        items=sorted(items,key=lambda c:c.key)
        result.extend(items[i] for i in sorted({0,len(items)//2,len(items)-1}))
    return result


def shadow_execution(env,candidate,gamma):
    # Copy together to preserve option -> environment -> bound block identity.
    shadow,bound=deepcopy((env,candidate))
    cursor=len(shadow.factorized_component_records)
    py_state=random.getstate()
    np_state=np.random.get_state()
    torch_state=torch.random.get_rng_state()
    try:
        execution=audit._execute_audit_macro(shadow,bound,gamma=gamma,
                    remaining_steps=3600-shadow.time_steps,evaluation=True)
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
        torch.random.set_rng_state(torch_state)
    values=shadow.factorized_component_records[cursor:]
    totals=tuple(sum(gamma**k*v[j] for k,v in enumerate(values)) for j in range(4))
    return execution,totals


def run(output):
    prepare(output)
    require(not (output/'e130-result.json').exists(),'Result already exists; analyze or use fresh output')
    agent=ref.SharedProgressAgent.from_checkpoint(torch.load(ref.CHECKPOINT,map_location='cpu',weights_only=False),
                                               device='cpu',resumable=False,seed=0)
    agent.Q_local.eval().requires_grad_(False)
    agent.Q_target.eval().requires_grad_(False)
    digest_before=ref.source.original.replay.parameter_digest(agent)
    updates,replay_size=agent.gradient_steps,len(agent.replay)
    native=agent.select
    rows=[]
    boundaries=0
    started=monotonic()
    stopped='episode_finished'
    def select(snapshot,*,training=False,epsilon=0.):
        nonlocal boundaries
        if boundaries>=BOUNDARIES: raise BoundedStop('boundary_limit')
        decision=native(snapshot,training=False,epsilon=0.)
        env=agent.context_environment
        before=ref.provenance._environment_digest(env)
        for candidate in candidates(decision,snapshot):
            prediction=predict(env,candidate,agent.config.gamma)
            execution,totals=shadow_execution(env,candidate,agent.config.gamma)
            rows.append(dict(boundary=boundaries,step=env.time_steps,key=candidate.key,
                kind=candidate.action_type.value,prediction=prediction,
                actual_duration=execution.duration,actual_components=totals,
                option_success=execution.option_success,failure=execution.failure_reason,
                **compare(prediction,execution.duration,totals,execution.option_success)))
        require(ref.provenance._environment_digest(env)==before,'Shadow execution changed native environment')
        boundaries+=1
        write(output/'progress.json',dict(boundaries=boundaries,rows=rows))
        if boundaries%4==0:
            print(f'E130 cost audit | boundaries={boundaries}/{BOUNDARIES} macros={len(rows)} '
                  f'mismatches={sum(not r["matches"] for r in rows)} step={env.time_steps}',flush=True)
        return decision
    def alarm(*args): raise BoundedStop('wall_clock_limit')
    previous=signal.signal(signal.SIGALRM,alarm)
    signal.alarm(WALL_SECONDS)
    try:
        with ref.cache_session(),ref.source.implementation(),patch.object(agent,'select',select),torch.no_grad():
            ref.source.original.episode(agent,ref.ARM,ref.SEED,False)
    except BoundedStop as exc: stopped=str(exc)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM,previous)
    require(ref.source.original.replay.parameter_digest(agent)==digest_before
            and agent.gradient_steps==updates and len(agent.replay)==replay_size,'Diagnostic learned or changed weights')
    require(ref.sha256(ref.CHECKPOINT)==contract()['checkpoint_sha256'],'Source checkpoint changed')
    result=dict(protocol=PROTOCOL,contract_sha256=contract()['contract_sha256'],
        status=stopped,boundaries=boundaries,rows=rows,wall_seconds=monotonic()-started,
        checkpoint_unchanged=True,training_runs=0,optimizer_steps=0,independent_test_opened=False)
    write(output/'e130-result.json',result)


def analyze(output):
    prepare(output)
    result=read(output/'e130-result.json')
    require(result['contract_sha256']==contract()['contract_sha256'],'Wrong result contract')
    rows=result['rows']
    coverage=Counter(r['kind'] for r in rows)
    mismatches=[r for r in rows if not r['matches']]
    sufficient=set(coverage)=={'accept','deliver','reconfigure','defer'} and result['boundaries']==BOUNDARIES
    report={k:v for k,v in result.items() if k!='rows'}
    report.update(samples=len(rows),coverage=dict(coverage),mismatches=len(mismatches),
        mismatch_examples=mismatches[:8],route_bound_mismatches=sum(r['kind']!='defer' for r in mismatches),
        gate='bounded_cost_fidelity_pass_not_global_proof' if sufficient and not mismatches else
             'do_not_launch_full_exact_macro_training',
        next_step='Verify stochastic/event-sensitive costs or restrict the exact term before changing the critic' if mismatches else
                  'Prepare a separately versioned matched continuation-only training pilot; retain grouping and guard')
    write(output/'e130-report.json',report)
    print(f'E130 | samples={len(rows)} coverage={dict(coverage)} mismatches={len(mismatches)} gate={report["gate"]}')
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('run','analyze'))
    parser.add_argument('--output',type=Path,default=OUTPUT)
    args=parser.parse_args()
    torch.set_num_threads(1)
    output=args.output.resolve()
    if args.command=='run': run(output)
    analyze(output)


if __name__=='__main__': main()
