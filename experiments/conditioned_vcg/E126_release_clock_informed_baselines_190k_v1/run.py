"""Clock-informed baseline comparison; same rules, objective scoring and instances."""
import argparse
import builtins
from contextlib import contextmanager
from functools import lru_cache
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from experiments.conditioned_vcg.E125_e123_matched_baseline_reference_189k_v1 import run as ref
from . import clock

ROOT = ref.ROOT
OUTPUT = ROOT/'results/vcg-conditioned-e126-release-clock-informed-baselines-190k-v1-repair2'
PROTOCOL = 'vcg_conditioned_e126_release_clock_informed_baselines_190k_v1_repair2'
baseline = ref.baseline
write,read,require = ref.write,ref.read,ref.require


@lru_cache(maxsize=1)
def contract():
    parent = ref.contract()
    paths = [Path(__file__),Path(clock.__file__),Path(__file__).with_name('run.sh'),
        ROOT/'tests/test_vcg_e126_release_clock_inputs.py',
        ROOT/'example/urgency_scheduler.py',ROOT/'example/Options/selector_v5.py',
        ROOT/'example/Options/AcceptStoreOption.py',ROOT/'example/Options/StrategicDeferOption.py',
        ROOT/'PSLAP/dynamic_yard.py',ROOT/'PSLAP/ga_policy.py',ROOT/'PSLAP/ga_capacity_aware.py',
        ROOT/'PSLAP/ga_optimizer.py',
        ROOT/'PSLAP/retrieval_context.py',ROOT/'PSLAP/retrieval_dispatch.py']
    value = {**parent,'protocol':PROTOCOL,'reference_contract_sha256':parent['contract_sha256'],
        'clock_intervention':'arrived-job due=arrival+duration at stored, inbound, pending and projected planning inputs',
        'unchanged':['instances','physical timestamps/durations','placement heuristic','retrieval/Defer rules',
                     'GA fitness','GA populations/generations/replicate seeds','J and accounting','VCG results'],
        'reference_report_sha256':ref.sha256(ref.OUTPUT/'e125-report.json'),
        'source_sha256':{**parent['source_sha256'],**{str(p.relative_to(ROOT)):ref.sha256(p) for p in paths}}}
    value.pop('contract_sha256')
    return {**value,'contract_sha256':ref.digest(value)}


@contextmanager
def adapter():
    settings = SimpleNamespace(PROTOCOL=PROTOCOL,INSTANCE_SEEDS=ref.SEEDS,METHODS=ref.METHODS,
        GA_REPLICATES=ref.GA_REPLICATES,contract=contract,digest=ref.digest)
    def readable(message,*args,**kwargs):
        if isinstance(message,str): message=message.replace('E116 ','E126 ')
        builtins.print(message,*args,**kwargs)
    with patch.object(baseline,'p',settings),patch.object(baseline,'print',readable,create=True):
        yield


def prepare(output):
    with adapter():
        result=baseline.prepare(output)
    identities={r['seed']:r['instance_id'] for r in read(output/'instances.json')['records']}
    require(all(identities[r['instance_seed']]==r['episode_instance_id']
                for v in ref.source_data()[1] for r in v['runs']),'Instance identity differs from VCG')
    return result


def run_one(output,method,seed,replicate):
    prepare(output)
    with adapter(),clock.enabled() as env_type:
        row=baseline.run_one(output,method,seed,replicate)
        if not row.get('reused',False):
            require(len(env_type.created)==1,'Expected one traced environment')
            env=env_type.created[0]
            schedule=[(b.label,b.arrival_step,b.storage_steps_needed) for b in env.blocks]
            require(schedule==env.e126_initial_schedule,'Planning clock mutated physical job schedule')
            row['planning_clock_audit']=dict(due_date='arrival_step+storage_steps_needed',
                physical_schedule_unchanged=True,calls=dict(env.e126_clock_calls),samples=env.e126_clock_samples,
                stored_timestamps_unchanged_by_adapter=True)
            require(env.e126_clock_calls['stored_snapshot']>0 and env.e126_clock_calls['assignment_view']>0,
                    'Release clock not observed by stored and inbound planners')
            if method=='ga': require(env.e126_clock_calls['ga_pending_view']>0,'GA pending clock bypassed')
            write(baseline.row_path(output,method,seed,replicate),row)
        else:
            require(row.get('planning_clock_audit',{}).get('physical_schedule_unchanged'),
                    'Saved coordinate lacks release-clock audit')
        return row


def analyze(output):
    prepare(output)
    with adapter():
        rows=[]
        for m in ref.METHODS:
            for s in ref.SEEDS:
                for rep in baseline.replicates(m):
                    path=baseline.row_path(output,m,s,rep)
                    if path.exists():
                        r=read(path)
                        require(r['contract_sha256']==contract()['contract_sha256']
                            and (r['method'],r['instance_seed'],r['replicate'])==(m,s,rep)
                            and r.get('planning_clock_audit',{}).get('physical_schedule_unchanged'),
                            'Saved row identity or clock audit differs')
                        rows.append(r)
        points,summaries=baseline._baseline_points(rows)
        validations=[baseline._e115_checkpoint_points(v) for v in ref.source_data()[1]]
        comparisons=[ref.comparison('late_window',ref.average_checkpoints(validations,ref.SEEDS),points),
                     ref.comparison(120,validations[-1],points)]
        report=dict(protocol=PROTOCOL,contract_sha256=contract()['contract_sha256'],
            status='complete' if len(rows)==25 else 'partial',expected_rows=25,observed_rows=len(rows),
            strict_complete_rows=sum(r['metrics']['strict_complete'] for r in rows),
            baseline_summaries=summaries,baseline_instance_points=points,comparisons=comparisons,
            original_storage_clock_baselines=read(ref.OUTPUT/'e125-report.json')['baseline_summaries'],
            training_runs=0,episode_runs=0,independent_test_opened=False,paper_evidence=False,
            scope='release-clock-informed existing heuristics; unchanged GA fitness; reused five-instance development panel')
        write(output/'e126-report.json',report)
        lines=['E126: release-clock-informed Dynamic PSLAP and GA versus frozen E123 A','',
            '| Method | Complete | J | Release MAE | Within ±20 | Outside wait | Storage MAE | Rehandles/100 | Steps/delivery |',
            '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
        entries=[(c['vcg']['method'],c['vcg']['strict_complete_instances'],c['vcg']['metrics']) for c in comparisons]
        entries += [(s['method']+'_release_clock',s['fully_complete_instances'],
                     s['metrics'] if s['fully_complete_instances']==5 else None) for s in summaries]
        for name,n,metrics in entries:
            values=([metrics[k] for k in ('objective_cost','release_clock_mae','within_tolerance_rate',
                'mean_wait_before_storage','storage_clock_mae','rehandles_per_100','steps_per_delivery')]
                if metrics else None)
            lines.append(f'| {name} | {n}/5 | '+(' | '.join(f'{v:.3f}' for v in values) if values else '— | — | — | — | — | — | —')+' |')
        lines += ['', 'Within ±20 is a fraction. Same J=2T+E+0.1L+10R, instances and physical accounting.',
            'Only the planning clock changes; actual storage timestamps/durations, algorithms, GA fitness and budgets remain unchanged.',
            'Average repeated checkpoints/GA replicates within instance; paired n=5. Nominal intervals are exploratory.',
            'No independent test, retraining or objective-aligned GA fitness. Historical VCG latency is not a controlled timing benchmark.']
        (output/'e126-table.md').write_text('\n'.join(lines)+'\n')
    return dict(status=report['status'],observed_rows=len(rows),expected_rows=25,
        strict_complete_rows=report['strict_complete_rows'],report=str(output/'e126-report.json'),
        table=str(output/'e126-table.md'),episode_runs=0,training_runs=0)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('prepare','run-one','run-all','analyze','status'))
    parser.add_argument('--output',type=Path,default=OUTPUT)
    parser.add_argument('--method',choices=ref.METHODS)
    parser.add_argument('--seed',type=int,choices=ref.SEEDS)
    parser.add_argument('--replicate',type=int,default=0)
    args=parser.parse_args()
    torch.set_num_threads(1)
    output=args.output.resolve()
    if args.command=='prepare': result=prepare(output)
    elif args.command in ('status','analyze'): result=analyze(output)
    else:
        if args.command=='run-one':
            require(args.method is not None and args.seed is not None,'Need --method and --seed')
            rows=[run_one(output,args.method,args.seed,args.replicate)]
        else:
            rows=[run_one(output,m,s,rep) for m in ref.METHODS for s in ref.SEEDS
                  for rep in (ref.GA_REPLICATES if m=='ga' else (0,))]
        result=analyze(output)
        result['episode_runs']=sum(not r.get('reused',False) for r in rows)
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
