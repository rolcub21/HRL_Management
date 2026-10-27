import argparse
import builtins
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from example.episode_instance import EpisodeInstance
from vcg_objective_audit import ObjectiveAuditSmallRoomsEnv
from experiments.conditioned_vcg.E78_original_vcg_base80_full_8x8_143k_v1.protocol import SPEC
from experiments.conditioned_vcg.E145_occupancy_only_pilot import run as prior

ROOT=prior.ROOT
OUTPUT=ROOT/'results/vcg-conditioned-e146-empty-start-working-day'
DAY=5400
MEAN_DURATION=1200
WORK_ALLOWANCE=30
CAP=16200
TARGETS=(.2,.5,.8)
SEEDS=(24060000,24060001,24060002)
LAST_ENV=None


def count(target,capacity=35):
    return int(target*capacity*DAY/MEAN_DURATION+.5)


def case(seed,target):
    if target not in TARGETS:
        raise ValueError('Undeclared day volume')
    env=prior.original.discovery.source.original.new_env('A_outflow_bounded')
    native=env.sample_episode_instance(seed)
    n=count(target,len(native.storage_positions))
    maximum=count(max(TARGETS),len(native.storage_positions))
    # Common random numbers across levels, independent arrival/duration streams.
    times=np.random.default_rng(seed+1461).integers(1,DAY,maximum-1).tolist()
    times=[0,*times]
    durations=np.maximum(1,np.random.default_rng(seed+1462).poisson(MEAN_DURATION,maximum)).tolist()
    jobs=sorted(zip(times[:n],durations[:n]),key=lambda x:x[0])
    instance=replace(native,arrival_rate=n/DAY,proc_mean=float(MEAN_DURATION),
        arrival_steps=tuple(a for a,p in jobs),storage_steps_needed=tuple(p for a,p in jobs))
    return dict(instance=instance.to_dict(),initial_positions=[],
        external_due_dates={f'B{i+1}':a+p for i,(a,p) in enumerate(jobs)},
        seed=seed,day_steps=DAY,mean_duration=MEAN_DURATION,nominal_overlap_target=target,
        incoming_count=n,capacity=len(native.storage_positions),
        nominal_overlap=n*MEAN_DURATION/(DAY*len(native.storage_positions)),
        planned_utilization=n*WORK_ALLOWANCE/DAY,
        arrivals='one job at0 required by simulator; N-1 iid discrete Uniform[1,5399], sorted; no arrivals after day-end')


class DayEnv(prior.evaluator.rollout.RecordedGridEnv):
    def __init__(self,value):
        global LAST_ENV
        instance=EpisodeInstance.from_dict(value['instance'])
        self._episode_instance=instance
        self._initial_positions=()
        self._external_dates=dict(value['external_due_dates'])
        self.initial_labels=set()
        ObjectiveAuditSmallRoomsEnv.__init__(self,timing_objective=SPEC,
            grid_rows=8,grid_cols=8,start_state=(1,1),door_cell=(0,6),pickup_cells=(1,6),
            exit_cells=[(7,4),(7,5),(7,6)],number_blocks=instance.number_blocks,
            choose_storage=False,arrival_rate=instance.arrival_rate,proc_mean=instance.proc_mean)
        self.reset(instance)
        self.day_work=self.total_work=self.day_stored_steps=0
        self.peak_stored=self.day_peak_stored=self.peak_outside=self.day_peak_outside=0
        LAST_ENV=self

    def levels(self):
        stored=sum(b.stored and not b.carrying and not b.delivered and
                   b.position in self.storage_positions for b in self.blocks)
        outside=sum(b.arrival_step<=self.time_steps and b.stored_time_step is None and not b.delivered
                    for b in self.blocks)
        return stored,outside

    def step(self,action):
        before=self.time_steps
        stored,outside=self.levels()
        result=super().step(action)
        busy=int(int(action)!=self.ACTION_IDS['WAIT'])
        self.total_work+=busy
        if before<DAY:
            self.day_work+=busy
            self.day_stored_steps+=stored
        after_stored,after_outside=self.levels()
        self.peak_stored=max(self.peak_stored,stored,after_stored)
        self.peak_outside=max(self.peak_outside,outside,after_outside)
        if self.time_steps<=DAY:
            self.day_peak_stored=max(self.day_peak_stored,stored,after_stored)
            self.day_peak_outside=max(self.day_peak_outside,outside,after_outside)
        return result

    def day_metrics(self):
        return dict(day_steps=DAY,observed_steps=self.time_steps,
            day_fully_observed=self.time_steps>=DAY or all(b.delivered for b in self.blocks),
            stored_peak=self.peak_stored,stored_peak_fraction=self.peak_stored/len(self.storage_positions),
            stored_day_peak=self.day_peak_stored,
            stored_day_mean_fraction=self.day_stored_steps/(DAY*len(self.storage_positions)),
            outside_queue_peak=self.peak_outside,outside_day_queue_peak=self.day_peak_outside,
            transporter_day_work_steps=self.day_work,transporter_day_utilization=self.day_work/DAY,
            transporter_total_utilization=self.total_work/max(1,self.time_steps),
            drain_steps=max(0,self.time_steps-DAY),
            unfinished_jobs=sum(not b.delivered for b in self.blocks),
            right_censored=not all(b.delivered for b in self.blocks))


def prepare(output):
    cases=[]
    for level,target in enumerate(TARGETS):
        for replicate,seed in enumerate(SEEDS):
            value=json.loads(json.dumps(case(seed,target)))
            path=output/'instances'/f'target-{target:g}-seed-{seed}.json'
            if path.exists() and prior.original.discovery.read(path)!=value:
                raise ValueError('Prepared day changed')
            if not path.exists():
                prior.original.discovery.write(path,value)
            cases.append(dict(index=level*3+replicate,target=target,replicate=replicate,seed=seed,
                incoming_count=value['incoming_count'],nominal_overlap=value['nominal_overlap'],
                planned_utilization=value['planned_utilization'],path=str(path.resolve()),sha256=prior.file_sha(path)))
    models=[]
    for seed in range(3):
        info=prior.original.discovery.read(prior.original.OUTPUT/f'models/seed-{seed}.json')
        path=prior.original.OUTPUT/info['file']
        if prior.file_sha(path)!=info['snapshot_sha256']:
            raise ValueError('Frozen source changed')
        models.append({**info,'file':str(path.resolve())})
    paths=[Path(__file__),Path(__file__).with_name('README.txt'),ROOT/'tests/test_vcg_e146_working_day.py',
        Path(prior.__file__),Path(prior.evaluator.__file__),
        ROOT/'experiments/conditioned_vcg/E137_grid_rollout_readiness_201k_v1/rollout.py']
    value=dict(protocol='e146_fixed_day_empty_start_v1',cases=cases,models=models,
        day_steps=DAY,duration='positive Poisson1200, same for all levels',work_allowance=WORK_ALLOWANCE,
        initial_inventory=0,geometry='original8x8,35storage cells',objective='unchanged release-clock J',
        arrival_information='past/current releases only; future schedule hidden to all methods',
        budgets=dict(execution_cap=CAP,macros=4000,certificate_nodes=2000,wall_seconds=1800),
        termination='all jobs delivered, not day-end; cap/wall/macro exhaustion is failure/truncation',
        normalization='frozen source3600 input units retained; clipping after3600 is disclosed transfer limitation',
        certified_shell='same original base model class, E137 certificate/guard/flow-event Defer counter',
        transfer='longer duration/random-arrival distribution differs from training in every level',
        limitations='one anchored arrival at0; finite empty-start nominal overlap approximation; '
                    'planning utilization is not a feasibility guarantee; plain/certified macro mismatch',
        sources={str(p.relative_to(ROOT)):prior.file_sha(p) for p in paths},
        training_runs=0,test_panel_opened=False)
    value['contract_sha256']=hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()
    value=json.loads(json.dumps(value))
    path=output/'contract.json'
    if path.exists() and prior.original.discovery.read(path)!=value:
        raise ValueError('Contract differs; use new output')
    if not path.exists():
        prior.original.discovery.write(path,value)
    return value


def analyze(output,contract):
    rows=[prior.original.discovery.read(p) for p in (output/'rows').glob('*/*.json')]
    lines=['# Fixed-day, empty-start workload pilot','','Targets are nominal overlap, NOT measured occupancy.','',
        '| Method | Target | Jobs/day | Instance | Status | J | Peak stored | Day work utilization | Peak outside |',
        '|---|---:|---:|---:|---|---:|---:|---:|---:|']
    for row in sorted(rows,key=lambda r:(r['regime'],r['label'])):
        case=contract['cases'][row['regime']];m=row.get('metrics') or {};d=row.get('day_statistics') or {}
        cost=f"{m['J']:.1f}" if m.get('cost_ranking_eligible') else '—'
        peak=f"{100*d['stored_peak_fraction']:.1f}%" if d else '—'
        util=f"{100*d['transporter_day_utilization']:.1f}%" if d else '—'
        lines.append(f"| {row['label']} | {case['target']:.0%} | {case['incoming_count']} | {case['seed']} | "
            f"{row['status']} | {cost} | {peak} | {util} | {d.get('outside_queue_peak','—')} |")
    lines+=['','Day utilization and average occupancy are lower bounds if truncated before day-end.',
        'Completion first; partial costs not ranked. All service/cost/latency components remain in row JSONs.',
        'Long durations are a common transfer shift for the frozen models; no retraining.',
        'Three instances per level: descriptive pilot, not independent test-panel evidence.']
    (output/'REPORT.md').write_text('\n'.join(lines)+'\n')
    result=dict(observed_rows=len(rows),expected_rows=54,complete_rows=sum(r['status']=='complete' for r in rows),
                training_runs=0,test_panel_opened=False)
    prior.original.discovery.write(output/'report.json',result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('prepare','run','status'))
    parser.add_argument('--output',type=Path,default=OUTPUT)
    args=parser.parse_args()
    torch.set_num_threads(1)
    contract=prepare(args.output)
    if args.command=='run':
        evaluator=prior.evaluator;rollout=evaluator.rollout
        native_run=rollout.run
        native_wall=evaluator.training.wall_limit
        def execute(env,method,**kwargs):
            kwargs['max_decisions']=4000
            return native_run(env,method,**kwargs)
        def bound(env):
            agent=prior.original.model_class(0).from_checkpoint(torch.load(contract['models'][0]['file'],
                map_location='cpu',weights_only=False),device='cpu')
            agent.bind_context_environment(env,wait_cap=80,required_jobs=len(env.blocks),max_steps=3600)
            agent.reset_episode_state()
            return agent
        def log(*values,**kwargs):
            builtins.print(*(str(v).replace('E140','E146').replace('regime=','day=').replace('/6','/9')
                             for v in values),**kwargs)
        try:
            with patch.object(rollout,'RecordedGridEnv',DayEnv),patch.object(rollout,'HORIZON',CAP),\
                 patch.object(rollout,'run',execute),patch.object(rollout.GridAgent,'bound',bound),\
                 patch.object(evaluator,'model_class',prior.original.model_class),\
                 patch.object(evaluator.training,'wall_limit',lambda _:native_wall(1800)),\
                 patch.object(evaluator,'print',log,create=True):
                for case in contract['cases']:
                    methods=[(m,None) for m in ('dynamic_pslap','ga','certified_dynamic')]
                    methods += [(f'original_seed{i["model_seed"]}_ep60',i) for i in contract['models']]
                    for label,info in methods:
                        path=args.output/f'rows/{label}/cell-{case["index"]}.json'
                        if path.exists():
                            continue
                        row=evaluator.run_row(args.output,contract,label,case,info)
                        row['day_statistics']=LAST_ENV.day_metrics()
                        prior.original.discovery.write(path,row)
                        analyze(args.output,contract)
        finally:
            analyze(args.output,contract)
    print(json.dumps(analyze(args.output,contract),indent=2))


if __name__=='__main__':
    main()
