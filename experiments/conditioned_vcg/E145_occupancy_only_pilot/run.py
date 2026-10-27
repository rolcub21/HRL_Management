"""Three paired development instances per fill; no training or reference dates."""
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import random
from unittest.mock import patch

import numpy as np
import torch

from experiments.conditioned_vcg.E141_original_frozen_workload_transfer_205k_v1 import run as original
from experiments.conditioned_vcg.E140_frozen_grid_behavior_comparison_204k_v1 import run as evaluator
from experiments.conditioned_vcg.E135_independent_grid_reference_199k_v1 import reference
from experiments.conditioned_vcg.E136_grid_interface_gate_200k_v1.controller import GridAgent
from experiments.conditioned_vcg.E136_grid_interface_gate_200k_v1.environment import GridEnv
from experiments.conditioned_vcg.E142_frozen_regime_gradient_audit.run import file_sha

ROOT=original.ROOT
OUTPUT=ROOT/'results/vcg-conditioned-e145-occupancy-only-pilot'
SEEDS=(24050000,24050001,24050002)
FILLS=(0.,.4,.7,.9)


def case(seed,fill,*,placement_offset=1451,date_offset=1452):
    if fill not in FILLS:
        raise ValueError('Undeclared occupancy')
    env=original.discovery.source.original.new_env('A_outflow_bounded')
    incoming=env.sample_episode_instance(seed)
    cells=sorted(incoming.storage_positions)
    capacity=len(cells)
    count=int(capacity*fill)
    random.Random(seed+placement_offset).shuffle(cells)
    # Independent streams: neither placement nor selection reads due dates.
    last_release=max(incoming.arrival_steps)
    dates=np.random.default_rng(seed+date_offset).integers(0,last_release+1,capacity).tolist()
    durations=np.maximum(1,np.random.default_rng(seed+1453).poisson(80,capacity)).tolist()
    instance=replace(incoming,arrival_steps=(0,)*count+incoming.arrival_steps,
        storage_steps_needed=tuple(durations[:count])+incoming.storage_steps_needed)
    dues={f'B{i+1}':int(dates[i]) for i in range(count)}
    dues.update({f'B{count+i+1}':a+p for i,(a,p) in enumerate(
        zip(incoming.arrival_steps,incoming.storage_steps_needed))})
    return dict(instance=instance.to_dict(),initial_positions=cells[:count],external_due_dates=dues,
        seed=seed,requested_fill=fill,actual_fill=count/capacity,initial_count=count,
        incoming_count=incoming.number_blocks,capacity=capacity,
        initial_due_rule=f'independent integer Uniform[0,{last_release}] residual deadlines at time0',
        initial_due_dates=dates[:count],incoming_arrivals=list(incoming.arrival_steps),
        incoming_durations=list(incoming.storage_steps_needed),
        initial_completion_feasibility='not established; no outcome-conditioned rejection or resampling')


def setup_metrics(value):
    env=GridEnv(value)
    blocked=[]
    for block in env.blocks[:value['initial_count']]:
        approach=reference.path(env,env.current_state,block.position,source=block.position)
        transport=any(reference.path(env,block.position,exit_cell,source=block.position) is not None
                      for exit_cell in env.exit_cells)
        if approach is None or not transport:
            blocked.append(block.label)
    dates=value['initial_due_dates']
    return dict(initial_count=value['initial_count'],actual_fill=value['actual_fill'],
        due_min=min(dates) if dates else None,due_max=max(dates) if dates else None,
        due_std=float(np.std(dates)) if dates else None,
        direct_retrieval_obstruction_count=len(blocked),obstructed_labels=blocked,
        obstruction_definition='no clear transporter approach OR no clear loaded route to an exit without relocation',
        completion_feasibility_verified=False)


def prepare(output):
    cases=[]
    setups=[]
    for level,fill in enumerate(FILLS):
        for replicate,seed in enumerate(SEEDS):
            value=json.loads(json.dumps(case(seed,fill)))
            path=output/'instances'/f'fill-{fill:g}-seed-{seed}.json'
            if path.exists() and original.discovery.read(path)!=value:
                raise ValueError('Prepared case changed')
            if not path.exists():
                original.discovery.write(path,value)
            index=level*3+replicate
            cases.append(dict(index=index,path=str(path.resolve()),sha256=file_sha(path),
                fill=fill,replicate=replicate,seed=seed,panel='pilot' if level<3 else 'near_full'))
            setups.append(dict(index=index,seed=seed,requested_fill=fill,**setup_metrics(value)))
    sources=[Path(__file__),Path(__file__).with_name('README.txt'),ROOT/'tests/test_vcg_e145_occupancy.py',
        Path(original.__file__),Path(evaluator.__file__),Path(reference.__file__),
        ROOT/'experiments/conditioned_vcg/E137_grid_rollout_readiness_201k_v1/rollout.py',
        ROOT/'experiments/conditioned_vcg/E27_certified_dynamic_sustained_99k/selector.py']
    models=[]
    for seed in range(3):
        info=original.discovery.read(original.OUTPUT/f'models/seed-{seed}.json')
        path=original.OUTPUT/info['file']
        if file_sha(path)!=info['snapshot_sha256']:
            raise ValueError('Frozen original checkpoint changed')
        models.append({**info,'file':str(path.resolve())})
    contract=dict(protocol='e145_occupancy_only_pilot_v1',cases=cases,models=models,
        original_layout='unchanged8x8',incoming='native36 jobs; gap30; native Poisson80; due=a+p',
        primary_fills=[0.,.4,.7],near_full=.9,
        near_full_trigger='all six plain Dynamic/GA rows at70% complete; then explicitly run near-full',
        initial_dates='Uniform integer[0,last incoming release], independent of all-cell random placement; no witness dates',
        placement='nested random prefixes of ALL usable storage cells; no protected corridor or due sorting',
        controller_alignment='certified Dynamic delegates to original.model_class(0) base; '
            'same E137 snapshot/certificate/Defer-counter/witness guard as frozen VCG',
        budgets=dict(wall_seconds=900,horizon=6000,certificate_nodes=2000,decisions=1000),
        claims='initial-occupancy transfer; descriptive pilot; initial recoverability NOT guaranteed',
        source_sha256={str(p.relative_to(ROOT)):file_sha(p) for p in sources},
        test_panel_opened=False,training_runs=0)
    contract['contract_sha256']=hashlib.sha256(json.dumps(contract,sort_keys=True).encode()).hexdigest()
    contract=json.loads(json.dumps(contract))
    path=output/'contract.json'
    if path.exists() and original.discovery.read(path)!=contract:
        raise ValueError('Contract differs; use new output')
    if not path.exists():
        original.discovery.write(path,contract)
    original.discovery.write(output/'setup.json',dict(rows=setups))
    return contract


def near_full_eligible(output):
    for label in ('dynamic_pslap','ga'):
        for index in (6,7,8):
            path=output/f'rows/{label}/cell-{index}.json'
            if not path.exists() or original.discovery.read(path)['status']!='complete':
                return False
    return True


def analyze(output,contract):
    rows=[original.discovery.read(p) for p in (output/'rows').glob('*/*.json')]
    lines=['# Occupancy-only frozen pilot','','Lower J is better; failures are retained, not ranked by partial cost.','',
        '| Method | Initial fill | Instance | Status | J | Rehandles | Wait |',
        '|---|---:|---:|---|---:|---:|---:|']
    for row in sorted(rows,key=lambda r:(r['regime'],r['label'])):
        cell=contract['cases'][row['regime']]
        m=row['metrics'] or {}
        cost=f"{m['J']:.1f}" if m.get('cost_ranking_eligible') else '—'
        lines.append(f"| {row['label']} | {cell['fill']:.0%} | {cell['seed']} | {row['status']} | {cost} | "
            f"{m.get('components',{}).get('rehandles','—')} | {m.get('mean_observed_admission_wait','—')} |")
    lines+=['','Three instances per level; no statistical superiority claim.',
        'Initial dates are spread residual deadlines, not a reconstructed stationary prehistory.',
        'Initial recoverability is unverified: failure does not alone establish that certification would rescue an instance.',
        'Plain compound retrieval versus certified elementary macro mismatch remains.',
        'Full timing, work, within-window, completion and latency/primitive records are in row JSONs.']
    (output/'REPORT.md').write_text('\n'.join(lines)+'\n')
    result=dict(observed_rows=len(rows),pilot_expected_rows=54,
        complete_rows=sum(r['status']=='complete' for r in rows),
        near_full_eligible=near_full_eligible(output),test_panel_opened=False,training_runs=0)
    original.discovery.write(output/'report.json',result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('prepare','run','status'))
    parser.add_argument('--panel',choices=('pilot','near-full'),default='pilot')
    parser.add_argument('--output',type=Path,default=OUTPUT)
    args=parser.parse_args()
    torch.set_num_threads(1)
    contract=prepare(args.output)
    if args.command=='run':
        if args.panel=='near-full' and not near_full_eligible(args.output):
            raise ValueError('Near-full trigger not met: retain/report primary failures first')
        cells=[c for c in contract['cases'] if c['panel']==('pilot' if args.panel=='pilot' else 'near_full')]
        def bound(env):
            base=original.model_class(0).from_checkpoint(torch.load(contract['models'][0]['file'],
                map_location='cpu',weights_only=False),device='cpu')
            base.bind_context_environment(env,wait_cap=80,required_jobs=len(env.blocks),max_steps=3600)
            base.reset_episode_state()
            return base
        try:
            with patch.object(evaluator,'model_class',original.model_class),patch.object(GridAgent,'bound',bound):
                for cell in cells:
                    for label in ('dynamic_pslap','ga','certified_dynamic'):
                        evaluator.run_row(args.output,contract,label,cell)
                    for info in contract['models']:
                        evaluator.run_row(args.output,contract,f'original_seed{info["model_seed"]}_ep60',cell,info)
        finally:
            analyze(args.output,contract)
    print(json.dumps(analyze(args.output,contract),indent=2))


if __name__=='__main__':
    main()
