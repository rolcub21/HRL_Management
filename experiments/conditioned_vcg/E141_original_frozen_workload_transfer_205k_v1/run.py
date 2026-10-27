import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
from unittest.mock import patch

import torch
from experiments.conditioned_vcg.E131_nominal_cost_residual_pilot_195k_v1 import run as discovery
from experiments.conditioned_vcg.E131_nominal_cost_residual_pilot_195k_v1 import model as original_model
from experiments.conditioned_vcg.E133_residual_two_seed_confirmation_197k_v1 import run as confirmation
from experiments.conditioned_vcg.E136_grid_interface_gate_200k_v1.controller import predict
from experiments.conditioned_vcg.E140_frozen_grid_behavior_comparison_204k_v1 import run as evaluator

ROOT=discovery.ROOT
OUTPUT=ROOT/'results/vcg-conditioned-e141-original-frozen-workload-transfer-205k-v1'


class FrozenOriginal(original_model.PilotAgent):
    def _admissible_prepared(self,snapshot):
        # Retain the OLD input normalization and architecture. Only the known
        # due-date source in nominal K changes from implicit a+p to explicit d.
        with patch.object(original_model,'predict',predict):
            return super()._admissible_prepared(snapshot)

    def bind_context_environment(self,env,**kwargs):
        kwargs['max_steps']=self.config.episode_horizon_steps
        return super().bind_context_environment(env,**kwargs)


def model_class(seed):
    class SeededOriginal(FrozenOriginal):
        @classmethod
        def from_checkpoint(cls,payload,**kwargs):
            kwargs['seed']=seed
            return super().from_checkpoint(payload,**kwargs)
    return SeededOriginal


def source(seed):
    folder=discovery.OUTPUT if seed==0 else confirmation.OUTPUT
    return folder/'residual_q'/f'seed-{seed}'


def freeze(output,seed):
    path=output/'models'/f'seed-{seed}.json'
    if path.exists():
        info=discovery.read(path)
        if evaluator.sha(output/info['file'])!=info['snapshot_sha256']:
            raise RuntimeError('frozen original checkpoint changed')
        return info
    directory=source(seed)
    original=directory/'latest.pth'
    identity=(original.stat().st_size,original.stat().st_mtime_ns)
    source_sha=evaluator.sha(original)
    payload=torch.load(original,map_location='cpu',weights_only=False)
    if payload.get('completed_training_episodes')!=60 or payload.get('e131_variant')!='residual_q':
        raise RuntimeError('requires original residual endpoint60, not mixture or another arm')
    expected=discovery.contract()['contract_sha256'] if seed==0 else confirmation.contract()['contract_sha256']
    key='e131_contract_sha256' if seed==0 else 'e133_contract_sha256'
    if payload.get(key)!=expected or (seed and payload.get('model_seed')!=seed):
        raise RuntimeError('original source identity differs')
    agent=model_class(seed).from_checkpoint(payload,device='cpu',resumable=False)
    snapshot=agent.checkpoint(include_replay=False,e141_original_episode=60)
    snapshot['agent_state'].pop('optimizer',None)
    destination=output/'models'/f'original-seed-{seed}-ep-60.pth'
    destination.parent.mkdir(parents=True,exist_ok=True)
    torch.save(snapshot,destination.with_suffix('.pth.tmp'))
    os.replace(destination.with_suffix('.pth.tmp'),destination)
    if identity!=(original.stat().st_size,original.stat().st_mtime_ns) or source_sha!=evaluator.sha(original):
        raise RuntimeError('original training source changed during snapshot')
    summary=discovery.read(directory/'summary.json')
    endpoint=next(x for x in summary['validation_history'] if x['checkpoint_episode']==60)
    info=dict(model_seed=seed,episode=60,file=str(destination.relative_to(output)),
        source=str(original),source_sha256=source_sha,snapshot_sha256=evaluator.sha(destination),
        weights_sha256=evaluator.parameter_digest(agent),input_horizon=agent.config.episode_horizon_steps,
        historical_native_J={str(x['instance_seed']):x['operator_cost'] for x in endpoint['runs']})
    discovery.write(path,info)
    return info


def prepare(output):
    discovery.prepare(discovery.OUTPUT)
    confirmation.prepare(confirmation.OUTPUT)
    paths=[*Path(__file__).parent.glob('*.py'),Path(__file__).with_name('run.sh'),
           Path(__file__).with_name('README.txt'),ROOT/'tests/test_vcg_e141_original_transfer.py',
           discovery.OUTPUT/'contract.json',confirmation.OUTPUT/'contract.json',
           Path(evaluator.__file__)]
    cases=[]
    # Native control recreates the exact earlier arrival/duration instances;
    # only episode accounting is routed through the explicit-date recorder.
    for index,seed in enumerate(discovery.source.p.VALIDATION_SEEDS):
        env=discovery.source.original.new_env('A_outflow_bounded')
        instance=env.sample_episode_instance(seed)
        env.reset(instance=instance)
        case=dict(instance=instance.to_dict(),initial_positions=[],
            external_due_dates={b.label:b.arrival_step+b.storage_steps_needed for b in env.blocks},
            panel='native_control',instance_seed=seed)
        case=json.loads(json.dumps(case))
        path=output/'instances'/f'native-{seed}.json'
        if path.exists() and discovery.read(path)!=case:
            raise RuntimeError('native control instance changed')
        if not path.exists():discovery.write(path,case)
        cases.append(dict(index=index,panel='control',instance_seed=seed,path=str(path),sha256=evaluator.sha(path)))
    for index in range(6):
        path=evaluator.training.p.OUTPUT/'instances'/f'cell-{index}-seed-{evaluator.training.p.DEV_BASE+index}.json'
        cases.append(dict(index=5+index,panel='workload',workload_regime=index,
                          path=str(path),sha256=evaluator.sha(path)))
    value=dict(protocol='vcg_conditioned_e141_original_frozen_workload_transfer_205k_v1',
        source_sha256={str(p.relative_to(ROOT)):evaluator.sha(p) for p in paths},cases=cases,
        original_model_seeds=[0,1,2],endpoint=60,weights='frozen, no mixture initialization or fitting',
        input_normalization='original checkpoint3600; original graph timing scale and same dimension',
        workload_change='arrival interval, initial inventory, released due-date distribution; geometry fixed8x8',
        epsilon=0,training_runs=0,test_panel_opened=False,wall_seconds_per_row=900,
        execution_horizon=6000,baselines=list(evaluator.BASELINES),
        limitation='reused development instances; native plain baselines have compound retrieval macros')
    value['contract_sha256']=hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()
    path=output/'contract.json'
    if path.exists() and discovery.read(path)!=value:
        raise RuntimeError('transfer contract differs; use new output')
    if not path.exists():discovery.write(path,value)
    for seed in (0,1,2):freeze(output,seed)
    return value


def analyze(output,contract):
    result=evaluator.analyze(output,contract)
    rows=[discovery.read(p) for p in (output/'rows').glob('*/*.json')]
    native=[]
    for row in rows:
        cell=contract['cases'][row['regime']]
        if cell['panel']=='control' and row['model_seed'] is not None:
            info=discovery.read(output/'models'/f'seed-{row["model_seed"]}.json')
            expected=info['historical_native_J'][str(cell['instance_seed'])]
            measured=(row['metrics'] or {}).get('J')
            native.append(dict(model_seed=row['model_seed'],instance_seed=cell['instance_seed'],
                historical_J=expected,measured_J=measured,
                difference=measured-expected if measured is not None else None,status=row['status']))
    discovery.write(output/'native-control-reproduction.json',dict(rows=native,
        interpretation='report reproduction discrepancies before interpreting transfer; no selecting checkpoints'))
    # E140 generic formatting calls indices regimes; replace with correct panel
    # labels because the first FIVE rows are native controls, not grid regimes.
    table=output/'e140-table.md'
    text=table.read_text()
    text=text.replace('Regimes 1/2: empty .60/.85; 3/4: 40% .60/.85; 5/6: approximately70% .60/.85.',
        'Indices 1–5: original native control instances. Indices 6–11: changed workloads '
        '(empty .60/.85;40% .60/.85;approximately70% .60/.85). Geometry is always8x8.')
    table.write_text(text)
    result['expected_rows']=len(contract['cases'])*6
    result['status']='complete' if result['observed_rows']==result['expected_rows'] else 'partial'
    report_path=output/'e140-report.json'
    report=discovery.read(report_path)
    report.update(expected_rows=result['expected_rows'],status=result['status'],
                  scope='original frozen episode60 models; native controls plus same-layout workload transfer')
    discovery.write(report_path,report)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('prepare','run','status'))
    parser.add_argument('--panel',choices=('control','workload','all'),default='all')
    parser.add_argument('--output',type=Path,default=OUTPUT)
    args=parser.parse_args()
    torch.set_num_threads(1)
    def terminate(*_):raise KeyboardInterrupt('termination requested')
    signal.signal(signal.SIGTERM,terminate)
    output=args.output.resolve()
    contract=prepare(output)
    if args.command=='run':
        try:
            with patch.object(evaluator,'model_class',model_class):
                for panel in ('control','workload'):
                    if args.panel not in (panel,'all'):continue
                    for label in evaluator.BASELINES:
                        for cell in contract['cases']:
                            if cell['panel']==panel:evaluator.run_row(output,contract,label,cell)
                    for seed in (0,1,2):
                        info=freeze(output,seed)
                        for cell in contract['cases']:
                            if cell['panel']==panel:
                                evaluator.run_row(output,contract,f'original_seed{seed}_ep60',cell,info)
        except KeyboardInterrupt:
            analyze(output,contract)
            raise SystemExit(130)
    print(json.dumps(analyze(output,contract),indent=2))


if __name__=='__main__':main()
