"""E144 v2: arrival-aware warm-start, 60-episode first stage only."""
import argparse
import hashlib
import json
from pathlib import Path
import signal
from unittest.mock import patch

import torch

from . import run as v1
from . import arrival

OUTPUT=v1.ROOT/'results/vcg-conditioned-e144-warm-start-mixture-arrival-v2-repair1'
NATIVE_VALIDATE=v1.validate
NATIVE_SAVE=v1.save


def save(output,contract,agent,history,validations):
    NATIVE_SAVE(output,contract,agent,history,validations)
    path=output/'seed-0/summary.json'
    summary=v1.base.read(path)
    summary['maximum_adaptation_episodes']=60
    summary['arrival_window_steps']=120
    summary['continue_decision']=decision(validations)
    v1.base.write(path,summary)


def decision(validations):
    endpoints={v['episode']:v for v in validations}
    if not all(n in endpoints for n in (0,30,60)):
        return dict(status='pending',continue_authorized=False)
    required=[endpoints[n] for n in (0,30,60)]
    if not all(all(r['strict_complete'] for r in v['rows']) for v in required):
        return dict(status='stop_reassess_validation_failure',continue_authorized=False)
    easy_limit=1.10*endpoints[0]['easy_J']
    easy_ok=all(v['easy_J']<=easy_limit for v in required[1:])
    hard=lambda v: sum(r['metrics']['J'] for r in v['rows'] if r['case'] in (7,8,9,10))
    j0,j30,j60=map(hard,required)
    improving=j60<=.95*j30 and j60<j0
    return dict(status='eligible_to_consider_continuation' if easy_ok and improving else 'stop_reassess',
        continue_authorized=False,eligible=easy_ok and improving,easy_limit=easy_limit,
        easy_retained=easy_ok,hard_improving=improving,hard_J=dict(ep0=j0,ep30=j30,ep60=j60),
        note='No automatic episodes beyond60; no test-panel or extra-seed authorization')


def prepare(output):
    # Reuse unchanged v1 data/reference authentication without overwriting its
    # contract. This parent directory is dedicated to v2 provenance only.
    parent=v1.prepare(output/'v1-provenance')
    paths=[Path(__file__),Path(arrival.__file__),Path(__file__).with_name('ARRIVAL_V2_README.txt'),
           v1.ROOT/'tests/test_vcg_e144_arrival.py']
    contract={**parent,'protocol':'e144_observed_arrival_warm_start_v2_repair1',
        'defer_counter_reset':'only flow progress, matching E137/E141 certified baseline; not arrival-only events',
        'maximum_adaptation_episodes':60,'episodes_per_regime':10,
        'validation_endpoints':[0,6,30,60],
        'arrival_features':arrival.FEATURES,'arrival_window_steps':120,
        'initial_inventory_counts_as_arrivals':False,'future_release_schedule_visible':False,
        'initialization':'source weights copied; added arrival projection exactly zero; fresh optimizer/replay',
        'criterion':'At0/30/60 all11 validation cases complete; easy J at30 and60 <=1.10*ep0; '
                    'sum four congested J at60 <=.95*ep30 and <ep0. Pause at60 regardless.',
        'sources':{**parent['sources'],**{str(p.relative_to(v1.ROOT)):v1.file_sha(p) for p in paths}}}
    contract.pop('contract_sha256')
    contract['contract_sha256']=hashlib.sha256(json.dumps(contract,sort_keys=True).encode()).hexdigest()
    contract=json.loads(json.dumps(contract))
    path=output/'contract.json'
    if path.exists() and v1.base.read(path)!=contract:
        raise ValueError('Arrival-v2 contract changed; choose fresh output')
    if not path.exists():
        v1.base.write(path,contract)
    return contract


def validate(output,contract,agent,endpoint):
    result=NATIVE_VALIDATE(output,contract,agent,endpoint)
    # Endpoint0 is also a no-learning prediction-preservation rollout check.
    if endpoint==0:
        for row in result['rows']:
            path=v1.original.OUTPUT/f'rows/original_seed0_ep60/cell-{row["case"]}.json'
            reference=v1.base.read(path)
            if row['strict_complete'] and abs(row['metrics']['J']-reference['metrics']['J'])>1e-3:
                raise RuntimeError(f'Warm-start reproduction differs on case {row["case"]}; investigate before training')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('prepare','train','status'))
    parser.add_argument('--through-episode',type=int,default=60)
    parser.add_argument('--output',type=Path,default=OUTPUT)
    parser.add_argument('--device',choices=('cuda',),default='cuda')
    args=parser.parse_args()
    if not 1<=args.through_episode<=60:
        parser.error('First-stage budget is1..60, not360')
    torch.set_num_threads(1)
    def terminate(*_):
        raise KeyboardInterrupt('termination requested')
    signal.signal(signal.SIGTERM,terminate)
    contract=prepare(args.output)
    with patch.object(v1,'WarmAgent',arrival.ArrivalAgent),patch.object(v1,'initialize',arrival.initialize),patch.object(v1,'validate',validate),patch.object(v1,'save',save),patch.object(v1.base,'episode',arrival.episode):
        if args.command=='train':
            result=v1.train(args.output,contract,args.through_episode,args.device)
        elif args.command=='status':
            path=args.output/'seed-0/summary.json'
            result=v1.base.read(path) if path.exists() else dict(status='not_started')
        else:
            result=dict(status='prepared_not_launched',maximum_adaptation_episodes=60)
    path=args.output/'seed-0/summary.json'
    if path.exists():
        gate=decision(v1.base.read(path)['validation_history'])
        v1.base.write(args.output/'continue-decision.json',gate)
        result['continue_decision']=gate
    result.pop('validation_history',None)
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
