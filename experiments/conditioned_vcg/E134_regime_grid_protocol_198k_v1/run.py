"""Freeze prospective choices or rescore an already recorded trajectory."""
import argparse
import json
from pathlib import Path
from .objective import Job,rescore
from . import protocol as p


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n')


def prepare(output):
    value=p.contract()
    path=output/'protocol.json'
    if path.exists():
        if json.loads(path.read_text())!=value: raise RuntimeError('E134 prospective design changed; use a fresh version')
    else:
        if output.exists() and any(output.iterdir()): raise RuntimeError('Need empty protocol output')
        write(path,value)
    print(f'E134 prospective design recorded: {path}; training and test gates remain CLOSED')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('prepare','rescore'))
    parser.add_argument('--output',type=Path,default=p.OUTPUT)
    parser.add_argument('--trajectory',type=Path)
    parser.add_argument('--rescore-output',type=Path)
    args=parser.parse_args()
    prepare(args.output.resolve())
    if args.command=='rescore':
        if not args.trajectory or not args.rescore_output: parser.error('Need --trajectory and --rescore-output')
        if args.rescore_output.exists(): raise RuntimeError('Never overwrite a trajectory or prior sensitivity result')
        payload=json.loads(args.trajectory.read_text())
        result=rescore([Job(**row) for row in payload['jobs']],observation_end=payload['observation_end'],
            equipment_work=payload['equipment_work'],rehandles=payload['rehandles'],
            strict_complete=payload['strict_complete'])
        import hashlib
        result.update(protocol_sha256=p.contract()['contract_sha256'],
            trajectory_sha256=hashlib.sha256(args.trajectory.read_bytes()).hexdigest())
        write(args.rescore_output,result)


if __name__=='__main__': main()
