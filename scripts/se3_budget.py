"""Reserve bounded Slurm allocations against the 256 GPUh rollout ceiling."""
import argparse
import fcntl
import json
import os
import subprocess
from pathlib import Path
import time


def reserve(schedule, ledger, stage, credit, command, submit=False):
    if credit is None or credit<0:
        raise ValueError('A checked remaining project credit balance is required')
    budget=json.loads(Path(schedule).read_text())
    choice,=[s for s in budget['stages'] if s['name']==stage]
    cost=choice['gpuh']
    path=Path(ledger);path.parent.mkdir(parents=True,exist_ok=True)
    # Lock the durable ledger even across independent submission clients.
    with path.open('a+') as f:
        fcntl.flock(f,fcntl.LOCK_EX);f.seek(0)
        rows=[json.loads(line) for line in f if line.strip()]
        spent=sum(r['reserved_gpuh'] for r in rows)
        if any(r['stage']==stage for r in rows):
            raise ValueError('Stage already reserved; diagnose/reconcile explicitly before a correction')
        if spent+cost>budget['ceiling_gpuh'] or credit<cost:
            raise ValueError('GPUh ceiling or available credit would be exceeded')
        if not command or Path(command[0]).name!='sbatch':
            raise ValueError('Expected an explicit sbatch command')
        print(json.dumps({'stage':stage,'nodes':choice['nodes'],'gpus':4*choice['nodes'],
              'hours':choice['hours'],'gpuh':cost,'prior_reserved_gpuh':spent,'command':command}),flush=True)
        # Require resource values to match the reservation; reject ambiguous overrides.
        expected={'--nodes':str(choice['nodes']), '--account':'a0117', '--time':str(round(choice['hours']*60))}
        for flag,value in expected.items():
            matches=[arg for arg in command if arg == flag or arg.startswith(flag+'=')]
            if matches != [flag+'='+value]:
                raise ValueError(f'Required resource argument: {flag}={value}')
        if submit:
            row={'stage':stage,'reserved_gpuh':cost,'time':time.time(),'command':command,'state':'submitting'}
            f.seek(0,2);f.write(json.dumps(row)+'\n');f.flush(); os.fsync(f.fileno())
            # Keep a reservation even on submission failure until reconciled; fail closed.
            result=subprocess.run(command,check=True,capture_output=True,text=True)
            print(result.stdout,flush=True)
            record=Path(str(path)+'.'+stage+'.receipt.json')
            record.write_text(json.dumps({'stdout':result.stdout,'time':time.time()},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--schedule',default='configs/clariden_se3/budget.json')
    p.add_argument('--ledger',required=True);p.add_argument('--stage',required=True)
    p.add_argument('--available-credit',type=float,required=True);p.add_argument('--submit',action='store_true')
    p.add_argument('command',nargs=argparse.REMAINDER);a=p.parse_args()
    reserve(a.schedule,a.ledger,a.stage,a.available_credit,a.command,a.submit)
