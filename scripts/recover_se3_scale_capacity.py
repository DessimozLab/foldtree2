"""One bounded capacity recovery after the width-80 CLI allowlist failure."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from scripts import submit_se3_scale_capacity as capacity
from scripts.se3_budget import reserve

PREFIX='scale_capacity_v3r1_'
ORIGINAL_JOBS={'debug_two':'3675418','width64':'3675419','width80':'3675420'}


def cancelled_without_start(rows,job):
    own=[row for row in rows if row[0]==str(job)]
    children=[row for row in rows if row[0].startswith(str(job)+'.')]
    if (len(own)!=1 or not own[0][1].startswith('CANCELLED') or int(own[0][2])!=0
            or own[0][3] not in ('None','Unknown','N/A','') or children):
        raise ValueError('Reservation reuse requires a cancelled, never-started job with no steps')
    return own[0]


def accounting():
    value=subprocess.check_output(['sacct','-j',','.join(ORIGINAL_JOBS.values()),'-n','-P',
        '--format=JobIDRaw,State,ElapsedRaw,Start'],text=True)
    rows=[line.split('|') for line in value.splitlines() if line.strip()]
    return {suffix:cancelled_without_start(rows,job) for suffix,job in ORIGINAL_JOBS.items()}


def submit_reusing(ledger,stage,parent,job,cmd,rows,submit):
    old_budget=json.loads(Path('configs/clariden_se3/scale_capacity_v3_budget.json').read_text())
    expected=next(row for row in old_budget['stages'] if row['name']==parent)
    for flag,value in (('--account','a0117'),('--partition','normal'),('--gpus-per-task','4'),
                       ('--ntasks-per-node','1'),('--nodes',str(expected['nodes'])),
                       ('--time',str(round(expected['hours']*60)))):
        if [arg for arg in cmd if arg==flag or arg.startswith(flag+'=')]!=[flag+'='+value]:
            raise ValueError('Recovery exceeds/mismatches the retained reservation resources')
    cancelled_without_start([rows],job)
    with ledger.open('a+') as stream:
        fcntl.flock(stream,fcntl.LOCK_EX);stream.seek(0)
        entries=[json.loads(line) for line in stream if line.strip()]
        existing=[row for row in entries if row['stage']==parent]
        if len(existing)!=1 or abs(existing[0]['reserved_gpuh']-expected['gpuh'])>1e-9:
            raise ValueError('Original reservation missing or changed')
        if any(row['stage']==stage or row.get('parent_reservation')==parent for row in entries):
            raise ValueError('The one-time recovery allowance was already used')
        record={'stage':stage,'reserved_gpuh':0,'parent_reservation':parent,
                'cancelled_job':job,'accounting':rows,'reservation_ceiling_gpuh':expected['gpuh'],
                'time':time.time(),'command':cmd,'state':'submitting',
                'reason':'Original dependent job never started; original reservation remains charged'}
        if not submit:return record
        stream.seek(0,2);stream.write(json.dumps(record)+'\n');stream.flush();os.fsync(stream.fileno())
        result=subprocess.run(cmd,check=True,capture_output=True,text=True)
        identifier=result.stdout.strip().split(';')[0]
        if not identifier.isdigit():raise ValueError('Unexpected Slurm receipt')
        record['job_id']=identifier
        return record


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--submit',action='store_true')
    parser.add_argument('--available-credit',type=float,required=True)
    args=parser.parse_args()
    code=Path(__file__).resolve().parents[1];os.chdir(code)
    capacity.validate_inputs(code,prefix=PREFIX)
    capacity.prepare_checks(code,prefix=PREFIX)
    prior=accounting()
    ledger=capacity.ROOT/'budget.jsonl'
    entries=[json.loads(line) for line in ledger.read_text().splitlines() if line.strip()]
    if any(row['stage'].startswith(PREFIX) for row in entries):
        raise ValueError('Capacity recovery already reserved/submitted; do not repeat')
    if sum(row['reserved_gpuh'] for row in entries)+4/3>256 or args.available_credit<44/3:
        raise ValueError('Insufficient rollout budget or confirmed credit for recovery')
    (code/'logs').mkdir(exist_ok=True)
    os.environ.update(TRAIN_WORKDIR=str(code),EDF_PATH=str(code/'alps/se3/extended-pinned.toml'),
        TRAIN_VENV=str(capacity.ROOT/'environments/028cc752968f218de4e5837f75b2e7a8a004162a'))
    suffixes=('debug_one','debug_two','width64','width80')
    for suffix in suffixes:
        cmd=capacity.command(code,PREFIX+suffix,prefix=PREFIX)
        if suffix!='debug_one':
            submit_reusing(ledger,PREFIX+suffix,capacity.PREFIX+suffix,ORIGINAL_JOBS[suffix],
                           cmd,prior[suffix],False)
        subprocess.run(cmd[:1]+['--test-only']+cmd[1:],check=True,capture_output=True,text=True)
    dependency=None
    for suffix in suffixes:
        stage=PREFIX+suffix
        cmd=capacity.command(code,stage,dependency,prefix=PREFIX)
        if suffix=='debug_one':
            reserve(code/'configs/clariden_se3/scale_capacity_v3r1_budget.json',ledger,stage,
                    args.available_credit,cmd,args.submit)
            record={'stage':stage,'reserved_gpuh':4/3,'command':cmd}
            if args.submit:
                receipt=json.loads(Path(str(ledger)+'.'+stage+'.receipt.json').read_text())
                record['job_id']=receipt['stdout'].strip().split(';')[0]
        else:
            record=submit_reusing(ledger,stage,capacity.PREFIX+suffix,ORIGINAL_JOBS[suffix],
                                  cmd,prior[suffix],args.submit)
        record.update(code_revision=code.name,dependency=dependency,
            controller_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            credit_basis='Prior user confirmation; within original 256 GPUh ceiling, not portal balance')
        print(json.dumps(record),flush=True)
        if args.submit:
            if not record['job_id'].isdigit():raise ValueError('Invalid job identifier')
            (capacity.ROOT/'artifacts'/(stage+'-submission.json')).write_text(json.dumps(record,indent=2)+'\n')
            dependency=record['job_id']


if __name__=='__main__':main()
