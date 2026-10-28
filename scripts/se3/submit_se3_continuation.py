"""Submit the authorized two-rate continuation, sequentially and within budget."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
from scripts.se3.se3_budget import reserve

ROOT=Path('/capstor/store/cscs/swissai/a0117/foldtree2/se3')
STAGES=('continue_lr5e4','continue_lr1e4')
PARENT_SHA='93ff91d18ff9b4b0df5396889f9cd6c4fc0c91f65f88775caf71d2c90487bb04'


def command(code,stage,dependency=None):
    cmd=['sbatch','--account=a0117','--partition=normal','--nodes=2','--time=120','--ntasks-per-node=1','--gpus-per-task=4','--cpus-per-task=8','--parsable',f'--job-name=ft2-se3-{stage}',f'--output={code}/logs/{stage}-%j.out']
    if dependency:cmd.append('--dependency=afterok:'+dependency)
    return cmd+[str(code/'alps/se3/train.sbatch'),'--config','configs/clariden_se3/'+stage+'.yaml','--resume-from',str(ROOT/'artifacts/continuation'/ (stage+'.ckpt'))]


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--submit',action='store_true');p.add_argument('--available-credit',type=float,required=True);a=p.parse_args()
    code=Path(__file__).resolve().parents[1]
    ledger=ROOT/'budget.jsonl';rows=[json.loads(x) for x in ledger.read_text().splitlines()]
    if sum(x['reserved_gpuh'] for x in rows)+32>256 or a.available_credit<32:raise ValueError('Insufficient paired budget')
    if any(x['stage'] in STAGES for x in rows):raise ValueError('Continuation already reserved')
    for stage in STAGES:
        if (ROOT/'runs'/stage).exists():raise ValueError('Output exists; explicit recovery required')
        path=ROOT/'artifacts/continuation'/(stage+'.ckpt');m=json.loads(path.with_suffix('.json').read_text())
        if m['parent_sha256']!=PARENT_SHA or hashlib.sha256(path.read_bytes()).hexdigest()!=m['branch_sha256']:raise ValueError('Branch checkpoint identity mismatch')
    os.chdir(code);(code/'logs').mkdir(exist_ok=True)
    os.environ.update(TRAIN_WORKDIR=str(code),EDF_PATH=str(code/'alps/se3/extended-pinned.toml'),TRAIN_VENV=str(ROOT/'environments/028cc752968f218de4e5837f75b2e7a8a004162a'))
    for stage in STAGES:
        cmd=command(code,stage);r=subprocess.run(cmd[:1]+['--test-only']+cmd[1:],capture_output=True,text=True)
        if r.returncode:raise RuntimeError(r.stderr)
    dependency=None
    for stage in STAGES:
        cmd=command(code,stage,dependency)
        reserve(code/'configs/clariden_se3/continuation_budget.json',ledger,stage,a.available_credit,cmd,a.submit)
        if a.submit:
            receipt=json.loads(Path(str(ledger)+'.'+stage+'.receipt.json').read_text());job=receipt['stdout'].strip().split(';')[0]
            if not job.isdigit():raise ValueError('Invalid submission receipt')
            (ROOT/'artifacts'/(stage+'-submission.json')).write_text(json.dumps({'job_id':job,'code_revision':code.name,'command':cmd,'parent_sha256':PARENT_SHA,'dependency':dependency,'credit_basis':'User confirmed sufficient project credit; existing 256 GPUh ceiling'},indent=2)+'\n')
            dependency=job

if __name__=='__main__':main()
