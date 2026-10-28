"""Queue the authorized bounded loss comparison behind recovery/topology gates."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
from scripts.se3.se3_budget import reserve

ROOT=Path('/capstor/store/cscs/swissai/a0117/foldtree2/se3')
STAGES=('loss_v1_debug_one','loss_v1_debug_two','loss_v1_control','loss_v1_local_distances','loss_v1_backbone','loss_v1_combined')
PARENT=ROOT/'runs/growth_fp64_8192/best.ckpt'
PARENT_SHA='93ff91d18ff9b4b0df5396889f9cd6c4fc0c91f65f88775caf71d2c90487bb04'
BUNDLE_SHA='b2c635e8cf30af851a63eb8f0836ff090fb5707b51523a6b709cca8b707877f3'


def command(code,stage,dependency=None):
    code=Path(code).resolve();is_check='debug' in stage;nodes=1 if stage.endswith('_one') else 2
    minutes=10 if is_check else 45
    cmd=['sbatch','--account=a0117','--partition=normal',f'--nodes={nodes}',f'--time={minutes}','--ntasks-per-node=1','--gpus-per-task=4','--cpus-per-task=8','--parsable',f'--job-name=ft2-se3-{stage}',f'--output={code}/logs/{stage}-%j.out']
    if dependency:cmd.append('--dependency=afterok:'+dependency)
    config='configs/clariden_se3/'+stage+'.yaml'
    train='bash "$TRAIN_WORKDIR/alps/se3/train.sbatch"'
    warm=' --warm-start '+str(PARENT)+' --allow-loss-change'
    if not is_check:return cmd+[str(code/'alps/se3/train.sbatch'),'--config',config,'--warm-start',str(PARENT),'--allow-loss-change']
    if stage.endswith('_one'):
        body=f'set -eu\n{train} --config {config}{warm} --stop-after-updates 1\n'
        body+=f'{train} --config {config} --resume-from {ROOT}/runs/{stage}/last.ckpt --stop-after-updates 2\n'
        body+=f'{train} --config configs/clariden_se3/loss_v1_debug_control.yaml{warm} --stop-after-updates 2\n'
        control='loss_v1_debug_control';exact=' --exact'
    else:
        body=f'set -eu\n{train} --config {config}{warm} --stop-after-updates 2\n';control='loss_v1_debug_control';exact=''
    check=f'set -eu\ncd "$TRAIN_WORKDIR"\nsource "$TRAIN_VENV/bin/activate"\npython -m scripts.se3.verify_se3_loss_checks {ROOT}/runs/{stage} {ROOT}/runs/{control}{exact} --output {ROOT}/artifacts/{stage}-gate.json\n'
    body+='srun --nodes=1 --ntasks=1 --environment="$EDF_PATH" --network=disable_rdzv_get --mpi=none --cpu-bind=none --kill-on-bad-exit=1 bash -c '+shlex.quote(check)+'\n'
    return cmd+['--wrap',body]


def validate_inputs(root=ROOT):
    if hashlib.sha256((root/'runs/growth_fp64_8192/best.ckpt').read_bytes()).hexdigest()!=PARENT_SHA:raise ValueError('Parent checkpoint identity mismatch')
    m=json.loads(Path('/iopsstor/scratch/cscs/dmoi/se3/bundle_growth_8192/manifest.json').read_text())
    identity=dict(m);expected=identity.pop('bundle_sha256')
    if expected!=BUNDLE_SHA or hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()!=expected:raise ValueError('Cache identity mismatch')
    for stage in STAGES+('loss_v1_debug_control',):
        if (root/'runs'/stage).exists():raise ValueError('Output already exists: '+stage)
    prior=[json.loads(x) for x in (root/'runs/growth_fp64_8192/events.jsonl').read_text().splitlines()]
    if len([x for x in prior if x['event']=='train_epoch'])!=16 or any(x['event']=='failure' for x in prior):raise ValueError('Parent training gate failed')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--submit',action='store_true');p.add_argument('--available-credit',type=float,required=True);a=p.parse_args()
    code=Path(__file__).resolve().parents[1];validate_inputs()
    ledger=ROOT/'budget.jsonl';rows=[json.loads(x) for x in ledger.read_text().splitlines()]
    if sum(x['reserved_gpuh'] for x in rows)+26>256 or a.available_credit<26:raise ValueError('Insufficient loss-comparison budget')
    if any(x['stage'] in STAGES for x in rows):raise ValueError('Loss comparison already reserved')
    os.chdir(code);(code/'logs').mkdir(exist_ok=True)
    os.environ.update(TRAIN_WORKDIR=str(code),EDF_PATH=str(code/'alps/se3/extended-pinned.toml'),TRAIN_VENV=str(ROOT/'environments/028cc752968f218de4e5837f75b2e7a8a004162a'))
    for stage in STAGES:
        cmd=command(code,stage);r=subprocess.run(cmd[:1]+['--test-only']+cmd[1:],capture_output=True,text=True)
        if r.returncode:raise RuntimeError('Preflight rejected '+stage+': '+r.stderr)
    dependency=None
    for stage in STAGES:
        cmd=command(code,stage,dependency)
        reserve(code/'configs/clariden_se3/loss_v1_budget.json',ledger,stage,a.available_credit,cmd,a.submit)
        if a.submit:
            r=json.loads(Path(str(ledger)+'.'+stage+'.receipt.json').read_text());job=r['stdout'].strip().split(';')[0]
            if not job.isdigit():raise ValueError('Invalid receipt')
            (ROOT/'artifacts'/(stage+'-submission.json')).write_text(json.dumps({'job_id':job,'stage':stage,'command':cmd,'code_revision':code.name,'parent_sha256':PARENT_SHA,'bundle_sha256':BUNDLE_SHA,'dependency':dependency,'credit_basis':'User confirmed sufficient credit; authorized rollout ceiling 256 GPUh'},indent=2)+'\n')
            dependency=job

if __name__=='__main__':main()
