#!/usr/bin/env python3
"""Release a stopped TCS full sweep only after recorded physical-affinity checks."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def completed(job):
    rows = subprocess.check_output([
        'sacct', '-j', str(job), '-n', '-P', '--format=JobIDRaw,State,ExitCode'],
        text=True).splitlines()
    rows = [row.split('|') for row in rows if row.strip()]
    if not any(row[0] == str(job) for row in rows):
        raise ValueError(f'No accounting for verification job {job}')
    if any(row[1] != 'COMPLETED' or row[2] != '0:0' for row in rows):
        raise ValueError(f'Verification job {job} did not finish successfully: {rows}')


def release(root, operations, probes, workflow_job):
    receipt = operations / 'resume-receipt.json'
    if receipt.exists():
        raise FileExistsError(f'Full sweep already released: {receipt}')
    active = subprocess.check_output(['squeue', '--me', '-h', '-o', '%i|%j'], text=True)
    if any('ft2-tcs-full' in line or 'ft2-tcs-pilot' in line for line in active.splitlines()):
        raise ValueError('An inference allocation is already queued/running')
    required = {}
    hosts = set()
    binary = root / 'foldtree2/foldtree2/raxml-ng/raxml-ng'
    worker = root / 'curnagl_physical_step.py'
    for job in probes:
        completed(job)
        path = root / 'affinity-checks' / str(job) / 'report.json'
        data = json.loads(path.read_text())
        if (not data['passed'] or len(data['checks']) != 45 or data['concurrency'] != 4
                or data['threads'] != 1 or data['step_cpus'] != 2
                or data['binary_sha256'] != sha(binary) or data['worker_sha256'] != sha(worker)):
            raise ValueError('RAxML verification identity/settings mismatch')
        for check in data['checks']:
            if (not check['passed'] or check['returncode'] != 0
                    or len(check['affinity']['affinity_pinned']) != 1
                    or check['affinity']['physical_cores_available'] < 1
                    or '--force' in check['command'] or 'rand{10},pars{10}' not in check['command']):
                raise ValueError('Incomplete/unsafe RAxML verification')
        hosts.add(data['hostname'])
        required[str(path)] = sha(path)
    if len(hosts) < 2:
        raise ValueError('Require successful RAxML checks on two different nodes')
    completed(workflow_job)
    path = operations / 'workflow-gate.json'
    gate = json.loads(path.read_text())
    candidate = operations / 'slurm-steps.smk'
    if (not gate['passed'] or gate['methods'] != 13 or gate['job_id'] != str(workflow_job)
            or gate['worker_sha256'] != sha(worker) or gate['workflow_sha256'] != sha(candidate)
            or gate['plan_sha256'] != sha(root / 'plan.json')):
        raise ValueError('Workflow verification does not match the proposed release')
    for name, digest in gate['files'].items():
        if sha(name) != digest:
            raise ValueError(f'Workflow verification output changed: {name}')
    required[str(path)] = sha(path)
    pilot = json.loads((root / 'results/pilot/validated.json').read_text())
    if pilot['scores'] != 26 or pilot['methods'] != 13 or pilot['families'] != 2:
        raise ValueError('Pilot validation is missing')
    state = json.loads((root / 'supervision.json').read_text())
    if state['status'] != 'needs_attention' or state['phase'] != 'full':
        raise ValueError('Deployment is not a stopped full sweep')
    base = operations / 'base-full.sbatch'
    batch = (base if base.exists() else root / 'full.sbatch').read_text()
    if '--resources mem_mb=96000 ' not in batch or '#SBATCH --cpus-per-task=32\n' not in batch:
        raise ValueError('Unexpected full allocation settings')
    batch = batch.replace('--resources mem_mb=96000 ', '--resources mem_mb=96000 slurm_cpus=32 ')
    source = root / 'foldtree2/workflow/curnagl_tcs.smk'
    probe_script = operations / 'check_curnagl_raxml_affinity.py'
    recovery_script = operations / 'prepare_curnagl_step_recovery.py'
    required.update({str(worker): sha(worker), str(source): sha(source),
                     str(probe_script): sha(probe_script),
                     str(recovery_script): sha(recovery_script),
                     str(operations / 'curnagl_physical_step.py'): sha(worker),
                     str(root / 'plan.json'): sha(root / 'plan.json'),
                     str(binary): sha(binary), str(root / 'slurm-steps.smk'): sha(candidate)})
    guard = ('import hashlib,json,pathlib\n'
             f'expected = {required!r}\n'
             'for name,digest in expected.items():\n'
             '    actual=hashlib.sha256(pathlib.Path(name).read_bytes()).hexdigest()\n'
             '    if actual != digest: raise RuntimeError("Release input changed: "+name)\n'
             'print("Physical-affinity release hashes verified", flush=True)\n')
    (operations / 'assert_release.py').write_text(guard)
    python = json.loads((root / 'plan.json').read_text())['config']['python']
    batch = batch.replace('set -euo pipefail\n',
                          f'set -euo pipefail\n{python} {operations / "assert_release.py"}\n', 1)
    controller = '/work/FAC/FBM/DBC/cdessim2/default/dmoi/miniconda3/envs/snakemake/bin/snakemake'
    if batch.count(controller) != 1:
        raise ValueError('Expected a single workflow invocation')
    batch = batch.replace(controller,
                          f'{python} -u {probe_script} --deployment {root}\n'
                          f'{python} {recovery_script} --deployment {root} '
                          '--attempt "$SLURM_JOB_ID" --all-ml\n' + controller, 1)
    preview = operations / 'full-verified.sbatch'
    preview.write_text(batch)
    subprocess.run(['sbatch', '--test-only', str(preview)], check=True)
    backup = operations / 'before-full-resume'
    backup.mkdir(exist_ok=False)
    for name in ('full.sbatch', 'slurm-steps.smk', 'slurm-step-workflow.json',
                 'supervision.json', 'operational-source.json'):
        if (root / name).exists():
            shutil.copy2(root / name, backup / name)
    shutil.copyfile(candidate, root / 'slurm-steps.smk')
    shutil.copyfile(preview, root / 'full.sbatch')
    record = {'source': str(source), 'source_sha256': sha(source),
              'workflow': str(root / 'slurm-steps.smk'), 'workflow_sha256': sha(candidate),
              'worker': str(worker), 'worker_sha256': sha(worker),
              'probe_jobs': probes, 'probe_nodes': sorted(hosts), 'workflow_job': workflow_job,
              'policy': 'single-thread ML inference for unfinished methods; '
                        'retain completed methods with their original thread signature; '
                        'two reserved CPUs per tool thread; topology-verified affinity'}
    (root / 'slurm-step-workflow.json').write_text(json.dumps(record, indent=2) + '\n')
    provenance = root / 'operational-source.json'
    hashes = json.loads(provenance.read_text()) if provenance.exists() else {}
    for name in ('full.sbatch', 'slurm-steps.smk', 'slurm-step-workflow.json', 'curnagl_physical_step.py'):
        hashes[name] = sha(root / name)
    for name in ('assert_release.py', 'workflow-gate.json', 'check_curnagl_raxml_affinity.py',
                 'curnagl_physical_step.py'):
        hashes[str((operations / name).relative_to(root))] = sha(operations / name)
    provenance.write_text(json.dumps(hashes, indent=2) + '\n')
    old_job = state['jobs'].pop('full')
    state.setdefault('previous_jobs', []).append({
        'phase': 'full', 'job_id': old_job, 'state': state.get('slurm_state'),
        'reason': 'User-authorized physical-affinity repair; two-node RAxML and 13-method workflow checks passed'})
    state.update(status='active', slurm_state='READY_FOR_SUBMISSION', physical_affinity_recovery=record)
    state.pop('reason', None)
    (root / 'supervision.json').write_text(json.dumps(state, indent=2) + '\n')
    activation = subprocess.check_output(['sbatch', '--parsable', str(root / 'advance.sbatch')], text=True).strip()
    result = {**record, 'activation_job': activation, 'previous_full_job': old_job,
              'user_authorization': '2026-10-19: queue TCS after RAxML verification succeeds'}
    receipt.write_text(json.dumps(result, indent=2) + '\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--deployment', required=True, type=Path)
    parser.add_argument('--operations', required=True, type=Path)
    parser.add_argument('--probe-jobs', required=True, nargs=2, type=int)
    parser.add_argument('--workflow-job', required=True, type=int)
    args = parser.parse_args()
    with (args.deployment / '.advance.lock').open('a') as advance_lock:
        fcntl.flock(advance_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with (args.deployment / '.supervision.lock').open('a') as state_lock:
            fcntl.flock(state_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            print(json.dumps(release(args.deployment, args.operations, args.probe_jobs, args.workflow_job)))
