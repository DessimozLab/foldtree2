#!/usr/bin/env python3
"""Render an operational TCS workflow with disjoint Slurm steps for each rule."""

import os as _bootstrap_os, sys as _bootstrap_sys
_bootstrap_root = _bootstrap_os.path.dirname(_bootstrap_os.path.dirname(_bootstrap_os.path.dirname(_bootstrap_os.path.abspath(__file__))))
if _bootstrap_root not in _bootstrap_sys.path:
    _bootstrap_sys.path.insert(0, _bootstrap_root)
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def render(source):
    prefix = "    shell: '"
    resource = '    resources: mem_mb='
    if source.count(prefix) != 6 or source.count(resource) != 6 or 'srun ' in source:
        raise ValueError('Expected the unmodified six-command TCS workflow')
    source = source.replace("PYTHON = CFG['python']", "PYTHON = CFG['python']\n"
                            "PHYSICAL_STEP = str(Path(PLAN_PATH).parent / 'curnagl_physical_step.py')")
    hook = '''
def inference_threads(wildcards):
    completed = Path(ROOT) / 'families' / wildcards.family / wildcards.method / 'completed.json'
    if completed.is_file():
        # The frozen driver still verifies the complete signature and output hashes.
        return int(json.loads(completed.read_text())['signature']['threads'])
    return 1 if PLAN['methods'][wildcards.method]['kind'] in {'AA', '3Di', 'FT2'} else CFG['threads']

'''
    source = source.replace('wildcard_constraints:', hook + 'wildcard_constraints:', 1)
    infer_start = source.index('rule infer:')
    source = source[:infer_start] + source[infer_start:].replace(
        "    threads: CFG['threads']", '    threads: inference_threads', 1)
    source = source.replace(resource, '    resources: slurm_cpus=lambda wildcards, threads: 2 * threads, mem_mb=')
    return source.replace(
        prefix,
        prefix + 'srun --exclusive --exact --ntasks=1 --cpus-per-task={resources.slurm_cpus} '
        '--hint=nomultithread --threads-per-core=1 --cpu-bind=cores --mem={resources.mem_mb}M '
        '{PYTHON:q} {PHYSICAL_STEP:q} --threads {threads} -- ',
    )


def install(root):
    source = root / 'foldtree2/workflow/curnagl_tcs.smk'
    output = root / 'slurm-steps.smk'
    manifest = root / 'slurm-step-workflow.json'
    worker = root / 'curnagl_physical_step.py'
    if output.exists() or manifest.exists() or worker.exists():
        raise FileExistsError('Operational step workflow already exists')
    content = source.read_bytes()
    rendered = render(content.decode()).encode()
    shutil.copyfile(Path(__file__).with_name(worker.name), worker)
    output.write_bytes(rendered)
    manifest.write_text(json.dumps({
        'source': str(source), 'source_sha256': hashlib.sha256(content).hexdigest(),
        'workflow': str(output), 'workflow_sha256': hashlib.sha256(rendered).hexdigest(),
        'worker': str(worker), 'worker_sha256': hashlib.sha256(worker.read_bytes()).hexdigest(),
        'policy': 'exclusive exact Slurm steps; two reserved CPUs per tool thread; '
                  'verified one hardware thread per physical core',
    }, indent=2) + '\n')
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--deployment', required=True, type=Path)
    args = parser.parse_args()
    print(install(args.deployment.resolve()))
