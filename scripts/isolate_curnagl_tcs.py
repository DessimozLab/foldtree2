#!/usr/bin/env python3
"""Render an operational TCS workflow with disjoint Slurm steps for each rule."""
import argparse
import hashlib
import json
from pathlib import Path


def render(source):
    prefix = "    shell: '"
    if source.count(prefix) != 6 or 'srun ' in source:
        raise ValueError('Expected the unmodified six-command TCS workflow')
    return source.replace(
        prefix,
        prefix + 'srun --exclusive --exact --ntasks=1 --cpus-per-task={threads} '
        '--cpu-bind=cores --mem={resources.mem_mb}M ',
    )


def install(root):
    source = root / 'foldtree2/workflow/curnagl_tcs.smk'
    output = root / 'slurm-steps.smk'
    manifest = root / 'slurm-step-workflow.json'
    if output.exists() or manifest.exists():
        raise FileExistsError('Operational step workflow already exists')
    content = source.read_bytes()
    rendered = render(content.decode()).encode()
    output.write_bytes(rendered)
    manifest.write_text(json.dumps({
        'source': str(source), 'source_sha256': hashlib.sha256(content).hexdigest(),
        'workflow': str(output), 'workflow_sha256': hashlib.sha256(rendered).hexdigest(),
        'policy': 'exclusive exact Slurm steps; one CPU per workflow thread',
    }, indent=2) + '\n')
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--deployment', required=True, type=Path)
    args = parser.parse_args()
    print(install(args.deployment.resolve()))
