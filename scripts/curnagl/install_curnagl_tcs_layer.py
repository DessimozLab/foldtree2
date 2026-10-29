#!/usr/bin/env python3
"""Install supplied wheels offline in a NEW deployment-local layer, never conda base."""

import os as _bootstrap_os, sys as _bootstrap_sys
_bootstrap_root = _bootstrap_os.path.dirname(_bootstrap_os.path.dirname(_bootstrap_os.path.dirname(_bootstrap_os.path.abspath(__file__))))
if _bootstrap_root not in _bootstrap_sys.path:
    _bootstrap_sys.path.insert(0, _bootstrap_root)
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys


def install(root, wheels):
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError('Use the Curnagl foldtree2 Python 3.12 interpreter')
    target = root / 'python-layer'
    if target.exists():
        raise FileExistsError('Runtime layer already exists; use a new layer/deployment')
    subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-index', '--no-deps',
                    '--target', str(target), *map(str, wheels)], check=True)
    hashes = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in target.rglob('*') if p.is_file() and '__pycache__' not in p.parts}
    record = {'python': sys.version, 'wheels': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in wheels},
              'files': hashes, 'shared_environment_modified': False}
    (root / 'runtime-layer.json').write_text(json.dumps(record, indent=2) + '\n')
    # Update generated operational job scripts BEFORE any inference jobs start.
    for batch in root.glob('*.sbatch'):
        lines = batch.read_text().splitlines()
        rewritten = []
        for line in lines:
            if line.startswith('export PYTHONPATH='):
                existing = shlex.split(line)[1].split('=', 1)[1]
                line = 'export PYTHONPATH=' + shlex.quote(str(target) + ':' + existing)
            rewritten.append(line)
        batch.write_text('\n'.join(rewritten) + '\n')
    print(json.dumps({'runtime_layer': str(target), 'files': len(hashes), 'shared_environment_modified': False}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--deployment', required=True, type=Path)
    parser.add_argument('wheels', nargs='+', type=Path)
    args = parser.parse_args()
    install(args.deployment.resolve(), [p.resolve() for p in args.wheels])
