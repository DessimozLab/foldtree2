#!/usr/bin/env python3
"""Preserve incomplete native 3Di attempts before a TCS allocation restarts."""

import os as _bootstrap_os, sys as _bootstrap_sys
_bootstrap_root = _bootstrap_os.path.dirname(_bootstrap_os.path.dirname(_bootstrap_os.path.dirname(_bootstrap_os.path.abspath(__file__))))
if _bootstrap_root not in _bootstrap_sys.path:
    _bootstrap_sys.path.insert(0, _bootstrap_root)
import argparse
import json
from pathlib import Path
import shutil


def preserve_incomplete(root, attempt):
    if not attempt.isdigit():
        raise ValueError('Recovery attempt must be a Slurm job identifier')
    plan = json.loads((root / 'plan.json').read_text())
    families = Path(plan['config']['output_dir']) / 'families'
    destination = root / 'recovery' / ('slurm-steps-' + attempt)
    if destination.exists():
        raise FileExistsError(destination)
    moved = []
    for family in plan['families']:
        native = families / family / '3Di/native'
        if not native.is_dir() or (native / 'completed.json').exists():
            continue
        if (native.parent / 'completed.json').exists():
            raise ValueError(f'Completed method has incomplete native provenance: {native}')
        destination.mkdir(parents=True, exist_ok=True)
        target = destination / family
        shutil.move(str(native), str(target))
        moved.append({'source': str(native), 'preserved': str(target)})
    if moved:
        (destination / 'preserved.json').write_text(json.dumps(moved, indent=2) + '\n')
    return moved


def preserve_incomplete_ml(root, attempt):
    """Preserve unfinished method attempts before changing their thread signature."""
    if not attempt.isdigit():
        raise ValueError('Recovery attempt must be a Slurm job identifier')
    plan = json.loads((root / 'plan.json').read_text())
    families = Path(plan['config']['output_dir']) / 'families'
    destination = root / 'recovery' / ('single-thread-' + attempt)
    if destination.exists():
        raise FileExistsError(destination)
    moved = []
    for family in plan['families']:
        directory = families / family
        if not directory.is_dir():
            continue
        for method in directory.iterdir():
            spec = plan['methods'].get(method.name, {})
            if (spec.get('kind') not in {'AA', '3Di', 'FT2'} or not method.is_dir()
                    or (method / 'completed.json').exists()):
                continue
            if (method / 'tcs/completed.json').exists():
                raise ValueError(f'Completed score has unfinished inference: {method}')
            target = destination / family / method.name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(method), str(target))
            moved.append({'source': str(method), 'preserved': str(target)})
    destination.mkdir(parents=True, exist_ok=True)
    (destination / 'preserved.json').write_text(json.dumps(moved, indent=2) + '\n')
    return moved


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--deployment', required=True, type=Path)
    parser.add_argument('--attempt', required=True)
    parser.add_argument('--all-ml', action='store_true')
    args = parser.parse_args()
    action = preserve_incomplete_ml if args.all_ml else preserve_incomplete
    print(json.dumps(action(args.deployment.resolve(), args.attempt)), flush=True)
