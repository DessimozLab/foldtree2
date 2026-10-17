#!/usr/bin/env python3
"""Preserve incomplete native 3Di attempts before a TCS allocation restarts."""
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--deployment', required=True, type=Path)
    parser.add_argument('--attempt', required=True)
    args = parser.parse_args()
    print(json.dumps(preserve_incomplete(args.deployment.resolve(), args.attempt)), flush=True)
