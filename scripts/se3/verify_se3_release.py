"""Verify deployable archive completeness before spending a Slurm allocation."""

import os as _bootstrap_os, sys as _bootstrap_sys
_bootstrap_root = _bootstrap_os.path.dirname(_bootstrap_os.path.dirname(_bootstrap_os.path.dirname(_bootstrap_os.path.abspath(__file__))))
if _bootstrap_root not in _bootstrap_sys.path:
    _bootstrap_sys.path.insert(0, _bootstrap_root)
import argparse
import json
import tarfile

REQUIRED = {
    'alps/se3/requirements.txt', 'alps/se3/prepare.sbatch', 'alps/se3/train.sbatch',
    'alps/se3/extended-pinned.toml', 'alps/se3/inspect_environment.py',
    'configs/clariden_se3/budget.json', 'configs/clariden_se3/debug_1nodes.yaml',
    'configs/clariden_se3/debug_2nodes.yaml', 'configs/clariden_se3/small.yaml',
    'configs/clariden_se3/medium.yaml', 'foldtree2/train_se3_lightning.py',
    'foldtree2/se3_cache.py', 'foldtree2/se3_validation.py', 'foldtree2/se3_gradient_accumulation.py',
    'foldtree2/src/se3_struct_decoder.py', 'foldtree2/src/geometry_seeds.py',
    'foldtree2/src/losses/se3_ablation.py', 'foldtree2/src/losses/fape.py',
    'scripts/se3/se3_budget.py', 'scripts/se3/se3_capacity_gate.py',
}


def verify(path):
    with tarfile.open(path, 'r:*') as archive:
        files = {m.name: m for m in archive.getmembers() if m.isfile()}
        missing = sorted(REQUIRED - set(files))
        if missing:
            raise ValueError('Incomplete SE3 release: ' + ', '.join(missing))
        if any(files[name].size == 0 for name in REQUIRED):
            raise ValueError('Empty release dependency')
        pins = archive.extractfile(files['alps/se3/requirements.txt']).read().decode()
        if 'gotennet-pytorch==0.3.1' not in pins or 'pytorch-lightning==2.6.0' not in pins:
            raise ValueError('Missing validated refiner dependency pins')
    return {'verified': True, 'required_files': len(REQUIRED)}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('archive')
    a = p.parse_args(); print(json.dumps(verify(a.archive)))
