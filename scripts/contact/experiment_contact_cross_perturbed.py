"""Run a matched spherical screen with distinct query/key projections."""

import os as _bootstrap_os, sys as _bootstrap_sys
_bootstrap_root = _bootstrap_os.path.dirname(_bootstrap_os.path.dirname(_bootstrap_os.path.dirname(_bootstrap_os.path.abspath(__file__))))
if _bootstrap_root not in _bootstrap_sys.path:
    _bootstrap_sys.path.insert(0, _bootstrap_root)
import argparse
import json
from pathlib import Path

from foldtree2.se3_cache import digest
from foldtree2.src.contact_cross_dot_perturbed import PerturbedCrossDotHead
import scripts.contact.experiment_contact_objectives as trainer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', default='runs/se3_distance_data_scaling_5120_20261022/comparison/bundle_5120')
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--train-count', type=int, default=128)
    parser.add_argument('--val-count', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=12)
    parser.add_argument('--pairs', type=int, default=512)
    parser.add_argument('--hidden', type=int, default=128)
    parser.add_argument('--contact-weight', type=float, default=.3)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    args.cases = [f'64:{args.contact_weight:g}']
    root = Path(args.output)
    if root.exists():
        raise FileExistsError(root)
    sources = [Path(__file__), Path('foldtree2/src/contact_cross_dot_perturbed.py'),
               Path('foldtree2/src/contact_dot_experiments.py')]
    hashes = {str(p): digest(p) for p in sources}
    trainer.PairwiseDistanceHead = PerturbedCrossDotHead
    try:
        trainer.run(args)
        if hashes != {str(p): digest(p) for p in sources}:
            raise RuntimeError('Cross-dot implementation changed during training')
    finally:
        if root.exists() and (root / 'provenance.json').exists():
            from scripts.se3.diagnose_se3_reconstruction import atomic_json
            provenance = json.loads((root / 'provenance.json').read_text())
            provenance['variant'] = {
                'model_class': 'foldtree2.src.contact_cross_dot_perturbed.PerturbedCrossDotHead',
                'name': 'cross_dot_perturbed', 'sources_sha256': hashes,
                'initial_key_perturbation_std': .001, 'perturbation_seed': 9281,
            }
            atomic_json(root / 'provenance.json', provenance)
            for index, path in enumerate(sources):
                (root / f'variant-source-{index}-{path.name}').write_bytes(path.read_bytes())


if __name__ == '__main__':
    main()
