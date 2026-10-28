"""Run a matched spherical-head screen with learned per-residue biases."""
import argparse
import json
from pathlib import Path

from foldtree2.se3_cache import digest
from foldtree2.src.contact_bias_experiment import ResidueBiasDistanceHead
import scripts.experiment_contact_objectives as trainer


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
    sources = [Path(__file__), Path('foldtree2/src/contact_bias_experiment.py')]
    source_hashes = {str(p): digest(p) for p in sources}
    # The override is process-local; the two main training jobs keep their code.
    trainer.PairwiseDistanceHead = ResidueBiasDistanceHead
    try:
        trainer.run(args)
        if source_hashes != {str(p): digest(p) for p in sources}:
            raise RuntimeError('Residue-bias implementation changed during training')
    finally:
        if root.exists() and (root / 'provenance.json').exists():
            from scripts.se3.diagnose_se3_reconstruction import atomic_json
            provenance = json.loads((root / 'provenance.json').read_text())
            provenance['variant'] = {
                'model_class': 'foldtree2.src.contact_bias_experiment.ResidueBiasDistanceHead',
                'contact_score': 'scale * cosine + global_bias + residue_bias_i + residue_bias_j',
                'initial_residue_bias': 0., 'sources_sha256': source_hashes,
            }
            atomic_json(root / 'provenance.json', provenance)
            for index, p in enumerate(sources):
                (root / f'variant-source-{index}-{p.name}').write_bytes(p.read_bytes())


if __name__ == '__main__':
    main()
