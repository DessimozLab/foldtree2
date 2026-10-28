"""Grow the training split while preserving an existing refiner holdout."""
import argparse
import json
from pathlib import Path
import sys
import torch
import yaml
from scripts.se3.prepare_se3_larger_cohort import fresh_splits, main as populate_cache
from foldtree2.se3_validation import base


def growth_splits(dataset, retained, train_count, max_residues=256, seed=42):
    train, val = retained['train_ids'], retained['val_ids']
    if len(set(train + val)) != len(train + val) or not val or train_count <= len(train):
        raise ValueError('Growth requires disjoint existing splits and more training structures')
    excluded = set(train + val) | set(retained['excluded_prior_ids'])
    new, empty, skipped = fresh_splits(dataset, excluded, train_count - len(train), 0,
                                      max_residues, seed)
    assert not empty
    return train + new, list(val), skipped


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    # Frame checks are small tensors; many CPU threads add substantial overhead.
    torch.set_num_threads(2)
    config = yaml.safe_load(Path(args.config).read_text())
    retained = json.loads(Path(config['retain_cohort']).read_text())
    if (config['train_count'], config['val_count'], config['max_residues']) != (8192, 512, 256):
        raise ValueError('Expected the authorized 8192-training / retained-512-validation trial')
    if (len(retained['train_ids']), len(retained['val_ids'])) != (4096, 512):
        raise ValueError('Expected the completed 4096/512 source cohort')
    output = Path(config['cohort_provenance'])
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        record = json.loads(output.read_text())
        if record['selection_config'] != config or record['identity'] != retained['identity']:
            raise ValueError('Growth cohort provenance changed')
    else:
        dataset = base.pdbgraphmk2.StructureDataset(retained['config']['dataset'])
        train, val, skipped = growth_splits(dataset, retained, config['train_count'],
                                           config['max_residues'], config['seed'])
        record = {
            'identity': retained['identity'], 'config': retained['config'],
            'selection_config': config, 'train_ids': train, 'val_ids': val,
            'skipped': skipped, 'excluded_prior_ids': retained['excluded_prior_ids'],
            'retained_cohort': config['retain_cohort'],
            'scope': 'Retained refiner holdout; additional training IDs exclude all earlier holdouts. Production pretraining independence not claimed',
        }
        output.write_text(json.dumps(record, indent=2) + '\n')
    sys.argv = [sys.argv[0], '--config', args.config]
    populate_cache()


if __name__ == '__main__':
    main()
