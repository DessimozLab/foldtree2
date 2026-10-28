"""Export existing mk2 graph tensors for frozen extraction on Clariden."""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import shutil
import torch
from foldtree2.se3_cache import digest, identity_hash
from foldtree2.se3_validation import base, frozen_entry, production_pair, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cohort', required=True)
    parser.add_argument('--previous-bundle', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    cohort = json.loads(Path(args.cohort).read_text())
    previous = json.loads((Path(args.previous_bundle) / 'manifest.json').read_text())
    identity = cohort['identity']
    pair = production_pair()
    if (identity != previous['input_identity'] or sha256(cohort['config']['dataset']) != identity['dataset_sha256']
            or any(sha256(pair[k]) != identity[k + '_sha256'] for k in pair)
            or hashlib.sha256(inspect.getsource(frozen_entry).encode()).hexdigest() != identity['preprocessing']['extractor_sha256']):
        raise ValueError('Frozen graph source/checkpoint/extractor provenance differs')
    if cohort['train_ids'][:4096] != previous['train_ids'] or cohort['val_ids'] != previous['val_ids']:
        raise ValueError('Retained splits differ')
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=False)
    (root / 'graphs').mkdir()
    dataset = base.pdbgraphmk2.StructureDataset(cohort['config']['dataset'])
    identifiers = [previous['train_ids'][0]] + cohort['train_ids'][4096:]
    graphs = {}
    for index, identifier in enumerate(identifiers):
        # Read the existing graph exactly; no graph generation or preprocessing.
        graph = dataset[identifier]
        name = hashlib.sha256(identifier.encode()).hexdigest() + '.pt'
        torch.save({'id': identifier, 'graph': graph.to_dict()}, root / 'graphs' / name)
        graphs[identifier] = {'name': name, 'sha256': digest(root / 'graphs' / name)}
        if (index + 1) % 256 == 0:
            print(json.dumps({'event': 'export_progress', 'count': index + 1, 'total': len(identifiers)}), flush=True)
    for key, path in pair.items():
        shutil.copyfile(path, root / (key + '.pt'))
    template = {k: v for k, v in previous.items() if k not in ('bundle_sha256', 'files', 'cohort_sha256', 'train_ids', 'val_ids')}
    template.update(cohort_sha256=digest(args.cohort), train_ids=cohort['train_ids'], val_ids=cohort['val_ids'])
    record = {'format': 'se3-existing-graph-export-v1', 'identity': identity,
              'cohort_sha256': digest(args.cohort), 'graphs': graphs,
              'parity_id': previous['train_ids'][0], 'bundle_template': template,
              'source_dataset': cohort['config']['dataset']}
    record['export_sha256'] = identity_hash(record)
    (root / 'manifest.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps({'event': 'complete', 'graphs': len(graphs), 'export_sha256': record['export_sha256']}), flush=True)


if __name__ == '__main__':
    main()
