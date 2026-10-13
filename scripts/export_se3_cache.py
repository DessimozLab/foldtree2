"""Export only selected frozen entries; original HDF5 stays on the workstation."""
import argparse
import json
import shutil
from pathlib import Path
import torch
from foldtree2.se3_cache import digest, identity_hash, open_bundle
from scripts.investigate_se3_scale import load_cohort_entries


def export(cohort, calibration, destination):
    # Performs the original dataset/checkpoint/extractor provenance checks locally.
    source, entries = load_cohort_entries(cohort)
    root = Path(destination)
    if root.exists():
        raise ValueError('Use a fresh bundle destination')
    root.mkdir(parents=True)
    (root / 'entries').mkdir()
    cal = json.loads(Path(calibration).read_text())
    if set(cal['train_ids']) & set(source['val_ids']):
        raise ValueError('Calibration overlaps validation')
    cache = Path(source['config']['cache']) / identity_hash(source['identity'])
    files = {}
    for identifier in source['train_ids'] + source['val_ids']:
        # Legacy filenames hash the raw ID, not JSON serialization.
        import hashlib
        name = hashlib.sha256(identifier.encode()).hexdigest() + '.pt'
        shutil.copyfile(cache / name, root / 'entries' / name)
        files[identifier] = {'name': name, 'sha256': digest(root / 'entries' / name),
                             'length': len(entries[identifier]['coords'])}
    manifest = {'format': 'se3-frozen-bundle-v1', 'input_identity': source['identity'],
                'cohort_sha256': digest(cohort), 'calibration_sha256': digest(calibration),
                'scale': cal['scale'], 'calibration_ids': cal['train_ids'],
                'train_ids': source['train_ids'], 'val_ids': source['val_ids'], 'files': files,
                'feature_dim': entries[source['train_ids'][0]]['features'].shape[-1],
                'scope': source.get('scope', 'Structure-level refiner holdout')}
    manifest['bundle_sha256'] = identity_hash(manifest)
    (root / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    open_bundle(root)
    print(json.dumps({'bundle': str(root), 'identity': manifest['bundle_sha256'], 'entries': len(files)}))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cohort', required=True); p.add_argument('--calibration', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args(); export(a.cohort, a.calibration, a.output)
