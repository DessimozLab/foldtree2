"""CPU-only stage-in and content verification of the finished growth cache."""
import hashlib
import json
from pathlib import Path
import shutil


def main():
    root = Path('/capstor/store/cscs/swissai/a0117/foldtree2/se3')
    source = root / 'cache/bundle_growth_8192'
    target = Path('/iopsstor/scratch/cscs/dmoi/se3/bundle_growth_8192')
    report = json.loads((root / 'artifacts/growth-extraction.json').read_text())
    manifest = json.loads((source / 'manifest.json').read_text())
    if not report['passed'] or report['bundle_sha256'] != manifest['bundle_sha256']:
        raise ValueError('Frozen extraction verification missing or differs')
    identity = dict(manifest)
    expected = identity.pop('bundle_sha256')
    if hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest() != expected:
        raise ValueError('Source manifest identity differs')
    shutil.copytree(source, target, dirs_exist_ok=True)
    if (source / 'manifest.json').read_bytes() != (target / 'manifest.json').read_bytes():
        raise ValueError('Copied manifest differs')
    for record in manifest['files'].values():
        path = target / 'entries' / record['name']
        if path.parent != target / 'entries':
            raise ValueError('Invalid cache entry path')
        h = hashlib.sha256()
        with path.open('rb') as f:
            for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
                h.update(chunk)
        if h.hexdigest() != record['sha256']:
            raise ValueError('Copied entry differs: ' + record['name'])
    result = {'passed': True, 'bundle_sha256': expected, 'entries': len(manifest['files']),
              'source': str(source), 'target': str(target)}
    (root / 'artifacts/growth-stage-in.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
