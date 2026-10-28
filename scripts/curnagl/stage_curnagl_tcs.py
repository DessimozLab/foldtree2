#!/usr/bin/env python3
"""Build a small explicit TCS deployment archive; excludes datasets and scratch files."""

import os as _bootstrap_os, sys as _bootstrap_sys
_bootstrap_root = _bootstrap_os.path.dirname(_bootstrap_os.path.dirname(_bootstrap_os.path.dirname(_bootstrap_os.path.abspath(__file__))))
if _bootstrap_root not in _bootstrap_sys.path:
    _bootstrap_sys.path.insert(0, _bootstrap_root)
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parents[1]


def build(destination, benchmark, original, foldmason):
    selected = {}
    def add(path, name):
        # Archive regular content, not dangling conda-library symlinks.
        path = Path(path).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.stat().st_size > 100 * 1024**2:
            raise ValueError(f'Oversized deployment input: {path}')
        selected[name] = path
    # Runtime source only; no notebooks, training datasets, runs, git metadata or tests.
    for path in (ROOT / 'foldtree2').rglob('*.py'):
        if not {'tests', 'notebooks', '__pycache__'} & set(path.relative_to(ROOT).parts):
            add(path, 'foldtree2/' + str(path.relative_to(ROOT)))
    for path in (ROOT / 'scripts').glob('*.py'):
        add(path, 'foldtree2/' + str(path.relative_to(ROOT)))
    for relative in ['configs/production_models.yaml', 'foldtree2/config/aaindex1.csv',
                     'foldtree2/config/3diphy_submats/Q.3Di.AF',
                     'foldtree2/mafft_tools/hex2maffttext', 'foldtree2/mafft_tools/maffttext2hex',
                     'foldtree2/madroot/mad', 'foldtree2/raxml-ng/raxml-ng',
                     'foldtree2/raxml-ng/LICENSE.txt', 'workflow/curnagl_tcs.smk']:
        add(ROOT / relative, 'foldtree2/' + relative)
    entries = json.loads((ROOT / 'configs/production_models.yaml').read_text())['models']
    for entry in entries:
        directory = ROOT / entry['directory']
        for key in ('encoder', 'decoder', 'mafft', 'raxml'):
            add(directory / entry[key], 'foldtree2/' + entry['directory'] + '/' + entry[key])
        for path in directory.glob(Path(entry['encoder']).stem + '*json'):
            add(path, 'foldtree2/' + str(path.relative_to(ROOT)))
    add(Path(benchmark) / 'src/topology_benchmark.py', 'benchmark/src/topology_benchmark.py')
    add(Path(original) / 'src/foldtree/foldseek2tree.py', 'original/src/foldtree/foldseek2tree.py')
    add(foldmason, 'bin/foldmason')
    # Ship its conda shared libraries as an isolated layer, not a global install.
    for library in ('libatomic.so.1', 'libz.so.1', 'libbz2.so.1.0', 'libgomp.so.1', 'libstdc++.so.6', 'libgcc_s.so.1'):
        add(Path(foldmason).parent.parent / 'lib' / library, 'lib/' + library)
    provenance = {'foldtree2_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                  'files': {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in selected.items()},
                  'bytes': sum(p.stat().st_size for p in selected.values()),
                  'note': 'Source content hashes are authoritative, including any working-tree differences.'}
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    with tarfile.open(destination, 'w:gz') as archive:
        for name, path in sorted(selected.items()):
            archive.add(path, arcname=name, recursive=False)
    destination.with_suffix(destination.suffix + '.manifest.json').write_text(json.dumps(provenance, indent=2) + '\n')
    print(json.dumps({'archive': str(destination), 'files': len(selected), 'bytes': provenance['bytes'],
                      'archive_bytes': destination.stat().st_size}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--benchmark', required=True, type=Path)
    parser.add_argument('--original', required=True, type=Path)
    parser.add_argument('--foldmason', required=True, type=Path)
    args = parser.parse_args()
    build(args.out, args.benchmark, args.original, args.foldmason)
