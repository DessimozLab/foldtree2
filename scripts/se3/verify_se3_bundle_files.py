"""Verify transfer custody using only the standard library, without loading tensors."""

import os as _bootstrap_os, sys as _bootstrap_sys
_bootstrap_root = _bootstrap_os.path.dirname(_bootstrap_os.path.dirname(_bootstrap_os.path.dirname(_bootstrap_os.path.abspath(__file__))))
if _bootstrap_root not in _bootstrap_sys.path:
    _bootstrap_sys.path.insert(0, _bootstrap_root)
import argparse
import hashlib
import json
from pathlib import Path


def verify(directory):
    root=Path(directory);manifest=json.loads((root/'manifest.json').read_text())
    identity=dict(manifest);expected=identity.pop('bundle_sha256')
    actual=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    if actual!=expected:raise ValueError('Manifest identity mismatch')
    ids=manifest['train_ids']+manifest['val_ids']
    if len(set(ids))!=len(ids) or set(ids)!=set(manifest['files']):raise ValueError('Cache split inventory mismatch')
    size=0
    for identifier,record in manifest['files'].items():
        path=root/'entries'/record['name']
        if path.parent!=root/'entries':raise ValueError('Cache filename escapes bundle')
        h=hashlib.sha256()
        with path.open('rb') as f:
            for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
        if h.hexdigest()!=record['sha256']:raise ValueError('Transferred cache hash mismatch: '+identifier)
        size+=path.stat().st_size
    result={'bundle_sha256':expected,'files':len(ids),'bytes':size,'verified':True}
    print(json.dumps(result));return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('bundle');p.add_argument('--output')
    a=p.parse_args();result=verify(a.bundle)
    if a.output:Path(a.output).write_text(json.dumps(result,indent=2)+'\n')
