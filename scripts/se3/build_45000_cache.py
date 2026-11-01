#!/usr/bin/env python3
"""Build 45k enriched cache from structalign HDF5, extending existing 10k."""
import hashlib, inspect, json, os, random, shutil, time
from pathlib import Path
from types import SimpleNamespace
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import torch
from foldtree2.se3_cache import digest, identity_hash, open_bundle, FrozenDataset
from foldtree2.se3_validation import base, frozen_entry, load_production_decoder, production_pair, sha256, tensor_digest
from foldtree2.src.graphs.pdbgraphmk2 import StructureDataset

DEVICE = torch.device('cuda:1')
TARGET = 45000
BUNDLE = Path('runs/se3_distance_data_scaling_45000_20261031/raw')
BUNDLE.mkdir(parents=True, exist_ok=False)
(BUNDLE / 'entries').mkdir()
source = open_bundle('runs/se3_distance_data_scaling_10240_20261024/expanded_raw')
existing = set(source['train_ids'] + source['val_ids'])
print(f'Starting from {len(existing)} existing entries, need {TARGET} total')

# Load enc/dec
pair = production_pair()
ds = StructureDataset('/mnt/data1/foldtree2-workstation-data/foldtree2/structalnfinal.h5')
probe = ds[list(ds.structlist)[0]].to(DEVICE)
encoder, _ = base.build_encoder(SimpleNamespace(pretrained_encoder_path=pair['encoder'], pretrained_encoder_full_path=pair['encoder']), probe, DEVICE)
encoder = encoder.float().eval().requires_grad_(False)
decoder = load_production_decoder(pair['decoder'], DEVICE).float().eval().requires_grad_(False)
before = [tensor_digest(m) for m in (encoder, decoder)]

# Select eligible additions
eligible = [i for i in ds.structlist if i not in existing]
random.Random(42).shuffle(eligible)
print(f'{len(eligible)} candidates available')

# Copy existing
src_entries = Path('runs/se3_distance_data_scaling_10240_20261024/expanded_raw/entries')
files = {}
for ident in existing:
    name = hashlib.sha256(ident.encode()).hexdigest() + '.pt'
    shutil.copyfile(src_entries / name, BUNDLE / 'entries' / name)
    files[ident] = {'name': name, 'sha256': digest(BUNDLE / 'entries' / name), 'length': 0}
print(f'Copied {len(files)} existing entries')

# Extract new entries
needed = TARGET - len(existing)
extracted, skipped, fresh_ids = 0, 0, []
t0 = time.time()
for idx, ident in enumerate(eligible):
    if extracted >= needed:
        break
    if (idx + 1) % 100 == 0:
        rate = (time.time() - t0) / max(extracted, 1)
        eta = (needed - extracted) * rate / 60
        print(f'  scan {idx+1}/{len(eligible)} extract {extracted}/{needed} skip {skipped} rate {rate:.1f}s/e ETA {eta:.0f}min')
    try:
        group = ds.h5dataset['structs'][ident]['node']
        n = group['res']['x'].shape[0]
        if not (64 <= n <= 256):
            skipped += 1; continue
        mask = torch.isfinite(torch.from_numpy(group['coords']['x'][:])).all(-1)
        if 'plddt' in group:
            confidence = torch.from_numpy(group['plddt']['x'][:]).reshape(-1)
            mask &= torch.isfinite(confidence) & (confidence >= .5)
        if float(mask.float().mean()) < .9:
            skipped += 1; continue
        graph = ds[ident].to(DEVICE)
        epoch = frozen_entry(graph, encoder, decoder, DEVICE)
        name = hashlib.sha256(ident.encode()).hexdigest() + '.pt'
        torch.save(epoch, BUNDLE / 'entries' / name)
        length = int(mask.sum())
        files[ident] = {'name': name, 'sha256': digest(BUNDLE / 'entries' / name), 'length': length}
        fresh_ids.append(ident); extracted += 1
    except Exception as exc:
        print(f'  skip {ident}: {exc}')
        skipped += 1
if before != [tensor_digest(m) for m in (encoder, decoder)]:
    raise RuntimeError('Frozen state changed')

# Build manifest — keep original val set, add new train entries
train_ids = list(source['train_ids']) + [i for i in fresh_ids if i not in source['val_ids']]
val_ids = list(source['val_ids'])
manifest = {'format': 'se3-frozen-bundle-v1', 'input_identity': source['input_identity'],
            'train_ids': train_ids, 'val_ids': val_ids, 'files': files,
            'feature_dim': source['feature_dim'],
            'scope': f'Extended from 10k to {len(train_ids)}/{len(val_ids)}'}
manifest['bundle_sha256'] = identity_hash(manifest)
(BUNDLE / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
print(f'Done: {len(train_ids)} train / {len(val_ids)} val | {BUNDLE}')
print(f'Extracted {extracted} new / {skipped} skipped / {len(existing)} existing')