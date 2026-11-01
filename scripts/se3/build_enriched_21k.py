#!/usr/bin/env python3
"""Build enriched cache: copy already-enriched entries + enrich new ones from structaln."""
import json, hashlib, os, shutil, time, torch
from pathlib import Path
from torch_geometric.data import HeteroData
from foldtree2.se3_cache import digest, open_bundle
from foldtree2.src.graphs.pdbgraphmk2 import StructureDataset
from foldtree2.se3_validation import production_pair

# Minimal atomic save
def atomic_save(obj, path):
    tmp = str(path) + '.tmp'
    torch.save(obj, tmp)
    os.replace(tmp, str(path))

DEVICE = torch.device('cuda:1')
SRC_ENRICHED = 'runs/se3_distance_data_scaling_10240_20261024/enriched'
RAW_CACHE = 'runs/se3_distance_data_scaling_45000_20261031/raw'
OUTPUT = 'runs/se3_distance_data_scaling_45000_20261031/enriched'
DATASET = '/home/dmoi/projects/foldtree2/foldtree2/structalnfinal.h5'

src = open_bundle(SRC_ENRICHED)
raw = open_bundle(RAW_CACHE)
already_enriched = set(src['train_ids'] + src['val_ids'])
needs_enrich = [i for i in raw['train_ids'] + raw['val_ids'] if i not in already_enriched]
print(f'{len(already_enriched)} already enriched, {len(needs_enrich)} need enrichment')

# Load decoder branches
pair = production_pair()
decoder = torch.load(pair['decoder'], map_location='cpu', weights_only=False)
aa_decoder = decoder.decoders['sequence_transformer'].to(DEVICE).eval()
for p in aa_decoder.parameters(): p.requires_grad_(False)
geometry = decoder.decoders['geometry_cnn']
temperature, bias = float(geometry.contact_temp), float(geometry.contact_bias)
print(f'Temperature={temperature:.4f} bias={bias:.4f}')

ds = StructureDataset(DATASET)

root = Path(OUTPUT); root.mkdir(parents=True, exist_ok=False)
(root / 'entries').mkdir()
files = {}
t0 = time.time()

# Copy already-enriched entries
for identifier in already_enriched:
    name = raw['files'][identifier]['name']
    shutil.copyfile(Path(SRC_ENRICHED) / 'entries' / name, root / 'entries' / name)
    files[identifier] = {**raw['files'][identifier], 'sha256': digest(root / 'entries' / name)}
print(f'Copied {len(files)} existing enriched entries')

# Process entries that need enrichment from scratch
for idx, identifier in enumerate(needs_enrich):
    if (idx + 1) % 100 == 0:
        rate = (time.time() - t0) / max(idx, 1)
        eta = (len(needs_enrich) - idx) * rate / 60
        print(f'  {idx+1}/{len(needs_enrich)} ETA {eta:.0f}min')
    name = raw['files'][identifier]['name']
    if identifier not in ds.structlist:
        print(f'  SKIP {identifier}: not in structaln HDF5')
        continue
    entry = torch.load(Path(RAW_CACHE) / 'entries' / name, map_location='cpu', weights_only=False)
    graph = ds[identifier].to(DEVICE)
    positions = graph['positions'].x.to(DEVICE).float()
    latent = entry['latent'].to(DEVICE)
    aa_input = HeteroData(); aa_input['res'].x = latent.clone(); aa_input['positions'].x = positions
    with torch.no_grad():
        logits = aa_decoder(aa_input)['aa']
    if logits.shape == (len(latent), 1, 20):
        logits = logits[:, 0, :]
    entry['aa_probs'] = logits.float().softmax(-1).cpu()
    contact = entry['contact_z'].float()
    entry['contact_probs'] = torch.sigmoid(temperature * (contact @ contact.T) + bias).clamp(1e-6, 1 - 1e-6)
    torch.save(entry, root / 'entries' / name)
    files[identifier] = {**raw['files'][identifier], 'sha256': digest(root / 'entries' / name)}
    if idx == 0:
        # Verify deterministic
        aa2 = aa_decoder(aa_input)['aa']
        if aa2.shape == (len(latent), 1, 20): aa2 = aa2[:, 0, :]
        assert torch.allclose(entry['aa_probs'], aa2.float().softmax(-1).cpu(), atol=1e-6), 'AA not deterministic'

# Build manifest
train_ids, val_ids = [], []
for i in raw['train_ids']:
    if i in files: train_ids.append(i)
for i in raw['val_ids']:
    if i in files: val_ids.append(i)
manifest = {k: v for k, v in raw.items() if k != 'bundle_sha256'}
manifest['files'] = files; manifest['train_ids'] = train_ids; manifest['val_ids'] = val_ids
manifest['decoder_enrichment'] = {'protocol': 'frozen-decoder-enrichment-v1',
    'contact_dim': 30, 'aa_dim': 20, 'temperature': temperature, 'bias': bias}
manifest['bundle_sha256'] = hashlib.sha256(
    json.dumps({k: v for k, v in manifest.items() if k != 'bundle_sha256'},
               sort_keys=True).encode()).hexdigest()
(root / 'manifest.json').write_text(json.dumps(manifest, indent=2))
open_bundle(OUTPUT)
print(f'Done! {len(train_ids)} train / {len(val_ids)} val -> {OUTPUT}')