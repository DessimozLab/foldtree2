#!/usr/bin/env python3
"""Direct enrichment: add aa_probs + contact_probs to raw cache entries."""
import json, hashlib, time, torch
from pathlib import Path
from torch_geometric.data import HeteroData
from foldtree2.se3_cache import digest, open_bundle
from foldtree2.src.graphs.pdbgraphmk2 import StructureDataset
from foldtree2.se3_validation import production_pair, sha256

DEVICE = torch.device('cuda:1')
CACHE = 'runs/se3_distance_data_scaling_45000_20261031/raw'
OUTPUT = 'runs/se3_distance_data_scaling_45000_20261031/enriched'
DATASET = '/home/dmoi/projects/foldtree2/foldtree2/structalnfinal.h5'

manifest = open_bundle(CACHE)
print(f'Enriching {len(manifest["train_ids"])} train + {len(manifest["val_ids"])} val')

# Load decoder branches
pair = production_pair()
decoder = torch.load(pair['decoder'], map_location='cpu', weights_only=False)
aa_decoder = decoder.decoders['sequence_transformer'].to(DEVICE).eval()
for p in aa_decoder.parameters(): p.requires_grad_(False)
geometry = decoder.decoders['geometry_cnn']
temperature = geometry.contact_temp
bias = geometry.contact_bias
print(f'Contact: temperature={float(temperature):.4f} bias={float(bias):.4f}')

# Load dataset for graph positions
ds = StructureDataset(DATASET)

# Create output
root = Path(OUTPUT)
root.mkdir(parents=True, exist_ok=False)
(root / 'entries').mkdir()
files, rows = {}, []
t0 = time.time()
all_ids = manifest['train_ids'] + manifest['val_ids']

for idx, identifier in enumerate(all_ids):
    if (idx + 1) % 100 == 0:
        eta = (len(all_ids) - idx) * (time.time() - t0) / max(idx, 1) / 60
        print(f'  {idx+1}/{len(all_ids)} ETA {eta:.0f}min')
    name = manifest['files'][identifier]['name']
    entry = torch.load(Path(CACHE) / 'entries' / name, map_location='cpu', weights_only=False)
    # AA probabilities
    graph = ds[identifier].to(DEVICE)
    positions = graph['positions'].x.to(DEVICE).float()
    latent = entry['latent'].to(DEVICE)
    aa_input = HeteroData()
    aa_input['res'].x = latent.clone()
    aa_input['positions'].x = positions
    with torch.no_grad():
        logits = aa_decoder(aa_input)['aa']
    if logits.shape == (len(latent), 1, 20):
        aa_logits = aa_logits[:, 0, :]
    aa_probs = aa_logits.float().softmax(-1).cpu()
    # Contact probabilities
    contact = entry['contact_z'].float()
    contact_probs = torch.sigmoid(temperature * (contact @ contact.T) + bias).clamp(1e-6, 1 - 1e-6)
    entry['aa_probs'] = aa_probs
    entry['contact_probs'] = contact_probs
    torch.save(entry, root / 'entries' / name)
    files[identifier] = {**manifest['files'][identifier],
                         'sha256': digest(root / 'entries' / name)}
    if idx == 0:
        # Repeat to verify determinism
        aa2 = aa_decoder({'res': HeteroData(x=latent), 'positions': HeteroData(x=positions)})['aa']
        if aa2.shape == (len(latent), 1, 20):
            aa2 = aa2[:, 0, :]
        assert torch.allclose(aa_probs, aa2.float().softmax(-1).cpu(), atol=1e-6), 'AA not deterministic'

# Build enriched manifest
enriched = dict(manifest)
enriched['files'] = files
enriched['decoder_enrichment'] = {
    'protocol': 'frozen-decoder-enrichment-v1',
    'contact_dim': 30, 'aa_dim': 20,
    'temperature': float(temperature), 'bias': float(bias),
}
enriched['bundle_sha256'] = hashlib.sha256(
    json.dumps({k: v for k, v in enriched.items() if k != 'bundle_sha256'},
               sort_keys=True).encode()).hexdigest()
(root / 'manifest.json').write_text(json.dumps(enriched, indent=2))
print(f'Done: {root} — {len(files)} entries enriched')