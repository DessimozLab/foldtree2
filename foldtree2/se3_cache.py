"""Portable, content-verified frozen production outputs and global batch order."""
import hashlib
import json
import random
from pathlib import Path
import torch
from torch.utils.data import Dataset, Sampler


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def identity_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def open_bundle(directory):
    root = Path(directory)
    manifest = json.loads((root / 'manifest.json').read_text())
    identity = dict(manifest)
    expected = identity.pop('bundle_sha256')
    if identity_hash(identity) != expected:
        raise ValueError('Cache manifest identity mismatch')
    train, val = manifest['train_ids'], manifest['val_ids']
    if len(set(train + val)) != len(train + val) or not train or not val:
        raise ValueError('Duplicate, overlapping or empty cache splits')
    if set(train + val) != set(manifest['files']):
        raise ValueError('Cache inventory differs from splits')
    for identifier, record in manifest['files'].items():
        path = root / 'entries' / record['name']
        if path.parent != root / 'entries' or digest(path) != record['sha256']:
            raise ValueError(f'Cache file hash mismatch: {identifier}')
    return manifest


class FrozenDataset(Dataset):
    def __init__(self, directory, manifest, split):
        self.root = Path(directory)
        self.manifest = manifest
        self.ids = manifest[split + '_ids']

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        identifier = self.ids[index]
        saved = torch.load(self.root / 'entries' / self.manifest['files'][identifier]['name'],
                           map_location='cpu', weights_only=False)
        if saved['id'] != identifier or saved['identity'] != self.manifest['input_identity']:
            raise ValueError('Cache payload provenance mismatch')
        result = dict(saved['entry'])
        if len(result['coords']) != self.manifest['files'][identifier]['length']:
            raise ValueError('Cache length mismatch')
        result['id'] = identifier
        return result


def global_order(lengths, seed, epoch, batch_size=16):
    if len(lengths) % batch_size:
        raise ValueError('Training count must be divisible by the global batch size')
    ordered = sorted(range(len(lengths)), key=lambda i: (lengths[i], i))
    groups = [ordered[i:i + batch_size] for i in range(0, len(ordered), batch_size)]
    random.Random(seed + epoch).shuffle(groups)
    return groups


class GlobalBatchSampler(Sampler):
    """Cursor is advanced by consumed optimizer updates, never by prefetch."""
    def __init__(self, lengths, state, rank, world_size, seed=42, global_batch=16):
        if global_batch % world_size:
            raise ValueError('World size must divide global batch')
        self.lengths, self.state = lengths, state
        self.rank, self.world_size = rank, world_size
        self.seed, self.global_batch = seed, global_batch

    def __iter__(self):
        groups = global_order(self.lengths, self.seed, self.state['epoch'], self.global_batch)
        for group in groups[self.state['cursor']:]:
            yield from group[self.rank::self.world_size]

    def __len__(self):
        return (len(self.lengths) // self.global_batch - self.state['cursor']) * (self.global_batch // self.world_size)


def single_entry(batch):
    if len(batch) != 1:
        raise ValueError('Only one structure per GPU is supported')
    return batch[0]
