"""Extend the frozen distance cohort to 5120 representatives, retaining its holdout."""

import os as _bootstrap_os, sys as _bootstrap_sys
_bootstrap_root = _bootstrap_os.path.dirname(_bootstrap_os.path.dirname(_bootstrap_os.path.dirname(_bootstrap_os.path.abspath(__file__))))
if _bootstrap_root not in _bootstrap_sys.path:
    _bootstrap_sys.path.insert(0, _bootstrap_root)
import argparse
import copy
import hashlib
import inspect
import json
from pathlib import Path
import random
import shutil
from types import SimpleNamespace

import torch

from foldtree2.se3_cache import digest, identity_hash, open_bundle
from foldtree2.se3_validation import (
    base, frozen_entry, load_production_decoder, production_pair, sha256, tensor_digest)
from scripts.se3.diagnose_se3_reconstruction import atomic_checkpoint, atomic_json
from scripts.se3.scale_distance_data import run as run_scaling, sources as scaling_sources


def select_additions(dataset, excluded, count, seed=42):
    ids = sorted(set(dataset.structlist) - set(excluded))
    random.Random(seed).shuffle(ids)
    selected, skipped = [], {}
    for index, identifier in enumerate(ids):
        if index and index % 512 == 0:
            print(json.dumps({'event': 'selection_progress', 'scanned': index,
                              'selected': len(selected), 'needed': count}), flush=True)
        group = dataset.h5dataset['structs'][identifier]['node']
        n = group['res']['x'].shape[0]
        reason = None
        if not 64 <= n <= 256:
            reason = 'length'
        else:
            mask = torch.isfinite(torch.from_numpy(group['coords']['x'][:])).all(-1)
            if 'plddt' in group:
                confidence = torch.from_numpy(group['plddt']['x'][:]).reshape(-1)
                mask &= torch.isfinite(confidence) & (confidence >= .5)
            valid = mask.nonzero().flatten()
            if float(mask.float().mean()) < .9:
                reason = 'confidence'
            elif len(valid) < 2 or int(valid[-1] - valid[0]) < 12:
                reason = 'no_nonlocal_pairs'
        if reason:
            skipped[reason] = skipped.get(reason, 0) + 1
            continue
        selected.append(identifier)
        if len(selected) == count:
            return selected, skipped
    raise ValueError(f'Insufficient eligible additions: {len(selected)} of {count}')


def sources():
    return {**scaling_sources(), 'scripts/se3/extend_distance_data.py': digest(__file__),
            'foldtree2/se3_validation.py': digest('foldtree2/se3_validation.py')}


def prepare(args):
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=False)
    source = Path(args.previous)
    previous = open_bundle(source)
    exclusion = open_bundle(args.exclude_bundle)
    if len(previous['train_ids']) != 2048 or len(previous['val_ids']) != 256:
        raise ValueError('Expected preceding 2048/256 raw cohort')
    if 'decoder_enrichment' in previous:
        raise ValueError('Extension requires raw frozen entries')
    identity = previous['input_identity']
    if exclusion['input_identity'] != identity:
        raise ValueError('Exclusion cohort identity differs')
    if sha256(args.dataset) != identity['dataset_sha256']:
        raise ValueError('Dataset identity changed')
    pair = production_pair()
    if any(sha256(pair[k]) != identity[k + '_sha256'] for k in pair):
        raise ValueError('Production checkpoints changed')
    extractor = hashlib.sha256(inspect.getsource(frozen_entry).encode()).hexdigest()
    if extractor != identity['preprocessing']['extractor_sha256']:
        raise ValueError('Frozen extractor changed')
    code = sources()
    for name in code:
        path = root / 'sources' / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(Path(name).read_bytes())
    dataset = base.pdbgraphmk2.StructureDataset(args.dataset)
    excluded = set(exclusion['train_ids'] + exclusion['val_ids'])
    excluded.update(previous['train_ids'] + previous['val_ids'])
    print(json.dumps({'event': 'selecting_additional', 'needed': 3072}), flush=True)
    additions, skipped = select_additions(dataset, excluded, 3072)
    atomic_json(root / 'selection.json', {
        'retained_train_ids': previous['train_ids'], 'additional_train_ids': additions,
        'val_ids': previous['val_ids'], 'excluded_previous_ids': sorted(excluded),
        'skipped_counts': skipped, 'sources': code,
        'eligibility': '64-256 residues, >=90% valid, nonlocal pair; seed42'})
    raw = root / 'expanded_raw'
    (raw / 'entries').mkdir(parents=True)
    manifest = copy.deepcopy(previous)
    manifest.pop('bundle_sha256')
    for record in manifest['files'].values():
        shutil.copyfile(source / 'entries' / record['name'], raw / 'entries' / record['name'])
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    probe = dataset[additions[0]].to(device)
    encoder, _ = base.build_encoder(SimpleNamespace(
        pretrained_encoder_path=pair['encoder'], pretrained_encoder_full_path=pair['encoder']), probe, device)
    if isinstance(encoder, base.FrozenProjectionEncoder):
        raise RuntimeError('Encoder fallback forbidden')
    encoder = encoder.float().eval().requires_grad_(False)
    decoder = load_production_decoder(pair['decoder'], device).float().eval().requires_grad_(False)
    before = [tensor_digest(m) for m in (encoder, decoder)]
    for index, identifier in enumerate(additions):
        entry = frozen_entry(dataset[identifier], encoder, decoder, device)
        if float(entry['mask'].float().mean()) < .9:
            raise ValueError('Extracted mask differs from selected eligibility')
        if index == 0:
            repeated = frozen_entry(dataset[identifier], encoder, decoder, device)
            for key, value in entry.items():
                if isinstance(value, torch.Tensor):
                    torch.testing.assert_close(value, repeated[key], atol=0, rtol=0, equal_nan=True)
        name = hashlib.sha256(identifier.encode()).hexdigest() + '.pt'
        atomic_checkpoint(raw / 'entries' / name, {'identity': identity, 'id': identifier, 'entry': entry})
        manifest['files'][identifier] = {
            'name': name, 'sha256': digest(raw / 'entries' / name), 'length': len(entry['coords'])}
        if (index + 1) % 32 == 0:
            print(json.dumps({'event': 'frozen_progress', 'count': index + 1, 'total': 3072}), flush=True)
    if before != [tensor_digest(m) for m in (encoder, decoder)]:
        raise RuntimeError('Frozen production models changed')
    if any(p.grad is not None for m in (encoder, decoder) for p in m.parameters()):
        raise RuntimeError('Frozen production models received gradients')
    if code != sources():
        raise ValueError('Extension sources changed')
    manifest.update(train_ids=previous['train_ids'] + additions,
                    parent_bundle_sha256=previous['bundle_sha256'], extension_sources=code)
    manifest['bundle_sha256'] = identity_hash(manifest)
    atomic_json(raw / 'manifest.json', manifest)
    open_bundle(raw)
    atomic_json(root / 'preparation.json', {'status': 'complete', 'train': 5120, 'val': 256})
    del encoder, decoder, probe
    torch.cuda.empty_cache()
    return raw


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--previous', default='runs/se3_distance_data_scaling_20261022/raw_bundle')
    parser.add_argument('--exclude-bundle', default='runs/se3_clariden/bundle_4096')
    parser.add_argument('--dataset', default='foldtree2/notebooks/structs_training_mk2.h5')
    parser.add_argument('--device', choices=['cuda:0', 'cuda:1'], default='cuda:1')
    args = parser.parse_args()
    if Path(args.output).exists():
        parser.error('Extension requires a fresh output directory')
    torch.set_num_threads(2)
    torch.manual_seed(42)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        raw = prepare(args)
        run_scaling(SimpleNamespace(
            output=str(Path(args.output) / 'comparison'), source=str(raw),
            previous=args.previous, dataset=args.dataset, device=args.device,
            counts=[4096, 5120], epochs=128, seconds=28800.))
        atomic_json(Path(args.output) / 'completion.json', {'status': 'complete', 'counts': [4096, 5120]})
    except Exception as exc:
        if Path(args.output).is_dir():
            atomic_json(Path(args.output) / 'failure.json', {'error': repr(exc)})
        raise


if __name__ == '__main__':
    main()
