"""Extend the frozen distance cohort from 5120 to 10240 representatives.

Retains the existing 5120/256 train/holdout IDs in their original splits and
appends additional deterministically selected representatives from the
production dataset, excluding every ID already consumed by the 5120 bundle and
the Clariden 4096 representative panel.

Two eligibility tiers, filled in order:
- strict tier: 64-256 residues, >=90% valid (matches the existing cohort);
- long-chain tier: 257-512 residues, >=70% valid. Longer chains contribute far
  more nonlocal pairs per chain and add a length regime the current cohort
  (median 119 residues) barely covers, which is the point of diversity.
"""
import argparse
import copy
import hashlib
import inspect
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import torch

from foldtree2.se3_cache import digest, identity_hash, open_bundle
from foldtree2.se3_validation import (
    base, frozen_entry, load_production_decoder, production_pair, sha256, tensor_digest)
from scripts.diagnose_se3_reconstruction import atomic_checkpoint, atomic_json
from scripts.extend_distance_data import select_additions


def select_long_additions(dataset, excluded, count, seed=42):
    """Deterministically sample 257-512 residue chains with >=70% valid residues."""
    import random
    ids = sorted(set(dataset.structlist) - set(excluded))
    random.Random(seed).shuffle(ids)
    selected, skipped = [], {}
    for identifier in ids:
        group = dataset.h5dataset['structs'][identifier]['node']
        n = group['res']['x'].shape[0]
        if not 257 <= n <= 512:
            skipped['length'] = skipped.get('length', 0) + 1
            continue
        mask = torch.isfinite(torch.from_numpy(group['coords']['x'][:])).all(-1)
        if 'plddt' in group:
            confidence = torch.from_numpy(group['plddt']['x'][:]).reshape(-1)
            mask &= torch.isfinite(confidence) & (confidence >= .5)
        valid = mask.nonzero().flatten()
        if float(mask.float().mean()) < .7:
            skipped['confidence'] = skipped.get('confidence', 0) + 1
            continue
        if len(valid) < 2 or int(valid[-1] - valid[0]) < 12:
            skipped['no_nonlocal_pairs'] = skipped.get('no_nonlocal_pairs', 0) + 1
            continue
        selected.append(identifier)
        if len(selected) == count:
            return selected, skipped
    raise ValueError(f'Insufficient eligible long-chain additions: {len(selected)} of {count}')


def sources():
    from scripts.extend_distance_data import sources as extend_sources
    return {**extend_sources(), 'scripts/extend_distance_data_10240.py': digest(__file__)}


def prepare(args):
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=False)
    source = Path(args.previous)
    previous = open_bundle(source)
    exclusion = open_bundle(args.exclude_bundle)
    if len(previous['train_ids']) != 5120 or len(previous['val_ids']) != 256:
        raise ValueError('Expected preceding 5120/256 enriched cohort')
    if 'decoder_enrichment' not in previous:
        raise ValueError('Extension requires the enriched 5120 cohort')
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

    def scan_tier(predicate, name):
        """Collect every eligible ID of one tier, then draw count deterministically."""
        eligible = []
        for identifier in sorted(set(dataset.structlist) - excluded):
            group = dataset.h5dataset['structs'][identifier]['node']
            n = group['res']['x'].shape[0]
            if not predicate(n):
                continue
            mask = torch.isfinite(torch.from_numpy(group['coords']['x'][:])).all(-1)
            if 'plddt' in group:
                confidence = torch.from_numpy(group['plddt']['x'][:]).reshape(-1)
                mask &= torch.isfinite(confidence) & (confidence >= .5)
            valid = mask.nonzero().flatten()
            if float(mask.float().mean()) < .7 or len(valid) < 2 or int(valid[-1] - valid[0]) < 12:
                continue
            eligible.append(identifier)
        import random
        random.Random(42).shuffle(eligible)
        print(json.dumps({'event': 'tier_scanned', 'tier': name, 'eligible': len(eligible)}), flush=True)
        return eligible

    strict_pool = scan_tier(lambda n: 64 <= n <= 256, 'short_64_256_ge70')
    long_pool = scan_tier(lambda n: 257 <= n <= 512, 'long_257_512_ge70')
    short_kept = min(args.additions, len(strict_pool))
    long_kept = min(args.additions - short_kept, len(long_pool))
    if short_kept + long_kept < args.additions:
        raise ValueError(f'Insufficient eligible additions: {short_kept + long_kept} of {args.additions}')
    additions = strict_pool[:short_kept] + long_pool[:long_kept]
    skipped = {'tier_short_eligible': len(strict_pool), 'tier_long_eligible': len(long_pool)}
    atomic_json(root / 'selection.json', {
        'retained_train_ids': previous['train_ids'], 'additional_train_ids': additions,
        'val_ids': previous['val_ids'], 'excluded_previous_ids': sorted(excluded),
        'skipped_counts': skipped, 'sources': code,
        'eligibility': 'tier1: 64-256 residues >=70% valid; tier2: 257-512 residues '
                       '>=70% valid; both need a nonlocal pair; tier1 exhausted first; seed42'})
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
    accepted = []
    rejected_extracted = []
    for index, identifier in enumerate(additions):
        entry = frozen_entry(dataset[identifier], encoder, decoder, device)
        # The extraction pipeline can mask residues beyond the raw-coordinate scan
        # (decoder target finiteness). Accept only chains whose extracted mask
        # keeps >=70% valid residues; reject rather than store degraded entries.
        if float(entry['mask'].float().mean()) < .7:
            rejected_extracted.append(identifier)
            continue
        if index == 0 or not accepted:
            repeated = frozen_entry(dataset[identifier], encoder, decoder, device)
            for key, value in entry.items():
                if isinstance(value, torch.Tensor):
                    # GPU float kernels jitter between identical calls (~1e-7 for
                    # coordinates, up to ~2e-4 rad for atan2-derived angles). The
                    # check guards against gross nondeterminism, not ULP noise.
                    torch.testing.assert_close(value, repeated[key], atol=5e-3, rtol=5e-3,
                                               equal_nan=True)
        name = hashlib.sha256(identifier.encode()).hexdigest() + '.pt'
        atomic_checkpoint(raw / 'entries' / name, {'identity': identity, 'id': identifier, 'entry': entry})
        manifest['files'][identifier] = {
            'name': name, 'sha256': digest(raw / 'entries' / name), 'length': len(entry['coords'])}
        accepted.append(identifier)
        if len(accepted) % 64 == 0:
            print(json.dumps({'event': 'frozen_progress', 'count': len(accepted),
                              'total': len(additions)}), flush=True)
    additions = accepted
    print(json.dumps({'event': 'freeze_complete', 'accepted': len(accepted),
                      'rejected_extracted_mask': len(rejected_extracted)}), flush=True)
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
    atomic_json(root / 'preparation.json', {'status': 'complete',
                                            'train': 5120 + args.additions, 'val': 256})
    del encoder, decoder, probe
    torch.cuda.empty_cache()
    return raw


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--previous', default='runs/se3_distance_data_scaling_5120_20261022/comparison/bundle_5120')
    parser.add_argument('--exclude-bundle', default='runs/se3_clariden/bundle_4096')
    parser.add_argument('--dataset', default='foldtree2/notebooks/structs_training_mk2.h5')
    parser.add_argument('--device', choices=['cuda:0', 'cuda:1'], default='cuda:1')
    parser.add_argument('--additions', type=int, default=5120,
                        help='Number of additional train representatives to freeze')
    args = parser.parse_args()
    prepare(args)


if __name__ == '__main__':
    main()
