"""Verify completion and numerical gates for the one-epoch normalization trial."""
import argparse
import json
import math
import statistics
from pathlib import Path
import torch
from foldtree2.se3_cache import open_bundle


def tensors(value):
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from tensors(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from tensors(item)


def assess(root, cache):
    root = Path(root)
    ck = torch.load(root / 'last.ckpt', map_location='cpu', weights_only=False)
    events = [json.loads(line) for line in (root / 'events.jsonl').read_text().splitlines()]
    config = ck['se3']['provenance']['training']
    manifest = open_bundle(cache)
    if manifest['bundle_sha256'] != ck['se3']['provenance']['bundle']:
        raise ValueError('Frozen bundle differs from checkpoint provenance')
    updates = [e for e in events if e['event'] == 'update']
    trains = [e for e in events if e['event'] == 'train_epoch']
    validations = [e for e in events if e['event'] == 'validation']
    expected_updates = config['train_count'] // config['global_batch']
    if not config.get('tensor_ff_input_norm') or config['epochs'] != 1:
        raise ValueError('Expected the explicit one-epoch normalization experiment')
    if ck['se3']['stream'] != {'epoch':1, 'cursor':0, 'updates':expected_updates}:
        raise ValueError('Incomplete training epoch')
    if [e['global_step'] for e in updates] != list(range(1, expected_updates + 1)):
        raise ValueError('Missing, duplicate or out-of-order optimizer updates')
    if len(trains) != 1 or trains[0]['effective_structures'] != config['train_count']:
        raise ValueError('Training structure count differs')
    if len(validations) != 1 or not validations[0]['complete_epoch']:
        raise ValueError('Expected exactly one complete-epoch validation')
    if list(root.glob('failure*')) or any(e['event'] == 'failure' for e in events):
        raise ValueError('Failure artifacts present')
    if not all(math.isfinite(e['loss']) and math.isfinite(e['gradient_norm'])
               and e['gradient_norm'] > 0 for e in updates):
        raise ValueError('Nonfinite losses, gradients or zero gradient update')
    states = list(tensors({'model':ck['state_dict'], 'optimizer':ck['optimizer_states']}))
    if any(t.dtype != torch.float32 for t in ck['state_dict'].values() if t.is_floating_point()):
        raise ValueError('Model parameters/buffers are not float32')
    if not all(bool(torch.isfinite(t).all()) for t in states if t.is_floating_point()):
        raise ValueError('Nonfinite model or optimizer state')
    rows = json.loads((root / 'validation-000.json').read_text())
    if len(rows) != config['val_count'] or len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Missing or duplicate validation structures')
    if {r['id'] for r in rows} != set(manifest['val_ids']):
        raise ValueError('Validation split membership differs')
    for row in rows:
        values = [row['geometry'], row['evaluation_fape']] + list(row['raw'].values()) + list(row['refined'].values())
        if not all(v is None or math.isfinite(v) for v in values):
            raise ValueError('Nonfinite structural metric')
    paired = {}
    for key in rows[0]['refined']:
        pairs = [(r['raw'][key], r['refined'][key]) for r in rows
                 if r['raw'][key] is not None and r['refined'][key] is not None]
        if pairs:
            paired[key] = {'raw':statistics.mean(x for x,y in pairs),
                           'refined':statistics.mean(y for x,y in pairs), 'count':len(pairs)}
    start = next(e['time'] for e in events if e['event'] == 'ready')
    elapsed = events[-1]['time'] - start
    return {'passed':True, 'updates':expected_updates, 'training_structures':config['train_count'],
            'validation_structures':len(rows), 'nonfinite_events':0,
            'training_geometry':trains[0]['geometry'], 'validation':validations[0]['metrics'],
            'paired_metrics':paired, 'elapsed_seconds':elapsed,
            'peak_gpu_memory_bytes':max(e['peak_memory_bytes'] for e in updates),
            'gradient_norm_min':min(e['gradient_norm'] for e in updates),
            'gradient_norm_max':max(e['gradient_norm'] for e in updates),
            'median_structures_per_second':statistics.median(e['structures_per_second'] for e in updates),
            'scope':'One local float32 epoch; does not establish long-run or distributed stability',
            'frozen_bundle_verified':True,
            'model_precision':'float32',
            'provenance':ck['se3']['provenance']}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('run'); p.add_argument('--output', required=True)
    p.add_argument('--cache', required=True)
    args = p.parse_args()
    result = assess(args.run, args.cache)
    Path(args.output).write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k:v for k,v in result.items() if k != 'provenance'}, allow_nan=False))
