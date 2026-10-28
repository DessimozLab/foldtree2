"""Prepare and run matched local conditioning trials; no automatic expansion."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import torch
import yaml

from foldtree2.se3_cache import FrozenDataset, digest, identity_hash, open_bundle
from foldtree2.train_se3_lightning import build_model, calibrated, source_identity
from foldtree2.se3_validation import contact_inputs, forward
from foldtree2.src.se3_struct_decoder import enable_tensor_ff_input_norm
from scripts.diagnose_se3_reconstruction import atomic_json, select_panel


VARIANTS = (
    ('batch_normal', 'batch', 'normal'),
    ('structure_normal', 'per_structure', 'normal'),
    ('batch_local', 'batch', 'local-only'),
    ('structure_local', 'per_structure', 'local-only'),
    ('structure_inferred', 'per_structure', 'inferred-only'),
    ('structure_seed', 'per_structure', 'seed-only'),
    ('layer_normal', 'layer', 'normal'),
    ('layer_local', 'layer', 'local-only'),
)


def verify_normalization(checkpoint_path, cache, device, output):
    """Diagnostic-only transfer of learned affine weights; never a warm start."""
    manifest = open_bundle(cache)
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    if checkpoint['provenance']['bundle_sha256'] != manifest['bundle_sha256']:
        raise ValueError('Normalization diagnostic checkpoint/cache mismatch')
    architecture = checkpoint['provenance']['architecture']
    original = enable_tensor_ff_input_norm(build_model(manifest['feature_dim'], architecture)).to(device)
    candidate = enable_tensor_ff_input_norm(build_model(
        manifest['feature_dim'], {**architecture, 'input_normalization': 'per_structure'})).to(device)
    original.load_state_dict(checkpoint['model'])
    omitted = {'bn.running_mean', 'bn.running_var', 'bn.num_batches_tracked'}
    candidate.load_state_dict({k: v for k, v in checkpoint['model'].items() if k not in omitted})
    dataset = FrozenDataset(cache, manifest, 'train')
    rows = []
    for identifier in checkpoint['provenance']['ids']:
        original.load_state_dict(checkpoint['model'])
        e = calibrated(dataset[dataset.ids.index(identifier)], manifest['scale'])
        e = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in e.items()}
        with torch.no_grad():
            original.eval()
            old_eval, _ = forward(original, e)
            original.train()
            old_train, old_angles = forward(original, e)
            candidate.train()
            new_train, new_angles = forward(candidate, e)
            candidate.eval()
            new_eval, eval_angles = forward(candidate, e)
        torch.testing.assert_close(new_train, old_train, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(new_angles, old_angles, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(new_eval, new_train, atol=0, rtol=0)
        torch.testing.assert_close(eval_angles, new_angles, atol=0, rtol=0)
        pred, angles = forward(candidate, e)
        (pred.square().mean() + angles.square().mean()).backward()
        gradients = [p.grad for p in candidate.parameters() if p.grad is not None]
        if not gradients or not all(torch.isfinite(g).all() for g in gradients):
            raise FloatingPointError('Normalization gate has nonfinite or missing gradients')
        candidate.zero_grad(set_to_none=True)
        rows.append({
            'id': identifier,
            'original_mode_max_coordinate_difference': float((old_eval - old_train).abs().max()),
            'candidate_mode_max_coordinate_difference': float((new_eval - new_train).abs().max()),
            'candidate_vs_original_train_max_difference': float((new_train - old_train).abs().max()),
        })
    atomic_json(Path(output), {
        'checkpoint_sha256': digest(checkpoint_path), 'bundle_sha256': manifest['bundle_sha256'],
        'sources': source_identity(), 'device': str(device), 'rows': rows,
        'finite_gradients': True, 'optimizer_steps': 0,
        'omitted_running_buffers': sorted(omitted),
        'scope': 'Diagnostic affine-weight transfer only; not a migrated production checkpoint',
    })


def contact_audit(entry):
    idx = torch.arange(len(entry['mask']))
    eligible = (idx[None] - idx[:, None] >= 12)
    eligible &= entry['mask'][:, None] & entry['mask'][None]
    target = torch.cdist(entry['target'].masked_fill(~entry['mask'][:, None], 0.),
                         entry['target'].masked_fill(~entry['mask'][:, None], 0.)) < 8.
    distance = torch.cdist(entry['coords'], entry['coords'])
    rows = {}
    for _, _, policy in VARIANTS:
        options = contact_inputs(entry, policy)
        predicted = options['dot_prod'][0].bool().clone()
        if options['use_distance_contacts']:
            predicted |= (distance < 8.) & (distance > 0.)
        predicted &= eligible
        truth = target & eligible
        shared = int((predicted & truth).sum())
        npred, ntrue = int(predicted.sum()), int(truth.sum())
        rows[policy] = {
            'predicted': npred, 'target': ntrue, 'true_positive': shared,
            'eligible': int(eligible.sum()),
            'precision': shared / npred if npred else None,
            'recall': shared / ntrue if ntrue else None,
        }
    return {'id': entry['id'], 'policies': rows}


def prepare(source, root, train_count, val_count, epochs):
    if train_count < 16 or train_count % 16 or val_count < 1 or epochs < 1:
        raise ValueError('Positive counts/epochs required; training count must be divisible by 16')
    manifest = open_bundle(source)
    root.mkdir(parents=True, exist_ok=False)
    bundle = root / 'bundle'
    (bundle / 'entries').mkdir(parents=True)
    selected = {}
    selection = {}
    for split, count in [('train', train_count), ('val', val_count)]:
        dataset = FrozenDataset(source, manifest, split)
        candidates, excluded = [], []
        for index, identifier in enumerate(dataset.ids):
            length = manifest['files'][identifier]['length']
            if 64 <= length <= 128:
                entry = dataset[index]
                if float(entry['mask'].float().mean()) >= .9:
                    candidates.append(identifier)
                else:
                    excluded.append(identifier)
        ids = select_panel({**manifest, 'train_ids': candidates, 'val_ids': []}, count)
        selected[split] = ids
        selection[split] = {'ids': ids, 'eligible': len(candidates),
                            'excluded_low_confidence': excluded,
                            'unselected_ids': [i for i in candidates if i not in set(ids)]}
    ids = selected['train'] + selected['val']
    if len(set(ids)) != len(ids) or set(manifest['calibration_ids']) & set(selected['val']):
        raise ValueError('Training, validation or calibration overlap')
    subset = {**manifest, 'train_ids': selected['train'], 'val_ids': selected['val'],
              'files': {i: manifest['files'][i] for i in ids},
              'parent_bundle_sha256': manifest['bundle_sha256'],
              'scope': 'Matched short-chain development pilot; no independent test claim'}
    subset.pop('bundle_sha256')
    subset['bundle_sha256'] = identity_hash(subset)
    for identifier in ids:
        name = subset['files'][identifier]['name']
        shutil.copyfile(Path(source) / 'entries' / name, bundle / 'entries' / name)
    atomic_json(bundle / 'manifest.json', subset)
    open_bundle(bundle)
    atomic_json(root / 'selection.json', selection)
    audit_dataset = FrozenDataset(bundle, subset, 'train')
    atomic_json(root / 'training_contact_audit.json', [
        contact_audit(calibrated(audit_dataset[i], subset['scale']))
        for i in range(min(16, len(audit_dataset)))])
    configs = {}
    for name, normalization, policy in VARIANTS:
        config = {
            'accelerator': 'cuda', 'nodes': 1, 'devices': 1,
            'cache': str(bundle.resolve()), 'output': str((root / name).resolve()),
            'seed': 42, 'global_batch': 16, 'max_residues': 256,
            'train_count': train_count, 'val_count': val_count, 'epochs': epochs,
            'learning_rate': .0005, 'scheduler': False, 'min_delta': 0.,
            'checkpoint_every': 16, 'checkpoint_metric': 'evaluation_fape',
            'tensor_ff_input_norm': True, 'gradient_accumulation_dtype': 'float64',
            'structural_metrics': True, 'input_baseline_metrics': True,
            'contact_policy': policy,
            'model': {'hidden': 64, 'depth': 4, 'heads': 4,
                      'input_normalization': normalization},
            'loss': {'protocol': 'balanced_geometry_v1',
                     'weights': {'local_distance': 1., 'bend': .5, 'torsion': .5}},
        }
        path = root / (name + '.yaml')
        path.write_text(yaml.safe_dump(config))
        configs[name] = digest(path)
    sources = {str(Path('foldtree2') / n): h for n, h in source_identity().items()}
    sources['scripts/run_se3_generalization.py'] = digest(__file__)
    sources['scripts/diagnose_se3_reconstruction.py'] = digest('scripts/diagnose_se3_reconstruction.py')
    for filename in sources:
        target = root / 'sources' / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(Path(filename).read_bytes())
    atomic_json(root / 'experiment.json', {
        'protocol': 'se3-conditioning-generalization-v1', 'source_bundle': manifest['bundle_sha256'],
        'bundle': subset['bundle_sha256'], 'configs': configs, 'sources': sources,
        'training_visits_per_case': train_count * epochs,
        'updates_per_case': train_count * epochs // 16,
        'timeout_seconds_per_case': 2700,
        'scope': 'Previously monitored development holdout; not a cluster-separated test',
    })


def verify_completion(root, config):
    expected_ids = set(open_bundle(config['cache'])['val_ids'])
    checkpoint = torch.load(root / 'last.ckpt', map_location='cpu', weights_only=False)
    stream = checkpoint['se3']['stream']
    expected = config['epochs'] * config['train_count'] // config['global_batch']
    if stream != {'epoch': config['epochs'], 'cursor': 0, 'updates': expected}:
        raise ValueError('Incomplete matched training stream')
    for name in ('state_dict', 'optimizer_states'):
        def finite(value):
            if isinstance(value, torch.Tensor):
                return bool(torch.isfinite(value).all())
            if isinstance(value, dict):
                return all(finite(v) for v in value.values())
            if isinstance(value, (list, tuple)):
                return all(finite(v) for v in value)
            return True
        if not finite(checkpoint[name]):
            raise FloatingPointError('Nonfinite final ' + name)
    events = [json.loads(line) for line in (root / 'events.jsonl').read_text().splitlines()]
    if any(e['event'] == 'failure' for e in events) or list(root.glob('failure-*.json')):
        raise ValueError('Training recorded a failure')
    validations = [e for e in events if e['event'] == 'validation']
    training = [e for e in events if e['event'] == 'train_epoch']
    if len(validations) != config['epochs'] or len(training) != config['epochs']:
        raise ValueError('Incomplete epoch history')
    if any(e['effective_structures'] != config['train_count'] for e in training):
        raise ValueError('Training sample count mismatch')
    for epoch in range(config['epochs']):
        rows = json.loads((root / f'validation-{epoch:03d}.json').read_text())
        if len(rows) != config['val_count'] or {r['id'] for r in rows} != expected_ids:
            raise ValueError('Validation sample count mismatch')
    best_epoch = min(range(len(validations)),
                     key=lambda i: validations[i]['metrics']['evaluation_fape'])
    best = torch.load(root / 'best.ckpt', map_location='cpu', weights_only=False)
    if best['se3']['stream']['epoch'] != best_epoch + 1:
        raise ValueError('Best checkpoint does not match FAPE-selected epoch')
    atomic_json(root / 'completion.json', {
        'stream': stream, 'best_epoch': best_epoch + 1,
        'best_metrics': validations[best_epoch]['metrics'],
        'last_sha256': digest(root / 'last.ckpt'),
        'best_sha256': digest(root / 'best.ckpt'),
        'finite_model_optimizer': True, 'complete': True,
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default='runs/se3_generalization_20261021')
    parser.add_argument('--cache', default='runs/se3_clariden/bundle_4096')
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--train-count', type=int, default=256)
    parser.add_argument('--val-count', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=16)
    parser.add_argument('--device', choices=['0', '1'], default='1')
    parser.add_argument('--verify-checkpoint')
    args = parser.parse_args()
    torch.set_num_threads(2)
    root = Path(args.root)
    if args.verify_checkpoint:
        root.mkdir(parents=True, exist_ok=True)
        verify_normalization(args.verify_checkpoint, args.cache,
                             torch.device('cuda:' + args.device), root / 'normalization_gate.json')
        print(json.dumps({'event': 'normalization_gate_passed'}), flush=True)
        return
    if args.prepare_only:
        prepare(args.cache, root, args.train_count, args.val_count, args.epochs)
        print(json.dumps({'event': 'prepared', 'root': str(root)}), flush=True)
        return
    try:
        experiment = json.loads((root / 'experiment.json').read_text())
        for name, _, _ in VARIANTS:
            for source, expected in experiment['sources'].items():
                if digest(source) != expected:
                    raise ValueError('Staged source changed: ' + source)
            path = root / (name + '.yaml')
            if digest(path) != experiment['configs'][name]:
                raise ValueError('Experiment configuration changed: ' + name)
            config = yaml.safe_load(path.read_text())
            manifest = open_bundle(config['cache'])
            if manifest['bundle_sha256'] != experiment['bundle']:
                raise ValueError('Experiment bundle changed')
            environment = {**os.environ, 'CUDA_VISIBLE_DEVICES': args.device,
                           'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2'}
            print(json.dumps({'event': 'launch', 'case': name, 'time': time.time()}), flush=True)
            with (root / (name + '.log')).open('x') as log:
                subprocess.run(
                    [sys.executable, '-u', '-m', 'foldtree2.train_se3_lightning',
                     '--config', str(path)], env=environment,
                    stdout=log, stderr=subprocess.STDOUT, check=True, timeout=2700)
            verify_completion(Path(config['output']), config)
            print(json.dumps({'event': 'case_complete', 'case': name, 'time': time.time()}), flush=True)
        atomic_json(root / 'queue_complete.json', {'cases': [v[0] for v in VARIANTS]})
    except Exception as exc:
        atomic_json(root / 'queue_failure.json', {'error': repr(exc), 'time': time.time()})
        raise


if __name__ == '__main__':
    main()
