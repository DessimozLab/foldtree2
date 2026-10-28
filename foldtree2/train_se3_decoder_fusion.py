"""Bounded serial decoder-fusion experiment, isolated from historical trainers."""
import argparse
import json
import os
from pathlib import Path
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import torch
import yaml

from foldtree2.se3_cache import FrozenDataset, digest, global_order, open_bundle
from foldtree2.se3_gradient_accumulation import GradientAccumulator
from foldtree2.src.decoder_feature_fusion import DecoderFusionRefiner, MODES
from foldtree2.src.losses.se3_balanced import BalancedGeometryLoss
from foldtree2.train_se3_lightning import calibrated, source_identity
from scripts.diagnose_se3_reconstruction import atomic_json, atomic_checkpoint, reconstruction_metrics


def fusion_sources():
    sources = {str(Path('foldtree2') / name): value for name, value in source_identity().items()}
    for name in ('foldtree2/src/decoder_feature_fusion.py', 'foldtree2/train_se3_decoder_fusion.py',
                 'scripts/diagnose_se3_reconstruction.py'):
        sources[name] = digest(name)
    return sources


def evaluate(model, entries, loss_fn):
    model.eval()
    rows = []
    with torch.no_grad():
        for entry in entries:
            pred, angles = model(entry)
            terms = loss_fn.components(pred, angles, entry)
            rows.append({
                'id': entry['id'], 'metrics': reconstruction_metrics(pred, angles, entry),
                'components': {k: float(v) if isinstance(v, torch.Tensor) else v
                               for k, v in terms.items()},
            })
    return rows


def mean_metrics(rows):
    return {key: sum(r['metrics'][key] for r in rows) / len(rows)
            for key in ('fape', 'tm_score', 'ca_rmsd', 'lddt', 'ca_bond_mae')}


def train(config, device):
    if config['mode'] not in MODES or config['epochs'] < 1:
        raise ValueError('Invalid fusion experiment configuration')
    root = Path(config['output'])
    root.mkdir(parents=True, exist_ok=False)

    def event(kind, **values):
        text = json.dumps({'event': kind, 'time': time.time(), **values}, allow_nan=False)
        with (root / 'events.jsonl').open('a') as stream:
            stream.write(text + '\n')
        print(text, flush=True)

    try:
        torch.set_num_threads(2)
        torch.manual_seed(42)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        if device.type == 'cuda':
            torch.cuda.set_device(device)
        manifest = open_bundle(config['cache'])
        if 'decoder_enrichment' not in manifest:
            raise ValueError('Fusion requires a versioned enriched cache')
        if len(manifest['train_ids']) != config['train_count'] or len(manifest['val_ids']) != config['val_count']:
            raise ValueError('Fusion split counts differ')
        if config['global_batch'] != 16 or len(manifest['train_ids']) % 16:
            raise ValueError('Matched fusion training requires complete global batches of 16')
        splits = {}
        baselines = {}
        for split in ('train', 'val'):
            dataset = FrozenDataset(config['cache'], manifest, split)
            entries = [calibrated(dataset[i], manifest['scale']) for i in range(len(dataset))]
            splits[split] = [
                {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in e.items()}
                for e in entries]
        baselines['val'] = [
            {'id': e['id'], 'metrics': reconstruction_metrics(e['coords'], None, e)}
            for e in splits['val']]
        atomic_json(root / 'input_baseline.json', baselines['val'])
        model = DecoderFusionRefiner(
            manifest['feature_dim'], manifest['decoder_enrichment']['contact_dim'],
            config['mode'], config['model']).to(device).float()
        optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=.01)
        accumulator = GradientAccumulator(model.parameters())
        loss_fn = BalancedGeometryLoss({'local_distance': 1., 'bend': .5, 'torsion': .5})
        provenance = {
            'protocol': 'se3-decoder-fusion-v1', 'config': config, 'sources': fusion_sources(),
            'bundle_sha256': manifest['bundle_sha256'], 'seed': 42,
            'train_ids': manifest['train_ids'], 'val_ids': manifest['val_ids'],
            'torch': torch.__version__, 'cuda': torch.version.cuda,
            'gpu': torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
            'scope': 'Reused development holdout; not independent of production pretraining',
            'parameters': sum(p.numel() for p in model.parameters()),
            'production_forward_calls_during_training': 0,
        }
        atomic_json(root / 'provenance.json', provenance)
        initial = evaluate(model, splits['val'], loss_fn)
        atomic_json(root / 'initial.json', initial)
        event('ready', mode=config['mode'], train=len(splits['train']), val=len(initial),
              initial_metrics=mean_metrics(initial))
        best = float('inf')
        updates = 0
        best_epoch = None
        started = time.monotonic()
        history = []
        lengths = [len(e['coords']) for e in splits['train']]
        for epoch in range(config['epochs']):
            model.train()
            losses, norms = [], []
            order = global_order(lengths, 42, epoch, 16)
            for group in order:
                optimizer.zero_grad(set_to_none=True)
                for index in group:
                    entry = splits['train'][index]
                    pred, angles = model(entry)
                    loss = loss_fn(pred, angles, entry)[0]
                    if not torch.isfinite(loss):
                        raise FloatingPointError('Nonfinite fusion training loss: ' + entry['id'])
                    loss.backward()
                    accumulator.collect()
                    losses.append(float(loss.detach()))
                accumulator.finish(16)
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                if norm <= 0:
                    raise RuntimeError('Zero fusion gradient')
                optimizer.step()
                updates += 1
                norms.append(float(norm))
                if updates % 8 == 0:
                    event('update', epoch=epoch + 1, update=updates, gradient_norm=float(norm))
            rows = evaluate(model, splits['val'], loss_fn)
            selection = sum(r['metrics']['fape'] for r in rows) / len(rows)
            atomic_json(root / f'validation-{epoch + 1:03d}.json', rows)
            if selection < best:
                best, best_epoch = selection, epoch + 1
            checkpoint = {
                'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                'epoch': epoch + 1, 'updates': updates, 'provenance': provenance,
                'rng': torch.get_rng_state(),
                'cuda_rng': torch.cuda.get_rng_state(device) if device.type == 'cuda' else None,
                'best_epoch': best_epoch, 'best_fape': best, 'resumable': False,
            }
            atomic_checkpoint(root / 'last.pt', checkpoint)
            if best_epoch == epoch + 1:
                atomic_checkpoint(root / 'best.pt', checkpoint)
                atomic_json(root / 'best_metrics.json', rows)
            record = {
                'epoch': epoch + 1, 'updates': updates, 'train_structures': len(losses),
                'val_structures': len(rows), 'train_objective': sum(losses) / len(losses),
                'validation': mean_metrics(rows), 'mean_gradient_norm': sum(norms) / len(norms),
                'gates': {name: float(p.detach().tanh()) for name, p in
                          model.fusion.named_parameters() if name.endswith('_gate')},
            }
            history.append(record)
            atomic_json(root / 'history.json', history)
            event('epoch_complete', **record)
        if provenance['sources'] != fusion_sources():
            raise ValueError('Fusion source changed during training')
        if not all(torch.isfinite(v).all() for v in model.state_dict().values()):
            raise FloatingPointError('Nonfinite final fusion model')
        for state in optimizer.state.values():
            if any(isinstance(v, torch.Tensor) and not torch.isfinite(v).all() for v in state.values()):
                raise FloatingPointError('Nonfinite final fusion optimizer')
        atomic_json(root / 'completion.json', {
            'complete': True, 'epochs': config['epochs'], 'updates': updates,
            'best_epoch': best_epoch, 'best_fape': best,
            'elapsed_seconds': time.monotonic() - started,
            'best_sha256': digest(root / 'best.pt'), 'last_sha256': digest(root / 'last.pt'),
            'finite_model_optimizer': True,
        })
        event('complete', updates=updates, best_epoch=best_epoch)
    except Exception as exc:
        event('failure', error=repr(exc))
        atomic_json(root / 'failure.json', {'error': repr(exc)})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--device', choices=['cpu', 'cuda:0', 'cuda:1'], required=True)
    args = parser.parse_args()
    train(yaml.safe_load(Path(args.config).read_text()), torch.device(args.device))


if __name__ == '__main__':
    main()