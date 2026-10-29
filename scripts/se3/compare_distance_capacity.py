"""Fixed-panel width comparison with matched-update and walltime landmarks."""

import os as _bootstrap_os, sys as _bootstrap_sys
_bootstrap_root = _bootstrap_os.path.dirname(_bootstrap_os.path.dirname(_bootstrap_os.path.dirname(_bootstrap_os.path.abspath(__file__))))
if _bootstrap_root not in _bootstrap_sys.path:
    _bootstrap_sys.path.insert(0, _bootstrap_root)
import argparse
import json
import math
import os
from pathlib import Path
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import torch
import torch.nn.functional as F

from foldtree2.se3_cache import FrozenDataset, digest, open_bundle
from foldtree2.src.distance_restraints import PairwiseDistanceHead, distance_labels, eligible_pairs
from scripts.se3.diagnose_distance_learning import describe, memorization_pass, sources as learning_sources
from scripts.se3.diagnose_se3_reconstruction import atomic_checkpoint, atomic_json


def sources():
    return {**learning_sources(), 'scripts/se3/compare_distance_capacity.py': digest(__file__)}


def train_width(records, manifest, width, root, args, event):
    output = root / f'width_{width}'
    output.mkdir()
    device = torch.device(args.device)
    torch.manual_seed(42)
    model = PairwiseDistanceHead(
        manifest['feature_dim'], manifest['decoder_enrichment']['contact_dim'],
        hidden=width).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=0.)
    data = []
    for entry, pairs in records:
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in entry.items()}
        pairs = pairs.to(device)
        labels, _ = distance_labels(inputs['target'], pairs, model.edges)
        data.append((inputs, pairs, labels))
    atomic_json(output / 'initial.json', describe(model, records, device))
    history, gate_step, best_step = [], None, 0
    best = float('inf')
    started = time.monotonic()
    next_landmark = 600.
    event('case_started', width=width, parameters=sum(p.numel() for p in model.parameters()))
    for step in range(1, args.steps + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        for inputs, pairs, labels in data:
            loss = F.cross_entropy(model(inputs, pairs), labels)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite capacity objective')
            (loss / len(data)).backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        elapsed = time.monotonic() - started
        landmark = elapsed >= next_landmark
        if step == 1 or step % args.log_every == 0 or landmark or step == args.steps or elapsed >= args.seconds:
            report = describe(model, records, device)
            elapsed = time.monotonic() - started
            if gate_step is None and memorization_pass(report):
                gate_step = step
            record = {'step': step, 'elapsed_seconds': elapsed,
                      'gradient_norm': float(norm), 'passed': memorization_pass(report), **report}
            history.append(record)
            atomic_json(output / 'history.json', history)
            checkpoint = {
                'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                'model_config': model.model_config(), 'report': report,
                'step': step, 'elapsed_seconds': elapsed,
                'ids': [e['id'] for e, _ in records],
                'bundle_sha256': manifest['bundle_sha256'], 'sources': sources(),
                'resumable': False}
            if report['nll'] < best:
                best, best_step = report['nll'], step
                atomic_checkpoint(output / 'best.pt', checkpoint)
            if landmark:
                atomic_checkpoint(output / f'walltime-{int(next_landmark):05d}.pt', checkpoint)
                while next_landmark <= elapsed:
                    next_landmark += 600.
            event('progress', width=width, step=step, elapsed_seconds=elapsed,
                  nll=report['nll'], accuracy=report['bin_accuracy'],
                  precision=report['contact_precision'], recall=report['contact_recall'],
                  contact_ap=report['contact_average_precision'])
        if elapsed >= args.seconds:
            break
    atomic_checkpoint(output / 'last.pt', checkpoint)
    saved = torch.load(output / 'best.pt', map_location=device, weights_only=False)
    model.load_state_dict(saved['model'])
    if describe(model, records, device) != saved['report']:
        raise ValueError('Capacity checkpoint reload mismatch')
    if any(not torch.isfinite(v).all() for state in optimizer.state.values()
           for v in state.values() if isinstance(v, torch.Tensor)):
        raise FloatingPointError('Nonfinite capacity optimizer state')
    summary = {
        'width': width, 'parameters': sum(p.numel() for p in model.parameters()),
        'updates': step, 'elapsed_seconds': time.monotonic() - started,
        'best_step': best_step, 'best': saved['report'], 'final': report,
        'first_gate_step': gate_step, 'reload_exact': True,
        'stop_reason': 'steps' if step == args.steps else 'time_limit'}
    atomic_json(output / 'completion.json', summary)
    event('case_complete', width=width, updates=step, best_nll=best)
    return summary


def run(args):
    if (args.steps < 1 or args.log_every < 1 or args.seconds <= 0
            or not math.isfinite(args.seconds) or not args.widths
            or len(set(args.widths)) != len(args.widths) or min(args.widths) < 1):
        raise ValueError('Invalid capacity experiment bounds')
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=False)

    def event(kind, **values):
        text = json.dumps({'event': kind, 'time': time.time(), **values}, allow_nan=False)
        with (root / 'events.jsonl').open('a') as stream:
            stream.write(text + '\n')
        print(text, flush=True)

    try:
        torch.set_num_threads(2)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        device = torch.device(args.device)
        if device.type == 'cuda':
            torch.cuda.set_device(device)
        panel = json.loads(Path(args.panel).read_text())
        manifest = open_bundle(args.cache)
        ids = panel['panel_ids']
        if (panel['bundle_sha256'] != manifest['bundle_sha256'] or len(ids) != 8
                or len(set(ids)) != 8 or not set(ids) <= set(manifest['train_ids'])
                or set(ids) & set(manifest['val_ids'])):
            raise ValueError('Capacity panel differs from original training panel')
        dataset = FrozenDataset(args.cache, manifest, 'train')
        records = []
        for identifier in ids:
            entry = dataset[dataset.ids.index(identifier)]
            records.append((entry, eligible_pairs(entry['mask'])))
        identity = sources()
        atomic_json(root / 'provenance.json', {
            'args': vars(args), 'sources': identity, 'panel_sha256': digest(args.panel),
            'ids': ids, 'bundle_sha256': manifest['bundle_sha256'],
            'torch': torch.__version__, 'seed': 42,
            'gpu': torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
            'protocol': 'distance-width-memorization-v1',
            'objective': 'equal-chain all-eligible-pair CE; no weight decay; AdamW1e-3 clip1',
            'selection': 'training NLL; record first acceptance without early stopping',
            'landmarks': 'every100 updates and first observation after each600 seconds',
            'scope': 'eight-chain training memorization, not held-out reconstruction'})
        for name in identity:
            path = root / 'sources' / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(Path(name).read_bytes())
        summaries = {}
        for width in args.widths:
            if identity != sources():
                raise ValueError('Capacity experiment source changed')
            summaries[str(width)] = train_width(records, manifest, width, root, args, event)
        if identity != sources() or open_bundle(args.cache)['bundle_sha256'] != manifest['bundle_sha256']:
            raise ValueError('Capacity source or cache changed')
        atomic_json(root / 'completion.json', summaries)
        event('complete')
    except Exception as exc:
        atomic_json(root / 'failure.json', {'error': repr(exc)})
        event('failure', error=repr(exc))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', default='runs/se3_decoder_enrichment_v3_20261021')
    parser.add_argument('--panel', default='runs/se3_distance_learning_long_20261021/provenance.json')
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', choices=['cpu', 'cuda:0', 'cuda:1'], default='cuda:1')
    parser.add_argument('--widths', type=int, nargs='+', default=[64, 128, 256])
    parser.add_argument('--steps', type=int, default=40000)
    parser.add_argument('--seconds', type=float, default=7200.)
    parser.add_argument('--log-every', type=int, default=100)
    run(parser.parse_args())


if __name__ == '__main__':
    main()
