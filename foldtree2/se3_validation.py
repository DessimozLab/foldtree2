"""Bounded residue-level production refiner experiments, cache and paired evaluation.

Run with python -m foldtree2.se3_validation --config configs/se3_validation_smoke.yaml.
"""
from __future__ import annotations
import argparse
import hashlib
import inspect
import resource
import json
import math
import os
from pathlib import Path
import platform
import random
import time
from types import SimpleNamespace

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch_geometric.data import HeteroData
from foldtree2 import learn_geometry_lightning as base
from foldtree2.learn_production_geometry_se3_lightning import load_production_decoder
from foldtree2.src.se3_struct_decoder import se3_denoiser
from foldtree2.src.losses.fape import equivariant_ca_frame_rotmat


def sha256(path):
    # Reuse a full-file digest only while inode, size, mtime and ctime agree.
    path = Path(path)
    stat = path.stat()
    signature = [str(path.resolve()), stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns]
    record = Path('runs/se3_validation/hashes') / (hashlib.sha256(str(path.resolve()).encode()).hexdigest() + '.json')
    if record.exists():
        saved = json.loads(record.read_text())
        if saved['signature'] == signature:
            return saved['sha256']
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    if signature != [str(path.resolve()), path.stat().st_dev, path.stat().st_ino, path.stat().st_size, path.stat().st_mtime_ns, path.stat().st_ctime_ns]:
        raise RuntimeError('File changed while hashing')
    record.parent.mkdir(parents=True, exist_ok=True)
    record.write_text(json.dumps({'signature': signature, 'sha256': h.hexdigest()}))
    return h.hexdigest()


def production_pair(manifest='configs/production_models.yaml'):
    entry, = [m for m in yaml.safe_load(Path(manifest).read_text())['models'] if m['size'] == 40]
    return {key: str(Path(entry['directory']) / entry[key]) for key in ('encoder', 'decoder')}


def tensor_digest(model):
    h = hashlib.sha256()
    for key, v in model.state_dict().items():
        h.update(key.encode())
        h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def split_structures(dataset, train_count, val_count, max_residues, seed):
    """Select complete chains by metadata; oversized chains are never truncated."""
    ids = list(dataset.structlist)
    random.Random(seed).shuffle(ids)
    selected, skipped = [], []
    for identifier in ids:
        group = dataset.h5dataset['structs'][identifier]
        n = group['node']['res']['x'].shape[0]
        if n < 3 or n > max_residues:
            skipped.append({'id': identifier, 'length': n, 'reason': 'length'})
            continue
        target = torch.from_numpy(group['node']['coords']['x'][:]).float()
        mask = torch.isfinite(target).all(-1)
        if 'plddt' in group['node']:
            confidence = torch.from_numpy(group['node']['plddt']['x'][:]).float().reshape(-1)
            mask &= torch.isfinite(confidence) & (confidence >= .5)
        if mask.sum() < 3:
            skipped.append({'id': identifier, 'length': n, 'reason': 'fewer_than_three_valid_residues'})
            continue
        try:
            frame_fape(target.masked_fill(~mask[:, None], 0.), target, mask)
        except ValueError:
            skipped.append({'id': identifier, 'length': n, 'reason': 'no_valid_defined_frames'})
            continue
        angle_values = torch.from_numpy(group['node']['bondangles']['x'][:])
        if not (torch.isfinite(angle_values) & mask[:, None]).any():
            skipped.append({'id': identifier, 'length': n, 'reason': 'no_valid_angles'})
            continue
        selected.append(identifier)
        if len(selected) == train_count + val_count:
            break
    if len(selected) != train_count + val_count:
        raise ValueError('Insufficient structures within the residue bound')
    return selected[:train_count], selected[train_count:], skipped


def length_bucket_batches(lengths, batch_size, seed):
    """Shuffle nearby-length batches without padding short chains to global maximum."""
    indices = sorted(range(len(lengths)), key=lambda i: lengths[i])
    batches = [indices[i:i + batch_size] for i in range(0, len(indices), batch_size)]
    random.Random(seed).shuffle(batches)
    return batches


def frozen_entry(graph, encoder, decoder, device):
    graph = base.ensure_edge_attrs_inplace(base.ensure_float32_inplace(graph.clone().to(device)))
    with torch.no_grad():
        z, _ = encoder(graph)
        tokens, _ = encoder.vector_quantizer.discretize_z(z)
        tokens = tokens.long().reshape(-1)
        codebook = base.GeometryFocusedModule._get_codebook_vectors(encoder, tokens)
        graph['res'].x = z
        capture = []
        hook = decoder.body['lin'].register_forward_pre_hook(lambda _m, inputs: capture.append(inputs[0].detach()))
        try:
            out = decoder(graph, contact_pred_index=None)
        finally:
            hook.remove()
        if len(capture) != 1 or capture[0].shape != (z.shape[0], 3):
            raise RuntimeError('Production bottleneck must be [N,3]')
        coords = capture[0]
        contacts = F.normalize(out['z'].float(), dim=-1)
        scores = contacts @ contacts.T
        n = len(z)
        separation = (torch.arange(n, device=device)[:, None] - torch.arange(n, device=device)[None, :]).abs()
        eligible = separation >= 3
        scores = scores.masked_fill(~eligible, -torch.inf)
        dot = torch.zeros(n, n, device=device, dtype=torch.bool)
        k = min(8, n)
        indices = scores.topk(k, dim=-1).indices
        dot.scatter_(1, indices, True)
        dot &= eligible
        dot |= dot.T.clone()
        dot |= (separation <= 1) & (separation > 0)
        angles = base.maybe_deg_to_rad_torch(base.node_x(graph, 'bondangles'))
        target = base.node_x(graph, 'coords')
        if target is None or target.shape != coords.shape:
            raise ValueError('Expected residue CA targets [N,3]')
        mask = torch.isfinite(target).all(-1)
        # Use the identical residue mask for baseline and refined metrics.
        plddt = base.node_x(graph, 'plddt')
        if plddt is not None:
            mask &= torch.isfinite(plddt.reshape(-1)) & (plddt.reshape(-1) >= .5)
        entry = {'features': torch.cat([z, codebook], -1), 'latent': z, 'codebook': codebook,
                 'tokens': tokens, 'coords': coords, 'contact_z': contacts, 'dot': dot,
                 'target': target, 'angles': angles, 'mask': mask,
                 'baseline_angles': out.get('angles')}
        for key in ['features', 'coords', 'contact_z']:
            if not torch.isfinite(entry[key]).all():
                raise ValueError(f'Nonfinite frozen {key}')
    return {k: v.detach().cpu() if isinstance(v, torch.Tensor) else v for k, v in entry.items()}


def frame_fape(pred, target, mask):
    # Mask frame stencils, not just frame centers; gaps must not define a frame.
    safe_target = target.masked_fill(~mask[:, None], 0.)
    pr, pu = equivariant_ca_frame_rotmat(pred)
    tr, tu = equivariant_ca_frame_rotmat(safe_target)
    frame_mask = mask.clone()
    frame_mask[1:] &= mask[:-1]
    frame_mask[:-1] &= mask[1:]
    if len(mask) > 2:
        frame_mask[0] &= mask[2]
        frame_mask[-1] &= mask[-3]
    frame_mask &= ~pu & ~tu
    if not frame_mask.any() or not mask.any():
        raise ValueError('No valid defined frames for FAPE')
    pd = pred[mask][None] - pred[frame_mask][:, None]
    td = safe_target[mask][None] - safe_target[frame_mask][:, None]
    pl = torch.einsum('fji,fpj->fpi', pr[frame_mask], pd)
    tl = torch.einsum('fji,fpj->fpi', tr[frame_mask], td)
    return (pl - tl).norm(dim=-1).clamp(max=10.).mean() / 10.


def losses(pred, angles, entry):
    mask = entry['mask']
    if mask.sum() < 3:
        raise ValueError('Fewer than three valid residues')
    fape = frame_fape(pred, entry['target'], mask)
    pd = torch.cdist(pred[mask], pred[mask])
    td = torch.cdist(entry['target'][mask], entry['target'][mask])
    geometry = F.smooth_l1_loss(pd, td) / 10. + fape
    target_angles = entry['angles'][..., :3]
    amask = torch.isfinite(target_angles) & mask[:, None]
    if not amask.any():
        raise ValueError('No valid target angles')
    angle = (1 - torch.cos(angles[..., :3][amask] - target_angles[amask])).mean()
    return geometry + .1 * angle, geometry, fape, angle


def forward(model, entry):
    g = HeteroData()
    g['res'].x = entry['features']
    out = model(g, coords_pred=entry['coords'], ft2_token_ids=entry['tokens'],
                edge_attr_dict={'dot_prod': entry['dot'][None], 'node_mask': entry['mask'],
                                'use_distance_contacts': True, 'distance_contact_cutoff': 8.})
    return out['coors_out_flat'], out['angles']


def ca_angle_errors(pred, target, mask):
    def angles(c):
        bonds = c[1:] - c[:-1]
        unit = F.normalize(bonds, dim=-1, eps=1e-8)
        bend = torch.acos((-unit[:-1] * unit[1:]).sum(-1).clamp(-1., 1.))
        b0, b1, b2 = -bonds[:-2], unit[1:-1], bonds[2:]
        v = b0 - (b0*b1).sum(-1,keepdim=True)*b1
        w = b2 - (b2*b1).sum(-1,keepdim=True)*b1
        torsion = torch.atan2((torch.cross(b1,v,dim=-1)*w).sum(-1), (v*w).sum(-1))
        return bend, torsion
    pa, ta = angles(pred), angles(target.masked_fill(~mask[:,None],0.))
    bm = mask[:-2] & mask[1:-1] & mask[2:]
    tm = mask[:-3] & mask[1:-2] & mask[2:-1] & mask[3:]
    result = {}
    for key, a, b, valid in zip(['ca_bend_mae_radians','ca_torsion_mae_radians'], pa, ta, [bm,tm]):
        diff = a[valid]-b[valid]
        result[key] = float(torch.atan2(diff.sin(),diff.cos()).abs().mean()) if valid.any() else None
    return result


def metrics(pred, angles, entry):
    mask = entry['mask']
    p, t = pred[mask], entry['target'][mask]
    pc, tc = p - p.mean(0), t - t.mean(0)
    u, _, vh = torch.linalg.svd(pc.T @ tc)
    sign = torch.ones(3, device=p.device)
    sign[-1] = torch.linalg.det(u @ vh)
    aligned = pc @ u @ torch.diag(sign) @ vh
    err = (aligned - tc).norm(dim=-1)
    pd, td = torch.cdist(p, p), torch.cdist(t, t)
    neighbors = (td < 15.) & (td > 0.)
    delta = (pd - td).abs()
    if not neighbors.any():
        raise ValueError('No lDDT pairs')
    lddt = torch.stack([(delta[neighbors] < cutoff).float().mean() for cutoff in (.5, 1., 2., 4.)]).mean()
    bond_mask = mask[1:] & mask[:-1]
    pb = (pred[1:] - pred[:-1]).norm(dim=-1)[bond_mask]
    tb = (entry['target'][1:] - entry['target'][:-1]).norm(dim=-1)[bond_mask]
    amask = torch.isfinite(entry['angles'][..., :3]) & mask[:, None]
    angle_error = None
    if angles is not None and amask.any():
        diff = angles[..., :3][amask] - entry['angles'][..., :3][amask]
        angle_error = float(torch.atan2(diff.sin(), diff.cos()).abs().mean())
    # tmtools performs TM-score optimized alignment, not a Kabsch approximation.
    from tmtools import tm_align
    result = tm_align(p.detach().cpu().double().numpy(), t.detach().cpu().double().numpy(),
                      'A' * len(p), 'A' * len(t))
    return {**ca_angle_errors(pred, entry['target'], mask), 'fape': float(frame_fape(pred, entry['target'], mask)),
            'ca_rmsd': float(err.square().mean().sqrt()), 'tm_score': result.tm_norm_chain2,
            'lddt': float(lddt), 'ca_bond_mae': float((pb - tb).abs().mean()),
            'ca_bond_deviation_3_8': float((pb - 3.8).abs().mean()),
            'angle_mae_radians': angle_error, 'valid_residues': int(mask.sum())}


def restore_training_state(checkpoint, provenance, model, optimizer, scheduler, device):
    if checkpoint['provenance'] != provenance:
        raise ValueError('Resume rejected: model, dataset, split or preprocessing provenance differs')
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])
    torch.set_rng_state(checkpoint['torch_rng'].cpu())
    if device.type == 'cuda':
        torch.cuda.set_rng_state(checkpoint['cuda_rng'].cpu(), device)
    return checkpoint['epoch'] + 1, checkpoint['global_step'], checkpoint['best'], checkpoint['bad']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--resume-from')
    parser.add_argument('--cache-only', action='store_true')
    args = parser.parse_args()
    try:
        run(args)
    except Exception as exc:
        # Persist startup/selection/cache failures as well as training failures.
        try:
            output = Path(yaml.safe_load(Path(args.config).read_text())['output'])
            output.mkdir(parents=True, exist_ok=True)
            event_path = output / 'events.jsonl'
            previous = event_path.read_text().splitlines() if event_path.exists() else []
            if not previous or json.loads(previous[-1]).get('event') != 'failure':
                with event_path.open('a') as f:
                    f.write(json.dumps({'event': 'failure', 'time': time.time(), 'error': repr(exc)}) + '\n')
        except (KeyError, OSError, ValueError):
            pass
        raise


def run(args):
    config = yaml.safe_load(Path(args.config).read_text())
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    base.pl.seed_everything(config['seed'], workers=True)
    device = torch.device(config.get('device', 'cuda:0'))
    output = Path(config['output']); output.mkdir(parents=True, exist_ok=True)
    def event(kind, **values):
        row = {'event': kind, 'time': time.time(), **values}
        with (output / 'events.jsonl').open('a') as f:
            f.write(json.dumps(row) + '\n')
        print(json.dumps(row), flush=True)
    pair = production_pair(config.get('manifest', 'configs/production_models.yaml'))
    event('hashing_dataset', path=config['dataset'])
    dataset_hash = sha256(config['dataset'])
    dataset = base.pdbgraphmk2.StructureDataset(config['dataset'])
    train_ids, val_ids, skipped = split_structures(dataset, config['train_count'], config['val_count'],
                                                 config['max_residues'], config['seed'])
    if config.get('overfit', False):
        val_ids = list(train_ids)
    identity = {'format': 1, 'dataset_sha256': dataset_hash,
                'encoder_sha256': sha256(pair['encoder']), 'decoder_sha256': sha256(pair['decoder']),
                'preprocessing': {'extractor_sha256': hashlib.sha256(inspect.getsource(frozen_entry).encode()).hexdigest(), 'coordinate_source': 'bottleneck', 'max_residues': config['max_residues'],
                                  'contact_top_k': 8, 'contact_min_sep': 3, 'contact_cutoff': 8.,
                                  'plddt_threshold': .5, 'features': 'latent+codebook'}}
    provenance = {'identity': identity, 'train_ids': train_ids, 'val_ids': val_ids,
                  'model': {'hidden': 32, 'depth': 2, 'heads': 2, 'dim_head': 16},
                  'optimizer': {'learning_rate': 5e-4, 'effective_batch_size': 8, 'clip': 1.},
                  'seed': config['seed'],
                  'implementation_sha256': sha256('foldtree2/src/se3_struct_decoder.py'),
                  'selection_sha256': hashlib.sha256(inspect.getsource(split_structures).encode()).hexdigest(),
                  'loss_sha256': hashlib.sha256((inspect.getsource(losses)+inspect.getsource(frame_fape)).encode()).hexdigest()}
    # Validate before replacing a run's audit record or doing expensive extraction.
    if args.resume_from:
        saved = torch.load(args.resume_from, map_location='cpu', weights_only=False)
        if saved['provenance'] != provenance:
            raise ValueError('Resume rejected: incompatible provenance')
        del saved
    elif (output / 'last.pt').exists():
        raise ValueError('Run already has a checkpoint; use --resume-from or a fresh output directory')
    (output / 'provenance.json').write_text(json.dumps({**provenance, 'config': config,
        'checkpoints': pair, 'skipped': skipped, 'versions': {'python': platform.python_version(),
        'torch': torch.__version__, 'cuda': torch.version.cuda, 'lightning': base.pl.__version__}}, indent=2))
    probe = dataset[train_ids[0]].to(device)
    encoder_args = SimpleNamespace(pretrained_encoder_path=pair['encoder'], pretrained_encoder_full_path=pair['encoder'])
    encoder, dim = base.build_encoder(encoder_args, probe, device)
    if isinstance(encoder, base.FrozenProjectionEncoder):
        raise RuntimeError('Production encoder failed to load; projection fallback forbidden')
    decoder = load_production_decoder(pair['decoder'], device).float().eval()
    encoder = encoder.float().eval()
    frozen_before = [tensor_digest(m) for m in (encoder, decoder)]
    cache_key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    cache_dir = Path(config['cache']) / cache_key; cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / 'identity.json').write_text(json.dumps(identity, indent=2))
    entries = {}
    for index, identifier in enumerate(train_ids + val_ids):
        path = cache_dir / (hashlib.sha256(identifier.encode()).hexdigest() + '.pt')
        if path.exists():
            cached = torch.load(path, weights_only=False)
            if cached['identity'] != identity or cached['id'] != identifier:
                raise ValueError('Incompatible frozen-output cache')
            entry = cached['entry']
        else:
            entry = frozen_entry(dataset[identifier], encoder, decoder, device)
            tmp = path.with_suffix('.tmp')
            torch.save({'identity': identity, 'id': identifier, 'entry': entry}, tmp)
            tmp.replace(path)
        if index == 0:
            uncached = frozen_entry(dataset[identifier], encoder, decoder, device)
            for key, value in entry.items():
                if isinstance(value, torch.Tensor):
                    torch.testing.assert_close(value, uncached[key], atol=1e-6, rtol=1e-6, equal_nan=True)
            event('cache_verified', id=identifier)
        entries[identifier] = entry
        if index % 32 == 0:
            event('cache_progress', count=index + 1, total=len(train_ids + val_ids))
    # Verify frozen extraction did not mutate production weights or buffers.
    if frozen_before != [tensor_digest(m) for m in (encoder, decoder)]:
        raise RuntimeError('Frozen production state changed during cache extraction')
    if args.cache_only:
        return
    model = se3_denoiser(2 * dim, [32], 3, 40, .25, dropout_p=0., depth=2,
                        heads=2, dim_head=16, num_atom_types=40).to(device).float()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3)
    start, step, best, bad = 0, 0, float('inf'), 0
    if args.resume_from:
        ckpt = torch.load(args.resume_from, map_location=device, weights_only=False)
        start, step, best, bad = restore_training_state(ckpt, provenance, model, optimizer, scheduler, device)
        event('resume', epoch=start, global_step=step)
        if config.get('early_stopping', False) and bad >= 5:
            start = config['epochs']
            event('resume_terminal_early_stop', patience=bad)
    def on_device(identifier):
        return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in entries[identifier].items()}
    # Full forward cache parity, including coordinates and scalar outputs.
    model.eval()
    with torch.no_grad():
        fresh = frozen_entry(dataset[train_ids[0]], encoder, decoder, device)
        fresh = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in fresh.items()}
        for a, b in zip(forward(model, on_device(train_ids[0])), forward(model, fresh)):
            torch.testing.assert_close(a, b, atol=1e-4, rtol=1e-4)
    event('cache_forward_verified')
    baseline = []
    with torch.no_grad():
        for identifier in val_ids:
            e = on_device(identifier)
            baseline.append({'id': identifier, **metrics(e['coords'], e['baseline_angles'], e)})
    (output / 'baseline.json').write_text(json.dumps(baseline, indent=2))
    profile_rows = []
    started = time.perf_counter()
    if device.type == 'cuda': torch.cuda.reset_peak_memory_stats(device)
    try:
        for epoch in range(start, config['epochs']):
            model.train(); optimizer.zero_grad(set_to_none=True)
            epoch_start = time.perf_counter(); rows, norms = [], []
            batches = length_bucket_batches([len(entries[i]['coords']) for i in train_ids], 1, config['seed'] + epoch)
            for i, (idx,) in enumerate(batches):
                sample_started = time.perf_counter()
                e = on_device(train_ids[idx]); pred, angle = forward(model, e)
                total, geom, fape, aloss = losses(pred, angle, e)
                if not torch.isfinite(total): raise FloatingPointError('Nonfinite loss')
                # Last short accumulation group uses its true effective size.
                group_size = min(8, len(batches) - (i // 8) * 8)
                (total / group_size).backward()
                if epoch == start:
                    if device.type == 'cuda': torch.cuda.synchronize(device)
                    n = len(e['coords'])
                    profile_rows.append({'id': train_ids[idx], 'residues': n, 'batch_size': 1,
                                         'dense_adjacency_bytes': n*n, 'padding_residues': 0,
                                         'forward_backward_seconds': time.perf_counter()-sample_started,
                                         'peak_memory_bytes': torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else 0})
                rows.append([float(total.detach()), float(geom.detach()), float(fape.detach()), float(aloss.detach())])
                if (i + 1) % 8 == 0 or i + 1 == len(batches):
                    grads = [p.grad for p in model.parameters() if p.grad is not None]
                    if not grads or not all(torch.isfinite(g).all() for g in grads):
                        raise FloatingPointError('Missing or nonfinite refiner gradients')
                    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                    if norm <= 0: raise RuntimeError('Zero SE3 gradient')
                    norms.append(float(norm)); optimizer.step(); optimizer.zero_grad(set_to_none=True); step += 1
                if any(p.grad is not None for m in (encoder, decoder) for p in m.parameters()):
                    raise RuntimeError('Frozen production parameters received gradients')
            model.eval(); val_rows = []
            with torch.no_grad():
                for identifier in val_ids:
                    e = on_device(identifier); pred, angle = forward(model, e)
                    val_rows.append([float(v) for v in losses(pred, angle, e)])
            mean_train, mean_val = np.mean(rows, axis=0), np.mean(val_rows, axis=0)
            score = float(mean_val[2]); scheduler.step(score)
            improved = score < best - float(config.get('early_stopping_min_delta', 0.))
            best, bad = (score, 0) if improved else (best, bad + 1)
            elapsed = time.perf_counter() - epoch_start
            event('epoch', epoch=epoch, global_step=step, train_loss=mean_train[0], train_geometry=mean_train[1],
                  train_fape=mean_train[2], train_angle=mean_train[3], val_loss=mean_val[0], val_geometry=mean_val[1],
                  val_fape=score, val_angle=mean_val[3], gradient_norm_mean=float(np.mean(norms)),
                  effective_train_count=len(rows), effective_val_count=len(val_rows), seconds=elapsed,
                  structures_per_second=(len(rows)+len(val_rows))/elapsed,
                  peak_memory_bytes=torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else 0,
                  peak_host_memory_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
            (output / 'length_profile.json').write_text(json.dumps(profile_rows, indent=2))
            ckpt = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
                    'epoch': epoch, 'global_step': step, 'best': best, 'bad': bad, 'provenance': provenance,
                    'torch_rng': torch.get_rng_state(),
                    'cuda_rng': torch.cuda.get_rng_state(device) if device.type == 'cuda' else None}
            torch.save(ckpt, output / 'last.tmp')
            (output / 'last.tmp').replace(output / 'last.pt')
            if improved:
                torch.save(ckpt, output / 'best.tmp')
                (output / 'best.tmp').replace(output / 'best.pt')
            if config.get('early_stopping', False) and bad >= 5:
                event('early_stopping', epoch=epoch); break
        best_path = output / 'best.pt'
        if not best_path.exists() and args.resume_from:
            candidate = Path(args.resume_from).parent / 'best.pt'
            best_path = candidate if candidate.exists() else Path(args.resume_from)
        ckpt = torch.load(best_path, map_location=device, weights_only=False)
        if ckpt['provenance'] != provenance:
            raise ValueError('Best checkpoint has incompatible provenance')
        model.load_state_dict(ckpt['model']); model.eval()
        paired, diagnostics = [], []
        with torch.no_grad():
            for identifier, before in zip(val_ids, baseline):
                e = on_device(identifier); pred, angle = forward(model, e)
                after = metrics(pred, angle, e)
                mask = e['mask']; bm = mask[1:] & mask[:-1]
                diag = {'id': identifier, 'unique_tokens': int(e['tokens'][mask].unique().numel())}
                for label, coords in [('baseline',e['coords']),('refined',pred),('target',e['target'])]:
                    valid = coords[mask]
                    diag[label+'_radius_rms'] = float((valid-valid.mean(0)).square().sum(-1).mean().sqrt())
                    diag[label+'_bond_mean'] = float((coords[1:]-coords[:-1]).norm(dim=-1)[bm].mean())
                valid_seed = e['coords'][mask]
                d = torch.cdist(valid_seed,valid_seed)
                offdiag = ~torch.eye(len(d),device=device,dtype=torch.bool)
                diag['distance_contact_density'] = float(((d<8.) & (d>0.))[offdiag].float().mean())
                diagnostics.append(diag)
                paired.append({'id': identifier, 'baseline': before, 'refined': after,
                               'delta': {k: after[k]-before[k] for k in after if isinstance(after[k], (int,float)) and isinstance(before.get(k), (int,float))}})
        (output / 'diagnostics.json').write_text(json.dumps(diagnostics, indent=2))
        (output / 'paired_metrics.json').write_text(json.dumps(paired, indent=2))
        if frozen_before != [tensor_digest(m) for m in (encoder, decoder)]:
            raise RuntimeError('Frozen state changed during training')
        event('complete', seconds=time.perf_counter()-started, frozen_unchanged=True,
              best_epoch=ckpt['epoch'], nonfinite_events=0, failures=0)
    except Exception as exc:
        event('failure', error=repr(exc)); raise


if __name__ == '__main__':
    main()
