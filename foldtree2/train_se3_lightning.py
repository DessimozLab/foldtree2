"""Cache-only Lightning DDP training of the validated residue SE(3) refiner.

The consumed global-batch cursor drives recovery. Lightning's iterator progress
is deliberately not replayed: loading restores optimizer/RNG explicitly, and a
fresh Trainer loop processes exactly the remaining stream.
"""
import argparse
from datetime import timedelta
import json
import math
import os
import platform
import random
import resource
import time
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
import torch.distributed as dist
import pytorch_lightning as pl
from pytorch_lightning.plugins.environments import TorchElasticEnvironment
from pytorch_lightning.strategies import DDPStrategy
from torch.utils.data import DataLoader, Subset
import yaml
from foldtree2.se3_cache import (FrozenDataset, GlobalBatchSampler, digest,
                               identity_hash, open_bundle, single_entry)
from foldtree2.se3_validation import forward, metrics
from foldtree2.src.geometry_seeds import lift_seed_with_local_normals
from foldtree2.se3_gradient_accumulation import GradientAccumulator
from foldtree2.src.losses.se3_ablation import AblationLoss
from foldtree2.src.se3_struct_decoder import (se3_denoiser, enable_stable_tensor_norm,
                                            enable_tensor_ff_input_norm)


def calibrated(entry, scale):
    result = dict(entry)
    center = entry['coords'][entry['mask']].mean(0)
    coords = ((entry['coords'] - center) * scale + center).masked_fill(~entry['mask'][:, None], 0.)
    result['coords'] = lift_seed_with_local_normals(coords, entry['features'], entry['mask'], .1)
    return result


def build_model(feature_dim, architecture):
    model = se3_denoiser(feature_dim, [architecture['hidden']], 3, 40, .25,
        dropout_p=0., depth=architecture['depth'], heads=architecture['heads'], dim_head=16,
        num_atom_types=40, coordinate_update='residual', zero_init_coordinate_update=True,
        coordinate_normalization='rms_radius', coordinate_residual_gain=32.,
        angle_output='sincos', device='cpu').float()
    if architecture['hidden'] > 32:
        enable_stable_tensor_norm(model)
    return model


def source_identity():
    files = ['train_se3_lightning.py', 'se3_cache.py', 'se3_validation.py', 'se3_gradient_accumulation.py',
             'src/se3_struct_decoder.py', 'src/geometry_seeds.py',
             'src/losses/se3_ablation.py', 'src/losses/fape.py', 'src/dynamictan.py',
             'src/quantizers.py', 'src/folding_refiner.py', 'src/manifold_hyper_connections.py']
    return {name: digest(Path(__file__).parent / name) for name in files}


def provenance(config, manifest):
    training = {k: v for k, v in config.items() if k not in ('cache', 'output', 'nodes', 'devices', 'accelerator')}
    return {'protocol': 'se3-lightning-v1', 'bundle': manifest['bundle_sha256'],
            'training': training, 'sources': source_identity(),
            'feature_dim': manifest['feature_dim'], 'input_identity':manifest['input_identity'],
            'preprocessing':{'scale':manifest['scale'],'lift_fraction':.1,'gain':32.}}


def check_resume(saved, current, topology, allow_change=False):
    if saved['provenance'] != current:
        raise ValueError('Resume rejected: code, model, cache, split or training configuration differs')
    if saved['topology'] != topology:
        if not allow_change or saved['stream']['cursor'] != 0:
            raise ValueError('Topology change requires explicit opt-in and an epoch boundary')


def check_warm_start_normalization(previous, current):
    if previous.get('tensor_ff_input_norm', False) != current.get('tensor_ff_input_norm', False):
        raise ValueError('Warm-start tensor feed-forward normalization mismatch')


class CacheData(pl.LightningDataModule):
    def __init__(self, directory, manifest, state, global_batch=16, seed=42):
        super().__init__()
        self.directory, self.manifest, self.state = directory, manifest, state
        self.global_batch, self.seed = global_batch, seed

    def train_dataloader(self):
        dataset = FrozenDataset(self.directory, self.manifest, 'train')
        sampler = GlobalBatchSampler([self.manifest['files'][i]['length'] for i in dataset.ids],
            self.state, self.trainer.global_rank, self.trainer.world_size, self.seed, self.global_batch)
        return DataLoader(dataset, batch_size=1, sampler=sampler, collate_fn=single_entry, num_workers=0)

    def val_dataloader(self):
        dataset = FrozenDataset(self.directory, self.manifest, 'val')
        # Every validation ID once; unequal rank lengths are allowed because no
        # per-batch collectives or buffer broadcasts occur in validation.
        indices = range(self.trainer.global_rank, len(dataset), self.trainer.world_size)
        return DataLoader(Subset(dataset, list(indices)), batch_size=1,
                          collate_fn=single_entry, num_workers=0)


class Refiner(pl.LightningModule):
    def __init__(self, config, manifest, resume=None, allow_topology_change=False):
        super().__init__()
        self.automatic_optimization = False
        self.config, self.manifest = config, manifest
        self.model = build_model(manifest['feature_dim'], config['model'])
        if not isinstance(config.get('tensor_ff_input_norm', False), bool):
            raise ValueError('tensor_ff_input_norm must be an explicit boolean')
        if config.get('tensor_ff_input_norm', False):
            enable_tensor_ff_input_norm(self.model)
        self.loss_fn = AblationLoss(True, True)
        self.stream = {'epoch': 0, 'cursor': 0, 'updates': 0}
        self.provenance = provenance(config, manifest)
        self.provenance['initialization'] = resume['se3']['provenance'].get('initialization', {'kind':'fresh'}) if resume else {'kind':'fresh'}
        self.resume = resume
        self.allow_change = allow_topology_change
        self.best, self.bad = float('inf'), 0
        self.scheduler = None
        self.validation_rows = []
        self.epoch_train = []
        self.timing = []
        self.bn_stats = []
        self.bn_start = None
        self.update_began = None
        self.last_end = time.perf_counter()
        self.stop_epoch = None
        self.stop_updates = None
        self.active_batch = None
        self.gradient_accumulator = None
        if config.get('gradient_accumulation_dtype', 'float32') not in ('float32','float64'):
            raise ValueError('gradient_accumulation_dtype must be float32 or float64')
        if resume:
            self.load_state_dict(resume['state_dict'])
            self.stream.update(resume['se3']['stream'])
            self.best, self.bad = resume['se3']['best'], resume['se3']['bad']

    def event(self, kind, **values):
        if self.global_rank == 0:
            record = {'event': kind, 'time': time.time(), 'epoch': self.stream['epoch'],
                      'global_step': self.stream['updates'], **values}
            text = json.dumps(record, allow_nan=False)
            root = Path(self.config['output']); root.mkdir(parents=True, exist_ok=True)
            with (root / 'events.jsonl').open('a') as f:
                f.write(text + '\n')
            print(text, flush=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config['learning_rate'], weight_decay=.01)
        if self.config.get('scheduler', False):
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=.5, patience=3)
        return optimizer

    def on_fit_start(self):
        topology = {'world_size': self.trainer.world_size, 'nodes': self.trainer.num_nodes}
        if self.resume:
            saved = self.resume['se3']
            check_resume(saved, self.provenance, topology, self.allow_change)
            self.optimizers().optimizer.load_state_dict(self.resume['optimizer_states'][0])
            if self.scheduler:
                self.scheduler.load_state_dict(saved['scheduler'])
            if saved['topology'] == topology:
                rng = saved['rng'][self.global_rank]
                self.epoch_train = list(rng.get('epoch_train', []))
                torch.set_rng_state(rng['torch'].cpu()); random.setstate(rng['python'])
                np.random.set_state(rng['numpy'])
                if self.device.type == 'cuda':
                    torch.cuda.set_rng_state(rng['cuda'].cpu(), self.device)
            else:
                pl.seed_everything(self.config['seed'] + self.global_rank, workers=True)
                self.event('topology_transition', previous=saved['topology'], current=topology,
                           bitwise_equivalence=False)
        self.event('ready', topology=topology, bundle=self.manifest['bundle_sha256'],
                   frozen_production_calls=0, initialization=self.provenance['initialization'], parameters=sum(p.numel() for p in self.model.parameters()))

    def training_step(self, batch, batch_idx):
        self.active_batch = {'phase':'train', 'id':batch['id'], 'length':len(batch['coords']),
                             'batch_idx':batch_idx}
        started = time.perf_counter()
        data_wait = started - self.last_end
        e = calibrated(batch, self.manifest['scale'])
        if not self.bn_stats:
            self.update_began = started
            bn = self.model.bn
            self.bn_start = (bn.running_mean.clone(), bn.running_var.clone(), bn.num_batches_tracked.clone())
        features = batch['features'][batch['mask']]
        self.bn_stats.append((features.mean(0).detach().cpu(), features.var(0, unbiased=True).detach().cpu()))
        optimizer = self.optimizers()
        accumulation = self.config['global_batch'] // self.trainer.world_size
        boundary = (batch_idx + 1) % accumulation == 0
        high_precision = self.config.get('gradient_accumulation_dtype') == 'float64'
        with optimizer.toggle_model(sync_grad=boundary and not high_precision):
            pred, angle = forward(self.model, e)
            values = self.loss_fn(pred, angle, e)
            if not all(torch.isfinite(v) for v in values):
                raise FloatingPointError(f'Nonfinite loss: {batch["id"]}')
            self.manual_backward(values[0] if high_precision else values[0] / accumulation)
            if high_precision:
                if self.gradient_accumulator is None:
                    self.gradient_accumulator = GradientAccumulator(self.model.parameters())
                self.gradient_accumulator.collect()
        self.epoch_train.append(float(values[1].detach()))
        if boundary:
            if high_precision:
                self.gradient_accumulator.finish(self.config['global_batch'])
            grads = [p.grad for p in self.model.parameters() if p.grad is not None]
            if not grads or not all(torch.isfinite(g).all() for g in grads):
                raise FloatingPointError('Nonfinite or missing gradients')
            norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1., error_if_nonfinite=True)
            if norm <= 0:
                raise RuntimeError('Zero refiner gradient')
            optimizer.step(); optimizer.zero_grad(set_to_none=True)
            self.merge_batchnorm()
            self.stream['cursor'] += 1; self.stream['updates'] += 1
            if self.device.type == 'cuda':
                torch.cuda.synchronize(self.device)
            elapsed = time.perf_counter() - self.update_began
            self.event('update', loss=float(values[0].detach()), gradient_norm=float(norm),
                       data_wait_seconds=data_wait, compute_seconds=time.perf_counter() - started,
                       length=len(batch['coords']), update_seconds=elapsed,
                       structures_per_second=self.config['global_batch'] / elapsed,
                       peak_host_memory_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                       peak_memory_bytes=torch.cuda.max_memory_allocated() if self.device.type == 'cuda' else 0)
            if self.stream['updates'] % self.config.get('checkpoint_every', 64) == 0 and self.stream['cursor'] < len(self.manifest['train_ids']) // self.config['global_batch']:
                self.save_atomic('last.ckpt')
        self.last_end = time.perf_counter()
        if self.stop_updates and self.stream['updates'] >= self.stop_updates:
            self.trainer.should_stop = True
        return values[0].detach()

    def merge_batchnorm(self):
        # Training BN normalizes each structure independently. Replay only its
        # running statistics in the global sample order, preserving serial eval.
        gathered = [None] * self.trainer.world_size
        if dist.is_initialized():
            dist.all_gather_object(gathered, self.bn_stats)
        else:
            gathered[0] = self.bn_stats
        bn = self.model.bn
        mean, variance, count = self.bn_start
        with torch.no_grad():
            for micro in range(len(self.bn_stats)):
                for rank in range(self.trainer.world_size):
                    m, v = gathered[rank][micro]
                    count += 1
                    momentum = bn.momentum if bn.momentum is not None else 1. / int(count)
                    mean = (1. - momentum) * mean + momentum * m.to(self.device)
                    variance = (1. - momentum) * variance + momentum * v.to(self.device)
            bn.running_mean.copy_(mean); bn.running_var.copy_(variance)
            bn.num_batches_tracked.copy_(count)
        self.bn_stats = []; self.bn_start = None

    def on_validation_epoch_start(self):
        self.validation_rows = []

    def validation_step(self, batch, batch_idx):
        self.active_batch = {'phase':'validation', 'id':batch['id'],
                             'length':len(batch['coords']), 'batch_idx':batch_idx}
        e = calibrated(batch, self.manifest['scale'])
        p, a = forward(self.model, e)
        values = self.loss_fn(p, a, e)
        if not all(torch.isfinite(v) for v in values):
            raise FloatingPointError(f'Nonfinite validation: {batch["id"]}')
        row = {'id': batch['id'], 'geometry': float(values[1]), 'evaluation_fape': float(values[2])}
        if self.config.get('structural_metrics', True):
            row['raw'] = metrics(batch['coords'], batch['baseline_angles'], batch)
            row['refined'] = metrics(p, a, e)
        self.validation_rows.append(row)

    def on_validation_epoch_end(self):
        keys = ['geometry', 'evaluation_fape']
        if self.validation_rows and 'refined' in self.validation_rows[0]:
            keys += ['refined.' + k for k in sorted(self.validation_rows[0]['refined'])]
        else:
            # All ranks must use the same keys even when some have no examples.
            if self.config.get('structural_metrics', True):
                keys += ['refined.' + k for k in sorted(['fape', 'ca_rmsd', 'tm_score', 'lddt',
                         'ca_bond_mae', 'ca_bond_deviation_3_8', 'angle_mae_radians', 'valid_residues',
                         'ca_bend_mae_radians', 'ca_torsion_mae_radians'])]
        sums = []
        for key in keys:
            values = [r['refined'].get(key[8:]) if key.startswith('refined.') else r[key]
                      for r in self.validation_rows]
            values = [v for v in values if v is not None]
            if not all(math.isfinite(v) for v in values):
                raise FloatingPointError(f'Nonfinite metric: {key}')
            sums.append([sum(values), len(values)])
        totals = torch.tensor(sums, dtype=torch.float64, device=self.device)
        if dist.is_initialized():
            dist.all_reduce(totals)
        if int(totals[0, 1]) != len(self.manifest['val_ids']):
            raise RuntimeError('Distributed validation sample count differs from cohort')
        means = {key: float(total / count) if count else None for key, (total, count) in zip(keys, totals)}
        complete = self.stream['cursor'] == len(self.manifest['train_ids']) // self.config['global_batch']
        self.improved = complete and means['geometry'] < self.best - self.config.get('min_delta', 0.)
        if complete:
            if self.improved:
                self.best, self.bad = means['geometry'], 0
            else:
                self.bad += 1
            if self.scheduler:
                self.scheduler.step(means['geometry'])
        self.event('validation', complete_epoch=complete, metrics=means, counts={k: int(v[1]) for k, v in zip(keys, totals)})
        gathered = [None] * self.trainer.world_size
        if dist.is_initialized():
            dist.all_gather_object(gathered, self.validation_rows)
        else:
            gathered[0] = self.validation_rows
        if self.global_rank == 0:
            rows = [r for part in gathered for r in part]
            if len({r['id'] for r in rows}) != len(rows):
                raise RuntimeError('Duplicate validation structures')
            root = Path(self.config['output'])
            (root / f'validation-{self.stream["epoch"]:03d}.json').write_text(json.dumps(rows, indent=2))
        if self.config.get('patience') and self.bad >= self.config['patience']:
            self.trainer.should_stop = True

    def on_train_epoch_end(self):
        totals = torch.tensor([sum(self.epoch_train), len(self.epoch_train)], dtype=torch.float64, device=self.device)
        if dist.is_initialized():
            dist.all_reduce(totals)
        if totals[1] <= 0:
            raise RuntimeError('No training structures consumed')
        self.event('train_epoch', geometry=float(totals[0] / totals[1]), effective_structures=int(totals[1]))
        complete = self.stream['cursor'] == len(self.manifest['train_ids']) // self.config['global_batch']
        if complete:
            self.stream['epoch'] += 1; self.stream['cursor'] = 0
            self.epoch_train = []
        self.save_atomic('last.ckpt')
        if complete and getattr(self, 'improved', False):
            self.save_atomic('best.ckpt')
        if self.stop_epoch and self.stream['epoch'] >= self.stop_epoch:
            self.trainer.should_stop = True

    def on_save_checkpoint(self, checkpoint):
        rng = {'epoch_train':list(self.epoch_train), 'torch': torch.get_rng_state(), 'python': random.getstate(), 'numpy': np.random.get_state(),
               'cuda': torch.cuda.get_rng_state(self.device) if self.device.type == 'cuda' else None}
        states = [None] * self.trainer.world_size
        if dist.is_initialized():
            dist.all_gather_object(states, rng)
        else:
            states[0] = rng
        checkpoint['se3'] = {'provenance': self.provenance, 'stream': dict(self.stream),
            'topology': {'world_size': self.trainer.world_size, 'nodes': self.trainer.num_nodes},
            'rng': states, 'best': self.best, 'bad': self.bad,
            'scheduler': self.scheduler.state_dict() if self.scheduler else None}

    def save_atomic(self, name):
        root = Path(self.config['output']); root.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        self.trainer.save_checkpoint(str(root / (name + '.tmp')))
        self.trainer.strategy.barrier()
        if self.global_rank == 0:
            (root / (name + '.tmp')).replace(root / name)
        self.trainer.strategy.barrier()
        self.event('checkpoint', path=name, seconds=time.perf_counter() - started,
                   cursor=self.stream['cursor'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--resume-from'); parser.add_argument('--warm-start')
    parser.add_argument('--allow-topology-change', action='store_true')
    parser.add_argument('--stop-after-epoch', type=int)
    parser.add_argument('--stop-after-updates', type=int)
    parser.add_argument('--profile', action='store_true')
    args = parser.parse_args()
    if args.resume_from and args.warm_start:
        parser.error('Choose resume or warm-start')
    config = yaml.safe_load(Path(args.config).read_text())
    if config['model'] not in ({'hidden':32,'depth':2,'heads':2}, {'hidden':64,'depth':4,'heads':4}):
        raise ValueError('This architecture failed float32 numerical gates; depth-six rollout remains blocked')
    if config['seed'] != 42 or config['global_batch'] != 16 or config['max_residues'] != 256:
        raise ValueError('Canonical seed/global batch/length must be 42/16/256')
    torch.set_num_threads(2); pl.seed_everything(42, workers=True)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    manifest = open_bundle(config['cache'])
    if len(manifest['train_ids']) != config['train_count'] or len(manifest['val_ids']) != config['val_count']:
        raise ValueError('Cache counts differ from experiment')
    if any(r['length'] > config['max_residues'] for r in manifest['files'].values()):
        raise ValueError('Oversized cached structure')
    if config['global_batch'] % (config['devices'] * config['nodes']):
        raise ValueError('Topology does not divide global batch')
    root = Path(config['output'])
    if (root / 'events.jsonl').exists() and not args.resume_from:
        raise ValueError('Use a fresh output or --resume-from')
    root.mkdir(parents=True, exist_ok=True)
    saved = torch.load(args.resume_from, map_location='cpu', weights_only=False) if args.resume_from else None
    module = Refiner(config, manifest, saved, args.allow_topology_change)
    module.stop_epoch = args.stop_after_epoch
    module.stop_updates = args.stop_after_updates or config.get('stop_after_updates')
    if args.warm_start:
        warm = torch.load(args.warm_start, map_location='cpu', weights_only=False)
        if 'se3' in warm:
            prior = warm['se3']['provenance']
            check_warm_start_normalization(prior['training'], config)
            if prior['training']['model'] != config['model'] or prior['sources'] != module.provenance['sources'] or prior['input_identity'] != manifest['input_identity'] or prior['preprocessing'] != module.provenance['preprocessing']:
                raise ValueError('Warm-start architecture or implementation mismatch')
            module.load_state_dict(warm['state_dict'])
        else:
            check_warm_start_normalization({}, config)
            spec = warm['provenance']['spec']
            if any(spec[k] != config['model'][k] for k in ('hidden', 'depth', 'heads')):
                raise ValueError('Legacy warm-start architecture mismatch')
            if warm['provenance']['input_identity'] != manifest['input_identity']:
                raise ValueError('Legacy warm-start frozen production identity mismatch')
            module.model.load_state_dict(warm['model'])
        module.provenance['initialization'] = {'kind':'warm_start','sha256':digest(args.warm_start)}
    if saved:
        check_resume(saved['se3'], module.provenance,
                     {'world_size': config['nodes'] * config['devices'], 'nodes': config['nodes']}, args.allow_topology_change)
    if int(os.environ.get('RANK', 0)) == 0:
        (root / 'provenance.json').write_text(json.dumps(module.provenance, indent=2))
        import importlib.metadata as metadata
        (root / 'software.json').write_text(json.dumps({'python': platform.python_version(),
            'torch': torch.__version__, 'cuda': torch.version.cuda,
            'packages': {p: metadata.version(p) for p in ['pytorch-lightning', 'gotennet-pytorch',
                        'e3nn', 'torch-geometric', 'einops', 'numpy']}}, indent=2))
    world = config['devices'] * config['nodes']
    timeout = config.get('distributed_timeout_seconds', 1800)
    if type(timeout) is not int or timeout <= 0:
        raise ValueError('distributed_timeout_seconds must be a positive integer')
    strategy = DDPStrategy(find_unused_parameters=True, broadcast_buffers=False,
                          timeout=timedelta(seconds=timeout)) if world > 1 else 'auto'
    plugins = [TorchElasticEnvironment()] if 'LOCAL_RANK' in os.environ else []
    remaining = config['epochs'] - module.stream['epoch']
    if remaining <= 0:
        raise ValueError('Requested epochs already completed')
    profiler = None
    if args.profile:
        from pytorch_lightning.profilers import PyTorchProfiler
        profiler = PyTorchProfiler(dirpath=str(root / 'profiles'),
            filename='rank-' + os.environ.get('RANK', '0'),
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1))
    trainer = pl.Trainer(profiler=profiler, accelerator=config['accelerator'], devices=config['devices'],
        num_nodes=config['nodes'], strategy=strategy, plugins=plugins, precision='32-true',
        max_epochs=remaining, num_sanity_val_steps=0, use_distributed_sampler=False,
        reload_dataloaders_every_n_epochs=1, enable_checkpointing=False, logger=False,
        enable_progress_bar=False, deterministic=True)
    try:
        trainer.fit(module, datamodule=CacheData(config['cache'], manifest, module.stream))
    except Exception as exc:
        failure = {'time':time.time(), 'rank':int(os.environ.get('RANK', 0)),
                   'event':'failure', 'error':repr(exc), 'stream':dict(module.stream),
                   'active_batch':module.active_batch}
        # Diagnostic only: an interrupted accumulation is not a resumable update.
        # No collectives here: another rank may already be blocked or have failed.
        try:
            snapshot = root / ('failure-state-rank-' + str(failure['rank']) + '.pt')
            torch.save({'state_dict':module.state_dict(), 'stream':dict(module.stream),
                        'active_batch':module.active_batch, 'provenance':module.provenance,
                        'resumable':False}, snapshot)
            failure['diagnostic_state'] = snapshot.name
        except Exception as capture_error:
            failure['diagnostic_capture_error'] = repr(capture_error)
        (root / ('failure-rank-' + str(failure['rank']) + '.json')).write_text(json.dumps(failure, indent=2))
        raise


if __name__ == '__main__':
    main()
