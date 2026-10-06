#!/usr/bin/env python3
"""Train missing production alphabets, generate matrices, and run local analyses.

Run from the repository root in the foldtree2 conda environment. Long jobs
write a separate log for each stage and only mark a stage complete on success.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def run(command, log):
    log.parent.mkdir(parents=True, exist_ok=True)
    print(f"Running {' '.join(map(str, command))}\nLog: {log}", flush=True)
    with log.open('a') as stream:
        subprocess.run(list(map(str, command)), cwd=ROOT, stdout=stream,
                       stderr=subprocess.STDOUT, check=True)


def digest(path):
    checksum = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            checksum.update(block)
    return checksum.hexdigest()


def check_convergence_acceptance(payload, acceptance, artifact_hashes, metrics_hash):
    """Accept only an explicit, artifact-bound waiver; never relabel convergence."""
    if payload.get('counting_method') != 'headerless_all_pair_occurrences_v2':
        raise ValueError('Corrected matrix counting is required')
    if payload.get('convergence', {}).get('is_converged'):
        return 'converged'
    if (not acceptance or acceptance.get('status') != 'accepted_nonconverged'
            or acceptance.get('authorization') != 'explicit_user_approval'
            or not acceptance.get('reason') or not acceptance.get('approved_at_utc')
            or acceptance.get('artifact_sha256') != artifact_hashes
            or acceptance.get('metrics_sha256') != metrics_hash):
        raise ValueError('Converged matrices or an explicit matching acceptance record are required')
    return 'accepted_nonconverged'


def validate_bundle(entry, require_convergence=False):
    import numpy as np
    import torch
    directory = ROOT / entry['directory']
    paths = {key: directory / entry[key] for key in ('encoder', 'decoder', 'mafft', 'raxml')}
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    encoder = torch.load(paths['encoder'], map_location='cpu', weights_only=False)
    decoder = torch.load(paths['decoder'], map_location='cpu', weights_only=False)
    size = entry['size']
    if encoder.num_embeddings != size:
        raise ValueError(f"Expected {size} states in {paths['encoder']}")
    for model in (encoder, decoder):
        if not all(torch.isfinite(v).all() for v in model.state_dict().values()):
            raise ValueError(f"Nonfinite checkpoint in {directory}")
    values = np.fromstring(paths['raxml'].read_text(), sep=' ')
    if len(values) != size * (size - 1) // 2 + size or not np.isfinite(values).all():
        raise ValueError(f"Invalid RAxML matrix for {size} states")
    if (values < 0).any() or (values[-size:] <= 0).any() or not np.isclose(values[-size:].sum(), 1, atol=1e-5):
        raise ValueError('Invalid RAxML rates/background frequencies')
    rows = [line.split() for line in paths['mafft'].read_text().splitlines() if line.strip()]
    if len(rows) != size * (size + 1) // 2 or not all(len(row) == 3 and np.isfinite(float(row[2])) for row in rows):
        raise ValueError(f"Invalid MAFFT matrix for {size} states")
    replacements = {'"': chr(248), '#': chr(247), '>': chr(249), '=': chr(250), '<': chr(251),
                    '-': chr(252), ' ': chr(253), '\r': chr(254), '\n': chr(255)}
    alphabet = {ord(replacements.get(chr(i + 1), chr(i + 1))) for i in range(size)}
    observed = {int(row[j], 16) for row in rows for j in (0, 1)}
    if observed != alphabet:
        raise ValueError(f'MAFFT symbols do not cover the full {size}-state codebook')
    report = {**entry, 'sha256': {k: digest(v) for k, v in paths.items()},
              'embedding_dim': encoder.out_channels, 'status': 'artifacts_validated'}
    metrics = directory / (paths['encoder'].stem + '_metrics.json')
    acceptance_path = directory / (paths['encoder'].stem + '_convergence_acceptance.json')
    if require_convergence or acceptance_path.exists():
        payload = json.loads(metrics.read_text())
        acceptance = json.loads(acceptance_path.read_text()) if acceptance_path.exists() else None
        report['matrix_acceptance'] = check_convergence_acceptance(
            payload, acceptance, report['sha256'], digest(metrics))
        if report['matrix_acceptance'] == 'accepted_nonconverged':
            report['convergence_acceptance'] = acceptance
            report['acceptance_sha256'] = digest(acceptance_path)
        report['convergence_sha256'] = digest(metrics)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sizes', type=int, nargs='+', default=[10, 20, 30, 40, 50])
    parser.add_argument('--stages', nargs='+', choices=['train', 'matrices', 'validate', 'experiments'],
                        default=['train', 'matrices', 'validate', 'experiments'])
    parser.add_argument('--config', default='configs/production_alphabet.yaml')
    parser.add_argument('--manifest', default='configs/production_models.yaml')
    parser.add_argument('--matrix-data', default='/mnt/data2/datasets')
    parser.add_argument('--matrix-dataset', default=None, help='Override the per-model reference graph dataset')
    parser.add_argument('--outdir', default='runs/production_alphabets')
    parser.add_argument('--families', default='families/Information_benchmark/marker_genes')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--max-families', type=int, help='Smoke-test limit; omit for full experiments')
    parser.add_argument('--wait-for-training-pid', type=int, help='Wait for an existing training workflow before dependent stages')
    parser.add_argument('--rebuild-matrices', action='store_true', help='Regenerate matrices in staging and preserve originals before promotion')
    parser.add_argument('--matrix-convergence-threshold', type=float, default=0.025)
    parser.add_argument('--matrix-convergence-patience', type=int, default=5)
    parser.add_argument('--matrix-convergence-ema-span', type=int, default=5)
    parser.add_argument('--reuse-staged-encoding', action='store_true',
                        help='Reuse staging encodings only if the staged encoder matches production')
    parser.add_argument('--matrix-update-interval', type=int, default=25)
    parser.add_argument('--additional-matrix-alignments', type=Path,
                        help='New independent AFDB families appended to the original reference set')
    args = parser.parse_args()
    os.chdir(ROOT)
    import yaml
    entries = json.loads(Path(args.manifest).read_text())['models']
    if set(args.sizes) - {entry['size'] for entry in entries}:
        parser.error('Sizes must be in the model manifest')
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    if args.wait_for_training_pid:
        print(f'Waiting for training workflow PID {args.wait_for_training_pid}', flush=True)
        while True:
            try:
                os.kill(args.wait_for_training_pid, 0)
            except ProcessLookupError:
                break
            time.sleep(30)
    reports = []
    for entry in entries:
        size = entry['size']
        if size not in args.sizes:
            continue
        directory = ROOT / entry['directory']
        encoder = directory / entry['encoder']
        decoder = directory / entry['decoder']
        stage = outdir / str(size)
        stage.mkdir(parents=True, exist_ok=True)
        if 'train' in args.stages and not (encoder.exists() and decoder.exists()):
            config = yaml.safe_load(Path(args.config).read_text())
            # Train into staging: a best checkpoint is not a completed training run.
            training_dir = stage / 'checkpoints'
            training_dir.mkdir(exist_ok=True)
            config.update(num_embeddings=size, model_name=f'production_{size}char',
                          output_dir=str(training_dir), device=args.device,
                          metrics_output=str(stage / 'training_metrics.json'))
            config_path = stage / 'training.yaml'
            if config_path.exists():
                previous = yaml.safe_load(config_path.read_text())
                ignored = {'output_dir', 'device', 'metrics_output', 'tensorboard_dir', 'run_name'}
                if {k: v for k, v in previous.items() if k not in ignored} != {k: v for k, v in config.items() if k not in ignored}:
                    raise ValueError(f'Training protocol changed in {stage}; use a fresh --outdir to preserve prior checkpoints')
            config_path.write_text(yaml.safe_dump(config, sort_keys=False))
            if not (stage / 'training_metrics.json').exists():
                run([sys.executable, '-u', '-m', 'foldtree2.learn_monodecoder', '--config', config_path], stage / 'train.log')
            run([sys.executable, '-u', 'scripts/evaluate_alphabet_checkpoint.py',
                 '--encoder', training_dir / encoder.name, '--decoder', training_dir / decoder.name,
                 '--dataset', config['dataset'], '--out', stage / 'best_pair_metrics.json',
                 '--val-seed', config.get('val_seed', 7), '--val-split', config.get('val_split', .1),
                 '--batch-size', config['batch_size'], '--device', args.device], stage / 'best_pair_validation.log')
            directory.mkdir(parents=True, exist_ok=True)
            import shutil
            for dest in (encoder, decoder):
                shutil.copy2(training_dir / dest.name, dest)
            shutil.copy2(stage / 'training_metrics.json', directory / 'training_metrics.json')
            shutil.copy2(config_path, directory / 'training.yaml')
            shutil.copy2(stage / 'best_pair_metrics.json', directory / 'best_pair_metrics.json')
        if 'matrices' in args.stages and (args.rebuild_matrices or not all((directory / entry[key]).exists() for key in ('mafft', 'raxml'))):
            import shutil
            # Also evaluate pairs promoted by a training supervisor launched
            # before selected-checkpoint evaluation was added to this script.
            if size in (10, 50) and (stage / 'training.yaml').exists() and not (directory / 'best_pair_metrics.json').exists():
                config = yaml.safe_load((stage / 'training.yaml').read_text())
                run([sys.executable, '-u', 'scripts/evaluate_alphabet_checkpoint.py',
                     '--encoder', encoder, '--decoder', decoder,
                     '--dataset', config['dataset'], '--out', stage / 'best_pair_metrics.json',
                     '--val-seed', config.get('val_seed', 7), '--val-split', config.get('val_split', .1),
                     '--batch-size', config['batch_size'], '--device', args.device,
                     '--threads', args.threads], stage / 'best_pair_validation.log')
                shutil.copy2(stage / 'best_pair_metrics.json', directory / 'best_pair_metrics.json')
            matrix_stage = stage / 'matrices'
            matrix_stage.mkdir(exist_ok=True)
            staged_encoder = matrix_stage / encoder.name
            encoded = matrix_stage / (encoder.stem + '_aln_encoded.fasta')
            reuse_encoding = args.reuse_staged_encoding and encoded.is_file()
            if reuse_encoding and (not staged_encoder.is_file() or digest(staged_encoder) != digest(encoder)):
                raise ValueError('Cached encoding encoder differs from production; refuse reuse')
            for path in (encoder, decoder):
                shutil.copy2(path, matrix_stage / path.name)
            # All approved production encoders now consume 857 residue features.
            matrix_dataset = args.matrix_dataset or 'foldtree2/structalnfinal.h5'
            run([sys.executable, '-u', '-m', 'foldtree2.makesubmat', '--modelname', encoder.stem,
                 '--modeldir', matrix_stage, '--datadir', args.matrix_data,
                 '--dataset', matrix_dataset, '--device', args.device, '--plot',
                 '--monitor-convergence', '--save-history', '--require-convergence',
                 '--convergence-threshold', args.matrix_convergence_threshold,
                 '--convergence-patience', args.matrix_convergence_patience,
                 '--convergence-ema-span', args.matrix_convergence_ema_span,
                 '--update-interval', args.matrix_update_interval,
                 *(['--additional-alignments-root', args.additional_matrix_alignments] if args.additional_matrix_alignments else []),
                 *([] if reuse_encoding else ['--encode_alns'])], stage / 'matrices.log')
            staged_entry = {**entry, 'directory': str(matrix_stage)}
            validate_bundle(staged_entry, require_convergence=True)
            existing = directory / entry['raxml']
            if existing.exists():
                old_hash = digest(existing)[:16]
                archive = directory / 'previous_matrices' / old_hash
                archive.mkdir(parents=True, exist_ok=True)
                for path in directory.glob(encoder.stem + '_*'):
                    if path.is_file() and path.suffix != '.pt':
                        shutil.copy2(path, archive / path.name)
                experiments = stage / 'experiments'
                if experiments.exists():
                    previous = stage / 'previous_experiments' / old_hash
                    previous.parent.mkdir(parents=True, exist_ok=True)
                    if previous.exists():
                        raise FileExistsError(f'Preserved experiment directory already exists: {previous}')
                    experiments.rename(previous)
            for path in matrix_stage.glob(encoder.stem + '_*'):
                if path.is_file() and path.suffix != '.pt':
                    shutil.copy2(path, directory / path.name)
        if 'validate' in args.stages or 'experiments' in args.stages:
            report = validate_bundle(entry)
            reports.append(report)
            (stage / 'artifacts.json').write_text(json.dumps(report, indent=2) + '\n')
        if 'experiments' in args.stages:
            command = [sys.executable, '-u', 'scripts/production_alphabet_experiments.py',
                       '--size', size, '--families', args.families, '--outdir', stage / 'experiments',
                       '--device', args.device, '--threads', args.threads]
            if args.max_families:
                command.extend(['--max-families', args.max_families])
            run(command, stage / 'experiments.log')
            print('Family preparation complete. Run compare_oma_representations.py after native AA/3Di/FT2 outputs exist; joint information/gain comparisons require 3Di.', flush=True)
    label = '_'.join(map(str, args.sizes))
    (outdir / f'artifacts_{label}.json').write_text(json.dumps(reports, indent=2) + '\n')


if __name__ == '__main__':
    main()
