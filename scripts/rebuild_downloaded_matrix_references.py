#!/usr/bin/env python3
"""Rebuild an expanded matrix using existing encodings and alignments only.

No downloads, graph conversion, or model training are performed. Use a new
output directory to preserve previous failed matrices and their histories.
"""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

from benchmark_common import ROOT, digest, write_json
from extend_matrix_references import concatenate_encoded_fastas


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--size', type=int, default=50)
    parser.add_argument('--reference-root', type=Path, required=True)
    parser.add_argument('--outdir', type=Path, required=True)
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--benchmark-outdir', type=Path)
    args = parser.parse_args()
    if not 1 <= args.threads <= 8:
        parser.error('Use 1–8 threads')
    args.outdir = args.outdir.resolve()
    args.reference_root = args.reference_root.resolve()
    if args.outdir.exists():
        parser.error('Use a new output directory to preserve prior builds')
    sys.path.insert(0, str(ROOT))
    from foldtree2.src.encoder import load_encoded_fasta
    from foldtree2.makesubmat import build_char_set
    entries = json.loads((ROOT / 'configs/production_models.yaml').read_text())['models']
    entry = next(e for e in entries if e['size'] == args.size)
    production = ROOT / entry['directory']
    original = ROOT / f'runs/production_alphabets/notebook_heldout_persistent_restart_20261001/{args.size}/matrices'
    sources = [original / (Path(entry['encoder']).stem + '_aln_encoded.fasta'),
               args.reference_root / 'additional_aln_encoded.fasta']
    frames = [load_encoded_fasta(p, replace=False) for p in sources]
    if set(frames[0].index) & set(frames[1].index):
        raise ValueError('Original and additional encoded accessions overlap')
    stage = args.outdir / str(args.size) / 'matrices'
    stage.mkdir(parents=True)
    for name in ['encoder', 'decoder']:
        shutil.copy2(production / entry[name], stage / entry[name])
    encoded = stage / (Path(entry['encoder']).stem + '_aln_encoded.fasta')
    concatenate_encoded_fastas(sources, encoded)
    combined = load_encoded_fasta(encoded, replace=False)
    if len(combined) != sum(map(len, frames)):
        raise ValueError('Combined FASTA lost records')
    build_char_set(combined, expected_size=args.size)
    alignment_root = args.reference_root / 'struct_align'
    write_json(args.outdir / 'input_provenance.json', {
        'protocol': 'existing_references_readerfix_v1', 'size': args.size,
        'sources': {str(p): digest(p) for p in sources},
        'combined_sha256': digest(encoded), 'encoded_records': len(combined),
        'additional_alignment_files': len(list(alignment_root.glob('*/allvall.csv'))),
        'reference_root': str(args.reference_root), 'threshold': .025,
        'ema_span': 5, 'patience': 5, 'update_interval': 25})
    write_json(args.outdir / 'status.json', {'status': 'building_matrices'})
    result = subprocess.run([sys.executable, '-u', ROOT / 'scripts/prepare_production_alphabets.py',
        '--sizes', str(args.size), '--stages', 'matrices', 'validate', '--outdir', args.outdir,
        '--matrix-data', '/mnt/data2/datasets', '--additional-matrix-alignments', alignment_root,
        '--rebuild-matrices', '--reuse-staged-encoding', '--device', 'cpu', '--threads', str(args.threads)],
        cwd=ROOT)
    metrics = stage / (Path(entry['encoder']).stem + '_metrics.json')
    convergence = json.loads(metrics.read_text()).get('convergence') if metrics.exists() else None
    write_json(args.outdir / 'status.json', {
        'status': 'converged' if result.returncode == 0 else 'not_converged' if convergence else 'failed',
        'returncode': result.returncode, 'convergence': convergence})
    if result.returncode:
        sys.exit(result.returncode)
    if args.benchmark_outdir:
        subprocess.run([sys.executable, '-u', ROOT / 'scripts/run_local_benchmark_queue.py',
            '--benchmark-only', '--sizes', str(args.size), '--threads', str(args.threads),
            '--alignment-root', ROOT / 'runs/local_benchmarks/notebook_frobenius_20260930/experiments',
            '--reuse-native-root', ROOT / 'runs/local_benchmarks/missing_alphabet_analysis_20261001',
            '--outdir', args.benchmark_outdir], cwd=ROOT, check=True)
        for mode in ['native', 'controlled']:
            subprocess.run([sys.executable, '-u', ROOT / 'scripts/benchmark_species_supermatrix.py',
                '--sizes', '10', '20', '30', '40', '50', '--column-mode', mode,
                '--experiment-root', ROOT / 'runs/local_benchmarks/notebook_frobenius_20260930/experiments',
                '--native-root', args.benchmark_outdir,
                '--outdir', args.benchmark_outdir / ('species_supermatrix_' + mode),
                '--threads', str(args.threads)], cwd=ROOT, check=True)


if __name__ == '__main__':
    main()
