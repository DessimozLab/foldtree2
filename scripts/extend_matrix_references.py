#!/usr/bin/env python3
"""Expand AFDB matrix references in bounded, resumable batches, using mk2 graphs.

Data stays on the data disk. Original families precede all new families when
counting; accession background counts remain unique across the combined input.
Only the production builder can promote matrices, after sustained convergence.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmark_common import digest, write_json


def concatenate_encoded_fastas(sources, destination):
    """Join encoded FASTA bytes without inserting blank sequence lines."""
    with Path(destination).open('wb') as output:
        for path in sources:
            last = b''
            with Path(path).open('rb') as source:
                for block in iter(lambda: source.read(1024 * 1024), b''):
                    output.write(block)
                    last = block[-1:]
            if last and last != b'\n':
                output.write(b'\n')


def select_members(table, excluded, max_families=5000, members=5, seed=42):
    """Reservoir sample five members, preserving the notebook's random sampling."""
    import numpy as np
    import pandas as pd
    # First pass counts clusters; chunking keeps the 813 MiB table off the heap.
    counts = pd.Series(dtype='int64')
    for chunk in pd.read_csv(table, sep='\t', header=None, names=['entryId', 'repId', 'taxId'],
                             usecols=['repId'], dtype=str, chunksize=500000):
        counts = counts.add(chunk.repId.value_counts(), fill_value=0)
    eligible = sorted(set(counts[counts >= members].index) - set(excluded))
    rng = np.random.default_rng(seed)
    chosen = set(rng.choice(eligible, min(max_families, len(eligible)), replace=False))
    selected, seen = {key: [] for key in chosen}, {key: 0 for key in chosen}
    for chunk in pd.read_csv(table, sep='\t', header=None, names=['entryId', 'repId', 'taxId'],
                             dtype=str, chunksize=500000):
        for entry, rep in chunk.loc[chunk.repId.isin(chosen), ['entryId', 'repId']].itertuples(index=False, name=None):
            seen[rep] += 1
            if len(selected[rep]) < members:
                selected[rep].append(entry)
            else:
                index = rng.integers(seen[rep])
                if index < members:
                    selected[rep][index] = entry
    return {key: sorted(set(value)) for key, value in sorted(selected.items())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--size', type=int, default=50)
    parser.add_argument('--data-root', type=Path, default=Path('/mnt/data2/datasets'))
    parser.add_argument('--outdir', type=Path, required=True)
    parser.add_argument('--max-families', type=int, default=5000)
    parser.add_argument('--batch-size', type=int, default=1000)
    parser.add_argument('--members', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--wait-for-service')
    parser.add_argument('--benchmark-outdir', type=Path)
    args = parser.parse_args()
    if min(args.max_families, args.batch_size, args.members, args.threads) < 1 or args.threads > 8:
        parser.error('Positive limits required; at most eight CPU threads')
    out = args.outdir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    status = out / 'status.json'
    if args.wait_for_service:
        while subprocess.run(['systemctl', '--user', 'show', args.wait_for_service, '-p', 'ActiveState', '--value'],
                             check=True, capture_output=True, text=True).stdout.strip() in {'active', 'activating', 'deactivating', 'reloading'}:
            write_json(status, {'status': 'waiting', 'service': args.wait_for_service})
            time.sleep(30)
    import torch
    torch.set_num_threads(args.threads)
    from foldtree2.src.pdbgraphmk2 import PDB2PyG
    from foldtree2.src.download_utils import download_pdb_subprocess
    from foldtree2.makesubmat import encode_structures
    entries = {e['size']: e for e in json.loads((ROOT / 'configs/production_models.yaml').read_text())['models']}
    entry = entries[args.size]
    production = ROOT / entry['directory']
    original = ROOT / f'runs/production_alphabets/notebook_heldout_persistent_restart_20261001/{args.size}/matrices'
    original_fasta = original / (Path(entry['encoder']).stem + '_aln_encoded.fasta')
    original_root = args.data_root / 'struct_align'
    excluded = {p.name for p in original_root.iterdir() if p.is_dir()}
    table = args.data_root / 'afdbclusters/1-AFDBClusters-entryId_repId_taxId.tsv'
    selection_path = out / 'selection.json'
    protocol = {'size': args.size, 'seed': args.seed, 'members': args.members,
                'max_families': args.max_families, 'batch_size': args.batch_size,
                'encoder_sha256': digest(production / entry['encoder']),
                'original_fasta_sha256': digest(original_fasta),
                'cluster_table_sha256': digest(table), 'original_family_ids': sorted(excluded)}
    write_json(status, {'status': 'selecting_AFDB_clusters'})
    if selection_path.exists():
        selection = json.loads(selection_path.read_text())
        if selection['protocol'] != protocol:
            raise ValueError('Extension protocol changed; use a new output directory')
        families = selection['families']
    else:
        families = select_members(table, excluded, args.max_families, args.members, args.seed)
        write_json(selection_path, {'protocol': protocol, 'families': families})
    additional = out / 'struct_align'
    additional.mkdir(exist_ok=True)
    stage = out / str(args.size) / 'matrices'
    stage.mkdir(parents=True, exist_ok=True)
    for name in ['encoder', 'decoder']:
        shutil.copy2(production / entry[name], stage / entry[name])
    model = torch.load(stage / entry['encoder'], map_location='cpu', weights_only=False)
    model.to('cpu').eval()
    model.device = torch.device('cpu')
    converter = PDB2PyG()
    graph_file = out / 'additional_graphs_mk2.h5'
    family_ids = list(families)
    for offset in range(0, len(family_ids), args.batch_size):
        batch = family_ids[offset:offset + args.batch_size]
        for family in batch:
            directory = additional / family
            if (directory / 'completed.json').exists():
                continue
            structs = directory / 'structs'
            structs.mkdir(parents=True, exist_ok=True)
            write_json(status, {'status': 'download_graphs_align', 'family': family,
                                'selected_families': len(family_ids), 'batch_start': offset})
            # Reuse local PDBs by copying, otherwise invoke the established AFDB downloader.
            def fetch(accession):
                target = structs / (accession + '.pdb')
                for folder in [args.data_root / 'structs', args.data_root / 'foldtree2/structs']:
                    source = folder / target.name
                    if not target.exists() and source.is_file() and source.stat().st_size:
                        shutil.copy2(source, target)
                return download_pdb_subprocess(accession, str(structs))
            with ThreadPoolExecutor(max_workers=4) as executor:
                paths = [Path(p) for p in executor.map(fetch, families[family]) if p and Path(p).stat().st_size]
            if len(paths) < 2:
                write_json(directory / 'failed.json', {'reason': 'fewer_than_two_downloads'})
                continue
            failed = converter.store_pyg(list(map(str, paths)), str(graph_file), verbose=False, output_mode='a')
            bad = {str(p) for p, _ in failed}
            good = [p for p in paths if str(p) not in bad]
            if len(good) < 2:
                write_json(directory / 'failed.json', {'reason': 'fewer_than_two_valid_mk2_graphs'})
                continue
            with (directory / 'foldseek.log').open('a') as log:
                # Explicit valid structure paths keep failed graphs out of the alignment DB.
                valid_structs = directory / 'valid_structs'
                valid_structs.mkdir(exist_ok=True)
                for p in good:
                    link = valid_structs / p.name
                    if not link.exists():
                        link.symlink_to(p.resolve())
                subprocess.run(['foldseek', 'easy-search', valid_structs, valid_structs,
                    directory / 'allvall.csv', directory / 'tmp', '--threads', str(args.threads),
                    '--format-output', 'query,target,fident,alnlen,mismatch,gapopen,qstart,qend,tstart,tend,evalue,bits,qaln,taln',
                    '--exhaustive-search', '1'], check=True, stdout=log, stderr=subprocess.STDOUT)
            write_json(directory / 'completed.json', {'family': family, 'graphs': len(good),
                'alignment_sha256': digest(directory / 'allvall.csv'), 'graph_version': 'pdbgraphmk2'})
        if not graph_file.exists():
            raise RuntimeError('No additional graph input was successfully generated')
        new_fasta = Path(encode_structures(model, str(out), 'additional', 'cpu', str(graph_file)))
        from foldtree2.src.encoder import load_encoded_fasta
        old = load_encoded_fasta(str(original_fasta), replace=False)
        new = load_encoded_fasta(str(new_fasta), replace=False)
        if set(old.index) & set(new.index):
            raise ValueError('New AFDB structures overlap existing encoded accessions; refuse duplicate evidence')
        combined = stage / (Path(entry['encoder']).stem + '_aln_encoded.fasta')
        # Preserve native FASTA escaping, codebook order and original input bytes.
        concatenate_encoded_fastas([original_fasta, new_fasta], combined)
        write_json(status, {'status': 'building_expanded_matrices', 'attempted_new_families': offset + len(batch)})
        build_started = time.time()
        with (out / 'matrix_build.log').open('a') as log:
            result = subprocess.run([sys.executable, '-u', ROOT / 'scripts/prepare_production_alphabets.py',
                '--sizes', str(args.size), '--stages', 'matrices', 'validate', '--outdir', out,
                '--matrix-data', args.data_root, '--additional-matrix-alignments', additional,
                '--rebuild-matrices', '--reuse-staged-encoding', '--device', 'cpu', '--threads', str(args.threads)],
                cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        metrics_path = stage / (Path(entry['encoder']).stem + '_metrics.json')
        if not metrics_path.exists() or metrics_path.stat().st_mtime < build_started:
            raise RuntimeError('Expanded matrix build failed before producing metrics')
        convergence = json.loads(metrics_path.read_text())['convergence']
        write_json(status, {'status': 'converged' if result.returncode == 0 else 'more_reference_families_needed',
                            'attempted_new_families': offset + len(batch), 'convergence': convergence})
        if result.returncode == 0:
            if args.benchmark_outdir:
                subprocess.run([sys.executable, '-u', ROOT / 'scripts/run_local_benchmark_queue.py',
                    '--benchmark-only', '--sizes', str(args.size), '--threads', str(args.threads),
                    '--alignment-root', ROOT / 'runs/local_benchmarks/notebook_frobenius_20260930/experiments',
                    '--reuse-native-root', ROOT / 'runs/local_benchmarks/missing_alphabet_analysis_20261001',
                    '--outdir', args.benchmark_outdir], cwd=ROOT, check=True)
                for mode in ['native', 'controlled']:
                    subprocess.run([sys.executable, '-u', ROOT / 'scripts/benchmark_species_supermatrix.py',
                        '--sizes', *map(str, sorted({10, 20, 30, 40, args.size})), '--column-mode', mode,
                        '--experiment-root', ROOT / 'runs/local_benchmarks/notebook_frobenius_20260930/experiments',
                        '--native-root', args.benchmark_outdir,
                        '--outdir', args.benchmark_outdir / ('species_supermatrix_' + mode),
                        '--threads', str(args.threads)], cwd=ROOT, check=True)
            return
        if convergence.get('is_converged') or convergence.get('criterion') != 'sustained_ema_frobenius_v1':
            raise RuntimeError('Matrix build failed for a reason other than EMA nonconvergence')
    raise RuntimeError('Bounded AFDB expansion exhausted without sustained convergence; matrices stay staged')


if __name__ == '__main__':
    main()
