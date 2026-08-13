import argparse
import glob
import os
import sys
import json

import h5py
import numpy as np
import torch
import tqdm

from foldtree2.src import pdbgraph
from foldtree2.src import pdbgraphmk2

def print_about():
    ascii_art = r'''
    
+-----------------------------------------------------------+
|                         foldtree2                          |
|                 pdb2pyg  (PDB → PyG graphs)                |
|          Structure → Contacts • Angles • Features          |
|                 Ready for PyTorch Geometric                |
|                      🧬   🧠   🌳                          |
+-----------------------------------------------------------+


    '''
    print(ascii_art)
    print("PDB to PyTorch Geometric Converter")
    print("-" * 50)
    print("Convert protein structure files (PDB) into PyTorch Geometric graph format")
    print("for neural network processing with FoldTree2.\n")
    print("This tool extracts structural features including:")
    print("  • Contact maps and hydrogen bonds")
    print("  • Secondary structure annotations")
    print("  • Backbone angles (phi, psi, omega)")
    print("  • Amino acid properties")
    print("  • Optional FoldX energy predictions")
    print("  • Optional ProDy interaction features\n")
    print("Project: https://github.com/DessimozLab/foldtree2")
    print("Contact: dmoi@unil.ch\n")
    print("Run with --help for usage instructions.")


def _configure_reproducibility():
    torch.manual_seed(0)
    np.random.seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _looks_like_glob(path):
    return any(token in path for token in ['*', '?', '['])


def _resolve_input_mode(input_path, requested_mode):
    if requested_mode != 'auto':
        return requested_mode

    if os.path.isfile(f'{input_path}.lookup'):
        return 'foldcomp-db'

    if input_path.lower().endswith('.fcz'):
        return 'fcz'

    if _looks_like_glob(input_path) and '.fcz' in input_path.lower():
        return 'fcz'

    if os.path.isdir(input_path):
        has_pdb = len(glob.glob(os.path.join(input_path, '*.pdb'))) > 0
        has_fcz = len(glob.glob(os.path.join(input_path, '*.fcz'))) > 0
        if has_fcz and not has_pdb:
            return 'fcz'

    return 'pdb'


def _resolve_structure_files(input_path, mode):
    if os.path.isdir(input_path):
        if mode == 'fcz':
            files = glob.glob(os.path.join(input_path, '*.fcz'))
        else:
            files = glob.glob(os.path.join(input_path, '*.pdb'))
    elif _looks_like_glob(input_path):
        files = glob.glob(input_path, recursive=True)
        if mode == 'fcz':
            files = [f for f in files if f.lower().endswith('.fcz')]
        else:
            files = [f for f in files if f.lower().endswith('.pdb')]
    else:
        files = [input_path]

    files = [f for f in files if os.path.isfile(f)]
    files = sorted(set(files))
    return files


def _checkpoint_manifest_path(output_h5):
    return f'{output_h5}.checkpoint.json'


def _load_checkpoint_state(manifest_path):
    if not os.path.exists(manifest_path):
        return set()
    with open(manifest_path, 'r', encoding='utf-8') as handle:
        payload = json.load(handle)
    completed = payload.get('completed', [])
    return {os.path.abspath(item) for item in completed}


def _save_checkpoint_state(manifest_path, completed_paths):
    payload = {'completed': sorted(str(path) for path in completed_paths)}
    with open(manifest_path, 'w', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=2)


def _filter_remaining_inputs(files, manifest_path):
    completed = _load_checkpoint_state(manifest_path)
    if not completed:
        return files
    remaining = []
    for path in files:
        if os.path.abspath(path) not in completed:
            remaining.append(path)
    return remaining


def _parse_foldcomp_ids(foldcomp_ids_arg):
    if not foldcomp_ids_arg:
        return None

    if os.path.isfile(foldcomp_ids_arg):
        ids = []
        with open(foldcomp_ids_arg, 'r', encoding='utf-8') as handle:
            for line in handle:
                entry = line.strip()
                if entry:
                    ids.append(entry)
        return ids

    ids = [item.strip() for item in foldcomp_ids_arg.split(',') if item.strip()]
    return ids or None


def _write_heterodata_group(structs_group, hetero_data):
    identifier = hetero_data.identifier
    if identifier in structs_group:
        return False
    struct_group = structs_group.create_group(identifier)

    node_group = struct_group.create_group('node')
    for node_type in hetero_data.node_types:
        if hetero_data[node_type].x is not None:
            type_group = node_group.create_group(node_type)
            type_group.create_dataset('x', data=hetero_data[node_type].x.numpy())

    edge_group = struct_group.create_group('edge')
    for edge_type in hetero_data.edge_types:
        edge_name = f'{edge_type[0]}_{edge_type[1]}_{edge_type[2]}'
        type_group = edge_group.create_group(edge_name)
        if hetero_data[edge_type].edge_index is not None:
            type_group.create_dataset('edge_index', data=hetero_data[edge_type].edge_index.numpy())
        if hasattr(hetero_data[edge_type], 'edge_attr') and hetero_data[edge_type].edge_attr is not None:
            type_group.create_dataset('edge_attr', data=hetero_data[edge_type].edge_attr.numpy())


def _store_foldcomp_db(converter, input_path, output_h5, ids=None, max_structures=None, verbose=False, chunk_size=256):
    if not os.path.isfile(f'{input_path}.lookup'):
        raise FileNotFoundError(
            f'Foldcomp DB lookup file not found: {input_path}.lookup. '
            'For Foldcomp DB mode, pass the DB root path (without .lookup).'
        )

    if ids is None:
        ids = pdbgraphmk2._load_foldcomp_ids(input_path)

    if max_structures is not None:
        ids = ids[:int(max_structures)]

    if len(ids) == 0:
        raise ValueError('No Foldcomp IDs found to encode.')

    dataset = pdbgraphmk2.FoldcompStructureDataset(
        input_path,
        ids=ids,
        converter=converter,
        cache_size=0,
        persistent_db=True,
        persistent_window=max(32, int(chunk_size)),
    )

    failed = []
    written = 0
    with h5py.File(output_h5, mode='a') as h5f:
        structs_group = h5f.require_group('structs')

        pbar = tqdm.tqdm(total=len(ids), desc='Encoding Foldcomp structures', unit='struct')
        try:
            for idx, entry_id in enumerate(ids):
                try:
                    graph = dataset[idx]
                    _write_heterodata_group(structs_group, graph)
                    written += 1
                except Exception as exc:
                    failed.append((entry_id, str(exc)))
                    if verbose:
                        print(f'Failed structure {entry_id}: {exc}')
                finally:
                    pbar.update(1)
        finally:
            pbar.close()

    if verbose:
        print(f'Successfully stored {written}/{len(ids)} structures from Foldcomp DB.')
    return failed


def build_parser():
    parser = argparse.ArgumentParser(description='Encode structures to PyTorch Geometric HDF5')
    parser.add_argument(
        'input_path',
        type=str,
        help=(
            'Input source: PDB directory, file path, glob pattern, .fcz input, '
            'or Foldcomp DB root path (path with companion .lookup).'
        ),
    )
    parser.add_argument('output_h5', type=str, help='Output file with PyTorch Geometric graphs')
    parser.add_argument('foldxdir', type=str, nargs='?', default=None, help='Legacy FoldX directory (PDB mode only)')
    parser.add_argument('--distance', type=float, default=15, help='Distance threshold for contact map (PDB mode; default: 15)')
    parser.add_argument('--add-prody', action='store_true', default=False, help='Add ProDy features in legacy PDB mode')
    parser.add_argument('--verbose', action='store_true', default=False, help='Verbose output')
    parser.add_argument('--multiprocessing', action='store_true', default=False, help='Use multiprocessing for file-based conversion')
    parser.add_argument('--ncpu', type=int, default=25, help='Number of CPUs for multiprocessing (default: 25)')
    parser.add_argument('--nstructs', type=int, default=None, help='Maximum number of structures to use after shuffling')
    parser.add_argument(
        '--checkpoint',
        action='store_true',
        default=False,
        help='Persist a resumable checkpoint manifest for long conversion jobs',
    )
    parser.add_argument(
        '--aapropcsv',
        type=str,
        default=None,
        help='Amino acid property CSV file (default: packaged foldtree2/config/aaindex1.csv)',
    )
    parser.add_argument(
        '--input-format',
        type=str,
        default='auto',
        choices=['auto', 'pdb', 'fcz', 'foldcomp-db'],
        help='Input format override (default: auto)',
    )
    parser.add_argument(
        '--foldcomp-ids',
        type=str,
        default=None,
        help='Foldcomp IDs as comma-separated list or path to text file (one ID per line)',
    )
    parser.add_argument(
        '--foldcomp-chunk-size',
        type=int,
        default=256,
        help='Chunk size for Foldcomp DB conversion (default: 256)',
    )
    parser.epilog = (
        'Example usage:\n'
        "  python encode_pdbs.py /path/to/pdbs output.h5\n"
        "  python encode_pdbs.py '/path/**/*.pdb' output.h5 /path/to/foldx\n"
        "  python encode_pdbs.py '/path/**/*.fcz' output.h5 --input-format fcz\n"
        "  python encode_pdbs.py /path/to/foldcomp_db_root output.h5 --input-format foldcomp-db"
    )
    return parser


def main(argv=None):
    argv = argv if argv is not None else sys.argv[1:]
    if '--about' in argv:
        print_about()
        return 0

    _configure_reproducibility()

    parser = build_parser()
    args = parser.parse_args(argv)

    mode = _resolve_input_mode(args.input_path, args.input_format)
    if args.verbose:
        print(f'Input mode: {mode}')

    # Keep legacy behavior for PDB mode, switch to mk2 for Foldcomp-compatible modes.
    if mode == 'pdb':
        converter = pdbgraph.PDB2PyG(aapropcsv=args.aapropcsv)
        files = _resolve_structure_files(args.input_path, mode='pdb')
        print(f'Found {len(files)} PDB files from {args.input_path}')
        np.random.shuffle(files)
        if args.nstructs is not None:
            files = files[:args.nstructs]

        if len(files) == 0:
            raise ValueError('No PDB files found for the provided input_path.')

        manifest_path = _checkpoint_manifest_path(args.output_h5) if args.checkpoint else None
        if manifest_path is not None:
            files = _filter_remaining_inputs(files, manifest_path)
            if args.verbose:
                print(f'Resuming from checkpoint: {len(files)} remaining of {len(files) + len(_load_checkpoint_state(manifest_path))} total')

        if len(files) == 0:
            if args.verbose:
                print('All files already processed according to checkpoint manifest.')
            return 0

        output_mode = 'a' if manifest_path and (os.path.exists(args.output_h5) or os.path.exists(manifest_path)) else 'w'
        if args.multiprocessing:
            converter.store_pyg_mp(
                files,
                filename=args.output_h5,
                foldxdir=args.foldxdir,
                verbose=args.verbose,
                add_prody=args.add_prody,
                ncpu=args.ncpu,
                checkpoint_manifest=manifest_path,
                output_mode=output_mode,
            )
        else:
            converter.store_pyg(
                files,
                filename=args.output_h5,
                foldxdir=args.foldxdir,
                verbose=args.verbose,
                add_prody=args.add_prody,
                distance=args.distance,
                checkpoint_manifest=manifest_path,
                output_mode=output_mode,
            )
        return 0

    converter = pdbgraphmk2.PDB2PyG(aapropcsv=args.aapropcsv)

    if args.foldxdir is not None and args.verbose:
        print('Warning: foldxdir is ignored for Foldcomp-compatible modes.')
    if args.add_prody and args.verbose:
        print('Warning: --add-prody is ignored for Foldcomp-compatible modes.')
    if args.distance != 15 and args.verbose:
        print('Warning: --distance is currently ignored for Foldcomp-compatible modes.')

    if mode == 'foldcomp-db':
        foldcomp_ids = _parse_foldcomp_ids(args.foldcomp_ids)
        failed = _store_foldcomp_db(
            converter=converter,
            input_path=args.input_path,
            output_h5=args.output_h5,
            ids=foldcomp_ids,
            max_structures=args.nstructs,
            verbose=args.verbose,
            chunk_size=args.foldcomp_chunk_size,
        )
        if args.verbose and failed:
            print(f'Failed structures: {len(failed)}')
        return 0

    files = _resolve_structure_files(args.input_path, mode='fcz')
    print(f'Found {len(files)} FCZ files from {args.input_path}')
    np.random.shuffle(files)
    if args.nstructs is not None:
        files = files[:args.nstructs]

    if len(files) == 0:
        raise ValueError('No FCZ files found for the provided input_path.')

    manifest_path = _checkpoint_manifest_path(args.output_h5) if args.checkpoint else None
    if manifest_path is not None:
        files = _filter_remaining_inputs(files, manifest_path)
        if args.verbose:
            print(f'Resuming from checkpoint: {len(files)} remaining of {len(files) + len(_load_checkpoint_state(manifest_path))} total')

    if len(files) == 0:
        if args.verbose:
            print('All files already processed according to checkpoint manifest.')
        return 0

    output_mode = 'a' if manifest_path and (os.path.exists(args.output_h5) or os.path.exists(manifest_path)) else 'w'
    if args.multiprocessing:
        failed = converter.store_pyg_mp_pool(
            files,
            filename=args.output_h5,
            ncpu=args.ncpu,
            verbose=args.verbose,
            checkpoint_manifest=manifest_path,
            output_mode=output_mode,
        )
    else:
        failed = converter.store_pyg(files, filename=args.output_h5, verbose=args.verbose, checkpoint_manifest=manifest_path, output_mode=output_mode)

    if args.verbose and failed:
        print(f'Failed structures: {len(failed)}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())