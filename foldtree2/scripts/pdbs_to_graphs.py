import os
import argparse
import glob
import h5py
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from foldtree2.src.pdbgraph import PDB2PyG

def find_pdbs_recursive(root_dir):
    pdb_files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            if fname.lower().endswith('.pdb'):
                pdb_files.append(os.path.join(dirpath, fname))
    return pdb_files

def main():
    parser = argparse.ArgumentParser(description="Convert PDB files to graph HDF5 dataset using PDB2PyG.")
    parser.add_argument("input_dir", help="Root directory to search for PDB files recursively")
    parser.add_argument("output_h5", help="Output HDF5 filename")
    parser.add_argument("--aapropcsv", default="config/aaindex1.csv", help="Amino acid property CSV")
    parser.add_argument("--mp", action="store_true", help="Use multiprocessing")
    parser.add_argument("--ncpu", type=int, default=4, help="Number of CPUs for multiprocessing")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    args = parser.parse_args()

    pdb_files = find_pdbs_recursive(args.input_dir)
    if args.verbose:
        print(f"Found {len(pdb_files)} PDB files.")

    converter = PDB2PyG(aapropcsv=args.aapropcsv)
    if args.mp:
        converter.store_pyg_mp(pdb_files, args.output_h5, verbose=args.verbose, ncpu=args.ncpu)
    else:
        converter.store_pyg(pdb_files, args.output_h5, verbose=args.verbose)

if __name__ == "__main__":
    main()
