from foldtree2.src import pdbgraph
import os
import glob
import torch
import numpy as np
import argparse
import sys

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
    
# command line arguments are an input directory with pdbs, a model file and an output directory
if __name__ == '__main__':
    if '--about' in sys.argv:
        print_about()
        sys.exit(0)
    # Setting the seed for reproducibility
    torch.manual_seed(0)
    np.random.seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Initialize converter with config
    converter = pdbgraph.PDB2PyG(
        aapropcsv='./config/aaindex1.csv'
    )
    
    # Set device to gpu if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    parser = argparse.ArgumentParser(description='Encode pdbs')
    parser.add_argument('input_path', type=str, help='Input directory with pdbs or glob pattern (e.g., /path/to/pdbs or "/path/**/*.pdb")')
    parser.add_argument('output_h5', type=str, help='Output file with pytorch geometric graphs of pdbs')
    parser.add_argument('foldxdir', type=str, nargs='?', default=None, help='foldx directory with foldx output for all pdbs')
    parser.add_argument('--distance', type=float, default=15, help='Distance threshold for contact map (default: 15)')
    parser.add_argument('--add-prody', action='store_true', default=False, help='Add ProDy features (default: True)')
    parser.add_argument('--verbose', action='store_true', default=False, help='Verbose output')
    parser.add_argument('--multiprocessing', action='store_true', default=False, help='Use multiprocessing for parallel processing')
    parser.add_argument('--ncpu', type=int, default=25, help='Number of CPUs for multiprocessing (default: 25)')
    parser.add_argument('--nstructs', type=int, default=None, help='Number of structures to use (random subsample if specified)')
    
    # Add help for the arguments
    parser.description = "Encode PDB files into PyTorch geometric graphs with optional FoldX data integration."
    parser.epilog = ("Example usage:\n"
                     "  python encode_pdbs.py /path/to/pdbs output.h5\n"
                     "  python encode_pdbs.py '/path/**/*.pdb' output.h5 /path/to/foldx")

    args = parser.parse_args()
    
    # Handle input path - can be directory or glob pattern
    if os.path.isdir(args.input_path):
        # It's a directory, find all PDB files in it
        files = glob.glob(os.path.join(args.input_path, '*.pdb'))
        input_source = args.input_path
    else:
        # It's a glob pattern
        files = glob.glob(args.input_path, recursive=True)
        input_source = args.input_path
    
    print(f"Found {len(files)} PDB files from {input_source}")
    # Shuffle the data for randomization
    np.random.shuffle(files)
    
    # Subsample if nstructs is specified
    if args.nstructs is not None:
        files = files[:args.nstructs]
    
    output_h5 = args.output_h5
    foldx = args.foldxdir
    
    # Create h5 dataset with pytorch geometric graphs
    # Using the same parameters as in the notebook
    if args.multiprocessing:
        converter.store_pyg_mp(
            files, 
            filename=output_h5, 
            foldxdir=foldx, 
            verbose=args.verbose, 
            add_prody=args.add_prody,
            ncpu=args.ncpu
        )
    else:
        converter.store_pyg(
            files, 
            filename=output_h5, 
            foldxdir=foldx, 
            verbose=args.verbose, 
            add_prody=args.add_prody,
            distance=args.distance
        )