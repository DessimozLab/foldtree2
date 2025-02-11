from converter import pdbgraph
import os
import torch
import argparse
    
# command line arguments are an input directory with pdbs, a model file and an output directory
if __name__ == '__main__':
    converter = pdbgraph.PDB2PyG()import glob
    #set device to gpu if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    parser = argparse.ArgumentParser(description='Encode pdbs')
    parser.add_argument('input_dir', type=str, help='Input directory with pdbs')
    parser.add_argument('input_glob', type=str, help='Input directory with pdbs')    
    parser.add_argument('output_h5', type=str, help='Output file with pytorch geometric graphs of pdbs')
    parser.add_argument( 'foldxdir' , type = str , help = 'foldx directory with foldx output for all pdbs')
    
    #add help for the arguments
    parser.description = "Encode PDB files into PyTorch geometric graphs with optional FoldX data integration."
    parser.epilog = ("Example usage:\n"
                     "  python encode_pdbs.py /path/to/pdbs '*.pdb' output.h5 /path/to/foldx")




    args = parser.parse_args()
    
    if args.input_glob:
        files = glob.glob( args.input_glob )
    else:
        files = glob.glob(os.path.join(args.input_dir, '*.pdb'))
    output_h5 = args.output_h5
    if args.foldxdir:
        foldx = args.foldxdir
    else:
        foldx = None
    #create h5 dataset with pytorch geometric graphs
    converter.store_pyg(pdbfiles, filename= output_h5, foldxdir = foldx , verbose = False)