import foldtree_ecddcd as ft
import glob
import os
import torch

# command line arguments are an input directory with pdbs, a model file and an output directory

def init_model(model_file):
    pass

def encode_pdbs(input_dir, model_file, output_dir):
    pass


if __name__ == '__main__':
    import argparse

    #set device to gpu if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    parser = argparse.ArgumentParser(description='Encode pdbs')
    parser.add_argument('input_dir', type=str, help='Input directory with pdbs')
    parser.add_argument('input_glob', type=str, help='Input directory with pdbs')
    
    parser.add_argument('model_file', type=str, help='Model file')
    parser.add_argument('output_file', type=str, help='Output file with encoded pdbs')
    parser.add_argument('output_h5', type=str, help='Output file with pytorch geometric graphs of pdbs')
    parser.add_argument('output_format', type=str, default='fasta', help='Output format')

    args = parser.parse_args()

    if args.input_glob:
        files = glob.glob( args.input_glob )
    else:
        files = glob.glob(os.path.join(args.input_dir, '*.pdb'))
    
    
    #create h5 dataset

    #load model

    #create torch data loader

    #predict encoded

    #save encoded to output file
