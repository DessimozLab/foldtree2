#!/usr/bin/env python3
"""
Command-line script version of makesubmat.ipynb
"""

import sys
import os
import argparse
import pickle
import glob
import pandas as pd
import numpy as np
import tqdm
import torch
import importlib
from matplotlib import pyplot as plt

# Optional: import custom modules if available
from src import AFDB_tools, foldseek2tree
from src.pdbgraph import PDB2PyG, StructureDataset
import src.foldtree2_ecddcd as ft2


def parse_args():
    parser = argparse.ArgumentParser(description="Generate substitution matrices and run alignments.")
    parser.add_argument('--modelname', type=str, default='small5_geo_graph', help='Model name to load')
    parser.add_argument('--modeldir', type=str, default='models/', help='Directory containing model .pkl files')
    parser.add_argument('--datadir', type=str, default='../../datasets/', help='Data directory')
    parser.add_argument('--outdir_base', type=str, default='../../results/foldtree2/', help='Base output directory')
    parser.add_argument('--download_structs', action='store_true', help='Download structure members')
    parser.add_argument('--nstructs', type=int, default=5, help='Number of structures to download per cluster representative')
    parser.add_argument('--align_structs', action='store_true', help='Align structures with foldseek')
    parser.add_argument('--encode_alns', action='store_true', help='Encode alignments')
    parser.add_argument('--plot', action='store_true', help='Show plots')
    parser.add_argument('--mafftmat', type=str, default=None, help='Output path for MAFFT matrix')
    parser.add_argument('--submat', type=str, default=None, help='Output path for RAxML substitution matrix')
    parser.add_argument('--dataset', type=str, default='structalignmk4.h5', help='Dataset name for structure encoding')
    parser.add_argument('--fident_thresh', type=float, default=0.3, help ='Identity threshold for pair counts')
    return parser.parse_args()

def ensure_dirs(outdir_base):
    matdir = os.path.join(outdir_base, 'matrices')
    treedir = os.path.join(outdir_base, 'trees')
    os.makedirs(outdir_base, exist_ok=True)
    os.makedirs(matdir, exist_ok=True)
    os.makedirs(treedir, exist_ok=True)
    return matdir, treedir


def load_model(modeldir, modelname):
    with open(os.path.join(modeldir, modelname + '.pkl'), 'rb') as f:
        encoder, decoder = pickle.load(f)
    return encoder, decoder


def read_reps(datadir):
    reps = pd.read_table(os.path.join(datadir, 'afdbclusters/1-AFDBClusters-entryId_repId_taxId.tsv'),
                        header=None, names=['entryId', 'repId', 'taxId'])
    return reps

def download_structs_fn(reps, datadir, n=5):
    if AFDB_tools is None:
        print("AFDB_tools not available. Skipping download.")
        return
    for rep in tqdm.tqdm(reps.repId.unique()):
        subdf = reps[reps['repId'] == rep]
        if len(subdf) < n:
            n = len(subdf)
        subdf = subdf.sample(n=n)
        for uniID in subdf['entryId']:
            AFDB_tools.grab_struct(uniID, structfolder=os.path.join(datadir, 'struct_align', rep, 'structs'))

def align_structs_fn(reps, datadir):
    if foldseek2tree is None:
        print("foldseek2tree not available. Skipping alignment.")
        return
    for rep in tqdm.tqdm(reps.repId.unique()):
        foldseek2tree.runFoldseek_allvall_EZsearch(
            infolder=os.path.join(datadir, 'struct_align', rep, 'structs'),
            outpath=os.path.join(datadir, 'struct_align', rep, 'allvall.csv')
        )

def encode_structures(encoder, modeldir, modelname, device, dataset):
    from torch_geometric.data import DataLoader
    struct_dat = StructureDataset(dataset)
    encoder_loader = DataLoader(struct_dat, batch_size=1, shuffle=False)
    def databatch2list(loader):
        for data in loader:
            data = data.to_data_list()
            for d in data:
                d = d.to(device)
                yield d
    encoder_loader = databatch2list(encoder_loader)
    encoder.encode_structures_fasta(encoder_loader, modeldir + modelname + '_aln_encoded.fasta')

def parse_encoded_fasta(fasta_path):
    """Parse encoded FASTA and return a DataFrame with sequences."""
    import tqdm
    seqdict = {}
    seqstr = ''
    ID = ''
    with open(fasta_path, 'r') as f:
        for line in tqdm.tqdm(f, desc="Parsing encoded FASTA"):
            if line.startswith('>'):
                if ID:
                    seqdict[ID] = seqstr
                ID = line[1:].strip()
                seqstr = ''
            else:
                seqstr += line.strip()
        if ID:
            seqdict[ID] = seqstr
    encoded_df = pd.DataFrame(seqdict.items(), columns=['protid', 'seq'])
    encoded_df.index = encoded_df.protid
    encoded_df = encoded_df.drop('protid', axis=1)
    return encoded_df

def build_char_set(encoded_df):
    """Build the set of all characters in the encoded sequences."""
    char_set = set()
    for seq in encoded_df.seq:
        char_set = char_set.union(set(seq))
    char_set = list(char_set)
    char_set.sort()
    return char_set

def compute_pair_counts_and_bg(alnfiles, encoded_df, char_set, fident_thresh=0.3):
    """Compute pair counts and background frequencies from alignments."""
    cols = 'query,target,fident,alnlen,mismatch,gapopen,qstart,qend,tstart,tend,evalue,bits,qaln,taln'.split(',')
    submat = np.zeros((256,256))
    background_freq = np.zeros(len(char_set))
    seqcount = 0
    for rep in tqdm.tqdm(alnfiles, desc="Processing alignments"):
        submat_chunk = np.zeros((256,256))
        aln_df = pd.read_table(rep)
        aln_df.columns = cols
        seqset = set()
        for q in aln_df['query'].unique():
            for t in aln_df['target'].unique():
                if q != t:
                    aln = aln_df[(aln_df['query'] == q) & (aln_df['target'] == t)]
                    if len(aln) > 0 and aln.fident.iloc[0] < fident_thresh:
                        aln = aln.iloc[0]
                        qaln = aln.qaln
                        taln = aln.taln
                        qaccession = q.split('.')[0]
                        taccession = t.split('.')[0]
                        if qaccession in encoded_df.index and taccession in encoded_df.index:
                            qz = str(encoded_df.loc[qaccession].seq[aln.qstart-1:aln.qend])
                            tz = str(encoded_df.loc[taccession].seq[aln.tstart-1:aln.tend])
                            if qaccession not in seqset:
                                background_freq += np.array([qz.count(c) for c in char_set])
                                seqset.add(qaccession)
                                seqcount += len(qz)
                            if taccession not in seqset:
                                background_freq += np.array([tz.count(c) for c in char_set])
                                seqset.add(taccession)
                                seqcount += len(tz)
                            if len(qz) == len(qaln.replace('-','')) and len(tz) == len(taln.replace('-','')):
                                qz_iter = iter(qz)
                                tz_iter = iter(tz)
                                qaln_ft2, taln_ft2 = [], []
                                for q_char in qaln:
                                    if q_char == '-':
                                        qaln_ft2.append(None)
                                    else:
                                        qaln_ft2.append(ord(next(qz_iter)))
                                for t_char in taln.strip():
                                    if t_char == '-':
                                        taln_ft2.append(None)
                                    else:
                                        taln_ft2.append(ord(next(tz_iter)))
                                alnzip = [ [a, b] for a, b in zip(qaln_ft2, taln_ft2) if a is not None and b is not None ]
                                alnzip = np.array(alnzip)
                                if alnzip.size > 0:
                                    submat_chunk[alnzip[:,0], alnzip[:,1]] += 1
        submat += submat_chunk
    return submat, background_freq, char_set

def compute_log_odds_from_counts(pair_counts, char_freqs, pseudocount=1e-20, log_base=np.e):
    n = pair_counts.shape[0]
    total_pairs = np.sum(pair_counts)
    obs_freq = (pair_counts + pseudocount) / (total_pairs + pseudocount * (n**2))
    char_freqs = char_freqs / np.sum(char_freqs)
    exp_freq = np.outer(char_freqs, char_freqs) + pseudocount
    ratio = obs_freq / exp_freq
    epsilon = 1e-15
    log_odds_matrix = np.log(ratio + epsilon) / np.log(log_base)
    return log_odds_matrix

def output_mafft_matrix(submat, char_set, outpath):
    def formathex(hexnum):
        if len(hexnum) == 3:
            return hexnum[0:2] + '0' + hexnum[2]
        else:
            return hexnum
    with open(outpath, 'w') as f:
        for i in range(len(char_set)):
            for j in range(len(char_set)):
                if i <= j:
                    stringi = char_set[i]
                    stringj = char_set[j]
                    hexi = formathex(hex(ord(stringi)))
                    hexj = formathex(hex(ord(stringj)))
                    f.write(f'{hexi} {hexj} {submat[i,j]}\n')

def output_raxml_matrix(log_odds, char_set, outpath):
    """Output a RAxML substitution matrix file."""
    with open(outpath, 'w') as f:
        f.write('x  ' + ' '.join(char_set) + '\n')
        for i, c1 in enumerate(char_set):
            row = [c1]
            for j, c2 in enumerate(char_set):
                val = log_odds[i, j]
                row.append(f"{val:.4f}")
            f.write('  '.join(row) + '\n')

def main():
    args = parse_args()
    # Set default output paths if not provided
    if args.mafftmat is None:
        args.mafftmat = args.modelname + '_mafftmat.mtx'
    if args.submat is None:
        args.submat = args.modelname + '_submat.txt'
    
    matdir, treedir = ensure_dirs(args.outdir_base)
    
    encoder, decoder = load_model(args.modeldir, args.modelname)
    print(encoder)
    print(decoder)

    print(encoder.num_embeddings)
    print( ' creating matrices in', matdir)
    print('modelname', args.modelname)

    reps = read_reps(args.datadir)
    print('reps', reps.head())

    if args.download_structs:
        download_structs_fn(reps, args.datadir)
    if args.align_structs:
        align_structs_fn(reps, args.datadir)

    if not os.path.exists(os.path.join(args.datadir, 'struct_align')):
        print("No structure alignments found. Please run --download_structs and --align_structs first.")
        sys.exit(1)
            
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    encoder = encoder.to(device)
    decoder = decoder.to(device)
    encoder.eval()
    decoder.eval()
    print(f"Using device: {device}")

    if args.encode_alns:
        encode_structures(encoder, args.modeldir, args.modelname, device, args.dataset)
    else:
        print("Skipping encoding of alignments, using existing encoded FASTA.")
        encoded_fasta = args.modeldir + args.modelname + '_aln_encoded.fasta'
    if not os.path.exists(encoded_fasta):
        print(f"Encoded FASTA file {encoded_fasta} not found. Please run encoding first.")
        sys.exit(1)

    encoded_df = parse_encoded_fasta(encoded_fasta)
    char_set = build_char_set(encoded_df)
    alnfiles = glob.glob(os.path.join(args.datadir, 'struct_align/*/allvall.csv'))
    pair_counts, background_freq, char_set = compute_pair_counts_and_bg(alnfiles, encoded_df, char_set)
    print(f"Pair counts shape: {pair_counts.shape}, Background frequencies shape: {background_freq.shape}")
    #save pair counts
    pair_counts_path = os.path.join(args.outdir_base, 'matrices', args.modelname + '_pair_counts.pkl')
    with open(pair_counts_path, 'wb') as f:
        pickle.dump((pair_counts, char_set), f)
    print(f"Pair counts saved to {pair_counts_path}")
    # Compute log odds matrix
    print("Computing log odds matrix...")
    background_freq = background_freq / np.sum(background_freq)
    log_odds = compute_log_odds_from_counts(pair_counts, background_freq)
    # Save MAFFT matrix
    mafftmat_path = os.path.join(args.outdir_base, 'matrices', args.mafftmat)
    output_mafft_matrix(pair_counts, char_set, mafftmat_path)
    print(f"MAFFT matrix written to {mafftmat_path}")
    # Save RAxML matrix
    raxmlmat_path = os.path.join(args.outdir_base, 'matrices', args.submat)
    output_raxml_matrix(log_odds, char_set, raxmlmat_path)
    print(f"RAxML matrix written to {raxmlmat_path}")

if __name__ == "__main__":
    main()