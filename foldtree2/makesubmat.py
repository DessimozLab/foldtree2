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
from foldtree2.src import AFDB_tools, foldseek2tree
from foldtree2.src.pdbgraph import PDB2PyG, StructureDataset
import foldtree2.src.foldtree2_ecddcd as ft2

def parse_args():
	parser = argparse.ArgumentParser(description="Generate substitution matrices and run alignments.")
	parser.add_argument('--modelname', type=str, default=None, help='Model name to load')
	parser.add_argument('--modeldir', type=str, default='models/', help='Directory containing model .pkl files')
	parser.add_argument('--datadir', type=str, default='../../datasets/', help='Data directory')
	parser.add_argument('--download_structs', action='store_true', help='Download structure members')
	parser.add_argument('--convert_to_pyg', action='store_true', help='Convert PDBs to PyG format')
	parser.add_argument('--nstructs', type=int, default=5, help='Number of structures to download per cluster representative')
	parser.add_argument('--align_structs', action='store_true', help='Align structures with foldseek')
	parser.add_argument('--encode_alns', action='store_true', help='Encode alignments')
	parser.add_argument('--plot', action='store_true', help='Show plots')
	parser.add_argument('--mafftmat', type=str, default=None, help='Output path for MAFFT matrix')
	parser.add_argument('--submat', type=str, default=None, help='Output path for RAxML substitution matrix')
	parser.add_argument('--dataset', type=str, default='structalignmk4.h5', help='Dataset name for structure encoding')
	parser.add_argument('--fident_thresh', type=float, default=0.3, help ='Identity threshold for pair counts')
	parser.add_argument('--rawcounts', action='store_true', help='Output raw pair counts to mafft matrix')

	return parser.parse_args()

def ensure_dirs(outdir_base):
	matdir = os.path.join(outdir_base, 'matrices')
	os.makedirs(outdir_base, exist_ok=True)
	os.makedirs(matdir, exist_ok=True)
	return matdir

def load_model(modeldir, modelname):
	with open(os.path.join(modeldir, modelname + '.pkl'), 'rb') as f:
		encoder, decoder = pickle.load(f)
	return encoder, decoder

def read_reps(datadir):
	#check if reps file exists
	reps_file = os.path.join(datadir, 'afdbclusters/1-AFDBClusters-entryId_repId_taxId.tsv')
	if not os.path.exists(reps_file):
		print(f"Reps file {reps_file} from AFDB clusters not found. Please ensure the file exists.")
		sys.exit(1)
	
	#read the reps file
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
	for rep in tqdm.tqdm(reps.repId.unique()):
		foldseek2tree.runFoldseek_allvall_EZsearch(
			infolder=os.path.join(datadir, 'struct_align', rep, 'structs'),
			outpath=os.path.join(datadir, 'struct_align', rep, 'allvall.csv')
		)

def find_recursive_pdbs(folder):
	#find all pdb files in folder and subfolders
	pdbfiles = []
	for root, dirs, files in os.walk(folder):
		for file in files:
			if file.endswith('.pdb') or file.endswith('.ent') or file.endswith('.pdb.gz'):
				pdbfiles.append(os.path.join(root, file))
	return pdbfiles

def convert_to_pyg(dataset, out_h5, foldxdir=None ):
	converter = PDB2PyG()
	pdbfiles = find_recursive_pdbs(dataset)
	print(f"Found {len(pdbfiles)} PDB files for conversion.")
	if len(pdbfiles) == 0:
		print("No PDB files found. Please check the dataset path.")
		sys.exit(1)
	converter.store_pyg(pdbfiles, filename=out_h5, foldxdir=foldxdir,  verbose=False)

def encode_structures(encoder, modeldir, modelname, device, dataset):
	from torch_geometric.data import DataLoader
	if os.path.exists(os.path.join( dataset )):
		print(f"Using existing dataset at {dataset}")
		struct_dat = StructureDataset(dataset)
	else:
		#convert pdbs to pyg
		print(f"Converting PDB files in {os.path.dirname(dataset)} to PyG format...")
		convert_to_pyg(os.path.dirname(dataset), dataset)
		struct_dat = StructureDataset(dataset)
	
	print(f"Loaded {len(struct_dat)} structures from {dataset}")
	encoder_loader = DataLoader(struct_dat, batch_size=1, shuffle=False)
	def databatch2list(loader):
		for data in loader:
			data = data.to_data_list()
			for d in data:
				d = d.to(device)
				yield d
	encoder_loader = databatch2list(encoder_loader)
	encoder.encode_structures_fasta(encoder_loader, os.path.join(modeldir, modelname + '_aln_encoded.fasta') , replace=True )
	print("Encoding complete. Encoded FASTA saved.")
	return os.path.join(modeldir, modelname + '_aln_encoded.fasta')

def build_char_set(encoded_df):
	"""Build the set of all characters in the encoded sequences."""
	char_set = set()
	for seq in encoded_df.seq:
		char_set = char_set.union(set(seq))
	char_set = list(char_set)
	char_set.sort()  # Sort to ensure consistent order
	print(f"Character set: {char_set}")
	print('ord', [ord(c) for c in char_set])
	print('hex', [hex(ord(c)) for c in char_set])
	print(f"Number of characters: {len(char_set)}")
	char_position_map = {char: i for i, char in enumerate(char_set)}
	print(f"Character position map: {char_position_map}")
	raxml_chars = """0 1 2 3 4 5 6 7 8 9 A B C D E F G H I J K L M N O P Q R S T U V W X Y Z ! " # $ % & ' ( ) * + , / : ; < = > @ [ \ ] ^ _ { | } ~""".split()
	raxml_charset = [ raxml_chars[char_position_map[c]] for c in char_set ]
	raxml_char_position_map = {c: i for i, c in enumerate(raxml_charset)}
	print(f"RAxML character set: {raxml_charset}")
	print(f"RAxML character position map: {raxml_char_position_map}")
	if len(raxml_charset) != len(char_set):
		print("Warning: RAxML character set length does not match original character set length.")
	# Ensure the character set is sorted and unique
	assert len(set(raxml_charset)) == len(raxml_charset), "RAxML character set contains duplicates."
	assert len(set(char_set)) == len(char_set), "Original character set contains duplicates."
	assert len(raxml_charset) == len(char_set), "RAxML character set length does not match original character set length."
	# Return both the character set and the position map
	return char_set, char_position_map , raxml_charset, raxml_char_position_map

def compute_pair_counts_and_bg(alnfiles, encoded_df, char_set, char_position_map, fident_thresh=0.3):
	"""Compute pair counts and background frequencies from alignments."""
	cols = 'query,target,fident,alnlen,mismatch,gapopen,qstart,qend,tstart,tend,evalue,bits,qaln,taln'.split(',')
	submat = np.zeros((len(char_set),len(char_set)))
	background_freq = np.zeros(len(char_set))
	seqcount = 0
	for rep in tqdm.tqdm(alnfiles, desc="Processing alignments"):
		submat_chunk = np.zeros((len(char_set),len(char_set)))
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
										qaln_ft2.append(char_position_map[next(qz_iter)])
								for t_char in taln.strip():
									if t_char == '-':
										taln_ft2.append(None)
									else:
										taln_ft2.append(char_position_map[next(tz_iter)])
								alnzip = [ [a, b] for a, b in zip(qaln_ft2, taln_ft2) if a is not None and b is not None ]
								alnzip = np.array(alnzip)
								if alnzip.size > 0:
									submat_chunk[alnzip[:,0], alnzip[:,1]] += 1
		submat += submat_chunk
	return submat, background_freq

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

def compute_raxml_compatible_matrix(pair_counts, char_freqs, raxml_charset, raxml_char_position_map, pseudocount=1e-20, log_base=np.e, scaling_factor=1.0):
	# Compute the log odds matrix as you already do
	log_odds_matrix = compute_log_odds_from_counts(pair_counts, char_freqs, pseudocount, log_base)
	
	# Exponentiate the log odds to get relative rates
	preliminary_rates = np.exp(log_odds_matrix * scaling_factor)
	
	# Symmetrize the matrix to ensure reversibility
	n = preliminary_rates.shape[0]
	rate_matrix = np.zeros_like(preliminary_rates)
	for i in range(n):
		for j in range(n):
			if i != j:
				rate_matrix[i, j] = (preliminary_rates[i, j] + preliminary_rates[j, i]) / 2.0
	# Set diagonal entries so that each row sums to zero
	for i in range(n):
		rate_matrix[i, i] = -np.sum(rate_matrix[i, :]) + rate_matrix[i, i]
	
	# Scale the matrix so that the expected substitution rate is 1
	# Calculate the expected rate: sum_i πᵢ * (-Qᵢᵢ)
	char_freqs = char_freqs / np.sum(char_freqs)
	expected_rate = -np.sum(char_freqs * np.diag(rate_matrix))
	rate_matrix = rate_matrix / expected_rate

	return rate_matrix, char_freqs

def output_mafft_matrix(submat, char_set, char_position_map, outpath):
	def formathex(hexnum):
		if len(hexnum) == 3:
			return hexnum[0:2] + '0' + hexnum[2]
		else:
			return hexnum
	reverse_char_map = {v: k for k, v in char_position_map.items()}
	with open(outpath, 'w') as f:
		for i in range(len(char_set)):
			for j in range(len(char_set)):
				if i <= j:
					stringi = reverse_char_map[i]
					stringj = reverse_char_map[j]
					hexi = formathex(hex(ord(stringi)))
					hexj = formathex(hex(ord(stringj)))
					f.write(f'{hexi} {hexj} {submat[i,j]}\n')

def output_raxml_matrix( matrix, background_frequencies, outpath):
	# Create the substitution matrix file
	#lower triangular matrix
	with open(outpath, "w") as f:
		for i in range(matrix.shape[0]):
			for j in range(matrix.shape[0]):
				if j < i:
					#format to 6 decimal places
					f.write(f" {matrix[i,j]:.6f}")
			f.write("\n")
		# Add the frequencies
		for i, freq in enumerate(background_frequencies):
			f.write(f"{freq:.6f} ")
		f.write("\n")
	return outpath


def main():
	args = parse_args()
	# Set default output paths if not provided
	if args.mafftmat is None:
		args.mafftmat = args.modelname + '_mafftmat.mtx'
	if args.submat is None:
		args.submat = args.modelname + '_submat.txt'
	if args.modelname is None:
		print("Error: --modelname must be specified.")
		sys.exit(1)
	model = os.path.join(args.modeldir, args.modelname)
	encoder = torch.load(model + '.pt', map_location=torch.device('cpu') , weights_only=False)
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	encoder = encoder.to(device)
	encoder.device = device

	encoder.eval()
	print(f"Using device: {device}")
	print(encoder)

	print(encoder.num_embeddings)
	outdir_base = args.modeldir
	matdir = ensure_dirs(outdir_base)

	print( ' creating matrices in', outdir_base)
	print('modelname', args.modelname)
	reps = None
	if args.download_structs and not os.path.exists(os.path.join(args.datadir, 'struct_align')):
		#make struct align directory
		os.makedirs(os.path.join(args.datadir, 'struct_align'), exist_ok=True)
		print("Downloading structure representatives...")
		reps = read_reps(args.datadir)
		print('reps', reps.head())
		download_structs_fn(reps, args.datadir)
	if args.convert_to_pyg:
		print("Converting PDB files to PyG format...")
		convert_to_pyg(os.path.join(args.datadir, 'struct_align'), args.dataset)
		print(f"Converted PDB files saved to {args.dataset}")
	if args.align_structs:
		if reps is None:
			reps = read_reps(args.datadir)
		align_structs_fn(reps, args.datadir)
	if not os.path.exists(os.path.join(args.datadir, 'struct_align')):
		print("No structure alignments found. Please run --download_structs and --align_structs first.")
		sys.exit(1)
	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
	encoder = encoder.to(device)
	encoder.device = device
	encoder.eval()
	print(f"Using device: {device}")
	if args.encode_alns:
		print("Encoding alignment structures...")
		encoded_fasta = encode_structures(encoder, args.modeldir, args.modelname, device, args.dataset)
	else:
		print("Skipping encoding of alignments, using existing encoded FASTA.")
		encoded_fasta = os.path.join(args.modeldir, args.modelname + '_aln_encoded.fasta')
	if not os.path.exists(encoded_fasta):
		print(f"Encoded FASTA file {encoded_fasta} not found. Please run encoding first.")
		sys.exit(1)
	encoded_df = ft2.load_encoded_fasta(encoded_fasta, alphabet=None, replace=False)
	char_set , char_position_map , raxml_charset, raxml_char_position_map = build_char_set(encoded_df)
	alnfiles = glob.glob(os.path.join(args.datadir, 'struct_align/*/allvall.csv'))
	print(f"Found {len(alnfiles)} alignment files.")
	if len(alnfiles) == 0:
		print("No alignment files found. Please run --align_structs first.")
		sys.exit(1)
	print(f"Processing {len(alnfiles)} alignment files...")
	pair_counts, background_freq = compute_pair_counts_and_bg(alnfiles, encoded_df, char_set , char_position_map, fident_thresh=args.fident_thresh)
	print(f"Pair counts shape: {pair_counts.shape}, Background frequencies shape: {background_freq.shape}")
	
	#save pair counts
	pair_counts_path = os.path.join(outdir_base, args.modelname + '_pair_counts.pkl')
	with open(pair_counts_path, 'wb') as f:
		pickle.dump((pair_counts, char_set, char_position_map , raxml_charset, raxml_char_position_map), f)
	print(f"Pair counts and char positions saved to {pair_counts_path}")

	# Compute log odds matrix
	print("Computing log odds matrix...")
	background_freq = background_freq / np.sum(background_freq)
	log_odds = compute_log_odds_from_counts(pair_counts, background_freq)
	# Save MAFFT matrix
	if args.mafftmat is None:
		args.mafftmat = args.modelname + '_mafftmat.mtx'
	if args.submat is None:
		args.submat = args.modelname + '_submat.txt'

	#save charmap 
	print("Outputting matrices...")
	# Save MAFFT matrix
	mafftmat_path = os.path.join(outdir_base, args.mafftmat)
	if args.rawcounts:
		print("Outputting raw pair counts to MAFFT matrix...")
		output_mafft_matrix(pair_counts, char_set, char_position_map, mafftmat_path)
	else:
		print("Outputting log odds matrix to MAFFT matrix...")
		output_mafft_matrix(log_odds, char_set, char_position_map, mafftmat_path)
	print(f"MAFFT matrix written to {mafftmat_path}")
	# Save RAxML matrix
	raxmlmat_path = os.path.join(outdir_base, args.submat)

	# Compute RAxML-compatible matrix
	raxml_matrix, char_freqs = compute_raxml_compatible_matrix(pair_counts, background_freq , raxml_charset, raxml_char_position_map, scaling_factor=1.0)
	# Output RAxML matrix
	assert len(raxml_charset) == len(char_set), "RAxML character set length mismatch"
	output_raxml_matrix(raxml_matrix, char_freqs, raxmlmat_path)
	
	print(f"RAxML matrix written to {raxmlmat_path}")

if __name__ == "__main__":
	main()