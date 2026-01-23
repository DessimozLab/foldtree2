#!/usr/bin/env python3
"""
makesubmat.py - Generate Structure-Based Substitution Matrices

This tool creates custom substitution matrices for phylogenetic analysis based on 
protein structural alignments. It uses trained FoldTree2 models to encode protein 
structures into discrete sequences, then builds substitution matrices from the

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
	return char_set, char_position_map from structural 
alignments to capture evolutionary relationships at the structural level.

The workflow consists of several steps:
1. Download representative protein structures from AFDB clusters
2. Convert PDB files to graph representations suitable for neural network processing
3. Align structures using Foldseek to identify homologous regions
4. Encode aligned structures using trained FoldTree2 models into discrete alphabets
5. Compute substitution frequencies from structural alignments
6. Generate MAFFT-compatible and RAxML-compatible substitution matrices

These matrices can then be used for structure-based phylogenetic inference, providing
an alternative to sequence-based methods that incorporates 3D structural information.


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
import foldtree2.src.encoder as ft2

def print_about():
	ascii_art = r'''

+-----------------------------------------------------------+
|                         foldtree2                          |
|          Structure-Based Substitution Matrix Generator      |
|     AFDB reps • Foldseek align • Discrete alphabets → MAT   |
|          MAFFT + RAxML matrices for phylogenetic inference  |
|                      🧬   🧠   🌳                          |
+-----------------------------------------------------------+


	'''
	print(ascii_art)
	print("Structure-Based Substitution Matrix Generator")
	print("-" * 50)
	print("Generate custom substitution matrices from protein structural alignments")
	print("for phylogenetic inference with FoldTree2.\n")
	print("This tool creates matrices by:")
	print("  • Downloading AFDB cluster representatives")
	print("  • Aligning structures with Foldseek")
	print("  • Encoding structures to discrete alphabets")
	print("  • Computing substitution frequencies")
	print("  • Generating MAFFT and RAxML matrices\n")
	print("Project: https://github.com/DessimozLab/foldtree2")
	print("Contact: dmoi@unil.ch\n")
	print("Run with --help for usage instructions.")

def parse_args():
	parser = argparse.ArgumentParser(
		description="""
Generate Structure-Based Substitution Matrices for Phylogenetic Analysis

This tool creates custom substitution matrices by:
1. Downloading protein structures from AFDB clusters
2. Performing structural alignments using Foldseek
3. Encoding structures with trained FoldTree2 models
4. Computing substitution frequencies from alignments
5. Generating matrices compatible with MAFFT and RAxML

WORKFLOW STEPS:
- Use --download_structs to fetch representative structures
- Use --convert_to_pyg to prepare structures for neural network processing
- Use --align_structs to create structural alignments with Foldseek
- Use --encode_alns to encode structures using trained models
- Final matrices are automatically generated from the encoded alignments

EXAMPLE USAGE:
# Complete workflow with a trained model
makesubmat --modelname my_model --download_structs --convert_to_pyg --align_structs --encode_alns

# Generate matrices from existing data
makesubmat --modelname my_model --encode_alns
		""",
		formatter_class=argparse.RawDescriptionHelpFormatter
	)
	
	# Core parameters
	parser.add_argument('--about', action='store_true',
						help='Show information about this tool and exit')
	parser.add_argument('--modelname', type=str, default=None, required=True,
						help='Name of trained FoldTree2 model to use for encoding (without .pt extension)')
	parser.add_argument('--modeldir', type=str, default='models/', 
						help='Directory containing trained model .pt files (default: models/)')
	parser.add_argument('--datadir', type=str, default='../../datasets/', 
						help='Base data directory containing AFDB clusters and structure files')
	
	# Workflow control flags
	parser.add_argument('--download_structs', action='store_true', 
						help='Download representative protein structures from AFDB clusters')
	parser.add_argument('--convert_to_pyg', action='store_true', 
						help='Convert downloaded PDB files to PyTorch Geometric format for neural network processing')
	parser.add_argument('--align_structs', action='store_true', 
						help='Perform all-vs-all structural alignments using Foldseek')
	parser.add_argument('--encode_alns', action='store_true', 
						help='Encode aligned structures using the trained model into discrete sequences')
	
	# Structure download parameters
	parser.add_argument('--nstructs', type=int, default=5, 
						help='Number of structures to download per cluster representative (default: 5)')
	
	# Output control
	parser.add_argument('--plot', action='store_true', 
						help='Generate and display visualization plots of the matrices')
	parser.add_argument('--mafftmat', type=str, default=None, 
						help='Output filename for MAFFT-compatible matrix (default: MODELNAME_mafftmat.mtx)')
	parser.add_argument('--submat', type=str, default=None, 
						help='Output filename for RAxML-compatible substitution matrix (default: MODELNAME_submat.txt)')
	
	# Processing parameters
	parser.add_argument('--dataset', type=str, default='structalignmk4.h5', 
						help='HDF5 dataset filename for storing PyG-converted structures')
	parser.add_argument('--fident_thresh', type=float, default=0.3, 
						help='Sequence identity threshold for including alignment pairs in matrix computation (default: 0.3)')
	parser.add_argument('--rawcounts', action='store_true', 
						help='Output raw substitution counts instead of log-odds scores in MAFFT matrix')

	return parser.parse_args()

def ensure_dirs(outdir_base):
	"""
	Create necessary output directories for storing matrices and intermediate files.
	
	Args:
		outdir_base (str): Base directory for outputs
		
	Returns:
		str: Path to matrices subdirectory
	"""
	matdir = os.path.join(outdir_base, 'matrices')
	os.makedirs(outdir_base, exist_ok=True)
	os.makedirs(matdir, exist_ok=True)
	return matdir

def load_model(modeldir, modelname):
	"""
	Load a trained FoldTree2 encoder-decoder model from pickle file.
	
	Args:
		modeldir (str): Directory containing model files
		modelname (str): Name of model file (without .pkl extension)
		
	Returns:
		tuple: (encoder, decoder) model objects
	"""
	with open(os.path.join(modeldir, modelname + '.pkl'), 'rb') as f:
		encoder, decoder = pickle.load(f)
	return encoder, decoder

def read_reps(datadir):
	"""
	Read AFDB cluster representatives file containing protein IDs and taxonomic info.
	
	This function loads the AlphaFold Database cluster file that maps protein entries
	to their cluster representatives, which is used to identify structurally similar
	proteins for alignment and matrix generation.
	
	Args:
		datadir (str): Base data directory containing afdbclusters subdirectory
		
	Returns:
		pd.DataFrame: DataFrame with columns ['entryId', 'repId', 'taxId']
	"""
	# Check if reps file exists
	reps_file = os.path.join(datadir, 'afdbclusters/1-AFDBClusters-entryId_repId_taxId.tsv')
	if not os.path.exists(reps_file):
		print(f"Reps file {reps_file} from AFDB clusters not found. "
			  f"Please ensure the file exists.")
		sys.exit(1)
	
	# Read the reps file
	reps = pd.read_table(
		os.path.join(datadir, 'afdbclusters/1-AFDBClusters-entryId_repId_taxId.tsv'),
		header=None, names=['entryId', 'repId', 'taxId']
	)
	return reps

def download_structs_fn(reps, datadir, n=5):
	"""
	Download protein structures for cluster representatives and their members.
	
	This function downloads a random subset of structures for each cluster
	representative to create diverse training data for the substitution matrix.
	Structures are organized by cluster representative in separate directories.
	
	Args:
		reps (pd.DataFrame): DataFrame with protein cluster information
		datadir (str): Base directory for storing structures
		n (int): Maximum number of structures to download per cluster
	"""
	if AFDB_tools is None:
		print("AFDB_tools not available. Skipping download.")
		return
	
	# Process each cluster representative
	for rep in tqdm.tqdm(reps.repId.unique()):
		subdf = reps[reps['repId'] == rep]
		
		# Adjust sample size if cluster is smaller than requested
		if len(subdf) < n:
			n = len(subdf)
		
		# Sample random subset of proteins from this cluster
		subdf = subdf.sample(n=n)
		
		# Download structures for sampled proteins
		for uniID in subdf['entryId']:
			AFDB_tools.grab_struct(
				uniID, 
				structfolder=os.path.join(datadir, 'struct_align', rep, 'structs')
			)

def align_structs_fn(reps, datadir):
	"""
	Perform structural alignments for each cluster representative.
	
	This function runs FoldSeek all-vs-all structural alignment for each
	cluster's structures. The resulting alignments are used to identify
	structurally similar regions for substitution matrix computation.
	
	Args:
		reps (pd.DataFrame): DataFrame with cluster representative information
		datadir (str): Base directory containing structure alignment data
	"""
	for rep in tqdm.tqdm(reps.repId.unique()):
		foldseek2tree.runFoldseek_allvall_EZsearch(
			infolder=os.path.join(datadir, 'struct_align', rep, 'structs'),
			outpath=os.path.join(datadir, 'struct_align', rep, 'allvall.csv')
		)

def find_recursive_pdbs(folder):
	"""
	Recursively find all PDB structure files in a directory tree.
	
	This utility function searches through all subdirectories to locate
	protein structure files in various formats (PDB, ENT, compressed PDB).
	
	Args:
		folder (str): Root directory to search for structure files
		
	Returns:
		list: List of full paths to all found structure files
	"""
	# Find all pdb files in folder and subfolders
	pdbfiles = []
	for root, dirs, files in os.walk(folder):
		for file in files:
			if (file.endswith('.pdb') or file.endswith('.ent') or 
				file.endswith('.pdb.gz')):
				pdbfiles.append(os.path.join(root, file))
	return pdbfiles

def convert_to_pyg(dataset, out_h5, foldxdir=None):
	"""
	Convert PDB structure files to PyTorch Geometric format.
	
	This function converts protein structure files to PyG graph objects
	that can be processed by the neural network encoder. The resulting
	data is stored in HDF5 format for efficient loading.
	
	Args:
		dataset (str): Directory containing PDB structure files
		out_h5 (str): Output HDF5 file path for converted data
		foldxdir (str, optional): Directory containing FoldX energy data
	"""
	converter = PDB2PyG()
	pdbfiles = find_recursive_pdbs(dataset)
	print(f"Found {len(pdbfiles)} PDB files for conversion.")
	
	if len(pdbfiles) == 0:
		print("No PDB files found. Please check the dataset path.")
		sys.exit(1)
	
	converter.store_pyg(pdbfiles, filename=out_h5, foldxdir=foldxdir,
						verbose=False)

def encode_structures(encoder, modeldir, modelname, device, dataset):
	"""
	Encode protein structures using a trained neural network encoder.
	
	This function processes protein structures through the FoldTree2 encoder
	to generate discrete structural tokens. These tokens represent structural
	features and are used to compute structure-based substitution matrices.
	
	Args:
		encoder: Trained neural network encoder model
		modeldir (str): Directory containing model files
		modelname (str): Name of the model being used
		device: PyTorch device for computation
		dataset (str): Path to structure dataset (HDF5 file)
		
	Returns:
		str: Path to encoded FASTA file containing structural tokens
	"""
	from torch_geometric.data import DataLoader
	
	# Load existing dataset or convert PDB files
	if os.path.exists(os.path.join(dataset)):
		print(f"Using existing dataset at {dataset}")
		struct_dat = StructureDataset(dataset)
	else:
		# Convert PDBs to PyG format
		print(f"Converting PDB files in {os.path.dirname(dataset)} to PyG format...")
		convert_to_pyg(os.path.dirname(dataset), dataset)
		struct_dat = StructureDataset(dataset)
	
	print(f"Loaded {len(struct_dat)} structures from {dataset}")
	encoder_loader = DataLoader(struct_dat, batch_size=1, shuffle=False)
	
	def databatch2list(loader):
		"""Convert batched data to individual structures on device."""
		for data in loader:
			data = data.to_data_list()
			for d in data:
				d = d.to(device)
				yield d
	
	encoder_loader = databatch2list(encoder_loader)
	
	# Encode structures and save as FASTA
	output_path = os.path.join(modeldir, modelname + '_aln_encoded.fasta')
	encoder.encode_structures_fasta(encoder_loader, output_path, replace=True)
	print("Encoding complete. Encoded FASTA saved.")
	return output_path

def build_char_set(encoded_df):
	"""
	Build the set of all structural tokens in the encoded sequences.
	
	This function extracts all unique structural tokens from the encoded
	protein sequences and creates a mapping for matrix indexing. These tokens
	represent discrete structural states learned by the neural network encoder.
	
	Args:
		encoded_df (pd.DataFrame): DataFrame with 'seq' column containing 
								  encoded structural sequences
								  
	Returns:
		tuple: (char_set, char_position_map) where char_set is a sorted list
			   of unique tokens and char_position_map maps tokens to indices
	"""
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
	"""
	Compute pair counts and background frequencies from structural alignments.
	
	This function processes structural alignment files to count co-occurrences
	of structural tokens in aligned positions. These counts form the basis
	for calculating log-odds scores in the substitution matrix.
	
	Args:
		alnfiles (list): List of alignment file paths
		encoded_df (pd.DataFrame): DataFrame with encoded structural sequences
		char_set (list): List of unique structural tokens
		char_position_map (dict): Mapping from tokens to matrix indices
		fident_thresh (float): Minimum fractional identity threshold for alignments
		
	Returns:
		tuple: (submat, background_freq, seqcount) containing count matrix,
			   background frequencies, and total sequence count
	"""
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
	"""
	Compute log-odds substitution scores from observed pair counts.
	
	This function calculates log-odds scores by comparing observed substitution
	frequencies to expected frequencies under a null model of independent mutations.
	The log-odds scores quantify how much more (or less) likely a substitution is
	compared to random chance.
	
	Args:
		pair_counts (np.ndarray): Matrix of observed substitution counts
		char_freqs (np.ndarray): Background frequencies for each structural token
		pseudocount (float): Small value added to prevent division by zero (default: 1e-20)
		log_base (float): Base for logarithm calculation (default: np.e for natural log)
		
	Returns:
		np.ndarray: Log-odds substitution matrix
	"""
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
	"""
	Compute a time-reversible rate matrix compatible with RAxML format.
	
	This function converts pair counts into a time-reversible substitution rate
	matrix suitable for phylogenetic inference with RAxML. The matrix is symmetrized
	to ensure reversibility, normalized so rows sum to zero, and scaled so the
	expected substitution rate equals 1.
	
	Args:
		pair_counts (np.ndarray): Matrix of observed substitution counts
		char_freqs (np.ndarray): Background frequencies for each structural token
		raxml_charset (list): List of RAxML-compatible character symbols
		raxml_char_position_map (dict): Mapping from RAxML characters to indices
		pseudocount (float): Small value to prevent division by zero (default: 1e-20)
		log_base (float): Base for logarithm in log-odds calculation (default: np.e)
		scaling_factor (float): Factor for scaling rate matrix (default: 1.0)
		
	Returns:
		tuple: (rate_matrix, char_freqs) containing the normalized rate matrix
				and background character frequencies
	"""
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
	"""
	Write substitution matrix in MAFFT-compatible format.
	
	This function outputs the substitution matrix in the format expected by
	MAFFT alignment software. The matrix is written as upper triangular with
	hexadecimal character codes for structural tokens.
	
	Args:
		submat (np.ndarray): Substitution score matrix
		char_set (list): List of structural token characters
		char_position_map (dict): Mapping from characters to matrix indices
		outpath (str): Output file path for the MAFFT matrix
	"""
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

def output_raxml_matrix(matrix, background_frequencies, outpath):
	"""
	Write substitution matrix in RAxML-compatible format.
	
	This function outputs the rate matrix in the format required by RAxML
	phylogenetic inference software. The matrix is written as a lower triangular
	matrix followed by background character frequencies.
	
	Args:
		matrix (np.ndarray): Substitution rate matrix
		background_frequencies (np.ndarray): Background frequencies for each character
		outpath (str): Output file path for the RAxML matrix
		
	Returns:
		str: Path to the output file
	"""
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
	"""
	Main function orchestrating the structure-based substitution matrix generation.
	
	This function coordinates the entire workflow:
	1. Parse command-line arguments
	2. Load the trained neural network encoder
	3. Read protein cluster representatives
	4. Download and process protein structures
	5. Encode structures to structural tokens
	6. Compute substitution matrix from structural alignments
	7. Output matrices in MAFFT and RAxML formats
	"""
	if '--about' in sys.argv:
		print_about()
		sys.exit(0)
		
	args = parse_args()
	
	# Set default output paths if not provided
	if args.mafftmat is None:
		args.mafftmat = args.modelname + '_mafftmat.mtx'
	if args.submat is None:
		args.submat = args.modelname + '_submat.txt'
	if args.modelname is None:
		print("Error: --modelname must be specified.")
		sys.exit(1)
	
	# Load the trained encoder model
	model = os.path.join(args.modeldir, args.modelname)
	encoder = torch.load(model + '.pt', map_location=torch.device('cpu'),
						weights_only=False)
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