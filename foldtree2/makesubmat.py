#!/usr/bin/env python3
"""
makesubmat.py - Generate Structure-Based Substitution Matrices

This tool creates custom substitution matrices for phylogenetic analysis based on 
protein structural alignments. It uses trained FoldTree2 models to encode protein 
structures into discrete sequences, then builds substitution matrices from structural 
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
import json
import pandas as pd
import numpy as np
import tqdm
import torch
import importlib
from matplotlib import pyplot as plt

# Optional: import custom modules if available
from foldtree2.src import AFDB_tools, foldseek2tree
from foldtree2.src.pdbgraph import PDB2PyG, StructureDataset
from foldtree2.src.download_utils import download_structures, verify_downloads
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
	parser.add_argument('--show-plots', action='store_true',
						help='Display generated plots interactively (useful in notebooks/GUI sessions)')
	parser.add_argument('--mafftmat', type=str, default=None, 
						help='Output filename for MAFFT-compatible matrix (default: MODELNAME_mafftmat.mtx)')
	parser.add_argument('--submat', type=str, default=None, 
						help='Output filename for RAxML-compatible substitution matrix (default: MODELNAME_submat.txt)')
	parser.add_argument('--convergence-plot-path', type=str, default=None,
						help='Output path for convergence plot PNG (default: MODELNAME_convergence.png in modeldir)')
	parser.add_argument('--final-matrices-plot-path', type=str, default=None,
						help='Output path for final matrices summary PNG (default: MODELNAME_final_matrices.png in modeldir)')
	parser.add_argument('--evolution-plot-path', type=str, default=None,
						help='Output path for matrix evolution PNG (default: MODELNAME_evolution_analysis.png in modeldir)')
	parser.add_argument('--metrics-json', type=str, default=None,
						help='Output path for convergence/statistics JSON (default: MODELNAME_metrics.json in modeldir)')
	parser.add_argument('--save-history', action='store_true',
						help='Store convergence history snapshots in the pair-counts pickle file')
	
	# Processing parameters
	parser.add_argument('--dataset', type=str, default='structalignmk4.h5', 
						help='HDF5 dataset filename for storing PyG-converted structures')
	parser.add_argument('--fident_thresh', type=float, default=0.3, 
						help='Sequence identity threshold for including alignment pairs in matrix computation (default: 0.3)')
	parser.add_argument('--monitor-convergence', action='store_true',
						help='Track notebook-style convergence metrics during alignment processing')
	parser.add_argument('--update-interval', type=int, default=5,
						help='Update interval (in alignment files) for convergence snapshots (default: 5)')
	parser.add_argument('--aln-limit', type=int, default=None,
						help='Optional limit on number of alignment files to process (default: all)')
	parser.add_argument('--convergence-threshold', type=float, default=0.01,
						help='Gradient-norm threshold used to label convergence (default: 0.01)')
	parser.add_argument('--live-plot', action='store_true',
						help='Render live convergence plots while processing alignments')
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

def download_structs_fn(reps, datadir, structdir=None, ncpu=20, timeout=30):
	"""
	Download protein structures for cluster representatives using robust multiprocessing.
	
	This function downloads structures for each cluster representative using the
	download_structures utility, which handles multiprocessing safely with proper
	pickling of subprocess calls instead of file objects.
	
	Structures are organized by cluster representative in separate directories
	for compatibility with downstream alignment steps.
	
	Args:
		reps (pd.DataFrame): DataFrame with protein cluster information (must have 'repId' column)
		datadir (str): Base directory for storing structures
		structdir (str): Optional override for output directory structure
		ncpu (int): Number of parallel processes for downloads
		timeout (int): Timeout per download in seconds
	"""
	if structdir is None:
		structdir = os.path.join(datadir, 'struct_align')
	
	# Ensure base directory exists
	os.makedirs(structdir, exist_ok=True)
	
	# Use the robust download_structures function
	print(f"Downloading {len(reps)} structures using multiprocessing...")
	successful, failed = download_structures(
		reps,
		nreps=None,  # Download all provided reps
		structdir=structdir,
		ncpu=ncpu,
		method='subprocess',
		timeout=timeout,
		verbose=False
	)
	
	print(f"\nDownload complete:")
	print(f"  Successfully downloaded: {len(successful)} structures")
	if failed:
		print(f"  Failed to download: {len(failed)} structures")
		if len(failed) <= 10:
			print(f"  Failed IDs: {failed}")
	
	return successful, failed

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
	raxml_chars = """0 1 2 3 4 5 6 7 8 9 A B C D E F G H I J K L M N O P Q R S T U V W X Y Z ! " # $ % & ' ( ) * + , / : ; < = > @ [ \\ ] ^ _ { | } ~""".split()
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

def compute_pair_counts_and_bg(
		alnfiles,
		encoded_df,
		char_set,
		char_position_map,
		fident_thresh=0.3,
		update_interval=5,
		aln_limit=None,
		monitor=None,
		live_plot=False,
		show_plots=False,
		plot_figsize=(18, 12),
		return_stats=False,
	):
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
		tuple: (submat, background_freq) by default. If return_stats=True,
			   returns (submat, background_freq, stats_dict).
	"""
	cols = 'query,target,fident,alnlen,mismatch,gapopen,qstart,qend,tstart,tend,evalue,bits,qaln,taln'.split(',')
	submat = np.zeros((len(char_set), len(char_set)))
	background_freq = np.zeros(len(char_set))
	seqcount = 0
	all_processed_seqs = set()
	files_processed = 0
	total_files_to_process = len(alnfiles) if aln_limit is None else min(len(alnfiles), aln_limit)

	aln_iter = tqdm.tqdm(alnfiles, desc="Processing alignments")
	for file_idx, rep in enumerate(aln_iter):
		if aln_limit is not None and file_idx >= aln_limit:
			print(f"Reached alignment file limit of {aln_limit}. Stopping processing.")
			break

		submat_chunk = np.zeros((len(char_set), len(char_set)))
		try:
			aln_df = pd.read_table(rep)
			aln_df.columns = cols
		except Exception as exc:
			print(f"Warning: Could not read {rep}: {exc}")
			continue

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
							qz = str(encoded_df.loc[qaccession].seq[aln.qstart - 1:aln.qend])
							tz = str(encoded_df.loc[taccession].seq[aln.tstart - 1:aln.tend])

							# Notebook behavior: count background frequencies once per accession globally.
							if qaccession not in all_processed_seqs:
								background_freq += np.array([qz.count(c) for c in char_set])
								all_processed_seqs.add(qaccession)
								seqcount += len(qz)
							if taccession not in all_processed_seqs:
								background_freq += np.array([tz.count(c) for c in char_set])
								all_processed_seqs.add(taccession)
								seqcount += len(tz)

							if len(qz) == len(qaln.replace('-', '')) and len(tz) == len(taln.replace('-', '')):
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
								alnzip = [[a, b] for a, b in zip(qaln_ft2, taln_ft2) if a is not None and b is not None]
								alnzip = np.array(alnzip)
								if alnzip.size > 0:
									submat_chunk[alnzip[:, 0], alnzip[:, 1]] += 1

		submat += submat_chunk
		files_processed += 1

		if monitor is not None and update_interval > 0:
			is_update_step = (files_processed % update_interval == 0)
			is_last_step = (files_processed == total_files_to_process)
			if is_update_step or is_last_step:
				if np.sum(background_freq) > 0 and np.sum(submat) > 0:
					bg_norm = background_freq / np.sum(background_freq)
					current_log_odds = compute_log_odds_from_counts(submat, bg_norm)
				else:
					current_log_odds = np.zeros_like(submat)
				monitor.update(files_processed, current_log_odds)
				if live_plot:
					fig = monitor.plot_convergence(figsize=plot_figsize)
					if show_plots:
						plt.show()
					plt.close(fig)

	stats = {
		'files_processed': files_processed,
		'processed_sequences': len(all_processed_seqs),
		'total_sequence_positions': int(seqcount),
	}
	if return_stats:
		return submat, background_freq, stats
	return submat, background_freq


class MatrixConvergenceMonitor:
	"""Track and visualize matrix convergence during iterative compilation."""

	def __init__(self, matrix_size, convergence_threshold=0.01):
		self.matrix_size = matrix_size
		self.convergence_threshold = convergence_threshold
		self.history = {
			'iteration': [],
			'frobenius_norm': [],
			'gradient_norm': [],
			'max_change': [],
			'mean_change': [],
			'nonzero_elements': [],
			'snapshots': [],
		}
		self.prev_matrix = None

	def update(self, iteration, current_matrix):
		"""Update convergence metrics with a new matrix snapshot."""
		frob_norm = float(np.linalg.norm(current_matrix, 'fro'))
		if self.prev_matrix is not None:
			gradient = current_matrix - self.prev_matrix
			grad_norm = float(np.linalg.norm(gradient, 'fro'))
			max_change = float(np.max(np.abs(gradient)))
			mean_change = float(np.mean(np.abs(gradient)))
		else:
			grad_norm = 0.0
			max_change = 0.0
			mean_change = 0.0

		nonzero = int(np.count_nonzero(current_matrix))

		self.history['iteration'].append(int(iteration))
		self.history['frobenius_norm'].append(frob_norm)
		self.history['gradient_norm'].append(grad_norm)
		self.history['max_change'].append(max_change)
		self.history['mean_change'].append(mean_change)
		self.history['nonzero_elements'].append(nonzero)
		self.history['snapshots'].append(current_matrix.copy())

		self.prev_matrix = current_matrix.copy()

	def plot_convergence(self, figsize=(18, 12)):
		"""Generate notebook-style convergence visualization."""
		fig, axes = plt.subplots(3, 3, figsize=figsize)
		iterations = self.history['iteration']

		if len(iterations) == 0:
			axes[1, 1].text(0.5, 0.5, 'No convergence snapshots yet', ha='center', va='center')
			for ax in axes.ravel():
				ax.axis('off')
			return fig

		axes[0, 0].plot(iterations, self.history['frobenius_norm'], 'b-', linewidth=2)
		axes[0, 0].set_xlabel('Iteration (Alignment Files)')
		axes[0, 0].set_ylabel('Frobenius Norm')
		axes[0, 0].set_title('Matrix Magnitude Over Time')
		axes[0, 0].grid(True, alpha=0.3)

		axes[0, 1].plot(iterations, self.history['gradient_norm'], 'r-', linewidth=2)
		axes[0, 1].set_xlabel('Iteration (Alignment Files)')
		axes[0, 1].set_ylabel('Gradient Norm')
		axes[0, 1].set_title('Rate of Change (Matrix Gradient)')
		axes[0, 1].grid(True, alpha=0.3)
		axes[0, 1].set_yscale('log')

		axes[0, 2].plot(iterations, self.history['max_change'], 'g-', label='Max', linewidth=2)
		axes[0, 2].plot(iterations, self.history['mean_change'], 'orange', label='Mean', linewidth=2)
		axes[0, 2].set_xlabel('Iteration (Alignment Files)')
		axes[0, 2].set_ylabel('Change Magnitude')
		axes[0, 2].set_title('Element-wise Changes')
		axes[0, 2].legend()
		axes[0, 2].grid(True, alpha=0.3)
		axes[0, 2].set_yscale('log')

		im0 = axes[1, 0].imshow(self.history['snapshots'][0], cmap='RdBu_r', aspect='auto', interpolation='nearest')
		axes[1, 0].set_title(f'Iteration {iterations[0]} (First)')
		plt.colorbar(im0, ax=axes[1, 0], fraction=0.046)

		mid_idx = len(self.history['snapshots']) // 2
		im1 = axes[1, 1].imshow(self.history['snapshots'][mid_idx], cmap='RdBu_r', aspect='auto', interpolation='nearest')
		axes[1, 1].set_title(f'Iteration {iterations[mid_idx]} (Middle)')
		plt.colorbar(im1, ax=axes[1, 1], fraction=0.046)

		im2 = axes[1, 2].imshow(self.history['snapshots'][-1], cmap='RdBu_r', aspect='auto', interpolation='nearest')
		axes[1, 2].set_title(f'Iteration {iterations[-1]} (Current)')
		plt.colorbar(im2, ax=axes[1, 2], fraction=0.046)

		axes[2, 0].plot(iterations, self.history['nonzero_elements'], color='purple', linewidth=2)
		axes[2, 0].set_xlabel('Iteration (Alignment Files)')
		axes[2, 0].set_ylabel('Count')
		axes[2, 0].set_title('Non-zero Matrix Elements')
		axes[2, 0].grid(True, alpha=0.3)

		if len(iterations) > 2:
			grad_norms = np.array(self.history['gradient_norm'])
			convergence_rate = np.diff(grad_norms)
			axes[2, 1].plot(iterations[1:], convergence_rate, 'c-', linewidth=2)
			axes[2, 1].set_xlabel('Iteration (Alignment Files)')
			axes[2, 1].set_ylabel('Delta Gradient Norm')
			axes[2, 1].set_title('Convergence Acceleration')
			axes[2, 1].grid(True, alpha=0.3)
			axes[2, 1].axhline(y=0, color='k', linestyle='--', alpha=0.5)
		else:
			axes[2, 1].axis('off')

		summary = self.get_convergence_summary()
		stats_text = (
			"Matrix Convergence Summary\n\n"
			f"Final Iteration: {iterations[-1]}\n"
			f"Final Frobenius Norm: {summary['final_frobenius_norm']:.4f}\n"
			f"Final Gradient Norm: {summary['final_gradient_norm']:.6f}\n"
			f"Mean Gradient Norm: {summary['mean_gradient_norm']:.6f}\n"
			f"Non-zero Elements: {self.history['nonzero_elements'][-1]} / {self.matrix_size**2}\n"
			f"Sparsity: {100.0 * summary['sparsity']:.2f}%\n\n"
			"Convergence Status:\n"
			f"{'CONVERGED' if summary['is_converged'] else 'STILL CHANGING'}"
		)
		axes[2, 2].text(0.02, 0.5, stats_text, fontsize=10, verticalalignment='center', family='monospace')
		axes[2, 2].axis('off')

		plt.tight_layout()
		return fig

	def get_convergence_summary(self):
		"""Return summary statistics about convergence."""
		if len(self.history['iteration']) == 0:
			return {
				'total_iterations': 0,
				'final_frobenius_norm': 0.0,
				'final_gradient_norm': 0.0,
				'mean_gradient_norm': 0.0,
				'max_gradient_norm': 0.0,
				'is_converged': False,
				'sparsity': 1.0,
			}

		grad_series = self.history['gradient_norm']
		mean_grad = float(np.mean(grad_series[1:])) if len(grad_series) > 1 else 0.0
		sparsity = 1.0 - self.history['nonzero_elements'][-1] / float(self.matrix_size ** 2)
		return {
			'total_iterations': len(self.history['iteration']),
			'final_frobenius_norm': float(self.history['frobenius_norm'][-1]),
			'final_gradient_norm': float(self.history['gradient_norm'][-1]),
			'mean_gradient_norm': mean_grad,
			'max_gradient_norm': float(np.max(self.history['gradient_norm'])),
			'is_converged': bool(self.history['gradient_norm'][-1] < self.convergence_threshold),
			'sparsity': float(sparsity),
		}


def save_figure(fig, outpath, show_plots=False):
	"""Save matplotlib figure and optionally display it."""
	if outpath is not None:
		outdir = os.path.dirname(outpath)
		if outdir:
			os.makedirs(outdir, exist_ok=True)
		fig.savefig(outpath, dpi=150, bbox_inches='tight')
		print(f"Saved plot to: {outpath}")
	if show_plots:
		plt.show()
	plt.close(fig)


def plot_final_matrices(pair_counts, log_odds_matrix, background_freq, outpath=None, show_plots=False):
	"""Generate notebook-style final matrix summary figure."""
	matrix_size = pair_counts.shape[0]
	fig, axes = plt.subplots(2, 2, figsize=(16, 14))

	im0 = axes[0, 0].imshow(pair_counts, cmap='viridis', aspect='auto', interpolation='nearest')
	axes[0, 0].set_title('Raw Pair Counts', fontsize=14, fontweight='bold')
	axes[0, 0].set_xlabel('Character Index')
	axes[0, 0].set_ylabel('Character Index')
	plt.colorbar(im0, ax=axes[0, 0], fraction=0.046)

	im1 = axes[0, 1].imshow(log_odds_matrix, cmap='RdBu_r', aspect='auto', interpolation='nearest', vmin=-2, vmax=2)
	axes[0, 1].set_title('Log-Odds Substitution Matrix', fontsize=14, fontweight='bold')
	axes[0, 1].set_xlabel('Character Index')
	axes[0, 1].set_ylabel('Character Index')
	plt.colorbar(im1, ax=axes[0, 1], fraction=0.046)

	axes[1, 0].bar(range(len(background_freq)), background_freq, color='steelblue', alpha=0.7)
	axes[1, 0].set_xlabel('Character Index')
	axes[1, 0].set_ylabel('Frequency')
	axes[1, 0].set_title('Background Character Frequencies', fontsize=14, fontweight='bold')
	axes[1, 0].grid(True, alpha=0.3, axis='y')

	diagonal = np.diag(log_odds_matrix)
	off_diagonal = log_odds_matrix[~np.eye(matrix_size, dtype=bool)]
	axes[1, 1].hist(diagonal, bins=30, alpha=0.7, label='Diagonal (same char)', color='green')
	axes[1, 1].hist(off_diagonal, bins=30, alpha=0.7, label='Off-diagonal (different char)', color='red')
	axes[1, 1].set_xlabel('Log-Odds Score')
	axes[1, 1].set_ylabel('Frequency')
	axes[1, 1].set_title('Distribution of Substitution Scores', fontsize=14, fontweight='bold')
	axes[1, 1].legend()
	axes[1, 1].grid(True, alpha=0.3, axis='y')

	plt.tight_layout()
	save_figure(fig, outpath, show_plots=show_plots)


def plot_evolution_analysis(monitor, matrix_size, outpath=None, show_plots=False):
	"""Generate notebook-style matrix evolution analysis plot."""
	if monitor is None or len(monitor.history['snapshots']) == 0:
		print("Skipping evolution analysis plot: no convergence snapshots available.")
		return

	fig, axes = plt.subplots(2, 2, figsize=(16, 10))
	iterations = monitor.history['iteration']

	default_positions = [
		(0, 0),
		(0, 1 if matrix_size > 1 else 0),
		(matrix_size // 2, matrix_size // 2),
		(min(5, matrix_size - 1), min(8, matrix_size - 1)),
	]

	for pos in default_positions:
		values = [snapshot[pos] for snapshot in monitor.history['snapshots']]
		label = f"[{pos[0]},{pos[1]}]{'(diag)' if pos[0] == pos[1] else ''}"
		axes[0, 0].plot(iterations, values, marker='o', linewidth=2, label=label)
	axes[0, 0].set_xlabel('Iteration (Alignment Files)')
	axes[0, 0].set_ylabel('Log-Odds Score')
	axes[0, 0].set_title('Evolution of Sample Matrix Elements', fontweight='bold')
	axes[0, 0].legend()
	axes[0, 0].grid(True, alpha=0.3)

	snapshot_array = np.array(monitor.history['snapshots'])
	element_variance = np.var(snapshot_array, axis=0)
	im = axes[0, 1].imshow(element_variance, cmap='hot', aspect='auto', interpolation='nearest')
	axes[0, 1].set_title('Element-wise Variance Across Iterations', fontweight='bold')
	axes[0, 1].set_xlabel('Character Index')
	axes[0, 1].set_ylabel('Character Index')
	plt.colorbar(im, ax=axes[0, 1], fraction=0.046)

	mean_abs_changes = []
	for i in range(1, len(monitor.history['snapshots'])):
		change = np.abs(monitor.history['snapshots'][i] - monitor.history['snapshots'][i - 1])
		mean_abs_changes.append(np.mean(change))

	if len(mean_abs_changes) > 0:
		axes[1, 0].plot(iterations[1:], mean_abs_changes, color='purple', linewidth=2, marker='o')
		axes[1, 0].set_xlabel('Iteration (Alignment Files)')
		axes[1, 0].set_ylabel('Mean Absolute Change')
		axes[1, 0].set_title('Average Element Change Per Iteration', fontweight='bold')
		axes[1, 0].set_yscale('log')
		axes[1, 0].grid(True, alpha=0.3)
	else:
		axes[1, 0].axis('off')

	if len(monitor.history['snapshots']) > 1:
		cumulative_change = np.abs(monitor.history['snapshots'][-1] - monitor.history['snapshots'][0])
		im = axes[1, 1].imshow(cumulative_change, cmap='plasma', aspect='auto', interpolation='nearest')
		axes[1, 1].set_title('Total Change from First to Last Iteration', fontweight='bold')
		axes[1, 1].set_xlabel('Character Index')
		axes[1, 1].set_ylabel('Character Index')
		plt.colorbar(im, ax=axes[1, 1], fraction=0.046)
	else:
		axes[1, 1].axis('off')

	plt.tight_layout()
	save_figure(fig, outpath, show_plots=show_plots)


def build_metrics_payload(
		pair_counts,
		background_freq,
		log_odds,
		monitor,
		matrix_size,
		processing_stats,
		fident_thresh,
	):
	"""Create a JSON-serializable metrics payload mirroring notebook summaries."""
	diagonal = np.diag(log_odds)
	off_diagonal = log_odds[~np.eye(matrix_size, dtype=bool)]
	payload = {
		'fident_threshold': float(fident_thresh),
		'matrix_size': int(matrix_size),
		'total_pair_counts': float(np.sum(pair_counts)),
		'nonzero_pairs': int(np.count_nonzero(pair_counts)),
		'sparsity': float(1.0 - (np.count_nonzero(pair_counts) / float(matrix_size ** 2))),
		'log_odds_min': float(np.min(log_odds)),
		'log_odds_max': float(np.max(log_odds)),
		'mean_diagonal_score': float(np.mean(diagonal)),
		'mean_off_diagonal_score': float(np.mean(off_diagonal)),
		'std_diagonal_score': float(np.std(diagonal)),
		'std_off_diagonal_score': float(np.std(off_diagonal)),
		'background_sum': float(np.sum(background_freq)),
		'processing': processing_stats,
	}
	if monitor is not None:
		payload['convergence'] = monitor.get_convergence_summary()
	else:
		payload['convergence'] = None
	return payload

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

	if args.plot and not args.monitor_convergence:
		args.monitor_convergence = True
	if args.live_plot and not args.monitor_convergence:
		args.monitor_convergence = True
	if args.update_interval < 1:
		print("Warning: --update-interval must be >= 1. Falling back to 1.")
		args.update_interval = 1
	if args.aln_limit is not None and args.aln_limit <= 0:
		print("Warning: --aln-limit must be > 0. Ignoring limit.")
		args.aln_limit = None

	safe_model_label = args.modelname.replace('/', '_')
	
	# Set default output paths if not provided
	if args.mafftmat is None:
		args.mafftmat = safe_model_label + '_mafftmat.mtx'
	if args.submat is None:
		args.submat = safe_model_label + '_submat.txt'
	if args.convergence_plot_path is None:
		args.convergence_plot_path = os.path.join(args.modeldir, safe_model_label + '_convergence.png')
	if args.final_matrices_plot_path is None:
		args.final_matrices_plot_path = os.path.join(args.modeldir, safe_model_label + '_final_matrices.png')
	if args.evolution_plot_path is None:
		args.evolution_plot_path = os.path.join(args.modeldir, safe_model_label + '_evolution_analysis.png')
	if args.metrics_json is None and (args.plot or args.monitor_convergence):
		args.metrics_json = os.path.join(args.modeldir, safe_model_label + '_metrics.json')
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
		successful, failed = download_structs_fn(reps, args.datadir)
		print(f"Download summary: {len(successful)} successful, {len(failed)} failed")
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

	max_files = len(alnfiles) if args.aln_limit is None else min(len(alnfiles), args.aln_limit)
	print(f"Processing up to {max_files} alignment files...")
	monitor = None
	if args.monitor_convergence:
		monitor = MatrixConvergenceMonitor(len(char_set), convergence_threshold=args.convergence_threshold)

	pair_counts, background_freq, processing_stats = compute_pair_counts_and_bg(
		alnfiles,
		encoded_df,
		char_set,
		char_position_map,
		fident_thresh=args.fident_thresh,
		update_interval=args.update_interval,
		aln_limit=args.aln_limit,
		monitor=monitor,
		live_plot=args.live_plot,
		show_plots=args.show_plots,
		return_stats=True,
	)
	print(f"Pair counts shape: {pair_counts.shape}, Background frequencies shape: {background_freq.shape}")
	print(f"Files processed: {processing_stats['files_processed']}")
	print(f"Processed sequences: {processing_stats['processed_sequences']}")
	print(f"Total sequence positions: {processing_stats['total_sequence_positions']}")
	
	#save pair counts
	pair_counts_path = os.path.join(outdir_base, safe_model_label + '_pair_counts.pkl')
	with open(pair_counts_path, 'wb') as f:
		pickle.dump((pair_counts, char_set, char_position_map , raxml_charset, raxml_char_position_map), f)
	print(f"Pair counts and char positions saved to {pair_counts_path}")

	if args.save_history and monitor is not None:
		history_path = os.path.join(outdir_base, safe_model_label + '_pair_counts_history.pkl')
		with open(history_path, 'wb') as f:
			pickle.dump({
				'pair_counts': pair_counts,
				'char_set': char_set,
				'char_position_map': char_position_map,
				'raxml_charset': raxml_charset,
				'raxml_char_position_map': raxml_char_position_map,
				'background_freq': background_freq,
				'convergence_history': monitor.history,
				'processing_stats': processing_stats,
			}, f)
		print(f"Pair counts + convergence history saved to {history_path}")

	# Compute log odds matrix
	print("Computing log odds matrix...")
	if np.sum(background_freq) <= 0:
		print("Background frequencies are all zero. Cannot compute log-odds matrix.")
		sys.exit(1)
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

	# Optional notebook-style plots and metrics outputs
	if monitor is not None and len(monitor.history['iteration']) > 0:
		fig = monitor.plot_convergence(figsize=(18, 12))
		save_figure(fig, args.convergence_plot_path, show_plots=args.show_plots)

	if args.plot:
		plot_final_matrices(
			pair_counts=pair_counts,
			log_odds_matrix=log_odds,
			background_freq=background_freq,
			outpath=args.final_matrices_plot_path,
			show_plots=args.show_plots,
		)
		plot_evolution_analysis(
			monitor=monitor,
			matrix_size=len(char_set),
			outpath=args.evolution_plot_path,
			show_plots=args.show_plots,
		)

	metrics_payload = build_metrics_payload(
		pair_counts=pair_counts,
		background_freq=background_freq,
		log_odds=log_odds,
		monitor=monitor,
		matrix_size=len(char_set),
		processing_stats=processing_stats,
		fident_thresh=args.fident_thresh,
	)

	if args.metrics_json is not None:
		metrics_dir = os.path.dirname(args.metrics_json)
		if metrics_dir:
			os.makedirs(metrics_dir, exist_ok=True)
		with open(args.metrics_json, 'w') as f:
			json.dump(metrics_payload, f, indent=2)
		print(f"Metrics JSON written to {args.metrics_json}")

	print("\nSummary metrics:")
	print(f"  Total pair counts: {metrics_payload['total_pair_counts']:.0f}")
	print(f"  Non-zero pairs: {metrics_payload['nonzero_pairs']}")
	print(f"  Sparsity: {metrics_payload['sparsity'] * 100:.2f}%")
	print(f"  Log-odds range: [{metrics_payload['log_odds_min']:.3f}, {metrics_payload['log_odds_max']:.3f}]")
	if metrics_payload['convergence'] is not None:
		print(f"  Final gradient norm: {metrics_payload['convergence']['final_gradient_norm']:.6f}")
		print(f"  Converged: {metrics_payload['convergence']['is_converged']}")

if __name__ == "__main__":
	main()