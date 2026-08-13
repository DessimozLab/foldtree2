import pickle
import torch
import glob
import subprocess
import Bio.PDB as PDB
#import torch_geometric hetero data
import torch_geometric
import multiprocessing as mp
import argparse
import numpy as np

from foldtree2.src import encoder as ecdr
from foldtree2.src import mono_decoders
from foldtree2.src.pdbgraphmk2 import PDB2PyG
from foldtree2.src.config_paths import resolve_aapropcsv_path
from torch_geometric.data import HeteroData

import traceback
import tqdm
import pandas as pd
import os
import ete3
from scipy import sparse
import sys


class treebuilder():
	def __init__ ( self , model , decoder_model = None, mafftmat = None , submat = None ,  raxml_path= None, charmaps=None, **kwargs ):
		"""Initialize a FoldTree2 tree builder with encoder, decoder, and tool paths.

		Parameters
		----------
		model : str
			Path to the serialized encoder checkpoint.
		decoder_model : str, optional
			Path to a serialized decoder checkpoint used for ancestral decoding.
		mafftmat : str, optional
			Path to the MAFFT text substitution matrix.
		submat : str, optional
			Path to the RAxML substitution matrix.
		raxml_path : str, optional
			Path to the raxml-ng executable.
		charmaps : str, optional
			Path to a pickled character mapping bundle.
		**kwargs : dict
			Optional runtime flags and paths (device, ncores, converters, etc.).
		"""

		#make fasta is shifted by 1 and goes from 1-248 included
		#0x01 – 0xFF excluding > (0x3E), = (0x3D), < (0x3C), - (0x2D), Space (0x20), Carriage Return (0x0d) and Line Feed (0x0a)
		#replace 0x22 or " which is necesary for nexus files and 0x23 or # which is also necesary		
		self.replace_dict = {chr(0):chr(246) , '"':chr(248) , '#':chr(247), '>' : chr(249), '=' : chr(250), '<' : chr(251), '-' : chr(252), ' ' : chr(253) , '\r' : chr(254), '\n' : chr(255) }
		self.rev_replace_dict = { v:k for k,v in self.replace_dict.items() }
		self.replace_dict_ord = { ord(k):ord(v) for k,v in self.replace_dict.items() }
		self.raxml_path = raxml_path
		self.raxmlng_path = raxml_path
		self.modelname = model.split('/')[-1].split('.')[0]
		self.model = model
		self.encoder = torch.load(model , map_location=torch.device('cpu') , weights_only=False)
		self.decoder = torch.load( decoder_model , map_location=torch.device('cpu') , weights_only=False ) if decoder_model is not None else None
		self.root = kwargs.get('root', False)
		self.overwrite = kwargs.get('overwrite', False)
		self.verbosity = self._normalize_verbosity(kwargs.get('verbosity', kwargs.get('verbose', 1)))

		if 'bs' in kwargs:
			self.bs = kwargs['bs']
		else:
			self.bs = False
		if 'redo' in kwargs:
			self.redo = kwargs['redo']
		else:
			self.redo = False

		if charmaps is None:
			self.rev_replace_dict_ord = { ord(v):ord(k) for k,v in self.replace_dict.items() }
			self.raxml_path = raxml_path
			#raxml alphabet
			self.raxmlchars = """0 1 2 3 4 5 6 7 8 9 A B C D E F G H I J K L M N O P Q R S T U V W X Y Z ! " # $ % & ' ( ) * + , / : ; < = > @ [ \ ] ^ _ { | } ~"""
			self.raxmlchars = self.raxmlchars.split()
			self.raxml_indices = {i:s for i,s in enumerate( self.raxmlchars ) }
			self.alphabet = [ chr(c+1) if chr(c+1) not in self.replace_dict else self.replace_dict[chr(c+1)] for c in range(self.encoder.num_embeddings) ]
			self.alphabet.sort()
			assert len(self.alphabet) == self.encoder.num_embeddings, f"Alphabet length {len(self.alphabet)} does not match num_embeddings {self.encoder.num_embeddings}"
			self.nchars = len(self.alphabet)
			self.map = { c:i for i,c in enumerate(self.alphabet)}
			self.revmap = { i:c for i,c in enumerate(self.alphabet)}

		else:
			self.log(f'loading charmaps from {charmaps}', level=2)
			
			'''
		save format is {
		'pair_counts': pair_counts,
		'char_set': char_set,
		'char_position_map': char_position_map,
		'raxml_charset': raxml_charset,
		'raxml_char_position_map': raxml_char_position_map,
		'background_freq': background_freq,
		'convergence_history': monitor.history
		
		}'''
			
			with open(charmaps, 'rb') as f:
				data = pickle.load(f)
			self.raxml_characters = data['raxml_charset']
			self.alphabet = data['char_set']	
			self.nchars = len(self.alphabet)
			self.map = data['char_position_map']
			self.revmap = { v:k for k,v in data['char_position_map'].items() }
			self.raxml_indices = data['raxml_char_position_map']
			self.rev_raxml_indices = { v:k for k,v in data['raxml_char_position_map'].items() }
			self.revmap_raxml = self.raxml_indices
			self.raxmlchars = data['raxml_charset']
		
		self.ordset = set([ ord(c) for c in self.alphabet ])
		self.aapropcsv = resolve_aapropcsv_path(kwargs.get('aapropcsv'))
		self.converter = PDB2PyG(aapropcsv=self.aapropcsv)

		#detect if we are using a GPU
		if 'device' in kwargs and kwargs['device'] is not None:
			self.device = torch.device(kwargs['device'])
		else:
			self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
			self.encoder = self.encoder.to(self.device)
			self.encoder.device = self.device
			if self.decoder is not None:
				self.decoder = self.decoder.to(self.device)
				self.decoder.device = self.device
		
		self.encoder.eval()
		if self.decoder is not None:
			self.decoder.eval()
		

		#load the mafftmat and submat matrices
		#if mafftmat == None or submat == None:
		#	raise ValueError('Need to provide mafftmat and submat')
		self.mafftmat = mafftmat
		self.submat = submat
		if 'maffttext2hex' in kwargs:
			self.maffttext2hex = kwargs['maffttext2hex']
		else:
			self.maffttext2hex = 'maffttext2hex'

		if 'maffthex2text' in kwargs:
			self.maffthex2text = kwargs['maffthex2text']
		else:
			self.maffthex2text = 'hex2maffttext'
		
		if 'ncores' in kwargs:
			self.ncores = kwargs['ncores']
		else:
			self.ncores = mp.cpu_count()

	@staticmethod
	def _normalize_verbosity(value):
		"""Normalize verbosity inputs to supported integer levels 0, 1, or 2."""
		if isinstance(value, bool):
			return 2 if value else 1
		if value is None:
			return 1
		try:
			level = int(value)
		except (TypeError, ValueError):
			level = 1
		return max(0, min(2, level))

	def log(self, message, level=1):
		"""Print a message when the current verbosity is high enough."""
		if self.verbosity >= level:
			print(message)

	def error(self, message, include_traceback=False):
		"""Always print important errors; traceback is shown only at verbosity 2."""
		print(message)
		if include_traceback and self.verbosity >= 2:
			print(traceback.format_exc())

	@staticmethod
	def formathex(hexnum):
			"""Normalize a hex token to MAFFT expected width.

			Parameters
			----------
			hexnum : str
				Hex string as returned by hex().

			Returns
			-------
			str
				Zero-padded 2-byte hex representation.
			"""
			if len(hexnum) == 3:
				return hexnum[0:2] + '0' + hexnum[2]
			else:
				return hexnum

	def run_mafft_textaln( self, infasta , outaln=None , matrix='mafft_submat.mtx' , mafft_path = 'mafft' ):
		"""Run MAFFT in text mode using a custom matrix and return output alignment path."""
		if outaln == None:
			outaln = infasta+'aln.txt'
		cmd = f'{mafft_path} --text --thread -1 --localpair --maxiterate 1000 --textmatrix {matrix} {infasta}  > {outaln}'
		self.log(cmd, level=2)
		subprocess.run(cmd, shell=True)
		return outaln

	def mafft_hex2ascii( self, intext , outfile , hex2text_path = './mafft_tools/hex2maffttext' ):
		"""Convert MAFFT hex-encoded alignment text back to ASCII symbols."""
		if outfile == None:

			outfile = intext.replace('.hex' , '.ASCII')
			self.log(f'outfile for ascii : {outfile}', level=2)
		#% /usr/local/libexec/mafft/hex2maffttext input.hex > input.ASCII
		cmd = f'{hex2text_path} {intext} > {outfile}'
		self.log(cmd, level=2)
		subprocess.run(cmd, shell=True)
		return outfile    

	def fasta2hex( self, intext , outfile  , maffttext2hex = './mafft_tools/maffttext2hex' ):
		"""Convert ASCII alignment text to MAFFT-compatible hex encoding."""
		#% /usr/local/libexec/mafft/maffttext2hex input.hex > input.ASCII
		if outfile == None:
			outfile = intext+'.hex'
		cmd = f'{maffttext2hex} {intext} > {outfile}'
		self.log(cmd, level=2)
		subprocess.run(cmd, shell=True)
		return outfile    

	def normal_mafft( self, infasta , outaln ):
		"""Run MAFFT with generic symbol support and return output alignment path."""
		cmd = f'mafft --anysymbol {infasta} > {outaln}'
		self.log(cmd, level=2)
		subprocess.run(cmd, shell=True)
		return outaln

	@staticmethod
	def struct2sequence(structfile):
		"""Extract a residue-name sequence from a PDB structure file.

		Returns the concatenated 3-letter residue names of amino acids.
		"""
		parser = PDB.PDBParser()
		structure = parser.get_structure('struct', structfile)
		seq = ''
		for model in structure:
			for chain in model:
				for residue in chain:
					if PDB.is_aa(residue):
						seq += residue.get_resname()
		return seq

	def struct_loader( self, structlist , converter):
		"""Yield PyG graph objects converted from an iterable of structure paths.

		Invalid structures are skipped with traceback logging.
		"""
		self.log('converting structures', level=1)
		for struct in tqdm.tqdm(structlist , desc='Converting structures', disable=self.verbosity == 0):
			try:
				data = self.converter.struct2pyg( struct )
				if self.device is not None:
					data = data.to(self.device)
				if data:
					yield data
			except:
				self.error(f'error {struct}', include_traceback=True)
				continue
	@staticmethod
	def aln2dcict(alnfile):
		"""Parse a FASTA-like alignment file into a dictionary keyed by sequence id."""
		with open( alnfile , 'r') as f:
			seqdict = {}
			seqstr = ''
			ID = ''
			for line in f:
				if line[0] == '>':
					seqdict[ID] = seqstr
					ID = line[1:].strip()
					seqstr = ''
				else:
					seqstr += line.strip()
			seqdict[ID] = seqstr
		return seqdict

	def encode_structblob(self , blob = None , outfile = None ):
		"""Encode a collection of PDB structures into a discrete FASTA representation.

		The input can be a glob pattern or a directory path ending with '/'.
		"""
		if blob[-1] == '/':
			structs = glob.glob(blob + '*.pdb')
		else:
			structs = glob.glob(blob)

		#you need at least 4 structs for a tree
		assert len(structs) >= 4, f'Need at least 4 structures to build a tree, found {len(structs)}'
		if outfile == None:
			outfolder = '/'.join( blob.split('/')[:-1] )
			outfile = outfolder + 'encoded.fasta'
		loader = self.struct_loader( structs , self.converter )
		self.encoder.encode_structures_fasta( loader , outfile , replace = True)
		return outfile

	def encode_structblob_raxml(self , blob = None , outfile = None ):
		"""Encode structures and remap encoded symbols to a RAxML-safe alphabet.

		This routine is intended for generating RAxML-ready FASTA files directly.
		"""
		#encode the structure into a fasta file that can be used by raxml
		#used for recoding foldmason alignments
		#otherwise use encode_structblob and align with mafft

		if blob[-1] == '/':
			structs = glob.glob(blob + '*.pdb')
		else:
			structs = glob.glob(blob)
		if outfile == None:
			outfolder = '/'.join( blob.split('/')[:-1] )
			outfile = outfolder + 'encoded.fasta'
		loader = self.struct_loader( structs , self.converter )
		outfile = self.encoder.encode_structures_fasta( loader , outfile)
		seqdict = self.aln2dcict( outfile )
		
		alndf = pd.DataFrame( seqdict.items() , columns=['protid', 'aln'] )
		alndf.index = alndf.protid
		alndf.drop( 'protid' , axis = 1 , inplace = True)
		alndf.drop( ''  , inplace = True)
		
		alndf['ord_aln'] = alndf.aln.map( lambda x: [ ord(c) if c!='-' else '-' for c in x.split() ] )
		alndf['seq_aln'] = alndf.ord_aln.map( lambda x: ''.join([ chr(c) if c !='-' else '-' for c in x ]) )	
		alndf['remap_int'] = alndf.seq_aln.map(lambda x : [ self.map[c] if c in self.map else '-' for c in x ] )
		alndf['remap_symbols'] = alndf['remap_int'].map( lambda x : ''.join([ self.raxml_indices[c] if c in self.raxml_indices else '-' for c in x ]) )
		#check that remap symbols only contains characters in raxml_indices
		remap_set = set(alndf['remap_symbols'].values.flatten())
		with open(outfile, 'w') as f:
			for i in alndf.index:
				f.write('>' + i + '\n' + alndf.loc[i].remap_symbols + '\n')
		return outfile
	
	def recode_aln( self, alnfile , encoded_fasta , outfile = None ):
		"""Attempt to recode an alignment using symbols from an encoded FASTA.

		Notes
		-----
		This method is currently experimental/incomplete and is not used by the
		main tree-building pipeline.
		"""
		#recode the alignment to the alphabet used by the model
		seqdict = self.aln2dcict( alnfile )
		alndf = pd.DataFrame( seqdict.items() , columns=['protid', 'aln'] )
		seqdict = self.aln2dcict( encoded_fasta )
		encoded_df = pd.DataFrame( seqdict.items() , columns=['protid', 'seq'] )
		#merge on protid
		alndf = alndf.merge( encoded_df , on='protid' , how='inner')

		if len( qz) == len( aln.replace('-','') ) and len( tz) == len( seq.replace('-','') ):
			qz = iter(qz)
			tz = iter(tz)

			#transfer the alignments to the embeddings                    
			qaln_ft2, taln_ft2 = [], []

			for q_char in qaln:
				if q_char == '-':
					qaln_ft2.append(None)
				else:
					qaln_ft2.append(ord(next(qz)))
					
			for t_char in taln.strip():
				if t_char == '-':
					taln_ft2.append(None)
				else:
					taln_ft2.append(ord(next(tz)))


	def replace_sp_chars(self, encoded_fasta, outfile = None  , verbose = False):
		"""Replace problematic special symbols in encoded FASTA sequences.

		The replacement map ensures downstream tools that expect restricted
		alphabets (for example MAFFT/RAxML/NEXUS workflows) can parse sequences.
		"""
		if outfile == None:
			outfile = encoded_fasta.replace('.fasta' , '_replaced.fasta')
		#load the encoded fasta
		with open(encoded_fasta) as encoded:
			seqstr = '' 
			ID = ''
			seqdict = {}
			for line in encoded:
				if line and line[0] == '>' and line[-1] == '\n':
					seqdict[ID] = seqstr
					ID = line[1:].strip()
					seqstr = ''
				else:
					seqstr += line.strip()
			if ID:
				seqdict[ID] = seqstr
			seqdict.pop('', None)
			encoded_df = pd.DataFrame( seqdict.items() , columns=['protid', 'seq'] )
		#replace the characters that aren't allowed
		encoded_df.seq = encoded_df.seq.map(lambda x : ''.join([ c if c not in self.replace_dict else self.replace_dict[c] for c in x]))
		encoded_df['ord'] = encoded_df.seq.map( lambda x: [ ord(c) for c in x] )
		if self._normalize_verbosity(verbose) >= 2 or self.verbosity >= 2:
			print(encoded_df.head())
		#write output to fasta
		with open( outfile, 'w') as f:
			for idx, row in encoded_df.iterrows():
				f.write('>' + row.protid + '\n' + row.seq + '\n')
		return outfile
	
	def encodedfasta2hex(self , encoded_fasta , outfile = None ):
		"""Convert encoded FASTA symbols to hex-encoded alignment text format."""
		with open(encoded_fasta, 'r') as f:
			if outfile == None:
				outfile = encoded_fasta.replace('.fasta' , '.hex')
				self.log(f'outfile for hex : {outfile}', level=2)
			with open(outfile , 'w') as g:
				for line in f:
					if line[0] == '>':
						g.write(line )
					else:
						hexstr = ''
						for char in line:
							o = ord(char)
							if o in self.replace_dict_ord:
								o = self.replace_dict_ord[o]
							hexstr += self.formathex(hex(o))[2:] + ' '
						g.write(hexstr + '\n')
		return outfile

	def read_textaln(self, aln_hexfile , outfile = None):
		"""Read a hex alignment and emit a remapped RAxML FASTA alignment file."""
		with open( aln_hexfile , 'r') as f:
			seqdict = {}
			seqstr = ''
			ID = ''
			for line in f:
				if line[0] == '>':
					seqdict[ID] = seqstr
					ID = line[1:].strip()
					seqstr = ''
				else:
					seqstr += line + ' '
			seqdict[ID] = seqstr
		
		alndf = pd.DataFrame( seqdict.items() , columns=['protid', 'hex_aln'] )
		alndf.index = alndf.protid
		alndf.drop( 'protid' , axis = 1 , inplace = True)
		alndf.drop( ''  , inplace = True)
		alndf['ord_aln'] = alndf.hex_aln.map( lambda x: [ int(c,16) if c!='--' else '-' for c in x.split() ] )
		alndf['seq_aln'] = alndf.ord_aln.map( lambda x: ''.join([ chr(c) if c !='-' else '-' for c in x ]) )
		alndf['remap_int'] = alndf.seq_aln.map(lambda x : [ self.map[c] if c in self.map else '-' for c in x ] )
		alndf['remap_symbols'] = alndf['remap_int'].map( lambda x : ''.join([ self.rev_raxml_indices[c] if c in self.rev_raxml_indices else '-' for c in x ]) )
		if outfile is None:
			outfile = aln_hexfile.replace('.hex' , '.raxml_aln.fasta')
		with open(outfile, 'w') as f:
			for i in alndf.index:
				f.write('>' + i + '\n' + alndf.loc[i].remap_symbols + '\n')
		return outfile

	def run_raxml_ng(self, fasta_file, matrix_file, nsymbols, output_prefix , iterations = 10 , cores = 8 , evoflags = {'+I'} ):
		"""Run RAxML-NG tree inference for a MULTI-state custom alphabet model.

		Returns the expected best-tree output path.
		"""
		raxmlng_path = self.raxml_path
		if raxmlng_path == None:
			raxmlng_path = 'raxml-ng'
		raxml_cmd = raxmlng_path  + '  --model MULTI'+str(self.nchars)+'_GTR{'+matrix_file+'}'+''.join(evoflags)+' --seed 12345 --threads auto{' + str(self.ncores) + '} --workers auto --msa '+fasta_file+' --prefix '+output_prefix
		if self.bs == True:
			raxml_cmd += ' --force perf_threads --bs-trees '+str(iterations)+' --bs-metric fbp'	
		if self.redo == True or self.overwrite == True:
			raxml_cmd += ' --redo'
		self.log(raxml_cmd, level=2)
		subprocess.run(raxml_cmd, shell=True)
		return output_prefix + '.raxml.bestTree'

	#ancestral reconstruction
	#raxml-ng --ancestral --msa ali.fa --tree best.tre --model HKY --prefix ASR

	def run_raxml_ng_ancestral_struct(self, fasta_file, tree_file, matrix_file, nsymbols, output_prefix):
		"""Run RAxML-NG ancestral state reconstruction for encoded structural alignments."""
		model = 'MULTI'+str(nsymbols)+'_GTR{'+matrix_file+'}+I'
		if self.raxmlng_path == None:
			self.raxmlng_path = 'raxml-ng'

		raxml_cmd = self.raxmlng_path + ' --ancestral --msa '+fasta_file+' --tree '+tree_file+' --model '+model+' --prefix '+output_prefix + ' --force perf_threads'
		if self.overwrite:
			raxml_cmd += ' --redo'
		self.log(raxml_cmd, level=2)
		subprocess.run(raxml_cmd, shell=True)
		return fasta_file.replace('raxml_aln.fasta' , 'raxml.ancestralStates')

	def madroot( self, treefile  , madroot_path = 'mad' ):
		"""Root a tree with MAD and return the rooted tree file path."""
		mad_cmd = f'{madroot_path} {treefile} '
		subprocess.run(mad_cmd, shell=True)
		return treefile+'.rooted'
	
	def ancestral2fasta(self, ancestral_file , outfasta = None ):
		"""Convert a RAxML ancestral states table into FASTA format."""
		if outfasta is None:
			outfasta = ancestral_file + '.fasta'
		with open( outfasta , 'w') as g:        
			with open( ancestral_file , 'r') as f:
				for l in f:
					words = l.split('	')
					if len(words) == 2:
						identifier, seq = words
						g.write('>' + identifier + '\n' + seq + '\n')
		return outfasta

	def ancestralfasta2df(self, outfasta ):
		"""Load ancestral FASTA into a DataFrame and map symbols back to indices."""
		aln_data = {}
		with open(outfasta, 'r') as f:
			for line in f:
				if line[0] == '>':
					ID = line[1:].strip()
					aln_data[ID] = ''
				else:
					aln_data[ID] += line.strip()
		ancestral_df = pd.DataFrame( aln_data.items() , columns=['protid', 'seq'] )
		#use rev map to convert back to ord
		ancestral_df['ord'] = ancestral_df.seq.map( lambda x: [ self.revmap_raxml[c] if c in self.revmap_raxml else '-' for c in x ] )
		return ancestral_df

	def decoder_reconstruction( self, ords , verbose = False):
		"""Decode discrete ancestral token ids into amino-acid sequence predictions.

		Builds a temporary heterograph required by the decoder and returns
		reconstructed sequence strings.
		"""
		data = HeteroData()
		ords = ords.to( self.device )
		z = self.encoder.vector_quantizer.embeddings( ords  ).to(self.device)
		edge_index = torch.tensor( [ [i,j] for i in range(z.shape[0]) for j in range(z.shape[0]) ]  , dtype = torch.long).T
		godnode_index = np.vstack([np.zeros(z.shape[0]), [ i for i in range(z.shape[0]) ] ])
		godnode_rev = np.vstack([ [ i for i in range(z.shape[0]) ] , np.zeros(z.shape[0]) ])
		#generate a backbone for the decoder
		data['res'].x = z
		data['res'].batch = torch.tensor([0 for i in range(z.shape[0])], dtype=torch.long)
		backbone, backbone_rev = self.converter.get_backbone( chainlen=z.shape[0] )
		backbone = sparse.csr_matrix(backbone)
		backbone_rev = sparse.csr_matrix(backbone_rev)
		backbone = self.converter.sparse2pairs(backbone)
		backbone_rev = self.converter.sparse2pairs(backbone_rev)
		positional_encoding = self.converter.get_positional_encoding( z.shape[0] , 256 )
		data['positions'].x = torch.tensor( positional_encoding, dtype=torch.float32).to( self.device )		
		data['res','backbone','res'].edge_index = torch.tensor(backbone,  dtype=torch.long ).to( self.device )
		data['res','backbonerev','res'].edge_index = torch.tensor(backbone_rev,  dtype=torch.long ).to( self.device )
		#add the godnode
		data['godnode'].x = torch.tensor(np.ones((1,5)), dtype=torch.float32).to( self.device )
		data['godnode4decoder'].x = torch.tensor(np.ones((1,5)), dtype=torch.float32).to( self.device )
		data['godnode4decoder', 'informs', 'res'].edge_index = torch.tensor(godnode_index, dtype=torch.long).to( self.device )

		# Repeat for godnode4decoder
		data['res', 'informs', 'godnode4decoder'].edge_index = torch.tensor(godnode_rev, dtype=torch.long).to( self.device )
		data['res', 'informs', 'godnode'].edge_index = torch.tensor(godnode_rev, dtype=torch.long).to( self.device )
		edge_index = edge_index.to( self.device )
		data = data.to( self.device )
		#decode_out = decoder(z , data.edge_index_dict[( 'res','contactPoints','res')] , data.edge_index_dict , poslossmod = 1 , neglossmod= 1 )
		allpairs = torch.tensor( [ [i,j] for i in range(z.shape[0]) for j in range(z.shape[0]) ]  , dtype = torch.long).T.to( self.device )
		out = self.decoder( data , allpairs )
		recon_x = out['aa'].detach().to('cpu') if 'aa' in out else None
		edge_probs = out['edge_probs'].detach().to('cpu').numpy() if 'edge_probs' in out else None
		amino_map = self.decoder.decoders['sequence_transformer'].amino_acid_indices
		revmap_aa = { v:k for k,v in amino_map.items() }
		edge_probs = edge_probs.reshape((z.shape[0], z.shape[0]))
		aastr = ''.join(revmap_aa[int(idx.item())] for idx in recon_x.argmax(dim=1) )
		res = {}
		res['aastr'] = aastr
		#res['edge_probs'] = edge_probs
		return res


	def run_site_likelihood_analysis(self, aln , tree ,  output_prefix = None):
		"""Run RAxML-NG site-likelihood evaluation on a fixed tree.

		Returns the generated .raxml.siteLH path when successful.
		"""
		self.log("Running site likelihood analysis...", level=1)
		#raxml command is  --force --evaluate --msa your_alignment.phy --model ### --tree fixed_tree.newick --site-lh
		self.log(f"Output prefix: {output_prefix}", level=2)
		model = f"MULTI{self.nchars}_GTR{{{self.submat}}}+I"
		if output_prefix is None:
				output_prefix = tree + '_siteLH_' + model
		
		cmd = [
			"raxml-ng",
			"--force",
			"--redo",
			"--msa", aln,
			"--model", model,
			"--tree", tree,
			"--sitelh",
			"--prefix", output_prefix
		]
		cmd = ' '.join(cmd)
		self.log(f"Running: {cmd}", level=2)
		subprocess.run(cmd, shell=True)
		
		#.raxml.siteLH
		sitelh_file = f"{output_prefix}.raxml.siteLH"
		if os.path.exists(sitelh_file):
			self.log(f"Site likelihood results saved to {sitelh_file}", level=1)
			return sitelh_file
		else:
			self.error("Site likelihood analysis did not produce results.")
			return None


	def structs2tree(self, structs , outdir = None , ancestral = False , raxml_iterations = 20 , raxml_path = None , output_prefix = None , verbose = False , **kwargs ):
		"""Run the full FoldTree2 structure-to-tree pipeline.

		Pipeline steps include structure encoding, symbol conversion, MAFFT
		alignment, RAxML tree inference, and optional ancestral reconstruction
		and MAD rooting.

		Returns
		-------
		dict
			Paths to generated intermediate and final outputs.
		"""
		#encode the structures
		if outdir is None:
			outdir = output_prefix

		step_verbosity = self._normalize_verbosity(kwargs.get('verbosity', verbose if verbose not in (None, False) else self.verbosity))
		previous_verbosity = self.verbosity
		self.verbosity = step_verbosity
		try:
			outfasta = os.path.join(outdir, self.modelname + 'encoded.fasta')
			if self.overwrite == True and os.path.exists(outfasta):
				if step_verbosity >= 1:
					print(f"Overwriting existing encoded FASTA at {outfasta}")
				os.remove(outfasta)

			encoded_fasta = self.encode_structblob( blob=structs , outfile = outfasta )

			#replace special characters
			#encoded_fasta = self.replace_sp_chars( encoded_fasta=encoded_fasta , outfile = outfasta , verbose = verbose)
			#convert to hex
			if step_verbosity >= 1:
				print('converting to hex for mafft')
			
			hexfasta = self.encodedfasta2hex( encoded_fasta , outfile = None  )
			# convert to ascii
			if step_verbosity >= 1:
				print('converting to ascii for mafft')

			asciifile = self.mafft_hex2ascii( hexfasta , outfile = None , hex2text_path = self.maffthex2text )
			if step_verbosity >= 2:
				print('asciifile:', asciifile)
			#run mafft text aln with custom submat
			if step_verbosity >= 1:
				print('running mafft')

			outaln = asciifile+'aln.txt'
			if not os.path.exists(outaln) and self.overwrite == False:
				mafftaln = self.run_mafft_textaln( asciifile , matrix=self.mafftmat , mafft_path = 'mafft'  )
			else:
				mafftaln = outaln
			if step_verbosity >= 2:
				print('mafftaln:', mafftaln)
			
			#convert the mafft aln to fasta
			if step_verbosity >= 1:
				print('converting mafft aln to hex fasta')
			mafftaln  = self.fasta2hex( mafftaln , outfile = None , maffttext2hex = self.maffttext2hex )
			#read the mafft aln
			alnfasta = self.read_textaln( mafftaln )
			#run raxml-ng
			if step_verbosity >= 1:
				print('running raxml-ng')
			if output_prefix is None:
				output_prefix = alnfasta.replace('.raxml_aln.fasta' , '')

			treefile = output_prefix + '.raxml.bestTree'

			if os.path.exists(treefile) and self.overwrite == False:
				pass
			else:
				treefile = self.run_raxml_ng( alnfasta , matrix_file= self.submat
					, nsymbols = self.nchars ,
					output_prefix = output_prefix ,
					iterations = raxml_iterations ,
						)

			#print the tree
			if step_verbosity >= 1:
				print('treefile:', treefile)
			tree = ete3.Tree(treefile, format=1)
			if step_verbosity >= 2:
				print(tree)

			if ancestral == True:
				#not tested yet
				ancestral_file = self.run_raxml_ng_ancestral_struct( alnfasta , treefile , self.submat , self.nchars , alnfasta.replace('.raxml_aln.fasta' , '') )
				ancestral_fasta = self.ancestral2fasta( ancestral_file )
				ancestral_df = self.ancestralfasta2df( ancestral_fasta )
				#decode the ancestral sequence
				if step_verbosity >= 2:
					print(ancestral_df.head())
				ords = ancestral_df.ord.values
				identifiers = ancestral_df.protid.values
				results = {}
				for l in tqdm.tqdm(range(ords.shape[0]), desc='decoding ancestral sequences', disable=step_verbosity == 0):
					res = self.decoder_reconstruction( torch.tensor(ords[l] , dtype=torch.long).T , verbose = verbose)
					results.update({ identifiers[l] : res } )
				#create a new dataframe with the decoded sequences
				results = pd.DataFrame.from_dict( results , orient='index' )
				if step_verbosity >= 2:
					print('decoded ancestral sequences:')
					print(results.head())
				#merge with ancestral df
				ancestral_df = ancestral_df.merge( results , left_on='protid' , right_index=True , how='left' )
				#write the ancestral dataframe to a file
				ancestral_df.to_csv( ancestral_fasta.replace('.aastr.fasta' , '.csv') )
				#write out aastr to a fasta
				with open( ancestral_fasta , 'w') as f:
					for i in ancestral_df.index:
						f.write('>' + ancestral_df.loc[i].protid + '\n' + ancestral_df.loc[i].aastr + '\n')
				ancestral_fasta = ancestral_fasta
			else:
				ancestral_fasta = None

			if self.root == True:
				if step_verbosity >= 1:
					print('rooting tree with mad')
				treefile = self.madroot( treefile )
				tree = ete3.Tree(treefile, format=1)
				if step_verbosity >= 2:
					print(tree)

			#return in dictionary form
			return { 'encoded_fasta': encoded_fasta, 'tree': treefile  , 'ancestral_fasta': ancestral_fasta  , 'alignment': alnfasta , 'mafft_aln': mafftaln, 'asciifile': asciifile, 'hexfasta': hexfasta }
		finally:
			self.verbosity = previous_verbosity
def print_about():
	"""Print a banner and project overview for FoldTree2."""
	ascii_art = r'''

                                 @@@@@@@@.                                 
                     %@@@@@@@ @@@@@@@@@@@@@@@@@@@@@@@                      
               /@@@@@@@@@@@@ . @@@@@@@@@@@@@@@@@@@@@@@@ .@@                
           @@@@@@@@@@@@@@@@  @  @@@@@@@@@@@@@@@@@@@@   /@@@@@@@            
       .@@@@@@@@@@@@@@@@@@@ @@@  @@@@@@@@@@@@@@@@  @% %@@@@@@@@@@@*        
     @@@@@@@@@@@@@@@@@@@@@ @@@@@  @@@@@@@@@@@@  @@@  @@@@@@@@@@@@@@@@&     
  %@@@@@@@@@@@@@@@@@@@@@@ #@@@@@,  @@@@@@@&  @@@@@  @@@@@@@@@@@@@@@@@@@@   
@@@@@@@@@@@@@@@   .  @@@( @@@@@@@#@ @@@( (@@@@@@ / @@@@@@@@@@@@@@@@@@@@@@@ 
@@@@@@@@@@@@@@ #@  *@ (@  @@@@@ @ @   %@@@@@@@@ * @@@@@@@@@@@@@@@@@@@@@@@@@
@@@@@@@@@@@@@ (  @@ .@@ @     @@*@@@@  @@@@@@.,  @@@@@@@@@@@@@@@@@@@@@@@@@@
@@@@@@@@@@@@  @@@@@@  @@.  @@@@@.@@@@@@* @@@ @  @@@@@@@@@@@@@@@@@@@@@@@@@@@
@@@@@@@@@@@@@@@@@ @@@  @@@ /@@#             @ *@@@@@ @@@@@@@@@@@@@@@@@@@@@@
@@@@@@@@@@@ *@@@@ %@@@  @@@  @@&#@@@@@@  @ @ #@@@@@# @@  @@@@@@@@@ @@@@@@@@
@@@@@@@   .%@  @@@ @@ @( @@@  @@(@@@@  @@@  @@@@@@@  # @.  ,@@@@ &@@@.@@@@@
@@@@@@@@@@@@@@  @   @@@@@ @ *@ @,@  @@@@#  @@@@@@@@  @@@@@@@@@     @@@@@@@@
@@@@@@@@@%@@@@@@@@   *@@@@@@@@@  %@@@@@@@@@@@@@@@@. @@@@@@@@ @       @@@@@@
@@@@@&@@@@# @@@ @@@@@&  @. @@@@@@@@@@@@@@@@@@@@@(# @@@@%(.  /@@@#@@@@@@@@@@
@@@@@@@@@  ,# @ @@@@@@@@@ * @@@@@@@@@@@@@@@@@. (@      ,@@@@@@@@ @@@&  @@@@
@@@@@@@@@@@@@@ &.@@@@@@@@@ @ @@@@@@@@@@@@/ @@/  @@@@@@@@@@@@@  % / .@@@@@@@
@@@@@@@@@@@@@@@   @@@@@@@@ @%/@@@@@@@@@ %@@ .@@@@@@@@@@@/  #@@ #(@@@@@@@@@@
@@@@@.      , .,.  ...,,**  @@@  @@@( @@               (@@@@@ %.@@@@@@@@@@@
@@@@@@@@  /@@@@@@@@@@@@@@@@@   @@@.(@.  *@@@@@@@@@@@@ .@@@@@ ( @@@(   *@@@@
@@@@@@@@@@@@@@&  @@@@@@@@@@@@@@    *  @@@@@@@@@@@@@@ @@@@ .(        &@@@@@@
@@@@@@@@@@  @@@@@@#   @@@@/    %@ @@& @@@@@@@@@@@@@@@ ,@   @@@@@@@@@@@@@@@@
@@@@@             #@%/            @.& @@@@@@@@@@( &@@  &@@@@@@@@    @@@@@@@
@@@@@@@@@  @@@@ *@@@ ,@@@@@@@@@@@ @@   @@@@&#(,@@*     .,* .( %@@@@@@@@@@@@
@@@@@@@@@@@@@  (@@@@@@@@@@@@@@@@*%@%*             ( (@@@@@@@@@ @@  ./%@@@@@
@@@@@@@@@@@ @@/@@@@@@@@@@@@@@@@@ @@ @  %@@@@@@@@@@@@*      &@@@#,@@* @@@@@@
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ @@ @%  @@@@@@@@@@@@@@@ /@@, @@@@@@@@@@@@@@@
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@&.@@ @   @@@@@@@@@@@@@@@@ @@@.,  @@@@@@@@@@@@
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ @@ @@  .@@@@@@@@@@@@@@@@@@@@@ @@@@@@@@@@@@@@
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@,%@/%@.  &.@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ 
  ,@@@@@@@@@@@@@@@@@@@@@@@@@@ @@ @@  .@ @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@   
     @@@@@@@@@@@@@@@@@@@@@@@ @@ @@@  (@ @@@@@@@@@@@@@@@@@@@@@@@@@@@@@      
        @@@@@@@@@@@@@@@@@@@&,@@ @@   @@(.@@@@@@@@@@@@@@@@@@@@@@@@@         
           (@@@@@@@@@@*      @ @@%   @@@     #@@@@@@@@@@@@@@@@@            
                @@@@@@@@@@@@@ *    /@@@(.@@@@@@@@@@@@@@@@@(                
                      @@@@@ /@@@@@@@@@@@@@@@@@@@@@@@%                      

				+-------------------------------------------+
				|                 foldtree2                 |
				|        Structural Phylogenetics & AI      |
				|              🧬   🧠   🌳                 |
				+-------------------------------------------+

	'''

	print(ascii_art)
	print("FoldTree2: Structural Phylogenetics and Ancestral Sequence Reconstruction")
	print("--------------------------------------------------------------------------------")
	print("FoldTree2 is a toolkit for encoding protein structures as sequences using deep learning,\n"
		  "enabling phylogenetic tree inference, ancestral structure/sequence reconstruction, and\n"
		  "custom alphabets for evolutionary analysis. It integrates structure encoding, alignment,\n"
		  "custom substitution matrices, and tree inference (RAxML-NG), supporting \n"
		  "structure-based workflows. FoldTree2 is designed for protein family analysis,\n"
		  "benchmarking, and exploring the evolution of protein folds.\n\n"
		  "NOTE: FoldTree2 is under heavy development and its interface, models, and workflows may change\n"
		  "as new features and improvements are added.\n\n"
		  "Project: https://github.com/DessimozLab/foldtree2\n"
		  "Contact: dmoi@unil.ch\n")
	print("Run with --help for usage instructions.")

def main():
	"""Command-line entry point for FoldTree2 tree building workflows."""
	if '--about' in sys.argv:
		print_about()
		sys.exit(0)

	
	# Example usage:
	# Run the script from the command line with:
	# python ft2treebuilder.py --model path/to/model --mafftmat path/to/mafft_matrix.mtx --submat path/to/substitution_matrix.mtx --structures "/path/to/structures/*.pdb" --ancestral
	# This command will load the model (from 'path/to/model.pkl'),
	# the MAFFT matrix, and the substitution matrix for RAxML.
	# It will process all PDB files matching the glob pattern,
	# perform the ancestral reconstruction, and output results accordingly.

	#otherwise, import the treebuilder class and use it programmatically.

	parser = argparse.ArgumentParser(description="CLI for running foldtree2 tree builder.")
	parser.add_argument("--about", action="store_true", help="Show information about FoldTree2 and exit.")

	parser.add_argument("--model", required=False, help="Path to the model and name (without encoder/decoder or .pth extension)")
	parser.add_argument("--encoder", required=False, default=None, help="Path to the encoder model (.pth file)")
	parser.add_argument("--decoder", required=False, default=None, help="Path to the decoder model (.pth file)")

	parser.add_argument("--mafftmat", required=False, default = None , help="Path to the MAFFT substitution matrix")
	parser.add_argument("--submat", required=False, default = None, help="Path to the substitution matrix for RAxML")
	parser.add_argument("--charmaps", required=False, default=None, help="Path to the character maps for encoding (if not specified, uses default)"	)
	parser.add_argument("--structures", required=True, help="Glob pattern for input structure files (e.g. '/path/to/structures/*.pdb')")
	parser.add_argument("--outdir", default=None, help="Output directory for results")
	parser.add_argument("--output_prefix", default=None, help="Output file prefix for encoded sequences")

	#paths to properties and executables
	parser.add_argument("--aapropcsv", default=None, help="Path to amino acid properties CSV file for PDB2PyG conversion (default: packaged foldtree2/config/aaindex1.csv)")
	parser.add_argument("--maffttext2hex", default='maffttext2hex', help="Path to maffttext2hex executable")
	parser.add_argument("--maffthex2text", default='hex2maffttext', help="Path to hex2maffttext executable")

	parser.add_argument("--ncores", type=int, default=8, help="Number of CPU cores to use for processing")
	parser.add_argument("--raxml_iterations", type=int, default=20, help="Number of RAxML iterations for tree inference")
	parser.add_argument("--raxmlpath", default='raxml-ng', help="Path to RAxML-NG executable")
	parser.add_argument("--verbosity", type=int, choices=[0, 1, 2], default=1, help="Verbosity level: 0=quiet, 1=step-level progress, 2=detailed command/debug output")
	parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
	parser.add_argument("--device", default=None, help="Device to run the model on (default: None, uses CPU or GPU if available)")
	parser.add_argument( "--bs", default=False, action="store_true", help="Enable bootstrapping in RAxML-NG" )
	parser.add_argument( "--redo", default=False, action="store_true", help="Enable --redo flag in RAxML-NG to overwrite existing results" )
	parser.add_argument( "--overwrite", default=False, action="store_true", help="Overwrite existing output files if they exist" )


	parser.add_argument("--root", default=False, action="store_true", help="Use mad root to root the tree after inference")

	# Ancestral reconstruction options
	parser.add_argument("--ancestral", action="store_true", help="Perform ancestral reconstruction")

	if len(sys.argv) == 1 or ('--help' in sys.argv) or ('-h' in sys.argv):
		print('No arguments provided. Use -h or --help for help.')
		print('Example command:')
		print('  python ft2treebuilder.py --model ./models/my_model --structures "/path/to/structures/*.pdb" --outdir ./results --ncores 8 --raxml_iterations 20 --ancestral')
		parser.print_help()
		sys.exit(0)

	if '--about' in sys.argv:
		print_about()
		sys.exit(0)
	args = parser.parse_args()
	if args.verbose:
		args.verbosity = 2

	if args.model is None and args.encoder is None:
		print('Model path is required. Use --model to specify the model path.')
		print( 'or use --encoder and --decoder to specify paths to encoder and decoder separately.' )
		sys.exit(1)

	try:
		args.aapropcsv = resolve_aapropcsv_path(args.aapropcsv)
	except FileNotFoundError as exc:
		print(str(exc))
		sys.exit(1)
	
	if args.model is not None:
		args.model = args.model.replace('.pt', '')
		args.model = args.model.replace('.pth', '')
		modeldir = '/'.join( args.model.split('/')[:-1] )
		encoder_path = os.path.join( args.model + '_encoder.pt' )
		decoder_path = os.path.join( args.model + '_decoder.pt' )
	
	if args.encoder is not None and args.decoder is not None:
		encoder_path = args.encoder
		decoder_path = args.decoder
	

	
	if args.verbosity >= 1:
		print('Using encoder path:', encoder_path)
		print('Using decoder path:', decoder_path)
	#check pth files exist

	if not os.path.exists(encoder_path) or not os.path.exists(decoder_path):
		if args.model is not None:
			print(f"Model files not found in {args.model}. Please ensure the encoder and decoder files are present.")
		else:
			print(f"Model files not found. Please ensure the encoder and decoder files are present.")
		sys.exit(1)

	if args.structures is None:
		print('Structures glob pattern is required. Use --structures to specify the glob pattern.')
		sys.exit(1)

	if args.structures[-1] == '/':
		args.structures += '*.pdb'
	elif not args.structures.endswith('.pdb'):
		args.structures += '.pdb'
	
	if args.outdir is not None:
		if not os.path.exists(args.outdir):
			os.makedirs(args.outdir , exist_ok=True)
	
	if args.output_prefix is None:
		if args.outdir is not None:
			args.output_prefix = os.path.join(args.outdir, encoder_path.split('/')[-1].replace('.pt', ''))
		else:
			args.output_prefix = encoder_path.split('/')[-1].replace('.pt', '')
		if not args.output_prefix.endswith('_'):
			args.output_prefix += '_'
	
	if args.mafftmat is None:
		args.mafftmat = encoder_path.replace('.pt', '_mafftmat.mtx')
	if args.submat is None:
		args.submat = encoder_path.replace('.pt', '_submat.txt')
	if args.charmaps is None:
		args.charmaps = encoder_path.replace('.pt', '_pair_counts.pkl')
	

	# Create an instance of treebuilder
	tb = treebuilder(model=encoder_path, mafftmat=args.mafftmat, decoder_model=decoder_path, submat=args.submat , raxml_path=args.raxmlpath,
	 aapropcsv=args.aapropcsv, maffttext2hex=args.maffttext2hex, maffthex2text=args.maffthex2text, ncores=args.ncores , charmaps=args.charmaps , device=args.device, 
	 bs=args.bs, redo=args.redo , verbose=args.verbose , verbosity=args.verbosity, root =args.root )

	# Generate tree from structures using the provided options
	tb.structs2tree(structs=args.structures, outdir=args.outdir, ancestral=args.ancestral, raxml_iterations=args.raxml_iterations , raxml_path=args.raxmlpath , output_prefix=args.output_prefix
				 , verbose=args.verbose  , verbosity=args.verbosity, aapropcsv=args.aapropcsv, maffttext2hex=args.maffttext2hex, ncores=args.ncores)

if __name__ == "__main__":
	main()
