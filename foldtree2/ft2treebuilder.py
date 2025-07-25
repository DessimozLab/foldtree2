import pickle
import torch
import glob
import subprocess
import Bio.PDB as PDB
#import torch_geometric hetero data
import torch_geometric
import multiprocessing as mp
import pebble
import argparse

from foldtree2.src import foldtree2_ecddcd as ft2
from foldtree2.src import mono_decoders
from foldtree2.src.pdbgraph import PDB2PyG


import traceback
import tqdm
import pandas as pd
import os
import ete3 
import sys

class treebuilder():
	def __init__ ( self , model , mafftmat = None , submat = None , n_state = None, raxml_path= None, charmaps=None, **kwargs ):

		#make fasta is shifted by 1 and goes from 1-248 included
		#0x01 – 0xFF excluding > (0x3E), = (0x3D), < (0x3C), - (0x2D), Space (0x20), Carriage Return (0x0d) and Line Feed (0x0a)
		#replace 0x22 or " which is necesary for nexus files and 0x23 or # which is also necesary		
		self.replace_dict = {chr(0):chr(246) , '"':chr(248) , '#':chr(247), '>' : chr(249), '=' : chr(250), '<' : chr(251), '-' : chr(252), ' ' : chr(253) , '\r' : chr(254), '\n' : chr(255) }
		self.rev_replace_dict = { v:k for k,v in self.replace_dict.items() }
		self.replace_dict_ord = { ord(k):ord(v) for k,v in self.replace_dict.items() }
		self.raxml_path = raxml_path
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
			print('loading charmaps from', charmaps)
			with open(charmaps, 'rb') as f:
				pair_counts, char_set, char_position_map , raxml_charset, raxml_char_position_map = pickle.load(f)
			self.raxml_characters = raxml_charset
			self.alphabet = char_set
			self.nchars = len(self.alphabet)
			self.map = char_position_map
			self.revmap = { v:k for k,v in char_position_map.items() }
			self.raxml_indices = raxml_char_position_map
			self.rev_raxml_indices = { v:k for k,v in raxml_char_position_map.items() }
			self.raxmlchars = raxml_charset
		self.ordset = set([ ord(c) for c in self.alphabet ])
		#load pickled model
		self.model = model
		self.encoder = torch.load(model + '_encoder.pth', map_location=torch.device('cpu') , weights_only=False)
		self.decoder = torch.load(model + '_decoder.pth', map_location=torch.device('cpu') , weights_only=False)

		if 'aapropcsv' in kwargs and kwargs['aapropcsv'] is not None:
			self.converter = PDB2PyG(aapropcsv=kwargs['aapropcsv'])
		else:
			self.converter = PDB2PyG()

		#detect if we are using a GPU
		if 'device' in kwargs and kwargs['device'] is not None:
			self.device = torch.device(kwargs['device'])
		else:
			self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
			self.encoder = self.encoder.to(self.device)
			self.decoder = self.decoder.to(self.device)
			self.encoder.device = self.device
		
		self.encoder.eval()
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
	def formathex(hexnum):
			if len(hexnum) == 3:
				return hexnum[0:2] + '0' + hexnum[2]
			else:
				return hexnum

	@staticmethod
	def run_mafft_textaln( infasta , outaln=None , matrix='mafft_submat.mtx' , mafft_path = 'mafft' ):
		if outaln == None:
			outaln = infasta+'aln.txt'
		cmd = f'{mafft_path} --text --thread -1 --localpair --maxiterate 1000 --textmatrix {matrix} {infasta}  > {outaln}'
		print(cmd)
		subprocess.run(cmd, shell=True)
		return outaln

	@staticmethod
	def mafft_hex2ascii( intext , outfile , hex2text_path = './mafft_tools/hex2maffttext' ):
		if outfile == None:

			outfile = intext.replace('.hex' , '.ASCII')
			print('outfile for ascii :', outfile)
		#% /usr/local/libexec/mafft/hex2maffttext input.hex > input.ASCII
		cmd = f'{hex2text_path} {intext} > {outfile}'
		print(cmd)
		subprocess.run(cmd, shell=True)
		return outfile    

	@staticmethod
	def fasta2hex( intext , outfile  , maffttext2hex = './mafft_tools/maffttext2hex' ):
		#% /usr/local/libexec/mafft/maffttext2hex input.hex > input.ASCII
		if outfile == None:
			outfile = intext+'.hex'
		cmd = f'{maffttext2hex} {intext} > {outfile}'
		print(cmd)
		subprocess.run(cmd, shell=True)
		return outfile    

	@staticmethod
	def normal_mafft( infasta , outaln ):
		cmd = f'mafft --anysymbol {infasta} > {outaln}'
		print(cmd)
		subprocess.run(cmd, shell=True)
		return outaln

	@staticmethod
	def struct2sequence(structfile):
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
		print( 'converting structures')
		for struct in tqdm.tqdm(structlist):
			try:
				data = self.converter.struct2pyg( struct )
				if data:
					yield data
			except:
				print('error', struct)
				print( traceback.format_exc() )
				continue
	@staticmethod
	def aln2dcict(alnfile):
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
					seqstr += line + ' '
			seqdict[ID] = seqstr
		return seqdict

	def encode_structblob(self , blob = None , outfile = None ):
		if blob[-1] == '/':
			structs = glob.glob(blob + '*.pdb')
		else:
			structs = glob.glob(blob)
		if outfile == None:
			outfolder = '/'.join( blob.split('/')[:-1] )
			outfile = outfolder + 'encoded.fasta'
		loader = self.struct_loader( structs , self.converter )
		self.encoder.encode_structures_fasta( loader , outfile , replace = True)
		return outfile

	def encode_structblob_raxml(self , blob = None , outfile = None ):
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
		if outfile == None:
			outfile = encoded_fasta.replace('.fasta' , '_replaced.fasta')
		#load the encoded fasta
		with open(encoded_fasta) as encoded:
			seqstr = '' 
			ID = ''
			seqdict = {}
			for line in encoded:
				if line[0] == '>' and line[-1] == '\n':
					seqdict[ID] = seqstr
					ID = line[1:].strip()
					seqstr = ''
				else:
					seqstr += line.strip()
			del seqdict['']
			encoded_df = pd.DataFrame( seqdict.items() , columns=['protid', 'seq'] )
		#replace the characters that aren't allowed
		encoded_df.seq = encoded_df.seq.map(lambda x : ''.join([ c if c not in self.replace_dict else self.replace_dict[c] for c in x]))
		encoded_df['ord'] = encoded_df.seq.map( lambda x: [ ord(c) for c in x] )
		if verbose:
			print(encoded_df.head())
		#write output to fasta
		with open( outfile, 'w') as f:
			for idx, row in encoded_df.iterrows():
				f.write('>' + row.protid + '\n' + row.seq + '\n')
		return outfile
	
	def encodedfasta2hex(self , encoded_fasta , outfile = None ):
		with open(encoded_fasta, 'r') as f:
			if outfile == None:
				outfile = encoded_fasta.replace('.fasta' , '.hex')
				print('outfile for hex :', outfile)
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

	def run_raxml_ng(self, fasta_file, matrix_file, nsymbols, output_prefix , iterations = 10 , cores = 8 , bs =False):
		raxmlng_path = self.raxml_path
		if raxmlng_path == None:
			raxmlng_path = 'raxml-ng'
		raxml_cmd = raxmlng_path  + ' --model MULTI'+str(self.nchars)+'_GTR{'+matrix_file+'} --redo  --all --bs-trees '+str(iterations)+' --seed 12345 --threads auto{' + str(self.ncores) + '} --workers auto --msa '+fasta_file+' --prefix '+output_prefix  + ' --force perf_threads'
		#raxml_cmd =raxmlng_path  + ' --model MULTI'+str(nsymbols)+'_GTR+I+G --redo  --all --bs-trees '+str(iterations)+' --seed 12345 --threads 8 --msa '+fasta_file+' --prefix '+output_prefix 
		print(raxml_cmd)
		subprocess.run(raxml_cmd, shell=True)
		return output_prefix + '.raxml.bestTree'

	#ancestral reconstruction
	#raxml-ng --ancestral --msa ali.fa --tree best.tre --model HKY --prefix ASR

	def run_raxml_ng_ancestral_struct(self, fasta_file, tree_file, matrix_file, nsymbols, output_prefix):
		model = 'MULTI'+str(nsymbols)+'_GTR{'+matrix_file+'}+I+G'
		if raxmlng_path == None:
			raxmlng_path = 'raxml-ng'
		raxml_cmd = raxmlng_path + ' --redo --ancestral --msa '+fasta_file+' --tree '+tree_file+' --model '+model+' --prefix '+output_prefix + ' --force perf_threads'
		print(raxml_cmd)
		subprocess.run(raxml_cmd, shell=True)
		return None

	def madroot( treefile  , madroot_path = 'mad' ):
		mad_cmd = f'{madroot_path} {treefile} '
		subprocess.run(mad_cmd, shell=True)
		return treefile+'.rooted'
	
	def ancestral2fasta( ancestral_file , outfasta ):
		with open( outfasta , 'w') as g:        
			with open( ancestral_file , 'r') as f:
				for l in f:
					words = l.split('	')
					if len(words) == 2:
						identifier, seq = words
						g.write('>' + identifier + '\n' + seq + '\n')
		return outfasta

	def ancestralfasta2df( outfasta ):
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
		ancestral_df['ord'] = ancestral_df.seq.map( lambda x: [ revmap_raxml[c] if c in revmap_raxml else '-' for c in x ] )
		return ancestral_df

	def decoder_reconstruction( self, ords , verbose = False):
		data = HeteroData()
		z = self.encoder.vector_quantizer.embeddings( ords  ).to('cpu')
		edge_index = torch.tensor( [ [i,j] for i in range(z.shape[0]) for j in range(z.shape[0]) ]  , dtype = torch.long).T
		godnode_index = np.vstack([np.zeros(z.shape[0]), [ i for i in range(z.shape[0]) ] ])
		godnode_rev = np.vstack([ [ i for i in range(z.shape[0]) ] , np.zeros(z.shape[0]) ])
		#generate a backbone for the decoder
		data['res'].x = z
		backbone, backbone_rev = self.converter.get_backbone( z.shape[0] )
		backbone = sparse.csr_matrix(backbone)
		backbone_rev = sparse.csr_matrix(backbone_rev)
		backbone = self.converter.sparse2pairs(backbone)
		backbone_rev = self.converter.sparse2pairs(backbone_rev)
		positional_encoding = self.converter.get_positional_encoding( z.shape[0] , 256 )
		data['positions'].x = torch.tensor( positional_encoding, dtype=torch.float32)
		data['res'].x = torch.cat([data['res'].x, data['positions'].x], dim=1)
		data['res','backbone','res'].edge_index = torch.tensor(backbone,  dtype=torch.long )
		data['res','backbonerev','res'].edge_index = torch.tensor(backbone,  dtype=torch.long )

		data['res','backbone','res'].edge_index = torch_geometric.utils.add_self_loops(data['res','backbone','res'].edge_index)[0]
		data['res','backbonerev','res'].edge_index = torch_geometric.utils.add_self_loops(data['res','backbonerev','res'].edge_index)[0]

		#add the godnode
		data['godnode'].x = torch.tensor(np.ones((1,5)), dtype=torch.float32)
		data['godnode4decoder'].x = torch.tensor(np.ones((1,5)), dtype=torch.float32)
		data['godnode4decoder', 'informs', 'res'].edge_index = torch.tensor(godnode_index, dtype=torch.long)

		# Repeat for godnode4decoder
		data['res', 'informs', 'godnode4decoder'].edge_index = torch.tensor(godnode_rev, dtype=torch.long)
		data['res', 'informs', 'godnode'].edge_index = torch.tensor(godnode_rev, dtype=torch.long)		
		edge_index = edge_index.to( device )
		data = data.to( self.device )
		
		#decode_out = decoder(z , data.edge_index_dict[( 'res','contactPoints','res')] , data.edge_index_dict , poslossmod = 1 , neglossmod= 1 )
		allpairs = torch.tensor( [ [i,j] for i in range(z.shape[0]) for j in range(z.shape[0]) ]  , dtype = torch.long).T.to( self.device )
		out = decoder( data.x_dict, data.edge_index_dict , allpairs )
		recon_x = out['aa'] if 'aa' in out else None
		edge_probs = out['edge_probs'] if 'edge_probs' in out else None
		print( edge_probs.shape)

		amino_map = decoder.decoders['sequence_transformer'].amino_acid_indices
		revmap_aa = { v:k for k,v in amino_map.items() }
		edge_probs = edge_probs.reshape((z.shape[0], z.shape[0]))
		aastr = ''.join(revmap_aa[int(idx.item())] for idx in recon_x.argmax(dim=1) )
		return aastr ,edge_probs , zgodnode , foldxout , r , t , angles

	def structs2tree(self, structs , outdir = None , ancestral = False , raxml_iterations = 20 , raxml_path = None , output_prefix = None , verbose = False , **kwargs ):
		#encode the structures
		
		outfasta = os.path.join(outdir, 'encoded.fasta')
		encoded_fasta = self.encode_structblob( blob=structs , outfile = outfasta )	
		#replace special characters
		#encoded_fasta = self.replace_sp_chars( encoded_fasta=encoded_fasta , outfile = outfasta , verbose = verbose)
		#convert to hex
		print('converting to hex for mafft')
		hexfasta = self.encodedfasta2hex( encoded_fasta , outfile = None  )
		# convert to ascii
		print('converting to ascii for mafft')
		asciifile = self.mafft_hex2ascii( hexfasta , outfile = None , hex2text_path = self.maffthex2text )
		print('asciifile:', asciifile)
		#run mafft text aln with custom submat
		print('running mafft')
		mafftaln = self.run_mafft_textaln( asciifile , matrix=self.mafftmat , mafft_path = 'mafft'  )
		#convert the mafft aln to fasta
		print('converting mafft aln to hex fasta')
		mafftaln  = self.fasta2hex( mafftaln , outfile = None , maffttext2hex = self.maffttext2hex )
		#read the mafft aln
		alnfasta = self.read_textaln( mafftaln )
		#run raxml-ng
		print('running raxml-ng')
		if output_prefix is None:
			output_prefix = alnfasta.replace('.raxml_aln.fasta' , '')

		treefile = self.run_raxml_ng( alnfasta , matrix_file= self.submat
			   , nsymbols = self.nchars ,
			   output_prefix = output_prefix ,
			   iterations = raxml_iterations ,
			    )
		#print the tree
		print('treefile:', treefile)
		tree = ete3.Tree(treefile, format=1)	
		print(tree)


		if ancestral == True:
			#not tested yet
			ancestral_file = self.run_raxml_ng_ancestral_struct( alnfasta , treefile , self.submat , self.nchars , alnfasta.replace('.raxml_aln.fasta' , '') )
			ancestral_fasta = self.ancestral2fasta( ancestral_file , outfasta )
			ancestral_df = self.ancestralfasta2df( outfasta )
			#decode the ancestral sequence
			ords = ancestral_df.ord.values
			for l in ords.shape[0]:
				res = decoder_reconstruction( ords[l] , verbose = verbose)	
				ancestral_df.loc[l , 'aastr'] = aastr
				ancestral_df.loc[l , 'edge_probs'] = edge_probs
				ancestral_df.loc[l , 'zgodnode' ] = zgodnode
				ancestral_df.loc[l , 'foldxout' ] = foldxout
				ancestral_df.loc[l , 'r' ] = r
				ancestral_df.loc[l , 't' ] = t
				ancestral_df.loc[l , 'angles' ] = angles
			#write the ancestral dataframe to a file
			ancestral_df.to_csv( outfasta.replace('.fasta' , '.csv') )
			#write out aastr to a fasta
			with open( outfasta.replace('.fasta' , '.aastr.fasta') , 'w') as f:
				for i in ancestral_df.index:
					f.write('>' + i + '\n' + ancestral_df.loc[i].aastr + '\n')
			
def print_about():
	ascii_art = r'''
	+-------------------------------+
	|       foldtree2     	   	    |	
	|  Structural Phylogenetics & AI|
	|   	🧬  🧠  🌳              |
	+-------------------------------+
	'''

	print(ascii_art)
	print("FoldTree2: Structural Phylogenetics and Ancestral Sequence Reconstruction")
	print("--------------------------------------------------------------------------------")
	print("FoldTree2 is a toolkit for encoding protein structures as sequences using deep learning,\n"
		  "enabling phylogenetic tree inference, ancestral structure/sequence reconstruction, and\n"
		  "custom alphabets for evolutionary analysis. It integrates structure encoding, alignment,\n"
		  "custom substitution matrices, and tree inference (RAxML-NG), supporting both sequence\n"
		  "and structure-based workflows. FoldTree2 is designed for protein family analysis,\n"
		  "benchmarking, and exploring the evolution of protein folds.\n\n"
		  "NOTE: FoldTree2 is under heavy development and its interface, models, and workflows may change\n"
		  "as new features and improvements are added.\n\n"
		  "Project: https://github.com/DessimozLab/foldtree2\n"
		  "Contact: dmoi@unil.ch\n")
	print("Run with --help for usage instructions.")

def main():
	if '--about' in sys.argv:
		print_about()
		sys.exit(0)

	parser = argparse.ArgumentParser(description="CLI for running foldtree2 tree builder.")
	parser.add_argument("--about", action="store_true", help="Show information about FoldTree2 and exit.")
	parser.add_argument("--model", required=True, help="Path to the model (without .pkl extension)")
	parser.add_argument("--modeldir" , required=False, default='./models', help="Directory containing the model files (if not specified, uses current directory)")
	parser.add_argument("--mafftmat", required=False, default = None , help="Path to the MAFFT substitution matrix")
	parser.add_argument("--submat", required=False, default = None, help="Path to the substitution matrix for RAxML")
	parser.add_argument("--charmaps", required=False, default=None, help="Path to the character maps for encoding (if not specified, uses default)"	)
	parser.add_argument("--structures", required=True, help="Glob pattern for input structure files (e.g. '/path/to/structures/*.pdb')")
	parser.add_argument("--outdir", default=None, help="Output directory for results")
	parser.add_argument("--output_prefix", default=None, help="Output file prefix for encoded sequences")

	#paths to properties and executables
	parser.add_argument("--aapropcsv", default='./foldtree2/config/aaindex1.csv', help="Path to amino acid properties CSV file for PDB2PyG conversion")
	parser.add_argument("--maffttext2hex", default='maffttext2hex', help="Path to maffttext2hex executable")
	parser.add_argument("--maffthex2text", default='hex2maffttext', help="Path to hex2maffttext executable")

	parser.add_argument("--ncores", type=int, default=8, help="Number of CPU cores to use for processing")
	parser.add_argument("--raxml_iterations", type=int, default=20, help="Number of RAxML iterations for tree inference")
	parser.add_argument("--raxmlpath", default='raxml-ng', help="Path to RAxML-NG executable")
	parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
	parser.add_argument("--n_state", type=int, default=40, help="Number of encoded states (default: 40)")
	parser.add_argument("--device", default=None, help="Device to run the model on (default: None, uses CPU or GPU if available)")
	# Ancestral reconstruction options
	parser.add_argument("--ancestral", action="store_true", help="Perform ancestral reconstruction")

	if len(sys.argv) == 1 or ('--help' in sys.argv) or ('-h' in sys.argv):
		print('No arguments provided. Use -h or --help for help.')
		print('Example command:')
		print('  python ft2treebuilder.py --model path/to/model --mafftmat path/to/mafft_matrix.mtx --submat path/to/substitution_matrix.txt --structures "/path/to/structures/*.pdb" --outdir ./results --ancestral')
		parser.print_help()
		sys.exit(0)

	if '--about' in sys.argv:
		print_about()
		sys.exit(0)
	args = parser.parse_args()
	if args.model is None:
		print('Model path is required. Use --model to specify the model path.')
		sys.exit(1)

	modelpath = os.path.join( args.modeldir, args.model )
	#check pth files exist
	if not os.path.exists(modelpath + '_encoder.pth') or not os.path.exists(modelpath + '_decoder.pth'):
		print(f"Model files not found in {args.modeldir}. Please ensure the model files are present.")
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
			os.makedirs(args.outdir)
	if args.output_prefix is None:
		args.output_prefix = 'encoded'
	else:
		if not args.output_prefix.endswith('_'):
			args.output_prefix += '_'
	
	if args.mafftmat is None:
		args.mafftmat = os.path.join(args.modeldir, args.model + '_mafftmat.mtx')
	if args.submat is None:
		args.submat = os.path.join(args.modeldir, args.model + '_submat.txt')
	if args.charmaps is None:
		args.charmaps = os.path.join(args.modeldir, args.model + '_pair_counts.pkl')
	

	# Example usage:
	# Run the script from the command line with:
	# python ft2treebuilder.py --model path/to/model --mafftmat path/to/mafft_matrix.mtx --submat path/to/substitution_matrix.mtx --structures "/path/to/structures/*.pdb" --ancestral
	# This command will load the model (from 'path/to/model.pkl'),
	# the MAFFT matrix, and the substitution matrix for RAxML.
	# It will process all PDB files matching the glob pattern,
	# perform the ancestral reconstruction, and output results accordingly.

	# Create an instance of treebuilder
	tb = treebuilder(model=modelpath, mafftmat=args.mafftmat, submat=args.submat , n_state=args.n_state , raxml_path=args.raxmlpath,
	 aapropcsv=args.aapropcsv, maffttext2hex=args.maffttext2hex, maffthex2text=args.maffthex2text, ncores=args.ncores , charmaps=args.charmaps , device=args.device)

	# Generate tree from structures using the provided options
	tb.structs2tree(structs=args.structures, outdir=args.outdir, ancestral=args.ancestral, raxml_iterations=args.raxml_iterations , raxml_path=args.raxmlpath , output_prefix=args.output_prefix
				 , verbose=args.verbose , n_state=args.n_state , aapropcsv=args.aapropcsv, maffttext2hex=args.maffttext2hex, ncores=args.ncores)

if __name__ == "__main__":
	main()
