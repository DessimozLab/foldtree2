#!/usr/bin/env python
# coding: utf-8

from utils import *


"""
datadir = '../../datasets/foldtree2/'
EPS = 1e-10

import wget
import copy
import importlib
import warnings 
import torch_geometric
import glob
import h5py
from scipy import sparse
from copy import deepcopy
import pebble 
import time
import h5py
import torch
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.utils import to_networkx
from torch_geometric.data import HeteroData
from torch_geometric.nn import  Linear , FiLMConv , TransformerConv , HGTConv , GINConv , GATConv , GCNConv , SAGEConv , MFConv
from torch_geometric.nn.dense import dense_diff_pool as DiffPool
from torch.nn import ModuleDict, ModuleList , L1Loss
from torch_geometric.nn import global_mean_pool
from torch_geometric.utils import negative_sampling
import os
import urllib.request
from urllib.error import HTTPError
import pytorch_lightning as L
import scipy.sparse
import tqdm
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.data import Data, Dataset
from pytorch_lightning.callbacks import ModelCheckpoint
from torch import Tensor
import numpy as np
import pandas as pd
import os
from Bio import PDB
import warnings
import pydssp
import polars as pl 
from Bio.PDB import PDBParser   
"""


#create a class for transforming pdb files to pyg 
class PDB2PyG:
	def __init__(self , aapropcsv = './aaindex1.csv'):
		aaproperties = pd.read_csv(aapropcsv, header=0)
		colmap = {aaproperties.columns[i]:i for i in range(len(aaproperties.columns))}
		aaproperties.drop( [ 'description' , 'reference'  ], axis=1, inplace=True)
		onehot = pd.get_dummies(aaproperties.columns.unique())
		#turn true into 1 and false into 0
		onehot = onehot.astype(int)
		aaindex = { c:onehot[c].argmax() for c in onehot.columns}
		aaproperties = pd.concat([aaproperties, onehot ] , axis = 0 )
		aaproperties = aaproperties.T
		aaproperties[aaproperties.isna() == True] = 0
		self.aaproperties = aaproperties
		self.onehot = onehot
		self.colmap = colmap
		#self.aaproperties =  pl.from_pandas(aaproperties)
		self.metadata = { 'edge_types': [ ('res','backbone','res') ,  ('res','contactPoints', 'res') , ('res','hbond', 'res') ] }
		self.aaindex = aaindex
		self.revmap_aa = {v:k for k,v in aaindex.items()}

	@staticmethod
	def read_pdb(filename):
		#silence all warnings
		warnings.filterwarnings('ignore')
		with warnings.catch_warnings():        
			parser = PDB.PDBParser()
			structure = parser.get_structure(filename, filename)
			chains = [ c for c in structure.get_chains() if len(list(c.get_residues())) > 1]
			return chains

	#return the phi, psi, and omega angles for each residue in a chain
	def get_angles(self,chain):
		aa_dict = {
		'ALA': 'A',
		'ARG': 'R',
		'ASN': 'N',
		'ASP': 'D',
		'CYS': 'C',
		'GLN': 'Q',
		'GLU': 'E',
		'GLY': 'G',
		'HIS': 'H',
		'ILE': 'I',
		'LEU': 'L',
		'LYS': 'K',
		'MET': 'M',
		'PHE': 'F',
		'PRO': 'P',
		'SER': 'S',
		'THR': 'T',
		'TRP': 'W',
		'TYR': 'Y',
		'VAL': 'V'
		}

		phi_psi_angles = []
		chain = [ r for r in chain if PDB.is_aa(r)]
		#sliding window of 3 residues
		polypeptides = [ chain[i:i+3] for i in range(len(chain)) if len(chain[i:i+4]) >= 3]
		#translate to single letter code
		residue = chain[0]
		residue_id = residue.get_full_id()
		phi_psi_angles.append({
				"Chain": residue_id[2],
				"Residue_Number": residue_id[3][1],
				"Residue_Name": residue.get_resname(),
				#translate 3 letter to 1 letter code
				"single_letter_code": aa_dict[residue.get_resname()],
				"Phi_Angle": 0,
				"Psi_Angle": 0
			})


		for poly_index, poly in enumerate(polypeptides):
			phi = None
			psi = None

			if len(poly) >= 3:
				c_minus_1 = poly[len(poly) - 3]["C"].get_vector()
				n = poly[len(poly) - 2]["N"].get_vector()
				ca = poly[len(poly) - 2]["CA"].get_vector()
				c = poly[len(poly) - 2]["C"].get_vector()

				# Calculate phi angle
				phi = PDB.calc_dihedral(c_minus_1, n, ca, c)
				n = poly[len(poly) - 2]["N"].get_vector()
				ca = poly[len(poly) - 2]["CA"].get_vector()
				c = poly[len(poly) - 2]["C"].get_vector()
				n_plus_1 = poly[len(poly) - 1]["N"].get_vector()

				# Calculate psi angle
				psi = PDB.calc_dihedral(n, ca, c, n_plus_1)
			residue = poly[0]
			residue_id = residue.get_full_id()

			phi_psi_angles.append({
				"Chain": residue_id[2],
				"Residue_Number": residue_id[3][1],
				"Residue_Name": residue.get_resname(),
				#translate 3 letter to 1 letter code
				"single_letter_code": aa_dict[residue.get_resname()],
				"Phi_Angle": phi,
				"Psi_Angle": psi
			})

		residue = chain[-1]
		residue_id = residue.get_full_id()

		phi_psi_angles.append({
				"Chain": residue_id[2],
				"Residue_Number": residue_id[3][1],
				"Residue_Name": residue.get_resname(),
				#translate 3 letter to 1 letter code
				"single_letter_code": aa_dict[residue.get_resname()],
				"Phi_Angle": 0,
				"Psi_Angle": 0
			})
		
		#transform phi and psi angles into a dataframe
		phi_psi_angles = pd.DataFrame(phi_psi_angles)
		#transform the residue names into single letter code
		return phi_psi_angles    

	@staticmethod
	def get_contact_points(chain, distance=25):
		contact_mat = np.zeros((len(chain), len(chain)))
		for i,r1 in enumerate(chain):
			for j,r2 in enumerate(chain):
				if i< j:
					if 'CA' in r1 and 'CA' in r2:
						if r1['CA'] - r2['CA'] < distance:
							contact_mat[i,j] =  r1['CA'] - r2['CA']
		contact_mat = contact_mat + contact_mat.T
		return contact_mat

	@staticmethod
	def get_contact_points_complex(chain1, chain2, distance=25):
		contact_mat = np.zeros((len(chain1), len(chain2)))
		for i,r1 in enumerate(chain1):
			for j,r2 in enumerate(chain2):
				if 'CA' in r1 and 'CA' in r2:
					if r1['CA'] - r2['CA'] < distance:
						contact_mat[i,j] =  r1['CA'] - r2['CA']
		return contact_mat


	@staticmethod
	def get_closest(chain):
		contact_mat = np.zeros((len(chain), len(chain)))
		for i,r1 in enumerate(chain):
			for j,r2 in enumerate(chain):
				contact_mat[i,j] =  r1['CB'] - r2['CB']
		#go through each row and select min
		for r in contact_mat.shape[0]:
			contact_mat[r, :][ contact_mat[r, :] != np.amin(contact_mat)] =  0
		return contact_mat

	@staticmethod
	def get_backbone(chain):
		backbone_mat = np.zeros((len(chain), len(chain)))
		backbone_rev_mat = np.zeros((len(chain), len(chain)))
		np.fill_diagonal(backbone_mat[1:], 1)
		np.fill_diagonal(backbone_rev_mat[:, 1:], 1)
		return backbone_mat, backbone_rev_mat

	@staticmethod
	def ret_hbonds(chain , verbose = False):
		#loop through all atoms in a structure
		#N,CA,C,O
		typeindex = {'N':0, 'CA':1 , 'C':2, 'O':3}
		#get the number of atoms in the chain
		#create a numpy array of zeros with the shape of (1, length, atoms, xyz)
		output = np.zeros((1, len(chain), len(typeindex), 3 ))
		for c, res in enumerate(chain):
			atoms = res.get_atoms()
			for at,atom in enumerate(atoms):
				if atom.get_name() in typeindex:
					output[ 0, c ,  typeindex[atom.get_name()] , : ]  = atom.get_coord()
		output = torch.tensor(output)
		if verbose:
			print(output.shape)
		mat =  pydssp.get_hbond_map(output[0])
		return mat

	#add the amino acid properties to the angles dataframe
	#one hot encode the amino acid properties
	
	def add_aaproperties(self, angles , verbose = False):
		if verbose == True:
			print(self.aaproperties , angles )
		nodeprops = angles.merge(self.aaproperties, left_on='single_letter_code', right_index=True, how='left')
		nodeprops = nodeprops.dropna()
		# Merge operation in Polars using join
		"""nodeprops = angles.join(
			self.aaproperties,
			left_on="single_letter_code",
			right_on=self.aaproperties.index,  # This assumes the index of aaproperties is set as a column; if not, adjust accordingly
			how="left"
		)
		
		# Dropping rows with missing values
		nodeprops = nodeprops.drop_nulls()
		nodeprops = nodeprops.to_pandas()"""
		return nodeprops

	@staticmethod
	def get_plddt(chain):
		'''
		Extracts the plddt (in the beta factor column) of the first atom of each residue in a PDB file and returns a descriptive statistics object.
		Parameters:
			pdb_path (str): The path to the PDB file.'''
		
		lddt=[]
		for res in chain:
			for at in res.get_atoms():
				lddt.append(at.get_bfactor())
				break
		return np.array([lddt]).T

	def get_delta_g(monomer):
		#calculate the delta g of the fold
		#run foldx in subprocess
		cmd = f'foldx --command=Stability --pdb={monomer}'
		os.system(cmd)
		#parse the foldx output
		
		#delete output

		#get the delta g
		return delta_g


	
	@staticmethod
	def get_positional_encoding(seq_len, d_model):
		"""
		Generates a positional encoding matrix.
		
		Args:
		seq_len: int, the length of the sequence.
		d_model: int, the dimension of the embedding.
		
		Returns:
		numpy array of shape (seq_len, d_model) representing positional encodings.
		"""
		positional_encoding = np.zeros((seq_len, d_model))
		position = np.arange(0, seq_len).reshape(-1, 1)
		div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10.0) / d_model))
		
		positional_encoding[:, 0::2] = np.sin(position * div_term)
		positional_encoding[:, 1::2] = np.cos(position * div_term)
		
		return positional_encoding
	
	@staticmethod
	def read_foldx_file(file = None , foldxdir = None , pdb = None):
		if not file and (not foldxdir or not pdb):
			raise 'provide a file, foldxdir or pdb'
		elif not file:
			file = f'{foldxdir}/{pdb}_0_ST.fxout'
		with open(file) as f:
			lines = f.read().split('\t')
			pdb = lines[0].split('/')[-1].split('.')[0]
			values = [ float( n ) for n in lines[ 1 : ] ]
		return pdb, values

	#create features from a monomer pdb file
	
	def create_features(self, monomerpdb, distance = 8, verbose = False , foldxdir = None):
		if type(monomerpdb) == str:    
			chain = self.read_pdb(monomerpdb)[0]
		else:
			chain = monomerpdb
		chain = [ r for r in chain if PDB.is_aa(r)]
		if len(chain) ==0:
			return None
		angles = self.get_angles(chain)

		bondangles = np.array(angles[['Phi_Angle', 'Psi_Angle']])

		if len(angles) ==0:
			return None
		angles = self.add_aaproperties(angles , verbose = verbose)
		angles = angles.dropna()
		angles = angles.reset_index(drop=True)
		angles = angles.set_index(['Chain', 'Residue_Number'])
		angles = angles.sort_index()
		angles = angles.reset_index()
		angles = angles.drop(['Chain', 'Residue_Number' , 'Residue_Name'], axis=1)
		
		if verbose:
			plt.imshow(angles.iloc[:,-20:])
			plt.show()
		aa = np.array(angles.iloc[:,-20:])
		contact_points = self.get_contact_points(chain, distance)
		if verbose:
			print('contacts' , contact_points.shape)
			plt.imshow(contact_points)
			plt.colorbar()
			plt.show()
		hbond_mat = np.array(self.ret_hbonds(chain, verbose))
		if verbose:
			print('hbond' , hbond_mat.shape)
			plt.imshow(hbond_mat)
			plt.colorbar()
			plt.show()

		#return the angles, amino acid properties, contact points, and hydrogen bonds
		#backbone is just the amino acid chain
		backbone , backbone_rev = self.get_backbone(chain)
		positional_encoding = self.get_positional_encoding( len(chain) , 256)

		if verbose:
			print('positions' , positional_encoding.shape)
			plt.imshow(positional_encoding)
			plt.colorbar()
			plt.show()
			
		#springmat = anm_analysis(monomerpdb)
		"""if verbose:
			print('spring' , springmat.shape)
			plt.imshow(springmat)
			plt.colorbar()
			plt.show()"""

		angles = pd.concat([angles,pd.DataFrame(positional_encoding)] , axis = 1 )
		
		
		vals = deepcopy(angles)
		vals = vals.dropna()
		vals = vals.drop( ['single_letter_code'] , axis = 1 )
		vals = vals.values
		vals = vals.astype('float32')

		if verbose:
			print('vals',vals.shape)   
			plt.imshow(vals)
			plt.colorbar()
			plt.show()


		#change the contac matrices to sparse matrices
		contact_points = sparse.csr_matrix(contact_points)
		#springmat = sparse.csr_matrix(springmat)
		backbone = sparse.csr_matrix(backbone)
		backbone_rev = sparse.csr_matrix(backbone)
		hbond_mat = sparse.csr_matrix(hbond_mat)
		plddt = self.get_plddt(chain)/100

		if verbose:
			print('plddt' , plddt.shape)
			plt.plot(plddt)
			plt.ylim([0,1])
			plt.show()

		if foldxdir:
			pdb, values = self.read_foldx_file(foldxdir = foldxdir, pdb = monomerpdb.split('/')[-1].split('.')[0])
			#add the foldx values to the features
			foldx_vals = np.array(values)
			if verbose:
				print('foldx' , foldx_vals.shape)
				print(foldx_vals)
		else:
			foldx_vals = None

		return angles, contact_points, 0 , hbond_mat, backbone , backbone_rev , positional_encoding , plddt , aa , bondangles , foldx_vals


	@staticmethod
	def sparse2pairs(sparsemat):
		sparsemat = scipy.sparse.find(sparsemat)
		return np.vstack([sparsemat[0],sparsemat[1]])

	def complex2pyg(self , pdbchain1, pdbchain2  , identifier=None,  verbose = False):
		#should be called with two biopython chains
		data = HeteroData()
		contact_points = self.get_contact_points_complex(pdbchain1, pdbchain2, distance = 15 )
		contact_points = sparse.csr_matrix(contact_points)
		contacts = self.sparse2pairs(contact_points)
		data['res','contactPointsComplex', 'res'].edge_index = torch.tensor(contacts, dtype=torch.long)
		return data
		
	def struct2pyg(self , pdbchain  , foldxdir= None , identifier=None,  verbose = False):
		data = HeteroData()
		#transform a structure chain into a pytorch geometric graph
		#get the adjacency matrices
		#try:
		xdata = self.create_features(pdbchain , verbose = verbose, foldxdir = foldxdir)
		#except:
		#	return None
		
		if xdata is not None:
			angles, contact_points, springmat , hbond_mat , backbone , backbone_rev , positional_encoding , plddt ,aa , bondangles , foldx_vals = xdata
		else:
			return None
		if len(angles) ==0:
			return None
		
		if type(pdbchain) == str:
			identifier = pdbchain.split('/')[-1].split('.')[0]
		#if the chain is a bio.pdb chain object
		elif type(pdbchain) == PDB.Chain.Chain:
			identifier = pdbchain.get_id()
		else:
			raise 'chain must be a string of a file or a Bio.PDB.Chain.Chain object'
		data.identifier = identifier

		angles = angles.drop(['single_letter_code'], axis=1)
		angles.fillna(0, inplace=True)
		#just keep the amino acid 1 hot encoding
		#add the amino acid 1 hot to dataset. use for training
		data['AA'].x = torch.tensor(aa, dtype=torch.float32)

		if foldx_vals is not None:
			#add the foldx values to the dataset
			data['Foldx'].x = torch.tensor(foldx_vals, dtype=torch.float32)

		data['bondangles'].x = torch.tensor(bondangles, dtype=torch.float32)
		data['plddt'].x = torch.tensor(plddt, dtype=torch.float32)
		data['positions'].x = torch.tensor( positional_encoding, dtype=torch.float32)
		#use the amino acid properties as the node features
		angles = torch.tensor(angles.values, dtype=torch.float32)
		data['res'].x = angles
		data['godnode'].x = torch.tensor(np.ones((1,5)), dtype=torch.float32)
		data['godnode4decoder'].x = torch.tensor(np.ones((1,5)), dtype=torch.float32)

		#get the edge features
		data['res','backbone','res'].edge_attr = torch.tensor(backbone.data, dtype=torch.float32)
		data['res','backbonerev','res'].edge_attr = torch.tensor(backbone_rev.data, dtype=torch.float32)
		data['res','contactPoints', 'res'].edge_attr = torch.tensor(contact_points.data, dtype=torch.float32)
		data['res','hbond', 'res'].edge_attr = torch.tensor(hbond_mat.data, dtype=torch.float)

		#fully_connected = sparse.csr_matrix(np.ones((len(angles), len(angles))))
		#data['res','fullyconnected','res'].edge_index = torch.tensor(self.sparse2pairs(fully_connected), dtype=torch.long)
		#data['res','springMat', 'res'].edge_attr = torch.tensor(springmat.data, dtype=torch.float32)
		
		backbone = self.sparse2pairs(backbone)
		backbone_rev = self.sparse2pairs(backbone_rev)
		contact_points = self.sparse2pairs(contact_points)
		hbond_mat = self.sparse2pairs(hbond_mat)
		#springmat = self.sparse2pairs(springmat)

		#get the adjacency matrices into tensors
		data['res','backbone','res'].edge_index = torch.tensor(backbone,  dtype=torch.long )
		data['res','backbonerev','res'].edge_index = torch.tensor(backbone_rev,  dtype=torch.long )
		data['res','contactPoints', 'res'].edge_index = torch.tensor(contact_points,  dtype=torch.long )    
		data['res','hbond', 'res'].edge_index = torch.tensor(hbond_mat,  dtype=torch.long )

		#data['res','springMat', 'res'].edge_index = torch.tensor(springmat,  dtype=torch.long )
		data['res','contactPoints', 'res'].edge_index ,  data['res','contactPoints', 'res'].edge_attr =torch_geometric.utils.to_undirected(  data['res','contactPoints', 'res'].edge_index , data['res','contactPoints', 'res'].edge_attr )
		data['res','hbond', 'res'].edge_index ,  data['res','hbond', 'res'].edge_attr =torch_geometric.utils.to_undirected(  data['res','hbond', 'res'].edge_index , data['res','hbond', 'res'].edge_attr )       
		
		# Create edges from godnode to residues
		sparse_godnode = np.vstack([[ i for i in range(len(angles))], np.zeros(len(angles))])
		#copy and flip rows
		sparse_godnode2res = np.vstack([np.zeros(len(angles)), [ i for i in range(len(angles)) ] ])



		# Assign edges
		data['res', 'informs', 'godnode'].edge_index = torch.tensor(sparse_godnode, dtype=torch.long)
		data['godnode', 'informs', 'res'].edge_index = torch.tensor(sparse_godnode2res, dtype=torch.long)

		# Repeat for godnode4decoder
		data['res', 'informs', 'godnode4decoder'].edge_index = torch.tensor(sparse_godnode, dtype=torch.long)
		data['godnode4decoder', 'informs', 'res'].edge_index = torch.tensor(sparse_godnode2res, dtype=torch.long)
		#data['godnode4decoder', 'informs', 'res'].edge_index = torch_geometric.utils.add_self_loops(data['godnode4decoder', 'informs', 'res'].edge_index)[0]

		#to undirected
		data['res','hbond', 'res'].edge_index ,  data['res','hbond', 'res'].edge_attr =torch_geometric.utils.to_undirected(  data['res','hbond', 'res'].edge_index , data['res','hbond', 'res'].edge_attr )
		data['res','backbone', 'res'].edge_index =torch_geometric.utils.to_undirected(  data['res','backbone', 'res'].edge_index )

		#add self loops
		data['res','backbone','res'].edge_index = torch_geometric.utils.add_self_loops(data['res','backbone','res'].edge_index)[0]
		data['res','backbonerev','res'].edge_index  = torch_geometric.utils.add_self_loops(data['res','backbonerev','res'].edge_index)[0]
		
		return data

	
	#create a function to store the pytorch geometric data in a hdf5 file
	def store_pyg_mp(self, pdbfiles, filename, verbose = True , ncpu = 4):
		#working but crashes after a while. cant figure out why
		#todo fix the crash
		with h5py.File(filename , mode = 'w') as f:
			#create structs list
			with pebble.ProcessPool(max_workers=ncpu) as pool:
				#map the pdb files to the struct2pyg function and get the results asynchonously
				results = pool.map( self.struct2pyg , pdbfiles , timeout=1000)                
				for pygdata in tqdm.tqdm(results.result(), total=len(pdbfiles)):
					
					if verbose:
						print(res)
						print(pdbfile)
					graph,identifier  = pygdata
					hetero_data = graph
					#hetero_data = self.struct2pyg(pdbfile, self.aaproperties)
					if hetero_data:
						f.create_group(identifier)
						for node_type in hetero_data.node_types:
							if hetero_data[node_type].x is not None:
								node_group = f.create_group(f'structs/{identifier}/node/{node_type}')
								node_group.create_dataset('x', data=hetero_data[node_type].x.numpy())
						# Iterate over edge types and their connections
						for edge_type in hetero_data.edge_types:
							# edge_type is a tuple: (src_node_type, relation_type, dst_node_type)
							edge_group = f.create_group(f'structs/{identifier}/edge/{edge_type[0]}_{edge_type[1]}_{edge_type[2]}')
							if hetero_data[edge_type].edge_index is not None:
								edge_group.create_dataset('edge_index', data=hetero_data[edge_type].edge_index.numpy())
							
							# If there are edge features, save them too
							if hasattr(hetero_data[edge_type], 'edge_attr') and hetero_data[edge_type].edge_attr is not None:
								edge_group.create_dataset('edge_attr', data=hetero_data[edge_type].edge_attr.numpy())
							#todo. store some other data. sequence. uniprot info etc.
					else:
						print('err' , pdbfile )
	
	#create a function to store the pytorch geometric data in a hdf5 file
	def store_pyg(self, pdbfiles, filename, foldxdir = None, verbose = True ):
		with h5py.File(filename , mode = 'w') as f:
			for pdbfile in  tqdm.tqdm( pdbfiles ):                    
				if verbose:
					print(pdbfile)
				hetero_data = self.struct2pyg(pdbfile , foldxdir = foldxdir)
				if hetero_data:
					identifier = hetero_data.identifier
					f.create_group(identifier)
					for node_type in hetero_data.node_types:
						if hetero_data[node_type].x is not None:
							node_group = f.create_group(f'structs/{identifier}/node/{node_type}')
							node_group.create_dataset('x', data=hetero_data[node_type].x.numpy())
					# Iterate over edge types and their connections
					for edge_type in hetero_data.edge_types:
						# edge_type is a tuple: (src_node_type, relation_type, dst_node_type)
						edge_group = f.create_group(f'structs/{identifier}/edge/{edge_type[0]}_{edge_type[1]}_{edge_type[2]}')
						if hetero_data[edge_type].edge_index is not None:
							edge_group.create_dataset('edge_index', data=hetero_data[edge_type].edge_index.numpy())
						
						# If there are edge features, save them too
						if hasattr(hetero_data[edge_type], 'edge_attr') and hetero_data[edge_type].edge_attr is not None:
							edge_group.create_dataset('edge_attr', data=hetero_data[edge_type].edge_attr.numpy())
						#todo. store some other data. sequence. uniprot info etc.
				else:
					print('err' , pdbfile )
	
	def store_pyg_complexdata(self, pdbfiles, filename, verbose = True ):
		with h5py.File(filename , mode = 'w') as f:
			for pdbfile in  tqdm.tqdm( pdbfiles ):
				#load the pdb and get chains
				try:
					chains = self.read_pdb(pdbfile)
				except:
					print('err' , pdbfile)
					continue
				
				if verbose:
					print(pdbfile)
					print( chains , [len(list(c.get_residues())) for c in chains] )
				if chains and len(chains) > 1:

					identifier = pdbfile.split('/')[-1].split('.')[0]
					
					for i,c in enumerate(chains):
						hetero_data = self.struct2pyg(c)
						if verbose:
							#print(hetero_data)
							pass
						
						if i == 0 and hetero_data:
							f.create_group(identifier)
						if hetero_data:
							for node_type in hetero_data.node_types:
								if hetero_data[node_type].x is not None:
									node_group = f.create_group(f'structs/{identifier}/chains/{i}/node/{node_type}')
									node_group.create_dataset('x', data=hetero_data[node_type].x.numpy())
							# Iterate over edge types and their connections
							for edge_type in hetero_data.edge_types:
								# edge_type is a tuple: (src_node_type, relation_type, dst_node_type)
								
								edge_group = f.create_group(f'structs/{identifier}/chains/{i}/edge/{edge_type[0]}_{edge_type[1]}_{edge_type[2]}')
								if hetero_data[edge_type].edge_index is not None:
									edge_group.create_dataset('edge_index', data=hetero_data[edge_type].edge_index.numpy())
								# If there are edge features, save them too
								if hasattr(hetero_data[edge_type], 'edge_attr') and hetero_data[edge_type].edge_attr is not None:
									edge_group.create_dataset('edge_attr', data=hetero_data[edge_type].edge_attr.numpy())
					for i,c1 in enumerate(chains):
						for j,c2 in enumerate(chains):
							if i < j:
								hetero_data = self.complex2pyg(c1, c2 )
								if hetero_data and hetero_data['res','contactPointsComplex', 'res'].edge_index.shape[1] > 0:
									if verbose:
										print('complex',hetero_data)
									
									for node_type in hetero_data.node_types:
										if hetero_data[node_type].x is not None:
											node_group = f.create_group(f'structs/{identifier}/complex/{i}_{j}/node/{node_type}')
											node_group.create_dataset('x', data=hetero_data[node_type].x.numpy())
									# Iterate over edge types and their connections
									for edge_type in hetero_data.edge_types:
										# edge_type is a tuple: (src_node_type, relation_type, dst_node_type)
										edge_group = f.create_group(f'structs/{identifier}/complex/{i}_{j}/edge/{edge_type[0]}_{edge_type[1]}_{edge_type[2]}')
										if hetero_data[edge_type].edge_index is not None:
											edge_group.create_dataset('edge_index', data=hetero_data[edge_type].edge_index.numpy())
										
										# If there are edge features, save them too
										if hasattr(hetero_data[edge_type], 'edge_attr') and hetero_data[edge_type].edge_attr is not None:
											edge_group.create_dataset('edge_attr', data=hetero_data[edge_type].edge_attr.numpy())
										#todo. store some other data. sequence. uniprot info etc.
								
				else:
					print('err' , pdbfile , 'not multichain')

class ComplexDataset(Dataset):
	def __init__(self, h5dataset  ):
		super().__init__()
		#keys should be the structures


		self.h5dataset = h5dataset

		if type(h5dataset) == str:
			self.h5dataset = h5py.File(h5dataset, 'r')
		self.structlist = list(self.h5dataset['structs'].keys())
	
	def __len__(self):
		return len(self.structlist)

	def __getitem__(self, idx):
		if type(idx) == str:
			f = self.h5dataset['structs'][idx]
		elif type(idx) == int:
			f = self.h5dataset['structs'][self.structlist[idx]]
		else:
			raise 'use a structure filename or integer'
		chaindata = {}

		chains = [ c for c in f['chains'].keys()]
		for chain in chains:
			hetero_data = HeteroData()        
			if type (idx) == int:
				hetero_data.identifier = self.structlist[idx]
			else:
				hetero_data.identifier = idx


			if 'node' in f['chains'][chain].keys():
				for node_type in f['chains'][chain]['node'].keys():
					node_group = f['chains'][chain]['node'][node_type]
					# Assuming 'x' exists
					if 'x' in node_group.keys():
						hetero_data[node_type].x = torch.tensor(node_group['x'][:])
			# Edge data
			if 'edge' in f['chains'][chain].keys():
				for edge_name in f['chains'][chain]['edge'].keys():
					edge_group = f['chains'][chain]['edge'][edge_name]
					src_type, link_type, dst_type = edge_name.split('_')
					edge_type = (src_type, link_type, dst_type)
					# Assuming 'edge_index' exists
					if 'edge_index' in edge_group.keys():
						hetero_data[edge_type].edge_index = torch.tensor(edge_group['edge_index'][:])
					
					# If there are edge attributes, load them too
					if 'edge_attr' in edge_group.keys():
						hetero_data[edge_type].edge_attr = torch.tensor(edge_group['edge_attr'][:])
			chaindata[chain] = hetero_data
		
		pairdata = {}
		pairs = [ c for c in f['complex'].keys()]
		for pair in pairs:
			hetero_data = HeteroData()        
			if type (idx) == int:
				hetero_data.identifier = self.structlist[idx]
			else:
				hetero_data.identifier = idx
			if 'node' in f['complex'][pair].keys():
				for node_type in f['complex'][pair]['node'].keys():
					node_group = f['complex'][pair]['node'][node_type]
					# Assuming 'x' exists
					if 'x' in node_group.keys():
						hetero_data[node_type].x = torch.tensor(node_group['x'][:])
			# Edge data
			if 'edge' in f['complex'][pair].keys():
				for edge_name in f['complex'][pair]['edge'].keys():
					edge_group = f['complex'][pair]['edge'][edge_name]
					src_type, link_type, dst_type = edge_name.split('_')
					edge_type = (src_type, link_type, dst_type)
					# Assuming 'edge_index' exists
					if 'edge_index' in edge_group.keys():
						hetero_data[edge_type].edge_index = torch.tensor(edge_group['edge_index'][:])
					
					# If there are edge attributes, load them too
					if 'edge_attr' in edge_group.keys():
						hetero_data[edge_type].edge_attr = torch.tensor(edge_group['edge_attr'][:])
			pairdata[pair] = hetero_data
		return chaindata, pairdata
	
class StructureDataset(Dataset):
	def __init__(self, h5dataset  ):
		super().__init__()
		#keys should be the structures


		self.h5dataset = h5dataset

		if type(h5dataset) == str:
			self.h5dataset = h5py.File(h5dataset, 'r')
		self.structlist = list(self.h5dataset['structs'].keys())
	
	def __len__(self):
		return len(self.structlist)

	def __getitem__(self, idx):
		if type(idx) == str:
			f = self.h5dataset['structs'][idx]
		elif type(idx) == int:
			f = self.h5dataset['structs'][self.structlist[idx]]
		else:
			raise 'use a structure filename or integer'
		data = {}
		hetero_data = HeteroData()
		
		if type (idx) == int:
			hetero_data.identifier = self.structlist[idx]
		else:
			hetero_data.identifier = idx

		if 'node' in f.keys():
			for node_type in f['node'].keys():
				node_group = f['node'][node_type]
				# Assuming 'x' exists
				if 'x' in node_group.keys():
					hetero_data[node_type].x = torch.tensor(node_group['x'][:])
		# Edge data
		if 'edge' in f.keys():
			for edge_name in f['edge'].keys():
				edge_group = f['edge'][edge_name]
				src_type, link_type, dst_type = edge_name.split('_')
				edge_type = (src_type, link_type, dst_type)
				# Assuming 'edge_index' exists
				if 'edge_index' in edge_group.keys():
					hetero_data[edge_type].edge_index = torch.tensor(edge_group['edge_index'][:])
				
				# If there are edge attributes, load them too
				if 'edge_attr' in edge_group.keys():
					hetero_data[edge_type].edge_attr = torch.tensor(edge_group['edge_attr'][:])
		#return pytorch geometric heterograph
		return hetero_data
	
class VectorQuantizerEMA(nn.Module):
	def __init__(self, num_embeddings, embedding_dim, commitment_cost, decay=0.99 , epsilon=1e-5, reset_threshold=100000, reset = True , klweight = 1 , diversityweight= 1 , entropyweight = 1 ):
		super(VectorQuantizerEMA, self).__init__()
		self.embedding_dim = embedding_dim
		self.num_embeddings = num_embeddings
		self.commitment_cost = commitment_cost
		self.decay = decay
		self.epsilon = epsilon
		self.reset_threshold = reset_threshold
		self.reset = reset
		# Initialize the codebook with uniform distribution
		self.diversityweight = diversityweight
		self.klweight= klweight
		self.entropyweight = entropyweight

		self.embeddings = nn.Embedding(num_embeddings, embedding_dim)
		self.embeddings.weight.data.uniform_(-1 / self.num_embeddings, 1 / self.num_embeddings)
		self.entropyweight= entropyweight
		self.diversityweight = diversityweight
		self.klweight = klweight

		# EMA variables
		self.register_buffer('ema_cluster_size', torch.zeros(num_embeddings))
		self.ema_w = nn.Parameter(self.embeddings.weight.clone())

		# Track usage of embeddings
		self.register_buffer('embedding_usage_count', torch.zeros(num_embeddings, dtype=torch.long))

	def forward(self, x):
		# Flatten input
		flat_x = x.view(-1, self.embedding_dim)

		# Compute distances between input and codebook embeddings
		distances = (torch.sum(flat_x**2, dim=1, keepdim=True)
					 + torch.sum(self.embeddings.weight**2, dim=1)
					 - 2 * torch.matmul(flat_x, self.embeddings.weight.t()))

		# Get the encoding that has the minimum distance
		encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
		encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=x.device)
		encodings.scatter_(1, encoding_indices, 1)

		# Quantize the latents by mapping to the nearest embeddings
		quantized = torch.matmul(encodings, self.embeddings.weight).view_as(x)

		# Compute the commitment loss
		e_latent_loss = F.mse_loss(quantized.detach(), x)
		q_latent_loss = F.mse_loss(quantized, x.detach())
		loss = q_latent_loss + self.commitment_cost * e_latent_loss

		# Regularization
		entropy_reg = entropy_regularization(encodings)
		diversity_reg = diversity_regularization(encodings)
		kl_div_reg = kl_divergence_regularization(encodings)

		# Combine all losses
		total_loss = loss - self.entropyweight*entropy_reg + self.diversityweight*diversity_reg + self.klweight*kl_div_reg

		# EMA updates
		if self.training:
			encodings_sum = encodings.sum(0)
			dw = torch.matmul(encodings.t(), flat_x)

			self.ema_cluster_size = self.ema_cluster_size * self.decay + (1 - self.decay) * encodings_sum
			self.ema_w = nn.Parameter(self.ema_w * self.decay + (1 - self.decay) * dw)

			n = self.ema_cluster_size.sum()
			self.ema_cluster_size = ((self.ema_cluster_size + self.epsilon) / (n + self.num_embeddings * self.epsilon) * n)

			self.embeddings.weight.data = self.ema_w / self.ema_cluster_size.unsqueeze(1)

			# Update usage count
			self.embedding_usage_count += encodings_sum.long()
			
			if self.reset== True:
				# Reset unused embeddings
				self.reset_unused_embeddings()

		# Straight-through estimator for the backward pass
		quantized = x + (quantized - x).detach()

		return quantized, total_loss

	def reset_unused_embeddings(self):
		"""
		Resets the embeddings that have not been used for a certain number of iterations.
		"""
		unused_embeddings = self.embedding_usage_count < self.reset_threshold
		num_resets = unused_embeddings.sum().item()
		if num_resets > 0:
			with torch.no_grad():
				self.embeddings.weight[unused_embeddings] = torch.randn((num_resets, self.embedding_dim), device=self.embeddings.weight.device)
			# Reset usage counts for the reset embeddings
			self.embedding_usage_count[unused_embeddings] = 0

	def discretize_z(self, x):
		# Flatten input
		flat_x = x.view(-1, self.embedding_dim)
		# Compute distances between input and codebook embeddings
		distances = (torch.sum(flat_x**2, dim=1, keepdim=True)
					 + torch.sum(self.embeddings.weight**2, dim=1)
					 - 2 * torch.matmul(flat_x, self.embeddings.weight.t()))
		# Get the encoding that has the minimum distance
		closest_indices = torch.argmin(distances, dim=1)
		
		# Convert indices to characters
		char_list = [chr(idx.item()) for idx in closest_indices]
		return closest_indices, char_list

	def string_to_hex(self, s):
		# if string is ascii, convert to hex
		if all(ord(c) < 248 for c in s):
			return s.encode().hex()
		else:
			#throw an error
			raise ValueError('String contains non-ASCII characters')
		
	def string_to_embedding(self, s):
		
		# Convert characters back to indices
		indices = torch.tensor([ord(c) for c in s], dtype=torch.long, device=self.embeddings.weight.device)
		
		# Retrieve embeddings from the codebook
		embeddings = self.embeddings(indices)
		return embeddings
	
	def ord_to_embedding(self, s):
		# Convert characters back to indices
		indices = torch.tensor([c for c in s], dtype=torch.long, device=self.embeddings.weight.device)
		# Retrieve embeddings from the codebook
		embeddings = self.embeddings(indices)
		return embeddings


# Define the regularization functions outside the class

def entropy_regularization(encodings):
	probabilities = encodings.mean(dim=0)
	entropy = -torch.sum(probabilities * torch.log(probabilities + 1e-10))
	return entropy

def diversity_regularization(encodings):
	probabilities = encodings.mean(dim=0)
	diversity_loss = torch.sum((probabilities - 1 / probabilities.size(0)) ** 2)
	return diversity_loss

def kl_divergence_regularization(encodings):
	probabilities = encodings.mean(dim=0)
	kl_divergence = torch.sum(probabilities * torch.log(probabilities * probabilities.size(0) + 1e-10))
	return kl_divergence


# residual block feed forward
class ResidualBlockFF(torch.nn.Module):
	def __init__(self, in_channels, hidden_channels , outchannels , nlayers = 2):
		super(ResidualBlockFF, self).__init__()
		self.ff_stack1 = torch.nn.Sequential( 
			Linear(in_channels, hidden_channels),
			torch.nn.ReLU(),
			#add nlayers
			*[torch.nn.Sequential(Linear(hidden_channels, hidden_channels), torch.nn.ReLU()) for i in range(nlayers)]
			, Linear(hidden_channels, in_channels) , torch.nn.ReLU()
			)
		self.final = torch.nn.Linear(in_channels, outchannels)

	def forward(self, x):
		xcopy = x
		x = self.ff_stack1(x)
		x = x+xcopy
		x = self.final(x)
		return F.relu(x)
	
from torch_geometric.nn import HeteroConv
class HeteroGAE_Encoder(torch.nn.Module):
	def __init__(self, in_channels, hidden_channels, layers ,  out_channels, num_embeddings, commitment_cost, metadata={} , encoder_hidden = 100 , dropout_p = 0.01 , EMA = False , average = False, reset_codes = True , nheads = 3 , separated = False , flavor = 'transformer'):
		super(HeteroGAE_Encoder, self).__init__()

		#save all arguments to constructor
		self.args = locals()
		self.args.pop('self')

		# Setting the seed
		L.seed_everything(42)
		# Ensure that all operations are deterministic on GPU (if used) for reproducibility
		torch.backends.cudnn.deterministic = True
		torch.backends.cudnn.benchmark = False        
		self.convs = torch.nn.ModuleList()

		self.metadata = metadata
		self.hidden_channels = hidden_channels
		self.out_channels = out_channels
		self.in_channels = in_channels
		self.encoder_hidden = encoder_hidden
		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		#batch norm
		self.nlayers = layers
		self.bn = torch.nn.BatchNorm1d(in_channels['res'])
		self.dropout = torch.nn.Dropout(p=dropout_p)
		self.separated = separated
		self.average = average
		
		for i in range(layers):
			layer = {}          
			for k,edge_type in enumerate( hidden_channels.keys() ):
				edgestr = '_'.join(edge_type)

				datain = edge_type[0]
				dataout = edge_type[2]

				#if ( 'res','informs','godnode4decoder') == edge_type:
				#	layer[edgestr] = TransformerConv( in_channels[datain] , hidden_channels[edge_type][i], heads = nheads , concat= False)
				#else:
				if flavor == 'transformer' or edge_type == ('res','informs','godnode'):
					layer[edge_type] = TransformerConv( (-1, -1) , hidden_channels[edge_type][i], heads = nheads , concat= False)
				if flavor == 'gat':
					layer[edge_type] = GATConv( (-1, -1) , hidden_channels[edge_type][i], heads = nheads )
				if flavor == 'gcn':
					layer[edge_type] = GCNConv( (-1, -1) , hidden_channels[edge_type][i])
				if flavor == 'sage':
					layer[edge_type] = SAGEConv( (-1, -1) , hidden_channels[edge_type][i])
				if flavor == 'mfconv':
					layer[edge_type] = MFConv( (-1, -1)  , hidden_channels[edge_type][i])  
				if flavor == 'FiLM':
					layer[edge_type] = FiLMConv( in_channels[datain] , hidden_channels[edge_type][i],                                               
					nn = 
					torch.nn.Sequential(
					torch.nn.Linear( in_channels[datain] , hidden_channels[edge_type][i]),
					torch.nn.ReLU(),
					torch.nn.Linear(hidden_channels[edge_type][i] , hidden_channels[edge_type][i] ) , 
					torch.nn.ReLU(),
					torch.nn.Linear(hidden_channels[edge_type][i] , 2 * hidden_channels[edge_type][i] ) , 
					torch.nn.ReLU() )
					)

				if flavor == 'GIN':
					layer[edge_type] = GINConv(                              
					nn = 
					torch.nn.Sequential(
					torch.nn.Linear( in_channels[datain] , hidden_channels[edge_type][i]),
					torch.nn.ReLU(),
					torch.nn.Linear(hidden_channels[edge_type][i] , hidden_channels[edge_type][i] ) , 
					torch.nn.ReLU(),
					torch.nn.Linear(hidden_channels[edge_type][i] ,  hidden_channels[edge_type][i] ) , 
					torch.nn.ReLU() )
					)
				if ( 'res','backbone','res') == edge_type and i > 0:
					in_channels['res'] = hidden_channels[( 'res','backbone','res')][i-1] + in_channels['godnode']
				else:
					if k == 0 and i == 0:
						in_channels[dataout] = hidden_channels[edge_type][i]
					if k == 0 and i > 0:
						in_channels[dataout] = hidden_channels[edge_type][i-1]
					if k > 0 and i > 0:                    
						in_channels[dataout] = hidden_channels[edge_type][i]
					if k > 0 and i == 0:
						in_channels[dataout] = hidden_channels[edge_type][i]
			
			conv = HeteroConv( layer  , aggr='mean')
			self.convs.append( conv )
		
		print('encoder convolutions')
		print(self.convs)
	
		#output dense layer
		self.out_dense = self.out_dense= torch.nn.Sequential(
			Linear(hidden_channels[('res', 'contactPoints', 'res')][-1] + 20 , encoder_hidden),
			torch.nn.SiLU(),
			Linear(encoder_hidden, encoder_hidden),
			torch.nn.SiLU(),
			Linear(encoder_hidden, encoder_hidden),
			torch.nn.SiLU(),
			Linear(encoder_hidden, encoder_hidden),
			torch.nn.SiLU(),
			Linear(encoder_hidden, encoder_hidden),
			torch.nn.SiLU(),
			Linear(encoder_hidden, encoder_hidden),
			torch.nn.SiLU(),
			Linear(encoder_hidden, encoder_hidden),
			torch.nn.SiLU(),
			Linear(encoder_hidden, out_channels),
			torch.nn.SiLU()
			)
		
		#initialize the vector quantizer
		self.vector_quantizer = VectorQuantizerEMA(num_embeddings, out_channels, commitment_cost , reset = reset_codes)
	
	def forward(self, xdata, edge_index):

		#xdata = {key: xdata[key] for key in ['res']}
		#edge_index_dict = {key: edge_index_dict[key] for key in [ ('res','backbone','res') ,  ('res','contactPoints', 'res') ] }
		xdata['res'] = self.bn(xdata['res'])
		xdata['res'] = self.dropout(xdata['res'])
		xaa = xdata['AA']
		for i,layer in enumerate(self.convs):
			xdata = layer(xdata, edge_index)
			#add relu to all convolutions in this layer
			for key in layer.convs.keys():
				key = key[2]
				xdata[key] = F.relu(xdata[key])
		xres = xdata['res']
		x = self.out_dense( torch.cat([xres,xaa], dim=1) )
		#x = self.out_dense( x )
		z_quantized,  qloss = self.vector_quantizer(x)
		return z_quantized, qloss

	def encode_structures( dataloader, encoder, filename = 'structalign.strct' ):
		#write with contacts 
		with open( filename , 'w') as f:
			for i,data in tqdm.tqdm(enumerate(dataloader)):
				data = data.to(self.device)
				z,qloss = self.forward(data.x_dict , data.edge_index_dict)
				strdata = self.vector_quantizer.discretize_z(z)
				identifier = structlist[i]
				f.write(f'\n//////startprot//////{identifier}//////\n')
				for char in strdata[1]:
					f.write(char)
				f.write(f'\n//////contacts//////{identifier}//////\n')
				#write the contacts stored in the data object
				contacts = data.edge_index_dict[( 'res','contactPoints','res')]
				#write a json object with the contacts
				contacts = contacts.detach().cpu().numpy()
				#convert edge index to a json object
				contacts = contacts.tolist()
				f.write(json.dumps(contacts))
				f.write(f'\n//////endprot//////\n')
				f.write('\n')
		return filename

	def encode_structures_fasta(self, dataloader, filename = 'structalign.strct.fasta' , verbose = False):
		#write an encoded fasta for use with mafft and raxml. only doable with alphabet size of less that 248
		#0x01 – 0xFF excluding > (0x3E), = (0x3D), < (0x3C), - (0x2D), Space (0x20), Carriage Return (0x0d) and Line Feed (0x0a)
		replace_dict = {chr(0):chr(246) , '"':chr(248) , '#':chr(247), '>' : chr(249), '=' : chr(250),
		 '<' : chr(251), '-' : chr(252), ' ' : chr(253) , '\r' : chr(254), '\n' : chr(255) }
		
		#check encoding size
		if self.vector_quantizer.num_embeddings > 248:
			raise ValueError('Encoding size too large for fasta encoding')
		
		with open( filename , 'w') as f:
			for i,data in tqdm.tqdm(enumerate(dataloader)):
				data = data.to(self.device)
				z,qloss = self.forward(data.x_dict, data.edge_index_dict)
				strdata = self.vector_quantizer.discretize_z(z)
				identifier = data.identifier
				f.write(f'>{identifier}\n')
				outstr = ''
				for char in strdata[0]:
					char = chr(char)
					if char in replace_dict:
						char = replace_dict[char]
					outstr += char
					f.write(char)
				f.write('\n')
				if verbose == True:
					print(identifier, outstr)
		return filename
	
	def encode_structures_numbers(self, dataloader, filename = 'structalign.strct.fasta' ):
		#write an encoded fasta with just numbers of the discrete characters
		#check encoding size
		if self.vector_quantizer.num_embeddings > 248:
			raise ValueError('Encoding size too large for fasta encoding')
		with open( filename , 'w') as f:
			for i,data in tqdm.tqdm(enumerate(dataloader)):
				data = data.to(self.device)
				z,qloss = self.forward(data.x_dict, data.edge_index_dict)
				strdata = self.vector_quantizer.discretize_z(z)
				identifier = data.identifier
				f.write(f'\n>{identifier}\n')
				for num in strdata[0]:
					f.write(str(num)+ ',')
				f.write('\n')
		return filename


	def load(self, modelfile):
		self.load_state_dict(torch.load(modelfile))
		self.eval()
		return self

	def save(self, modelfile):
		torch.save(self.state_dict(), modelfile)
		return modelfile
	
	def ret_config(self):
		return {'in_channels': self.in_channels, 'hidden_channels': self.hidden_channels, 'out_channels': self.out_channels, 'num_embeddings': self.vector_quantizer.num_embeddings, 'commitment_cost': self.vector_quantizer.commitment_cost, 'metadata': self.metadata}

	def save_config(self, configfile):
		with open(configfile , 'w') as f:
			json.dump(self.ret_config(), f)
		return configfile

	def load_from_config(config):
		return HeteroGAE_Encoder(**config)
	
	def encode_structures( dataloader, encoder, filename = 'structalign.strct' ):
		#write with contacts 
		with open( filename , 'w') as f:
			for i,data in tqdm.tqdm(enumerate(dataloader)):
				data = data.to(self.device)
				z,qloss = self.forward(data.x_dict , data.edge_index_dict)
				strdata = self.vector_quantizer.discretize_z(z)
				identifier = structlist[i]
				f.write(f'\n//////startprot//////{identifier}//////\n')
				for char in strdata[1]:
					f.write(char)
				f.write(f'\n//////contacts//////{identifier}//////\n')
				#write the contacts stored in the data object
				contacts = data.edge_index_dict[( 'res','contactPoints','res')]
				#write a json object with the contacts
				contacts = contacts.detach().cpu().numpy()
				#convert edge index to a json object
				contacts = contacts.tolist()
				f.write(json.dumps(contacts))
				f.write(f'\n//////endprot//////\n')
				f.write('\n')
		return filename

	def encode_structures_fasta(self, dataloader, filename = 'structalign.strct.fasta' , verbose = False , alphabet = None , replace = False):
		#write an encoded fasta for use with mafft and iqtree. only doable with alphabet size of less that 248
		#0x01 – 0xFF excluding > (0x3E), = (0x3D), < (0x3C), - (0x2D), Space (0x20), Carriage Return (0x0d) and Line Feed (0x0a)
		replace_dict = { '>' : chr(249), '=' : chr(250), '<' : chr(251), '-' : chr(252), ' ' : chr(253) , '\r' : chr(254), '\n' : chr(255) }
		#check encoding size
		if self.vector_quantizer.num_embeddings > 248:
			raise ValueError('Encoding size too large for fasta encoding')
		
		if alphabet is not None:
			print('using alphabet')
			print(alphabet)
		
		with open( filename , 'w') as f:
			for i,data in tqdm.tqdm(enumerate(dataloader)):
				data = data.to(self.device)
				z,qloss = self.forward(data.x_dict , data.edge_index_dict)
				strdata = self.vector_quantizer.discretize_z(z)
				identifier = data.identifier
				f.write(f'>{identifier}\n')
				outstr = ''
				for char in strdata[0]:
					#start at 0x01
					if alphabet is not None:
						char = alphabet[char]
					else:
						char = chr(char+1)
					
					if replace and char in replace_dict:
						char = replace_dict[char]
					outstr += char
					f.write(char)

				f.write('\n')

				if verbose == True:
					print(identifier, outstr)
		return filename
	
	def encode_structures_numbers(self, dataloader, filename = 'structalign.strct.fasta' ):
		#write an encoded fasta with just numbers of the discrete characters
		#check encoding size
		if self.vector_quantizer.num_embeddings > 248:
			raise ValueError('Encoding size too large for fasta encoding')
		with open( filename , 'w') as f:
			for i,data in tqdm.tqdm(enumerate(dataloader)):
				data = data.to(self.device)
				z,qloss = self.forward(data.x_dict, data.edge_index_dict)
				strdata = self.vector_quantizer.discretize_z(z)
				identifier = data.identifier
				f.write(f'\n>{identifier}\n')
				for num in strdata[0]:
					f.write(str(num)+ ',')
				f.write('\n')
		return filename


	def load(self, modelfile):
		self.load_state_dict(torch.load(modelfile))
		self.eval()
		return self

	def save(self, modelfile):
		torch.save(self.state_dict(), modelfile)
		return modelfile
	
	def ret_config(self):
		return {'in_channels': self.in_channels, 'hidden_channels': self.hidden_channels, 'out_channels': self.out_channels, 'num_embeddings': self.vector_quantizer.num_embeddings, 'commitment_cost': self.vector_quantizer.commitment_cost, 'metadata': self.metadata}

	def save_config(self, configfile):
		with open(configfile , 'w') as f:
			json.dump(self.ret_config(), f)
		return configfile

	def load_from_config(config):
		return HeteroGAE_Encoder(**config)


class TransformerDecoder(nn.Module):
	def __init__(self, embedding_dim, num_layers=6, nhead=8, dropout=0.1, seq_len=128):
		super(TransformerDecoder, self).__init__()
		self.embedding_dim = embedding_dim
		self.seq_len = seq_len

		# Setting the seed
		L.seed_everything(42)
		# Ensure that all operations are deterministic on GPU (if used) for reproducibility
		torch.backends.cudnn.deterministic = True
		torch.backends.cudnn.benchmark = False
		self.convs = torch.nn.ModuleList()

		#self.bn = torch.nn.BatchNorm1d(encoder_out_channels)
		self.metadata = metadata
		self.hidden_channels = hidden_channels
		self.in_channels = encoder_out_channels
		self.amino_acid_indices = amino_mapper
		self.bn = torch.nn.BatchNorm1d(encoder_out_channels)
		self.revmap_aa = { v:k for k,v in amino_mapper.items() }
		self.dropout = torch.nn.Dropout(p=dropout)
		
		# Transformer decoder
		self.transformer = nn.Transformer(
			d_model=embedding_dim,
			nhead=nhead,
			num_encoder_layers=num_layers,
			num_decoder_layers=num_layers,
			dropout=dropout
		)
		
		# Linear projection from transformer output to sequence logits (e.g., for amino acids)
		self.to_sequence = nn.Linear(embedding_dim, 20)  # 20 for amino acid sequence classification (AAs)
		
		# Optional contact prediction head
		self.contact_head = nn.Linear(embedding_dim, 1)  # Sigmoid for binary contact prediction



	def forward(self, quantized_latents):
		"""
		Args:
			quantized_latents (Tensor): Quantized embeddings from the encoder (VQ output), shape (batch_size, seq_len, embedding_dim)
		Returns:
			sequence_logits (Tensor): Logits for sequence reconstruction (batch_size, seq_len, 20)
			contact_map (Tensor): Predicted contact map (batch_size, seq_len, seq_len)
		"""
		# Apply transformer decoder
		# (For simplicity, we'll use quantized_latents as both encoder and decoder input in the transformer)
		transformer_output = self.transformer(quantized_latents, quantized_latents)
		
		# Project to sequence space (e.g., amino acids)
		sequence_logits = self.to_sequence(transformer_output)
		
		# Predict contacts
		batch_size, seq_len, _ = transformer_output.shape
		contact_map = torch.sigmoid(self.contact_head(transformer_output)).view(batch_size, seq_len, 1)  # Pairwise contacts

		return sequence_logits, contact_map


class HeteroGAE_Decoder(torch.nn.Module):
	def __init__(self, in_channels = {'res':10 , 'godnode4decoder':5 , 'foldx':23}, xdim=20, hidden_channels={'res_backbone_res': [20, 20, 20]}, layers = 3,  AAdecoder_hidden = 20 
			  ,PINNdecoder_hidden = 10, contactdecoder_hidden = 10, nheads = 3 , Xdecoder_hidden=30, metadata={}, amino_mapper= None  , flavor = None, dropout= .1):
		super(HeteroGAE_Decoder, self).__init__()
		# Setting the seed
		L.seed_everything(42)
		# Ensure that all operations are deterministic on GPU (if used) for reproducibility
		torch.backends.cudnn.deterministic = True
		torch.backends.cudnn.benchmark = False
		self.convs = torch.nn.ModuleList()
		
		in_channels_orig = copy.deepcopy(in_channels )

		#self.bn = torch.nn.BatchNorm1d(encoder_out_channels)
		self.metadata = metadata
		self.hidden_channels = hidden_channels
		self.in_channels = in_channels
		self.amino_acid_indices = amino_mapper
		self.nlayers = layers
		self.bn = torch.nn.BatchNorm1d(in_channels['res'])

		self.revmap_aa = { v:k for k,v in amino_mapper.items() }
		self.dropout = torch.nn.Dropout(p=dropout)
		
		for i in range(layers):
			layer = {}          
			for k,edge_type in enumerate( hidden_channels.keys() ):
				edgestr = '_'.join(edge_type)

				datain = edge_type[0]
				dataout = edge_type[2]

				#if ( 'res','informs','godnode4decoder') == edge_type:
				#	layer[edgestr] = TransformerConv( in_channels[datain] , hidden_channels[edge_type][i], heads = nheads , concat= False)
				#else:
				if flavor == 'transformer' or edge_type == ('res','informs','godnode4decoder'):
					layer[edge_type] = TransformerConv( (-1, -1) , hidden_channels[edge_type][i], heads = nheads , concat= False)
				if flavor == 'gat':
					layer[edge_type] = GATConv( (-1, -1) , hidden_channels[edge_type][i], heads = nheads )
				if flavor == 'gcn':
					layer[edge_type] = GCNConv( (-1, -1) , hidden_channels[edge_type][i])
				if flavor == 'sage':
					layer[edge_type] = SAGEConv( (-1, -1) , hidden_channels[edge_type][i])
				if flavor == 'mfconv':
					layer[edge_type] = MFConv( (-1, -1)  , hidden_channels[edge_type][i])  
				if flavor == 'FiLM':
					layer[edge_type] = FiLMConv( in_channels[datain] , hidden_channels[edge_type][i],                                               
					nn = 
					torch.nn.Sequential(
					torch.nn.Linear( in_channels[datain] , hidden_channels[edge_type][i]),
					torch.nn.ReLU(),
					torch.nn.Linear(hidden_channels[edge_type][i] , hidden_channels[edge_type][i] ) , 
					torch.nn.ReLU(),
					torch.nn.Linear(hidden_channels[edge_type][i] , 2 * hidden_channels[edge_type][i] ) , 
					torch.nn.ReLU() )
					)

				if flavor == 'GIN':
					layer[edge_type] = GINConv(                              
					nn = 
					torch.nn.Sequential(
					torch.nn.Linear( in_channels[datain] , hidden_channels[edge_type][i]),
					torch.nn.ReLU(),
					torch.nn.Linear(hidden_channels[edge_type][i] , hidden_channels[edge_type][i] ) , 
					torch.nn.ReLU(),
					torch.nn.Linear(hidden_channels[edge_type][i] ,  hidden_channels[edge_type][i] ) , 
					torch.nn.ReLU() )
					)

				
				if ( 'res','backbone','res') == edge_type and i > 0:
					in_channels['res'] = hidden_channels[( 'res','backbone','res')][i-1] + in_channels['godnode4decoder']
				else:
					if k == 0 and i == 0:
						in_channels[dataout] = hidden_channels[edge_type][i]
					if k == 0 and i > 0:
						in_channels[dataout] = hidden_channels[edge_type][i-1]
					if k > 0 and i > 0:                    
						in_channels[dataout] = hidden_channels[edge_type][i]
					if k > 0 and i == 0:
						in_channels[dataout] = hidden_channels[edge_type][i]
			

			conv = HeteroConv( layer  , aggr='mean')
			self.convs.append( conv )
		
		print('decoder convs')
		print( self.convs)

		print( 'batchnorm' , self.bn)
		print( 'dropout' , self.dropout)
		


		self.sigmoid = nn.Sigmoid()

		self.lin = torch.nn.Sequential(
				torch.nn.Linear( self.hidden_channels[('res', 'backbone', 'res')][-1] , Xdecoder_hidden),
		)
		
		print( in_channels_orig['res'] )  
		print( Xdecoder_hidden)

		self.aadecoder = torch.nn.Sequential(
				torch.nn.Linear(in_channels_orig['res'] + Xdecoder_hidden , AAdecoder_hidden[0]),
				torch.nn.ReLU(),
				torch.nn.Linear(AAdecoder_hidden[0], AAdecoder_hidden[1] ) ,
				torch.nn.ReLU(),
				torch.nn.Linear(AAdecoder_hidden[1], AAdecoder_hidden[2] ) ,
				torch.nn.ReLU(),

				torch.nn.Linear(AAdecoder_hidden[2], xdim),
				torch.nn.LogSoftmax(dim=1) )
		
		self.contactdecoder = torch.nn.Sequential(
				torch.nn.Linear( Xdecoder_hidden , contactdecoder_hidden[0]),
				torch.nn.ReLU(),
				torch.nn.Linear(contactdecoder_hidden[0], contactdecoder_hidden[1] ) ,
				torch.nn.ReLU(),
				torch.nn.Linear(contactdecoder_hidden[1], contactdecoder_hidden[2] ) ,
				torch.nn.ReLU(),

				torch.nn.Linear(contactdecoder_hidden[2], Xdecoder_hidden),
				)
		

		self.godnodedecoder = torch.nn.Sequential(
				torch.nn.Linear(in_channels['godnode4decoder'] , PINNdecoder_hidden[0]),
				torch.nn.ReLU(),
				torch.nn.Linear(PINNdecoder_hidden[0], PINNdecoder_hidden[1] ) ,
				torch.nn.ReLU(),
				torch.nn.Linear(PINNdecoder_hidden[1], PINNdecoder_hidden[2] ) ,
				torch.nn.ReLU(),
				torch.nn.Linear(PINNdecoder_hidden[2], in_channels['foldx']) )
		
		print('aadecoder', self.aadecoder)
		print('lin' ,  self.lin)
		print( 'sigmoid' ,  self.sigmoid)

	def forward(self, z , xdata, edge_index, contact_pred_index, **kwargs):
		z = self.bn(z)
		#copy z for later concatenation
		inz = z
		xdata['res'] = z
		for i,layer in enumerate(self.convs):
			xdata = layer(xdata, edge_index)
			for key in layer.convs.keys():
				key = key[2]
				xdata[key] = F.relu(xdata[key])
			#context = xdata['godnode4decoder'].repeat(xdata['res'].shape[],1)
			#xdata['res'] = torch.cat( [xdata['res'], xdata['godnode4decoder'] ] , axis = 1)
		z = xdata['res']
		zgodnode = xdata['godnode4decoder']
		#pass through resnet decoder first
		#decoder_in =  torch.cat( [inz,  z] , axis = 1)
		#z_decoder = self.decoder( decoder_in )
		z = self.lin( z )
		#decode aa
		aa = self.aadecoder( torch.cat( [inz,  z ], axis = 1 ) )
		foldx_pred = self.godnodedecoder( xdata['godnode4decoder'] )
		if contact_pred_index is None:
			return aa, None, zgodnode , foldx_pred
		sim_matrix = (z[contact_pred_index[0]] * z[contact_pred_index[1]]).sum(dim=1)
		#find contacts
		edge_probs = self.sigmoid(sim_matrix)
		return aa,  edge_probs , zgodnode , foldx_pred
		

	def x_to_amino_acid_sequence(self, x_r):
		"""
		Converts the reconstructed 20-dimensional matrix to a sequence of amino acids.

		Args:
			x_r (Tensor): Reconstructed 20-dimensional tensor.

		Returns:
			str: A string representing the sequence of amino acids.
		"""
		# Find the index of the maximum value in each row to get the predicted amino acid
		indices = torch.argmax(x_r, dim=1)
		
		# Convert indices to amino acids
		amino_acid_sequence = ''.join(self.amino_acid_indices[idx.item()] for idx in indices)
		
		return amino_acid_sequence

	def load(self, modelfile):
		self.load_state_dict(torch.load(modelfile))
		self.eval()
		return self

	def save(self, modelfile):
		torch.save(self.state_dict(), modelfile)
		return modelfile
	
	def ret_config(self):
		return { 'encoder_out_channels': self.in_channels, 'xdim': 20, 'hidden_channels': self.hidden_channels, 'out_channels_hidden': self.out_channels_hidden, 'metadata': self.metadata, 'amino_mapper': self.amino_acid_indices }

	def save_config(self, configfile):
		with open(configfile , 'w') as f:
			json.dump(self.ret_config(), f)
		return configfile

	def load_from_config(config):
		return HeteroGAE_Encoder(**config)


def recon_loss(z , xdata,edge_index, pos_edge_index, decoder , poslossmod=1, neglossmod=1) -> Tensor:
	r"""Given latent variables :obj:`z`, computes the binary cross
	entropy loss for positive edges :obj:`pos_edge_index` and negative
	sampled edges.

	Args:
		xdata (HeteroData): The input data containing node features and edge indices.
		pos_edge_index (torch.Tensor): The positive edges to train against.
		decoder (torch.nn.Module, optional): The decoder model. (default: :obj:`None`)
		poslossmod (float, optional): The positive loss modifier. (default: :obj:`1`)
		neglossmod (float, optional): The negative loss modifier. (default: :obj:`1`)
	"""
	
	pos = decoder(z , xdata, edge_index, pos_edge_index )[1]
	pos_loss = -torch.log(pos + EPS).mean()
	neg_edge_index = negative_sampling( pos_edge_index, xdata['res'].size(0))
	neg = decoder(z , xdata, edge_index , neg_edge_index)[1]
	neg_loss = -torch.log((1 - neg) + EPS).mean()
	
	return poslossmod * pos_loss + neglossmod * neg_loss

#define loss for x reconstruction   
def x_reconstruction_loss(x, recon_x):
	"""
	compute the loss over the node feature reconstruction.
	"""
	return F.l1_loss(recon_x, x)


#amino acid onehot loss for x reconstruction
def aa_reconstruction_loss(x, recon_x):
	"""
	compute the loss over the node feature reconstruction.
	using categorical cross entropy
	"""
	x = torch.argmax(x, dim=1)
	#recon_x = torch.argmax(recon_x, dim=1)
	return F.cross_entropy(recon_x, x)

def gaussian_loss(mu , logvar , beta= 1.5):
	'''
	
	add beta to disentangle the features
	
	'''
	kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
	return beta*kl_loss




def save_model(model, optimizer, epoch, file_path):
	"""
	Save the model's state dictionary, optimizer's state dictionary, and other metadata to a file.

	Args:
		model (torch.nn.Module): The model to save.
		optimizer (torch.optim.Optimizer): The optimizer used for training.
		epoch (int): The current epoch number.
		file_path (str): The file path to save the model to.
	"""
	torch.save({
		'epoch': epoch,
		'model_state_dict': model.state_dict(),
		'optimizer_state_dict': optimizer.state_dict(),
		'model_class': model.__class__.__name__,
		'model_args': model.args,
		'model_kwargs': model.kwargs,
	}, file_path)



def load_model(file_path):
	"""
	Load the model's state dictionary, optimizer's state dictionary, and other metadata from a file.

	Args:
		file_path (str): The file path to load the model from.

	Returns:
		model (torch.nn.Module): The loaded model.
		optimizer (torch.optim.Optimizer): The loaded optimizer.
		epoch (int): The epoch number to resume training from.
	"""
	checkpoint = torch.load(file_path)

	# Dynamically import the module containing the model class
	model_module = importlib.import_module(__name__)

	# Instantiate the model with the saved arguments
	model_class = getattr(model_module, checkpoint['model_class'])
	model = model_class(*checkpoint['model_args'], **checkpoint['model_kwargs'])

	# Load the saved state dictionary into the model
	model.load_state_dict(checkpoint['model_state_dict'])

	# Assuming the optimizer is Adam, you can modify this to match your optimizer
	optimizer = torch.optim.Adam(model.parameters())
	optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

	epoch = checkpoint['epoch']

	return model, optimizer, epoch
