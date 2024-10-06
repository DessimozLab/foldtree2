
#this is a script to convert pdb files into pytorch geometric data

datadir = '../../datasets/foldtree2/'
EPS = 1e-10


import wget
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
from torch_geometric.nn import SAGEConv , Linear , FiLMConv , TransformerConv , FeaStConv , GATConv , GINConv , GatedGraphConv
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
from torch_geometric.data import HeteroData
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

#create a class for transforming pdb files to pyg 
class PDB2PyG:
    def __init__(self , aapropcsv= './aaindex1.csv'):
        aaproperties = pd.read_csv('./aaindex1.csv', header=0)
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
        self.metadata = { 'edge_types': [ ('res','backbone','res') , ('res','backbonerev','res') ,  ('res','contactPoints', 'res') , ('res','hbond', 'res') ] }
        self.aaindex = aaindex

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
                "single_letter_code": PDB.Polypeptide.three_to_one(residue.get_resname()),
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
                "single_letter_code": PDB.Polypeptide.three_to_one(residue.get_resname()),
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
                "single_letter_code": PDB.Polypeptide.three_to_one(residue.get_resname()),
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


    #create features from a monomer pdb file
    
    def create_features(self, monomerpdb, distance = 8, verbose = False):
        if type(monomerpdb) == str:    
            chain = self.read_pdb(monomerpdb)[0]
        else:
            chain = monomerpdb
        chain = [ r for r in chain if PDB.is_aa(r)]
        if len(chain) ==0:
            return None
        angles = self.get_angles(chain)
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
            
        return angles, contact_points, 0 , hbond_mat, backbone , backbone_rev , positional_encoding , plddt , aa


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
        
    def struct2pyg(self , pdbchain  , identifier=None,  verbose = False):
        data = HeteroData()
        #transform a structure chain into a pytorch geometric graph
        #get the adjacency matrices
        try:
            xdata = self.create_features(pdbchain)
        except:
            return None
        
        if xdata is not None:
            angles, contact_points, springmat , hbond_mat , backbone , backbone_rev , positional_encoding , plddt ,aa = xdata
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
        data['plddt'].x = torch.tensor(plddt, dtype=torch.float32)
        data['positions'].x = torch.tensor( positional_encoding, dtype=torch.float32)
        #use the amino acid properties as the node features
        angles = torch.tensor(angles.values, dtype=torch.float32)
        data['res'].x = angles

        #get the edge features
        data['res','backbone','res'].edge_attr = torch.tensor(backbone.data, dtype=torch.float32)
        data['res','backbonerev','res'].edge_attr = torch.tensor(backbone_rev.data, dtype=torch.float32)
        data['res','contactPoints', 'res'].edge_attr = torch.tensor(contact_points.data, dtype=torch.float32)
        data['res','hbond', 'res'].edge_attr = torch.tensor(hbond_mat.data, dtype=torch.float)
        #data['res','springMat', 'res'].edge_attr = torch.tensor(springmat.data, dtype=torch.float32)

        backbone = self.sparse2pairs(backbone)
        backbone_rev = self.sparse2pairs(backbone_rev)
        contact_points = self.sparse2pairs(contact_points)
        hbond_mat = self.sparse2pairs(hbond_mat)
        springmat = self.sparse2pairs(springmat)
        #get the adjacency matrices into tensors
        data['res','backbone','res'].edge_index = torch.tensor(backbone,  dtype=torch.long )
        data['res','backbonerev','res'].edge_index = torch.tensor(backbone_rev,  dtype=torch.long )
        data['res','contactPoints', 'res'].edge_index = torch.tensor(contact_points,  dtype=torch.long )    
        data['res','hbond', 'res'].edge_index = torch.tensor(hbond_mat,  dtype=torch.long )
        #data['res','springMat', 'res'].edge_index = torch.tensor(springmat,  dtype=torch.long )
        data['res','contactPoints', 'res'].edge_index ,  data['res','contactPoints', 'res'].edge_attr =torch_geometric.utils.to_undirected(  data['res','contactPoints', 'res'].edge_index , data['res','contactPoints', 'res'].edge_attr )
        #add self loops
        data['res','backbone','res'].edge_index = torch_geometric.utils.add_self_loops(data['res','backbone','res'].edge_index)[0]
        data['res','backbonerev','res'].edge_index  = torch_geometric.utils.add_self_loops(data['res','backbonerev','res'].edge_index)[0]
        data['res','backbone', 'res'].edge_index =torch_geometric.utils.to_undirected(  data['res','backbone', 'res'].edge_index )
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
    def store_pyg(self, pdbfiles, filename, verbose = True ):
        with h5py.File(filename , mode = 'w') as f:
            for pdbfile in  tqdm.tqdm( pdbfiles ):                    
                if verbose:
                    print(pdbfile)
                hetero_data = self.struct2pyg(pdbfile)
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

