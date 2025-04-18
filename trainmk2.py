#!/usr/bin/env python
# coding: utf-8

# In[172]:


import wget
#download an example pdb file
filename = './1eei (1).pdb'
url = 'https://files.rcsb.org/download/1EEI.pdb'
#filename = wget.download(url)
datadir = '../../datasets/foldtree2/'


# In[ ]:


import torch_geometric


# In[ ]:


# Standard libraries
import os
import glob
# For downloading pre-trained models
import urllib.request
from urllib.error import HTTPError

# PyTorch Lightning
import pytorch_lightning as L

# PyTorch
import torch
import scipy.sparse
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# PyTorch geometric
from torch_geometric.data import Data, Dataset
from torch_geometric.data import HeteroData

# PL callbacks
from pytorch_lightning.callbacks import ModelCheckpoint
from torch import Tensor


# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
aaproperties = pd.read_csv('./aaindex1.csv', header=0)
print(aaproperties.columns, len(aaproperties.columns))

#for i,w in  enumerate(list(aaproperties.description )):
#    print(i,w)

colmap = {aaproperties.columns[i]:i for i in range(len(aaproperties.columns))}
aaproperties.drop( [ 'description' , 'reference'  ], axis=1, inplace=True)
print(aaproperties.columns, len(aaproperties.columns))



start_onehot = len(aaproperties)+2
print(len(aaproperties))


# In[ ]:


#one hot encoding
#get dummies of the index
onehot = pd.get_dummies(aaproperties.columns.unique())
#turn true into 1 and false into 0
onehot = onehot.astype(int)
print(onehot)


# In[ ]:




# In[ ]:


aaindex = { c:onehot[c].argmax() for c in onehot.columns}
aaproperties = pd.concat([aaproperties, onehot ] , axis = 0 )
print(aaproperties.head())
print(len(aaproperties))


# In[ ]:


aaproperties = aaproperties.T
aaproperties[aaproperties.isna() == True] = 0
plt.figure( figsize=(10,10) )
plt.imshow(aaproperties)
plt.show()
plt.imshow( aaproperties.iloc[:,-20:])
plt.show()


# In[ ]:


from Bio import PDB
import warnings
from matplotlib import pyplot as plt
import numpy as np
import pydssp
from Bio.PDB import PDBParser   
import numpy as np
filename = './1eei.pdb'

def read_pdb(filename):
    #silence all warnings
    warnings.filterwarnings('ignore')

    with warnings.catch_warnings():        
        parser = PDB.PDBParser()
        structure = parser.get_structure(filename, filename)
        chains = [ c for c in structure.get_chains()]
        return chains

#return the phi, psi, and omega angles for each residue in a chain
def get_angles(chain):

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


def get_closest(chain):
    contact_mat = np.zeros((len(chain), len(chain)))
    for i,r1 in enumerate(chain):
        for j,r2 in enumerate(chain):
            contact_mat[i,j] =  r1['CB'] - r2['CB']
    #go through each row and select min
    for r in contact_mat.shape[0]:
        contact_mat[r, :][ contact_mat[r, :] != np.amin(contact_mat)] =  0
    return contact_mat


def get_backbone(chain):
    backbone_mat = np.zeros((len(chain), len(chain)))
    backbone_rev_mat = np.zeros((len(chain), len(chain)))
    np.fill_diagonal(backbone_mat[1:], 1)
    np.fill_diagonal(backbone_rev_mat[:, 1:], 1)
    return backbone_mat, backbone_rev_mat

def ret_hbonds(chain , verbose = False):
    #loop through all atoms in a structure
    struct = PDBParser().get_structure('1eei', filename)

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
def add_aaproperties(angles, aaproperties = aaproperties , verbose = False):
    if verbose == True:
        print(aaproperties , angles )
    nodeprops = angles.merge(aaproperties, left_on='single_letter_code', right_index=True, how='left')
    nodeprops = nodeprops.dropna()

    #generate 1 hot encoding for each amino acid
    #one_hot = pd.get_dummies(nodeprops['single_letter_code']).astype(int)
    #nodeprops = nodeprops.join(one_hot)
    #nodeprops = nodeprops.drop(columns=['single_letter_code'])
    return nodeprops

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


# In[ ]:


"""

from prody import *
from pylab import *
import warnings
def anm_analysis(filename):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        prot = parsePDB( filename  , )
        calphas2 = prot.select('calpha')
        anm = ANM('ANM analysis')
        anm.buildHessian(calphas2)
        anm.calcModes()
        cov = anm.getCovariance()
        cov[ cov < 0] = -cov[ cov < 0]
        logcov = np.log(cov)
        #get the top .5% of the covariance matrix
        top = np.percentile(logcov, 99.5)
        logcov[ logcov < top] = 0
        
        
        return logcov

cov = anm_analysis(filename)
print(cov.shape)

plt.imshow(cov)
plt.colorbar()
plt.show()

#print the number of 0 entries
print( np.sum(cov != 0) /np.sum(cov == 0))
print( np.sum(cov != 0) )

"""


# In[ ]:


import numpy as np
import matplotlib.pyplot as plt

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

# Example usage:
seq_len = 200  # Length of the sequence
d_model = 256  # Dimension of the embeddings

positional_encoding = get_positional_encoding( seq_len , d_model)

# Plotting the positional encoding for visualization
plt.figure(figsize=(15, 5))
plt.imshow(positional_encoding, aspect='auto', cmap='viridis')
plt.colorbar()
plt.xlabel('Embedding Dimensions')
plt.ylabel('Sequence Length')
plt.title('Positional Encoding')
plt.show()


# In[ ]:


#create features from a monomer pdb file
from scipy import sparse
from copy import deepcopy
def create_features(monomerpdb, aaproperties, distance = 8, verbose = False):
    if type(monomerpdb) == str:    
        chain = read_pdb(monomerpdb)[0]
    else:
        chain = monomerpdb
    chain = [ r for r in chain if PDB.is_aa(r)]
    angles = get_angles(chain)
    if len(angles) ==0:
        return None
    angles = add_aaproperties(angles, aaproperties)
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

    contact_points = get_contact_points(chain, distance)
    if verbose:
        print('contacts' , contact_points.shape)
        plt.imshow(contact_points)
        plt.colorbar()
        plt.show()


    hbond_mat = np.array(ret_hbonds(chain, verbose))
    if verbose:
        print('hbond' , hbond_mat.shape)
        plt.imshow(hbond_mat)
        plt.colorbar()
        plt.show()
    #return the angles, amino acid properties, contact points, and hydrogen bonds
    #backbone is just the amino acid chain
    backbone , backbone_rev = get_backbone(chain)
    positional_encoding = get_positional_encoding( len(chain) , 256)

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
    plddt = get_plddt(chain)/100
    if verbose:
        print('plddt' , plddt.shape)
        plt.plot(plddt)
        plt.ylim([0,1])
        plt.show()
        
    return angles, contact_points, 0 , hbond_mat, backbone , backbone_rev , positional_encoding , plddt , aa


# In[ ]:


angles, contact_points, blank , hbond_mat, backbone , backbone_rev , positional_encoding , plddt ,aa = create_features(filename, aaproperties, distance = 10, verbose = True)


# In[ ]:


print(angles.shape)
print( 'one hot aa' , angles.iloc[:,-20:] ) 


# In[ ]:


"""
#make a networkx graph from the contact points
import networkx as nx
import matplotlib.pyplot as plt
import colour

def make_graph(contact_points):
    G = nx.Graph()
    G.add_nodes_from(range(contact_points.shape[0]))

    for i in range(contact_points.shape[0]):
        for j in range(contact_points.shape[1]):
            if contact_points[i,j] != 0:
                G.add_edge(i,j, weight = contact_points[i,j] / 2)
    return G

#plot the contact point graph
#make the edges thicker for smaller weights
#make the nodes colors from a column of the angles dataframe


red = colour.Color("red")
blue = colour.Color("blue")
crange = [ c.hex_l for c in list(red.range_to(blue, 101))]


#add a color column to the angles dataframe
angles['color'] = [ int(100*(a-angles[111].min())/angles[111].max()) for a in angles[111]]
angles['color'] = [ crange[a] for a in angles['color']]

print(angles)

G = make_graph(contact_points)
edges = G.edges()
weights = [ G[u][v]['weight'] for u,v in edges]
#change the color of the nodes
colors = [ angles.loc[u]['color'] for u in G.nodes() ]
#change the size of the nodes$
sizes = [ angles.loc[u][110] for u in G.nodes() ]
#keep sizes between 0 and 1000  
sizes = [ 1000* (s - min(sizes))/max(sizes)  for s in sizes]
print( colors)
print( sizes)
plt.figure(figsize=(10,10))
pos = nx.spring_layout(G)
#draw nodes with colors

#get amino acid sequence from structure

chain = read_pdb(filename)[0]
chain = [ r for r in chain if PDB.is_aa(r)]
seq = [ r.get_resname() for r in chain ]

#label nodes with sequence
labels = { i:aa for i,aa in enumerate(angles.single_letter_code)}

nx.draw_networkx_nodes(G, pos = pos, node_color = colors, node_size = sizes  )

#draw edges with weights
nx.draw_networkx_edges(G, pos = pos, width = weights , alpha = 0.25)
#draw labels
nx.draw_networkx_labels(G, pos = pos, labels = labels , font_color='whitesmoke')

"""


# In[ ]:


#write a function to store sparse matrices in an hdf5 file for many pdb files
import h5py
def store_features( pdbfiles, aaproperties, filename, verbose = False):
    #create a hdf5 file
    with h5py.File(filename, 'pdbfiles') as f:
        for pdbfile in pdbfiles:
            if verbose:
                print(pdbfile)
            angles, contact_points, springmat, backbone = create_features(pdbfile, aaproperties, verbose)
            #store the features in the hdf5 file
            f.create_dataset(pdbfile + '_angles', data=angles)
            f.create_dataset(pdbfile + '_contact_points', data=contact_points)
            f.create_dataset(pdbfile + '_springmat', data=springmat)
            f.create_dataset(pdbfile + '_backbone', data=backbone)


# In[ ]:


import pandas as pd
cols = 'repId_isDark_nMem_repLen_avgLen_repPlddt_avgPlddt_LCAtaxId'.split('_')
repdf = pd.read_table(datadir+ './afdbclusters/2-repId_isDark_nMem_repLen_avgLen_repPlddt_avgPlddt_LCAtaxId.tsv')
repdf.columns = cols
print(repdf.head())


# In[ ]:


import multiprocessing as mp
import tqdm
import os
import numpy as np
import wget 

def download_pdb(rep ,structdir = datadir+'structs/'):
    url = f'https://alphafold.ebi.ac.uk/files/AF-{rep}-F1-model_v4.pdb'
    #check if file exists
    if os.path.exists( structdir + rep + '.pdb'):
        return structdir + rep + '.pdb'
    filename = wget.download(url, out=structdir + rep + '.pdb')
    return filename

def download(repdf , nreps = 100 , structdir = datadir +'structs/'):
    if not os.path.exists(structdir):
        os.makedirs(structdir)
    reps = repdf.repId.unique()
    if nreps:
        #select a random sample of representatives
        reps = np.random.choice(reps, nreps)
    with mp.Pool(20) as p:
        filenames = p.map(download_pdb, tqdm.tqdm(reps))
        return filenames


# In[173]:


#download(repdf, nreps = 1000 , structdir = '../datasets/foldtree2/structs/' )


# In[174]:


AVAIL_GPUS = min(1, torch.cuda.device_count())
BATCH_SIZE = 256 if AVAIL_GPUS else 64
# Path to the folder where the datasets are/should be downloaded
DATASET_PATH = os.environ.get("PATH_DATASETS", "data/")
# Path to the folder where the pretrained models are saved
CHECKPOINT_PATH = os.environ.get("PATH_CHECKPOINT", "saved_models/GNNs/")

#create the h5 dataset from the pdb files
import glob
import h5py

# Setting the seed
L.seed_everything(42)

# Ensure that all operations are deterministic on GPU (if used) for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# In[175]:


import pebble 

def sparse2pairs(sparsemat):
    sparsemat = scipy.sparse.find(sparsemat)
    return np.vstack([sparsemat[0],sparsemat[1]])

def struct2pyg(pdbchain , aaproperties= aaproperties , start_onehot = start_onehot , verbose = False):
    data = HeteroData()
    #transform a structure chain into a pytorch geometric graph
    #get the adjacency matrices
    
    xdata = create_features(pdbchain, aaproperties)
    if data is not None:
        angles, contact_points, springmat , hbond_mat , backbone , backbone_rev , positional_encoding , plddt ,aa = xdata
    else:
        return None
    if len(angles) ==0 :
        return None
    
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

    backbone = sparse2pairs(backbone)
    backbone_rev = sparse2pairs(backbone_rev)
    contact_points = sparse2pairs(contact_points)
    hbond_mat = sparse2pairs(hbond_mat)
    springmat = sparse2pairs(springmat)
    
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
    
    #data['res','contactPoints', 'res'].edge_index = torch_geometric.utils.add_self_loops(data['res','contactPoints', 'res'].edge_index)[0]
    #data['res','hbond_mat', 'res'].edge_index = torch_geometric.utils.add_self_loops(data['res','hbond_mat', 'res'].edge_index)[0]

    #normalize features

    #data['res'].x = torch_geometric.transforms.NormalizeFeatures(data['res'].x)
    #data['res','contact_points', 'res'].edge_attr = torch_geometric.transforms.normalize_edge_attr(data['res','contact_points', 'res'].edge_attr)
    #data['res','spring_mat', 'res'].edge_index = torch_geometric.transforms.normalize_edge_attr(data['res','spring_mat', 'res'].edge_attr)
    #data['res','hbond_mat', 'res'].edge_attr = torch_geometric.transforms.normalize_edge_attr(data['res','hbond_mat', 'res'].edge_attr)

    return data

#create a function to store the pytorch geometric data in a hdf5 file
def store_pyg(pdbfiles, aaproperties, filename, verbose = True):
    with h5py.File(filename , mode = 'w') as f:
        #create structs list

        for pdbfile in tqdm.tqdm(pdbfiles):
            if verbose:
                print(pdbfile)
            hetero_data = struct2pyg(pdbfile, aaproperties)
            if hetero_data:
                f.create_group(pdbfile)

                for node_type in hetero_data.node_types:
                    if hetero_data[node_type].x is not None:
                        node_group = f.create_group(f'{pdbfile}/node/{node_type}')
                        node_group.create_dataset('x', data=hetero_data[node_type].x.numpy())
                        
                # Iterate over edge types and their connections
                for edge_type in hetero_data.edge_types:
                    # edge_type is a tuple: (src_node_type, relation_type, dst_node_type)
                    edge_group = f.create_group(f'{pdbfile}/edge/{edge_type[0]}_{edge_type[1]}_{edge_type[2]}')
                    if hetero_data[edge_type].edge_index is not None:
                        edge_group.create_dataset('edge_index', data=hetero_data[edge_type].edge_index.numpy())
                    
                    # If there are edge features, save them too
                    if hasattr(hetero_data[edge_type], 'edge_attr') and hetero_data[edge_type].edge_attr is not None:
                        edge_group.create_dataset('edge_attr', data=hetero_data[edge_type].edge_attr.numpy())


                    #todo. store some other data. sequence. uniprot info etc.
            else:
                print('err' , pdbfile )



#create a function to store the pytorch geometric data in a hdf5 file
def store_pyg_mp(pdbfiles, aaproperties, filename, verbose = True , ncpu = 4):
    with h5py.File(filename , mode = 'w') as f:
        #create structs list

        #use pebble to multiprocess the pdb files and turn them into pyg objects
        #open process pool with ncpu workers
        with pebble.ProcessPool(max_workers=ncpu) as pool:
            #map the pdb files to the struct2pyg function and get the results asynchonously
            results = pool.map(lambda x: struct2pyg(x, aaproperties), pdbfiles)
            for pdbfile in tqdm.tqdm(pdbfiles_res):
                if hetero_data:
                    f.create_group(pdbfile)
                    for node_type in hetero_data.node_types:
                        if hetero_data[node_type].x is not None:
                            node_group = f.create_group(f'{pdbfile}/node/{node_type}')
                            node_group.create_dataset('x', data=hetero_data[node_type].x.numpy())
                    # Iterate over edge types and their connections
                    for edge_type in hetero_data.edge_types:
                        # edge_type is a tuple: (src_node_type, relation_type, dst_node_type)
                        edge_group = f.create_group(f'{pdbfile}/edge/{edge_type[0]}_{edge_type[1]}_{edge_type[2]}')
                        if hetero_data[edge_type].edge_index is not None:
                            edge_group.create_dataset('edge_index', data=hetero_data[edge_type].edge_index.numpy())
                        
                        # If there are edge features, save them too
                        if hasattr(hetero_data[edge_type], 'edge_attr') and hetero_data[edge_type].edge_attr is not None:
                            edge_group.create_dataset('edge_attr', data=hetero_data[edge_type].edge_attr.numpy())


                        #todo. store some other data. sequence. uniprot info etc.
                else:
                    print('err' , pdbfile )
        


# In[176]:


metadata = { 'edge_types': [ ('res','backbone','res') , ('res','backbonerev','res') ,  ('res','contactPoints', 'res') , ('res','hbond', 'res') ] }


# In[177]:


struct2pyg( filename, aaproperties, verbose=True)


# In[178]:


#pdbfiles_structalign = glob.glob(datadir + 'struct_align/*/structs/*.pdb')
#store_pyg(pdbfiles_structalign, aaproperties, filename='structs_structalign_new.h5', verbose = False)


# In[179]:


#pdbfiles = glob.glob(datadir+'structs/*.pdb')
#store_pyg(pdbfiles, aaproperties, filename='structs_plddt.h5', verbose = False)


# In[180]:


class StructureDataset(Dataset):
    def __init__(self, h5dataset , key = datadir+'structs' ):
        super().__init__()
        #keys should be the structures
        self.structures = h5dataset[key]
        self.structlist = list(self.structures.keys())
        
    def __len__(self):
        return len(self.structures)

    def __getitem__(self, idx):
        if type(idx) == str:
            f = self.structures[idx]
        elif type(idx) == int:
            f = self.structures[self.structlist[idx]]
        else:
            raise 'use a structure filename or integer'
        data = {}
        hetero_data = HeteroData()
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
    
    


# In[181]:


class StructAlnDataset(Dataset):
    def __init__(self, h5dataset , structlist ):
        super().__init__()
        #keys should be the structures

        self.h5dataset = h5dataset

        self.structlist = structlist
        paths = { s.split('/')[-1] : s for s in structlist }
        reps = {s:s.split('/')[2] for s in structlist }
        self.reps = reps
        self.paths = paths
        self.structs =  [ s.split('/')[-1] for s in structlist  ]
        
    def __len__(self):
        return len(self.structlist)

    def __getitem__(self, idx):    
        if type(idx) == str:
            pass
        elif type(idx) == int:
            idx = self.structlist[idx]            
        else:
            raise 'use a structure filename or integer'
        f = self.h5dataset[idx]
        
        hetero_data = HeteroData()
        
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


# In[182]:


import time
import h5py

f = h5py.File('./structs_structalign_new.h5' , 'r')
struct_dat = StructAlnDataset(f, structlist = glob.glob(datadir + 'struct_align/*/structs/*.pdb'))
ndim = struct_dat[20]['res'].x.shape[1]
print( ndim)

import torch
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.utils import to_networkx
from torch_geometric.data import HeteroData
import networkx as nx
from torch_geometric.data import HeteroData


def remove_self_loops_edge_index(edge_index):
    """
    Removes self-loops from the edge_index of a graph.
    """
    mask = edge_index[0] != edge_index[1]
    return edge_index[:, mask]

def convert_hetero_to_networkx(hetero_data):
    G = nx.MultiDiGraph()  # Use MultiDiGraph to support multiple edge types
    # Add nodes with type as an attribute
    for node_type in hetero_data.node_types:
        for node_id in range(hetero_data[node_type].num_nodes):
            # Node identifier format: (node_type, node_id)
            G.add_node((node_type, node_id), node_type=node_type)
    # Add edges
    for edge_type in hetero_data.edge_types:
        src_type, _, dst_type = edge_type
        edge_indices = hetero_data[edge_type].edge_index
        edge_indices = remove_self_loops_edge_index(edge_indices)
        for i in range(edge_indices.shape[1]):  # Iterate through each edge
            src_id, dst_id = edge_indices[:, i].tolist()
            # Edge identifier format: ((src_type, src_id), (dst_type, dst_id))
            G.add_edge((src_type, src_id), (dst_type, dst_id), edge_type=edge_type)
    return G

def plot_hetero_graph_with_curved_edges(data):
    print(data)
    G = convert_hetero_to_networkx(data)
    print(G)
    pos = nx.spring_layout(G)  # General layout if no positions are provided

    # Calculate offset for curved edges to avoid overlap
    edge_count = {}
    for src, dst, key in G.edges(keys=True):
        edge_count[(src, dst)] = edge_count.get((src, dst), 0) + 1
    # Draw nodes
    plt.figure(figsize=(12, 12))
    nx.draw_networkx_nodes(G, pos, node_color='skyblue', node_size=70)
    
    edge_type_colors = {}
    unique_edge_types = data.edge_types
    # Generate a color map from matplotlib, or use a predefined set of colors
    colors = plt.get_cmap('tab20')(range(len(unique_edge_types)))
    for i, edge_type in enumerate(unique_edge_types):
        edge_type_colors[edge_type] = colors[i]

    # Draw edges with curvature
    for (src, dst), count in edge_count.items():
        for i in range(count):
            edge_key = list(G[src][dst])[i]
            style = G[src][dst][edge_key]
            curvature = 0.1 * (i - count // 2)
            nx.draw_networkx_edges(
                G, pos, edgelist=[(src, dst)],
                connectionstyle=f'arc3, rad={curvature}',
                arrowstyle='-|>',
                edge_color = edge_type_colors.get(edge_type, 'black'),  # Default color is black
                width=style.get('weight', 1),
                alpha = .25
            )

    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=10, font_color='black')
    plt.title('Heterogeneous Graph with Curved Edges')
    plt.axis('off')  # Turn off the axis
    plt.show()


# In[ ]:


from torch_geometric.data import HeteroData
#import sageconv
from torch_geometric.nn import SAGEConv , Linear , FiLMConv , TransformerConv , FeaStConv , GATConv , GINConv , GatedGraphConv
#import module dict and module list
from torch.nn import ModuleDict, ModuleList , L1Loss
from torch_geometric.nn import global_mean_pool
#import negative sampling
from torch_geometric.utils import negative_sampling


EPS = 1e-10


def recon_loss( z: Tensor, pos_edge_index: Tensor , backbone:Tensor = None , decoder = None , poslossmod = 1 , neglossmod= 1) -> Tensor:
    r"""Given latent variables :obj:`z`, computes the binary cross
    entropy loss for positive edges :obj:`pos_edge_index` and negative
    sampled edges.

    Args:
        z (torch.Tensor): The latent space :math:`\mathbf{Z}`.
        pos_edge_index (torch.Tensor): The positive edges to train against.
        neg_edge_index (torch.Tensor, optional): The negative edges to
            train against. If not given, uses negative sampling to
            calculate negative edges. (default: :obj:`None`)
    """
    
    pos =decoder(z, pos_edge_index, { ( 'res','backbone','res'): backbone } )[1]
    #turn pos edge index into a binary matrix
    pos_loss = -torch.log( pos + EPS).mean()
    neg_edge_index = negative_sampling(pos_edge_index, z.size(0))
    neg = decoder(z ,  neg_edge_index, { ( 'res','backbone','res'): backbone } )[1]
    neg_loss = -torch.log( ( 1 - neg) + EPS ).mean()
    return poslossmod*pos_loss + neglossmod*neg_loss

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


# In[ ]:


class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super(VectorQuantizer, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost

        self.embeddings = nn.Embedding(num_embeddings, embedding_dim)
        self.embeddings.weight.data.uniform_(-1 / self.num_embeddings, 1 / self.num_embeddings)

    def forward(self, x):
        # Flatten input
        flat_x = x.view(-1, self.embedding_dim)

        # Calculate distances
        distances = (torch.sum(flat_x**2, dim=1, keepdim=True)
                     + torch.sum(self.embeddings.weight**2, dim=1)
                     - 2 * torch.matmul(flat_x, self.embeddings.weight.t()))

        # Get the encoding that has the min distance
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=x.device)
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize the latents
        quantized = torch.matmul(encodings, self.embeddings.weight).view_as(x)

        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), x)
        q_latent_loss = F.mse_loss(quantized, x.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        # Straight-through estimator
        quantized = x + (quantized - x).detach()
        return quantized, loss

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

    

class HeteroGAE_Encoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_embeddings, commitment_cost, metadata={} , encoder_hidden = 100 , dropout_p = 0.05):
        super(HeteroGAE_Encoder, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.metadata = metadata
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.encoder_hidden = encoder_hidden
        #batch norm
        self.bn = torch.nn.BatchNorm1d(in_channels)
        self.dropout = torch.nn.Dropout(p=dropout_p)
        for i in range(len(hidden_channels)):
            self.convs.append(
                torch.nn.ModuleDict({
                    '_'.join(edge_type): SAGEConv(in_channels if i == 0 else hidden_channels[i-1], hidden_channels[i])
                    for edge_type in metadata['edge_types']
                })
            )
        #self.lin = Linear(hidden_channels[-1], out_channels)
        self.out_dense= torch.nn.Sequential(
            torch.nn.Linear(hidden_channels[-1] + 20 , self.encoder_hidden) ,
            torch.nn.ReLU(),
            torch.nn.Linear(self.encoder_hidden, self.encoder_hidden) ,
            torch.nn.ReLU(),
            torch.nn.Linear(self.encoder_hidden, self.out_channels) ,
            torch.nn.Tanh()
            )
        
        self.vector_quantizer = VectorQuantizer(num_embeddings, out_channels, commitment_cost)

    def forward(self, x, xaa, edge_index_dict):
        x = self.bn(x)
        for i, convs in enumerate(self.convs):
            # Apply the graph convolutions and average over all edge types
            x = [conv(x, edge_index_dict[tuple(edge_type.split('_'))]) for edge_type, conv in convs.items()]
            x = torch.stack(x, dim=0).mean(dim=0)
            x = F.relu(x) if i < len(self.hidden_channels) - 1 else x
        
        x = self.out_dense( torch.cat([x,xaa], dim=1) )
        z_quantized, vq_loss = self.vector_quantizer(x)
        return z_quantized, vq_loss

class HeteroGAE_VariationalQuantizedEncoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_embeddings, commitment_cost, metadata={}):
        super(HeteroGAE_VariationalQuantizedEncoder, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.metadata = metadata
        self.hidden_channels = hidden_channels
        latent_dim = out_channels
        self.latent_dim = out_channels
        self.out_channels = out_channels
        self.in_channels = in_channels
        
        for i in range(len(hidden_channels)):
            self.convs.append(
                torch.nn.ModuleDict({
                    '_'.join(edge_type): SAGEConv(in_channels if i == 0 else hidden_channels[i-1], hidden_channels[i])
                    for edge_type in metadata['edge_types']
                })
            )
        self.fc_mu = Linear(hidden_channels[-1], latent_dim)
        self.fc_logvar = Linear(hidden_channels[-1], latent_dim)
        self.vector_quantizer = VectorQuantizer(num_embeddings, latent_dim, commitment_cost)

    def forward(self, x, edge_index_dict):

        for i, convs in enumerate(self.convs):
            # Apply the graph convolutions and average over all edge types
            x = [conv(x, edge_index_dict[tuple(edge_type.split('_'))]) for edge_type, conv in convs.items()]
            x = torch.stack(x, dim=0).mean(dim=0)
            x = F.ReLu(x) if i < len(self.hidden_channels) - 1 else x
        
        # Obtain the mean and log variance for the latent variables
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        
        # Reparameterization trick
        z = self.reparameterize(mu, logvar)
        
        # Vector quantization
        z_quantized, vq_loss = self.vector_quantizer(z)
        
        return z_quantized, vq_loss, mu, logvar

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu


class HeteroGAE_Decoder(torch.nn.Module):
    def __init__(self, encoder_out_channels, xdim=20, hidden_channels={'res_backbone_res': [20, 20, 20]}, out_channels_hidden=20, nheads = 1 , Xdecoder_hidden=30, metadata={}, amino_mapper= None):
        super(HeteroGAE_Decoder, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.metadata = metadata
        self.hidden_channels = hidden_channels
        self.out_channels_hidden = out_channels_hidden
        self.in_channels = encoder_out_channels
        self.amino_acid_indices = amino_mapper
        for i in range(len(self.hidden_channels[('res', 'backbone', 'res')])):
            self.convs.append(
                torch.nn.ModuleDict({
                    '_'.join(edge_type): SAGEConv(self.in_channels if i == 0 else self.hidden_channels[edge_type][i-1], self.hidden_channels[edge_type][i]  )
                    for edge_type in [('res', 'backbone', 'res')]
                })
            )

        self.lin = Linear(hidden_channels[('res', 'backbone', 'res')][-1], self.out_channels_hidden)
        
        self.sigmoid = nn.Sigmoid()

        self.decoder = torch.nn.Sequential(
            torch.nn.Linear( self.in_channels + self.out_channels_hidden , Xdecoder_hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(Xdecoder_hidden, Xdecoder_hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(Xdecoder_hidden, Xdecoder_hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(Xdecoder_hidden, Xdecoder_hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(Xdecoder_hidden, xdim),
            torch.nn.LogSoftmax(dim=1)         )
        

    def forward(self, z , edge_index, backbones, **kwargs):
        
        #copy z for later concatenation
        inz = z
        for layer in self.convs:
            for edge_type, conv in layer.items():
                z = conv(z, backbones[tuple(edge_type.split('_'))])
                z = F.relu(z)
        z = self.lin(z)
        x_r = self.decoder(  torch.cat( [inz,  z] , axis = 1) )
        sim_matrix = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=1)
        edge_probs = self.sigmoid(sim_matrix)

        #plddt_r = self.plddt_decoder(z)
        
        #return x_r, plddt_r,  edge_probs
        return x_r,  edge_probs

    
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
        amino_acid_sequence = ''.join(self.amino_mapper[idx.item()] for idx in indices)
        
        return amino_acid_sequence
    


# In[187]:


import torch
import torch.nn.functional as F
from torch_geometric.nn import VGAE
from torch.optim import Adam
from torch_geometric.data import DataLoader
import pickle
#create a training loop for the GAE model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device( 'cpu')
print(device)


train_loop = True
V_train_loop = False

if train_loop == True:
#add positional encoder channels to input
    encoder = HeteroGAE_Encoder(in_channels=ndim, hidden_channels=[ 400 ]*3 , out_channels=250, metadata=metadata , num_embeddings=248, commitment_cost=1 , encoder_hidden=500 , EMA = True)
#encoder = HeteroGAE_VariationalQuantizedEncoder(in_channels=ndim, hidden_channels=[100]*3 , out_channels=25, metadata=metadata , num_embeddings=256  , commitment_cost= 1.5 )

decoder = HeteroGAE_Decoder(encoder_out_channels = encoder.out_channels , 
                            hidden_channels={ ( 'res','backbone','res'):[ 500 ] * 5  } , 
                            out_channels_hidden= 150 , metadata=metadata , amino_mapper = aaindex , Xdecoder_hidden=100 )

encoder_save = 'encoder_mk2_aa_50_AAq_transformermk3_248'
decoder_save = 'decoder_mk2_aa_50_AAq_transformermk3_248'

betafactor = 2
#put encoder and decoder on the device
encoder = encoder.to(device)
decoder = decoder.to(device)
# Create a DataLoader for training
total_loss_x = 0
total_loss_edge = 0
total_vq=0
total_kl = 0
total_plddt=0
# Training loop


#load model if it exists
if os.path.exists(encoder_save):
    encoder.load_state_dict(torch.load(encoder_save ))
if os.path.exists(decoder_save):
    decoder.load_state_dict(torch.load(decoder_save  ))


if train_loop == True:
    train_loader = DataLoader(struct_dat, batch_size=40, shuffle=True)

    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=0.001)
    encoder.train()
    decoder.train()
    
    xlosses = []
    edgelosses = []
    vqlosses = []
    plddtlosses = []

    edgeweight = 1
    xweight = 1
    vqweight = 2
    plddtweight = 1

    for epoch in range(1000):
        for data in tqdm.tqdm(train_loader):
            data = data.to(device)
            optimizer.zero_grad()
            z,vqloss = encoder.forward(data['res'].x, data['AA'].x , data.edge_index_dict)
            
            #add positional encoding to give to the decoder
            edgeloss = recon_loss(z , data.edge_index_dict[( 'res','contactPoints','res')]
                                  , data.edge_index_dict[( 'res','backbone','res')], decoder)
            
            recon_x , edge_probs = decoder(z, data.edge_index_dict[( 'res','contactPoints','res')] , data.edge_index_dict )        
    
            xloss = aa_reconstruction_loss(data['AA'].x, recon_x)
            #plddtloss = x_reconstruction_loss(data['plddt'].x, recon_plddt)
            loss = xweight*xloss + edgeweight*edgeloss + vqweight*vqloss #+ plddtloss
            loss.backward()
            optimizer.step()
            total_loss_edge += edgeloss.item()
            total_loss_x += xloss.item()
            total_vq += vqloss.item()
            #total_plddt += plddtloss.item()

        if epoch % 10 == 0 :
            #save model
            torch.save(encoder.state_dict(), encoder_save)
            torch.save(decoder.state_dict(), decoder_save)
        
        """
        for loss in [( total_loss_x , xlosses , xweight ), (total_loss_edge, edgelosses, edgeweight), ( total_vq, vqlosses, vqweight ) ]:
            loss[1].append(loss[0])
            #calculate the mean delta loss for past 10 epochs
            if len(loss[1]) > 10:
                loss[1].pop(0)
                mean_loss = np.mean(loss[0:5])
                #calculate the delta loss for the last 5 epochs
                delta_loss = np.mean(loss[1][-5:])
                delta_loss = delta_loss- mean_loss

                if delta_loss < 0:
                    #reduce learning rate
                    for g in optimizer.param_groups:
                        g['lr'] = g['lr'] * 0.9
                        print('reducing learning rate to ', g['lr'])
        """

        print(f'Epoch {epoch}, AALoss: {total_loss_x:.4f}, Edge Loss: {total_loss_edge:.4f}, vq Loss: {total_vq:.4f}') #, plddt Loss: {total_plddt:.4f}')
        total_loss_x = 0
        total_loss_edge = 0
        total_vq = 0
        #total_plddt = 0
    torch.save(encoder.state_dict(), encoder_save)
    torch.save(decoder.state_dict(), decoder_save)


# In[190]:


torch.save(encoder.state_dict(), encoder_save)
torch.save(decoder.state_dict(), decoder_save)

with open('encoderk2.pkl', 'wb') as f:
    pickle.dump(encoder, f)
with open('decodermk2.pkl', 'wb') as f:
    pickle.dump(decoder, f)


# In[ ]:


#load encoder and decoder
encoder.load_state_dict(torch.load(encoder_save))
decoder.load_state_dict(torch.load(decoder_save))


# In[ ]:


#predict the embeddings for a structure
def predict_structure(structure, encoder, decoder):
    encoder.eval()
    decoder.eval()
    
    data = struct2pyg(structure, aaproperties)
    data = data.to(device)
    z,qloss = encoder.forward(data['res'].x , data['AA'].x,  data.edge_index_dict)
    #create all vs all edge index
    edge_index = torch.tensor( [ [i,j] for i in range(z.shape[0]) for j in range(z.shape[0]) ] ).T
    #z = torch.cat([ data['positions'].x , z] , axis = 1)
    recon_x, edge_probs = decoder(z , edge_index , data.edge_index_dict )
    

    
    return z, recon_x, edge_probs , data


# In[ ]:


from matplotlib import pyplot as plt

def plot_embeddings(filename, encoder, decoder):
    z, recon_x, edge_probs,data = predict_structure(filename, encoder, decoder)
    z = z.detach().cpu().numpy()
    edge_probs = edge_probs.detach().cpu().numpy()
    print(edge_probs.shape)
    #reshape the edge probabilities into a matrix
    edge_probs = edge_probs.reshape((z.shape[0], z.shape[0]))
    print(edge_probs.shape)
    print(edge_probs)
    #get edge index for the structure
    #plot the edge probabilities
    plt.figure(figsize=(10,10))
    plt.title('Edge Probabilities ' + filename)
    plt.imshow(1-edge_probs)
    plt.colorbar()
    plt.show()

    #plot the distance matrix
    chain = read_pdb(filename)[0]
    chain = [ r for r in chain if PDB.is_aa(r)]
    distances = np.zeros((len(chain), len(chain)))
    for i in range(len(chain)):
        for j in range(len(chain)):
            distances[i,j] = chain[i]['CA'] - chain[j]['CA']
    plt.figure(figsize=(10,10))
    plt.title('distmat ' + filename)

    plt.imshow(distances)
    plt.colorbar()

    distances[distances>10] = 0
    plt.figure(figsize=(10,10))
    plt.title('contactmat ' + filename)

    plt.imshow(distances)
    plt.colorbar()
    #transform the embeddings to the quantized string
    encoded_string = encoder.vector_quantizer.discretize_z(torch.tensor(z).to(device))
    print('encoded', encoded_string)
    plt.show()
    
    plt.imshow(F.log_softmax(recon_x , dim=1 ).detach().cpu().numpy())
    plt.colorbar()
    plt.show()

    plt.imshow(data['AA'].x.detach().cpu().numpy())
    plt.colorbar()
    plt.show()

    plt.imshow( data['AA'].x.detach().cpu().numpy() - torch.argmax(F.log_softmax(recon_x , dim=1 ) ,dim=1).detach().cpu().numpy())   
    plt.colorbar()
    plt.show()
    
#plot a few embeddings from example structures
for structure in structures[0:10]:
    print(structure)
    plot_embeddings(structure, encoder, decoder)


# In[ ]:


#predict the embeddings for a few structures
structures = glob.glob('./structs/*.pdb')
zvals_stack = []
edge_probs_stack = []

for structure in structures[0:10]:
    z, recon_x, edge_probs = predict_structure(structure, encoder, decoder)
    z = z.detach().cpu().numpy()
    zvals_stack.append(z)
    
    edge_probs_stack.append(edge_probs.detach().cpu().numpy())
    print(z.shape)

zvals = np.vstack(zvals_stack)
#edge_probs_stack = np.vstack(edge_probs_stack)
print(zvals.shape)


# In[ ]:


plot_embeddings(filename, encoder, decoder)


# In[ ]:


#make a an examples folder and copy the example structures there
import shutil

if not os.path.exists('./examples'):
    os.makedirs('./examples')
for structure in structures[0:10]:
    shutil.copy(structure, './examples/' + structure.split('/')[-1] )


# In[ ]:


#add the encoded structures to the struct align hdf5
import time
import h5py
pdbfiles_structalign = glob.glob(datadir + 'struct_align/*/structs/*.pdb')
f2 = h5py.File('./structs_structalign.h5' , 'r')
struct_dat2 = StructAlnDataset(f2,pdbfiles_structalign )
struct_dat2[10]
aln_loader = DataLoader( struct_dat2 , batch_size=1, shuffle=False)
#encode each element in the dataset and store in a new hdf5 file


# In[ ]:


import json
#turn structures into chains of characters. store the contact points in a json object
#start with ML model with only structural characters. if that works add the contact points
def encode_structures( dataloader, encoder, decoder, filename = 'structalign.strct.fasta' , structlist = None):
    with open( filename , 'w') as f:
        for i,data in tqdm.tqdm(enumerate(dataloader)):
            data = data.to(device)
            z,qloss = encoder.forward(data['res'].x , data['AA'].x , data.edge_index_dict)
            strdata = encoder.vector_quantizer.discretize_z(z)
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
encode_structures(aln_loader, encoder, decoder, filename = 'structalign.strct.fasta' , structlist=struct_dat2.structlist)


