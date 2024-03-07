import pandas as pd
from Bio import PDB
import warnings
import numpy as np
import torch
import pydssp
from Bio.PDB import PDBParser   
import numpy as np
from prody import *
import networkx as nx
#transform the contact matrices into a networkx multigraph 
import colour 
from torch_geometric.data import HeteroData
import scipy.sparse
import torch


aaproperties = pd.read_csv('./aaindex1.csv', header=0)
aaproperties.drop( [ 'description' , 'reference'  ], axis=1, inplace=True)
#one hot encoding
onehot= np.fill_diagonal(np.zeros((20,20)), 1)
#append to the dataframe
aaproperties = pd.concat([aaproperties, pd.DataFrame(onehot) ], axis=1)


def anm_analysis(filename):
    prot = parsePDB( filename)
    calphas2 = prot.select('calpha')
    anm = ANM('ANM analysis')
    anm.buildHessian(calphas2)
    anm.calcModes()

    cov = anm.getCovariance()

    return eigvals, eigvecs, cov

    
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
    print(polypeptides)


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
    #transform phi and psi angles into a dataframe
    phi_psi_angles = pd.DataFrame(phi_psi_angles)
    #transform the residue names into single letter code
    return phi_psi_angles    

def get_contact_points(chain, distance):
    contact_mat = np.zeros((len(chain), len(chain)))
    for i,r1 in enumerate(chain):
        for j,r2 in enumerate(chain):
            if i< j:
                if 'CA' in r1 and 'CA' in r2:
                    if r1['CA'] - r2['CA'] < distance:
                        contact_mat[i,j] =  r1['CA'] - r2['CA']
    contact_mat = contact_mat + contact_mat.T
    return contact_mat


def ret_hbonds(chain , verbose = False):
    #loop through all atoms in a structure
    struct = PDBParser().get_structure('1eei', filename)

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

def add_aaproperties(angles, aaproperties):
    nodeprops = angles.merge(aaproperties, left_on='single_letter_code', right_index=True, how='left')
    nodeprops = nodeprops.dropna()

    #generate 1 hot encoding for each amino acid

    one_hot = pd.get_dummies(nodeprops['single_letter_code'])
    nodeprops = nodeprops.join(one_hot)
    nodeprops = nodeprops.drop(columns=['single_letter_code'])

    return nodeprops

#create features from a chain

def create_features(chain, aaproperties, verbose = False):
    angles = get_angles(chain)
    angles = add_aaproperties(angles, aaproperties)
    angles = angles.dropna()
    angles = angles.reset_index(drop=True)
    angles = angles.drop(['Residue_Name', 'single_letter_code'], axis=1)
    angles = angles.set_index(['Chain', 'Residue_Number'])
    angles = angles.sort_index()
    angles = angles.dropna()
    angles = angles.reset_index()
    angles = angles.drop(['Chain', 'Residue_Number'], axis=1)
    angles = angles.values
    angles = angles.astype('float32')
    angles = torch.tensor(angles)
    
    if verbose:
        print(angles.shape)
    
    
    return angles





def tensor_to_multigraph(adjacency_tensor):
    # Initialize a MultiGraph
    G = nx.MultiGraph()
    num_nodes = adjacency_tensor.shape[1]
    G.add_nodes_from(range(num_nodes))
    colors = [ c.hex_l for c in  colour.Color('red').range_to(colour.Color('green'), adjacency_tensor.shape[0]) ]
    # Iterate through the adjacency matrices in the tensor
    for i, adj_matrix in enumerate(adjacency_tensor):
        # Add nodes to the MultiGraph
        # Iterate through the rows and columns of the adjacency matrix to add edges
        for row in range(num_nodes):
            for col in range(num_nodes):
                if adjacency_tensor[i,row, col] != 0:
                    # Add an edge with weight (if needed) to the MultiGraph
                    G.add_edge(row, col, weight=adjacency_tensor[i,row, col] , color = colors[i],  layer= i )
    return G

def sparse2pairs(sparsemat):
    sparsemat = scipy.sparse.find(sparsemat)
    return np.vstack([sparsemat[0],sparsemat[1]])

def struct2pyg(pdbfile):
    data = HeteroData()
    pdbchain = read_pdb(pdbfile)[0]
    #transform a structure chain into a pytorch geometric graph
    #get the adjacency matrices
    backbone = scipy.sparse.coo_matrix(  get_backbone(pdbchain) )
    contact_points = scipy.sparse.coo_matrix( get_contact_points(pdbchain, 8) )
    hbond_mat = scipy.sparse.coo_matrix(  ret_hbonds(pdbchain) )
    eigvals, eigvecs, cov = anm_analysis(pdbfile)

    #get the adjacency matrices into tensors
    data['res','backbone','res'].edge_index = torch.tensor(backbone,  dtype=torch.long )
    data['res','contact_points', 'res'].edge_index = torch.tensor(contact_points,  dtype=torch.long )    
    data['res','hbond_mat', 'res'].edge_index = torch.tensor(hbond_mat,  dtype=torch.long )

    #get the node features
    angles = get_angles(pdbchain)
    angles = add_aaproperties(angles, aaproperties)
    angles = angles.set_index('Residue_Number')
    angles = angles.sort_index()
    angles = angles.drop(['Residue_Name', 'single_letter_code'], axis=1)
    angles = angles.drop(['Chain'], axis=1)
    angles = torch.tensor(angles.values, dtype=torch.float)
    data['res'].x = angles

    #get the edge features
    data['res','backbone','res'].edge_attr = torch.tensor(backbone.data, dtype=torch.float)
    data['res','contact_points', 'res'].edge_attr = torch.tensor(contact_points.data, dtype=torch.float)
    data['res','hbond_mat', 'res'].edge_attr = torch.tensor(hbond_mat.data, dtype=torch.float)
    data['res','cov_anm', 'res'].edge_attr = torch.tensor(cov, dtype=torch.float)

    #add self loops
    data['res','backbone','res'].edge_index = torch_geometric.utils.add_self_loops(data['res','backbone','res'].edge_index)[0]
    data['res','contact_points', 'res'].edge_index = torch_geometric.utils.add_self_loops(data['res','contact_points', 'res'].edge_index)[0]
    data['res','hbond_mat', 'res'].edge_index = torch_geometric.utils.add_self_loops(data['res','hbond_mat', 'res'].edge_index)[0]

    #normalize features

    data['res'].x = torch_geometric.utils.normalize_features(data['res'].x)
    data['res','backbone','res'].edge_attr = torch_geometric.utils.normalize_edge_attr(data['res','backbone','res'].edge_attr)
    data['res','contact_points', 'res'].edge_attr = torch_geometric.utils.normalize_edge_attr(data['res','contact_points', 'res'].edge_attr)
    data['res','hbond_mat', 'res'].edge_attr = torch_geometric.utils.normalize_edge_attr(data['res','hbond_mat', 'res'].edge_attr)
    data['res','cov_anm', 'res'].edge_attr = torch_geometric.utils.normalize_edge_attr(data['res','cov_anm', 'res'].edge_attr)

    return data
