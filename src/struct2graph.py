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
from torch_geometric.data import HeteroData
import scipy.sparse
import torch
from prody import *
import pandas as pd


aaproperties = pd.read_csv('./aaindex1.csv', header=0)
aaproperties.drop( [ 'description' , 'reference'  ], axis=1, inplace=True)
#one hot encoding
onehot= np.fill_diagonal(np.zeros((20,20)), 1)
#append to the dataframe
aaproperties = pd.concat([aaproperties, pd.DataFrame(onehot) ], axis=1)



aaproperties = pd.read_csv('./aaindex1.csv', header=0)
aaproperties.drop( [ 'description' , 'reference'  ], axis=1, inplace=True)
print(aaproperties.columns)
print( len(aaproperties) )
#todo create the s2g class to handle the conversion of structures to graphs
#one hot encoding
onehot= np.fill_diagonal(np.zeros((20,20)), 1)
onehot = pd.DataFrame(onehot)
#change to integers instead of bool
onehot = onehot.astype(int)
#append to the dataframe
aaproperties = aaproperties.T
aaproperties = pd.concat([aaproperties, onehot ], axis=1)






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

def get_backbone(chain):
    contact_mat = np.zeros((len(chain), len(chain)))
    #fill diagonal with 1s
    np.fill_diagonal(contact_mat, 1)
    return contact_mat

def ret_hbonds(chain , verbose = False):
    #loop through all atoms in a structure
    struct = PDBParser().get_structure('1eei', filename)
    typeindex = {'N':0, 'O':1, 'C':2}
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
    one_hot = pd.get_dummies(nodeprops['single_letter_code']).astype(int)
    nodeprops = nodeprops.join(one_hot)
    nodeprops = nodeprops.drop(columns=['single_letter_code'])
    return nodeprops


def anm_analysis(filename):
    prot = parsePDB( filename)
    calphas2 = prot.select('calpha')
    anm = ANM('ANM analysis')
    anm.buildHessian(calphas2)
    anm.calcModes()

    cov = anm.getCovariance()
    cov[ cov < 0] = -cov[ cov < 0]

    logcov = np.log(cov)
    #get the top 10% of the covariance matrix
    top = np.percentile(logcov, 90)
    logcov[ logcov < top] = 0
    return logcov
