import os.path as osp
import torch
from torch_geometric.data import Data, Dataset
from torch_geometric.data import HeteroData
import scipy.sparse

def sparse2pairs(sparsemat):
    sparsemat = scipy.sparse.find(sparsemat)
    return np.vstack([sparsemat[0],sparsemat[1]])

def struct2pyg(pdbchain , aaproperties= aaproperties):
    data = HeteroData()
    #transform a structure chain into a pytorch geometric graph
    #get the adjacency matrices
    angles, contact_points, springmat , backbone = create_features(pdbchain, aaproperties)
    
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
    data['res','spring_mat', 'res'].edge_attr = torch.tensor(springmat.data, dtype=torch.float)

    backbone = sparse2pairs(backbone)
    contact_points = sparse2pairs(contact_points)
    hbond_mat = sparse2pairs(hbond_mat)
    springmat = sparse2pairs(springmat)

    #get the adjacency matrices into tensors
    data['res','backbone','res'].edge_index = torch.tensor(backbone,  dtype=torch.long )
    data['res','contact_points', 'res'].edge_index = torch.tensor(contact_points,  dtype=torch.long )    
    data['res','hbond_mat', 'res'].edge_index = torch.tensor(hbond_mat,  dtype=torch.long )
    data['res','spring_mat', 'res'].edge_index = torch.tensor(springmat,  dtype=torch.long )

    #add self loops
    data['res','backbone','res'].edge_index = torch_geometric.utils.add_self_loops(data['res','backbone','res'].edge_index)[0]
    data['res','contact_points', 'res'].edge_index = torch_geometric.utils.add_self_loops(data['res','contact_points', 'res'].edge_index)[0]
    data['res','hbond_mat', 'res'].edge_index = torch_geometric.utils.add_self_loops(data['res','hbond_mat', 'res'].edge_index)[0]

    #normalize features

    data['res'].x = torch_geometric.utils.normalize_features(data['res'].x)
    data['res','backbone','res'].edge_attr = torch_geometric.utils.normalize_edge_attr(data['res','backbone','res'].edge_attr)
    data['res','contact_points', 'res'].edge_attr = torch_geometric.utils.normalize_edge_attr(data['res','contact_points', 'res'].edge_attr)
    data['res','hbond_mat', 'res'].edge_attr = torch_geometric.utils.normalize_edge_attr(data['res','hbond_mat', 'res'].edge_attr)

    return data

#create a function to store the pytorch geometric data in a hdf5 file

def store_pyg(pdbfiles, aaproperties, filename, verbose = False):
    with h5py.File(filename, 'pdbfiles') as f:
        for pdbfile in pdbfiles:
            if verbose:
                print(pdbfile)
            data = struct2pyg(pdbfile, aaproperties)
            f.create_group(pdbfile)
            for key in data.keys():
                f.create_dataset(pdbfile + '/' + key, data=data[key])



class StructureDataset(Dataset):
    def __init__(self, h5dataset):
        super().__init__()
        #keys should be the structures
        self.structures = h5dataset

    def __len__(self):
        return len(self.structures)

    def __getitem__(self, idx):
        structure = self.structures[idx]
        data = {}
        for key in structure.keys():
            data[key] = torch.tensor(structure[key])
        #return pytorch geometric heterograph
        data = HeteroData(data)
        return data
    


#create pytorch lightning dataloaders
    
from torch_geometric.data import DataLoader
import pytorch_lightning as pl
class StructureDataModule(pl.LightningDataModule):
    def __init__(self, structuredir, batch_size=32):
        super().__init__()
        self.structuredir = structuredir
        self.batch_size = batch_size
        self.dataset = StructureDataset(self.structuredir)

    def setup(self, stage=None):
        #split the dataset into training, validation, and test sets
        self.train_dataset, self.val_dataset, self.test_dataset = torch.utils.data.random_split(self.dataset, [int(0.8*len(self.dataset)), int(0.1*len(self.dataset)), int(0.1*len(self.dataset))])

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)