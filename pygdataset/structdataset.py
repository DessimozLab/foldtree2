# This file contains the classes for the StructureDataset and ComplexDataset classes.
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
    