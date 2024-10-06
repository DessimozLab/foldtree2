
#!/usr/bin/env python
# coding: utf-8


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




class HeteroGAE_Pairwise_Decoder(torch.nn.Module):
    #we don't need to decode to aa... just contact probs
    def __init__(self, encoder_out_channels, xdim=20, hidden_channels={'res_backbone_res': [20, 20, 20]}, out_channels_hidden=20, nheads = 1 , Xdecoder_hidden=30, metadata={}, amino_mapper= None):
        super(HeteroGAE_Pairwise_Decoder, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.metadata = metadata
        self.hidden_channels = hidden_channels
        self.out_channels_hidden = out_channels_hidden
        self.in_channels = encoder_out_channels
        for i in range(len(self.hidden_channels[('res', 'backbone', 'res')])):
            self.convs.append(
                torch.nn.ModuleDict({
                    '_'.join(edge_type): SAGEConv(self.in_channels if i == 0 else self.hidden_channels[edge_type][i-1], self.hidden_channels[edge_type][i]  )
                    for edge_type in [('res', 'backbone', 'res')]
                })
            )
        self.lin = Linear(hidden_channels[('res', 'backbone', 'res')][-1], self.out_channels_hidden)
        self.sigmoid = Sigmoid()
    def forward(self, z1, z2, edge_index, backbones, **kwargs):
        zs = []
        for z in [z1,z2]:
            inz = z
            for layer in self.convs:
                for edge_type, conv in layer.items():
                    z = conv(z, backbones[tuple(edge_type.split('_'))])
                    z = F.relu(z)
            z = self.lin(z)
            zs.append(z)
        sim_matrix = (zs[0][edge_index[0]] * zs[1][edge_index[1]]).sum(dim=1)
        edge_probs = self.sigmoid(sim_matrix)
        
        return edge_probs




class VectorQuantizerEMA(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost, decay=0.99, epsilon=1e-5, reset_threshold=100000, reset = True):
        super(VectorQuantizerEMA, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.epsilon = epsilon
        self.reset_threshold = reset_threshold
        self.reset = reset
        # Initialize the codebook with uniform distribution
        self.embeddings = nn.Embedding(num_embeddings, embedding_dim)
        self.embeddings.weight.data.uniform_(-1 / self.num_embeddings, 1 / self.num_embeddings)

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
        total_loss = loss - entropy_reg + diversity_reg + kl_div_reg

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
    def __init__(self, in_channels, hidden_channels, out_channels, num_embeddings, commitment_cost, metadata={} , encoder_hidden = 100 , dropout_p = 0.05 , EMA = False , reset_codes = True ):
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
        if EMA == False:
            self.vector_quantizer = VectorQuantizer(num_embeddings, out_channels, commitment_cost)
        else:
            self.vector_quantizer = VectorQuantizerEMA(num_embeddings, out_channels, commitment_cost , reset = reset_codes)
        
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

    def encode_structures( dataloader, encoder, filename = 'structalign.strct' ):
        #write with contacts 
        with open( filename , 'w') as f:
            for i,data in tqdm.tqdm(enumerate(dataloader)):
                data = data.to(self.device)
                z,qloss = self.forward(data['res'].x , data['AA'].x , data.edge_index_dict)
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
        #write an encoded fasta for use with mafft and iqtree. only doable with alphabet size of less that 248
        #0x01 – 0xFF excluding > (0x3E), = (0x3D), < (0x3C), - (0x2D), Space (0x20), Carriage Return (0x0d) and Line Feed (0x0a)
        replace_dict = { '>' : chr(249), '=' : chr(250), '<' : chr(251), '-' : chr(252), ' ' : chr(253) , '\r' : chr(254), '\n' : chr(255) }
        #check encoding size
        if self.vector_quantizer.num_embeddings > 248:
            raise ValueError('Encoding size too large for fasta encoding')
        
        with open( filename , 'w') as f:
            for i,data in tqdm.tqdm(enumerate(dataloader)):
                data = data.to(self.device)
                z,qloss = self.forward(data['res'].x , data['AA'].x , data.edge_index_dict)
                strdata = self.vector_quantizer.discretize_z(z)
                identifier = data.identifier
                f.write(f'>{identifier}\n')
                outstr = ''
                for char in strdata[0]:
                    #start at 0x01
                    char = chr(char+1)
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
                z,qloss = self.forward(data['res'].x , data['AA'].x , data.edge_index_dict)
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
    




class HeteroGAE_VariationalQuantizedEncoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_embeddings, commitment_cost, metadata={}):
        super(HeteroGAE_VariationalQuantizedEncoder, self).__init__()
        #save all arguments to constructor
        self.args = locals()
        self.args.pop('self')
        
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
        
        # Setting the seed
        L.seed_everything(42)
        # Ensure that all operations are deterministic on GPU (if used) for reproducibility
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

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
