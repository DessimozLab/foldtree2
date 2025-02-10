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
import torch
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.utils import to_networkx
from torch_geometric.data import HeteroData
from torch_geometric.nn import Linear, AGNNConv , TransformerConv, GATv2Conv, GCNConv, SAGEConv, MFConv , GENConv , JumpingKnowledge
from einops import rearrange
from torch_geometric.nn.dense import dense_diff_pool as DiffPool
from torch.nn import ModuleDict, ModuleList, L1Loss
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn.aggr import SoftmaxAggregation
from torch_geometric.utils import negative_sampling
import os
import urllib.request
from urllib.error import HTTPError
import pytorch_lightning as L
import scipy.sparse
import tqdm
import src.egnlayer as egnlayer
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.data import Data, Dataset
from pytorch_lightning.callbacks import ModelCheckpoint
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import HeteroConv
from datasketch import WeightedMinHashGenerator , MinHashLSHForest
import numpy as np
import pandas as pd
from Bio import PDB
import pydssp
import polars as pl
from Bio.PDB import PDBParser
import torch.nn.functional as F
from titans_pytorch import NeuralMemory
import src.losses.fafe as fafe 
EPS = 1e-15

datadir = '../../datasets/foldtree2/'
