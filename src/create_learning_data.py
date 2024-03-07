
#command line arguments
import sys
import os
import argparse
import numpy as np
import AFDB_tools as afdb
import dask.dataframe as dd
import dask.delayed as delayed
import struct2graph as s2g
#import pytorch geometric dataset sqlite backend
from torch_geometric.data import Data , SQLiteDatabase
import random

def pyg2disk(graph,i,graphdir):
    torch.save(graph , graphdir + str(i) + '.pt')

#parse the command line arguments
parser = argparse.ArgumentParser(description='Create learning data for the protein folding project')
parser.add_argument('input', type=str, help='input file')
parser.add_argument('struct_dir', type=str, help='output structure directory')
parser.add_argument( 'graph_db', type=str, help='output graph directory')
parser.add_argument( 'nsamples', type=int, help='number of sample structs')

inputcsv = sys.argv[1]
struct_dir = sys.argv[2]
graph_dir = sys.argv[3]
nsamples = sys.argv[4]

#load the data from the afdb cluster dataset using dask dataframe
#load the data
df = dd.read_csv( inputcsv )

#download pdbs for representative structures
pdb_ids = list(df['pdb_id'].unique().compute())
pdb_ids = random.shuffle(pdb_ids)[0:nsamples]
#use delayed to parallelize the download

#download the pdbs
results = [delayed(afdb.grab_struct)(pdb_id, struct_dir) for pdb_id in pdb_ids]
results = delayed(results)
results.compute()

#create a graph for each protein
structs= os.listdir(struct_dir)
results = [delayed(s2g.struct2pyg)(pdbfile) for pdbfile in structs]
results = [delayed(pyg2disk)(graph,i,graphdir) for graph in results]
results.compute()

