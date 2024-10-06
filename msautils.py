import sys
sys.path.append('/home/dmoi/projects/foldtree2')

import torch
import torch.nn.functional as F
from Bio import Phylo, AlignIO, SeqIO
from scipy.special import gamma as gamma_function
import numpy as np

# Helper function to read a tree (returns a Phylo object)
def read_tree(tree_file):
    return Phylo.read(tree_file, 'newick')

# Helper function to read a multiple sequence alignment (MSA)
def read_msa(msa_file, format='fasta'):
    return AlignIO.read(msa_file, format)

def msa2array(msa):
    #use biopython to read the alignment
    msa = AlignIO.read(msa, 'fasta')
    index = {seq.id: i for i, seq in enumerate(msa)}
    return index, np.array([list(rec) for rec in msa], np.character)

class msaarray:
    def __init__(self, msa):
        self.index, self.array = msa2array(msa)
        self.n, self.L = self.array.shape
        self.alphabet = np.unique(self.array)
        self.alphabet_size = len(self.alphabet)

    def __getitem__(self, i):
        if type(i) is slice:
            return self.array[i]
        else:
            return self.array[self.index[i]]

    def __len__(self):
        return self.n

    def __iter__(self):
        for i in self.index:
            yield i


def read_seq(seq_file, format='fasta'):
    print(seq_file)
    try:
        return SeqIO.read(seq_file, format)
    except:
        print('Error reading sequence file')
        print('Trying to read as a list of sequences')
        return [ s for s in SeqIO.parse(seq_file, format) ]
