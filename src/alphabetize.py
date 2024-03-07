#command line arguments

import sys
import os
import argparse
import numpy as np
from Bio import PDB
import warnings
import numpy as np
import struct2graph as s2g


#create a file format for the contact points and spring matrices of known structures
#contains the position and contact points of each residue of a set of proteins 
#should be a human readable format
def get_contact_points_ut(chain, distance):
    #return upper triangular contact map where x<y
    contact_mat = np.zeros((len(chain), len(chain)))
    for i,r1 in enumerate(chain):
        for j,r2 in enumerate(chain):
            if i< j:
                if 'CA' in r1 and 'CA' in r2:
                    if r1['CA'] - r2['CA'] < distance:
                        contact_mat[i,j] =  r1['CA'] - r2['CA']
    return contact_mat

def struct2contactpoints(s, radius = 8):
    chain = read_pdb(s)[0]
    contact_mat = get_contact_points_ut(chain, radius )
    #transform to sparse matrix
    contact_mat = scipy.sparse.csr_matrix(contact_mat)
    #sparse to pairs 
    contact_mat = s2g.sparse2pairs(contact_mat)
    return contact_mat

def structs2contactpts(pdbfiles, radius = 8 , threads = 1):
    contacts = []
    if threads == 1:
        for pdbfile in pdbfiles:
            contacts.append( struct2contactpoints(pdbfile, radius))
    if threads != 1:
        with mp.Pool(threads) as p:
            contacts = p.map(struct2contactpoints, [ (pdbfile, radius) for pdbfile in pdbfiles])
    return contacts

#use a trained encoder to encode a structure
def encode_structure(model, pdbfile, aaproperties, verbose = False):
    #encode the structure
    data = s2g.struct2pyg(pdbchain , aaproperties= aaproperties)
    return data

#output embeddings for a set of structures
def encode_structures(model, pdbfiles, aaproperties, threads = 1,  verbose = False):
    graphs = []
    if threads == 1:
        for pdbfile in pdbfiles:
            graphs.append( encode_structure(model, pdbfile, aaproperties, verbose = verbose))
    if threads != 1:
        with mp.Pool(threads) as p:
            graphs = p.map(encode_structure, [ (model, pdbfile, aaproperties, verbose) for pdbfile in pdbfiles])
    #create a pytorch geometric dataloader from the data objects
    dataset = geom_data.DataLoader(graphs, batch_size=1)
    #predict embeddings for the structures
    with torch.no_grad():
        model.eval()
        embeddings = [] 
        for batch in dataset:
            #move the batch to the device
            batch = batch.to(model.device)
            #predict the embeddings
            embeddings.append( model.encoder(batch.x, batch.edge_index) )
    return embeddings

#use a kmeans clustering algorithm to cluster the embeddings
#this will define our alphabet
from sklearn.cluster import KMeans
def cluster_embeddings(embeddings, nclusters = 256):
    kmeans = KMeans(n_clusters=nclusters, random_state=0).fit(embeddings)
    return kmeans

#use the kmeans model to predict the clusters for a set of embeddings
def predict_clusters(kmeans, embeddings):
    return kmeans.predict(embeddings)

def clusternum2ascii(cluster):
    return chr(cluster + 65)

def struct2ascii(embeddings, kmeans):
    clusters = list(predict_clusters(kmeans, embeddings))
    asciistr = ''.join([ clusternum2ascii(cluster) for cluster in clusters ])
    return asciistr

def structs2ascii(outfile , structures, kmeans, model, aaproperties , threads = 1 , output_contacts = False):
    #turn structures into a fasta with ascii strings
    embeddings = encode_structures(model, structures, aaproperties, threads = threads,  verbose = False)
    if output_contacts == True:
        contactpts = structs2contactpts(structures, radius = 8 , threads = threads )
    with open(outfile, 'w') as f:
        for i,structure in enumerate(structures):
            contacts = contactpts[i]
            asciistr = struct2ascii(embeddings[i], kmeans)
            f.write('>' + structure + '\n')
            #comma separated contact points
            if output_contacts == True:
                f.write('#x:' + ','.join( contacts[0] ) + '\n' )
                f.write('#y:' + ','.join( contacts[1] ) + '\n' )
            #ascii string of structural embeddings            
            f.write(asciistr + '\n')
    return outfile

#if the process is main
if __name__ == '__main__':
    #parse the command line arguments
    parser = argparse.ArgumentParser(description='Create learning data for the protein folding project')
    parser.add_argument( 'struct_dir', type=str, help='output structure directory')
    parser.add_argument( 'encoder' , type=str, help='path to the encoder model')
    parser.add_argument( 'outfile', type=str, help='output file')
    parser.add_argument( 'radius', type=int, help='radius for contact points in Angstroms')
    parser.add_argument( 'threads', type=int, help='number of threads')
    parser.add_argument( 'output_contacts', type=bool, help='output contact points')
    args = parser.parse_args()

    struct_dir = args.struct_dir
    encoder = args.encoder
    outfile = args.outfile
    radius = args.radius
    threads = args.threads
    output_contacts = args.output_contacts

