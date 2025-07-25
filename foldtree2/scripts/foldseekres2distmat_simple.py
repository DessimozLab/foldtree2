from src import foldseek2tree
import numpy as np
import pandas as pd

res = pd.read_table(snakemake.input[0], header = None)
print(res.head())
#get the folder of the input file
infolder = snakemake.input[0].split('/')[:-1]
infolder = ''.join( [i + '/' for i in infolder])+'/'
res[0] = res[0].map(lambda x :x.replace('.pdb', ''))
res[1] = res[1].map(lambda x :x.replace('.pdb', ''))

if snakemake.params.fmt is None:
    res.columns = 'query,target,fident,alnlen,mismatch,gapopen,qstart,qend,tstart,tend,evalue,bits,lddt,lddtfull,alntmscore'.split(',')
else:
    res.columns = snakemake.params.fmt.split(',')
ids = list( set(list(res['query'].unique()) + list(res['target'].unique())))
pos = { protid : i for i,protid in enumerate(ids)}
kernels = ['fident', 'alntmscore', 'lddt']

#set kernel columns to float
for k in kernels:
    res[k] = res[k].astype(float)
#change nan to 0
res = res.fillna(0)
matrices = { k:np.zeros((len(pos), len(pos))) for k in kernels }
print(res)

#calc kernel for tm, aln score, lddt
for idx,row in res.iterrows():
    for k in matrices:
        matrices[k][pos[row['query']] , pos[row['target']]] += row[k]
        matrices[k][pos[row['target']] , pos[row['query']]] += row[k]

for i,k in enumerate(matrices):
    matrices[k] /= 2
    matrices[k] = 1-matrices[k]
    print(matrices[k], np.amax(matrices[k]), np.amin(matrices[k]) )
    if k == 'fident':
        bfactor=.93
    else:
        bfactor=1
    tajima =  foldseek2tree.Tajima_dist(matrices[k] + 10 **-5 , bfactor=bfactor )    
    np.fill_diagonal(tajima, 0)
    matrices[k] = tajima
    np.save( infolder + k + '_distmat.npy' , matrices[k])
    distmat_txt = foldseek2tree.distmat_to_txt( ids , matrices[k] , snakemake.output[i] )
