import sys
sys.path.append('/home/dmoi/projects/foldtree2')
#read the afdb clusters file
import pandas as pd
import numpy as np
import glob
import os
#autoreload
import pickle
from src import AFDB_tools
import tqdm

datadir = '../../datasets/foldtree2/'

repdirs = '../../datasets/foldtree2/struct_align/'
dl_structs = False
from src import foldseek2tree
aln_structs = True


#read the afdb rep file
reps = pd.read_table( datadir + 'afdbclusters/1-AFDBClusters-entryId_repId_taxId.tsv', header=None, names=['entryId', 'repId', 'taxId'] )
print( 'reps' , reps.head() )
#make a structure alignment directory
if not os.path.exists( 'struct_align' ):
    os.makedirs( 'struct_align' )

#make a directory for each cluster representative
for rep in reps['repId']:
    if not os.path.exists( datadir +'struct_align/' + rep  ):
        os.makedirs(datadir + 'struct_align/' + rep  )
    if not os.path.exists( datadir+ 'struct_align/' + rep  + '/structs/'):
        os.makedirs( datadir+ 'struct_align/' + rep + '/structs/' )


#download n struct members for each cluster
if dl_structs == True:
    n = 5
    for rep in tqdm.tqdm(reps.repId.unique() ):
        subdf = reps[ reps['repId'] == rep ]
        if len(subdf) < n:
            n = len(subdf)
        subdf = subdf.sample( n = n  )
        subdf = subdf.head( n )
        #download the structures
        for uniID in subdf['entryId']:
            AFDB_tools.grab_struct(uniID , structfolder=datadir+'struct_align/' + rep  + '/structs/')


#for each folder in struct_align, align the structures with all vs all using foldseek

if aln_structs == True:
    for rep in tqdm.tqdm(reps.repId.unique() ):
        #align the structures
        #check if there are structures in the folder
        if len( glob.glob( datadir + 'struct_align/' + rep + '/structs/*' ) ) > 0:
            foldseek2tree.runFoldseek_allvall_EZsearch( infolder= datadir + 'struct_align/' + rep  + '/structs/', outpath=datadir+'struct_align/' + rep + '/allvall.csv' )
