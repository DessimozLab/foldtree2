#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import networkx as nx
#get known complexes
datadir = '/home/dmoi/datasets/foldtree2/complexes/'
complexdf = pd.read_csv(datadir+'contactDefinition.txt' , sep='\t')
print(complexdf.head())

complexdf['accession'] = complexdf.code.map(lambda x: x.split('_')[0])


import glob
pdbs = glob.glob(datadir+'BU_all_renum/*.pdb')
#remove fixed 
pdbs = [p for p in pdbs if 'fixed' not in p]
print(len(pdbs))


codes = { f.split('/')[-1].split('.')[0]:f for f in pdbs}
complexdf['pdbfile'] = complexdf['code'].map(codes)
print(complexdf.head())


#sample 2000 codes
import random
sub = complexdf.code.unique().tolist()
random.shuffle(sub)
sub = sub[:2000]
sub = complexdf[complexdf.code.isin(sub)]
print(sub.head())
print( len(sub) )



print( sub.pdbfile.unique() )
print( len(sub.pdbfile.unique() ) )



#run pdbfixer on all pdbs
from pdbfixer import PDBFixer
from openmm.app import PDBFile
import tqdm
import os
import pebble 
import concurrent.futures

def fix_pdb(pdbfile):
    try:
        if os.path.exists(pdbfile.replace('.pdb', '_fixed.pdb')):
            return pdbfile.replace('.pdb', '_fixed.pdb')
        
        fixer = PDBFixer(filename=pdbfile)  
        fixer.findMissingResidues()
        fixer.findNonstandardResidues()
        fixer.replaceNonstandardResidues()
        fixer.removeHeterogens(True)
        fixer.findMissingAtoms()
        fixer.addMissingAtoms()
        outfile = pdbfile.replace('.pdb', '_fixed.pdb')
        with open(outfile, 'w') as w:
            PDBFile.writeFile(fixer.topology, fixer.positions, w)
        return outfile
    except Exception as e:
        print(e)
        return None
    

fix_pdbs = False
if fix_pdbs == True:
    with pebble.ProcessPool() as pool:
        futures = pool.map(fix_pdb, tqdm.tqdm(sub.pdbfile.unique().tolist()), timeout=60, workers=8, chunksize=8)
        results = []
        for future in tqdm.tqdm(concurrent.futures.as_completed(futures)):
            try:
                result = future.result()
                print(result)
                results.append(result)
            except TimeoutError as e:
                print(e)
            except Exception as e:
                print("Error processing PDB:", e)

    # Add fixed filepaths to sub
    sub['fixed_pdb'] = results
    print( sub.head())


fixed = glob.glob(datadir+'BU_all_renum/*_fixed.pdb')
print( len(fixed) )
print( fixed[:20] )


#get pdb ids from fixed
fixed = {f.split('/')[-1].split('_')[0]:f for f in fixed}
print( len(fixed) )
#filter complex df to only fixed pdbs
sub = complexdf[complexdf.accession.isin(fixed.keys())]
sub['fixed_pdb'] = sub.accession.map(fixed)
print(sub.head())
print( len(sub) )

#make a complex dataset of pytorch geometric objects
import foldtree2_ecddcd as ft2
#reload ft2
import importlib
importlib.reload(ft2)
converter = ft2.PDB2PyG()
converter.store_pyg_complexdata(sub.fixed_pdb.unique() , datadir+'pyg_complexes.h5')

