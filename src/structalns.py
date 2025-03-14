import subprocess
import os
import glob
import toytree
import tqdm
import pandas as pd
from subprocess import PIPE, Popen
import shlex
from Bio import SeqIO
import random
import itertools
import numpy as np


def remove_seeds( alnfile):
    """
    this function removes seeds from an alignment
    """
    aln = SeqIO.parse(alnfile, 'fasta')
    sequences = []
    for s in aln:
        sequences.append(s)
    #write new aln
    
    with open(alnfile, 'w') as f:
        for s in sequences:
            f.write('>' + str(s.id).replace('seed','').replace('_' , '') + '\n')
            f.write(str(s.seq) + '\n')
    
    return alnfile

def Fident(str1,str2 , verbose = False):
    #minlen= min( (len(str1),len(str2))  )
    #str1 = str1[:minlen]
    #str2 = str2[:minlen]
    str1 = np.array(list(str1))
    str2 = np.array(list(str2))            
    return len(np.where( (str1 == str2 ) & (str1 != '-' ) & (str2 != '-')  )[0]) / len(str1)

def copyaln( aln, seq):
    seqiter = iter(seq)
    newaln = ''
    for i,char in enumerate(aln):
        if char == '-':
            newaln += '-'
        else:
            newaln+=next(seqiter)
    return newaln

def read_dbfiles3di(  AADB , threeDidb):
    #find positions 
    threeDiseq = [ l.strip().replace('\x00','') for l in open(threeDidb)]
    lookup = AADB+'.lookup'
    ids = [ l.split()[1].strip().replace('.pdb', '').split('/')[-1] for l in open(lookup)]
    AAs = [ l.strip().replace('\x00','') for l in open(AADB)]

    mapper3di = dict(zip(ids,threeDiseq))
    mapperAA = dict(zip(ids,AAs))
    
    return mapper3di, mapperAA

def calc_fident_crossaln(row , verbose = False):
    #amino acid representations of alns using AAand3di or just 3di
    qaln_2, taln_2 = row.qaln , row.taln
    #start and stop of aln
    
    qstart_2, qend_2, tstart_2 , tend_2 = row.qstart, row.qend , row.tstart , row.tend
    #indexing starts at 1...
    
    #3di of the query and target
    structQ, structT = row['3diq'], row['3dit']
    AAq, AAt = row['AAq'], row['AAt']

    #add gaps
    t3diAA_newgaps = copyaln(taln_2, structT[tstart_2-1:tend_2]) 
    q3diAA_newgaps = copyaln(qaln_2, structQ[qstart_2-1:qend_2])
    row = pd.Series( { '3di_qaln_mode2':q3diAA_newgaps , '3di_taln_mode2':t3diAA_newgaps })
    #return columns
    return row

def get_leafset( treenode ):
    """
    this function returns the leafset of a node
    """
    if treenode.is_leaf():
        return [treenode.name]
    else:
        return treenode.get_leaf_names()



def mafft_profile(aln1,aln2, outprofile , submat = None):
    """
    this function aligns two alignments using clustalo
    """
    #make profile
    cmd = 'clustalo --is-profile --profile1 {} --profile2 {} > {}'.format( aln1,aln2, outprofile)
    print(cmd)
    subprocess.run(cmd , shell=True)
    return outprofile



def mafft_addfull(aln1,aln2, outprofile , submat = None):
    """
    this function aligns two alignments using MAFFT
    """
    #make profile
    profile = aln1 + '.profile'
    if submat is not None:
        cmd = 'mafft --textmatrix {} --addfull {} {} > {}'.format(submat, aln1,aln2, outprofile)
    else:
        cmd = 'mafft --addfull  {} {} > {}'.format(aln1,aln2, outprofile)
    print(cmd)

    subprocess.run(cmd , shell = True)
    return outprofile


def sub2fasta( sub, outfile , fastacol1='qaln' , fastacol2='taln' ):
    with open(outfile, 'w') as f:
        f.write('>' + sub['query'] + '\n')
        f.write(sub[fastacol1] + '\n')
        f.write('>' + sub['target'] + '\n')
        f.write(sub[fastacol2] + '\n')    
    return outfile

def retalns(allvall, leafname1,leafname2):
    sub = allvall[allvall['query'].isin( leafname1)]
    sub = sub[sub['target'].isin(leafname2)]
    sub = sub[sub['query'] != sub['target']]
    #get max prot lenght aligned
    sub['alnlen'] = sub.apply(lambda x: max(x['qend'] - x['qstart'] , x['tend'] - x['tstart']) , axis = 1)
    sub = sub[sub['alnlen'] == sub['alnlen'].max()]
    if len(sub)==0:
        print(leafname1, leafname2)
        raise Exception('no sub')
    return sub.iloc[0]

def get_fasta_leafset(fasta):
    """
    this function returns the leafset of a fasta file
    """
    aln = SeqIO.parse(fasta, 'fasta')
    leafset = []
    for s in aln:
        leafset.append(s.id)
    return leafset

#traverse tree from root to leaves recursively
def traverse_tree_merge_mafft( treenode, topleafset, allvall , alnfolder , submat = None , verbose = False):
    """
    this function traverses a tree from root to leaves recursively
    it returns a dictionary with the iteratively built alignment
    """
    if verbose == True:
        print('traverse', treenode.name , treenode.is_leaf() , treenode.leafset)
    
    if treenode.is_leaf():
        print(treenode, treenode.name)
        topleafset.remove(treenode.name)
        #if the node is a leaf, then we need to add it to the alignment with one of the pivots in the current leafset
        #select the alignment of the leaf with itself
        sub = allvall[allvall['query'].isin( [treenode.name] )]
        sub = sub.iloc[0]
        
        assert len(sub) > 0


        with open(alnfolder + treenode.name + '_inter.fasta', 'w') as f:
            f.write('>' + sub['query'] + '\n')
            f.write(sub['AAq'] + '\n')
        with open(alnfolder + treenode.name + '_inter.3di.fasta', 'w') as f:
            f.write('>' + sub['query'] + '\n')
            f.write(sub['3diq'] + '\n')
        treenode.aln = alnfolder + treenode.name + '_inter.fasta'
        treenode.aln3di = alnfolder + treenode.name + '_inter.3di.fasta'
        return treenode.aln, treenode.aln3di
    
    else:
        childalns3di = {}
        childalnsAA = {}
       
        #treenode.leafset = get_leafset(treenode)
        #get the intersection of the child leafsets
        treenode.leafset = get_leafset(treenode)
        children = treenode.get_children()
        
        if len(children) == 2 and children[0].is_leaf() and children[1].is_leaf():
            #treat the case of a cherry
            print('cherry', children[0].name , children[1].name)
            treenode.aln = sub2fasta( retalns(allvall, [children[0].name] , [children[1].name]) , alnfolder + treenode.name + '_inter.fasta')
            treenode.aln3di = sub2fasta( retalns(allvall, [children[0].name] , [children[1].name]) , alnfolder + treenode.name + '_inter.3di.fasta' , fastacol1='3di_qaln_mode2' , fastacol2='3di_taln_mode2')
            return treenode.aln, treenode.aln3di
        
        else:
            #not a cherry. one or both sides is a subtree
            print('not cherry', treenode.name  )
            print( 'children', [c.name for c in children])
            for c in treenode.get_children():
                #make sub aln for each child
                if verbose == True:
                    print('traverse', c.name , c.is_leaf() , c.leafset)
                if not c.aln:
                    c.aln,c.aln3di = traverse_tree_merge_mafft(c , treenode.leafset , allvall, alnfolder , verbose = verbose , submat=submat)
                childalnsAA[c] = { 'fasta': c.aln  }
                childalns3di[c] = { 'fasta': c.aln3di  }
            
                

            if len(children) == 2:
                c1,c2 = children
                
                if c1.is_leaf():
                    treenode.aln = mafft_addfull(childalnsAA[c1]['fasta'], childalnsAA[c2]['fasta'], alnfolder + treenode.name + '_inter.fasta' )
                    treenode.aln3di = mafft_addfull(childalns3di[c1]['fasta'], childalns3di[c2]['fasta'], alnfolder + treenode.name + '_inter3di.fasta' , submat =submat )

                elif c2.is_leaf():
                    treenode.aln = mafft_addfull(childalnsAA[c2]['fasta'], childalnsAA[c1]['fasta'], alnfolder + treenode.name + '_inter.fasta' )
                    treenode.aln3di = mafft_addfull(childalns3di[c1]['fasta'], childalns3di[c2]['fasta'], alnfolder + treenode.name + '_inter3di.fasta' , submat = submat)

                else:
                    with open(childalnsAA[c2]['fasta'], 'r') as f:
                        c2seqs = f.read().count('>')
                    with open(childalnsAA[c1]['fasta'], 'r') as f:
                        c1seqs = f.read().count('>')
                    
                    if c1seqs > c2seqs:
                        treenode.aln = mafft_profile(childalnsAA[c1]['fasta'], childalnsAA[c2]['fasta'], alnfolder + treenode.name + '_inter.fasta' )
                        treenode.aln3di = mafft_profile(childalns3di[c1]['fasta'], childalns3di[c2]['fasta'], alnfolder + treenode.name + '_inter3di.fasta' , submat =submat )
                    else:
                        treenode.aln = mafft_profile(childalnsAA[c2]['fasta'], childalnsAA[c1]['fasta'], alnfolder + treenode.name + '_inter.fasta' )
                        treenode.aln3di = mafft_profile(childalns3di[c2]['fasta'], childalns3di[c1]['fasta'], alnfolder + treenode.name + '_inter3di.fasta' , submat =submat )
                    
            elif len(children) > 2 and treenode.up == None:
                print('final aln')
                print('childalnsAA', childalnsAA)
                print('childalns3di', childalns3di)
                

                print([(c.aln,c.aln3di,type(c.aln), type(c.aln3di)) for c in children])
                
                children = [c for c in treenode.get_children() if c.aln and c.aln3di]
                
                for c in children:
                    #print alns
                    print('aln' + c.name)
                    with open(childalnsAA[c]['fasta'], 'r') as f:
                        print(f.read())
                    print('aln3di' + c.name )
                    with open(childalns3di[c]['fasta'], 'r') as f:
                        print(f.read())

                c1,c2 = children[0],children[1]
                if c1.is_leaf():
                    rootfasta = mafft_addfull(childalnsAA[c1]['fasta'], childalnsAA[c2]['fasta'], alnfolder + treenode.name + '_root.fasta' )
                    rootfasta3di = mafft_addfull(childalns3di[c1]['fasta'], childalns3di[c2]['fasta'], alnfolder + treenode.name + '_inter3di.fasta' , submat =submat )
                elif c2.is_leaf():
                    rootfasta = mafft_addfull(childalnsAA[c2]['fasta'], childalnsAA[c1]['fasta'], alnfolder + treenode.name + '_root.fasta' )
                    rootfasta3di = mafft_addfull(childalns3di[c1]['fasta'], childalns3di[c2]['fasta'], alnfolder + treenode.name + '_inter3di.fasta' , submat = submat)
                else:                
                    rootfasta = mafft_profile(childalnsAA[c1]['fasta'], childalnsAA[c2]['fasta'], alnfolder + treenode.name + '_root.fasta' )
                    rootfasta3di = mafft_profile(childalns3di[c1]['fasta'], childalns3di[c2]['fasta'], alnfolder + treenode.name + '_inter3di.fasta' , submat =  submat)
                
                print('aln1')
                with open(rootfasta , 'r') as f:
                    print(f.read())
                with open(rootfasta3di , 'r') as f:
                    print(f.read())
                
                for i,c in enumerate(children[1:]):
                    if c.is_leaf():
                        rootfasta = mafft_addfull( childalnsAA[c]['fasta'] , rootfasta, rootfasta +'.iter' )
                        rootfasta3di = mafft_addfull( childalns3di[c]['fasta'], rootfasta3di , rootfasta3di+'.iter'  , submat =submat )
                    else:
                        rootfasta = mafft_profile(rootfasta, childalnsAA[c]['fasta'], rootfasta+'.iter' )
                        rootfasta3di = mafft_profile(rootfasta3di, childalns3di[c]['fasta'], rootfasta3di+'.iter'  , submat =submat )
                    print('aln'+str(i))
                    with open(rootfasta , 'r') as f:
                        print(f.read())
                    with open(rootfasta3di , 'r') as f:
                        print(f.read())
                

                treenode.aln = rootfasta
                treenode.aln3di = rootfasta3di
                
                with open(treenode.aln , 'r') as f:
                    print(f.read())
                with open(treenode.aln3di , 'r') as f:
                    print(f.read())
                
            if verbose == True:
                #check if node is root  
                if treenode.up == None:
                    print('final aln')
                    print('childalnsAA', childalnsAA)
                    print('childalns3di', childalns3di)
            return treenode.aln, treenode.aln3di


def remove_redundant( alignment ):
    """
    this function removes redundant sequences from an alignment
    """
    aln = SeqIO.parse(alignment, 'fasta')
    seqs = []
    ids = []
    for s in aln:
        if s.id not in ids:
            seqs.append(s)
            ids.append(s.id)
    
    with open(alignment, 'w') as f:
        for s in seqs:
            f.write('>' + s.id + '\n')
            f.write(str(s.seq) + '\n')
    return alignment

#remove all alns except the final merged one
def cleanup( filedir ):
    """
    this function removes all alns except the final merged one
    """
    for f in glob.glob(filedir + '*inter.fasta'):
        os.remove(f)