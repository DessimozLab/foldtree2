import snakemake.utils
snakemake.utils.min_version("7.8.0")
snake_dir = workflow.basedir
rootdir = ''.join([ sub + '/' for sub in snake_dir.split('/')[:-1] ] )
print(' benchmark all running in ' , rootdir)
configfile: rootdir+ "workflow/config/config_vars.yaml"
# remote homologues search parameters

foldseekpath = config["foldseek_path"]
if foldseekpath == 'provided':
	foldseekpath = rootdir + "foldseek/foldseek"

if 'folder' in config:
	if type(config['folder']) == str:
		folders = [config['folder']]
	else:
		folders = config['folder']
else:
	folders = glob_wildcards("{folders}/identifiers.txt").folders
	config['folder']  = folders

if config['fam_limit'] > 0:
	folders = folders[:config['fam_limit']]
	config['folder']  = folders

if config['no_treescores'] and config['no_treescores']== True:
	no_treescores= True
else:
	no_treescores = False

models = ['monodecoders_smallGAE', 'monodecoders_bigGAE', 'monodecoders_heteroGAE' ]


rule all:
input:
	# ML3ditree trees and scores
	expand("{folder}/templateX{alntype}.nx.treefile.rooted.final", folder=folders, alntype=["3di","AA"]),
	expand("{folder}/templateX{alntype}.nx.treefile.rooted.final.treescore", folder=folders, alntype=["3di","AA"]),
	# FT2 tree and TCS score
	expand("{folder}/{model}_FT2_tree.nwk", folder=folders, model=models),
	expand("{folder}/{model}_FT2_tree.nwk.tcs", folder=folders, model=models),
	# Fold_tree1 tree and TCS score
	expand("{folder}/{model}_foldtree1_tree.nwk", folder=folders, model=models),
	expand("{folder}/{model}_foldtree1_tree.nwk.tcs", folder=folders, model=models),
	#regular raxmlng tree and TCS score
	


module tcs_benchmark:
	# here, plain paths, URLs and the special markers for code hosting providers (see below) are possible.
	snakefile: "tcs_benchmark"
	config: config
use rule * from tcs_benchmark as BM_*

module align:
	# here, plain paths, URLs and the special markers for code hosting providers (see below) are possible.
	snakefile: "align"
	config: config
use rule * from FidentXaln as AL_*

module raxmlng:
	# here, plain paths, URLs and the special markers for code hosting providers (see below) are possible.
	snakefile: "raxmlng"
	config: config
use rule * from iqtreeXsingle as RM_*

module foldtree:
	# here, plain paths, URLs and the special markers for code hosting providers (see below) are possible.
	snakefile: "fold_tree"
	config: config
use rule * from fold_tree as FTI_*

module foldtree2:
	# here, plain paths, URLs and the special markers for code hosting providers (see below) are possible.
	snakefile: "ft2"
	config: config
use rule * from fold_tree as FTII_*


module ML3di:
	# here, plain paths, URLs and the special markers for code hosting providers (see below) are possible.
	snakefile: "ml3ditree"
	config: config
use rule * from fold_tree as ML3di_*


