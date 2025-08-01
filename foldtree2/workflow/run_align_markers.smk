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

models = ['monodecoders_smallGAE', 'monodecoders_bigGAE', 'monodecoders_heteroGAE' ]

rule all:
	input:
		expand("{folder}/{folder}_aln.fasta", folder = folders ),
		expand('{folder}/{folder}_alnFT2{model}.fasta', folder = folders, alntype = alntype, restype = ['AA', '3di']),
		expand("{folder}/{alntype}_xaln_tree.PP.nwk.rooted.final", folder = folders , alntype = ['3di','AA']),
		expand("{folder}/templateX{alntype}.nx.treefile.rooted.final", folder = folders , alntype = alntype),
		expand('{folder}/alnfident_dist_{restype}_{alntype}.json', folder = folders, alntype = alntype, restype = ['AA', '3di']),
		expand( "{folder}/plddt.json" , folder = folders ) ,
		expand("{folder}/{mattype}_{alntype}_{exp}_struct_tree.PP.nwk.rooted", folder = folders, mattype = mattypes, alntype = alntypes, exp = exp),
		#expand("{folder}/alnAA_3di.fasta.treefile.rooted.final.treescore", folder = folders ),
		#expand("{folder}/{alntype}_xaln_tree.PP.nwk.rooted.final.treescore", folder = folders , alntype = ['3di','AA'] ),
		#expand("{folder}/templateX{alntype}.nx.treefile.rooted.final.treescore", folder = folders , alntype = alntype),
		#expand( "{folder}/{mattype}_{alntype}_{exp}_treescores_struct_tree.json" , folder = folders , mattype = mattypes , alntype = alntypes , exp=exp),
		#expand( "{folder}/treescores_sequences.{aligner}.json" , folder = folders , aligner = aligners),
		#expand( "{folder}/treescores_sequences_iq.{aligner}.json" , folder = folders , aligner = aligners),
		
module benchmarking:
	# here, plain paths, URLs and the special markers for code hosting providers (see below) are possible.
	snakefile: "benchmarking"
	config: config

use rule * from benchmarking as BM_*


module FidentXaln:
	# here, plain paths, URLs and the special markers for code hosting providers (see below) are possible.
	snakefile: "FidentXaln"
	config: config

use rule * from FidentXaln as FX_*


module tcoffee:
	# here, plain paths, URLs and the special markers for code hosting providers (see below) are possible.
	snakefile: "tcoffee"
	config: config

use rule * from tcoffee as TC_*


module iqtreeXsingle:
	# here, plain paths, URLs and the special markers for code hosting providers (see below) are possible.
	snakefile: "iqtreeXsingle"
	config: config

use rule * from iqtreeXsingle as IX_*

