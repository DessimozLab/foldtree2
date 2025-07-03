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

