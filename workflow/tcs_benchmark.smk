import snakemake.utils
snakemake.utils.min_version("7.8.0")

snake_dir = workflow.basedir

rootdir = ''.join([ sub + '/' for sub in snake_dir.split('/')[:-1] ] )

print(' benchmark tcs running in ' , rootdir)

configfile: rootdir+ "workflow/config/config_vars.yaml"

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


def get_mem_mb(wildcards, attempt):
	return attempt * 20000

rule calc_tax_score:
	conda:
		"foldtree2"
	input:
		"{folder}/sequence_dataset.csv",
		"{folder}/{treetype}.nwk.rooted"
	output:
		"{folder}/{mattype}_{alntype}_{exp}_treescores_struct_tree.json"
	log:
		"{folder}/logs/{mattype}_{alntype}_{exp}_struct_tree_scoring.log"
	script:
		"../src/calctreescores.py"


rule calc_tax_score_iq:
	conda:
		"foldtree2"
	input:
		"{folder}/sequence_dataset.csv",
		"{folder}/sequences.aln.{aligner}.fst.treefile.rooted"
	output:
		"{folder}/treescores_sequences_iq.{aligner}.json"
	log:
		"{folder}/logs/iq_scoring.{aligner}.log"
	script:
		"../src/calctreescores.py"

rule mad_root_struct:
	conda:
		"foldtree2"
	input:
		"{folder}/{mattype}_{alntype}_{exp}_struct_tree.PP.nwk"
	output:
		"{folder}/{mattype}_{alntype}_{exp}_struct_tree.PP.nwk.rooted"
	log:
		"{folder}/logs/{mattype}_{alntype}_{exp}_struct_madroot.log"
	shell:
		rootdir+'madroot/mad {wildcards.folder}/{wildcards.mattype}_{wildcards.alntype}_{wildcards.exp}_struct_tree.PP.nwk'

rule dl_ids_structs:
	input:
		"{folder}/sequence_dataset.csv",
	output:
		#"{folder}/sequences.fst",
		"{folder}/finalset.csv",
	conda: 
		#"config/fold_tree.yaml"
		"foldtree",
	log:
		"{folder}/logs/dlstructs.log",
	params:
		filtervar=config["filter"],
		cath=config["cath"],
		filtervar_min=config["filter_min"],
		filtervar_avg=config["filter_avg"],
		custom_structs=config["custom_structs"],
		clean_folder=config["clean_folder"],
	script:
		"../src/dl_structs.py"