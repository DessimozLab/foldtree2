#once snakemake is installed use the following command to test the struct tree
import snakemake.utils
snakemake.utils.min_version("7.8.0")
snake_dir = workflow.basedir
rootdir = ''.join([ sub + '/' for sub in snake_dir.split('/')[:-1] ] )




print('ml3ditree running in ' , rootdir)

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
	folders = glob_wildcards("{folders}/finalset.csv").folders
	config['folder']  = folders

if config["iqtree_redo"] == True:
	redo = " -redo"
else:
	redo = ""

alntype = ['3di','AA', ]#'foldtree']

print( 'custom submat' , rootdir+config["mafft_submat3di"])

def get_mem_mb(wildcards, attempt):
	return attempt * 20000

def get_time(wildcards, attempt):
	return min(attempt,3) * 60


rule iqtreex:
	threads: 20 
	resources:
		mem_mb=get_mem_mb,
		time=get_time,
	conda:
		#"config/fold_tree.yaml"
		"foldtree"
	input:
		ancient("{folder}/alnAA_{alntype}.fasta"),
		ancient("{folder}/aln3di_{alntype}.fasta"),
		ancient("{folder}/templateX{alntype}.nx")
	output:
		"{folder}/templateX{alntype}.nx.ckp.gz",
		"{folder}/templateX{alntype}.nx.treefile"	
	params:
		cores=config["iqtree_cores"],
	log:
		"{folder}/logs/3diAA_iqtree_{alntype}.log"
	shell:
		'iqtree -p {wildcards.folder}/templateX{wildcards.alntype}.nx -seed 42 -nt AUTO -n 50 ' + redo


rule iqtree_template:
	conda:
		#"config/fold_tree.yaml"
		"foldtree"
	input:
		ancient("{folder}/alnAA_{alntype}.fasta"),
		ancient("{folder}/aln3di_{alntype}.fasta"),
	output:
		"{folder}/templateX{alntype}.nx",
	params:
		submat=rootdir+config["submat3di"]
	log:
		"{folder}/logs/template_iqtree_{alntype}.log"
	script:
		#todo change this to use the partition file and the model
		'../src/create_iqtree_template.py' 

rule cross_alns:
	#transfer 3di logic to to AA seq	
	#transfer seq aln logic to 3di sequences
	conda:
		#"config/fold_tree.yaml"
		"foldtree"
	input:
		ancient("{folder}/alnAA_AA.fasta"),
		ancient("{folder}/aln3di_3di.fasta"),
	output:
		"{folder}/alnAA_3di.fasta",
		"{folder}/aln3di_AA.fasta",
	log:
		"{folder}/logs/cross_alns.log"
	script:
		"../scripts/crossalns.py"

rule mafft_textaln:
	#use the 3di submat to aln the 3di sequences
	conda:
		#"config/fold_tree.yaml"
		"foldtree"
	input:
		ancient("{folder}/seq3di.fasta"),
	output:
		"{folder}/aln3di_3di.fasta",
	params:
		submat=rootdir+config["mafft_submat3di"]
	shell:
		'mafft --textmatrix {params.submat} {input} > {output}'

rule mafft_seq:
	#use a normal sequence alignment tool to align the AA sequences
	conda:
		#"config/fold_tree.yaml"
		"foldtree"
	input:
		ancient("{folder}/sequences.fst"),
	output:
		"{folder}/alnAA_AA.fasta",
	shell:
		'mafft {input} > {output}'

rule scrape_alns:
	conda:
		#"config/fold_tree.yaml"
		"foldtree"
	input:
		ancient("{folder}/allvall_1.csv"),
		ancient("{folder}/outdb"),
		ancient("{folder}/outdb_ss")
	output:
		"{folder}/sequences.fst",
		"{folder}/seq3di.fasta",
	params:
		dbname="outdb",
		submat=rootdir+config["mafft_submat3di"]
	log:
		"{folder}/logs/scrape_alns.log"
	script:
		"../src/add_structalns.py"

rule out_xalndf:
	#use 3di and AA from foldseek
	#scrape the 3di alphabet and use alignment logic to calculate new fident
	#working on merging the alns progressively for a blend of struct and seq
	conda:
		#"config/fold_tree.yaml"
		"foldtree"
	input:
		ancient("{folder}/allvall_1.csv"),
		ancient("{folder}/outdb"),
		ancient("{folder}/outdb_ss")
	output:
		"{folder}/allvall_1_xaln.csv",
	params:
		dbname="outdb",
		submat=rootdir+config["mafft_submat3di"]
	log:
		"{folder}/logs/scrape_alns.log"
	script:
		"../src/output_xalndf.py"

rule foldseek_createdb:
	conda:
		#"config/fold_tree.yaml"
		"foldtree"
	input:
		ancient("{folder}/finalset.csv")
	output:
		"{folder}/outdb",
		"{folder}/outdb_ss"
	log:
		"{folder}/logs/foldseekallvall.log"
	shell:
		foldseekpath + " createdb {wildcards.folder}/structs/ {wildcards.folder}/outdb"