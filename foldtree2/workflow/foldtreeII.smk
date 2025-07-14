
# Snakemake workflow for alignment and treebuilding using ft2treebuilder.py and command-line tools

import snakemake.utils
snakemake.utils.min_version("7.8.0")
snake_dir = workflow.basedir
rootdir = ''.join([ sub + '/' for sub in snake_dir.split('/')[:-1] ] )
print(' benchmark all running in ' , rootdir)
configfile: rootdir+ "workflow/config/config_vars.yaml"
# remote homologues search parameters



if 'folder' in config:
    if type(config['folder']) == str:
        folders = [config['folder']]
    else:
        folders = config['folder']
else:
    folders = glob_wildcards("{folders}/structs/").folders
    config['folder']  = folders

if 'model' in config:
    print('running ft2 snakemake with ' , model)
else:
    raise 'you must specify a model ' 

rule all:
    input:
        expand( "{folder}/{model}_FT2_tree.nwk" , folder=folders ) 

rule encode_structures:
    conda : 
        "foldtree2"
    input:
        pdbs=expand("{indir}/*.pdb", indir="{indir}")
    output:
        fasta="results/encoded.fasta"
    params:
        model="models/model",
        script="../scripts/encode_structures_to_fasta.py"
    shell:
        "python {params.script} {params.model} '{wildcards.indir}/*.pdb' --outfile {output.fasta}"

rule replace_special_chars:
    conda :
        "foldtree2"
    input:
        fasta="{folder}/encoded.fasta"
    output:
        replaced_fasta="results/encoded_replaced.fasta"
    script:
        "../scripts/replace_sp_chars.py"

rule encodedfasta2hex:
    conda :
        "foldtree2"
    input:
        fasta="{folder}/encoded_replaced.fasta"
    output:
        hex="{folder}/encoded.hex"
    shell:
        "python ../scripts/encodedfasta2hex.py {input.fasta} {output.hex}"

rule mafft_hex2fasta:
    conda :
        "foldtree2"
    input:
        hex="{folder}/encoded.hex"
    output:
        ascii="{folder}/encoded.ASCII"
    params:
        hex2maffttext = config["hex2maffttext"]
    shell:
        "{params.hex2maffttext} {input.hex} > {output.ascii}"

rule run_mafft_textaln:
    conda :
        "foldtree2"
    input:
        ascii="results/encoded.ASCII"
    output:
        aln="results/aln.txt"
    params:
        matrix="models/mafft_submat.mtx"
    shell:
        "mafft --text --localpair --maxiterate 1000 --textmatrix {params.matrix} {input.ascii} > {output.aln}"

rule fasta2hex:
    conda :
        "foldtree2"
    input:
        aln="results/aln.txt"
    output:
        aln_hex="results/aln.hex"
    params:
        maffttext2hex = config["maffttext2hex"]
    shell:
        "{params.maffttext2hex} {input.aln} > {output.aln_hex}"

rule read_textaln:
    conda :
        "foldtree2"
    input:
        aln_hex="results/aln.hex"
    output:
        raxml_aln_fasta="results/aln.raxml_aln.fasta"
    script:
        "../scripts/read_textaln.py"

rule run_raxml_ng:
    conda :
        "foldtree2"
    input:
        fasta="results/aln.raxml_aln.fasta"
    output:
        tree="results/tree.nwk"
    params:
        matrix="models/submat.txt",
        nsymbols=20,
        raxmlng="./raxml-ng"
    shell:
        "{params.raxmlng} --model MULTI{params.nsymbols}_GTR{{{params.matrix}}}+I+G --redo --all --bs-trees 20 --seed 12345 --threads 8 --msa {input.fasta} --prefix results/tree && cp results/tree.raxml.bestTree {output.tree}"

rule ancestral_reconstruction:
    conda :
        "foldtree2"
    input:
        model="models/model",
        mafftmat="models/mafft_submat.mtx",
        submat="models/submat.txt",
        structures="{indir}/*.pdb"
    output:
        tree="{folder}/{model}_FT2_tree.nwk",
        ancestral_csv="{folder}/{model}_ancestral.csv",
        ancestral_fasta="{folder}/{model}_ancestral.aastr.fasta"
    params:
        script="../ft2treebuilder.py"
    shell:
        "python {params.script} --model {input.model} --mafftmat {input.mafftmat} --submat {input.submat} --structures '{input.structures}' --outdir {wildcards.folder} --ancestral"