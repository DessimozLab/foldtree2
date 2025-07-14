import glob
import os

# Get root directory from config or default to current directory
ROOT_DIR = config.get("root_dir", ".")

# Function to find all sequences.fst files and extract folder names
def get_folders_with_sequences(root_dir="."):
    """Find all directories containing sequences.fst files"""
    folders = []
    search_pattern = os.path.join(root_dir, "**/sequences.fst")
    for seq_file in glob.glob(search_pattern, recursive=True):
        folder = os.path.dirname(seq_file)
        if folder and folder != root_dir:  # Skip if sequences.fst is in root directory
            folders.append(folder)
    return folders

# Get all folders containing sequences.fst files
FOLDERS = get_folders_with_sequences(ROOT_DIR)

rule all:
    input:
        expand("{folder}/alnAA_AA.raxml.bestTree", folder=FOLDERS)

rule mafft_aa:
    conda:
        "foldtree2"
    input:
        fasta="{folder}/sequences.fst"
    output:
        aln="{folder}/alnAA_AA.fasta"
    shell:
        "mafft {input.fasta} > {output.aln}"

rule raxmlng_aa:
    conda:
        "foldtree2"
    input:
        aln="{folder}/alnAA_AA.fasta"
    output:
        tree="{folder}/alnAA_AA.raxml.bestTree"
    params:
        raxmlng="./raxml-ng",
        model="GTR+I+G"
    shell:
        "{params.raxmlng} --msa {input.aln} --model PROTGTR+I+G --prefix {wildcards.folder}/alnAA_AA.raxml --threads 8 --seed 12345 --redo"
