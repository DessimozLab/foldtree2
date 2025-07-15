import glob
import os

from snakemake.utils import min_version
min_version("6.0")

configfile: "./config/config_vars.yaml"


# Get root directory and all relevant options from config
ROOT_DIR = config.get("root_dir", ".")
ANCESTRAL = config.get("ancestral", False)
MODEL = config.get("model", None)
MAFFTMAT = config.get("mafftmat", None)
SUBMAT = config.get("submat", None)
OUTDIR = config.get("outdir", None)
OUTPUT_PREFIX = config.get("output_prefix", "encoded")
AAPROPCSV = config.get("aapropcsv", None)
MAFFTTEXT2HEX = config.get("maffttext2hex", "/usr/local/libexec/mafft/maffttext2hex")
NCORES = config.get("ncores", None)
RAXML_ITER = config.get("raxml_iterations", None)
VERBOSE = config.get("verbose", False)
N_STATE = config.get("n_state", None)
RAXMLPATH = config.get("raxmlpath", None)

# Find all folders containing a 'structs' subfolder
def get_struct_folders(root_dir):
    folders = []
    search_pattern = os.path.join(root_dir, "**/structs")
    for structs_dir in glob.glob(search_pattern, recursive=True):
        folder = os.path.dirname(structs_dir)
        if folder and folder != root_dir:
            folders.append(folder)
    return folders

FOLDERS = get_struct_folders(ROOT_DIR)


rule all:
    input:
        expand("{folder}/ft2_encoded_replaced.ASCIIaln.txt.raxml.bestTree.rooted", folder=FOLDERS)

rule ft2treebuilder:
    conda:
        "foldtree2"
    input:
        structs_dir="{folder}/structs"
    output:
        done="{folder}/ft2treebuilder.done"
        tree="{folder}/ft2_encoded_replaced.ASCIIaln.txt.raxml.bestTree"
        aln="{folder}/ft2_encoded_replaced.ASCIIaln.txt.raxml_aln.fasta.raxml_aln.fasta"
    params:
        ancestral=ANCESTRAL,
        model=MODEL,
        mafftmat=MAFFTMAT,
        submat=SUBMAT,
        outdir=OUTDIR,
        output_prefix=OUTPUT_PREFIX,
        aapropcsv=AAPROPCSV,
        maffttext2hex=MAFFTTEXT2HEX,
        ncores=NCORES,
        raxml_iterations=RAXML_ITER,
        verbose=VERBOSE,
        n_state=N_STATE,
        raxmlpath=RAXMLPATH
    shell:
        (
            "python ft2treebuilder.py " +
            (f"--model {params.model} " if params.model else "") +
            (f"--mafftmat {params.mafftmat} " if params.mafftmat else "") +
            (f"--submat {params.submat} " if params.submat else "") +
            (f"--structures {input.structs_dir} " if input.structs_dir else "") +
            (f"--outdir {params.outdir} " if params.outdir else "") +
            (f"--output_prefix {params.output_prefix} " if params.output_prefix else "") +
            (f"--aapropcsv {params.aapropcsv} " if params.aapropcsv else "") +
            (f"--maffttext2hex {params.maffttext2hex} " if params.maffttext2hex else "") +
            (f"--ncores {params.ncores} " if params.ncores else "") +
            ("--ancestral " if params.ancestral else "") +
            (f"--raxml_iterations {params.raxml_iterations} " if params.raxml_iterations else "") +
            ("--verbose " if params.verbose else "") +
            (f"--n_state {params.n_state} " if params.n_state else "") +
            (f"--raxmlpath {params.raxmlpath} " if params.raxmlpath else "") +
            "> {output.done}"
        )

rule madroot:
    input:
        tree="{folder}/ft2_encoded_replaced.ASCIIaln.txt.raxml.bestTree"
    output:
        rooted="{folder}/ft2_encoded_replaced.ASCIIaln.txt.raxml.bestTree.rooted"
    shell:
        "{config.madroot} {input.tree}"