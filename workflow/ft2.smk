import glob
import os

# Get root directory and all relevant options from config
ROOT_DIR = config.get("root_dir", ".")
ANCESTRAL = config.get("ancestral", False)
MODEL = config.get("model", None)
MAFFTMAT = config.get("mafftmat", None)
SUBMAT = config.get("submat", None)
OUTDIR = config.get("outdir", None)
RAXML_ITER = config.get("raxml_iterations", None)
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
        expand("{folder}/ft2treebuilder.done", folder=FOLDERS)

rule ft2treebuilder:
    conda:
        "foldtree2"
    input:
        structs_dir="{folder}/structs"
    output:
        done="{folder}/ft2treebuilder.done"
    params:
        ancestral=ANCESTRAL,
        model=MODEL,
        mafftmat=MAFFTMAT,
        submat=SUBMAT,
        outdir=OUTDIR,
        raxml_iterations=RAXML_ITER,
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
            ("--ancestral " if params.ancestral else "") +
            (f"--raxml_iterations {params.raxml_iterations} " if params.raxml_iterations else "") +
            (f"--n_state {params.n_state} " if params.n_state else "") +
            (f"--raxmlpath {params.raxmlpath} " if params.raxmlpath else "") +
            "> {output.done}"
        )
