#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --array=1-424
#SBATCH -c 24
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --gres-flags=enforce-binding
#SBATCH --mem 200G
#SBATCH -t 6:00:00
#SBATCH --output=array_%A-%a.out 
#SBATCH --output=array_%A-%a.out    # Standard output and error log
# Set the path to download dir

INPUT_DIR=.    # Set the appropriate path to your supporting data
OUTDIR=./models

file=$(ls ${INPUT_DIR}*.fasta | sed -n ${SLURM_ARRAY_TASK_ID}p)
echo $file

module purge
module load singularity

export SINGULARITY_BINDPATH="/scratch,/dcsrsoft,/users,/work,/reference"
#singularity run --nv /dcsrsoft/singularity/containers/alphafold-v2.1.1.sif 
python3 /dcsrsoft/singularity/containers/run_alphafold_2.2.0.py --data-dir /reference/alphafold/20220414 --cpus 24 --use-gpu --fasta-paths ${file} --output-dir ${OUTDIR}

echo 'done'



