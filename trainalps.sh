#!/bin/bash
#SBATCH --job-name=ft2-Train
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --environment=gemma-pytorch
#SBATCH --account=A-prep01

source ${VENV_PATH}/bin/activate

python /capstor/store/cscs/swissai/prep01/foldtree2/foldtree2/python scaling_experiment.py --dataset /capstor/scratch/cscs/dmoi/structs_trainingalpstest.h5 --hidden-size 500 --dataset-fractions 0.5 1.0 --epochs 30  --learning-rate 0.0001 --batch-size 16 