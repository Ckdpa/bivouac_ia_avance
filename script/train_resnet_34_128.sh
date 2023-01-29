#!/bin/bash

#Submit this script with: sbatch thefilename

#SBATCH -p short
#SBATCH --time=06:00:00   # walltime
#SBATCH --ntasks=4  # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --gres=gpu:4
#SBATCH -J "test-resnet34-128"   # job name
#SBATCH --mail-user=antoine.vergnaud@epita.fr   # email address
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END


# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE


echo Job starting
python3.8 train_script_resnet34_128.py
