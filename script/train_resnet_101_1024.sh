#!/bin/bash

#Submit this script with: sbatch thefilename

#SBATCH -p short
#SBATCH --time=06:00:00   # walltime
#SBATCH --ntasks=16   # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --gres=gpu:4
#SBATCH -J "test-resnet101-1024"   # job name
#SBATCH --mail-user=felix.wirth@epita.fr   # email address
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END


# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE


echo Job starting
cd data
python3.8 train_script_resnet101_1024.py
