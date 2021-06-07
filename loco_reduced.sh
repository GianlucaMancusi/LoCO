#!/bin/bash
#SBATCH --job-name=loco_reduced
#SBATCH --output=out/out.loco_reduced.txt
#SBATCH --error=out/err.loco_reduced.txt
#SBATCH --nodes=1
#SBATCH --cpus-per-task=9
#SBATCH --gres=gpu:1

source activate loco_env

python main.py --exp_name loco_reduced