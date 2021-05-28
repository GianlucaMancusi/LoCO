#!/bin/bash
#SBATCH --job-name=loco_norm
#SBATCH --output=out/out.loco_norm.txt
#SBATCH --error=out/err.loco_norm.txt
#SBATCH --nodes=1
#SBATCH --cpus-per-task=9
#SBATCH --gres=gpu:1

source activate loco_env

python main.py --exp_name loco_norm