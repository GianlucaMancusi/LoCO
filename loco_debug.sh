#!/bin/bash
#SBATCH --job-name=loco_debug
#SBATCH --output=out/out.loco_debug.txt
#SBATCH --error=out/err.loco_debug.txt
#SBATCH --nodes=1
#SBATCH --cpus-per-task=9
#SBATCH --gres=gpu:1
#SBATCH --time 24:00:00


source activate loco_env

python main.py --exp_name loco_debug