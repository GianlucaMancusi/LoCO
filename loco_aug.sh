#!/bin/bash
#SBATCH --job-name=loco_aug
#SBATCH --output=out/out.loco_aug.txt
#SBATCH --error=out/err.loco_aug.txt
#SBATCH --nodes=1
#SBATCH --cpus-per-task=9
#SBATCH --gres=gpu:1
#SBATCH --time 8:00:00



source activate loco_env

python main.py --exp_name loco_aug