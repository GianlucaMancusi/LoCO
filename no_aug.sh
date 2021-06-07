#!/bin/bash
#SBATCH --job-name=no_aug
#SBATCH --output=out/out.no_aug.txt
#SBATCH --error=out/err.no_aug.txt
#SBATCH --nodes=1
#SBATCH --cpus-per-task=9
#SBATCH --gres=gpu:1
#SBATCH --requeue                   ### On failure, requeue for another try


source activate loco_env

python main.py --exp_name no_aug