#!/bin/bash
#SBATCH --job-name=fix_metrics
#SBATCH --output=out/out.fix_metrics.txt
#SBATCH --error=out/err.fix_metrics.txt
#SBATCH --nodes=1
#SBATCH --cpus-per-task=9
#SBATCH --gres=gpu:1
#SBATCH --requeue                   ### On failure, requeue for another try


source activate loco_env

python main.py --exp_name fix_metrics