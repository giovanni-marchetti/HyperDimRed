#!/usr/bin/env bash
#SBATCH --output=/Midgard/home/%u/phd/HyperDimRed/runs/%J_slurm.out
#SBATCH --error=/Midgard/home/%u/phd/HyperDimRed/runs/%J_slurm.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=fatn@kth.se
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=9
#SBATCH --mem=10GB
#SBATCH --constrain="arwen"

# Check job environment
echo "JOB:  ${SLURM_JOB_ID}"
echo "TASK: ${SLURM_ARRAY_TASK_ID}"
echo "HOST: $(hostname)"
echo ""

# Activate conda
source "${HOME}/miniconda3/etc/profile.d/conda.sh"
conda activate HyperDimRed1


config_yaml='/Midgard/home/farzantn/phd/HyperDimRed/configs/params.yaml'
train_file='/Midgard/home/farzantn/phd/HyperDimRed/train_wandb_remote.py'
project_name='hyperbolic_smell'

python -m wandb_on_slurm $config_yaml $train_file $project_name