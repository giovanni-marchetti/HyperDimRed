#!/usr/bin/env bash
#SBATCH --output=/Midgard/home/%u/phd/corrupted_channels/SimCLRv2/scripts/runs/%J_slurm.out
#SBATCH --error=/Midgard/home/%u/phd/corrupted_channels/SimCLRv2/scripts/runs/%J_slurm.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=fatn@kth.se
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=9
#SBATCH --mem=50GB
#SBATCH --constrain="belegost|smaug|khazadum"
#SBATCH --array=0-3

# Check job environment
echo "JOB:  ${SLURM_JOB_ID}"
echo "TASK: ${SLURM_ARRAY_TASK_ID}"
echo "HOST: $(hostname)"
echo ""

# Activate conda
source "${HOME}/miniconda3/etc/profile.d/conda.sh"
conda activate HyperDimRed


project_name='hyperbolic_smell'
sweep_id='vvxmax2q'
entity_name='farzaneh-kth'
wandb login "128ba237ed5b081911f0564e6c8d34eb864f78cf"
wandb agent $sweep_id --project $project_name --entity $entity_name