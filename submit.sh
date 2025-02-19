#!/bin/sh
#SBATCH --account=def-einack-ab
#SBATCH --nodes=1
#SBATCH --gres=gpu:v100:8 ####t4:2 
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --time=0-00:30            # time (DD-HH:MM)
#SBATCH --job-name=v100test
#SBATCH --array=1-1
#SBATCH --output=t4_gp4_logs/%x-%j-%N.out
source vna-torch/bin/activate
# Set a unique MASTER_PORT using SLURM_ARRAY_TASK_ID
export MASTER_PORT=$((12355 + SLURM_ARRAY_TASK_ID))
python src/main_DDP.py  $SLURM_ARRAY_TASK_ID

#export MASTER_PORT=$((12355 + 0))
