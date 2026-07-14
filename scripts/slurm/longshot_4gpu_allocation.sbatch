#!/bin/bash
#SBATCH --job-name=gcw-longshot-4gpu
#SBATCH --partition=agent-xlong-15,agent-xlong
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=500G
#SBATCH --output=slurm_out/longshot-4gpu-%j.out

set -euo pipefail

echo "4-GPU allocation ready on $(hostname)"
echo "job_id=${SLURM_JOB_ID}"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"

while true; do
  sleep 3600
done
