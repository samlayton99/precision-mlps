#!/bin/bash
#SBATCH --job-name=expD06-pilot
#SBATCH --partition=gpu
#SBATCH --account=lab
#SBATCH --qos=lab-gpu2
#SBATCH --array=0-1%2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=02:15:00
#SBATCH --output=/workspace/junmiaoh/experiments/precision-mlps/logs/slurm-%A_%a.log

set -euo pipefail
: "${SLURM_JOB_ID:?Submit this file with sbatch}"
case "${SLURM_ARRAY_TASK_ID}" in
    0) optimizer=gd ;;
    1) optimizer=adam ;;
    *) exit 2 ;;
esac

experiment_root=/workspace/junmiaoh/experiments/precision-mlps
cd "${experiment_root}/code"
export JAX_PLATFORMS=cuda
export JAX_ENABLE_X64=true
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS=1
# Slurm sets CUDA_VISIBLE_DEVICES. Do not override its device assignment.
# The 3-hour cumulative worker cap includes the approximately 1 hour already
# charged to each optimizer; this allocation has 15 minutes of extra margin.
srun --ntasks=1 --gres=gpu:1 \
    flock --nonblock "${experiment_root}/worker_${optimizer}.lock" \
    "${experiment_root}/.venv/bin/python" -u \
    -m experiments.expD06_fixed_center_scales.campaign \
    --root "${experiment_root}" --optimizer "${optimizer}" \
    --boundary --gpu-hours 3 --batch-size 4
