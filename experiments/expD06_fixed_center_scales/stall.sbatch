#!/bin/bash
#SBATCH --job-name=expD06-stall
#SBATCH --partition=gpu
#SBATCH --account=lab
#SBATCH --qos=lab-gpu2
#SBATCH --array=0-1%2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --time=02:05:00
#SBATCH --output=/workspace/junmiaoh/experiments/precision-mlps/logs/stall-%A_%a.log

set -euo pipefail
: "${SLURM_JOB_ID:?Submit with sbatch}"
experiment_root=/workspace/junmiaoh/experiments/precision-mlps
cd "${experiment_root}/code"
export JAX_PLATFORMS=cuda JAX_ENABLE_X64=true XLA_PYTHON_CLIENT_PREALLOCATE=false
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
srun --ntasks=1 --gres=gpu:1 \
    flock --nonblock "${experiment_root}/stall_${SLURM_ARRAY_TASK_ID}.lock" \
    "${experiment_root}/.venv/bin/python" -u -m experiments.expD06_fixed_center_scales.continue_stall \
    --root "${experiment_root}" --seed "${SLURM_ARRAY_TASK_ID}"
