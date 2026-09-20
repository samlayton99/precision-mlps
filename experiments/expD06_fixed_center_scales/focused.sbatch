#!/bin/bash
#SBATCH --job-name=expD06-focused
#SBATCH --partition=gpu
#SBATCH --account=lab
#SBATCH --qos=lab-gpu2
#SBATCH --array=0-1%2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=/workspace/junmiaoh/experiments/precision-mlps/logs/focused-%A_%a.log

set -euo pipefail
: "${SLURM_JOB_ID:?Submit with sbatch}"
experiment_root=/workspace/junmiaoh/experiments/precision-mlps
cd "${experiment_root}/code"
export JAX_PLATFORMS=cuda JAX_ENABLE_X64=true XLA_PYTHON_CLIENT_PREALLOCATE=false
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
srun --ntasks=1 --gres=gpu:1 \
    flock --nonblock "${experiment_root}/focused_${SLURM_ARRAY_TASK_ID}.lock" \
    "${experiment_root}/.venv/bin/python" -u -m experiments.expD06_fixed_center_scales.focused \
    --root "${experiment_root}" --shard "${SLURM_ARRAY_TASK_ID}"
