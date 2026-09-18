#!/bin/bash
#SBATCH --job-name=expD06-ratios
#SBATCH --partition=gpu
#SBATCH --account=lab
#SBATCH --qos=lab-gpu2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --array=0-1%2
#SBATCH --time=02:00:00
#SBATCH --output=/workspace/junmiaoh/experiments/precision-mlps/logs/ratios-%A_%a.log

set -euo pipefail
experiment_root=/workspace/junmiaoh/experiments/precision-mlps
cd "${experiment_root}/code"
export JAX_PLATFORMS=cuda JAX_ENABLE_X64=true XLA_PYTHON_CLIENT_PREALLOCATE=false
export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2
D06_MODULE="${D06_MODULE:-ratio}"
srun --ntasks=1 flock "${experiment_root}/ratios_${SLURM_ARRAY_TASK_ID}.lock" \
    "${experiment_root}/.venv/bin/python" -u -m "experiments.expD06_fixed_center_scales.${D06_MODULE}" \
    --root "${experiment_root}" --seed "${SLURM_ARRAY_TASK_ID}" "$@"
