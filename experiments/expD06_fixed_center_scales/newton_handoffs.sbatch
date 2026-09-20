#!/bin/bash
#SBATCH --job-name=expD06-newton-handoffs
#SBATCH --partition=gpu
#SBATCH --account=lab
#SBATCH --qos=lab-gpu2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --array=0-1%2
#SBATCH --time=00:30:00
#SBATCH --output=/workspace/junmiaoh/experiments/precision-mlps/logs/newton-handoffs-%A_%a.log
set -euo pipefail
experiment_root=/workspace/junmiaoh/experiments/precision-mlps
cd "${experiment_root}/code"
export JAX_PLATFORMS=cuda JAX_ENABLE_X64=true XLA_PYTHON_CLIENT_PREALLOCATE=false
export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2
srun --ntasks=1 flock "${experiment_root}/newton_handoffs_${SLURM_ARRAY_TASK_ID}.lock" \
    "${experiment_root}/.venv-curvature/bin/python" -u -m experiments.expD06_fixed_center_scales.newton_handoffs \
    --root "${experiment_root}/runs/newton_handoffs" --worker "${SLURM_ARRAY_TASK_ID}" --require-gpu \
    --ssbroyden-source "${experiment_root}/dependencies/ssbroyden-4c87785" "$@"
