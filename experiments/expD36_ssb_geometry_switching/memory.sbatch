#!/bin/bash
#SBATCH --job-name=ssb-memory
#SBATCH --partition=gpu
#SBATCH --account=lab
#SBATCH --qos=lab-gpu2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --time=00:20:00
#SBATCH --output=/tmp/ssb-memory-%j.log
set -euo pipefail
experiment_root=/workspace/junmiaoh/experiments/precision-mlps
cd "${experiment_root}/ssb-geometry-code"
export JAX_PLATFORMS=cuda JAX_ENABLE_X64=true XLA_PYTHON_CLIENT_PREALLOCATE=false
export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2
srun --ntasks=1 "${experiment_root}/.venv-curvature/bin/python" -u \
  -m experiments.expD36_ssb_geometry_switching.memory \
  --root "${experiment_root}/runs/ssb_geometry_switching" \
  --source "${experiment_root}/dependencies/ssbroyden-4c87785" "$@"
