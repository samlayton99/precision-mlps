#!/bin/bash
#SBATCH --job-name=optimization-exploration
#SBATCH --partition=gpu
#SBATCH --account=lab
#SBATCH --qos=lab-gpu2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --time=00:20:00
#SBATCH --output=/tmp/exploration-%j.log
set -euo pipefail
experiment_root=/workspace/junmiaoh/experiments/precision-mlps
cd "${experiment_root}/exploration-code"
export EXPLORATION_SOURCE_COMMIT="$(cat .source_commit)"
export JAX_PLATFORMS=cuda JAX_ENABLE_X64=true XLA_PYTHON_CLIENT_PREALLOCATE=false
export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MPLCONFIGDIR=/tmp/exploration-mpl
srun --ntasks=1 "${experiment_root}/.venv-race/bin/python" -u \
  -m experiments.expD35_optimization_exploration.run \
  --root "${experiment_root}/runs/optimization_exploration" --require-gpu "$@"
