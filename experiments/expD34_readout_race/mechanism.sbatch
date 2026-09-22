#!/bin/bash
#SBATCH --job-name=d34-mechanism
#SBATCH --partition=gpu
#SBATCH --account=lab
#SBATCH --qos=lab-gpu2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
set -euo pipefail
campaign_root=$1
module=$2
shift 2
cd "${campaign_root}/code"
export RACE_SOURCE_COMMIT="$(cat .source_commit)"
export JAX_PLATFORMS=cuda JAX_ENABLE_X64=true XLA_PYTHON_CLIENT_PREALLOCATE=false
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
srun --ntasks=1 /workspace/junmiaoh/experiments/precision-mlps/.venv-race/bin/python -u \
  -m "experiments.expD34_readout_race.${module}" "$@"
