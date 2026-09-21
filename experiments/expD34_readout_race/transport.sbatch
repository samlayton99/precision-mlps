#!/bin/bash
#SBATCH --job-name=d34-transport
#SBATCH --partition=gpu
#SBATCH --account=lab
#SBATCH --qos=lab-gpu2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:15:00
set -euo pipefail
# Submit with --array=...%1 and an isolated committed checkout as argument 1.
campaign_root=$1
shift
cd "${campaign_root}/code"
export RACE_SOURCE_COMMIT="$(cat .source_commit)"
export JAX_PLATFORMS=cuda JAX_ENABLE_X64=true XLA_PYTHON_CLIENT_PREALLOCATE=false
export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MPLCONFIGDIR=/tmp/d34-transport-mpl
srun --ntasks=1 /workspace/junmiaoh/experiments/precision-mlps/.venv-race/bin/python -u \
  -m experiments.expD34_readout_race.transport_run --max-seconds 810 "$@"
