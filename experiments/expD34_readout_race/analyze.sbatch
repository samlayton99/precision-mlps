#!/bin/bash
#SBATCH --job-name=race-analysis
#SBATCH --partition=gpu
#SBATCH --account=lab
#SBATCH --qos=lab-gpu2
#SBATCH --gres=none
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/workspace/junmiaoh/experiments/precision-mlps/logs/race-analysis-%j.log
set -euo pipefail
experiment_root=/workspace/junmiaoh/experiments/precision-mlps
cd "${experiment_root}/race-code"
export CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu JAX_ENABLE_X64=true
export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MPLCONFIGDIR=/tmp/race-mpl
srun --ntasks=1 "${experiment_root}/.venv-race/bin/python" -u \
  -m experiments.expD34_readout_race.analyze --root "${experiment_root}/runs/readout_race" "$@"
