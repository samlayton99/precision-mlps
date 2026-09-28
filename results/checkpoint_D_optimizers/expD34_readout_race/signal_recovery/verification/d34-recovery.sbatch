#!/bin/bash
#SBATCH --job-name=d34-recovery
#SBATCH --partition=gpu
#SBATCH --account=lab
#SBATCH --qos=lab-gpu2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:25:00
#SBATCH --output=/workspace/junmiaoh/experiments/precision-mlps/recovery-a752d80/output/slurm-%j.out
set -euo pipefail
cd /workspace/junmiaoh/experiments/precision-mlps/recovery-a752d80/code
export JAX_ENABLE_X64=true
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OPENBLAS_NUM_THREADS=2
export OMP_NUM_THREADS=2
export MPLCONFIGDIR=/tmp/d34-recovery-mpl
export RACE_SOURCE_COMMIT=a752d80
srun /workspace/junmiaoh/experiments/precision-mlps/.venv-race/bin/python -u -m experiments.expD34_readout_race.replay_recovery --evidence ../inputs --output ../output
