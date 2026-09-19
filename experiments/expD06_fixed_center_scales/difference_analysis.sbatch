#!/bin/bash
#SBATCH --job-name=expD06-difference-analysis
#SBATCH --partition=gpu
#SBATCH --account=lab
#SBATCH --qos=lab-gpu2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=none
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/workspace/junmiaoh/experiments/precision-mlps/logs/difference-analysis-%j.log

set -euo pipefail
experiment_root=/workspace/junmiaoh/experiments/precision-mlps
cd "${experiment_root}/code"
export CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu JAX_ENABLE_X64=true
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MPLCONFIGDIR=/tmp/d06-difference-matplotlib
srun --ntasks=1 "${experiment_root}/.venv/bin/python" -u -m experiments.expD06_fixed_center_scales.difference_analysis \
    --root "${experiment_root}/runs/conditioning" --output "${experiment_root}/runs/conditioning_analysis" "$@"
