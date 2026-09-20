#!/bin/bash
#SBATCH --job-name=expD06-stall-analysis
#SBATCH --partition=gpu
#SBATCH --account=lab
#SBATCH --qos=lab-gpu2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --gres=none
#SBATCH --time=01:00:00
#SBATCH --output=/workspace/junmiaoh/experiments/precision-mlps/logs/stall-analysis-%j.log

set -euo pipefail
: "${SLURM_JOB_ID:?Submit with sbatch}"
experiment_root=/workspace/junmiaoh/experiments/precision-mlps
cd "${experiment_root}/code"
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=true CUDA_VISIBLE_DEVICES=''
export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2
srun --ntasks=1 --gres=none "${experiment_root}/.venv/bin/python" -u \
    -m experiments.expD06_fixed_center_scales.stall_analysis --root "${experiment_root}" "$@"
