#!/bin/bash
#SBATCH --job-name=gamma-probe-train
#SBATCH --partition=gpu
#SBATCH --account=lab
#SBATCH --qos=lab-gpu2
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:55:00
#SBATCH --output=/tmp/gamma-probe-train-%j.log
set -euo pipefail
cd "${PROBE_CODE:?}"
export JAX_PLATFORMS=cuda JAX_ENABLE_X64=true XLA_PYTHON_CLIENT_PREALLOCATE=false
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4
export PROBE_SOURCE_COMMIT="$(cat .source_commit)"
probe_python=/workspace/junmiaoh/experiments/precision-mlps/.venv/bin/python
srun "$probe_python" -u -m experiments.expD36_frozen_gamma_probe.train \
  --root "${PROBE_OUTPUT:?}" --map "${PROBE_MAP:?}" --require-gpu --seconds 3100
