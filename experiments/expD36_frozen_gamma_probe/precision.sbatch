#!/bin/bash
#SBATCH --job-name=gamma-probe-precision
#SBATCH --partition=gpu
#SBATCH --account=lab
#SBATCH --qos=lab-gpu2
#SBATCH --ntasks=1
#SBATCH --gres=none
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:40:00
#SBATCH --output=/tmp/gamma-probe-precision-%j.log
set -euo pipefail
cd "${PROBE_CODE:?}"
export CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu JAX_ENABLE_X64=true
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8
export PROBE_SOURCE_COMMIT="$(cat .source_commit)"
probe_python=/workspace/junmiaoh/experiments/precision-mlps/.venv/bin/python
srun "$probe_python" -m pytest -q tests/test_expD36_frozen_gamma_probe.py tests/test_expD36_frozen_training.py tests/test_expD36_precision.py
srun "$probe_python" -u -m experiments.expD36_frozen_gamma_probe.precision --root "${PROBE_OUTPUT:?}" --seconds 2100
