#!/bin/bash
#SBATCH --job-name=gamma-probe-screen
#SBATCH --partition=gpu
#SBATCH --account=lab
#SBATCH --qos=lab-gpu2
#SBATCH --ntasks=1
#SBATCH --gres=none
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:20:00
#SBATCH --output=/tmp/gamma-probe-screen-%j.log
set -euo pipefail
cd "${PROBE_CODE:?Set PROBE_CODE to the immutable source snapshot}"
export CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu JAX_ENABLE_X64=true
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8
export PROBE_SOURCE_COMMIT="$(cat .source_commit)"
probe_python=/workspace/junmiaoh/experiments/precision-mlps/.venv/bin/python
srun "$probe_python" -m pytest -q tests/test_expD36_frozen_gamma_probe.py
srun "$probe_python" -u -m experiments.expD36_frozen_gamma_probe.screen --root "${PROBE_OUTPUT:?}"
