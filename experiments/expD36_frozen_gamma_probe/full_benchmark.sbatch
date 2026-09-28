#!/bin/bash
#SBATCH --job-name=gamma-full-benchmark
#SBATCH --partition=gpu
#SBATCH --account=lab
#SBATCH --qos=lab-gpu2
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:10:00
#SBATCH --output=/tmp/gamma-full-benchmark-%j.log
set -euo pipefail
cd "${PROBE_CODE:?}"
export JAX_ENABLE_X64=true XLA_PYTHON_CLIENT_PREALLOCATE=false
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8
export PROBE_SOURCE_COMMIT="$(cat .source_commit)"
probe_python=/workspace/junmiaoh/experiments/precision-mlps/.venv/bin/python
probe_flags=()
if [[ "${PROBE_CPU:-0}" == 1 ]]; then
  export CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu
  probe_flags+=(--cpu)
  srun "$probe_python" -m pytest -q tests/test_expD36_full_sweep.py
else
  export JAX_PLATFORMS=cuda
fi
srun "$probe_python" -u -m experiments.expD36_frozen_gamma_probe.full_benchmark \
  --root "${PROBE_OUTPUT:?}" "${probe_flags[@]}"
