#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --account=lab
#SBATCH --qos=lab-gpu2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=00:30:00
set -euo pipefail
cd "${PROBE_CODE:?}"
export JAX_ENABLE_X64=true XLA_PYTHON_CLIENT_PREALLOCATE=false
export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2
export PROBE_SOURCE_COMMIT="$(cat .source_commit)"
probe_python=/workspace/junmiaoh/experiments/precision-mlps/.venv/bin/python
if [[ "${PROBE_STAGE:?}" == train ]]; then
  export JAX_PLATFORMS=cuda
else
  export CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu
fi
srun "$probe_python" -u -m experiments.expD36_frozen_gamma_probe.cap_campaign \
  "$PROBE_STAGE" --root "${PROBE_OUTPUT:?}" "${@}"
