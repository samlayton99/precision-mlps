#!/usr/bin/env bash
#SBATCH --partition=gpu
#SBATCH --account=lab
#SBATCH --qos=lab-gpu2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=00:55:00
#SBATCH --job-name=gamma-full

set -euo pipefail
: "${PROBE_CODE:?}" "${PROBE_OUTPUT:?}" "${PROBE_STAGE:?}"
cd "$PROBE_CODE"
export PROBE_SOURCE_COMMIT
PROBE_SOURCE_COMMIT=$(cat source_commit.txt)
export JAX_ENABLE_X64=true OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export MPLCONFIGDIR="/tmp/gamma-full-mpl-${SLURM_JOB_ID}"
PYTHON=/workspace/junmiaoh/experiments/precision-mlps/.venv/bin/python
if [[ "$PROBE_STAGE" == screen ]]; then
    export CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu
    srun "$PYTHON" -m experiments.expD36_frozen_gamma_probe.full_screen --root "$PROBE_OUTPUT" --seconds 850
else
    export JAX_PLATFORMS=cuda
    srun "$PYTHON" -m experiments.expD36_frozen_gamma_probe.full_worker --root "$PROBE_OUTPUT" --worker "$PROBE_WORKER" --seconds 3200
fi
