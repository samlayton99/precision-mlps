#!/usr/bin/env bash
#SBATCH --partition=gpu
#SBATCH --account=lab
#SBATCH --qos=lab-gpu2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=none
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:20:00
#SBATCH --job-name=gamma-audit

set -euo pipefail
: "${PROBE_CODE:?}" "${PROBE_OUTPUT:?}" "${PROBE_STAGE:?}"
cd "$PROBE_CODE"
export PROBE_SOURCE_COMMIT
PROBE_SOURCE_COMMIT=$(cat source_commit.txt)
export CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu JAX_ENABLE_X64=true
export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2
PYTHON=/workspace/junmiaoh/experiments/precision-mlps/.venv/bin/python
if [[ "$PROBE_STAGE" == precision ]]; then
    srun "$PYTHON" -m experiments.expD36_frozen_gamma_probe.full_precision --root "$PROBE_OUTPUT" --seconds 650 --workers 4
    srun "$PYTHON" -m experiments.expD36_frozen_gamma_probe.full_reference --root "$PROBE_OUTPUT" --seconds 450
else
    export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8
    srun "$PYTHON" -m experiments.expD36_frozen_gamma_probe.full_diagnostics --root "$PROBE_OUTPUT" --seconds 850
fi
