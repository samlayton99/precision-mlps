#!/usr/bin/env bash
# Submit inside a CPU-only Slurm allocation; helpers must already be uploaded.
set -euo pipefail
: "${SLURM_JOB_ID:?Submit this script through Slurm}"
export CUDA_VISIBLE_DEVICES=""
export JAX_PLATFORMS=cpu
export JAX_ENABLE_X64=true
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
D34_CAMPAIGN=/workspace/junmiaoh/experiments/precision-mlps/mechanism-0365cd3-20260923
D34_PYTHON=${D34_PYTHON:-/workspace/junmiaoh/experiments/precision-mlps/.venv-feedback/bin/python}
D34_EVIDENCE="$D34_CAMPAIGN/evidence/dilation_b52df33"
cd "$D34_CAMPAIGN/code"
mkdir -p "$D34_EVIDENCE"
# Run the focused tests separately before submitting this preparation job.
srun "$D34_PYTHON" -m experiments.expD34_readout_race.mechanism_dilation_repair \
    --input "$D34_CAMPAIGN/evidence/inputs/development.npz" \
    --output "$D34_EVIDENCE/late_development.npz" --nref 128 --cohort development
srun "$D34_PYTHON" -m experiments.expD34_readout_race.mechanism_dilation_repair \
    --input "$D34_CAMPAIGN/evidence/inputs/confirmation.npz" \
    --output "$D34_EVIDENCE/late_confirmation.npz" --nref 128 --cohort confirmation
srun "$D34_PYTHON" -m experiments.expD34_readout_race.mechanism_dilation_repair \
    --input "$D34_CAMPAIGN/evidence/width_inputs/N512_fork20000.npz" \
    --output "$D34_EVIDENCE/wide_N512.npz" --nref 512 --cohort development
srun "$D34_PYTHON" -m experiments.expD34_readout_race.mechanism_dilation_repair \
    --input "$D34_CAMPAIGN/evidence/width_inputs/N1024_fork20000.npz" \
    --output "$D34_EVIDENCE/wide_N1024.npz" --nref 1024 --cohort development
