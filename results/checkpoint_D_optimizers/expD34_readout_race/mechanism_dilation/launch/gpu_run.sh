#!/usr/bin/env bash
# Root reviews preparation and commits source before submitting with one GPU.
# First allocation <=30 minutes; all campaign GPU allocations together <=1 hour.
set -euo pipefail
: "${SLURM_JOB_ID:?Submit this script through Slurm}"
: "${D34_SOURCE_COMMIT:?Set the reviewed source commit at submission}"
export JAX_PLATFORMS=cuda
export JAX_ENABLE_X64=true
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
D34_CAMPAIGN=/workspace/junmiaoh/experiments/precision-mlps/mechanism-0365cd3-20260923
D34_PYTHON=${D34_PYTHON:-/workspace/junmiaoh/experiments/precision-mlps/.venv-feedback/bin/python}
D34_EVIDENCE="$D34_CAMPAIGN/evidence/dilation_b52df33"
cd "$D34_CAMPAIGN/code"
# All six prepared arms are selected, including both fixed repair references.
# Each srun retains Slurm's GPU mask; the runner verifies its allocation/device.
for D34_PANEL in late_development late_confirmation wide_N512 wide_N1024; do
    srun "$D34_PYTHON" -m experiments.expD34_readout_race.mechanism_dilation_run \
        --input "$D34_EVIDENCE/$D34_PANEL.npz" --output "$D34_EVIDENCE/${D34_PANEL}_run" \
        --eta .002 --steps 20000 --backend gpu
done
srun "$D34_PYTHON" -m experiments.expD34_readout_race.mechanism_dilation_run \
    --input "$D34_EVIDENCE/late_development.npz" --output "$D34_EVIDENCE/halfstep_run" \
    --targets moment5,mixed_sine,gauss_left,bump_right,step_right,kink_abs \
    --eta .001 --steps 40000 --backend gpu
