#!/usr/bin/env bash
#SBATCH --job-name=section34
#SBATCH --partition=gpu
#SBATCH --account=lab
#SBATCH --qos=lab-gpu2
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:30:00
set -euo pipefail
cd "${SECTION34_CODE:?}"
export JAX_ENABLE_X64=true JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8
srun bash -c '
  set -euo pipefail
  test -n "${SLURM_JOB_ID:-}"
  test -n "${SLURM_STEP_ID:-}"
  test -n "${CUDA_VISIBLE_DEVICES:-}"
  scontrol show job "$SLURM_JOB_ID"
  scontrol show step "$SLURM_JOB_ID.$SLURM_STEP_ID"
  flags=()
  if [[ -n "${SECTION34_STEPS:-}" ]]; then flags+=(--steps "$SECTION34_STEPS"); fi
  /workspace/junmiaoh/experiments/precision-mlps/.venv/bin/python -u "${SECTION34_RUNNER:?}" \
    --input "${SECTION34_INPUT:?}" --config "${SECTION34_CONFIG:?}" \
    --output "${SECTION34_OUTPUT:?}" --require-gpu --self-test "${flags[@]}"
'
