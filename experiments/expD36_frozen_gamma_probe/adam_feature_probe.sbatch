#!/usr/bin/env bash
#SBATCH --job-name=adam-feature-probe
#SBATCH --partition=gpu
#SBATCH --account=lab
#SBATCH --qos=lab-gpu2
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
set -euo pipefail
cd "${PROBE_ROOT:?}"
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
  if [[ -n "${PROBE_STEPS:-}" ]]; then flags+=(--steps "$PROBE_STEPS"); fi
  runner=adam_feature_probe_run.py
  input=${PROBE_INPUT:-input.npz}
  config=adam_feature_probe_config.json
  if [[ "${PROBE_KIND:-frozen}" == joint ]]; then
    runner=adam_joint_probe_run.py
    input=joint_input.npz
    config=adam_joint_probe_config.json
  fi
  /workspace/junmiaoh/experiments/precision-mlps/.venv/bin/python -u "$runner" \
    --input "$input" --config "$config" \
    --output "${PROBE_OUTPUT:?}" --require-gpu --self-test "${flags[@]}"
'
