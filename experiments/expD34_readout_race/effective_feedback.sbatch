#!/usr/bin/env bash
#SBATCH --job-name=d34-feedback
#SBATCH --partition=gpu
#SBATCH --account=lab
#SBATCH --qos=lab-gpu2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
set -euo pipefail

# Submit only after reserving this allocation in the shared campaign budget.
# Arguments: checkout python inputs output arm reservation [horizon] [max-seconds] [indices] [predictions] [eta] [degree].
checkout=$1
python_bin=$2
inputs=$3
output=$4
arm=$5
reservation=$6
horizon=${7:-20000}
max_seconds=${8:-1500}
indices=${9:-}
predictions=${10:-}
eta=${11:-.002}
degree=${12:-65}
extra=()
if [[ -n "${indices}" ]]; then extra+=(--indices "${indices}"); fi
if [[ -n "${predictions}" ]]; then extra+=(--predictions "${predictions}"); fi
test -n "${reservation}"
cd "${checkout}"
export JAX_ENABLE_X64=true JAX_PLATFORMS=cuda
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
export D34_EXECUTION_PLATFORM=slurm D34_BUDGET_RESERVATION="${reservation}"
# Preserve the scheduler's CUDA_VISIBLE_DEVICES; verify inside the actual step.
srun bash -c '
  set -euo pipefail
  test -n "${SLURM_JOB_ID:-}"
  test -n "${SLURM_STEP_ID:-}"
  test -n "${CUDA_VISIBLE_DEVICES:-}"
  scontrol show job "${SLURM_JOB_ID}"
  scontrol show step "${SLURM_JOB_ID}.${SLURM_STEP_ID}"
  exec "$@"
' bash "${python_bin}" -m experiments.expD34_readout_race.effective_feedback run \
  --inputs "${inputs}" --output "${output}" --arm "${arm}" \
  --eta "${eta}" --degree "${degree}" --horizon "${horizon}" --max-seconds "${max_seconds}" \
  --backend slurm "${extra[@]}"
