#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --account=lab
#SBATCH --qos=lab-gpu2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --no-requeue
set -euo pipefail
campaign_root=$1
phase=$2
cd "${campaign_root}/code"
export RACE_SOURCE_COMMIT=0f3385d JAX_ENABLE_X64=true
export XLA_PYTHON_CLIENT_PREALLOCATE=false OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLBACKEND=Agg
python_bin=/workspace/junmiaoh/experiments/precision-mlps/.venv-race/bin/python
if [[ ${phase} == prepare || ${phase} == analysis ]]; then
  export CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu
  srun "${python_bin}" - "${campaign_root}/modal" <<'PY'
import hashlib, json, pathlib, sys
root = pathlib.Path(sys.argv[1])
assert (root/'long/gd/state.npz').is_file(), 'Missing flattened GD bundle'
assert (root/'long/adam/state.npz').is_file(), 'Missing flattened Adam bundle'
for name, expected in json.loads((root/'transfer_hashes.json').read_text()).items():
    assert hashlib.sha256((root/name).read_bytes()).hexdigest() == expected, name
print('All transferred checkpoint hashes verified', flush=True)
PY
else
  export JAX_PLATFORMS=cuda
fi
case "${phase}" in
  prepare)
    srun "${python_bin}" -m pytest -q tests/test_expD34_plateau_followup.py tests/test_expD34_plateau_outputs.py tests/test_expD34_plateau_run.py tests/test_expD34_plateau_runtime.py tests/test_expD34_plateau_dense.py tests/test_expD34_plateau_gate.py
    srun "${python_bin}" -u -m experiments.expD34_readout_race.plateau_followup prepare --source "${campaign_root}/modal" --output "${campaign_root}/late"
    ;;
  analysis)
    srun "${python_bin}" -u -m experiments.expD34_readout_race.plateau_results --root "${campaign_root}/modal" --output "${campaign_root}/initial-analysis"
    ;;
  pilot)
    srun "${python_bin}" -u -m experiments.expD34_readout_race.plateau_followup pilot --source "${campaign_root}/modal" --output "${campaign_root}/late"
    ;;
  probe)
    srun "${python_bin}" -u -m experiments.expD34_readout_race.plateau_followup probe --output "${campaign_root}/late" --seed "${SLURM_ARRAY_TASK_ID}" --max-seconds 1080
    ;;
  *) exit 2 ;;
esac
