#!/bin/bash
#SBATCH --job-name=joint-conditioning-setup
#SBATCH --partition=gpu
#SBATCH --account=lab
#SBATCH --qos=lab-gpu2
#SBATCH --gres=none
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=/workspace/junmiaoh/experiments/precision-mlps/logs/joint-setup-%j.log
set -euo pipefail
export CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu
experiment_root=/workspace/junmiaoh/experiments/precision-mlps
if [[ ${SLURM_STEP_ID:-} == '' ]]; then
    exec srun --ntasks=1 bash "$0"
fi
# Reuse the existing CUDA/JAX packages without changing the previous environment.
if [[ ! -f ${experiment_root}/.venv-curvature/bin/python ]]; then
    "${experiment_root}/.venv/bin/python" -m venv "${experiment_root}/.venv-curvature"
fi
"${experiment_root}/.venv-curvature/bin/python" - <<'PY'
from pathlib import Path
import site
Path(site.getsitepackages()[0], 'existing_jax_environment.pth').write_text('/workspace/junmiaoh/experiments/precision-mlps/.venv/lib/python3.11/site-packages\n')
PY
"${experiment_root}/.venv-curvature/bin/python" -m pip install --no-deps \
    'equinox==0.13.8' 'lineax==0.1.1' 'jaxtyping==0.3.11' 'wadler-lindig==0.1.7' 'typing_extensions==4.15.0' \
    -e "${experiment_root}/dependencies/ssbroyden-4c87785/optimistix"
cd "${experiment_root}/code"
export PYTHONPATH=. SSBROYDEN_SOURCE="${experiment_root}/dependencies/ssbroyden-4c87785"
"${experiment_root}/.venv-curvature/bin/python" - <<'PY'
import jax,optax,equinox,lineax,optimistix
print('versions',jax.__version__,optax.__version__,equinox.__version__,lineax.__version__,optimistix.__version__)
print('devices',jax.devices())
PY
"${experiment_root}/.venv-curvature/bin/python" -m pytest -q tests/test_expD06_joint_conditioning.py
"${experiment_root}/.venv-curvature/bin/python" -m experiments.expD06_fixed_center_scales.joint_conditioning \
    --root "${experiment_root}/runs/joint_conditioning" --prepare
