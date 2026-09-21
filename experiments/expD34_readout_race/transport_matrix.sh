#!/bin/bash
set -euo pipefail
# Slurm array wrapper. Submit ONE phase at a time, with --array=...%1.
# Resource flags come from the submission; transport.sbatch launches the step.
campaign_root=$1
phase=$2
index=${SLURM_ARRAY_TASK_ID:?Requires a Slurm array}
args=()
case "${phase}:${index}" in
  adequacy:0) name=degree9; args=(--degree 9);;
  adequacy:1) name=degree17; args=(--degree 17);;
  adequacy:2) name=degree65; args=(--degree 65);;
  adequacy:3) name=law8; args=(--nodes 8);;
  adequacy:4) name=law12; args=(--nodes 12);;
  adequacy:5) name=law16; args=(--nodes 16);;
  adequacy:6) name=law12full; args=(--nodes 12 --degree -1);;
  forecast:0) name=fresh33; args=(--seeds 10 11 12 13 14);;
  forecast:1) name=halfstep; args=(--seeds 0 1 2 --eta .001);;
  validation:0) name=freshfull; args=(--degree -1 --seeds 10 11 12 13 14);;
  validation:1) name=slow33; args=(--kappa .1 --seeds 10 11 12);;
  validation:2) name=slowfull; args=(--degree -1 --kappa .1 --seeds 10 11 12);;
  validation:3) name=fast33; args=(--kappa 10 --seeds 10 11 12);;
  validation:4) name=fastfull; args=(--degree -1 --kappa 10 --seeds 10 11 12);;
  validation:5) name=width89; args=(--width 89 --seeds 0 1 2);;
  validation:6) name=width89full; args=(--width 89 --degree -1 --seeds 0 1 2);;
  validation:7) name=width353; args=(--width 353 --seeds 0 1 2);;
  validation:8) name=width353full; args=(--width 353 --degree -1 --seeds 0 1 2);;
  *) echo "Unknown campaign case ${phase}:${index}" >&2; exit 2;;
esac
exec bash "${campaign_root}/code/experiments/expD34_readout_race/transport.sbatch" \
  "${campaign_root}" --output "${campaign_root}/runs/${name}" "${args[@]}"
