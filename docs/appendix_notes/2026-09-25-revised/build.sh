#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
notes=(01_true_pbit_implementation 02_finite_kernel_ratios 03_geometry_readout 04_activation_bandwidth 05_tapout_comparisons)
if command -v tectonic >/dev/null 2>&1; then
  for note in "${notes[@]}"; do
    tectonic --untrusted "$note.tex"
  done
elif command -v pdflatex >/dev/null 2>&1; then
  for note in "${notes[@]}"; do
    pdflatex -interaction=nonstopmode -halt-on-error "$note.tex"
    pdflatex -interaction=nonstopmode -halt-on-error "$note.tex"
  done
else
  echo 'Install Tectonic or a standard LaTeX distribution to rebuild the notes.' >&2
  exit 1
fi
