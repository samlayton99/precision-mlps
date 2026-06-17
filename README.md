# precision-mlps

Research project on machine-epsilon-precision MLPs. We already have an explicit
construction (Quadrature/Quasi-Interpolation, "QI") that lets single-hidden-layer
tanh MLPs approximate smooth 1-D functions to ~1e-15, demonstrating the
theoretical capacity of MLPs. The aim now is to find an *optimizer* that can
learn such functions through training rather than via the construction.

## Documentation

- `docs/future_experiments.md` — the design spec / experiment roadmap.
- `src/construction/README.md` — QI construction quick reference.
- `CLAUDE.md` — research question, architecture map, and QI construction facts.
- `papers/QIs_workshop.pdf` — the main paper. Section 3 (construction) is not yet
  updated there; read `papers/section3_rewrite.tex` for the current Section 3 and
  `papers/practical_implementation.tex` for the fp64/mpmath implementation details.
- `docs/explanation.md` — conceptual walkthrough of the math and `qi_mpmath.py`.

## Setup

```bash
python3 -m venv ~/.venvs/precisionMLPs        # keep venvs outside iCloud-synced dirs
~/.venvs/precisionMLPs/bin/pip install torch mpmath numpy scipy pyyaml pytest matplotlib
# run from the repo root; `python -m pytest` puts the repo on sys.path so `src` imports
~/.venvs/precisionMLPs/bin/python -m pytest -q -m "not slow"   # ~70 fast tests
```

## Hardware

All computation runs in float64. CUDA supports it; Apple MPS does **not** (it
raises on float64), so MPS is never selected. Falls back to CPU. These are tiny
1-D problems, so CPU is the expected device.
