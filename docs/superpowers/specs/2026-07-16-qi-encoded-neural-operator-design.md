# expF10 -- QI-encoded neural operators: how much does the QI help? (design)

**Date:** 2026-07-16
**Checkpoint:** F (applications) -- a *data-driven* direction, distinct from the
physics-solve accuracy-ceiling program (expF08/expF09).
**Status:** approved (first-pass scope), pending implementation plan

## Motivation

Everything so far is a training-free per-instance solve. This experiment enters
the **data-driven neural-operator regime** deliberately, to test a specific
idea: does inserting the QI (frozen-geometry ridge) representation as a
**fixed, spectral-quality continuous encoder/decoder** around a learned operator
help -- and by how much? The learned part only has to learn the *operator*; the
QI handles the *representation*.

Key structural property exploited: for a **fixed** input grid, QI-encoding is a
**fixed linear projection** `c = Phi^+ a_grid`, where `Phi = rows_2d(grid, ...)`
is the ridge basis evaluated at the grid points and `Phi^+` its truncated
pseudo-inverse (precomputed once). Decoding is `u = Phi_out c`. Both `Phi^+` and
`Phi_out` can be built at **any** resolution from the same continuous geometry,
so a coefficient-space operator is resolution-agnostic by construction -- the
hypothesised win over a plain FFT-based FNO, which is tied to its training grid.

## What we compare (three configs, same task, matched capacity)

Task: the Darcy map `a(x,y) -> u(x,y)`, learned from data.

- **A -- QI-coeff operator.** `a_grid -> c_a = Phi^+_in a_grid -> [MLP] -> c_u
  -> u = Phi_out c_u`. No FFT; the operator is an MLP in QI-coefficient space.
- **B -- QI -> FNO.** `a_grid -> c_a -> a_canon = Phi_canon c_a` (QI-resample onto
  a fixed canonical grid) `-> [FNO] -> u`. QI is a continuous front-end
  resampler feeding a standard FNO.
- **C -- plain FNO (control).** `a_grid -> [FNO] -> u`. The baseline that tells us
  whether the QI does anything.

## Data

FNO Darcy sets on disk (`/scr/cdeng/continuous-mlps/data/fno_datasets_jax/`,
keys `x`=coefficient, `y`=solution): `darcy_train_421` (4000 x 421^2),
`darcy_test_421` (1000 x 421^2), plus native low-res `darcy_*_16` (16^2). The
same problem at multiple resolutions is exactly what the invariance study needs.

**First pass:** source `darcy_*_421`; downsample to a **64^2** training
resolution; `N_train = 1000`, `N_test = 200`. Cross-resolution eval downsamples
the test set to `{16, 32, 64, 128, 256}`. Compute: one **A100-40GB** is
available, so training is minutes, not hours.

## Architecture detail

- **QI encode/decode (`qi_codec.py`, reuses `core`/expF08 primitives).**
  `radon_geometry(W, lam)` -> geometry; `basis(P) = rows_2d(P, dirs, offs, gamma,
  [((0,0),1.0)])` is `[n_pts, W+10]` (identity operator -> tanh values + poly
  tail). `encode(a_grid, grid) = pinv(basis(grid), rcond) @ a_flat`;
  `decode(c, P) = basis(P) @ c`. `Phi^+` is cached per grid. Latent dim
  `D = W + 10`; first pass `W = 576` (`D = 586`).
- **A -- coeff MLP:** `D -> hidden -> ... -> D`, ~3-4 layers, GELU, width ~1024.
  Trains on `(c_a, c_u)` pairs (both from the fixed codec); loss is rel L2 of the
  decoded field vs the reference on the output grid.
- **FNO (`fno2d.py`, small custom torch, ~100 lines):** lift `1->width` (1x1
  conv), `K=4` spectral-conv layers (rfft2 -> keep `modes` low frequencies ->
  per-mode complex linear -> irfft2, plus a pointwise skip), project
  `width->1`. `width=32`, `modes=12` first pass. Shared by B and C.
- **B wrapper:** the QI-resample `a_canon = decode(encode(a_grid, grid_in),
  grid_canon)` feeds the FNO at the fixed canonical grid (64^2).
- **Fair comparison:** report parameter counts; keep optimizer/epochs/lr
  identical across A/B/C (Adam, ~100 epochs, cosine lr); tune only within each
  config's natural knobs.

## Metrics (= "how much does the QI help")

1. **Standard accuracy** -- mean test rel L2 at the 64^2 training resolution,
   A vs B vs C. The headline number.
2. **Discretization invariance** -- train at 64^2, evaluate zero-shot at
   `{16,32,64,128,256}`; rel L2 vs test resolution per config. Hypothesis: A
   stays ~flat (coefficient space is resolution-agnostic), C degrades off 64^2,
   B in between.
3. **Data efficiency** -- `N_train in {100, 300, 1000}` at 64^2; test rel L2 per
   config. Does the fixed QI basis generalise from fewer samples?
4. **Diagnostic -- QI input reconstruction** -- rel L2 of `decode(encode(a)) - a`
   vs `W`, on smooth vs rough (Darcy) inputs. This *bounds* how much A/B can
   help and makes the rough-coefficient caveat explicit.

## Outputs

- Code: `experiments/expF10_qi_operator/{qi_codec.py, fno2d.py, models.py,
  data.py, run.py}` (`run.py`: `--smoke`, `--config {A,B,C,all}`,
  `--eval-invariance`).
- Results: `results/checkpoint_F_applications/expF10_qi_operator/`
  (`data.json` metrics gitignored; figures + `expF10_results.md` tracked).
- Update `results/checkpoint_F_applications/expF_results.md`.

## Tests (`tests/test_expF10_qi_operator.py`)

1. **Codec round-trips a smooth field to high precision** -- `decode(encode(f))`
   vs `f` for `f = sin(pi x) sin(pi y)` on a grid reaches `< 1e-8` rel L2 (the
   fixed Phi^+/Phi are correct).
2. **Codec is resolution-transferable** -- encode on a 32^2 grid, decode on a
   64^2 grid, still reconstructs the smooth field `< 1e-6` (the property config A
   relies on).
3. **Rough-field reconstruction is bounded, not exact** -- a Darcy coefficient
   sample reconstructs with rel L2 in a documented band (e.g. 1e-3..1e-1), so the
   caveat is asserted, not assumed.
4. **FNO forward shape** -- `fno2d` maps `[b,1,H,W] -> [b,1,H,W]` and is
   differentiable (a loss.backward() runs).
5. **Each model does a forward+backward on a tiny batch** (A/B/C smoke), and the
   data loader returns matching `(a, u)` shapes at a requested resolution.

Run under `uv run --extra dev pytest` (torch + A100).

## Success criteria

A *measurement* regardless of outcome. The framed question: **quantify the QI's
help** -- (i) the accuracy delta of A and B vs the plain-FNO control C at fixed
resolution, and (ii) the invariance-robustness delta (how much less A/B degrade
off the training resolution). A null result ("QI matches FNO at fixed res but
wins on invariance", or even "QI doesn't help on rough Darcy") is a real,
reportable finding -- the point is the honest measurement.

## Non-goals (deferred)

- NS operator learning (Darcy first).
- Learning the QI geometry (it stays frozen -- that is the point).
- Beating SOTA FNO tuning; this is a controlled A/B/C at matched capacity.
- The amortized-hybrid direction (learn input->exact-solve-coeffs) -- a separate
  follow-up if A looks promising.

## Reproduce

```
uv run --extra dev python experiments/expF10_qi_operator/run.py --config all
uv run --extra dev python experiments/expF10_qi_operator/run.py --eval-invariance
```
