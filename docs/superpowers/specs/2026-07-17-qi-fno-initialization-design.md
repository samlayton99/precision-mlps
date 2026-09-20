# expF11 -- QI-based FNO initialization (three methods): design

**Date:** 2026-07-17
**Checkpoint:** F (applications) -- data-driven, builds directly on expF10.
**Status:** approved (first-pass, initial-signal scope), pending implementation plan

## Motivation

expF10 found the QI helps an FNO as an *input encoder* (config B beat plain FNO
~27%) but not as a standalone operator, and that init/representation matters most
in the low-data regime. "QI as initialization" is a core repo theme
(`src/construction/initialize.py`, expD05) but only for MLPs -- an FNO has no
centers/gamma to copy into, so a QI *FNO* init must initialize the FNO to
*behave* QI-like. This experiment tests three such methods for **initial
signal**, each against a random-init control, in the low-data regime where init
should matter.

The teacher is the QI **physics solve**: for a coefficient `a`, the expF08 Darcy
collocation solver produces `u_QI(a)` (~2.8e-3 accurate) from the PDE alone -- no
dataset labels. That the QI solve is *more accurate than the data-driven FNO
(~7%)* is what makes it a plausible initializer.

## Shared substrate (reuse expF10)

Same `fno2d.FNO2d` (width 32, 12 modes, 4 layers), `data.load_darcy` (64^2,
area-average downsampled from darcy_421), `qi_codec` (fixed encode/decode). New:
`qi_solve.py` -- wraps expF08's `core.solve_square` + `DarcyCoefficient`
(spline-surrogate the 64^2 coefficient, sigma-presmooth, solve
`-a lap u - grad a . grad u = 1/4`, u=0 on boundary) to return `u_QI(a)` on the
64^2 grid. Control **D0** = random-init FNO trained on the labeled set
(= expF10 config C).

## The three methods (each vs D0)

- **(1) Physics-pretrain-as-init.** Pretrain the FNO on `(a, u_QI(a))` over a
  pool of `N_qi` coefficients (used *unlabeled* -- dataset `u` never seen), then
  fine-tune on a small labeled set `N_lab`. Init = "an FNO that already
  approximately solves the PDE." (First pass: pretrain on the first `N_qi` train
  coefficients via `u_QI`; fine-tune on the first `N_lab` of those via `u_ref`
  -- overlap allowed. Disjoint pretrain/label pools is a follow-up.)
- **(2) Warm-start residual.** Train the FNO to predict `u_ref - u_QI(a)`;
  inference `u_hat = u_QI(a) + FNO(a)`. The FNO learns only the correction.
  *Caveat (reported): inference pays a QI solve per instance -- not amortized.*
- **(3) Spectral-weight init from QI.** Initialize the FNO's first spectral-conv
  layer to reproduce the QI-resample (low-pass) operator `a -> decode(encode(a))`
  -- computed by a one-shot least-squares fit of that layer's per-mode weights to
  the QI-resample action on random inputs -- so the FNO *starts* by doing QI
  smoothing, then trains normally on labels. The most exploratory; may reduce to
  config B. Fallback if the fit is unstable: a fixed low-pass mode profile at the
  QI bandwidth.

## What we measure (initial signal)

1. **Low-data accuracy** -- test rel L2 at `N_lab in {100, 300}` (64^2), each
   method vs D0. Headline: does QI-init beat random init when labels are scarce?
2. **Convergence speed** -- training/val loss vs epoch for each method vs D0
   (does QI-init start lower / converge faster?).
3. **Cost accounting** -- QI-target generation time (one-time), train time, and
   for (2) the per-instance inference solve cost.

## First-pass scope

64^2; reuse the expF10 FNO; `N_qi = 500` QI solves (~15-25 min on top of
training); `N_lab in {100, 300}`; `N_test = 200`; single seed. Adam, rel-L2
loss. All configs share the same FNO hyperparameters -- this is an A/B on
*initialization only*.

## Outputs

- Code: `experiments/expF11_qi_fno_init/{qi_solve.py, init_methods.py, run.py}`
  (reusing expF10 modules via sys.path); `run.py` CLI `--method {D0,1,2,3,all}`,
  `--smoke`, `--plot`.
- Results: `results/checkpoint_F_applications/expF11_qi_fno_init/`
  (`*.json` gitignored; figures + `expF11_results.md` tracked).
- Update `results/checkpoint_F_applications/expF_results.md`.

## Tests (`tests/test_expF11_qi_fno_init.py`)

1. **`qi_solve` produces a sane field** -- `u_QI(a)` for one Darcy coefficient at
   32^2 is finite, zero on the boundary (Dirichlet), and its rel L2 vs the
   dataset `u_ref` is in a documented band (e.g. `< 5e-2`), confirming it is a
   real (if approximate) solution, not noise.
2. **Method (1) pretrain lowers the starting loss** -- an FNO pretrained on a
   handful of `(a,u_QI)` pairs has lower *initial* labeled-set loss than a
   random-init FNO (the init does something).
3. **Method (3) spectral init runs and changes weights** -- applying the QI
   spectral init to a fresh FNO changes its first spectral-conv weights and the
   net still does a forward+backward at 64^2 (and at 32^2, low-res).
4. **`run.train_eval(method, cfg)` returns finite `test_rel_l2`** for each of
   D0/1/2/3 on a tiny smoke config.

Run under `uv run --extra dev pytest` (torch + A100).

## Success criteria

A *measurement*. The framed question: for each of the three methods, does a
QI-derived initialization beat random init in the low-data regime (accuracy)
and/or converge faster? Any outcome is reportable -- including "physics
pretraining helps at N=100 but the gap closes by N=300", or "warm-start wins on
accuracy but is not amortized", or "spectral init just recovers config B".

## Non-goals (deferred)

- Full data-regime sweep / SOTA FNO tuning (this is initial signal).
- NS (Darcy only).
- Making method (2) amortized (removing the inference-time solve) -- noted as the
  obvious follow-up if it wins on accuracy.
- Multi-seed error bars.

## Reproduce

```
uv run --extra dev python experiments/expF11_qi_fno_init/run.py --method all
uv run --extra dev python experiments/expF11_qi_fno_init/run.py --plot
```
