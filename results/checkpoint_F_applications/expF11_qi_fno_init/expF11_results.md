# expF11 -- QI-based FNO initialization: three methods

**Status:** drafted (initial signal, single seed, small FNO). Builds on expF10;
teacher targets from the expF08 Darcy solver.

## TL;DR

Three ways to let the QI physics solve `u_QI(a)` inform an FNO on low-data
Darcy, each vs a random-init control (D0). Test rel L2 (64^2):

| method | N=100 | N=300 |
|---|---|---|
| D0 -- random init | 0.189 | 0.121 |
| **1 -- physics-pretrain-as-init** | **0.092** | 0.091 |
| **2 -- warm-start residual** | **0.0081** | **0.0078** |
| 3 -- QI-bandwidth spectral init | 0.189 | 0.121 |

- **Warm-start residual (2) is dramatically effective:** `u_hat = u_QI(a) +
  FNO(a)` reaches **~0.008 rel L2 -- ~15-23x better than random init, and better
  than the `u_QI` teacher itself (~1e-2)**. It is nearly label-independent
  (0.0081 @ N=100 vs 0.0078 @ N=300): `u_QI` carries the solution and the FNO
  only learns a small correction. **Caveat: not amortized** -- inference pays a
  QI solve per instance.
- **Physics-pretraining (1) is a genuine low-data init win:** 2x better than
  random at N=100 (0.092 vs 0.189), and its pretrained floor barely moves with
  more labels (0.092 -> 0.091), so the advantage is largest exactly where labels
  are scarce (the gap to D0 narrows as N grows: 0.092 vs 0.121 at N=300).
- **Spectral init (3) is a null result:** identical to random (0.189 / 0.121).
  The QI-bandwidth low-pass envelope on the spectral weights washes out under
  Adam -- the most speculative method did nothing.

## Question

An FNO has no centers/gamma to copy into (unlike the MLP QI-init in
`src/construction/initialize.py`), so a "QI FNO initialization" must make the FNO
*behave* QI-like. Which of three QI-derived initializations helps a low-data
Darcy FNO, and by how much: pretrain-as-init, warm-start residual, or a spectral
init? The QI physics solve `u_QI(a)` (~1e-2 at 64^2, *better than the 7% FNO*) is
the teacher.

## Experiment design

- **Teacher.** `qi_solve.u_qi(a)` = the expF08 Darcy collocation solve
  (spline-surrogate the 64^2 coefficient, sigma=4 pre-smooth, W=576), ~0.7 s each,
  disk-cached; 500 train + 200 test solves generated once.
- **Substrate.** expF10's FNO (width 32, 12 modes, 4 layers), `data`, `qi_codec`
  -- an A/B on *initialization only*. Control **D0** = random init.
- **Methods.** (1) pretrain FNO on `(a, u_QI)` over 500 coefficients, fine-tune
  on `N_lab` labels; (2) train on `u_ref - u_QI`, infer `u_QI + FNO`; (3) scale
  the spectral-conv weights by the QI-resample radial frequency gain.
- **Regime.** `N_lab in {100, 300}`, N_test 200, 64^2, Adam, rel-L2 loss,
  single seed, one A100. Metrics: low-data accuracy + convergence
  (`init_accuracy.png`, `init_convergence.png`).

## Results

See the table above. Reading the two figures:
- **Accuracy** (`init_accuracy.png`): method 2 sits ~1.5 orders below everything
  else at both label budgets; method 1 is a clear middle tier that is flat in N;
  method 3 overlaps D0 exactly.
- **Convergence** (`init_convergence.png`, N=100): method 2 starts and stays far
  below (the `u_QI` warm start means epoch 0 is already good); method 1 starts
  near D0 but settles lower (the pretrained features transfer); method 3 tracks
  D0 throughout.

**Cost.** Teacher generation: 700 solves x ~0.7 s ~= 8 min (one-time, cached).
Training: 8-45 s per config (method 1 adds a 100-epoch pretrain). Method 2's
inference additionally costs one QI solve per test instance (~0.7 s) -- the
non-amortized caveat.

## Conclusions

1. **Warm-starting the FNO with the QI solve is the standout** -- ~0.008 rel L2,
   better than the teacher, nearly label-free. But it keeps the QI solver in the
   inference loop, so it is a *precision* win, not an *amortization* win.
2. **Physics-pretraining is a real, cheap initialization** that helps most in the
   low-data regime and gives a label-independent floor -- the cleanest "QI init"
   result, and it *is* amortized (no solve at inference).
3. **The spectral-weight init did nothing** here -- a QI low-pass envelope on the
   Fourier weights is too weak a prior to survive training.

## Open questions / next

- **Amortize method 2:** replace the inference-time QI solve with a *learned*
  cheap approximation of `u_QI` (or method-1's pretrained net) as the warm start
  -- can we keep most of the 0.008 without the per-instance solve?
- **Disjoint pretrain/label pools** (method 1 pretrained on coefficients with no
  labels at all) -- the true semi-supervised test.
- **Stronger spectral init:** a per-mode operator fit to `u_QI` (not just the
  low-pass envelope), or freezing the init for a few warmup epochs so Adam does
  not immediately erase it.
- **Scale up** (bigger FNO, more labels) to see whether method 1's advantage
  persists once D0 is well-trained.
