# Exp0B -- QI vs Lstsq Coefficient Closeness

**Question:** on a *fixed* tanh geometry (shared centers + gamma), do QI and
least-squares find the *same solution*?

Code: `experiments/exp0B_coeff_diff/` (`run.py`, `config.yaml`,
`coeff_compare.py`; tests in `tests/test_exp0B.py`). Run:
`python3 experiments/exp0B_coeff_diff/run.py`.

## Experiments run (exactly as set up in the code)

Common setup (fp64 throughout): lambda=0.30, Kc=160, eval grid N_EVAL=2048. For
each (target, N) the QI construction fixes the geometry (centers + gamma); the
lstsq readout is solved on the *same* centers+gamma. Compared vectors are the
augmented weights `beta_qi = [a_coeffs ; c0]` and `beta_ls = [v ; bias]`.

Metrics (defined and unit-tested in `coeff_compare.py`):
- `ratio_full` = `max_i |beta_qi_i - beta_ls_i| / max(mean|beta_qi|, mean|beta_ls|)`
  -- largest element-wise coefficient deviation in units of the typical
  coefficient, over the full vector.
- `ratio_row` = the same ratio after projecting both vectors onto the row space
  of the augmented design `[Phi, 1]` (see next subsection).
- `fun_linf` = `max|f_qi - f_ls|` on the eval grid.
- `qi_fit_resid`, `ls_fit_resid` = `||A beta - y|| / ||y||` on the train grid
  (sanity: did each method actually fit the data).

### What the row-space projection is (which matrix, what operation)

The matrix is the augmented training design matrix `A = [Phi | 1]`, shape
`n_train x (W+1)`, one row per training point:
`A[i,:] = [tanh(gamma*(x_i - c_1)), ..., tanh(gamma*(x_i - c_W)), 1]`. A
coefficient vector `beta = [outer weights ; bias]` lives in `R^(W+1)`.

The **row space of A** is the span of A's rows -- a subspace of `R^(W+1)` of
dimension `rank(A) ~ N+12`. Its orthogonal complement is `null(A)` (~110-dim
here). The model's outputs at the training points are exactly `A beta`, and
`A beta` is unchanged by adding any vector in `null(A)` (since `A * beta_null =
0`). So the row space is precisely the part of the coefficients the data can see;
the null space is invisible to it.

The projection `P beta` is the orthogonal projection of `beta` onto `row(A)` --
equivalently `P = V_r V_r^T` (right singular vectors of A with singular value
above `1e-11 * s_max`), or `P = A^+ A`, or "the minimum-norm coefficient vector
that produces the same outputs `A beta`." `ratio_row` applies this same `P` to
both `beta_qi` and `beta_ls` and takes the deviation ratio of the results.

Two sweeps:
1. **vs width.** targets = {sine, sine_8pi, runge, sine_mixture, exp, abs_cubed}
   (one per category) x N in {32, 64, 96, 128}; n_train = max(512, 2W).
2. **vs lstsq sample count.** target = sine; N in {32, 64, 96, 128};
   n_train in {256, 512, 1024, 2048, 4096}.

## Results

`data.json` holds two lists, `width_sweep` and `nsamples_sweep` (plus `lambda`,
`Kc`). Each row is one (target, N, n_train) config with: `ratio_full`,
`ratio_row`, `fun_linf`/`fun_rel_l2` (function diff), `raw_l2`/`raw_linf` (raw
coefficient diff), `row_mean_qi`/`row_mean_ls` (mean |coeff| in the row space),
`rank`/`null_dim`, and `qi_fit_resid`/`ls_fit_resid` (relative fit residuals).

Representative numbers (sweep 1; full table in `data.json`):

| target | N | ratio_full | ratio_row | fun_linf | qi_fit | ls_fit |
|---|---|---|---|---|---|---|
| sine | 128 | 1.6e0 | 1.5e-4 | 4.7e-12 | ~1e-12 | ~1e-13 |
| sine_8pi | 128 | 1.6e0 | 1.7e-4 | 9.5e-12 | 6.7e-12 | 3.1e-13 |
| sine_mixture | 128 | 2.1e0 | 1.1e-3 | 2.0e-11 | 1.3e-11 | 1.5e-12 |
| exp | 128 | 1.3e2 | 2.7e-3 | 2.1e-13 | 8.7e-14 | 1.2e-14 |
| runge | 128 | 4.7e-2 | 2.3e-4 | 1.8e-12 | 1.7e-12 | 1.1e-14 |
| abs_cubed | 128 | 1.2e2 | 5.5e-1 | 2.4e-7 | 7.7e-8 | 4.8e-8 |

### Figures

**`coeff_ratio_full_vs_row.png` (headline, sweep 1).** Deviation ratio vs width;
each color a target, solid = full space, dashed = row space.
*Result:* solid lines sit near 1 -- the worst coefficient disagrees by ~one
typical coefficient (exp/abs_cubed higher, ~1e1-1e2) -- while dashed lines for
converged smooth targets drop to 1e-4-1e-3. The vertical gap between a target's
two lines *is* the null-space freedom. Lines that never drop (abs_cubed;
coarse-N runge/mixture) are unconverged constructions, not agreement.

**`coeff_diff_vs_width.png` (sweep 1).** Two panels vs width: top = raw
coefficient diff (`raw_l2` dotted, `raw_linf` solid), bottom = function diff
(`fun_rel_l2` dotted, `fun_linf` solid).
*Result:* the raw coefficient diff stays O(1) at every N, while the function diff
falls to ~1e-12 as N grows for smooth targets (abs_cubed plateaus ~1e-7). Same
geometry, same function -- different weights.

**`coeff_diff_vs_nsamples.png` (sweep 2).** Same two panels, but vs `n_train`
(color = N, target = sine).
*Result:* every line is flat -- increasing the lstsq sample count from 256 to
4096 changes neither the coefficient diff nor the function diff.

### Why the coefficients are underdetermined (and why more data does not help)

The readout solve `A beta = y` (`A = [Phi, 1]`) is rank-deficient: `rank(A) ~
N+12` of `~N+120` columns, a ~110-dim null space at every width. The columns are
near-collinear on `[-1,1]` -- adjacent `tanh` bumps overlap at large gamma, and
the ~59 halo neurons per side saturate to near-constants on the domain -- so the
features span only an ~N-dim space and ~110 weight combinations are invisible to
the data. This is a property of the **columns (geometry)**, not the rows: adding
samples scales all singular values by ~`sqrt(n_train)` without changing their
spread, so rank and null space are unchanged. `coeff_diff_vs_nsamples.png`
confirms this empirically (flat lines). It is fixed only by changing the geometry
(smaller gamma, fewer/spread centers, no halo) or by regularization (which picks
a representative, as min-norm lstsq already does) -- never by more data.

### Note on the row-space measurement

Getting `ratio_row` to read honestly required two choices, recorded in the code:
include the bias (else exp's DC term -- in `c0` for QI, split between `v` and
`bias` for lstsq -- fakes a ~2.5x disagreement), and use the 1e-11 SVD cutoff
(the last decade of singular values carries spurious disagreement).

## Conclusions

(Approved by Sam.)

1. QI and lstsq do **not** produce the same coefficients, but the entire
   difference lives in the ~110-dim null space the data cannot see: they are the
   **same data-visible solution**, free in the invisible directions.
2. The row-space agreement is **obvious / forced**, not independent evidence. It
   follows purely from both methods fitting the same data: `lstsq` already
   returns the min-norm solution (`beta_ls = A^+ y`, so `P beta_ls = beta_ls`),
   and `P beta_qi = A^+ A beta_qi = A^+ (QI's outputs)`, so
   `P beta_qi - beta_ls = A^+ (A beta_qi - y) = A^+ * (QI's fit residual)`. The
   row-space coefficient difference is just QI's data-fitting residual pushed
   through the pseudoinverse (small because QI fits; the `1e-4` size vs the
   `1e-12` residual is `A^+` amplifying by `1/s_min` near the cutoff -- that is
   conditioning, not agreement). So `ratio_row` restates the function agreement
   (`fun_linf`) in coefficient coordinates and adds nothing to it. The
   independent, non-obvious finding is conclusion 1: the full-space O(1)
   difference, i.e. the null-space freedom.

<!-- Proposed but NOT yet approved (do not treat as conclusions):
     - that this reframes the weight-blowup violation as a question of which
       null-space representative an optimizer selects. Pending Sam's review. -->
