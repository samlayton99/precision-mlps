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

Data: `results/exp0B_coeff_diff/data.json`. Figures (this folder):
- `coeff_ratio_full_vs_row.png` -- headline (full vs row-space ratio vs width).
- `coeff_diff_vs_width.png` -- raw coefficient diff and function diff vs width.
- `coeff_diff_vs_nsamples.png` -- the same vs lstsq sample count.

Representative numbers (sweep 1; full table in `data.json`):

| target | N | ratio_full | ratio_row | fun_linf | qi_fit | ls_fit |
|---|---|---|---|---|---|---|
| sine | 128 | 1.6e0 | 1.5e-4 | 4.7e-12 | ~1e-12 | ~1e-13 |
| sine_8pi | 128 | 1.6e0 | 1.7e-4 | 9.5e-12 | 6.7e-12 | 3.1e-13 |
| sine_mixture | 128 | 2.1e0 | 1.1e-3 | 2.0e-11 | 1.3e-11 | 1.5e-12 |
| exp | 128 | 1.3e2 | 2.7e-3 | 2.1e-13 | 8.7e-14 | 1.2e-14 |
| runge | 128 | 4.7e-2 | 2.3e-4 | 1.8e-12 | 1.7e-12 | 1.1e-14 |
| abs_cubed | 128 | 1.2e2 | 5.5e-1 | 2.4e-7 | 7.7e-8 | 4.8e-8 |

Observed (factual):
- Full-space ratio is O(1)-O(100) for every smooth target; the rank of `[Phi,1]`
  is ~N+12 of ~N+2*halo+1 columns, i.e. a ~110-dim null space at every width.
- For converged smooth targets the row-space ratio is 1e-4-1e-3 and `fun_linf`
  is ~1e-12; the row-space mean magnitudes of the two vectors match to ~4 digits.
- The cases that stay high in BOTH ratio_row and the fit residuals are exactly
  the unconverged ones: abs_cubed (only C^1) at all N, and runge/sine_mixture at
  N<=64. There both methods are far from the target (fit resid 1e-5-1e-7).
- Sweep 2: holding N fixed and varying n_train from 256 to 4096 leaves
  ratio_full, ratio_row, and fun_linf flat.
- Getting ratio_row to read honestly required two choices, recorded in the code:
  include the bias (else exp's DC term -- in `c0` for QI, split between `v` and
  `bias` for lstsq -- fakes a ~2.5x disagreement), and use the 1e-11 SVD cutoff
  (the last decade of singular values carries spurious disagreement).

### How to read `coeff_ratio_full_vs_row.png`

Each color is a target; solid = full-space ratio, dashed = row-space ratio.
Solid lines sit near 1 (the worst coefficient disagrees by ~one typical
coefficient). Dashed lines for converged smooth targets drop to 1e-4-1e-3; the
vertical gap between a target's solid and dashed line *is* the null-space
freedom. Lines that do not drop (abs_cubed; coarse-N runge/mixture) are
unconverged constructions, not genuine agreement.

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
