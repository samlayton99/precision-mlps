# expA03 -- QI vs lstsq: coefficient closeness and the readout null space

**Status: conclusion approved by Sam**

## TL;DR

- On the same geometry, QI and lstsq are the same function but different coefficients: per-coefficient they differ by $O(1)$--$O(10^2)$, yet agree as functions to ~$10^{-12}$.
- The whole difference lives in the ~108-dim null space of $A=[\Phi,\mathbf 1]$. lstsq picks the min-norm representative; QI does not.

## Question

On a fixed geometry, do QI and least squares find the same solution? If not, where does the difference live -- in directions the data sees, or directions it can't?

## Experiment design

Fix the QI geometry ($\lambda=0.30$, $K_c=160$, fp64). The augmented design matrix is $A=[\Phi\mid\mathbf 1]$ with $\Phi_{ik}=\tanh(\gamma(x_i-c_k))$, and a coefficient vector $\beta=[a;b]$ produces the outputs $A\beta$. Solve the readout two ways on the *same* $A$: QI's coefficients $\beta_{QI}$ vs least squares $\beta_{LS}=A^{+}y$. Compare with three lenses:

- **full-space deviation ratio** $\dfrac{\max_i|\beta_{QI,i}-\beta_{LS,i}|}{\max(\overline{|\beta_{QI}|},\overline{|\beta_{LS}|})}$ -- the worst single-coefficient disagreement in units of a typical coefficient;
- the **same ratio projected onto $\mathrm{row}(A)$** (via $P=A^{+}A$) -- restricted to the directions the data can see;
- the **function difference** $\|f_{QI}-f_{LS}\|_\infty$ on the eval grid;

plus each method's fit residual $\|A\beta-y\|/\|y\|$. Two sweeps: vs width (6 targets, one per category, $N\in\{32,\dots,128\}$) and vs lstsq sample count ($n_\text{train}\in\{256,\dots,4096\}$, sine).

**Code & data.** `experiments/expA03_coeff_nullspace/` (`run.py`, `coeff_compare.py`; tests in `tests/test_expA03_coeff_nullspace.py`). Data: `data.json`. Figures: `coeff_ratio_full_vs_row.png` (headline), `coeff_diff_vs_width.png`, `coeff_diff_vs_nsamples.png`.

## Results

- **Same function, different coefficients.** The full-space deviation is $O(1)$ (worst coefficient off by about one typical coefficient; `exp`/`abs_cubed` reach $O(10^2)$) while the function difference falls to ~$10^{-12}$ as width grows.
- **The disagreement is entirely null-space.** $A$ is rank-deficient ($\dim\ker(A)\approx108$ at every width) because adjacent $\tanh$ bumps overlap and halo neurons saturate to near-constants. Those invisible directions are exactly where QI and lstsq differ.
- **More data doesn't help:** sweeping the sample count leaves both the coefficient and function differences flat -- adding rows scales all singular values equally, leaving the null space unchanged.

### Figures

- **`coeff_ratio_full_vs_row.png`** (headline) -- deviation ratio vs width, one color per target, solid = full-space, dashed = row-space. Solid lines sit near 1 (raw disagreement); dashed lines drop to ~$10^{-4}$ for converged smooth targets. The vertical gap between a target's two lines *is* the null-space freedom; lines that never drop are unconverged constructions.
- **`coeff_diff_vs_width.png`** -- raw coefficient diff stays $O(1)$ at every $N$ while the function diff falls to ~$10^{-12}$.
- **`coeff_diff_vs_nsamples.png`** -- every line flat: sample count changes nothing.

## Additional details

- The row-space agreement is *forced* once both methods fit ($P\beta=A^{+}(A\beta)$ depends only on the output function), so it restates "both fit the same target" and is not independent evidence -- the real finding is the full-space $O(1)$ difference.

## Conclusions

The coefficients are not the same, and the entire difference is null-space freedom: QI and lstsq are the same data-visible solution, with lstsq choosing the min-norm representative. (Approved by Sam.)

## Open questions

None
