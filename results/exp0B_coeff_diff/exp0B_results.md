# Exp0B -- QI vs Lstsq Coefficient Closeness

**Question:** on a *fixed* tanh geometry (shared centers and bandwidth), do QI and least-squares find the *same solution*?

Code: `experiments/exp0B_coeff_diff/` (`run.py`, `config.yaml`, `coeff_compare.py`; tests in `tests/test_exp0B.py`). Run: `python3 experiments/exp0B_coeff_diff/run.py`.

## Notation

The geometry is fixed by the QI construction: $W$ neurons at centers $c_k$ with bandwidth $\gamma$. The training feature matrix is

$$ \Phi_{ik} = \tanh\!\big(\gamma\,(x_i - c_k)\big), \qquad i = 1,\dots,n_{\text{train}},\quad k = 1,\dots,W. $$

We append a constant column for the bias and call the result the **augmented design matrix**

$$ A = [\,\Phi \mid \mathbf{1}\,] \in \mathbb{R}^{\,n_{\text{train}}\times(W+1)}. $$

A coefficient vector $\beta = [\,\mathbf{a}\;;\,b\,] \in \mathbb{R}^{W+1}$ (outer weights $\mathbf{a}$ plus bias $b$) defines the model outputs at the training points by $A\beta$. We compare $\beta_{QI} = [\,\mathbf{a}_{QI};\,c_0\,]$ against the least-squares solution $\beta_{LS} = [\,\mathbf{v};\,b\,]$, both on the *same* $A$. Let $y_i = f(x_i)$ be the target samples.

## Experiments run (exactly as set up in the code)

Common setup (fp64 throughout): $\lambda = 0.30$, $K_c = 160$, eval grid $N_{\text{eval}} = 2048$. For each (target, $N$) the QI construction fixes $\gamma$ and the centers; the lstsq readout is solved on those same centers.

**Metrics** (defined and unit-tested in `coeff_compare.py`):

- Deviation ratio (full space), "how big is the worst single-coefficient disagreement, in units of the typical coefficient size?":

$$ \mathrm{ratio_{full}} = \frac{\displaystyle\max_i \big|\beta_{QI,i} - \beta_{LS,i}\big|}{\max\big(\overline{|\beta_{QI}|},\,\overline{|\beta_{LS}|}\big)}, \qquad \overline{|\beta|} = \frac{1}{W+1}\sum_i |\beta_i|. $$

- $\mathrm{ratio_{row}}$: the same ratio after projecting both vectors onto the row space of $A$ (defined below).
- Function difference: $\mathrm{fun\_linf} = \big\|f_{QI} - f_{LS}\big\|_\infty$ on the eval grid.
- Fit residuals: $\mathrm{qi\_fit} = \|A\beta_{QI} - y\|/\|y\|$ and $\mathrm{ls\_fit} = \|A\beta_{LS} - y\|/\|y\|$ on the train grid -- did each method actually fit the data.

### What the row-space projection is (which matrix, what operation)

The matrix is the augmented training design matrix $A$ above. Each **row** of $A$ is one training point's feature vector $\big[\tanh(\gamma(x_i - c_1)),\dots,\tanh(\gamma(x_i - c_W)),\,1\big] \in \mathbb{R}^{W+1}$.

The **row space** $\mathrm{row}(A)$ is the span of those rows -- a subspace of $\mathbb{R}^{W+1}$ of dimension $\mathrm{rank}(A)$. Its orthogonal complement is the null space $\ker(A)$. The model outputs are $A\beta$, and for any $n \in \ker(A)$,

$$ A(\beta + n) = A\beta \quad\text{because}\quad An = 0. $$

So $\ker(A)$ is exactly the set of coefficient directions the data cannot see; $\mathrm{row}(A)$ is the part it pins down.

The projection $P\beta$ is the orthogonal projection of $\beta$ onto $\mathrm{row}(A)$. Equivalent forms:

$$ P = V_r V_r^\top = A^{+}A, \qquad P\beta = A^{+}(A\beta), $$

where $V_r$ are the right singular vectors of $A$ with singular value above $10^{-11}\,\sigma_{\max}$, and $A^{+}$ is the pseudoinverse. The last form says $P\beta$ is the **minimum-norm coefficient vector that produces the same outputs $A\beta$**. $\mathrm{ratio_{row}}$ applies this same $P$ to both $\beta_{QI}$ and $\beta_{LS}$ and takes their deviation ratio.

**Two sweeps:**

1. **vs width.** targets $\in$ {sine, sine_8pi, runge, sine_mixture, exp, abs_cubed} (one per category) $\times\; N \in \{32, 64, 96, 128\}$; $n_{\text{train}} = \max(512,\,2W)$.
2. **vs lstsq sample count.** target $=$ sine; $N \in \{32, 64, 96, 128\}$; $n_{\text{train}} \in \{256, 512, 1024, 2048, 4096\}$.

## Results

`data.json` holds two lists, `width_sweep` and `nsamples_sweep` (plus `lambda`, `Kc`). Each row is one (target, $N$, $n_{\text{train}}$) config with `ratio_full`, `ratio_row`, `fun_linf`/`fun_rel_l2`, `raw_l2`/`raw_linf` (raw coefficient diff), `row_mean_qi`/`row_mean_ls` (mean $|\beta|$ in the row space), `rank`/`null_dim`, and `qi_fit_resid`/`ls_fit_resid`.

Representative numbers (sweep 1, $N=128$; full table in `data.json`):

| target | $\mathrm{ratio_{full}}$ | $\mathrm{ratio_{row}}$ | $\mathrm{fun\_linf}$ | qi_fit | ls_fit |
|---|---|---|---|---|---|
| sine | 1.6e0 | 1.5e-4 | 4.7e-12 | ~1e-12 | ~1e-13 |
| sine_8pi | 1.6e0 | 1.7e-4 | 9.5e-12 | 6.7e-12 | 3.1e-13 |
| sine_mixture | 2.1e0 | 1.1e-3 | 2.0e-11 | 1.3e-11 | 1.5e-12 |
| exp | 1.3e2 | 2.7e-3 | 2.1e-13 | 8.7e-14 | 1.2e-14 |
| runge | 4.7e-2 | 2.3e-4 | 1.8e-12 | 1.7e-12 | 1.1e-14 |
| abs_cubed | 1.2e2 | 5.5e-1 | 2.4e-7 | 7.7e-8 | 4.8e-8 |

### Figures

**`coeff_ratio_full_vs_row.png` (headline, sweep 1).** Deviation ratio vs width; each color a target, solid $= \mathrm{ratio_{full}}$, dashed $= \mathrm{ratio_{row}}$. *Result:* solid lines sit near $1$ -- the worst coefficient disagrees by about one typical coefficient (exp/abs_cubed higher, $\sim 10$ to $10^2$) -- while dashed lines for converged smooth targets drop to $10^{-4}$ to $10^{-3}$. The vertical gap between a target's two lines *is* the null-space freedom. Lines that never drop (abs_cubed; coarse-$N$ runge/mixture) are unconverged constructions, not agreement.

**`coeff_diff_vs_width.png` (sweep 1).** Two panels vs width: top $=$ raw coefficient diff ($\|\beta_{QI}-\beta_{LS}\|_2$ dotted, $\|\cdot\|_\infty$ solid), bottom $=$ function diff ($\mathrm{fun\_rel\_l2}$ dotted, $\mathrm{fun\_linf}$ solid). *Result:* the raw coefficient diff stays $\mathcal{O}(1)$ at every $N$, while the function diff falls to $\sim 10^{-12}$ as $N$ grows for smooth targets (abs_cubed plateaus $\sim 10^{-7}$). Same geometry, same function, different weights.

**`coeff_diff_vs_nsamples.png` (sweep 2).** Same two panels, but vs $n_{\text{train}}$ (color $= N$, target $=$ sine). *Result:* every line is flat -- increasing the lstsq sample count from $256$ to $4096$ changes neither the coefficient diff nor the function diff.

### Why the coefficients are underdetermined (and why more data does not help)

The readout solve $A\beta = y$ is rank-deficient:

$$ \mathrm{rank}(A) \approx N + 12 \quad\text{of}\quad W+1 \approx N + 120 \text{ columns}, \qquad \dim\ker(A) \approx 108 \text{ at every width.} $$

The columns are near-collinear on $[-1,1]$: adjacent $\tanh$ bumps overlap at large $\gamma$, and the $\approx 59$ halo neurons per side saturate to near-constants on the domain. So the features span only an $\approx N$-dimensional space, and $\approx 108$ weight combinations are invisible to the data.

This is a property of the **columns (geometry)**, not the rows. Sampling the same smooth features at more points gives $\Phi^\top\Phi \approx n_{\text{train}}\,M$ for the continuous feature Gram matrix $M$, so

$$ \sigma_i(\Phi) \approx \sqrt{n_{\text{train}}}\;\sqrt{\mu_i(M)}. $$

Adding rows scales **all** singular values by $\sqrt{n_{\text{train}}}$ without changing their *spread*; the rank and null space are set by $M$ (the geometry) and are unchanged. `coeff_diff_vs_nsamples.png` confirms this empirically (flat lines). It is fixed only by changing the geometry (smaller $\gamma$, fewer/spread centers, no halo) or by regularization (which *picks* a representative, as min-norm lstsq already does) -- never by more data.

### Note on the row-space measurement

Reading $\mathrm{ratio_{row}}$ honestly required two choices, recorded in the code: include the bias (else exp's DC term -- in $c_0$ for QI, split between $\mathbf{v}$ and $b$ for lstsq -- fakes a $\sim 2.5\times$ disagreement), and use the $10^{-11}$ singular-value cutoff (the last decade of singular values sits at the fp64 floor and carries spurious disagreement).

## Conclusions

(Approved by Sam.)

**1. The coefficients are not the same; the difference is entirely null-space.** $\mathrm{ratio_{full}}$ is $\mathcal{O}(1)$ to $\mathcal{O}(10^2)$ -- lstsq uses systematically smaller (min-norm) weights -- yet $\mathrm{fun\_linf}\sim 10^{-12}$ for converged smooth targets. The entire disagreement lives in the $\approx 108$-dimensional $\ker(A)$ the data cannot see. QI and lstsq are the **same data-visible solution**, free in the invisible directions.

**2. The row-space agreement is forced once both fit, not independent evidence.** The projection depends *only on the output function*, since

$$ P\beta = A^{+}(A\beta) $$

is a function of $A\beta$ alone -- it cannot see how $\beta$ was computed (convolution, Toeplitz solve, lstsq). It maps any $\beta$ to the unique min-norm representative of its function. Now $\beta_{LS} = A^{+}y$ is already that representative, and

$$ P\beta_{QI} - \beta_{LS} = A^{+}(A\beta_{QI}) - A^{+}y = A^{+}\big(A\beta_{QI} - y\big) = A^{+}\,r_{QI}, $$

where $r_{QI} = A\beta_{QI} - y$ is QI's fit residual. So *given* QI fits the target, $P\beta_{QI} \approx \beta_{LS}$ automatically -- regardless of the completely different process. (The size $\sim 10^{-4}$ rather than the residual $\sim 10^{-12}$ is $A^{+}$ amplifying by $1/\sigma_{\min}$ near the cutoff; that is conditioning, not agreement.)

The genuinely non-obvious fact is the premise "QI fits the target," which comes from the Toeplitz-plus-convolution construction working -- and it is measured by $\mathrm{qi\_fit}\sim 10^{-12}$, **not** by the row-space comparison. So $\mathrm{ratio_{row}}$ only restates "both fit the same function" (already in $\mathrm{fun\_linf}$ and $\mathrm{qi\_fit}$); it adds nothing, and it does *not* independently show the two methods are related. The independent, non-obvious finding remains conclusion 1: the full-space $\mathcal{O}(1)$ difference, i.e. the null-space freedom.

<!-- Proposed but NOT yet approved (do not treat as conclusions): that this reframes the weight-blowup violation as a question of which null-space representative an optimizer selects. Pending Sam's review. -->
