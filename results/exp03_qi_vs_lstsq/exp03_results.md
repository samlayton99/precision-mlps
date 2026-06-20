# Exp03 -- QI vs least-squares readout, four-way

**Status: draft -- pending Sam's review and sign-off on conclusions.**

## Question

On identical QI geometry (the same centers and $\gamma$), is the QI convolution formula fundamentally better at recovering the readout than a direct least-squares solve, or do the two agree once arithmetic precision is held fixed? To separate "which method" from "which arithmetic," each method is run in both fp64 and mpmath, giving a four-way comparison: QI mpmath, QI fp64, lstsq fp64, lstsq mpmath. If QI's advantage is the formula, it should persist at equal precision; if it is just arithmetic, the gap should close when both get extended precision.

## What the code runs

`experiments/exp03_qi_vs_lstsq/run.py` (sweep documented in `config.yaml`). All construction uses numpy/mpmath; the readout solves use numpy lstsq (fp64) and an mpmath truncated-SVD pseudoinverse (mpmath). Eval is on $2048$ equispaced points in $[-1,1]$.

- Widths $N$: $\{32, 64, 96, 128\}$.
- Bandwidth sweep: $\lambda = \gamma h \in \{0.20, 0.25, 0.30\}$ with $h = 2/N$.
- Targets (4): `sine`, `runge`, `exp`, `sine_mixture`.
- Geometry is shared across all four methods at each cell: the centers and $\gamma$ come from the QI mpmath construction ($K_c = 160$, `halo = default_halo(N, lambda)`), so only the readout-recovery method differs. Halo grows toward small $\lambda$, so the working width $w = N + 2\,\text{halo} + 1$ ranges from $151$ to $305$.

The four method variants at each (target, $N$, $\lambda$):

1. QI mpmath -- full QI construction in extended precision ($\text{mp\_dps} = 50$), `precision="mpmath"`.
2. QI fp64 -- full QI construction in fp64, `precision="fp64"`.
3. lstsq fp64 -- `numpy` least-squares on the augmented feature matrix $[\Phi, \mathbf 1]$, trained on $\max(512, 2w)$ equispaced points.
4. lstsq mpmath -- truncated-SVD pseudoinverse of $[\Phi, \mathbf 1]$ built and solved in mpmath ($\text{mp\_dps} = 50$, singular values kept above $10^{-15}\,s_{\max}$), result rounded to fp64.

Metrics per cell (prediction vs target on the eval grid): eval $L_\infty = \max_x |\hat f(x) - f(x)|$, and eval relative $L_2 = \lVert \hat f - f\rVert_2 / \lVert f\rVert_2$. The run also logs the mpmath-SVD rank, column count, and solve time.

Metric note: the two lstsq methods minimize the train residual on the training grid, but the reported numbers are eval $L_\infty$ / relative $L_2$ on a separate eval grid -- a different norm from the training objective, so any comparison against QI is empirical, not definitional.

## Results

Data: `results/exp03_qi_vs_lstsq/data.json` (48 cells: 4 targets $\times$ 4 widths $\times$ 3 $\lambda$).

Eval $L_\infty$ on `sine` at $\lambda = 0.25$ (the established four-way table; reproduced from `data.json` and consistent with `results/results.md`):

| $N$ | QI mpmath | QI fp64 | lstsq fp64 | lstsq mpmath |
|---|---|---|---|---|
| 32 | 1.7e-14 | 8.8e-11 | 1.6e-13 | 1.6e-15 |
| 64 | 3.0e-15 | 1.7e-10 | 6.6e-14 | 8.2e-16 |
| 96 | 2.7e-15 | 3.3e-11 | 2.8e-13 | 1.0e-15 |
| 128 | 2.3e-15 | 1.3e-10 | 1.4e-13 | 8.0e-16 |

The same four-way ordering holds on the other smooth targets at $\lambda = 0.25$. For `exp` at $N = 128$: QI mpmath $6.2\times10^{-15}$, QI fp64 $9.2\times10^{-11}$, lstsq fp64 $1.4\times10^{-13}$, lstsq mpmath $1.8\times10^{-15}$. For `runge` at $N = 128$: QI mpmath $2.4\times10^{-15}$, QI fp64 $4.1\times10^{-11}$, lstsq fp64 $2.1\times10^{-14}$, lstsq mpmath $2.2\times10^{-16}$. On the harder targets the methods cluster at the resolution-limited error rather than the fp64 floor: `runge` at $N = 32$, $\lambda = 0.25$ sits at $\sim 1\times10^{-4}$ ($L_\infty$) for all four; `sine_mixture` at $N = 32$, $\lambda = 0.25$ at $\sim 4\times10^{-3}$ for all four.

The two relevant comparisons within the table: at equal precision (QI fp64 vs lstsq fp64, and QI mpmath vs lstsq mpmath) the lstsq column is at least as accurate as the QI column at every cell; and the QI fp64 column is roughly two to three orders worse than QI mpmath at $\lambda = 0.25$, while lstsq fp64 stays within about two orders of lstsq mpmath. The mpmath SVD keeps only $\approx N$ singular values (e.g. rank $46$ of $174$ columns at $N=32$, $142$ of $270$ at $N=128$), consistent with the underdetermined readout reported in exp04/exp05.

### How to read the figures

**`qi_vs_learn_linf.png`** -- a $2\times2$ grid, one panel per target (`sine`, `runge`, `exp`, `sine_mixture`). Within each panel the x-axis is $\lambda = \gamma h$ over $\{0.20, 0.25, 0.30\}$ and the y-axis is eval $L_\infty$ on a log scale. Color encodes width $N$ ($32/64/96/128$); marker and line style encode method (circle/dashed = QI mpmath, triangle/dashed = QI fp64, square/solid = lstsq fp64, diamond/solid = lstsq mpmath). Read it by comparing, at a fixed color, the four markers at each $\lambda$: the two solid (lstsq) curves sit at or below the matching-precision dashed (QI) curve, and the QI fp64 triangles ride well above the others except near $\lambda = 0.30$, where QI fp64 re-enters its viable regime and closes most of the gap.

**`qi_vs_learn_rel_l2.png`** -- identical layout (same panels, same color/marker encoding, same $\lambda$ axis), but the y-axis is eval relative $L_2$ instead of $L_\infty$. Read it the same way; relative $L_2$ is the energy-normalized error, so it runs a little below $L_\infty$ but tells the same method-ordering story. The two figures together show the ordering is not an artifact of one norm.

## Conclusions

Plainly visible in the data (eval $L_\infty$ / relative $L_2$ on the four targets, $N \in \{32,64,96,128\}$, $\lambda \in \{0.20,0.25,0.30\}$):

- At equal arithmetic precision the least-squares readout is at least as accurate as the QI construction on this shared geometry: lstsq fp64 $\le$ QI fp64 and lstsq mpmath $\le$ QI mpmath at every cell of the table. The QI convolution formula does not beat a direct solve once precision is held fixed.
- The large fp64 margin (QI fp64 $\sim 10^{-10}$--$10^{-11}$ at $\lambda = 0.25$ vs lstsq fp64 $\sim 10^{-13}$) is because QI fp64 is outside its fp64-viable regime at $\lambda = 0.25$ (its fp64 optimum is near $\lambda \approx 0.30$, where the gap largely closes); least squares is insensitive to this choice of $\lambda$.
- The comparison is in eval $L_\infty$ / relative $L_2$, a different norm from the lstsq training objective, so this is an empirical result on these targets and grids, not a definitional one.
- On the resolution-limited targets (`runge`, `sine_mixture` at small $N$) all four methods cluster at the same error, so the method ordering above is only resolved once the target is well-approximated; there the floor is approximation error, not arithmetic.
