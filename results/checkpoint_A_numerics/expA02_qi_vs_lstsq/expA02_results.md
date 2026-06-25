# expA02 -- QI construction vs least-squares readout

**Status: conclusions approved by Sam.**

## TL;DR

- On the same geometry, lstsq is at least as accurate as the QI convolution in every cell (48/48), at both fp64 and mpmath.
- In fp64 -- what we train and eval in -- lstsq wins decisively (~$10^{-13}$ vs QI's ~$10^{-10}$). The Toeplitz machinery is unnecessary.

## Question

Does solving the readout matrix directly beat the QI convolution, at both fp64 and mpmath?

## Experiment design

All four methods -- QI and lstsq, each in fp64 and mpmath -- share the same geometry from the QI construction: centers $c_k$ and bandwidth $\gamma=\lambda/h$ ($h=2/N$, $K_c=160$, halo growing toward small $\lambda$ so the working width is $w=N+2\,\text{halo}+1$). Only the readout-recovery step differs -- QI uses its Toeplitz/convolution coefficients; lstsq solves the augmented system $[\Phi,\mathbf 1]\beta=y$ with $\Phi_{ik}=\tanh(\gamma(x_i-c_k))$, in fp64 via numpy and in mpmath via a truncated-SVD pseudoinverse (singular values kept above $10^{-15}s_\max$). Sweep: 4 targets x widths $\{32,64,96,128\}$ x $\lambda\in\{0.20,0.25,0.30\}$ (48 cells), scored on a 2048-point grid by eval $L_\infty=\max_x|\hat f-f|$ and rel $L_2=\|\hat f-f\|/\|f\|$. Note: lstsq minimizes the *train* residual but is scored on a separate eval grid (a different norm), so the win is empirical, not definitional.

**Code & data.** `experiments/expA02_qi_vs_lstsq/` (`run.py`, `config.yaml`). Data: `data.json` (48 cells). Figures: `qi_vs_learn_linf.png`, `qi_vs_learn_rel_l2.png`.

## Results

lstsq $\le$ QI at equal precision in all 48 cells. The fp64 margin is large because QI-fp64 only works in a narrow band near $\lambda=0.30$: at $\lambda=0.25$ it has already degraded to ~$10^{-10}$ while lstsq stays flat at ~$10^{-13}$. In mpmath both reach machine $\varepsilon$. On targets that are unresolved at small $N$ (runge, mixture) all four methods cluster at the same resolution-limited error.

### Figures

- **`qi_vs_learn_linf.png`** -- 2x2 grid, one panel per target; x = $\lambda$, y = eval $L_\infty$ (log), color = width, solid = lstsq, dashed = QI. Compare the four curves at one color: solid lstsq sits at or below the matching-precision dashed QI, and QI-fp64 rides well above the rest except near $\lambda=0.30$.
- **`qi_vs_learn_rel_l2.png`** -- same layout in relative $L_2$. Same ordering, confirming it isn't an artifact of one norm.

## Conclusions

- On shared geometry, lstsq is empirically superior to the QI Toeplitz construction: at least as accurate at equal precision everywhere, far better in fp64. Drop the convolution -- fix the geometry and solve the readout.

## Open questions

None.
