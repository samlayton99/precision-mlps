# expA01 -- Numerics sanity checks

**Status: conclusions approved by Sam.**

## TL;DR

- The precision floor is real, not a numerics artifact.
- Solver choice matters on fixed QI geometry: lstsq/svd reach the floor; anything that forms $\Phi^\top\Phi$ (ridge, normal equations) loses ~7 decades.
- The halo is necessary -- removing it collapses every solver, and extended precision doesn't rescue it.
- fp64 $\tanh$ is accurate to 1 ulp; activation is never the bottleneck.

## Question

Before blaming the gap on optimization, is the floor set by the linear solve, the eval grid, the activation arithmetic, or tolerances -- rather than the geometry?

## Experiment design

The readout feature matrix is $\Phi_{ik}=\tanh(\gamma(x_i-c_k))$, solved with an appended bias column $[\Phi,\mathbf 1]$. One target ($\sin\pi x$), QI geometry ($K_c=160$, equispaced centers), widths $N\in\{16,\dots,256\}$, fp64 everywhere with mpmath ($\mathrm{dps}=50$) only as a reference. Six checks, one suspect each:

- **Construction baseline:** build QI in fp64 and mpmath; measure eval $L_\infty=\max_x|q-f|$ and rel $L_2=\|q-f\|/\|f\|$ across densities $n_\text{eval}\in\{256,\dots,8192\}$, with/without Kahan summation.
- **Readout solvers:** freeze $\gamma$ and centers at the QI values; solve the readout six ways -- lstsq, qr, svd, ridge at $\alpha\in\{0,10^{-14},10^{-12}\}$ -- on the *full* (halo+interior) vs *interior-only* geometry.
- **Eval-density sweep:** recompute $L_\infty$ across all $n_\text{eval}$ to test grid-independence.
- **Conditioning:** $\mathrm{cond}(\Phi)$ vs $\mathrm{cond}(\Phi^\top\Phi)$ -- the squaring incurred by forming the normal equations.
- **tanh stability:** $\max|\tanh_\text{fp64}-\tanh_\text{mpmath}|$ over $\Phi$ entries, where $|z|=\gamma(x-c)$ reaches $\approx N\lambda$ near the edges.
- **Extended-precision solve** ($N\le64$): solve the normal equations $(\Phi^\top\Phi)v=\Phi^\top y$ in mpmath, keep fp64 $v$, check whether the floor drops.

**Code & data.** `experiments/expA01_numerics_sanity/` (`run.py`, `config.yaml`). Outputs: `construction.jsonl`, `readout_solves.jsonl`, `density_sweep.jsonl`, `conditioning.jsonl`, `tanh_stability.jsonl`, `mp_solve.jsonl`, plus `summary.txt`. No figures.

## Results

- **Construction hits its floor, grid-independent:** mpmath ~$1.7\times10^{-15}$, fp64 ~$5\times10^{-12}$; refining the eval grid (256$\to$8192 pts) moves $L_\infty$ within one significant figure.
- **Solver choice tracks the $\Phi^\top\Phi$ squaring:** lstsq/svd reach ~$10^{-13}$--$10^{-14}$; ridge/normal-equations ~$10^{-6}$--$10^{-7}$; qr matches eval error but inflates weights (~$10^3$). On the interior geometry $\mathrm{cond}(\Phi)\sim10^{7}$--$10^{9}$ but $\mathrm{cond}(\Phi^\top\Phi)$ is ~7--9 decades worse.
- **Halo necessary:** interior-only collapses every solver to ~$10^{-4}$--$10^{-6}$, and mpmath normal equations don't recover it -- a geometry limit, not arithmetic.
- **Activation fine:** fp64 $\tanh$ differs from mpmath by exactly 1 ulp at every width.

## Conclusions

The floor is real and set by geometry plus fp64 arithmetic -- not by solves, eval grid, or activation. Use lstsq/svd (never $\Phi^\top\Phi$), and keep the halo. (Approved by Sam.)

## Open questions

None -- this clears the numerical suspects.
