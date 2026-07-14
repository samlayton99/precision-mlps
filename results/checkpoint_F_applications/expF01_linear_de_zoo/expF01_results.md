# expF01 -- linear differential-equation zoo: solve, don't train

**Status: draft -- conclusions pending Sam's sign-off.**

## TL;DR

- The QI-geometry collocation solve reaches the fp64 floor on 8 of 9 linear problems with zero training: all three 1D ODEs (floor by $W\approx200$), all three 2D steady PDEs ($8\times10^{-14}$ / $9\times10^{-15}$ / $1.2\times10^{-13}$ at $W=2304$, including the inflow-BC transport and a third-order operator), and 2 of 3 space-time PDEs (advection $3\times10^{-13}$, heat $1\times10^{-13}$ at $W=4096$).
- The one exception is dispersive: space-time Airy ($u_t + u_{xxx} = 0$) plateaus at rel $L_2\sim3\times10^{-8}$. The residual field is smooth and pinned to $\sim10^{-12}$ at the boundaries but $\sim10^{-9}$ in the interior -- a stability-constant gap (weak interior control by boundary data), not a resolution failure.
- Third-order operators show the $\gamma^3$ roundoff drift: the 1D order-3 ODE hits $3\times10^{-15}$ at $W=173$ and then *rises* to $\sim10^{-11}$ by $W=1845$. Best practice is the smallest width that resolves the solution, not the largest affordable.
- Full suite (126 lstsq solves + all figures/gifs) runs in ~7 minutes on CPU.

## Question

Does the frozen-geometry + lstsq recipe solve *actual* linear ODEs/PDEs -- across operator order (1/2/3), domain type (interval, disk, space-time rectangle), and condition type (IVP, Dirichlet, Neumann, inflow, Cauchy, IC) -- and to what precision as width grows?

## Experiment design

Model: $u(p) = \sum_m a_m \tanh(\gamma(w_m^\top p - t_m)) + \text{poly}(p)$, where poly is all monomials of total degree $\le 3$ (absorbing the operator's kernel). Geometry is frozen: 1D uses the uniform QI grid with halo $R=\max(70, 0.4N)$ and $\gamma = \lambda/h$ at $\lambda=0.25$; 2D uses the expE01 Radon tensor ridges ($\sqrt W$ directions $\times$ $\sqrt W$ offsets, collar 1.25 disk / 1.6 square, $\gamma = \lambda/h_\text{ref}$, $h_\text{ref}=2.8/\sqrt W$). For a linear operator $L=\sum_i s_i D_i$ the PDE rows are $(L\Phi)_{jm} = \sum_i s_i(p_j)\,\gamma^{o_i}\pi_i(w_m)\,\psi^{(o_i)}(Z_{jm})$ with every tanh derivative $\psi^{(o)}$ a closed-form polynomial in $\tanh$; condition rows are dictionary (or derivative-dictionary) rows at boundary/initial points. One stacked min-norm lstsq (`rcond=1e-13`) per cell; PDE block scaled to $O(1)$ by its max entry, each condition block weighted $\sqrt{n_\text{pde}/n_\text{block}}$.

The nine problems (operator, condition type, exact solution used to manufacture $f$ and condition values):

| category | order | problem | conditions |
|---|---|---|---|
| 1D ODE | 1 | $u' + (2+\sin\pi x)u = f$ (variable coeff.) | IVP $u(-1)$ |
| 1D ODE | 2 | $u'' + 0.4u' + 100u = f$ (damped oscillator) | $u(-1)$, $u'(1)$ |
| 1D ODE | 3 | $u''' + 4u' = f$ (steady dispersion) | $u(-1), u'(-1), u(1)$ |
| 2D steady | 1 | $b\cdot\nabla u + u = f$ (transport + decay) | Dirichlet on inflow arc only |
| 2D steady | 2 | $-\Delta u + 4u = f$ (screened Poisson) | Dirichlet on circle |
| 2D steady | 3 | $u_{xxx} + u_{yyy} + u = f$ (stress test) | Cauchy ($u$, $\partial u/\partial n$) on circle |
| space-time | 1 | $u_t + u_x = 0$ (traveling pulse) | IC + inflow |
| space-time | 2 | $u_t = 0.15\,u_{xx}$ (two decaying modes) | IC + Dirichlet |
| space-time | 3 | $u_t + u_{xxx} = 0$ (two dispersive modes) | IC + $u(\pm1,t)$ + $u_x(-1,t)$ |

Space-time problems are posed on $(x,\tau)$ with $\tau = 2t-1$ (so $\partial_t = 2\partial_\tau$) and solved as anisotropic 2D problems on the square; the three space-time exact solutions are classical (forcing $\equiv 0$), so those solves are driven entirely by the IC/BC rows. Sweeps: 1D $N\in\{8,\dots,1024\}$ at fixed $\lambda=0.25$; 2D $W\in\{144,\dots,2304\}$ (time category to 4096) with $\lambda$ swept on a 7-point grid at anchor width 1024 per problem, then a 3-point local refinement around that optimum at every width (best kept per cell). Collocation: $4W$ uniform (1D) / $5W$ area-uniform random, single seed (2D); eval on grids strictly finer than the collocation spacing (1D $3\times$; 2D $241^2$). Metrics: rel $L_2 = \|\hat u - u^*\|_2/\|u^*\|_2$ and absolute $L_\infty$ on the eval grid. Every hand-coded forcing, derivative, and condition value is verified against finite differences at startup (`problems.verify_all`); tanh derivative formulas are separately FD-verified.

**Code & data.** `experiments/expF01_linear_de_zoo/` (`run.py`, `problems.py`). Data: `data.json`. Figures: `error_vs_width.png` (deliverable), `function_representations/{ode1d,pde2d_steady}/order{1,2,3}.png`, `function_representations/pde2d_time/order{1,2,3}.gif`.

## Results

- **1D: at the floor almost immediately.** All three ODEs reach $10^{-14}$--$10^{-15}$ by $W\approx160$--$205$; the mixed Dirichlet+Neumann and the 3-condition third-order problem behave identically to the IVP. The order-3 curve then drifts up to $\sim10^{-11}$ at $W=1845$: the $\gamma^3$ amplification of fp64 roundoff in the PDE block (order-2 shows a milder echo, order-1 none).
- **2D steady: all three descend steeply to the floor.** At $W=2304$: transport $8.3\times10^{-14}$, screened Poisson $9.2\times10^{-15}$, third-order $1.2\times10^{-13}$ (rel $L_2$). Inflow-only boundary data and Cauchy data both work; best $\lambda$ sits at $0.20$--$0.29$ and drifts down with width, as in expE01. The error field is spatially uniform across the disk (no boundary/interior structure) on the resolved problems.
- **Space-time: advection and heat reach the floor; Airy does not.** Advection descends $6\times10^{-2}\to3\times10^{-13}$ and heat $2\times10^{-3}\to1\times10^{-13}$ over the sweep. Airy stalls at rel $L_2\approx1.6$--$3.5\times10^{-8}$ from $W=576$ onward ($L_\infty\sim5\times10^{-7}$), non-monotone in width. Its residual-vs-$x$ trace is smooth and boundary-pinned: $\sim10^{-12}$ at $x=\pm1$ where data lives, $\sim10^{-9}$ mid-domain. The lstsq residual is at its floor while the solution error is not, i.e. the operator+BC system controls the interior weakly -- consistent with the known delicacy of boundary-value dispersion (which boundary conditions make Airy well-posed depends on the direction of dispersion), not with under-resolution.

### Figures

- **`error_vs_width.png`** (deliverable) -- $3\times3$ grid, rows = operator order, columns = {1D ODE, 2D steady, space-time}; each panel rel $L_2$ (solid) and $L_\infty$ (dashed) vs total width, log-log, with the $10^{-13}$ reference line. Read for: floors in columns 1--2 and the top of column 3; the order-3 1D upward drift; the Airy plateau (bottom right).
- **`function_representations/ode1d/order{1,2,3}.png`** -- exact vs solved overlay (left) and $|\hat u - u^*|$ on log scale (right) at each problem's best cell.
- **`function_representations/pde2d_steady/order{1,2,3}.png`** -- 3D surface of $u^*$ (left) and $\log_{10}|\hat u - u^*|$ heatmap on the disk (right). The order-2 map is uniform at $\sim10^{-14.5}$.
- **`function_representations/pde2d_time/order{1,2,3}.gif`** -- $u(x,t)$ animated over $t\in[0,1]$ (exact solid, solved dashed, locked axes) with the log-scale error profile beside it. order1: the pulse crosses the domain at error $\sim10^{-13}$; order2: two heat modes decay; order3: the dispersive case, where the error panel shows the boundary-pinned interior plateau.

## Additional details

- Collocation randomness is a single seed; the 2D numbers move by well under a decade across re-draws in spot checks, but no formal seed study was run.
- The anchored-$\lambda$ policy (full sweep only at $W=1024$, $\pm0.04$ refinement elsewhere) matches the full-sweep optimum wherever both were computed; the $\lambda$ basin is wide, as in expC03/expE01.
- Runtime: 126 solves + figures + gifs $\approx$ 7 min single-machine CPU. The largest single solve ($W=4096$: a $20{,}490\times4106$ lstsq) takes $\sim$40 s.

## Conclusions

*Proposed, pending Sam.* The frozen QI geometry + one linear solve handles practical linear differential problems -- variable coefficients, all tested condition types, orders 1--3, in 1D and 2D including space-time -- at or near the interpolation fp64 floor, with no training. The two deviations are numerical-analysis phenomena, not method failures: $\gamma^r$ roundoff growth at large width for high-order operators, and a residual-to-error stability gap for boundary-value dispersion (Airy).

## Open questions

- **Airy / dispersive stability gap.** Is the $10^{-8}$ plateau movable by better boundary data (Cauchy on both sides, or the well-posed BC count for the dispersion direction), heavier BC weighting, or is it intrinsic to least-squares collocation of dispersive IBVPs?
- **Width selection for high-order operators.** The order-3 drift implies an optimal finite width; can the a posteriori residual pick it automatically?
- Seed-average the 2D cells (single collocation seed here).
