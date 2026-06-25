# expC03 -- The lambda basin: center and width of the optimal bandwidth

**Status: conclusions approved by Sam.**

## TL;DR

- On the ideal geometry the optimum is $\lambda^*\approx0.25$, essentially constant in width -- a flat basin, runge included.
- The basin is wide, so the inner weight/bias magnitude is $\gamma^*=\lambda^* N/2$, i.e. $\gamma^*/N\approx0.10$. This is the robust answer to "what magnitude should the inner weights be."
- **Tangent (Sam, open):** at large $N$ a faint second near-floor region appears near $\lambda\approx0.05$ (for runge it even edges out $0.25$).

## Question

On the ideal uniform geometry, what bandwidth is optimal vs width, and how wide is the basin around it?

## Experiment design

For each (target, width) on the uniform grid + halo ($h=2/N$, halo $=\texttt{default\_halo}(N,0.25)$, centers $-1+nh$), sweep $\lambda\in[0.02,0.60]$ at step $0.005$ (117 points), set $\gamma=\lambda/h$ for all neurons, fit the readout by fp64 lstsq with a bias column on $\max(2048,4W)$ points, and evaluate on 8001 points. Every neuron is $\tanh(\gamma(x-c))$ with shared $\gamma$, so the inner-weight magnitude is $\gamma=\lambda N/2$ and the question reduces to the dimensionless optimum $\lambda^*$, from which $\gamma^*=\lambda^* N/2$ follows. The basin center/width is extracted from the rel-$L_2$ curve by a locked estimator (**M2 + mode seeking**): a tent-weighted posterior around the robust floor (weight $w(\lambda)=\max(0,\,1-(\log_{10}E-E_0)/D)$, $D=2$ decades) gives a center and $\pm$range, then the center is snapped to the actual dip within the inner band. A cell is *unresolved* if its best rel $L_2>10^{-10}$. Grid: 6 targets x widths $\{32,\dots,1024\}$.

**Code & data.** `experiments/expC03_lambda_basin/` (`run.py`; `--full`, estimator comparison via `--check`). Data: `full_sweep.json`, `basin_method_check.json`. Figures: `lambda_vs_size.png` (deliverable), `basin_grid_m2ms.png`, `basin_check_{m1,m2,m2ms}.png`.

## Results

- **Flat band near $\lambda^*\approx0.25$, width-independent.** Every resolved cell's center lands in $\approx[0.14,0.27]$ with no strong width trend; $\lambda=0.25$ is near-optimal across the family. Resolved floors are machine-limited.
- **runge looks like the exception but isn't:** its center drifts down ($\approx0.22\to0.14$) but the curve is shallow there, so $\lambda=0.25$ stays near-floor -- wander in a flat basin, not a real law.
- **Roughness sets a resolution wall:** `abs_cubed` resolves only at $N=1024$; the rougher targets are unresolved at $N=32$. Wherever a target resolves, it lands in the same band.

### Figures

- **`lambda_vs_size.png`** (deliverable) -- 6 rows (targets) x 2 cols (vs $N$, vs $W$); y = $\lambda^*$ on a fixed $[0,0.45]$ axis, green line = center, green ribbon = basin range, dashed reference at $0.25$. A flat green line = width-independent bandwidth; the ribbon shows how wide the near-optimal band is.
- **`basin_grid_m2ms.png`** -- error vs $\lambda$ per (target, width) with the basin band shaded; check the band sits on the bottom of each U.
- **`basin_check_{m1,m2,m2ms}.png`** -- the estimator comparison that locked M2 + mode-seeking.

## Conclusions

The optimal bandwidth is a flat band at $\lambda^*\approx0.25$, constant in width (runge included), giving $\gamma^*/N\approx0.10$ as the precision-admitting inner-weight magnitude. $\lambda=0.25$ is the bandwidth of record. (Approved by Sam.)

## Open questions

- **The second mode near $\lambda\approx0.05$** (Sam): as $N$ grows a faint second near-floor region appears at small $\lambda$. Aliasing, or does width make the small-bandwidth regime attainable? Worth checking whether scaling keeps opening it.
