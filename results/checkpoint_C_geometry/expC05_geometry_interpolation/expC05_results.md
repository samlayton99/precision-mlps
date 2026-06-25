# expC05 -- Geometry interpolation: what makes the QI geometry findable

**Status: draft -- IN REVISION. The RMS weights rerun fixed the scale/center confound, but a *second* artifact was then found: linearly interpolating the weight direction drags the ~half of Xavier weights that are negative through zero bandwidth, clustered at s~0.5 (a rank-collapse "pinch"). For tanh the sign is free, so the fix is to interpolate magnitudes with signs fixed and normalize by mean-abs (hold the total bandwidth budget). A de-confounded rerun is pending; only the endpoints (s=0 spread vs s=1 uniform) are currently trustworthy. weight+bias mode also pending. Conclusions pending Sam.**

## TL;DR

- Ranking the ingredients at the viable bandwidth: $\lambda$ is first-order (spans 10--13 decades); **center uniformity is soft** (~1--2 decades); **per-neuron bandwidth uniformity is hard** (2--6 decades, less for exp).
- The weights-mode finding is now clean: with centers pinned and the bandwidth scale held at $\gamma$ (rms), a non-uniform (Xavier) per-neuron bandwidth is 2--6 decades worse than all-equal-$\gamma$ and improves *monotonically* toward uniform -- so uniform bandwidth is a near-hard requirement, decoupled from scale and center placement.
- **Weight+bias (provisional):** in raw $(w,b)$ coordinates weight and bias must reach $\gamma$-scale *together* (a diagonal valley; off-diagonal the centers $-b/w$ collapse and the basis loses rank). This argues for the $\gamma(x-c)$/log-$\gamma$ reparameterization (expD03) -- but this mode still has the scale confound and is being rerun.

## Question

Between a random init and the QI geometry, which ingredient matters how much -- $\lambda$, center uniformity, per-neuron bandwidth uniformity -- and (weight+bias) are the weight and bias requirements coupled?

## Experiment design

The ideal QI inner layer is $\gamma(x-c_i)$: every weight $=\gamma$, bias $=-\gamma c_i$, centers on the uniform grid. From a genuine Glorot/Xavier init, interpolate toward the ideal along three modes, lstsq fp64 at each grid point:

- **centers:** weight fixed at $\gamma$; interpolate positions $c(t)=(1-t)c^\text{xav}+t\,c^\text{unif}$; sweep $\gamma=\texttt{logspace}(0.25,N)$. Axes $(t,\lambda)$.
- **weights (de-confounded):** centers pinned to the uniform grid and the bandwidth *scale* held at $\gamma$; interpolate only the weight *direction*. With $\hat w=w^\text{xav}/\mathrm{rms}(w^\text{xav})$ and $u(s)=(1-s)\hat w+s\mathbf 1$, set $w(s)=\gamma\,u(s)/\mathrm{rms}(u(s))$ and $b(s)=-w(s)\,c^\text{unif}$. Then $\mathrm{rms}(w)=\gamma$ and center $-b/w=c^\text{unif}$ hold *exactly* for all $s$ (asserted at runtime: both drifts $<10^{-15}$). Sweep $\gamma$; axes $(s,\lambda)$ with $\lambda=\mathrm{rms}(w)\,h$. Single seed.
- **weight+bias:** interpolate both raw parameters, $w(s)=(1-s)w^\text{xav}+s\gamma$ and $b(t)=(1-t)b^\text{xav}+t(-\gamma c^\text{unif})$, $\gamma$ fixed. Axes $(s,t)$; $(0,0)$=Xavier, $(1,1)$=QI. *(Straight-line raw interpolation: changes magnitude as well as pattern -- the confound below; a de-confounded rerun is pending.)*

4 targets, 4 widths ($W\in\{205,\dots,921\}$), metric eval rel $L_2$.

**Code & data.** `experiments/expC05_geometry_interpolation/` (`common.py`, `run_centers.py`, `run_weights.py`, `run_weightbias.py`). Data: `{centers,weights,weightbias}/data.json`. Figures: `<mode>/seed_1/interp_<target>.png`, `weightbias/lambda_sweeps.png`. Notebook: `interp_viz.ipynb`.

## Results

- **$\lambda$ dominates** (all modes): sweeping $\lambda$ moves the error 10--14 decades; geometry effects at the best $\lambda$ are second-order. With ideal weights, Xavier centers cost only ~1--2 decades -- **center uniformity is soft**.
- **Per-neuron bandwidth uniformity (endpoints only -- mid-path contaminated).** At the viable $\lambda$, the Xavier-spread endpoint ($s{=}0$) is 2--6 decades worse than the uniform endpoint ($s{=}1$); e.g. sine $N{=}128$: $s{=}0\sim8\times10^{-11}$ vs $s{=}1\sim6\times10^{-14}$. **But the path between them is NOT monotonic:** it dips into a rank-collapse "pinch" at $s{\sim}0.5$ where the negative-Xavier-weight cohort crosses zero bandwidth (verified: effective rank falls ~15%, the rms-renorm boosts survivors to ~$2\gamma$). That mid-path structure is a sign-crossing artifact of the direction interpolation, not a fact about uniformity -- which is why the mean-abs / magnitude-only rerun (status) is needed before claiming anything beyond the endpoints.
- **Weight+bias diagonal valley (provisional).** Only the joint corner reaches the floor; moving weights ahead of biases is worse than the Xavier start -- a diagonal valley. Mechanism: $-b/w$ collapses off-diagonal and the basis loses rank (verified: error tracks numerical rank, an independent lstsq reproduces it, target-independent). Caveat: this mode still confounds magnitude with pattern (below).

### Figures

- **`centers/seed_1/interp_<target>.png`** -- 4x3 grid (rows = width): col 1 heatmap (read the horizontal $\lambda$-band; the Xavier-center edge is only mildly bright -- soft), col 2 error-vs-$\lambda$ U-curves, col 3 error-vs-$t$ slices.
- **`weights/seed_1/interp_<target>.png`** -- same 4x3 layout, x = weight uniformness $s$, y = $\lambda=\mathrm{rms}(w)\,h$. The $s$ axis is now a clean pattern axis (scale and centers fixed): read the strong, monotonic darkening toward $s{=}1$ at the viable $\lambda$ -- non-uniform bandwidth is bright (bad), uniform sits at the floor.
- **`weightbias/seed_1/interp_<target>.png`** -- axes weight $s$ vs bias $t$; read the diagonal valley and the bright $s>t$ triangle. *(Provisional -- scale-confounded.)*
- **`weightbias/lambda_sweeps.png`** -- ideal-geometry U-curves with $\lambda^*$ marked (smooth targets flat-bottomed; only sine_8pi sharp).

## Additional details

- **Weights mode is now de-confounded.** The earlier $(1-s)w^\text{xav}+s\gamma$ interpolation changed the weight magnitude (hence the effective bandwidth) from ~0.1 to $\gamma$ *and* moved the centers via $-b/w$, so the $s$ axis was not orthogonal to $\lambda$. The rms-normalized, center-pinned construction fixes both exactly, so the weights-mode result is a pure weight-pattern effect. (RMS chosen as the scale: a vector's canonical magnitude is $\|w\|_2$, it is the native norm of the lstsq readout, and it is robust to the sign zero-crossings that make the log-natural geometric mean unusable here.)
- **Weight+bias still carries the confound:** its straight-line raw interpolation changes magnitude as well as pattern, so the diagonal-valley reading is partly scale-balance. A de-confounded rerun (same RMS / center-pinning idea on both axes) is pending; treat that conclusion as provisional until then.
- **runge lead:** runge does slightly better off-uniform in the centers mode (at $N=64$, ~$7\times10^{-12}$ vs the uniform grid's ~$3\times10^{-9}$), consistent with curvature-clustering -- but incidental to this seed; on hold until a deterministic test.

## Conclusions

*Proposed, pending Sam.* At the viable bandwidth the geometry requirements are asymmetric: **center placement is forgiving (~1--2 decades), but per-neuron bandwidth uniformity is strict (2--6 decades)** -- both now cleanly separated from the bandwidth scale (weights mode, rms-normalized, centers pinned). The weight+bias coupling (diagonal valley -> argument for reparameterization) is provisional pending its own de-confounded rerun.

## Open questions

- **De-confound the weight+bias mode** (next): rerun with the magnitude held and centers pinned, to see whether the diagonal valley survives once scale is removed.
- **Reparameterization test:** does $\gamma(x-c)$/log-$\gamma$ remove the coupling barrier? (expD03.)
- **Curvature-clustering** (on hold): a deterministic test of whether clustering centers at high curvature beats the uniform grid.
