# expG01 -- interactive geometry / generalization explorer

**Status:** live tool (not a batch experiment). First of the Checkpoint G interactive explorers.

## What it is

A local Dash web app for building intuition about how the fixed-geometry + SVD-min-norm readout resolves a target, and how it generalizes into held-out gaps. Set the bandwidth, width, target, sampling, and a slide-able hold-out mask; every change re-solves the readout on the unmasked points and redraws. It is the interactive companion to the Checkpoint C "precision vs generalization (mask the data)" question.

## Run it

```
~/.venvs/precisionMLPs/bin/python experiments/expG01_interactive_explorer/run.py
```

Then open **http://127.0.0.1:8050** in a browser. (Requires `dash` in the venv; already installed.)

## Controls (each is a slider + a linked text box)

- **solve** -- least squares (SVD min-norm readout on the training points) or QI (closed-form quasi-interpolant on the full grid, tanh only). QI ignores the data mask, center mask, and #train samples -- it needs the whole grid -- so it acts as a full-information reference.
- **λ** -- bandwidth; $\gamma=\lambda/h$, $h=2/N$.
- **N** -- grid width. The halo is sized as in the batch experiments (`default_halo(N, lambda_star=0.25)`); total neurons $W=N+2\,\text{halo}+1$ is shown in the footer.
- **# train samples** -- equispaced points on $[-1,1]$ used for the fit.
- **# test samples** -- equispaced test grid; the count is rounded up to the nearest prime so it does not align with the train grid (footer shows the value used).
- **activation** -- tanh (the machine-precision case), gelu, relu, swish, or sigmoid; the feature matrix uses the chosen activation.
- **halo** -- a "default" toggle (on -> auto-fills `default_halo(N, 0.25)` and locks the box, updating with $N$) plus a manual box (ghost nodes per side) when unchecked. Shrinking it makes the boundary error climb -- e.g. sine at $N=128$ stays at the floor for halo $\ge 5$ but jumps to $\sim 10^{-7}$ at halo $=1$.
- **y-noise** -- a toggle + log-scale slider ($\sigma=10^k$) that adds Gaussian noise to the training targets (least-squares only, fixed seed). This exposes the conditioning: the masked-gap reconstruction is numerical analytic continuation, which amplifies error by $\sim 10^5$. Noise-free the gap sits at $\sim 10^{-4}$; $\sigma=10^{-6}$ pushes it to $\sim 10^{-1}$; $\sigma=10^{-4}$ destroys it -- the noise-free assumption is load-bearing (cf. expB01).
- **function** $f(x)$ -- arbitrary text expression, parsed with sympy (e.g. `sin(2*pi*x)`, `1/(1+25*x**2)`, `exp(x)`); a preset dropdown fills common targets.
- **mask** -- toggle on/off + a range slider for the held-out interval $[a,b]$; when on, train points inside $[a,b]$ are dropped from the fit. A "drop centers too" toggle additionally removes the neurons centered inside $[a,b]$ from the geometry (least-squares only).

A note on the gap: with least squares, the min-norm SVD readout fills a masked gap with a straight line between the endpoints. With $\gamma=O(N)$ each tanh is a sharp step of height $2v_k$ at its center; the net rise across the gap is fixed by the boundary values, and minimizing $\|v\|_2$ for a fixed sum makes every $v_k$ equal -- equal steps at equal spacing, i.e. a linear ramp.

## Panels

- **Left** -- target $f$ (blue) vs approximation (red, thinner, semi-transparent); the mask interval is shaded. With y-noise on, the scheme switches to: true $f$ dotted black, approximation red, and the noisy training data blue.
- **Middle** -- signed residual $f-\hat f$ vs $x$ on a symmetric-log $y$-axis (exponent tick labels; a fixed zero line dead-center, mirrored above/below), with red dotted lines at the max and min residual (the $L_\infty$ envelope); mask shaded.
- **Right** -- 3x2 table of rel $L_2$ and $L_\infty$ over the test range, for: **entire** range, **unmasked** (where data is present), **masked** (held-out gap).

## Solver

Reuses `src/construction`: center lattice + `default_halo`, per-center $\gamma$, and `solve_readout_with_bias(..., method="svd")` (truncated-SVD min-norm). The feature matrix is built for the selected activation (tanh reproduces `src.readout.build_phi`). Same fp64 path as expC04.

Sanity checks (headless): sine at $\lambda=0.25$, $N=128$, no mask reaches rel $L_2\approx3.6\times10^{-14}$ (the fp64 floor). Runge with a $[-0.3,0.3]$ mask gives unmasked rel $L_2\approx2.3\times10^{-14}$ but masked $\approx1.9\times10^{-1}$ -- the single-scale geometry cannot reconstruct the held-out peak.

## Code & data

- App: `experiments/expG01_interactive_explorer/app.py` (solver + layout + callbacks)
- Launcher: `experiments/expG01_interactive_explorer/run.py`
