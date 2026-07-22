# How the high-input-dimension geometry is initialized (expF04)

This explains the QI initialization used in `expF04_qi_init_real_data` (the `all20_2layers` runs and the `all20_2layers_zero_readout` variant): how the repo's 1D quasi-interpolant construction is transported to a hidden layer whose input is a high-dimensional feature vector, not a scalar on $[-1,1]$. The code is `qi_ridge_init_layer_` in `experiments/expF04_qi_init_real_data/model.py`.

## The object we are trying to reproduce

In 1D the construction places a hidden neuron at a center $c_k$ on a grid of spacing $h$ and gives it pre-activation $\gamma\,(x - c_k)$, i.e. weight $w_k = \gamma$ and bias $b_k = -\gamma c_k$, with

$$\gamma = \frac{\lambda}{h}, \qquad \lambda = 0.25 .$$

$\gamma$ is the inner scale (how sharply $\tanh$ turns), and it is tied to the grid: as the width $N$ grows, $h$ shrinks like $1/N$, so $\gamma = O(N)$ and the $N$ centers pack into $[-1,1]$ together. That coupled growth (scale up, centers pack) is what makes the construction reach machine precision. Keep it in mind; it is the single property the high-d lift must not break.

## Why the naive lift fails

A hidden neuron on a $d$-dimensional input has pre-activation $w\cdot x + b$. The obvious lift is to pick a random weight direction $u$ (a unit vector in $\mathbb{R}^d$), set $w = \gamma\,u$, and sweep a scalar center $c_k$ as before, giving $\gamma\,(u\cdot x - c_k)$.

This looks right and is almost right. The failure is one of scale matching. The projection $t = u\cdot x$ has some spread that depends on the data and on $d$, and if the center sweep $\{c_k\}$ does not live on the same range as $t$, the term $\gamma\,c_k$ is swamped by $\gamma\,(u\cdot x)$ and every neuron sees essentially the same saturated input. The 1D grid assumed $x \in [-1,1]$; a raw high-d projection is not on that range, so the sweep and the signal fall out of alignment.

## The fix: ridge bundles

The transport that preserves the coupled-growth property is to sweep along one direction at a time and match the sweep to that direction's own projected data range.

Partition the $N$ hidden neurons into $M$ bundles of $P$ neurons each. In this experiment $P = \lfloor\sqrt{N}\rfloor$, so at width $N=256$ each bundle holds $P = 16$ neurons and there are $M = 16$ bundles. Each bundle is one "ridge": it shares a single random unit direction $u_m$ and carries a full 1D quasi-interpolant sweep along it.

For bundle $m$:

1. Draw a random unit direction $u_m \in \mathbb{R}^d$.
2. Project a sample of the layer's input onto it, $t = x\cdot u_m$, and take a robust half-range $A_m = \operatorname{quantile}(|t|,\,0.999)$. The quantile (rather than the max) keeps a few outlier projections from stretching the range.
3. Lay a 1D sweep of $P$ centers across $[-A_m, A_m]$, spacing $h_m = 2A_m / P$, and set the per-ridge scale

$$\gamma_m = \frac{\lambda}{h_m} = \frac{\lambda\,P}{2A_m}.$$

4. Give every neuron $k$ in the bundle $w_k = \gamma_m\, u_m$ and $b_k = -\gamma_m\, c_k$, so its pre-activation is $\gamma_m\,(u_m\cdot x - c_k)$ with the sweep now on the same range as the signal.

Because the centers are sampled on $[-A_m, A_m]$ and $\gamma_m$ is set from that same $A_m$, the scale and the packing grow together per ridge exactly as they did in 1D: $\gamma_m = O(P)$ while $P$ centers fill the range. This is the property the naive lift destroys and the bundle restores.

## The corner case this avoids

Setting $P = 1$ (every neuron its own direction, $M = N$ bundles) is the degenerate corner: $\gamma$ would still scale like $O(N)$, but with a single center per direction there is nothing packing alongside it, so the coupled-growth property is lost. The bundle size $P = \sqrt{N}$ is the deliberate middle: enough neurons per direction to form a real sweep, enough directions to cover the input space. (At the opposite corner, $d=1$ and $P=N$, one bundle recovers the original 1D construction exactly, with $\gamma = \lambda / (2/N) = 0.125\,N$.)

## The "scaled" variant actually used

`qi_ridge_init_layer_` takes a flag `uniform_centers`. The experiment calls it with `uniform_centers=False` (the "scaled" variant): instead of a uniform grid on $[-A_m, A_m]$, the $P$ centers are sampled directly from the observed projections $t = x\cdot u_m$. Directions and the $\gamma_m$ scale are identical to the uniform version; only the exact center placement differs. This isolates "getting the scale right" from "getting the exact sweep right," and the scaled placement won in the earlier `ridge_real` comparison, so it is the scheme carried into the two-layer runs.

## How the two-layer configs apply it

The model is `SimpleMLP2`: `d_in -> 256 -> 256 -> d_out`, with `fc1`, `fc2` the two hidden layers and `fc3` the readout.

- **qi1** applies the ridge init to `fc1` only. Its input is the raw feature vector, so the projections in step 2 use the data directly.
- **qi2** applies it to `fc1` and `fc2`. For `fc2` the "input" is the post-`fc1` hidden activations $\sigma(\mathrm{fc1}(x))$ on a sample batch, so `fc2` gets fresh random directions that tile the hidden representation and centers matched to *its* projected range. Everything downstream of the projection is identical; only what feeds the projection changes.
- **baseline** leaves all three layers at PyTorch defaults.

In the `all20_2layers_zero_readout` variant this init is unchanged; the only difference is that after the chosen scheme sets `fc1`/`fc2`, the readout `fc3` (weight and bias) is zeroed so the network output starts at exactly $0$.

## Where to look

- `experiments/expF04_qi_init_real_data/model.py` — `qi_ridge_init_layer_` (the ridge init) and `SimpleMLP2` (the architecture).
- `experiments/expF04_qi_init_real_data/all20_2layers/run.py` — how `qi1`/`qi2` call the init.
- `experiments/expF04_qi_init_real_data/all20_2layers_zero_readout/run.py` — same init, readout zeroed.
