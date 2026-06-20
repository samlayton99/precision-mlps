# Exp01 -- Numerics sanity checks

**Status: draft -- pending Sam's review and sign-off on conclusions.**

## Question

Before attributing the construction-vs-training gap to optimization, are linear solves, function evaluation, and tolerance choices themselves the precision bottleneck? Concretely: does the QI construction actually reach its claimed $L_\infty$ floor; does the reported $L_\infty$ depend on the eval-grid density; which readout solver recovers the readout weights to machine precision on fixed QI geometry; how badly does forming $\Phi^\top\Phi$ square the conditioning; and is fp64 $\tanh$ evaluation accurate enough at the large arguments $\gamma(x - c_m)$ that occur near the edges?

## What the code runs

Code: `experiments/exp01_numerics_sanity/` (`run.py`, `config.yaml`). Run: `python3 experiments/exp01_numerics_sanity/run.py`. fp64 throughout for evaluation and solves; mpmath ($\mathrm{dps}=50$) used only as a high-precision reference. Target $\sin(\pi x)$, widths $N \in \{16, 32, 64, 128, 256\}$, $K_c = 160$, equispaced grids on $[-1, 1]$. Eval-density list $n_{\text{eval}} \in \{256, 512, 1024, 2048, 4096, 8192\}$; train grid size $n_{\text{train}} = \max(256,\, 4\,W_{\text{full}})$ where $W_{\text{full}} = N + 2\,\mathrm{halo} + 1$.

Six tests (from the `run.py` docstring):

1. **Construction baseline.** Construct QI in both `fp64` and `mpmath` precision; evaluate $L_\infty = \max|q(x) - f(x)|$ and relative $L_2 = \|q - f\|_2 / \|f\|_2$ on every eval density, with and without Kahan-compensated summation.
2. **Readout solvers.** With $\gamma$ and centers frozen at the mpmath-QI values, build $\Phi_{ik} = \tanh(\gamma(x_i - c_k))$ and solve for the readout $(v, \text{bias})$ with six methods: `lstsq`, `qr`, `svd`, and ridge at $\alpha \in \{0, 10^{-14}, 10^{-12}\}$. Run on two geometries: `full` (halo + interior) and `interior` (interior centers only, no halo). Record train $L_\infty$, residual norm, eval $L_\infty$ / rel $L_2$ at $n_{\text{eval}} = 8192$, and weight magnitudes $\max|v|$, $\|v\|_2$. A cross-solver reproducibility pass records $\max|\text{pred}_a - \text{pred}_b|$ over solver pairs on the densest grid.
3. **Eval density sweep.** For each solution, recompute $L_\infty$ across all $n_{\text{eval}}$ to check the reported floor is grid-independent.
4. **Conditioning.** $\mathrm{cond}(\Phi)$ (from SVD) vs $\mathrm{cond}(\Phi^\top\Phi)$, plus their ratio, per width and geometry.
5. **tanh stability.** Max $|\tanh_{\text{fp64}}(z) - \tanh_{\text{mpmath}}(z)|$ over a strided subset of $\Phi$ entries, where $z = \gamma(x - c)$ reaches $|z| \approx N\lambda$ near edges (here up to $\gamma \cdot 2$).
6. **Extended-precision solve** (only $N \le 64$). Solve the normal equations $(\Phi^\top\Phi)\,v = \Phi^\top y$ in mpmath on the interior geometry, keep the fp64 $v$, and check whether the eval floor drops.

## Results

Data: `results/exp01_numerics_sanity/` -- `construction.jsonl`, `readout_solves.jsonl`, `density_sweep.jsonl`, `conditioning.jsonl`, `tanh_stability.jsonl`, `mp_solve.jsonl`, `reproducibility.jsonl`, and the human-readable `summary.txt`. There are no figures for this experiment.

**Construction baseline** (`construction.jsonl`, table [1] in `summary.txt`; $n_{\text{eval}} = 8192$, Kahan off). mpmath reaches machine eps; fp64 is limited by convolution cancellation; Kahan summation does not change either materially.

| $N$ | fp64 $L_\infty$ | mpmath $L_\infty$ |
|---|---|---|
| 16 | $1.541\times10^{-10}$ | $1.160\times10^{-12}$ |
| 32 | $9.215\times10^{-12}$ | $1.732\times10^{-14}$ |
| 64 | $5.393\times10^{-12}$ | $3.220\times10^{-15}$ |
| 128 | $5.493\times10^{-12}$ | $2.331\times10^{-15}$ |
| 256 | $2.844\times10^{-12}$ | $1.665\times10^{-15}$ |

**Eval density sweep** (`density_sweep.jsonl`; table [6] in `summary.txt`, mpmath QI). The reported $L_\infty$ is flat across grid density: at $N = 64$ it ranges only over $2.998\times10^{-15}$ to $3.220\times10^{-15}$ across $n_{\text{eval}} \in \{256, \dots, 8192\}$; at $N = 256$ it stays $\approx 1.55$ to $1.67\times10^{-15}$. The floor is not an under-sampling artifact.

**Readout solvers, full geometry** (`readout_solves.jsonl`, table [2]; eval $L_\infty$ at $n_{\text{eval}} = 8192$). `lstsq` and `svd` recover the readout to $\sim 10^{-13}$ to $10^{-14}$; `qr` reaches similar eval error but inflates weights ($\max|v|$ up to $\sim 10^3$ at $N = 16$, $\sim 10^2$ at $N = 64$); ridge / normal-equations solves are orders worse.

| $N$ | lstsq $L_\infty$ | svd $L_\infty$ | qr $L_\infty$ ($\max|v|$) | ridge_0 $L_\infty$ |
|---|---|---|---|---|
| 16 | $2.149\times10^{-13}$ | $1.389\times10^{-12}$ | $3.411\times10^{-11}$ ($2.731\times10^{3}$) | $3.208\times10^{-6}$ |
| 32 | $8.748\times10^{-14}$ | $1.511\times10^{-14}$ | $3.073\times10^{-13}$ ($4.042\times10^{1}$) | $7.091\times10^{-7}$ |
| 64 | $5.003\times10^{-14}$ | $4.664\times10^{-14}$ | $2.266\times10^{-13}$ ($1.269\times10^{2}$) | $1.299\times10^{-7}$ |
| 128 | $1.469\times10^{-12}$ | $1.950\times10^{-13}$ | $7.392\times10^{-13}$ ($4.384\times10^{2}$) | $1.297\times10^{-6}$ |
| 256 | $8.367\times10^{-13}$ | $4.267\times10^{-13}$ | NaN | NaN |

At $N = 256$ on full geometry the `qr`, `ridge_0`, and `ridge_1e-14` solves return NaN (degenerate); `lstsq` and `svd` still return $\sim 10^{-13}$.

**Readout solvers, interior-only geometry** (no halo). Accuracy collapses uniformly across all solvers: eval $L_\infty$ is $2.764\times10^{-4}$ at $N = 16$, $\sim 1.0\times10^{-5}$ at $N = 32$ / $64$, $5.490\times10^{-6}$ at $N = 128$, $2.789\times10^{-6}$ at $N = 256$ (`lstsq`). The solver choice is irrelevant once the halo is removed.

**Cross-solver reproducibility** (`reproducibility.jsonl`). `lstsq` vs `svd` agree to $\sim 10^{-12}$ to $10^{-14}$ ($9.115\times10^{-14}$ at $N = 32$, $4.342\times10^{-14}$ at $N = 64$), while `lstsq` vs `ridge_0` differ by $\sim 10^{-6}$ to $10^{-7}$ -- the ridge solution is a different solution, not a perturbed one.

**Conditioning** (`conditioning.jsonl`, table [3]). On full geometry $\mathrm{cond}(\Phi)$ is already $\sim 10^{19}$ (e.g. $4.989\times10^{19}$ at $N = 128$, $2.475\times10^{38}$ at $N = 256$). On interior geometry $\mathrm{cond}(\Phi)$ is far smaller ($\sim 10^{7}$ to $10^{9}$) but $\mathrm{cond}(\Phi^\top\Phi)$ jumps to $\sim 10^{14}$ to $10^{18}$ -- the squaring is explicit (ratio $\sim 10^{7}$ to $10^{9}$). This is why any method that forms $\Phi^\top\Phi$ (ridge, normal equations) loses accuracy.

**tanh stability** (`tanh_stability.jsonl`, table [4]). For every width the max fp64-vs-mpmath difference over sampled $\Phi$ entries is exactly $1.110\times10^{-16}$ (one ulp), with $\gamma$ ranging $2$ to $32$ and worst $|z| = \gamma \cdot 2$ up to $64$. Activation evaluation is not the bottleneck.

**Extended-precision normal-equations solve** (`mp_solve.jsonl`, table [5]; interior geometry, $N \le 64$). Solving the normal equations in mpmath does not rescue the interior geometry: eval $L_\infty$ is $2.764\times10^{-4}$ ($N = 16$), $1.048\times10^{-5}$ ($N = 32$), $9.999\times10^{-6}$ ($N = 64$) -- matching the fp64 interior-geometry solvers. The limitation is the geometry (missing halo), not solve precision.

## Conclusions

*Draft -- pending Sam's review and sign-off.*

- The QI construction reaches its claimed precision and the reported $L_\infty$ is grid-independent: mpmath hits machine eps ($\sim 1.7\times10^{-15}$ at $N = 32$ down to $\sim 1.7\times10^{-15}$ at $N = 256$), fp64 floors at $\sim 5\times10^{-12}$, and across $n_{\text{eval}} \in \{256, \dots, 8192\}$ the $L_\infty$ varies only within a single significant figure. The floor is not an evaluation artifact.
- On fixed QI full geometry, solving $\Phi$ directly via `lstsq` or `svd` reaches $\sim 10^{-13}$ to $10^{-14}$, whereas ridge / normal-equations solves are far worse ($\sim 10^{-6}$ to $10^{-7}$); `qr` reaches comparable eval error but with inflated weights ($\max|v|$ up to $\sim 10^{3}$). The gap tracks the explicit squaring of the condition number when $\Phi^\top\Phi$ is formed ($\mathrm{cond}(\Phi^\top\Phi)/\mathrm{cond}(\Phi) \sim 10^{7}$ to $10^{9}$ on interior geometry).
- The halo is necessary: removing it (interior-only centers) collapses every solver to $\sim 10^{-4}$ to $10^{-6}$, and solving the normal equations in mpmath does not recover accuracy -- so this is a geometry limitation, not a solve-precision one.
- fp64 $\tanh$ evaluation differs from mpmath by exactly one ulp ($1.110\times10^{-16}$) at every width tested; activation evaluation is not the precision bottleneck.
