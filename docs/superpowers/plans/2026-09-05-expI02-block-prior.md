# expI02 block prior -- implementation plan

> **For agentic workers:** executed inline in this session (superpowers:executing-plans); the repo's CLAUDE.md forbids subagents unless asked. Steps use checkbox syntax.

**Goal:** build the corrected QI block family (fixed mesh, free projection, coefficient coordinates, calibration + band penalty) and run the seven studies of the spec, producing six figures and a writeup that states a prior.

**Architecture:** one experiment folder with four focused files: `qi2.py` (models, optimizers, fit loop, diagnostic), `tasks.py` (data, targets, PDEs), `plots.py` (six figures from JSON), `run.py` (studies as subcommands, resumable JSON). Solvers and the Gauss-Newton finisher are imported from `experiments/expI01_compositional_qi/qiblocks.py`, which is not edited.

**Tech stack:** torch 2.4 (fp64 CPU for precision studies, fp32 CPU/MPS otherwise), numpy, sklearn (Friedman1, ridge), dysts (Lorenz), matplotlib. Run with `uv run --extra dev python ...`.

**Spec:** `docs/superpowers/specs/2026-09-05-expI02-block-prior-design.md`

## Global constraints

- $\lambda_\star=0.25$; bank centers cell-centered on $[-1,1]$; $\gamma=\lambda_\star/h$; never moved; no forward normalization of any kind.
- Calibration target $s_\star=0.4$; band penalty tolerances $|\mu|\le0.25$, $0.15\le s\le0.6$, excursion $\rho=0.9$; default $\beta=10^{-2}$; whitening cap $\kappa=10^{3}$.
- Readout solve: QR then SVD, rcond $10^{-14}$ (`qiblocks.lstsq_bias`); never normal equations.
- Every comparison: same data, seed, steps, final solve; counts from one function per model; MLP widths from `mlp_width_for`.
- Legends above axes; precision axes fixed to $[10^{-15},10^{1}]$; trajectories recorded at `log_every` steps.
- Results under `results/checkpoint_I_depth_theory/expI02_block_prior/` (JSON per study, `figures/`); writeup `expI02_results.md` there.
- No emojis; no new `.md` beyond the writeup.

---

### Task 1: `qi2.py` core -- bank, coordinates, mixer, layer, stack, MLP, counts

**Files:** create `experiments/expI02_block_prior/qi2.py`; test `tests/test_expI02_block_prior.py`.

**Produces:**
```python
LAM = 0.25
def bank_centers(N: int) -> tuple[torch.Tensor, float]          # (c [N], h)
class Bank(nn.Module):  # buffers c; attrs N, h, gamma; forward(p [B,M]) -> [B,M,N]
def coord_transform(N: int, kind: str, cap: float = 1e3, n_ref: int = 4001) -> torch.Tensor  # P [N,N]; kind in {"raw","val","white","val_white"}
def smooth_profile(N: int, gen: torch.Generator, degree: int = 3) -> torch.Tensor  # raw tanh coefficients w [N] of a random smooth function
class Mixer(nn.Module):  # theta [M,N,K] (rank None) or a [M,S], Phi [S,N,K]; bias [K]; buffer P
    def forward(self, H) -> [B,K]         # einsum over (H @ P) and theta
    def coeffs(self) -> [M,N,K]           # raw C = P theta (rank expanded)
    def readout_view(self) -> (W [M*N,K], b [K])   # rank None only
    def counts(self) -> dict(params, macs)
class QILayer(nn.Module):  # __init__(d_in, d_out, N, M=None, proj: bool, rank=None, coords="raw", cap=1e3, residual=False, seed=0)
    def pre(self, z) -> [B,M]             # bank input: z @ V.T if proj else z
    def forward(self, z) -> [B,d_out]     # residual: z + mixer(bank(pre(z)))
    def hidden(self, z) -> [B,M,N]
class QIStack(nn.Module):  # __init__(layers); forward, feats(x) [B,M_last*N_last], readout(), solve_readout(X,Y,rcond), nonlinear_params(), update_ranges(X) (no-op, for qiblocks.gauss_newton), counts(), pres(x) -> list of [B,M_l] bank inputs
class MLP(nn.Module):      # __init__(d, widths, d_out=1, act="tanh"|"relu", seed); feats, readout, solve_readout, nonlinear_params, update_ranges, counts, pres (empty)
def mlp_width_for(target: int, d: int, key: str, depth: int = 2, d_out: int = 1, act="tanh") -> int
def make_block(d, M, N1, K, N2, d_out=1, rank=None, coords="raw", cap=1e3, depth=2, residual=False, seed=0) -> QIStack
```

- [ ] Write failing tests:
```python
def test_coordinates_are_a_reparametrization():   # same function whichever P; "val" equals w_j=(a_j-a_{j-1})/2
def test_whitener_caps_the_spectrum():           # eig of Gram(B_ref P) == min(1,(cap*s/s_max)^2) to 1e-8
def test_one_layer_hits_1d_floor():              # N=128, solved readout on sin 2 pi x: rel L2 < 1e-11
def test_smooth_init_identical_across_coords():  # make_block(...coords=k) for all k give the same initial function to 1e-10
def test_counts_and_width_matching():            # mlp_width_for returns w with params >= target, w-1 below
def test_residual_layer_is_identity_plus_delta()
```
- [ ] Run, confirm failures (module missing).
- [ ] Implement `qi2.py` core per the interfaces above (import `tsvd_solve, lstsq_bias, rel_l2` from `qiblocks`).
- [ ] Run tests, pass.

### Task 2: calibration and band penalty

**Files:** modify `experiments/expI02_block_prior/qi2.py`; test file above.

**Produces:**
```python
S_STAR = 0.4; BAND = dict(mu_max=0.25, s_min=0.15, s_max=0.6, rho=0.9)
@torch.no_grad()
def calibrate(model, X, s_star=S_STAR, s_resid=0.1) -> None   # layer by layer: rescale proj rows / mixer output per channel, set bias; residual layers: increment sd = s_resid; MLP: no-op
def band_penalty(pres: list[torch.Tensor]) -> torch.Tensor      # sum over layers and channels of the four hinge terms; differentiable
```
- [ ] Tests: `test_calibrate_lands_on_s_star` (mean |mu| < 1e-9, |sd - 0.4| < 1e-9 for every layer of a 3-layer stack), `test_band_penalty_zero_inside_positive_outside`.
- [ ] Implement; pass.

### Task 3: optimizers, fit loop, solved-head diagnostic, finisher

**Files:** modify `qi2.py`; test file.

**Produces:**
```python
def power_iteration_L(model, loss_fn, params, iters=10) -> float          # top Hessian eigenvalue of loss at current params (autograd hvp)
def make_optimizer(model, kind: str, lr=5e-3, L=None) -> tuple[torch.optim.Optimizer, callable]  # kind in {"adam","momentum","mixed"}; returns (opt, lr_at(t))
def fit(model, Xtr, Ytr, Xte, Yte, *, optimizer="adam", head="varpro", steps=3000, beta=1e-2, rcond=1e-14,
        log_every=100, loss="mse", batch=None, epochs=None, warmup=50, gn_iters=0, seed=0) -> dict
# log keys: step, train, test_trained, test_solved, band (per-layer [mu,sd,frac_out] at log steps), final, final_trained, final_gn (if gn_iters), time, counts
# head: "trained" (readout is a gradient parameter; test_solved via refit_eval), "periodic" (re-solve every 250), "varpro" (re-solve every step)
# loss "ce": readout trained; test_* are error rates; test_solved = one-hot LS refit of the head
```
- [ ] Tests: `test_power_iteration_positive_and_stable` (L>0; momentum lr=1/L on a frozen-bank quadratic decreases the loss monotonically for 50 steps), `test_fit_logs_solved_head_gap` (head="trained" on 1-D sine: `test_solved[-1] <= test_trained[-1]`), `test_fit_ce_runs` (tiny 2-class problem, error rate < 0.2).
- [ ] Implement; pass.

### Task 4: `tasks.py` -- data, targets, PDEs

**Files:** create `experiments/expI02_block_prior/tasks.py`; test file.

**Produces:**
```python
def analytic(name, d, n_train, n_test=10000, seed=0, dtype=torch.float64) -> dict(Xtr,Ytr,Xte,Yte,r,x0,oracle)  # names: gauss_bump, radial_runge, fast_waves, composition, product_peak, three_bumps, random_ridges; ports the notebook exactly (anchors, ball, inner 0.9 test); oracle = (V, Pfn) or None
def friedman1(seed, n=2000, noise=1.0) ; def lorenz_map(seed, n=4000, stride=20) ; def uci(name, seed)  # name in {concrete, energy}; cached under results/.../data/
def fashion_mnist(cache_dir) -> (Xtr uint8 [60000,784], ytr, Xte, yte)   # raw IDX from the zalando github mirror, cached
def standardize(D) -> D   # inputs to zero mean unit sd on train; targets likewise for real data; returns the scalers
PDES = {"poisson1d": ..., "poisson2d": ..., "burgers": ...}
def pde(name, seed=0, n_col=2048, n_bc=256) -> dict(X_col, X_bc, u_bc, X_test, u_test, residual: callable(model, X) -> [n], linear: bool, d)
```
- [ ] Tests: `test_analytic_matches_notebook_row` (E0-style: 1-D block solved on `sin2pi` < 1e-9), `test_pde_manufactured_residual_zero` (residual of the exact solution wrapped as a model < 1e-8 for poisson1d/poisson2d; burgers exact from expF13 < 1e-6), `test_fashion_loader_shapes` (skipped without cache).
- [ ] Implement; pass.

### Task 5: `run.py` + `plots.py` -- A0 and fig 1

**Files:** create `experiments/expI02_block_prior/run.py`, `experiments/expI02_block_prior/plots.py`.

**Produces:** `run.py {a0,a1,a2,b1,b2,b3,b4,plot} [--quick] [--workers k]`; each study writes `results/checkpoint_I_depth_theory/expI02_block_prior/<study>.json` keyed by run id, skipping finished keys; `plots.py fig1(res_dir)` .. `fig6(res_dir)`.

- [ ] A0: frozen 1-D bank, 4 targets x 4 coords x 3 optimizers x 1000 steps, log every 25; fig1 as in the spec.
- [ ] Run `--quick`, then full; inspect fig1.

### Task 6: A1 mechanism ladder and fig 2
- [ ] Arms in order: `old` (notebook recipe: unit-normalized dirs, regrid, zero Phi, raw, Adam VarPro) reproduced by a flag in `make_block`/`fit` (`legacy=True`: normalization + regrid, no calibration/penalty); `fixed` (+fixed mesh, calibration, penalty, still zero-ish init? no: smooth init is a separate rung) ... exact rungs: `old`, `mesh` (fixed mesh + calib + penalty, raw, Adam, VarPro, smooth init), `val`, `white`, `val_white`, `momentum` (on `white`), `mixed`, `trained_head`, `periodic_head`; then on the winner: beta in {0, 1e-3, 1e-1}, cap in {1e2, 1e4}, rank in {1, 2}; +GN column (20 iters) on every rung; 3 seeds on the top three.
- [ ] fig2: recipe ladder + solved-head gap.

### Task 7: A2 scaling and fig 3
- [ ] Winner recipe at d=4 and d=5 on fast waves / composition / product peak: N1 in {8,16,32,64}, N2 in {32,64,128}, M in {4,8,16}, K in {1,2,4}, depth in {2,3,4} with residual on/off; MLP widths {16,32,64,128,256}; oracle floor from `analytic(...).oracle`. fig3.

### Task 8: B1 and B2 and fig 4
- [ ] B1: d=5, six targets, block/shallow/MLP@params/MLP@FLOPs, 3 seeds, 3000 steps (+GN 20 on the QI arms as a separate field).
- [ ] B2: fp32; Friedman1, Lorenz map, concrete, energy; block, MLP@params, ridge; 3 splits; RMSE on standardized targets.
- [ ] fig4.

### Task 9: B3 Fashion-MNIST and fig 5
- [ ] fp32, device chosen by a 3-batch timing; block (M=64,N1=8,K=64,N2=8, trained head in coords) and a smaller one; tanh MLP@params, ReLU MLP@params, logistic regression; 10 epochs, batch 256, Adam lr 1e-3 cosine; band penalty on; log test accuracy per epoch and wall clock. fig5.

### Task 10: B4 PINNs and fig 6
- [ ] fp64; poisson1d, poisson2d, burgers; block vs MLP@params; Adam 5000 steps on residual + 10 x BC loss; linear PDEs: final head solve on the stacked operator system for both models. fig6.

### Task 11: writeup
- [ ] `results/checkpoint_I_depth_theory/expI02_block_prior/expI02_results.md` in the repo's standard structure; the prior (predictions) stated first, each marked held / failed / untested; every figure with its how-to-read; conclusions marked speculative.
