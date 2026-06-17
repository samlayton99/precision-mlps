# precisionMLPs — Repository Walkthrough

> Orientation document for picking this project back up. It explains what is actually in the
> repo, where to find it, how the heavy-lifting code maps to the theory, what is verified, and
> where the code is (or may be) wrong. Written after a full read of the code, the papers, the
> sibling source-of-truth repo `continuous-mlps`, and after running the test suite.

---

## 1. TL;DR

- **What this is.** A research codebase investigating whether a *training* strategy can learn the
  same machine-precision solutions that an *explicit construction* (Quadrature/Quasi-Interpolation,
  "QI") already produces for single-hidden-layer `tanh` MLPs on 1-D functions.
- **The construction works and is faithful to theory.** I verified it reaches **2.7e-15 (machine
  eps)** via the mpmath path and **5.4e-12** via the fast fp64 path — exactly what the papers
  predict. The `src/` core library is solid and well-tested (**65 tests pass**).
- **The science is only partly done.** Of 11 experiments, **3 are implemented** (exp00, exp01,
  exp0A) and have already produced the key insight; **8 are stubs** (exp02–exp09: docstring +
  `# TODO`). Two of the stubs (exp08, exp09) also need new `src/` machinery before they can be written.
- **There are real latent bugs.** The most dangerous is a silent `gamma_exp` initialization trap
  that would make γ scale as O(1) instead of O(N) — i.e. it would *manufacture violation #1*, the
  exact thing the project studies. See §7. None of them affect the construction; they affect
  training experiments that haven't been written yet.

**Where to start reading:** `future_experiments.md` (the design spec) → `src/construction/qi_mpmath.py`
(the heavy lifting) → `experiments/exp0A_QI_vs_learn/run.py` + `results/results.md` (what's been learned).

---

## 2. The research question and the theory it rests on

**Goal (CLAUDE.md, `future_experiments.md`):** close the gap between explicit construction (~1e-15)
and gradient training (~1e-10). The papers (`papers/QIs_workshop.pdf`, `papers/section3_rewrite.tex`) prove a
one-hidden-layer `tanh` MLP *can represent* a machine-precision interpolant; the open question is
whether an optimizer can *find* it.

**The construction in one breath.** On `[-1,1]` with `N` intervals, grid `x_k = -1 + kh`, `h = 2/N`:
set bandwidth `γ = λ*/h` (so the dimensionless `λ = γh = λ*` is held fixed); solve a Toeplitz system
for target-independent *cardinal coefficients* `c_j`; convolve them with the target **derivative**
`g'` to get outer weights `a_n`; pin a bias `c0` so `q(-1)=g(-1)`. Result:
`q(x) = c0 + Σ_n a_n · tanh(γ(x - x_n))`.

**Theorem 1** (`section3_rewrite.tex`) bounds the error by four exponentially-decaying terms
(resolution `e^{-c/h}`, aliasing `e^{-c/λ}`, halo `e^{-cλR}`, stencil `e^{-cλK_c}`), giving
**exponential convergence in width**, `‖f - f̃_W‖ ≤ A·e^{-αW}`.

**The three violations** (why trained nets fail), and what the theory says each *should* look like:

| # | Violation | Construction (correct) | Trained net (broken) |
|---|-----------|------------------------|----------------------|
| 1 | **γ scaling** | `γ = λ*·N/2`, grows **O(N)**; `λ` plateaus | `γ = O(1)`, so `λ→0` into the ill-conditioned regime |
| 2 | **Outer weights** | `|a_m| = O(1)`, bounded in width | diverge / blow up |
| 3 | **Rank saturation** | every center used; `Φ` near full effective rank | effective rank collapses; power concentrates |

**Success criterion (CLAUDE.md):** across widths {32,64,128,256,…}, over 3–5 seeds, on the
6-category target family, error decays like `O(log 1/ε)` and reaches **eval rel-L2 ≤ 1e-13** and
**L∞ at machine eps**, *without* initializing from the construction.

**Two precision regimes** (`papers/practical_implementation.tex`): the fp64 Toeplitz solve becomes
ill-conditioned at small `λ`, and — more importantly — the convolution cancels `O(300)`-magnitude
alternating `c_j` terms, flooring fp64 at ~1e-12. `mpmath` (30 dps here) does the offline precompute
in high precision and **rounds the coefficients to fp64**; the model/training/eval all still run in
fp64. This is *not* an fp64 violation — analogous to `numpy.pi`.

---

## 3. Repository map

```
papers/                     The theory. Read section3_rewrite.tex + practical_implementation.tex.
docs/                       Project docs.
  WALKTHROUGH.md            This file — repo orientation.
  future_experiments.md     *** THE DESIGN SPEC *** numbered sections map 1:1 onto exp00..exp09.
  thoughts.md               Scratchpad (Socratic framing notes).
src/                        Core library (PyTorch, all fp64). Solid, well-tested.
  __init__.py               Sets torch fp64 default + DEVICE (CUDA/CPU; MPS excluded, no fp64).
  config/{schema,loader}    ExperimentConfig dataclasses; YAML load + sweep expansion.
  models/
    layers.py               GammaLinear / GammaExpLinear / StandardLinear inner layers.
    mlp.py                  QIMlp: inner_layer -> tanh -> readout. Exposes features(x), accessors.
    freeze.py               requires_grad freezing helpers.
  construction/
    qi_mpmath.py            *** THE HEAVY LIFTING *** QI construction, fp64 + mpmath, caching.
    readout.py              Phi matrix + readout solve (lstsq/qr/svd/ridge), with-bias variant.
    initialize.py           Project a QIResult into model params; init-and-freeze; readout re-solve.
    README.md               Construction-package quick reference (lives next to the code).
  data/{targets,sampling,dataset}   9 targets / 6 categories; samplers; dataset builder.
  training/{optimizers,losses,train_loop,metrics}   Multi-stage Adam->LBFGS, losses, metric schema.
experiments/                One folder per experiment (config.yaml + run.py). 3 done, 8 stubs.
results/                    JSONL/JSON outputs + results.md (the running lab notebook).
tests/                      Unit tests for the core library (71 pass; mpmath ones marked `slow`).
scripts/                    sweep_machine_eps.py, sweep_qi_convergence.py (standalone probes).
```

---

## 4. The heavy lifting: `src/construction/qi_mpmath.py` ↔ theory

This is the file that matters most. It is a faithful port of
`continuous-mlps/src/construction/explicit_quasi_interpolant.py` (the source of truth). The mapping
from code to the paper's algorithm is exact:

| Step | Theory (`practical_implementation.tex`) | Code | Verdict |
|------|------------------------------------------|------|---------|
| Grid + bandwidth | `h=2/N`, `γ=λ*/h`, `x_k=-1+kh` | `qi_mpmath.py:436-437` | ✓ γ grows **O(N)** |
| Toeplitz for `c_j` | `T_{r,j}=h·Kd((r-j)h)`, solve `Tc = e_0·h`, `Kd=γ·sech²(γx)` | `_build_toeplitz_c_f64:104`, `_build_toeplitz_c_mpmath:213`; RHS `b[Kc]=h` | ✓ center entry is `h`, not 1 |
| Convolve **derivative** | `a_n = Σ_k c_k·g'(x_{n-k})`, Kahan summation | `_build_a_*_kahan:146,259` | ✓ uses `g'`, Kahan-compensated |
| Bias | `c0 = g(-1) - Σ_n a_n·tanh(γ(-1-x_n))` | `_compute_c0_*:168,291` | ✓ (plain sum, not Kahan — see §7 minor) |
| Interpolant | `q(x)=c0+Σ a_n tanh(γ(x-x_n))` | `evaluate_qi:534` | ✓ |
| Halo | ghost nodes outside `[-1,1]`, width `W=N+2R+1` | `default_halo:49`, center grid `:492/506` | ✓ |
| Caching | `c_j` target-independent, keyed by `(λ*,Kc,N,precision,dps)` | `_cache_key:319`, `results/qi_cache/` | ✓ |

**Empirically verified (I ran this):** `sin(2πx)`, `N=64`, eval on 4001 points, Kahan eval:

```
fp64    width=183  γ=9.600  λ=0.30   L∞ = 5.39e-12     (matches predicted cancellation floor)
mpmath  width=205  γ=8.000  λ=0.25   L∞ = 2.67e-15     (true machine epsilon)
```

`γ = λ*·N/2` holds exactly (9.6 = 0.30·64/2; 8.0 = 0.25·64/2). The fp64 path floors at ~1e-12
*by design* (convolution cancellation); machine eps requires mpmath. **This is correct and on-theory.**

**`QIResult`** (`qi_mpmath.py:65`) is an immutable dataclass of pure construction data (centers,
`a_coeffs`, `c0`, `γ`, `λ`, halo, …) — it never references a model. `initialize.py` copies it into a
`QIMlp`.

**Parameter facts that are load-bearing** (corroborated by both papers *and* the source-of-truth
sweep data): `λ* = 0.30` (fp64) / `0.25` (mpmath); `Kc = 160`; halo grows with N. The known
non-working values (`λ*=1.5`, `Kc=12`) are confirmed by the `continuous-mlps` sweep never selecting
them at the optimum.

---

## 5. Model and training stack

**`QIMlp`** (`models/mlp.py:18`): `readout(tanh(inner_layer(x)))`. Three inner layers
(`models/layers.py`):

- `GammaLinear` — stores `γ` and `centers` directly; forward `γ·(x-center)`. Natural QI match;
  gradient w.r.t. `γ` vanishes ~O(1/N) at large γ (the core difficulty the project studies).
- `GammaExpLinear` — stores `log_gamma`; effective `γ = exp(log_gamma)/h`. Reparam so
  `d/d(log_gamma) = O(1)`. **Has an initialization trap — see §7.**
- `StandardLinear` — plain `nn.Linear(1,W)`; baseline for the reparameterization experiment.

γ's O(N) scaling is **not enforced by any layer** — it comes purely from initializing from the
construction. The model imposes no scaling on its own.

**Training** (`training/`): multi-stage (default Adam 30k → LBFGS 5k), losses `mse`/`lp`/
`hybrid_boundary`, optimizers Adam/AdamW/SGD/LBFGS, optional periodic readout re-solve.
`MetricsCollector` (`metrics.py`) logs a fixed 19-metric schema per eval step (loss, L∞, rel-L2,
γ/λ stats, outer-weight norms, feature rank/stable-rank) — this is the uniform measurement layer the
violations are diagnosed through.

---

## 6. Experiments: status and what each does

**Only exp00, exp01, exp0A are implemented.** exp02–exp09 are stubs (docstring + `# TODO` with
commented pseudocode); confirmed by `results/` having output only for the three implemented ones.

| Exp | Topic | Status | What it does / would do |
|-----|-------|--------|--------------------------|
| **exp00** | Numerics sanity | **Done** | Rules out numerical (not optimization) floor: construction in fp64/mpmath, readout solver comparison (lstsq/qr/svd/ridge), `cond(Φ)` vs `cond(ΦᵀΦ)`, tanh fp64-vs-mpmath stability. Outputs 7 JSONL + summary. |
| **exp01** | λ tradeoff | **Done (most mature)** | Confirms the U-shaped error-vs-λ curve. Sweeps targets×λ×width, QI vs least-squares on same geometry. `plot_consolidated.py` builds the 3×4 figure. |
| **exp0A** | QI vs learned readout | **Done** | 4-way on identical geometry: QI(mpmath/fp64) vs lstsq(mpmath/fp64). **Key result: lstsq ≥ QI in 48/48 configs given the geometry.** |
| exp02 | Basin stability | **Stub** | Drift from QI under low-LR optimizers; perturbation profiles; QI↔trained interpolation. |
| exp03 | Geometry ladder | **Stub (lynchpin)** | 7-level constraint relaxation (full construction → fully free). The planned "where is precision lost" experiment. |
| exp04 | Hessian | **Stub** | Eigenspectrum at QI vs trained. *Needs new Hessian helpers in `src/` — none exist.* |
| exp05 | Φ conditioning | **Stub** | `cond(Φ)` vs N, λ. Largely overlaps exp00's conditioning section. |
| exp06 | Objective mismatch | **Stub** | Compare loss functions. *Stub references `weighted_mse`/Chebyshev losses not in `losses.py`.* |
| exp07 | Noise sensitivity | **Stub** | X/Y-noise robustness. |
| exp08 | Reparameterization | **Stub** | Head-to-head of layer parameterizations. *global-bandwidth & dimensionless-center layers not implemented.* Uses `gamma_exp` → hits the §7 trap. |
| exp09 | VarPro | **Stub** | Variable-projection reduced objective. *No VarPro code exists anywhere in `src/`.* Uses `gamma_exp` → hits the §7 trap. |

**The science so far** (`results/results.md`): with mpmath, both QI *and* a plain least-squares
readout reach machine eps; **lstsq actually beats QI on identical geometry**; fp64-lstsq ≈
mpmath-QI (~1e-13). Conclusion already drawn: **the geometry (γ, centers) is the hard part — given
correct geometry, even fp64 lstsq hits 1e-13.** So the open problem reduces to: *can an optimizer
discover the geometry?* This is why `future_experiments.md` now points at exp03 (geometry ladder) and
exp09 (VarPro / reduced-coordinate training) as the most promising paths.

---

## 7. Bugs found — all now FIXED

None of these touched the construction (which was already correct). They were in the model/training
glue and would have bitten when the stubbed *training* experiments get written. **All were fixed in a
cleanup pass, each with a regression test (suite: 71 passing).** Originals kept below for the record.

| # | Issue | Fix | Test |
|---|-------|-----|------|
| A | `gamma_exp` init collapsed γ to O(1) | `initialize.py` sets `inner.h = qi.h` | `test_gamma_exp_init_recovers_construction_gamma` |
| B | LBFGS lr cosine-annealed | `build_scheduler` returns None for LBFGS | `test_scheduler_skips_lbfgs` |
| C | Readout re-solve ignored freeze/config | gated on trainable readout, uses configured method/α | `test_readout_resolve_respects_freeze` |
| D | `svd` solve divided by ~0 | truncates at `rcond·s_max`, reports rank | `test_svd_truncates_tiny_singular_values` |
| E | Metrics λ used wrong `h` | infers `h` from center spacing | `test_metrics_h_inferred_from_center_spacing` |
| F | `c0` boundary sum not compensated | uses `math.fsum` | covered by precision tests |
| H | Dead code / stale docs | MPS branch removed, dead config fields removed, docstrings fixed, empty `sandbox.ipynb` deleted | config-load smoke test |

Also: the fp64 convolution was vectorized (Python Kahan loop → sliding-window matmul), ~20ms→~0.5ms
with no usable precision loss (still 6.4e-12). Finding G (seed only varies `uniform` sampling) was not
a bug — it's now documented in `dataset.py`.

<details><summary>Original findings (pre-fix), ordered by severity</summary>

**A. `gamma_exp` initialization trap — HIGH (confirmed first-hand).**
`QIMlp.__init__` (`mlp.py:28`) forwards `**layer_kwargs` to `get_layer` but never injects `h`, so
`GammaExpLinear` defaults to `h=1.0` (`layers.py:68`). `initialize_from_construction` then sets
`log_gamma = log(λ*)` (`initialize.py:58`), giving effective `γ = λ*/h = λ*` — **O(1), not O(N)** —
unless the caller explicitly passes `h=2/N`. This silently *manufactures violation #1*, the exact
pathology the project is trying to study. exp08 and exp09 both use `gamma_exp`. **Fix before writing
them:** have `QIMlp`/`initialize` propagate `h=qi.h`, or assert `λ = γh ≈ λ*` after init.

**B. LBFGS gets cosine-annealed LR by default — MEDIUM (confirmed first-hand).**
`OptimizerStageConfig.use_cosine_schedule` defaults `True` (`schema.py:73`); the default LBFGS stage
(`schema.py:83`) doesn't override it; `build_scheduler` returns a `CosineAnnealingLR` whenever it's
true (`optimizers.py:61`); and `train_loop.py:98` calls `scheduler.step()` every step regardless of
optimizer. So the default LBFGS `lr=1.0` is annealed to `1e-6` over the stage — almost certainly
unintended for a strong-Wolfe line-search optimizer. **Fix:** set `use_cosine_schedule: false` on
LBFGS stages (or skip the scheduler when `name=="lbfgs"`).

**C. Periodic readout re-solve ignores freezing and config — MEDIUM (confirmed first-hand).**
`train_loop.py:103` calls `initialize_with_readout_solve(..., method="lstsq")` gated only by
`readout_solve_every>0`. It writes via `no_grad().copy_`, so it mutates the readout **even if frozen**,
and it hardcodes `lstsq`, ignoring `init.readout_solve`/`ridge_alpha`. Harmless at the default
(`readout_solve_every=0`), but a correctness trap once enabled.

**D. No singular-value truncation in `svd`/`lstsq` readout solves — MEDIUM (reported).**
`readout.py` `solve_readout` `svd` branch divides by all singular values; plain `lstsq` uses
`rcond=None`. In the ill-conditioned regime this project targets, that is numerically risky.
(For contrast, exp0A's *own* mpmath pseudo-inverse truncates at `1e-15·s_max`, and `continuous-mlps`
uses `rcond=1e-13`.) **Consider** `rcond≈1e-13` to match the source of truth.

**E. Metrics `h` mismatch — LOW (diagnostic only, reported).**
`metrics.py` computes `h=(b-a)/width` while the construction uses `h=2/N` with `width=N+1` (interior)
or `N+2R+1` (full). So the reported `λ=γh` is off by an O(1/N) factor from `qi.lambda_val`. Affects
only the logged λ diagnostic, not training.

**F. `c0` boundary sum is not Kahan-compensated — LOW (confirmed).**
The convolution uses Kahan (`qi_mpmath.py:146/259`) but the `c0` boundary sum uses plain summation
(`_compute_c0_f64:173`). `continuous-mlps` Kahan-compensates here too. Immaterial in practice (the
mpmath probe still hits 2.67e-15), but it's a divergence from the source of truth.

**G. Seed only varies `uniform` sampling + noise — LOW (reported, by design).**
`dataset.py` makes `equispaced`/`chebyshev`/`qi_grid` train data seed-independent. So multi-seed runs
with the default `equispaced` sampling differ only via model init / training stochasticity — fine,
but worth knowing when interpreting "3–5 seed" variance.

**H. Dead code / stale docs — LOW (reported).**
`DEVICE` (`src/__init__.py`) is never used — everything runs on CPU (and MPS can't do fp64 anyway).
`ModelConfig.activation`, `ConstructionConfig.gamma`/`enabled` are never read. Docstrings say
`default_halo = max(50, 0.4N)` but the code floor is `ceil(35/(2λ))` (=59 at λ=0.30, 70 at λ=0.25);
docs say "6 targets" but there are 9 functions across 6 categories.

(Note: `ConstructionConfig.enabled` is kept — it's set in 6 experiment YAMLs and the strict loader
needs it; only the truly-unreferenced `activation` and `construction.gamma` were removed.)

</details>

---

## 8. Fidelity to `continuous-mlps` (the source of truth)

The construction is a faithful port. All seven high-value correctness points from the reference repo
hold in `qi_mpmath.py`: Toeplitz RHS center `=h`; `γ=λ*/h ∝ N`; Kahan convolution; halo ghost
centers with `W=N+2R+1`; `c0` pinned to `q(-1)=g(-1)`; convolution of the **derivative** `g'`;
`Kc=160`, `λ=0.30/0.25` regime.

Divergences, all benign:

- **mpmath dps:** this repo uses `mp_dps=30`; `continuous-mlps` uses 120–128. 30 is sufficient
  because the output is fp64 anyway (verified: 2.67e-15). 
- **`c0` Kahan:** reference Kahan-compensates the `c0` sum; this repo doesn't (§7-F).
- **Readout `rcond`:** reference uses `rcond=1e-13`; this repo's lstsq uses `rcond=None` (§7-D).
- **Objective:** reference tracks L∞ of both `q` and `q'`; this repo measures `q` only.
- **Model parameterization:** reference `GammaLinear` holds `γ` via a fixed `a_scale` with
  `gamma=1`; this repo stores `γ` directly. Mathematically equivalent.

Note: in `continuous-mlps` the QI machinery lives in `src/construction/explicit_quasi_interpolant.py`
and `experiments/junmi_1d/CPU_theoretical_verification/`, **not** in its `src/train/models/`
spectral/barycentric stack — don't be misled by those files.

---

## 9. How to resume

1. **Environment**: venv at `~/.venvs/precisionMLPs` (outside iCloud, per global CLAUDE.md). Run tests
   with `~/.venvs/precisionMLPs/bin/python -m pytest -q -m "not slow"` (70 pass; drop `-m "not slow"`
   for the full 71 including the ~35s cold mpmath construction test).
2. **The §7 bugs are fixed** — `gamma_exp` (exp08/exp09) now initializes correctly, so you can build
   those experiments without manufacturing the very pathology you measure.
3. **Build exp03 (geometry ladder) next** — `results.md` and `future_experiments.md` both flag it as
   the lynchpin: it localizes exactly which constraint relaxation breaks precision.
4. **For exp04/exp06/exp08/exp09**, note the missing `src/` machinery called out in §6 (Hessian
   helpers, extra losses, global-bandwidth/dimensionless layers, a VarPro objective) — these need to
   be written and unit-tested first, with the construction-reaches-machine-eps test as the template.

> Pre-existing quirk (not addressed): `exp01` and `exp0A` `config.yaml` files use a `targets` field
> not in `ExperimentConfig`, so they fail `load_config` — but those run.py scripts don't use the
> loader (they hardcode their target lists), so it's harmless. The other 9 configs load fine.

---

*Verified: 71/71 tests pass (incl. slow mpmath); construction still reaches 6.4e-12 (fp64) and
2.67e-15 (mpmath) on `sin(2πx)`, N=64, matching theory. All §7 bugs fixed with regression tests; the
fp64 convolution was vectorized (~40x faster construct) with precision preserved.*
