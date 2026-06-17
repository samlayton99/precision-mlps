# Explanation: the math, `qi_mpmath.py`, and how it all maps to the paper

This is a conceptual walkthrough. The goal is that by the end you can read
`src/construction/qi_mpmath.py` and see the paper's equations staring back at you,
and you understand *why* each piece exists — including the two things that are
easy to conflate (the **Toeplitz solve** and the **least-squares / Φ solve**) and
the thing that sounds contradictory (using **mpmath** when "the point is fp64").

Paper: `papers/QIs_workshop.pdf` ("Constructing Machine-Precision Neural Networks
with Quasi-Interpolants"). Section 3 in that PDF is stale; the current construction
is `papers/section3_rewrite.tex`, and the fp64/mpmath details are in
`papers/practical_implementation.tex`.

---

## 0. What you're probably actually confused about

Before the details, let me name the four conceptual knots, because every specific
question you asked is a symptom of one of them.

1. **You think there is "a solve," singular.** There are *two completely different*
   linear-algebra problems in this codebase, and they answer different questions:
   - **Toeplitz solve** → the *cardinal coefficients* `c_j`. This is part of the
     **construction**. It is target-independent. It defines *how to interpolate at all*.
   - **Least-squares solve with the Φ matrix** → the *outer (readout) weights*.
     This is an *alternative* to the construction's convolution formula, and it's
     also exactly the object the paper uses to diagnose trained networks (§4.3).

   Once you see these as two separate machines, half the confusion dissolves.

2. **You don't see how a matrix solve "is" the math.** The math says: "find a
   cardinal function `L_h` that equals 1 at its own node and 0 at every other
   node." That sentence *is* a system of linear equations. Writing the requirement
   down at the grid points produces a matrix, and because the grid is uniform and
   the kernel only cares about *distance*, that matrix is Toeplitz. Toeplitz isn't
   a different method; it's the linear-algebra shadow of the interpolation
   condition.

3. **You think "fp64" means "every computation must be fp64."** It doesn't. It
   means the *representation* — the model's weights, its forward pass, its
   gradients, its evaluation — lives in fp64. The construction computes a handful
   of *constants* (the weights of a specific MLP). You can compute constants as
   accurately as you like and then round them to fp64, exactly like `numpy.pi`.
   mpmath is how we compute the *best possible fp64 constants*.

4. **You don't see where Φ / least squares connects to the paper.** It's not a
   side-quest. It is literally the paper's §4.2 ("fix the geometry, fit only the
   output coefficients") and §4.3 ("freeze the hidden layer, form `Φa=b`, ask how
   compressible Φ is"). The repo's `readout.py` is that experiment's engine.

Keep these four in mind; the rest of the document earns them.

---

## 1. The object we're building

A one-hidden-layer `tanh` MLP from `ℝ → ℝ` is just a weighted sum of shifted,
scaled `tanh`s plus a bias:

$$ \tilde f(x) \;=\; b \;+\; \sum_{m} w_m \,\tanh\!\big(\gamma\,(x - x_m)\big). $$

- `x_m` — the **centers** (where each `tanh` sits).
- `γ` — the **bandwidth** / steepness (how sharp each `tanh` is). One shared `γ`
  in the construction.
- `w_m` — the **outer / readout weights**.
- `b` — a single **bias**.

In code this is `QIMlp`: `readout(tanh(inner_layer(x)))`, where `inner_layer`
computes `γ·(x − center)` (`src/models/layers.py`, `GammaLinear`) and `readout` is
`nn.Linear(width, 1)` holding `w_m` and `b`.

The research question: can we *choose* `(γ, x_m, w_m, b)` so that `tilde f`
matches a target `f` to ~10⁻¹⁵ (machine epsilon)? The paper's Section 3 answers
**yes, explicitly** — that explicit recipe is the QI construction, and
`qi_mpmath.py` is its implementation. (The harder, open question — can *gradient
training* find such weights — is Section 4 and the rest of the repo.)

Everything below is: *how do we pick those numbers?*

---

## 2. Why `tanh`, and why we secretly work with the derivative

Quasi-interpolation (next section) is a theory about **kernels** — bump-like
functions `K` you place at each sample and add up. The natural kernel here is

$$ K(u) = \operatorname{sech}^2(u), $$

a symmetric bump. The magic fact (paper Eq. `kernel-derivative`):

$$ \operatorname{sech}^2 = \tanh', \qquad\text{i.e. } K = \psi' \text{ with } \psi=\tanh. $$

So a sum of **bumps** is the derivative of a sum of **tanhs**. If we build a great
kernel approximation of `f'` using `sech²` bumps,

$$ f'(x) \approx \sum_m a_m \, K_\gamma(x - x_m), \qquad K_\gamma(x):=\gamma K(\gamma x), $$

then integrating once turns each `sech²` bump back into a `tanh`, and the bump
weights `a_m` become the MLP's outer weights (paper Eq. `chain-rule-realization`,
`mlp-def`):

$$ f(x) \approx b + \sum_m a_m \,\tanh(\gamma(x - x_m)). $$

The constant of integration is the bias `b`. (For `tanh`/`sech²` the "integration
polynomial" is just this one constant; for GELU/Swish it would be a linear term —
paper Eq. `integration-poly`.)

**This is why the construction convolves with `g'`, not `g`.** The kernel lives at
the derivative level; the `tanh` MLP is its antiderivative. Hold onto that — it
explains a line of code that otherwise looks arbitrary.

The scaled kernel `K_γ(x) = γ·K(γx)` is a bump of width `~1/γ` and unit mass.
Larger `γ` = narrower, taller bump. The grid spacing is `h`. The single most
important quantity in the whole theory is the **dimensionless bandwidth**

$$ \boxed{\;\lambda := \gamma h\;} \qquad\text{(paper Eq. \texttt{lambda-def})} $$

which measures "how many grid points fit under one bump." It is the knob that
trades off two competing errors (Section 6).

---

## 3. Quasi-interpolation: reconstructing `f` from samples

Classical interpolation finds one global polynomial through all samples — it
blows up (Runge phenomenon). Quasi-interpolation instead reconstructs `f` as a
**translation-invariant** sum: put the *same* local cardinal function `L_h` at
every node and weight it by the sample there (paper Eq. `qh-infinite`):

$$ (Q_h f)(x) \;=\; \sum_{k\in\mathbb{Z}} f(x_k)\, L_h(x - x_k), \qquad x_k = -1 + kh. $$

For this to actually interpolate — to reproduce the samples and converge nicely —
`L_h` must be a **cardinal function**: it equals 1 at its own node and 0 at every
other node:

$$ L_h(0) = 1, \qquad L_h(jh) = 0 \ \text{ for integer } j\neq 0, \qquad\text{i.e. } L_h(jh)=\delta_{j,0}. $$

That property is what makes `Q_h f` reproduce data and have clean error behavior.
**The entire job of the Toeplitz solve is to build this `L_h`.** That's the link
you were missing: "Toeplitz" is not a separate idea — it's how we *enforce the
cardinal property*.

### 3.1 Building `L_h` out of kernels

We don't have `L_h` in closed form. We *build* it as a combination of our kernel
bumps placed on the grid, with unknown coefficients `c_j` (paper Eq.
`truncated-cardinal`):

$$ L_h(x) \;=\; \sum_{j} c_j \, K_\gamma(x - jh). $$

Now impose the cardinal property at the grid points. Plug `x = kh`:

$$ L_h(kh) = \sum_j c_j\, K_\gamma(kh - jh) = \sum_j c_j\, K_\gamma\big((k-j)h\big) \;=\; \delta_{k,0}. $$

Read that last equation carefully — it is a **linear system in the unknowns
`c_j`**. One equation per node `k`. The coefficient multiplying `c_j` in equation
`k` is `K_γ((k−j)h)`.

### 3.2 Why it's Toeplitz (and what Toeplitz means)

A **Toeplitz matrix** is one that is constant along each diagonal: entry `(k, j)`
depends only on the *difference* `k − j`, not on `k` and `j` separately. Picture
it: every row is the previous row shifted by one.

Our matrix entry is `K_γ((k−j)h)` — it depends only on `k − j`. So it's Toeplitz,
*automatically*, and the reason is physical: the kernel only cares about the
**distance** between node `k` and node `j`, and the grid is **uniform**, so the
relationship between node 5 and node 7 is identical to that between node 100 and
node 102. Shift-invariance of the setup ⇒ constant diagonals ⇒ Toeplitz.

Writing `T_{k,j} = h\,K_γ((k−j)h)` and the right-hand side as the unit spike `e_0`
(times `h`, a normalization), the cardinal condition becomes simply

$$ \boxed{\,T\,c = h\,e_0\,} \qquad\text{(paper appendix \emph{cardinal computation}; code uses RHS } b[K_c]=h). $$

Solve this linear system → you get the coefficients `c_j` → you have `L_h` → you
can quasi-interpolate. The `c_j` are called **cardinal coefficients**. Two facts
make them special:

- **They do not depend on the target `f`.** They depend only on `(λ, K_c)` — the
  shape of the kernel relative to the grid. So you solve the Toeplitz system once
  and reuse `c_j` for every target. (This is exactly why the code caches them to
  disk; see §7.)
- **They decay but oscillate.** `|c_0| ≈ 300+` and they alternate in sign as `|j|`
  grows. This alternation is the seed of the fp64 precision problem (§8).

> There's also a Fourier picture (paper Eq. `fourier-character`):
> `Ĉ_h(ω) = h / D_h(ω)` where `D_h` sums the kernel's Fourier transform over all
> aliases, and the `c_j` are the Fourier coefficients of that ratio. "Normalize the
> kernel's spectrum so the sum-of-shifts is flat" is the frequency-domain way of
> saying "make `L_h` cardinal." The Toeplitz solve is the real-space equivalent.
> Same `c_j`, two derivations.

### 3.3 Why `K_c` (truncation) and `halo R`

Two practical truncations turn the infinite theory into a finite computation:

- **`K_c` (stencil half-width):** we only keep `c_j` for `|j| ≤ K_c` (the sum in
  §3.1 is finite). Since `c_j` decays, this is fine if `K_c` is big enough.
  `K_c = 160` is the working value. (`K_c = 12` is famously *not* enough — the
  coefficients haven't decayed yet.)
- **`halo R`:** near the domain edges `x = ±1`, an interpolation node needs
  neighbors that fall *outside* `[−1,1]`. The halo adds `R` "ghost" centers on each
  side so edge nodes still have a full stencil. The MLP keeps those ghost `tanh`s —
  their centers sit outside `[−1,1]`. (Strikingly, the paper's §4.2 reports that
  *trained* networks also push some centers outside the domain — they rediscover
  the halo.)

Total hidden width: `W = N + 2R + 1` (interior nodes plus both halos).

---

## 4. From cardinal coefficients to the MLP weights

Now we have `c_j` (the interpolation machinery). To approximate a *specific*
target we do two cheap, target-dependent steps.

**Step A — convolve to get outer weights** (paper Eq. `single-kernel-sum`).
Re-indexing the quasi-interpolant into a single sum of kernels gives outer weights

$$ a_m \;=\; \sum_{|j|\le K_c} c_j \, g'(x_{m-j}). $$

This is a **discrete convolution** of the cardinal coefficients with the sampled
target derivative `g'`. (Remember §2: the kernel lives at the derivative level, so
we feed it `g'`.) One `a_m` per center.

**Step B — fix the bias from a boundary condition.** We pin the antiderivative so
the MLP matches the target at the left endpoint, `tilde f(−1) = g(−1)`:

$$ b \;=\; g(-1) - \sum_m a_m \,\tanh\!\big(\gamma(-1 - x_m)\big). $$

That's the whole construction. The final MLP is

$$ \tilde f(x) = b + \sum_m a_m \,\tanh(\gamma(x - x_m)), \qquad \gamma = \lambda^\*/h,\ \ x_m = -1 + mh. $$

Notice `γ = λ\*/h = λ\*·N/2`: as you add neurons (`N↑`), `γ` **grows linearly**.
That O(N) growth of the bandwidth is the central structural prediction of the
paper — and the thing trained networks fail to do (§9).

---

## 5. `qi_mpmath.py`, line by line, against the math

Now the file reads like the math above. Here's the map (function → equation):

| Code | Math | Section |
|---|---|---|
| `default_halo(N, λ)` | choose `R` so halo-truncation `e^{-c₃λR}` is below ε | §3.3 |
| `_build_toeplitz_c_f64 / _mpmath` | solve `T c = h·e₀`, `T_{k,j}=h·K_γ((k−j)h)`, `K_γ=γ·sech²(γx)` | §3.1–3.2 |
| `_build_a_f64 / _build_a_mpmath_kahan` | `a_m = Σ_j c_j g'(x_{m−j})` (convolution) | §4 Step A |
| `_compute_c0_f64 / _mpmath` | `b = g(−1) − Σ a_m tanh(γ(−1−x_m))` | §4 Step B |
| `construct_qi(...)` | orchestrates: `c_j` → `a_m` → `b`, returns `QIResult` | §3–4 |
| `evaluate_qi(qi, x)` | `tilde f(x) = b + Σ a_m tanh(γ(x−x_m))` | §4 |

Walking `construct_qi` top to bottom:

1. **Defaults & geometry** (`qi_mpmath.py:431-437`): pick `λ\*` (0.30 fp64 / 0.25
   mpmath), `halo`, then `h = 2/N`, `γ = λ\*/h`. This is `λ = γh` made concrete.
2. **Cardinal coefficients** (`:439-482`): build/solve the Toeplitz system for
   `c_j` — *with caching*, because `c_j` is target-independent (§7). The matrix is
   `T_{k,j} = h·γ·sech²(γ(k−j)h)`; RHS is `h` at the center index. This is §3.2.
3. **Convolution** (`:484-511`): sample `g'` on the extended grid and convolve with
   `c_j` to get `a_m`. (`_build_a_f64` does this as one vectorized sliding-window
   matmul; the mpmath path uses compensated summation — see §8.) This is §4-A.
4. **Bias** (`:493-510`): pin `tilde f(−1)=g(−1)` to get `b` (`c0` in code). §4-B.
5. **Package** (`:513-531`): split interior vs halo, return an immutable
   `QIResult` carrying `centers, a_coeffs, c0, γ, λ, halo, K_c`. Pure data; no
   model. `initialize.py` later copies these numbers into a `QIMlp`'s parameters.

That's it. There is no training, no gradient, no optimizer in this file. It is a
deterministic recipe that emits the weights of one specific MLP.

---

## 6. The λ tradeoff (why there's a "right" bandwidth)

Why not just crank `K_c`, `R`, `N` and pick any `λ`? Because of the error bound
(paper Theorem 1, Eq. `main-error-bound`). The total error splits into four
exponentially-decaying pieces:

$$ \|f - \tilde f\|_\infty \;\le\;
\underbrace{C_1(\lambda)e^{-c_1/h}}_{\text{resolution}} +
\underbrace{C_2 e^{-c_2/\lambda}}_{\text{aliasing}} +
\underbrace{C_3(\lambda)e^{-c_3\lambda R}}_{\text{halo}} +
\underbrace{C_4(\lambda)e^{-c_4\lambda K_c}}_{\text{stencil}}. $$

Look at how `λ` appears with opposite signs:

- **Large `λ`** (wide bumps relative to grid): the **aliasing** term `e^{−c₂/λ}`
  is large — fat bumps overlap and can't resolve detail. Bad.
- **Small `λ`** (narrow bumps): aliasing shrinks, *but* the prefactors
  `C₁, C₃, C₄` **blow up** algebraically as `λ→0` — the construction becomes
  ill-conditioned. Bad.

So there's a **U-shaped** error-vs-`λ` curve with a sweet spot around
`λ\* ≈ 0.25–0.30`. That's why those exact values are hard-coded, and why
`λ=1.5` (too aliased) and tiny `λ` (too ill-conditioned) both fail. **exp01 is the
experiment that traces this U-curve and confirms the sweet spot** (§10).

The deep consequence (paper §3 end, and the abstract): to *stay* at the good `λ\*`
as you refine the grid (`h = 2/N → 0`), you must grow `γ = λ\*/h ∝ N`. Holding
`λ` fixed *forces* `γ = O(N)`. Conversely, if `γ` stays `O(1)` while `N` grows
(what optimizers do), then `λ = γh → 0` and you fall off the left side of the U
into the ill-conditioned regime. This is the seed of "violation #1."

---

## 7. The cache (a small but important detail)

`c_j` depends only on `(λ\*, K_c, N, precision, mp_dps)` — **not** on the target.
The mpmath Toeplitz solve costs ~30–55 s cold. So `qi_mpmath.py` caches `c_j` to
`results/qi_cache/` keyed by those parameters (fp64 as `.npz`, mpmath as
full-precision text). First call for a given config pays the cost; every later
call (any target, any seed) reloads in ~0.25 s and just redoes the cheap
convolution + bias. This is why running a whole sweep of targets at fixed `(N, λ)`
is fast.

---

## 8. The Φ matrix and least squares — the *other* solve

Everything so far computes the outer weights `a_m` by the **convolution formula**.
But here is a completely different way to get outer weights, and it's the one your
question about "least squares / the Φ matrix" is really about.

**Setup:** suppose you've *already fixed* the geometry — the centers `x_m` and the
bandwidth `γ` (say, from the construction). The only things left to choose are the
outer weights `v_m` and bias. The model is now **linear** in those weights:

$$ \tilde f(x_i) = \sum_m v_m \,\underbrace{\tanh(\gamma(x_i - x_m))}_{\Phi_{i m}} + b. $$

Stack the training points `x_i` as rows and the centers `x_m` as columns. The
matrix

$$ \boxed{\;\Phi_{im} = \tanh\big(\gamma\,(x_i - x_m)\big)\;} $$

is the **feature matrix** (a.k.a. design matrix). Column `m` is "what neuron `m`
outputs across all the data points"; row `i` is "all neuron activations at point
`x_i`." Then fitting the readout is just the linear least-squares problem

$$ \min_{v}\ \|\Phi\,v - y\|_2^2, \qquad y_i = f(x_i). $$

This is `src/construction/readout.py`: `build_phi` builds `Φ`; `solve_readout`
solves the least-squares system (via `lstsq` / `qr` / `svd` / ridge);
`solve_readout_with_bias` adds a ones-column for `b`. (In the paper's §4.3
notation, `Φa=b` with `Φ_ij = tanh(γ_j(x_i−x_j))` and the bias mode removed via
`b_i = f(x_i) − f̄`. Same object.)

### 8.1 Two solves, side by side — do not conflate these

| | **Toeplitz solve** (§3) | **Least-squares / Φ solve** (§8) |
|---|---|---|
| Unknowns | cardinal coefficients `c_j` | outer/readout weights `v_m` |
| Matrix | `T_{k,j}=h K_γ((k−j)h)` — Toeplitz, square | `Φ_{im}=tanh(γ(x_i−x_m))` — rectangular (data × neurons) |
| Depends on target? | **No** (target-independent) | **Yes** (right-hand side is `y=f(x)`) |
| Role | *defines* the interpolation operator `L_h` | *fits* the outer weights given fixed geometry |
| Where used | the QI **construction** (`qi_mpmath.py`) | the **alternative readout** (`readout.py`); paper §4.2, §4.3 |
| Output feeds | convolution → `a_m` | directly the `v_m` |

The QI construction (Toeplitz + convolution) and the least-squares readout are
**two routes to the outer weights of the same fixed-geometry MLP**. The QI route
uses a fixed analytic formula (`a_m = Σ c_j g'(x_{m−j})`); the least-squares route
directly minimizes the residual on the data.

### 8.2 Why the repo bothers with both (and what it found)

If both fill in the outer weights of the same model, why compare them? Because it
*isolates the source of error*. **exp0A** builds the *same* QI geometry and then
fills the readout four ways: QI-formula in mpmath, QI-formula in fp64,
least-squares in fp64, least-squares in mpmath. Result (real numbers, sine,
`λ\*=0.25`, from `results/exp0A_QI_vs_learn/data.json`):

| width | QI (mpmath) | QI (fp64) | lstsq (fp64) | lstsq (mpmath) |
|---:|---:|---:|---:|---:|
| 173 | 1.7e-14 | 8.8e-11 | 1.6e-13 | 1.6e-15 |
| 205 | 3.0e-15 | 1.7e-10 | 6.6e-14 | 8.2e-16 |
| 269 | 2.3e-15 | 1.3e-10 | 1.4e-13 | 8.0e-16 |

Two punchlines, both important:

1. **Given the right geometry, plain least squares is as good as or better than
   the QI formula** — `lstsq(mpmath)` beats `QI(mpmath)` in **48/48** configs.
   Least squares directly minimizes the residual; the QI convolution is a fixed
   formula that's slightly suboptimal. So the clever convolution is *not* where the
   magic is.
2. **The hard part is the geometry, not the readout.** Once `(γ, x_m)` are correct,
   even fp64 least squares reaches ~1e-13. The outer weights are an easy linear
   solve. **This is the pivotal finding of the repo so far:** it reframes the open
   problem as "*can an optimizer discover the geometry (`γ ∝ N`, centers on a
   grid)?*" — because the readout is trivial once the geometry is right.

This is also precisely the paper's logic: §4.2 fixes the geometry and fits only
the output coefficients (a least-squares solve on Φ) to show MLPs *can* hit the
fp64 floor; §4.3 then uses the same Φ to diagnose *why trained nets don't*.

---

## 9. mpmath vs fp64 — "I thought the whole point was fp64"

This is the knot worth slowing down on.

**What "fp64" actually refers to.** The *model* is fp64: its weights are fp64
numbers, its forward pass, gradients, and evaluation all run in fp64. That is the
deliverable and it never changes. The construction's job is to produce the *fp64
numbers* `(γ, x_m, a_m, b)` that make an fp64 MLP accurate. Computing those
numbers is an **offline precomputation** — it is not part of the model.

**The `numpy.pi` analogy (this is the whole point).** `numpy.pi` is a single fp64
constant. Nobody computed it *in* fp64 by an fp64 algorithm; π was computed to
enormous precision by other means and then rounded once to the nearest fp64
double. Using `numpy.pi` does not "violate fp64." The QI coefficients are the same
kind of thing: constants for a specific MLP. Computing them in 30-digit mpmath and
then rounding to fp64 gives you the *best fp64 constant set* — it does not make the
model anything other than fp64.

**Why fp64 *computation* of the constants isn't good enough.** Two fp64 hazards in
the construction:

1. *Toeplitz conditioning.* As `λ` shrinks toward the high-precision regime, the
   Toeplitz matrix `T` becomes ill-conditioned; an fp64 solve loses digits. (This
   is why the fp64 path backs off to `λ=0.30` while mpmath can use `λ=0.25`.)
2. *Catastrophic cancellation in the convolution (the dominant one).* Recall the
   `c_j` reach `|c_0| ≈ 338` and **alternate in sign**. The convolution
   `a_m = Σ_j c_j g'(x_{m−j})` adds ~321 terms of magnitude ~hundreds whose true
   sum is `O(1)`. Adding big numbers of alternating sign to get a small answer is
   the textbook setup for **catastrophic cancellation**: you lose roughly
   `log10(300) ≈ 2.5` digits. fp64 has ~16 digits, so you floor out around
   `10^{-12}` — not `10^{-15}`.

So the **fp64 path is cancellation-limited at ~1e-12**, by arithmetic, not by the
method. The **mpmath path** does the Toeplitz solve and the convolution in 30-digit
arithmetic (so cancellation costs 2.5 of 30 digits, irrelevant), then rounds the
*final* `a_m, b` to fp64 — landing at true machine epsilon ~`2–3e-15`.

You can see both regimes directly in `results/exp00_sanity/summary.txt`
(target=sine):

| N | fp64 L∞ | mpmath L∞ |
|---:|---:|---:|
| 32 | 9.2e-12 | 1.7e-14 |
| 64 | 5.4e-12 | 2.8e-15 |
| 128 | 5.5e-12 | 2.0e-15 |
| 256 | 2.8e-12 | 1.4e-15 |

The fp64 column is stuck at the cancellation floor; the mpmath column rides down to
machine epsilon. **Same construction, same final fp64 model — only the precision of
the offline coefficient computation differs.**

**Practical rule** (encoded in the experiments): use **mpmath** when the QI
solution is a *fixed reference you compare against* (you want it exact — exp02
basin, exp03 ladder, exp04 Hessian). Use **fp64** when you're *training, sweeping,
or initializing* (you want speed and 1e-12 is plenty). The repo even mitigates the
fp64 floor with compensated summation (Kahan in the mpmath convolution; `math.fsum`
in the fp64 bias) so it gets as close to the floor as fp64 allows.

---

## 10. Current results, in depth

There are three implemented experiments (`exp00`, `exp01`, `exp0A`) plus the
`results/setup/` convergence probes. Together they establish the *foundations* the
paper's Section 4 rests on. Here's what each actually shows.

### 10.1 `results/setup/` — the construction works and scales

The convergence probes confirm the headline claim of Section 3: as width grows,
the QI construction's error falls geometrically and bottoms out at the precision
floor of its arithmetic (mpmath → ~1e-15; fp64 → ~1e-12). This is the empirical
version of "machine-precision interpolation is achievable," and it's verified in
the test suite too (`tests/test_construction.py::TestPrecisionFlag::
test_mpmath_path_reaches_machine_eps`).

### 10.2 `exp00` — it's not a numerics bug

Before blaming optimization for the training gap, rule out that the *floor itself*
is a numerical artifact. exp00 does this exhaustively (`results/exp00_sanity/`):

- **Construction baseline** (table in §9): fp64 floors at ~1e-12, mpmath at ~1e-15,
  cleanly and reproducibly. The floor is real and well-understood (cancellation),
  not a bug.
- **Readout solvers** (`summary.txt` section [2]): on the QI geometry,
  least-squares and SVD recover the outer weights to ~1e-13 with small weights
  (`v_max ≈ 6–10`), while naive QR can blow up (`v_max ≈ 2700`) and ridge with any
  nonzero penalty destroys precision. **Lesson: the readout solve is easy but you
  must use a stable, truncating solver** — which is exactly why `readout.py`'s SVD
  path now truncates tiny singular values.
- **Geometry matters enormously:** the same solve on the *interior-only* geometry
  (no halo) degrades to ~1e-4. The halo is not optional.
- **Conditioning, tanh stability, density sweeps:** all confirm the eval metric is
  trustworthy and `cond(Φ)` vs `cond(ΦᵀΦ)` behave as expected (squaring the
  condition number — why you never form the normal equations when you can avoid it).

Takeaway: **the precision floor is arithmetic, not a solver/eval artifact.** The
training gap is therefore a real optimization phenomenon, not a measurement bug.

### 10.3 `exp01` — the λ U-curve is real

exp01 sweeps `λ` across widths and targets, comparing full QI vs least-squares on
the same geometry (`results/exp01_lambda_tradeoff/`, plots
`consolidated_linf.png`). It confirms the Theorem-1 tradeoff empirically:

- A clear **U-shaped** error-vs-`λ` curve, minimized around **`λ ≈ 0.23–0.26`**.
- Below ~0.15: ill-conditioning (the diverging prefactors) dominates.
- Above ~0.5: aliasing dominates.
- The optimum is shared across targets and widths — i.e. `λ\*` is a property of the
  *method*, not the function. This is what licenses hard-coding `λ\* ≈ 0.25–0.30`.

Why it matters for the big question: unconstrained training lets `λ` drift toward 0
(because `γ` stays `O(1)` while `h→0`). exp01 shows that's precisely the
ill-conditioned side of the U — quantifying *why* drifting `λ` can't reach high
precision.

### 10.4 `exp0A` — geometry is the bottleneck (the key result)

Covered in §8.2. The 4-way comparison on identical geometry shows: (1) least
squares ≥ QI-formula given the geometry, and (2) with correct geometry even fp64
least squares hits ~1e-13. **Conclusion the repo is currently operating under: the
outer weights are a trivial linear solve; the entire difficulty is getting the
optimizer to discover the *geometry* — `γ ∝ N` and centers on a grid.** That is
the thesis that points the remaining experiments at the geometry (exp02 basin,
exp03 ladder) and at reparameterization/VarPro (exp08/exp09).

### 10.5 What is *not* done yet

`exp02`–`exp09` are scaffolded but unimplemented (docstring + `# TODO`). They are
the planned diagnostics that would fill in the paper's Section 4.3 story
(basin stability, the geometry ladder, the Hessian, Φ-conditioning, objective
shaping, noise, reparameterization, VarPro). See `docs/future_experiments.md`.

---

## 11. How this maps to Section 4 (Experiments) of the paper

Now the payoff. The paper's `QIs_workshop.pdf` Section 4 is organized exactly
around the construction-vs-training gap, and every part of it has a counterpart in
this repo. (Note: the *figures* in the PDF were produced from a fuller pipeline;
the repo's implemented experiments cover the foundations and some of §4.2, with
§4.3's diagnostics scaffolded as exp02–09.)

**The paper's framing (abstract + Fig. 1).** "Optimization, not expressivity, is
the bottleneck." Fig. 1 shows three panels that *are* the thesis: (left) QI kernels
with halo nodes outside `[−1,1]`; (middle) relative L₂ vs width — the explicit QI
interpolant rides down to the fp64 floor while trained MLPs plateau ~3 orders
higher; (right) mean `λ` vs width — QI plateaus at `λ\*`, trained networks drive
`λ → 0`. Your `results/setup/` and exp01 plots are the repo's versions of the
middle and right panels.

**§4.1 — "Direct training fails to reach machine precision" (Fig. 3).** Across
optimizers (Adam, BFGS, LBFGS, LM, SSBroyden, …), activations (GELU/tanh/sech/…),
and targets, training shows geometric gains at small width then **saturates far
above fp64**, with widening giving diminishing/non-monotone returns.
- *Repo counterpart:* not yet implemented as a training experiment — this is the
  motivation the diagnostic experiments serve. The infrastructure exists
  (`src/training/`, the multi-stage Adam→LBFGS loop, the metric schema), so this is
  the most direct "fill in Section 4.1" task. exp00 establishes that the gap these
  curves show is genuinely optimization, not numerics.

**§4.2 — "MLPs can realize quasi-interpolants" (Fig. 4).** Fix the grid and
bandwidth `γ`, so training reduces to fitting only the **output coefficients** —
i.e. a least-squares solve on the Φ features. Result: the learned coefficients
match the samples at grid points, and the interpolant converges geometrically to
the fp64 floor (for Gaussian and sinc kernels). Trained nets also place centers
outside the domain (the halo).
- *Repo counterpart:* **this is exactly exp0A and the `readout.py` Φ machinery.**
  "Fix geometry, fit output coefficients by least squares" is `build_phi` +
  `solve_readout`. exp0A's finding (fp64 lstsq ≈ 1e-13 on correct geometry) is the
  repo's version of Fig. 4c. The §8.1 "two solves" distinction is precisely the
  §3-vs-§4.2 distinction in the paper.

**§4.3 — "Empirical diagnosis" (Figs. 5–7).** Three sub-findings, the three
"violations":
- *Weight-magnitude scaling (Fig. 5).* Explicit QI grows `γ` with width (keeping
  `λ=γh` constant) and keeps outer weights `O(1)`; trained nets keep `γ=O(1)` (so
  `λ→0`) and use much larger outer weights — relying on cancellation among
  overlapping features. → **violation #1 (γ scaling) and #2 (weight blowup).**
  - *Repo counterpart:* the metric schema already logs `γ`, `λ=γh`, and outer-weight
    norms at every eval step (`src/training/metrics.py`), so this plot drops out of
    any training run; exp02/exp03/exp08 are designed to produce it.
- *Rank saturation / node utilization (Fig. 6).* Freeze the hidden layer, form the
  feature system `Φa=b` with `Φ_ij = tanh(γ_j(x_i−x_j))`, and use OMP to count how
  many neurons you can delete while keeping error below a threshold. Trained nets
  are **much more compressible** (a few neurons do the work — rank saturation); QI
  nets spread the work across all neurons. → **violation #3 (rank saturation).**
  - *Repo counterpart:* `Φ` is `readout.build_phi`; the metric schema already
    computes `feature_rank` and `feature_stable_rank` from the SVD of `Φ`. The OMP
    pruning curve is the planned exp03/exp05 analysis. This is the *same Φ* you
    asked about — here used as a diagnostic rather than a solver.
- *Endpoint Hessian curvature alone doesn't explain it (Fig. 7).* QI solutions
  don't have systematically larger top-Hessian curvature than trained ones, so the
  gap isn't simple endpoint conditioning — it's more consistent with a
  *representational* mismatch (how width is used). → motivates exp04 (Hessian) and
  the reduced-coordinate view (exp09 VarPro).

**§Discussion.** "The first explicit MLP construction achieving machine precision
with `log(1/ε)` parameter scaling, realizable in fp64; optimization, not
expressivity, is the bottleneck." Everything in this repo is in service of that one
sentence: Section 3 (the construction, = `qi_mpmath.py`) proves the representation
*exists* in fp64; Section 4 (the experiments, = `experiments/` + `readout.py`'s Φ
machinery) is the campaign to show training can't find it *yet* — and to figure out
how to make it.

---

## 12. One-paragraph summary

`qi_mpmath.py` is the paper's **Section 3 construction**: it builds a specific
fp64 `tanh` MLP that interpolates a target to machine precision. It does so by (1)
solving a **Toeplitz** system for target-independent **cardinal coefficients**
`c_j` — that solve *is* the cardinal/interpolation condition written at grid
points, Toeplitz because the uniform grid + distance-only kernel make it
shift-invariant; (2) **convolving** `c_j` with the target's derivative to get outer
weights `a_m` (derivative because `sech² = tanh'`); (3) pinning a **bias** at the
boundary. The **Φ matrix / least-squares** solve is a *separate* tool — it fits the
outer weights of a *fixed-geometry* MLP directly, and it's both an alternative to
the convolution (exp0A) and the paper's diagnostic for trained networks (§4.2,
§4.3). **mpmath vs fp64** is not a contradiction: the model is always fp64; mpmath
just computes the offline coefficient *constants* accurately enough to dodge the
catastrophic cancellation that floors the fp64 computation at ~1e-12, yielding fp64
constants good to ~1e-15 — the `numpy.pi` move. And the whole repo is the empirical
build-out of the paper's thesis: the machine-precision MLP *exists* (Section 3 ✓);
the open problem is teaching an optimizer to *find its geometry* (Section 4, in
progress).
