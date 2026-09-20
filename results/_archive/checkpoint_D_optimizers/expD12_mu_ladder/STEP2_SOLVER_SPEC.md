# Step 2: the least-squares solver, full spec

**Status: draft, except the section marked APPROVED. Sources: expD08-expD13 plus `results/checkpoint_D_optimizers/expD09_2nd_order_regime/DAMPED_GAUSS_NEWTON.md`. Every number is measured; anything asserted without measurement is marked *untested*.**

**This is the only writeup of expD12 (the mu ladder) and expD13 (the drift ladder); those experiments have no `*_results.md` of their own.** It is also where the mu-schedule control rules live (IV.5, IV.6) and where the guard that expD14 later violated is derived (V.6, V.7). Read IV.5-IV.7 and V.6-V.8 before designing any damping schedule.

---

# APPROVED (Sam, 2026-07-31): the linear block separates by rows

## 1. The linear block is blocked, and each block is solved on its own

Approved 2026-07-31.

A weight matrix $W \in \mathbb{R}^{o\times h}$ mapping features $\phi \in \mathbb{R}^{h}$ to outputs has $o\cdot h$ parameters, but its least-squares problem does **not** have $o\cdot h$ coupled unknowns. Writing $w_j$ for row $j$ of $W$:

$$\sum_i \|W\phi_i - y_i\|^2 \;=\; \sum_{j=1}^{o}\;\sum_i \big(w_j^{\top}\phi_i - y_{ij}\big)^2 .$$

Each row appears in exactly one inner sum, so this is $o$ **independent** least-squares problems that all share the *same* design matrix $\Phi \in \mathbb{R}^{n\times h}$. Equivalently the Jacobian is $J = I_o \otimes \Phi$ and the Gram is block diagonal with $o$ identical blocks $\Phi^{\top}\Phi$.

**Consequences.**
- One factorization of $\Phi$ serves every block. The blocks can be solved in parallel or sequentially against that shared factor; there is nothing to combine afterwards.
- State is $h^2/2$ (one triangular factor) plus $h\cdot o$ (the transformed right-hand sides) — **not** $(oh)^2/2$.
- The governing size is $h$, the **input** width of the layer. The output dimension $o$ enters only linearly.

**Verified** (`experiments/expD12_mu_ladder/scale_audit.py`, $h{=}60$, $o{=}7$, $\kappa{=}10^{13}$): the shared-factor route reaches residual $1.66\times10^{-15}$ using 2,250 floats; a brute-force solve of the full $(no)\times(oh)$ Kronecker system reaches $1.14\times10^{-15}$ using 88,410 floats. Same answer, $39\times$ less state, and the ratio grows like $o^2$.

**The trap.** Treating the layer as one least-squares problem in all $o\cdot h$ unknowns costs $(oh)^2/2$ state. At $o \sim h$ that is $h^4$ and it is fatal. The separation above is exact and free, but it has to be taken deliberately.

**Scope limit.** This separation requires the model output to be *linear* in the block, i.e. the block is the final linear map before an MSE loss. It does **not** hold for a hidden layer: there the Jacobian rows are $G_i \otimes \phi_i^{\top}$ (Khatri–Rao, not Kronecker), which does not block-diagonalize. It also does not hold for cross-entropy, where the Gauss–Newton matrix is $J^{\top}HJ$ with a softmax Hessian $H$ that couples the outputs.


---


---

# Part I — Context

## The goal

Close the gap between the explicit QI construction (~$10^{-15}$) and what training achieves (~$10^{-10}$). Success is defined in `CLAUDE.md`: as width grows, eval relative $L_2 \le 10^{-13}$ and $L_\infty$ consistent with machine epsilon, over 3–5 seeds on the target-family matrix, **without** initializing from the constructive solution.

## The three-step program

1. **Step 1** — a general-purpose optimizer (Adam). Done, trivial.
2. **Step 2** — an iterative least-squares solver that scales the way Adam does. *This document.*
3. **Step 3** — lobotomize Adam and stitch the solver in: decide which parameters qualify, schedule the damping, and certify block structure.

## The hypothesis step 2 serves

During training there exists a subset $L \subseteq P$ of parameters that is **locally linear** in the model output: $f(x) \approx h(x)\,v$ for $v = \theta_L$. Solving that least-squares subproblem is a well-defined, well-understood task. The catch is that the linearity is only local, and it is worst at the start of training.

## The role of step 2, and how it fits

Step 2 owns exactly one thing: **given $L$, its certified block structure, and a damping $\mu$, produce the parameter update for $L$.** It does not choose $L$, does not schedule $\mu$, and does not certify blocks — those are step 3.

Inside the training loop:

```
for each training step:
    one forward + backward pass
    Adam updates  P \ L        (unchanged, standard)
    STEP 2 updates L           (damped Gauss-Newton, this document)
...at the end, once:
    freeze the geometry
    STEP 2 terminal solve on L (to machine epsilon / the noise floor)
```

Damping $\mu$ is the confidence dial. Large $\mu$ makes the step a scaled gradient step; small $\mu$ makes it the exact least-squares solve. Step 2 must behave correctly across that whole range and degrade gracefully, because early in training the local-linearity assumption is poor.

---

# Part II — Requirements, and why each exists

| # | Requirement | Why it exists |
|---|---|---|
| R1 | Iterative | The per-step update must be interruptible and budgetable |
| R2 | $O(d)$ memory, Adam's class | Adam stores $2P$; the solver may not blow that up. $d=\lvert L\rvert$ |
| R3 | $O(1)$ iterations per step | Each iteration costs real work; the update must stay the same order as an Adam step |
| R4 | Per-step cost $\sim$ one forward-backward | Same reason; a $10\times$ optimizer is not adoptable |
| R5 | Robust to batching, down to $b = d/64$ | Real training does not get to choose $b$ |
| R6 | Front-loaded | Most of the progress must arrive in the first few iterations |
| R7 | Robust to $\Phi$ drifting between steps | Adam is moving the geometry underneath |
| R8 | $L$ may change membership between steps | Step 3 re-decides; no state may be keyed to a fixed $L$ |
| R9 | Reaches machine epsilon or the noise floor **once**, at the end | This is where the precision goal is actually delivered |
| R10 | Graceful failure, no regime-dependent behaviour | An optimizer with cliffs is not adoptable |

**Conceded, deliberately:** the per-step phase does **not** reach machine epsilon. It is not supposed to. All the precision comes from the terminal solve.

---

# Part III — The accounting, in brief

With $P$ total parameters, $d = \lvert L\rvert$, $n$ rows per step, $T$ iterations per step, $F$ = flops of one training step:

$$\textbf{per step: } \frac{4Tnd}{F} = \frac{2Td}{3P}\ \text{of an Adam step} \qquad\qquad \textbf{terminal: } \Theta\!\big(\max_\beta h_\beta^2\big)\ \text{memory}$$

The per-step cost is **independent of batch size** and is 0.3%–3.3% for any realistic $\lvert L\rvert/P$. The terminal solve is governed by the **largest certified-decoupled block**, never by $\lvert L\rvert$: single-GPU ceiling $h_{\max}\approx1.4\times10^5$, roughly $10^6$ across a 64-GPU pod. Representative: SD-1.5 U-Net final conv 33 MB and 0.0001% of the run; a wide U-Net 531 MB.

---

# Part IV — The method

This part is method only. No justifications; those are Part V.

## IV.0 Notation

| symbol | meaning |
|---|---|
| $P$ | all model parameters |
| $L$ | the locally-linear subset, $d = \lvert L\rvert$ |
| $\beta$ | index over certified-decoupled blocks of $L$ |
| $J$ | Jacobian of the model output w.r.t. $\theta_L$, $n \times d$ |
| $r$ | residual $f(x)-y$, length $n$ |
| $\mu$ | damping (Tikhonov / Levenberg–Marquardt) |
| $\sigma_1$ | largest singular value of $J$ |
| $\alpha$ | $\sqrt{\mu}/\sigma_1$ — the **scale-free damping**; $\kappa_\mu = 1/\alpha$ exactly |
| $\phi^{(i)}_{\ell-1}$ | input to layer $\ell$ for sample $i$ (cached by the forward pass) |
| $\delta^{(i)}_\ell$ | backprop delta at layer $\ell$ for sample $i$ |
| $T$ | LSQR iterations this step |
| $c$ | number of stored reorthogonalization vectors |

## IV.1 What the forward-backward pass must retain

Ordinary training frees the per-sample deltas as soon as they are reduced into the parameter gradient. **Step 2 requires them retained** for the layers touching $L$:

- $\phi^{(i)}_{\ell-1}$, shape $(n, h_\ell)$ — already cached for the backward pass, no extra cost.
- $\delta^{(i)}_\ell$, shape $(n, o_\ell)$ — must be kept.

For a final-layer block this is free: $\Phi$ *is* the cached input activation and $\delta$ *is* the residual.

## IV.2 The Jacobian operators

Never form $J$. Both matvecs come from the cache:

$$(Jv)_i \;=\; \sum_\ell \big(\delta^{(i)}_\ell\big)^{\!\top} V_\ell\, \phi^{(i)}_{\ell-1} \qquad\qquad (J^{\top}u)_\ell \;=\; \sum_i u_i\, \delta^{(i)}_\ell \big(\phi^{(i)}_{\ell-1}\big)^{\!\top}$$

where $V_\ell$ is $v$ reshaped to layer $\ell$'s weight shape. Each costs $2nd$ flops and **no network traversal**.

```python
def Jv(V, phi, delta):                       # -> (n,)
    return sum(((phi[l] @ V[l].T) * delta[l]).sum(1) for l in layers)

def JTu(u, phi, delta):                      # -> list of layer-shaped arrays
    return [ (delta[l] * u[:, None]).T @ phi[l] for l in layers ]
```

## IV.3 Damping, applied implicitly

The damped problem $\min_v \|Jv-r\|^2 + \mu\|v\|^2$ is ordinary least squares on the augmented system

$$\begin{pmatrix} J \\ \sqrt{\mu}\,I \end{pmatrix} v \;\approx\; \begin{pmatrix} r \\ 0\end{pmatrix}$$

**Never materialize the $d\times d$ identity block.** Apply it through the operator:

```python
sq = sqrt(mu)
def A_mv(v):   return concat([ Jv(v),  sq * v ])          # length n + d
def A_rmv(u):  return JTu(u[:n]) + sq * u[n:]             # length d
```

## IV.4 The per-step solver — LSQR with damping, $c = 0$

Standard Golub–Kahan bidiagonalization with Givens rotations (Paige–Saunders). State: **5 vectors of length $d$**. No factor, no preconditioner, no cross-step state.

```
input: A_mv, A_rmv, rhs = r, d, mu, alpha, T_max
u = concat([rhs, zeros(d)]);  beta = ||u||;  u /= beta;  b0 = beta
v = A_rmv(u);                 alph = ||v||;  v /= alph
w = v.copy();  x = zeros(d)
phibar = beta;  rhobar = alph;  anorm2 = alph**2

for it in 1..T_max:
    un   = A_mv(v) - alph * u
    beta = ||un||
    if beta <= brk * sqrt(anorm2): break            # breakdown guard, brk = 1e-14
    u    = un / beta

    vn     = A_rmv(u) - beta * v
    alph   = ||vn||
    anorm2 += alph**2 + beta**2
    if alph <= brk * sqrt(anorm2): break
    v      = vn / alph

    rho    = hypot(rhobar, beta)                    # Givens
    cs, sn = rhobar/rho, beta/rho
    theta  = sn * alph
    rhobar = -cs * alph
    phi    = cs * phibar
    phibar = sn * phibar

    x += (phi / rho) * w                            # the iterate
    w  = v - (theta / rho) * w

    # --- stopping test (see IV.6)
    anorm = sqrt(anorm2)
    atr   = abs(phibar * alph * cs)                 # ||A^T r - mu x||, free
    test2 = atr / (anorm * phibar)
    if test2 <= alpha: break

return x
```

The update applied to the model is $\theta_L \leftarrow \theta_L - x$ (sign per the residual convention $r = f-y$).

## IV.5 Setting $\alpha$ and $\mu$

Parameterize by $\alpha$, never by $\mu$ directly; $\mu = (\alpha\sigma_1)^2$.

$\sigma_1$ is available for free: after the first level's solve, LSQR's running `sqrt(anorm2)` is a $\|J\|_F$ estimate, and the largest Ritz value of the bidiagonal approximates $\sigma_1$. Alternatively use Marquardt scaling (IV.9), which removes the need for $\sigma_1$ entirely.

$\alpha$ descends over training. **Step 3 owns the schedule**, subject to the hard floor in IV.7.

## IV.6 The three control rules

**(a) Stopping rule — within a level.**

$$\text{stop when}\quad \text{test}_2 \;=\; \frac{\|A^{\top}r - \mu x\|}{\|A\|\,\|r\|} \;\le\; \alpha$$

This is LSQR/LSMR's standard `atol` test. All three quantities are already computed.

**(b) Iteration cap.**

$$T_{\max}(\alpha) \;=\; \min\big(T_{\text{hard}},\; 5\,\alpha^{-3/4}\big)$$

$T_{\text{hard}}$ is your per-step compute budget. **If the cap binds before the stopping test fires, that is the handoff signal** — stop descending $\alpha$ and go to the terminal solve.

**(c) Damping floor — the drift/noise guard.**

$$\alpha \;\ge\; r_{\text{entry}} \;=\; \frac{\|y - \Phi_k w\|}{\|y\|}\quad\text{measured at level entry, before solving}$$

**Never damp finer than the residual you walked in with.**

## IV.7 The per-step algorithm, assembled

```
# once per training step
fwd_bwd()                                  # retain phi and per-sample delta
adam_update(P \ L)

r_entry = ||y - f(x)|| / ||y||
alpha   = max(alpha_sched, r_entry)        # (c) drift/noise floor
mu      = (alpha * sigma1)**2
T_max   = min(T_hard, 5 * alpha**-0.75)    # (b) cap

for each certified block beta:             # exact decoupling; independent
    x_beta = lsqr_damped(J_beta, r, mu, T_max, stop = test2 <= alpha)
    theta[beta] -= x_beta

if cap_bound_before_stop:  signal_handoff()
```

Blocks sharing a design matrix ($J = I_o\otimes\Phi$, a multi-output layer) share the Krylov work and require identical $T$.

## IV.8 The terminal solve

Run **once**, at the end, per certified block.

**Preconditions.** The geometry must be **frozen** — Adam stopped on $P\setminus L$. Non-negotiable.

**Method.** LSQR with $\mu = 0$ and **full reorthogonalization**, one-sided (V only):

- After computing $v_{\text{new}}$, orthogonalize against all stored $v$'s by two-pass classical Gram–Schmidt (CGS2), vectorized:
  ```python
  z = z - Q.T @ (Q @ z)
  z = z - Q.T @ (Q @ z)        # twice is enough
  ```
- Do **not** reorthogonalize $u$. Store only $V$.
- Breakdown guard `brk = 1e-14`.
- Budget: $\approx 0.9\,r$ iterations, $r$ = numerical rank.
- State: $r\cdot d$ floats.
- Warm start from the ladder's endpoint is fine but optional.

**Getting the rows, when $b < 4d$.** The terminal solve needs $n \gtrsim 4d$ rows of Jacobian to resolve $d$ unknowns. If the batch is smaller than that, they must be accumulated. Do this in a dedicated accumulation phase **after $L$ and the geometry are both frozen** — never as running state during training (see IV.8.1). Two options:

- **Raw rows** — buffer $4d$ rows of $[\,J\;|\;r\,]$. State $4d\cdot d$. Simple; use it when it fits.
- **Streaming QR** — fold each batch into an upper-triangular factor $R$ and discard the rows. State $d^2/2$, independent of how much data you have seen, and **exact at any batch size including $b=1$**. Then run the terminal solve against $R$ instead of the raw rows ($d\times d$ operator instead of $n\times d$).

```python
M = zeros((0, d+1))
for each batch:
    M = vstack([M, hstack([J_batch, r_batch[:, None]])])
    if M.shape[0] >= d + 1:                 # ONLY factor a TALL buffer
        M = qr(M, mode='r')[:d+1]
R, z = M[:d, :d], M[:d, d]
```

**Never accumulate the Gram $\sum J^{\top}J$ instead.** Same state, but it squares the condition number ($10^{28}$ at $\kappa=10^{14}$) and dies at $10^{-4}$.

## IV.8.1 Membership and freezing

- **During the per-step phase**, $L$ may change freely. The solver holds no state keyed to $L$ — each step builds a fresh Krylov space from the current residual — so there is nothing to invalidate. *(Argued from construction; not measured.)*
- **Before the terminal solve**, $L$ must be **frozen**, together with the geometry. Row accumulation begins only after both are fixed, because a triangular factor or row buffer indexed by $L$ is invalidated by any membership change.
- **When a coordinate crosses the boundary** (enters or leaves $L$), its Adam moment estimates are stale. Re-initialization policy is unspecified here and belongs to step 3.

## IV.9 Optional: per-coordinate damping (Marquardt)

Replace $\mu I$ with $D = \mu\,\mathrm{diag}(J^{\top}J)$. Scale-invariant, costs nothing, removes the need for $\sigma_1$. If step 3 supplies per-coordinate confidences $\mu_i$, use $D = \mathrm{diag}(1/\mu_i)$ **with a clamp on $\max_i \mu_i$** (IV.10).

## IV.10 Constants

| constant | value | role |
|---|---|---|
| stopping tolerance | $\text{test}_2 \le 1.0\,\alpha$ | within-level stop |
| cap multiplier | 5 | safety net |
| cap exponent | $3/4$ | inside the measured CI; range $0.59$–$0.84$ |
| breakdown guard | $10^{-14}$ | plateau $10^{-14}$–$10^{-16}$ |
| reorth passes | 2 (CGS2) | terminal solve |
| terminal budget | $0.9\,r$ | iterations |
| Krylov state | 5 vectors | per-step |

---

# Part V — Why every decision is what it is

## V.1 Why damping is the whole idea

Damping is not a fudge. The damped system's singular values are $\sqrt{\sigma_i^2+\mu}$, so **every $\sigma_i \ll \sqrt\mu$ collapses onto the single value $\sqrt\mu$** — the spectral tail becomes one cluster.

That matters because of expD11's central law: **the stored orthogonality a Krylov solver needs is set by the number of distinct spectral scales, not by $d$ or $\kappa$.** Measured on synthetic spectra with identical $d$, $r$, $\kappa$ and target energy: a 2- or 4-level spectrum reaches $10^{-15}$ with $c=0$ — zero stored state — while a gapless spectrum needs $c=r$. QI feature matrices are gapless (median $\sigma_i/\sigma_{i+1} = 1.03$ at $N{=}768$), which is why the undamped problem is hard.

Damping manufactures the clustering. Measured: $c^\ast = 0$ for every $\alpha$ down to a damped optimum of $3\times10^{-6}$. **That is why $O(d)$ state and $O(1)$ iterations are achievable at all.**

## V.2 Why $\alpha$, not $\mu$

$\kappa_\mu = 1/\alpha$ **exactly** — measured to two digits on all matrices tested. $\mu$ itself is not transferable: $\sigma_1$ spanned 0.76 to 4196 across the test set. $\alpha$ is dimensionless and portable.

## V.3 Why the per-step phase cannot reach machine epsilon

Two independent measurements:

- Without stored orthogonality, convergence is **algebraic**: $\epsilon \sim T^{-1.6}$. Reaching $10^{-14}$ would need $T\sim10^8$.
- Every preconditioner tested **caps** around $10^{-6}$ regardless of orthogonalization — block-Jacobi at two block sizes and three truncation levels, pivoted-QR with backward-stable triangular solve, and the structured QI cardinal-coefficient operator. Loosening the truncation makes it *worse* ($5\times10^{-2}$). And $\kappa(\Phi M)\ge\kappa(\Phi)/\kappa(M)$ forces any effective preconditioner to itself span $10^{14}$.

Hence the concession is structural, not a tuning failure.

## V.4 Why $\text{test}_2 \le \alpha$

Scored against an oracle that stops at $1.3\times$ the exact damped optimum, over 75 levels on 4 matrices, static and drifting:

| rule | acc (med/p90/worst) | work (med/p90) |
|---|---|---|
| $\text{test}_2\le0.1\alpha$ | 0.78 / 0.97 / 1.0 | **2.00** / 3.19 |
| **$\text{test}_2\le\alpha$** | **1.00 / 1.34 / 1.6** | **1.00 / 1.46** |
| $\text{test}_2\le10\alpha$ | **3.15** / 4.31 / 5.8 | 0.10 / 0.50 |
| fixed budget $3\alpha^{-0.7}$ | 0.78 / 1.00 / 1.1 | **8.62 / 31.97** |
| held-out plateau | 2.14 / 4.17 / 5.8 | 0.11 / **3.79** |

Under drift the winner holds at acc 1.00/1.20/1.9, work 1.00/1.45. It sits on the oracle in both regimes. The tolerance scales with $\alpha$ because a level converges to accuracy $\approx\alpha$ (measured slope 0.87–1.36 across all seven matrices — each decade of $\alpha$ buys a decade of accuracy).

## V.5 Why the cap exponent is $3/4$, and how much to trust it

$\alpha^{-1/2}$ is the textbook $O(\sqrt\kappa)$ CG bound and would be the natural default — **the data rejects it.** Fitting per matrix (pooling was wrong: the constants differ 40x):

| matrix | slope | 95% CI |
|---|---|---|
| QI-2D N=2048 | -0.587 | [-0.631, -0.542] |
| random 2-D spectrum | -0.603 | [-0.637, -0.570] |
| QI-2D N=2048 (drift) | -0.663 | [-0.764, -0.561] |
| QI-1D N=256 | -0.736 | [-0.749, -0.723] |
| QI-1D N=256 (drift) | -0.741 | [-0.751, -0.731] |
| random gapless d=2048 | -0.839 | [-0.861, -0.817] |
| random gapless d=2048 (drift) | -0.840 | [-0.873, -0.807] |

Mean $-0.70$, CI $[-0.77,-0.63]$; $-0.5$ falls outside every individual interval. The $\sqrt\kappa$ bound is a worst-case Chebyshev result that is loose exactly for gapless spectra, and the ordering confirms the mechanism — the gapless random matrix is worst ($-0.84$, near $\kappa^1$), the lower-effective-rank 2-D cases best.

**But the exponent should be read as $3/4$, not as a fitted decimal.** Four matrices with a 40x spread in their constants do not support two significant figures. $-3/4$ sits inside the CI, near the middle of the per-matrix range, and errs on the generous side. Measured against the 81 recorded levels, $\min(4096, 5\alpha^{-3/4})$ **cuts none of them short**, against 6/81 at $-0.6$ — so it behaves as a safety net rather than as a binding budget, which is its only job.

Practically the choice is second-order anyway: over the $\alpha$ range in use the candidate exponents differ by 1.1x-2.5x, and $T_{\text{hard}}$ dominates below $\alpha\approx10^{-5}$.

## V.6 Why the damping floor $\alpha \ge r_{\text{entry}}$ — the most important guard

expD13 ran the ladder with the geometry drifting and converging (perturbation $\eta$ tied to $\alpha$). It exposed a failure mode invisible on a frozen $\Phi$: **the observable error and the true error diverge when drift outruns damping.** At the handoff level:

| $\Phi$ | drift | observable | actual | ratio |
|---|---|---|---|---|
| QI-1D | $10\alpha$ | 2.3e-7 | 2.7e-7 | 1.2 |
| QI-2D | $\alpha$ | 3.4e-8 | 4.8e-7 | **14** |
| QI-2D | $10\alpha$ | 2.9e-4 | 9.0e-3 | **31** |
| random d=2048 | $\alpha$ | 5.0e-5 | 9.8e-1 | **19,849** |

The random matrix reported $5\times10^{-5}$ having made *essentially no true progress*. Mechanism: $\|w\| \sim \|y\|/\sigma_{\min}$ is enormous at $\kappa=10^{14}$, so a $10^{-5}$ rotation of the feature space produces an $O(1)$ output change. **There is no error signal warning you.**

The fix is free. The residual at level entry estimates the drift to within $2\times$ over four decades (QI-2D, $r_{\text{entry}}/\eta$ = 2.14, 1.21, 1.05, 0.75, 0.50).

**And it unifies three guards into one.** $r_{\text{entry}}$ is bounded below by *every* irreducible error source:
- geometry drift $\eta$ → $r_{\text{entry}} \gtrsim \eta$
- label noise $\sigma$ → $r_{\text{entry}} \gtrsim \sigma$ (consistent with the measured $0.27\sigma$ statistical floor)
- approximation error of the feature space → $r_{\text{entry}} \gtrsim \epsilon_{\text{approx}}$

So the single rule $\alpha \ge r_{\text{entry}}$ prevents over-solving against all three, with no extra machinery and no hyperparameter.

## V.7 Why $\mu$ should *not* track the geometry noise

I hypothesized $\mu$ should be matched to the drift as a variance budget. **Measured false.** Sweeping $\eta \in \{0, 0.1\alpha, \alpha, 10\alpha\}$ shows no sweet spot at matching — less drift is monotonically better, and the terminal solve washes out the difference entirely. What survives is only the one-sided version: don't drive $\alpha$ *below* the drift (V.6).

## V.8 Why the terminal solve is separate, frozen, and full-reorth

- **Separate**, because per-step damping structurally cannot reach the floor (V.3).
- **Frozen**, because expD08 measured that after an exact solve a single Adam step re-injects $\|v\|\eta$ of error. Spending the terminal solve while the geometry still moves destroys it.
- **Full reorthogonalization**, because plain LSQR against the same operator caps at $10^{-6}$ while full reorth reaches $2\times10^{-15}$ — measured side by side.
- **One-sided (V only)**, because it matches two-sided accuracy exactly (2.57e-16 vs 2.92e-16 on QI-2D N=2048; 2.12e-15 vs 2.05e-15 on random d=2048) at **5–7× less memory** and ~1.5× faster.
- **$0.9r$ iterations**, because the terminal cost is $0.87$–$0.93\,r$ across 1-D, 2-D, random, and $d$ from 261 to 2060 — a 7% spread, the tightest law in the dataset.
- **Full-reorth LSQR rather than an SVD**, because it uses $2h^3$ flops against the SVD's $20h^3$ and reaches the same floor.

**It works.** All seven test matrices reach or beat their truncated-SVD floors — $2.8\times10^{-15}$ to $4.9\times10^{-13}$ — and it does so **regardless of what the drift did during the ladder** (verified at $\eta = 0, 0.1\alpha, \alpha, 10\alpha$).

## V.9 Why the breakdown guard is load-bearing

With reorthogonalization the Krylov space is exhausted after $\sim r$ steps; the next vector is pure rounding noise and normalizing it destroys the iterate. Measured on the terminal solve:

| guard | random d=512 | QI-1D N=256 | random d=2048 |
|---|---|---|---|
| $10^{-12}$ | 9.7e-13 | 4.1e-14 | 2.6e-13 |
| **$10^{-14}$** | **2.7e-15** | **3.0e-15** | **2.1e-15** |
| $10^{-16}$ | 2.0e-15 | 3.0e-15 | 2.7e-15 |
| $0$ | NaN | NaN | NaN |

$10^{-12}$ stops at $\sim0.79r$ and leaves the smallest directions unresolved — **two orders lost to a single constant.**

## V.10 Why batching is not a problem

Damping makes a rank-deficient batch well-posed: $(J_b^{\top}J_b + \mu I)$ is invertible even when $b < d$. Measured with repeated damped steps on fresh batches at $O(d)$ state: $b{=}d$ gives 1.8e-4 and $b{=}d/64$ (7 rows) gives 9.0e-4 — **5× degradation across a 64× batch reduction**. Unchanged under drift.

This also replaces the ad-hoc $\tau \le b/2$ clamp the undamped streaming solver needed; damping supplies the bound from the objective instead of as a hack.

If a triangular factor is being maintained, streaming QR is exact at **any** batch size including $b{=}1$ (5.15e-15 vs an 8.80e-15 full-rows reference), whereas Gram accumulation at identical state dies at $10^{-4}$ because it squares $\kappa$ ($10^{28}$ at $\kappa=10^{14}$).

## V.11 Why the Jacobian is free, and the one condition

Verified to machine precision against autograd: $Jv$ agrees to 9.8e-17 and $J^{\top}u$ to 4.2e-16 **for a random $u$, not the residual**. The mechanism is $\partial f_i/\partial W_{\ell,jk} = \delta^{(i)}_{\ell,j}\phi^{(i)}_{\ell-1,k}$ with both factors cached.

**The condition:** this requires a scalar output per sample, so that $\delta^{(i)}_\ell = r_i \cdot \partial f_i/\partial a_\ell$ and $r_i$ divides out. With genuinely coupled multi-output the cached delta is $\sum_m r_m\,\partial f_m/\partial a_\ell$ — already contracted, not recoverable. Step 3's block-decoupling oracle is what supplies the separation.

## V.12 Why blocks must be *certified*, not guessed

Solving blocks independently is exact **only** when the off-block-diagonal is identically zero. Blocking a genuinely coupled problem is block-coordinate descent, and with *exact* per-block solves it stalls:

| | 1 sweep | 10 sweeps | 60 sweeps | floor |
|---|---|---|---|---|
| QI-1D, blocks of 64 | 1.4e-1 | 9.9e-3 | **3.9e-3** | 8.8e-15 |
| QI-1D, **full** | 1.3e-4 | **2.8e-15** | 2.8e-15 | 8.8e-15 |
| random, blocks of 128 | 5.0e-6 | 1.3e-6 | **7.1e-7** | 3.1e-15 |
| random, **full** | 1.5e-4 | **1.1e-15** | 1.1e-15 | 3.1e-15 |

Twelve orders short and not converging. **A wrong certification costs ~9 orders with no error signal.**

## V.13 Why per-coordinate damping is optional and needs a clamp

Scalar $\mu I$ commutes with $J^{\top}J$, which is exactly why the tail collapses to a point. A general diagonal does not commute and the tail smears into a band. Measured, holding the geometric-mean damping fixed:

| spread $\max\mu_i/\min\mu_i$ | $\kappa$ | best accuracy |
|---|---|---|
| 1 (scalar) | 4.4e4 | 3.1e-6 |
| $10^4$ | 4.4e5 | 4.0e-5 |
| $10^8$ | 4.2e6 | 1.4e-4 |
| Marquardt $\mu\,\mathrm{diag}(J^{\top}J)$ | 4.4e4 | 3.1e-6 |

The governing law is $\kappa_D \approx \sigma_1\sqrt{\max_i \mu_i}$ — **the single most-trusted coordinate sets the conditioning for all $d$ of them.** Hence the clamp. Marquardt scaling is free and is the recommended default.

## V.14 Why stateless

$L$ may change membership between steps (R8). Any factor or preconditioner keyed to $L$ is invalidated when it does — and at $O(d^2)$ neither fits anyway. Restarting a fresh Krylov space each step costs nothing here because $T$ is $O(1)$ by design. Statelessness is a *consequence* of the design, not a compromise.

---

# Part VI — Tips, tricks, and warnings

## Tricks worth keeping

1. **The Jacobian action is free** — but only if you retain per-sample deltas instead of reducing them straight into the gradient.
2. **Apply damping through the operator**, never by stacking $\sqrt\mu I$ (238 MB at $d=2060$, rebuilt every level).
3. **Vectorize reorthogonalization** as $z \mathrel{-}= Q(Q^{\top}z)$ twice — 12× faster than a per-vector loop at $d=2048$, agreeing to 1.9e-15.
4. **One-sided reorthogonalization** in the terminal solve: same accuracy, 5–7× less memory.
5. **Solve against $R$, not the rows**, when a factor is maintained: $d\times d$ instead of $n\times d$, and $R$ holds all history.
6. **One factorization serves all blocks** that share a design matrix. 196,608 blocks cost the same as 4.
7. **$\sigma_1$ comes free** from LSQR's running `anorm` / largest Ritz value — or use Marquardt scaling and skip it.
8. **The cap binding is your handoff trigger.** No separate detector needed.

## Warnings

1. **Never form the Gram.** It squares $\kappa$ and dies at $10^{-4}$. Streaming QR at the same state holds $5\times10^{-15}$.
2. **Never trust the residual as a progress measure while the geometry moves.** Up to 19,849× optimistic, silently. Use $\alpha \ge r_{\text{entry}}$.
3. **Never run the terminal solve on a moving geometry.** One Adam step after it re-injects $\|v\|\eta$.
4. **Never guess block structure.** ~9 orders, no error signal.
5. **Never set the breakdown guard at $10^{-12}$ or at $0$.** Two orders, or NaN.
6. **Don't expect $O(\sqrt\kappa)$ iteration counts.** Measured $\alpha^{-0.59}$ to $\alpha^{-0.84}$; use $3/4$ and treat it as a bracket, not a law.
7. **Don't over-solve a level.** $\text{test}_2\le0.1\alpha$ doubles the work for essentially nothing.
8. **Heavy label noise plus drift degrades gracefully but measurably** — at $\sigma=10^{-2}$ the terminal solve overshoots the statistical floor by 4–20×.
9. **QI-scale research nets are the worst case for the accounting**, not the best. With $P\approx3d$ the GN step costs 1.7× an Adam step; on a real deep model it is ~0.1%. Do not read the research-scale cost as representative.

## Known gaps

- The damping floor $\alpha\ge r_{\text{entry}}$ is validated on two matrices at one drift schedule. The mechanism is straightforward but the evidence is thinner than for the stopping rule.
- The cap exponent is problem-dependent ($-0.59$ to $-0.84$) and comes from four matrices. $3/4$ is a conservative round number inside the interval, not a measured constant; do not treat it as universal.
- Nothing here has been run with Adam actually in the loop. Every drift result uses a synthetic schedule.
- **Membership change is argued, not measured.** The claim rests on the solver holding no state keyed to $L$, which is true by construction, but no experiment has varied $L$ between steps. The Adam moment handoff for coordinates crossing the boundary is entirely unspecified.
- The terminal accumulation phase (IV.8) has been measured for accuracy at every batch size, but never end-to-end after a real training run — in expD12/expD13 the full row pool was always resident.
- The interaction between a GN step on $L$ and Adam moving $P\setminus L$ in the same step is untested.
- Per-coordinate confidences $\mu_i$ from step 3 are untested; only scalar and Marquardt damping were measured.
