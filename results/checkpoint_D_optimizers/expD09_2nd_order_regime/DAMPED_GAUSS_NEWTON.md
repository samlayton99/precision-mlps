# Damped Gauss-Newton on the linear parameter block

**The mu math, written out as Sam specified. Moved here from `results/` because it was untracked. Directly relevant to the open mu-schedule question; read alongside `results/checkpoint_D_optimizers/expD12_mu_ladder/STEP2_SOLVER_SPEC.md` IV.5-IV.7.**

**Status:** math written down as specified by Sam; the three measured laws are data-obvious. The step-3 recommendation at the end is proposed, not approved.

**Notation warning.** This project already uses $\lambda$ for the QI bandwidth, $\lambda^\ast = \gamma h$. The damping parameter is written $\mu$ everywhere below. Sam's original note used $\lambda$ for the damping; it is the same object.

## The setup

MSE loss over $n$ samples, with $r(\theta) \in \mathbb{R}^n$ the residual vector:

$$L(\theta) = \tfrac12\|f(x,\theta) - y\|^2 = \tfrac12\|r(\theta)\|^2, \qquad r(\theta) = f(x,\theta) - y.$$

Differentiating, with $J = \partial f/\partial\theta \in \mathbb{R}^{n\times d}$ the Jacobian of the model output with respect to the parameters:

$$g \;=\; \nabla_\theta L \;=\; J^{\top} r.$$

Now split the parameters into a block $\theta_L$ that enters $f$ **linearly** and everything else. For that block $f = J_L\,\theta_L$ exactly, with $J_L$ independent of $\theta_L$. Writing $J$ for $J_L$, the Gauss-Newton step is the least-squares solution

$$\delta \;=\; \arg\min_{\delta}\;\|J\delta - r\|^2 \;=\; (J^{\top}J)^{-1} J^{\top} r, \qquad \theta_L \leftarrow \theta_L - \delta .$$

**Which parameters qualify.** For `QIMlp` the readout weights and bias are *exactly* linear in $f$, so $J_L$ is precisely the feature matrix $\Phi$ and the step is exact, not an approximation. Inner-layer parameters (centers, $\gamma$) are only *locally* linear, and there $J_L$ is the Jacobian block stored by the backward pass. The distinction matters because $\mu$ ends up doing two different jobs in the two cases.

## Why the undamped step fails

Two independent failures, and they need separating:

1. **$J$ is rank-deficient and ill-conditioned**, so $(J^{\top}J)^{-1}$ does not exist. For QI features $\kappa(\Phi)\approx10^{14}$ with numerical rank $r\approx0.57d$. Squaring to form $J^{\top}J$ makes this strictly worse — $\kappa(J^\top J)=\kappa(J)^2$ — which is why the working code never forms a Gram.
2. **Linearity is only local** for the non-readout blocks. The larger $\|\delta\|$, the worse the linear model predicts the true residual. This failure does *not* apply to the readout block at all.

## Damping

Add a Tikhonov term:

$$\delta_\mu \;=\; \arg\min_{\delta}\;\|J\delta - r\|^2 + \mu\|\delta\|^2 \;=\; (J^{\top}J + \mu I)^{-1} J^{\top} r .$$

This is Levenberg-Marquardt. It fixes both failures at once: the shifted matrix is invertible for any $\mu>0$, and $\|\delta_\mu\| \le \|g\|/\mu$ bounds the step so it stays inside the region where the linear model is trusted.

The two limits are the ones Sam identified:

$$\mu \to 0: \quad \delta_\mu \to J^{+}r \quad \text{(full least-squares step)}, \qquad\qquad \mu \to \infty: \quad \delta_\mu \to \frac{J^{\top}r}{\mu} = \frac{g}{\mu} \quad \text{(gradient descent, learning rate } 1/\mu).$$

So $\mu$ is a single dial interpolating between gradient descent and the exact solve, and it is a *confidence* dial: small $\mu$ when the linear model is trusted, large $\mu$ when it is not.

## What damping does to the spectrum

This is the part that connects to expD11. The damped problem is ordinary least squares on an augmented system:

$$\min_\delta \left\| \begin{pmatrix} J \\ \sqrt{\mu}\,I \end{pmatrix}\delta - \begin{pmatrix} r \\ 0\end{pmatrix} \right\|^2 ,$$

whose singular values are $\sqrt{\sigma_i^2 + \mu}$ where $\sigma_i$ are those of $J$. Therefore:

- every $\sigma_i \gg \sqrt{\mu}$ is left alone,
- every $\sigma_i \ll \sqrt{\mu}$ **collapses onto the single value $\sqrt{\mu}$**.

The damped operator has an effective rank $r_\mu = \#\{\sigma_i > \sqrt{\mu}\}$ above the damping, and one large cluster below it. Its condition number is

$$\kappa_\mu = \sqrt{\frac{\sigma_1^2+\mu}{\sigma_r^2+\mu}} \;\approx\; \frac{\sigma_1}{\sqrt{\mu}} \quad (\mu \gg \sigma_r^2).$$

expD11 established that the stored orthogonality a Krylov solver needs is set by the number of distinct spectral scales, not by $d$ or $\kappa$. Damping manufactures exactly that clustering. So the prediction is $c^\ast(\mu) \approx r_\mu$, and in particular $c^\ast = 0$ whenever $\mu$ is large enough to swallow the tail.

**In terms of the solution error**, damping shrinks each spectral component by $\sigma_i^2/(\sigma_i^2+\mu)$, so directions with $\sigma_i^2 \ll \mu$ are simply not solved. The damped optimum is a floor you cannot iterate past: to reach accuracy $\epsilon$ you must first choose $\mu$ small enough to admit it.

## Measured

All on QI-1D ($\Phi_{ik} = \tanh(\gamma(x_i-c_k))$ augmented with a bias column, $\lambda^\ast = 0.30$), $N \in \{256,512\}$, evaluation relative $L_2$ on a held-out 4001-point grid. One LSQR iteration is one $Jv$ plus one $J^{\top}u$, i.e. one forward-backward.

**1. Damping removes the memory requirement, until you ask for the floor.** $c^\ast = 0$ — plain LSQR, $O(d)$ state, no stored vectors — for every $\mu$ down to $r_\mu = 128$, where the damped optimum is $3.1\times10^{-6}$ ($N{=}256$) and $1.1\times10^{-4}$ ($N{=}512$). Only as $\mu \to 0$ does $c^\ast$ jump to $r$. The prediction $c^\ast\approx r_\mu$ holds at the ends but is loose in between: at $N{=}512$, $r_\mu=200$ still gives $c^\ast=0$.

**2. The price is that convergence is algebraic, not geometric.** With $c=0$ the accuracy after $T$ iterations follows $\epsilon \sim T^{-1.6}$, and the curves for *all* small $\mu$ coincide until the damping binds — meaning the iteration count is itself acting as the regularizer, and $\mu$ only matters once it is large enough to stop you sooner. Measured at $N{=}256$: $T{=}2 \to 9.5\times10^{-2}$, $T{=}8\to4.9\times10^{-3}$, $T{=}32\to4.9\times10^{-4}$, $T{=}128\to5.7\times10^{-5}$. Extrapolating that rate to $10^{-14}$ needs $T \sim 10^{8}$ iterations. With $c=r$ stored vectors the same target is reached in $2r \approx 550$.

**3. Damped steps are robust to batching at $O(d)$ state.** Repeated damped steps on fresh random batches, $T{=}8$ iterations per step, 400 steps: $b{=}d$ gives $1.8\times10^{-4}$ and $b{=}d/64$ (7 rows) gives $9.0\times10^{-4}$ — 5x degradation across a 64x batch reduction. This is the requirement-5 behaviour that the undamped streaming solver needed the ad-hoc $\tau \le b/2$ clamp to fake; damping supplies it from the objective instead. Only $\mu = 10^{2}$ (over-damped, essentially gradient descent) is materially worse.

**Code & data.** `experiments/expD11_batching/damping.py` → `results/checkpoint_D_optimizers/expD11_batching/damping.jsonl`, figure `figures/E_damping.png`. Solver with the window dial: `experiments/expD11_batching/window_law.py`. Underlying spectral law: `results/checkpoint_D_optimizers/expD11_batching/expD11_results.md`.

### Figure

- **`E_damping.png`.** Three panels. *E1*: $c^\ast$ (symlog $y$, so $0$ is on the axis; dotted line at $r$) against the accuracy of the damped optimum ($x$ reversed, so demanding more accuracy moves right); one line per width. Look for the flat run along $c^\ast=0$ that breaks upward only past $10^{-6}$. *E2*: eval rel $L_2$ against iteration budget $T$, log-log, one line per $r_\mu$, with a dashed $T^{-1.6}$ reference. Look for all small-$\mu$ curves lying on top of each other and parallel to the dashed line — algebraic, not geometric. *E3*: best eval over 400 batched steps against batch fraction $b/d$ ($x$ reversed), one line per $\mu$. Look for the flat bundle of small-$\mu$ lines and the separated over-damped $\mu{=}10^2$ line above them.

## What this does and does not buy

**Does.** A damped step is fully compliant with the step-2 requirement list at $O(d)$ state — three or four vectors, Adam's class or better. It is iterative, needs no setup, costs a handful of forward-backwards, front-loads hard (four iterations buy two orders), tolerates $b = d/64$, and degrades gracefully rather than catastrophically because $\mu$ bounds the step. It also gives a principled replacement for the $\tau$-clamp. As a *general-purpose* optimizer that is monotonically better than Adam, this is the strongest candidate the campaign has produced.

**Does not.** It does not reach machine precision, and it does not move the expD11 wall. Damping buys cheapness by removing the small singular directions from the problem — but those directions are exactly what the last nine orders of accuracy live in. The final $\mu \to 0$ solve is the same undamped ill-conditioned problem as before, needing $c \approx r$ stored vectors, i.e. $\Theta(d^2)$ state. Sam's own framing — "as the $\mu$ gets small enough, we do a final solve with full preconditioning" — is the right shape, and that final solve is precisely where the wall sits, unchanged.

So damping does make the math easier, and it makes the *bulk* of the optimization feasible at gradient-step cost. It relocates the hard part rather than removing it: instead of one impossible solve, there is a long cheap phase plus one expensive terminal solve.

## Does the terminal solve shrink? No.

Two ways of exploiting the damped phase were tested; both fail, and the second fails in the direction that matters.

**L1 — a warm start does not shrink the terminal solve's memory.** Starting the undamped solve from the damped solution $w_\mu$, so that the top $r_\mu$ directions are already resolved, leaves $c^\ast$ **completely unchanged**: $c^\ast = 256$ at $N{=}256$ and $512$ at $N{=}512$, at every $r_\mu$ tested. Even with 244 of 273 directions pre-resolved to $1.3\times10^{-9}$, the terminal solve still needs $c^\ast = 256$, against a naive prediction of $r - r_\mu = 29$.

The reason is structural, and it is worth stating because it kills a whole family of ideas: **a warm start changes the residual, not the operator.** The Krylov filter polynomial $q$ must satisfy $q(0)=1$ and be small across the *whole* spectrum of the operator it is handed, regardless of where the current residual happens to have energy. Pre-resolving directions does not remove them from the operator, so it does not shorten the polynomial.

**L2 — the $\mu$-ladder makes things monotonically worse.** Peeling the spectrum in $K$ descending damping stages, discarding stored vectors between stages so peak state stays $\Theta(d\cdot c)$ with $c$ fixed: accuracy degrades monotonically in $K$ at every $c$. At $N{=}256$, $c{=}64$: $K{=}1 \to 4.0\times10^{-7}$, $K{=}16 \to 2.6\times10^{-5}$. It never approaches the floor. This is the restart penalty — the hope was that each stage's small $\kappa_\mu$ would make restarting cheap, and it does not compensate.

So the damped phase does not make the terminal solve smaller. There is one cheap phase and one expensive terminal solve, and the terminal solve is exactly the $\Theta(d\cdot r)$ problem expD11 characterized.

## Per-coordinate damping

Replace $\mu I$ with $D = \mathrm{diag}(1/\mu_i)$, where $\mu_i$ is a per-coordinate confidence that coordinate $i$ is linear (high confidence, low penalty):

$$\delta \;=\; (J^{\top}J + D)^{-1}J^{\top}r, \qquad D = \mathrm{diag}(d_i), \quad d_i = 1/\mu_i .$$

**Why it is harder, in one line.** Scalar $\mu I$ commutes with $J^{\top}J$, so the damped singular values are exactly $\sqrt{\sigma_i^2+\mu}$ and the whole tail collapses onto the single value $\sqrt{\mu}$ — one cluster, which is precisely what removes the stored-state requirement. A general diagonal does **not** commute, so the tail no longer collapses to a point; it smears across a band.

**What it actually costs, measured.** Holding the geometric-mean damping fixed and varying only the spread of $\{d_i\}$ over $10^0$ to $10^8$:

| spread $\max_i\mu_i/\min_i\mu_i$ | $\kappa$ of damped operator | best accuracy | $c^\ast$ |
|---|---|---|---|
| $1$ (scalar $\mu I$) | $4.4\times10^{4}$ | $3.1\times10^{-6}$ | 0 |
| $10^{2}$ | $1.4\times10^{5}$ | $2.0\times10^{-5}$ | 0 |
| $10^{4}$ | $4.4\times10^{5}$ | $4.0\times10^{-5}$ | 0 |
| $10^{8}$ | $4.2\times10^{6}$ | $1.4\times10^{-4}$ | 0 |
| Marquardt $\mu\,\mathrm{diag}(J^{\top}J)$ | $4.4\times10^{4}$ | $3.1\times10^{-6}$ | 0 |

The measured $\kappa$ matches $\sigma_1/\sqrt{\min_i d_i}$ to plotting accuracy. That is the governing law and it is sharper than "spread is bad":

$$\kappa_D \;\approx\; \frac{\sigma_1}{\sqrt{\min_i d_i}} \;=\; \sigma_1\sqrt{\max_i \mu_i}.$$

**The condition number is set by the single most-trusted coordinate, not by the average or the spread as such.** One over-confident coordinate degrades the solve for all $d$ of them. The practical consequence is a clamp: cap $\max_i \mu_i$, and the rest of the diagonal can be as varied as the confidence estimates like.

Note also what per-coordinate damping does *not* cost: $c^\ast$ stays 0 in every variant. The damage is to conditioning, and therefore to accuracy at a fixed iteration budget — not to memory.

**Marquardt's choice $D = \mu\,\mathrm{diag}(J^{\top}J)$ is free.** It matches scalar damping on both $\kappa$ and accuracy, and it is scale-invariant: substituting $\delta = \mathrm{diag}(J^{\top}J)^{-1/2}z$ turns it into scalar damping on a column-equilibrated Jacobian. It is also the 1963 original and what every production LM implements, so it wins the practicality-gate toss-up over any novel confidence heuristic. Recommended default; treat a learned $\mu_i$ as a deviation from it that has to earn its keep.

## Does the solve have to be iterative?

No, but relaxing that does not relax the memory constraint. A direct solve of $(J^{\top}J + D)\delta = J^{\top}r$ needs either the $d\times d$ Gram or a factor of the augmented system; both are $\Theta(d^2)$ storage, which is the disqualified class, and forming the Gram also squares the condition number. Direct methods only fit if the operator has exploitable structure — banded, Toeplitz, or Kronecker. The Kronecker route ($J^{\top}J \approx A\otimes B$, i.e. K-FAC) genuinely fits the memory budget and is battle-tested, but it is an *approximation* of the operator, which puts it in the preconditioner class that expD11 measured to cap around $10^{-6}$. It would be a fine way to make the damped phase cheaper; it is not a route to the floor.

## Open questions

- Does the standard LM trust-region update (grow $\mu$ on a rejected step, shrink on an accepted one) remain reliable at $\kappa\sim10^{14}$, or does the gain-ratio test break down? The ladder result (L2) says a *pre-scheduled* descent in $\mu$ is harmful, which makes the adaptive rule the only remaining schedule worth testing.
- $\mu$ has units of $\sigma^2$. A scale-free $\mu = \alpha\sigma_1^2$ with $\sigma_1$ from cheap power iteration would make the dial transferable across widths and dtypes; untested.
- Where should $\max_i\mu_i$ be clamped in practice, and can the confidence estimates be derived from something already computed in the backward pass rather than a new statistic?
- All panels are 1-D. The 2-D check has not been run.
