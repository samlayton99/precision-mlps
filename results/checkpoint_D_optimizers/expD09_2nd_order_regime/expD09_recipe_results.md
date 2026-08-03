# Block-QR whitening + LSMR

> **This file is the canonical recipe.** There is no longer a copy under `experiments/`.
> It lives here because `results/**/*_results.md` is the only thing git tracks in this tree.

**Written for a reader with no prior context and no assumed background in numerical linear algebra.** Sections 1 to 6 build the method from scratch, defining every object as it appears. Sections 7 onward are the measured results and are the reference material. If you already know Krylov solvers and QR factorization, skip to Section 6.

Companions: `results/checkpoint_D_optimizers/expD09_2nd_order_regime/SUBPROBLEM.md` (the problem statement and the full experimental history, including everything that failed), `experiments/expD09_2nd_order_regime/validation/README.md` (the validation sweep), and the figures and run data alongside this file.

**Which solver ran, and where.** Two different Krylov solvers produced the numbers in this document, and the distinction was previously undocumented:

| numbers | code path | solver |
|---|---|---|
| the headline nine cells (Section 8.0), integration measurements (Section 9) | `solvers.py::_blockqr` | **LSMR** (SciPy's `lsmr`, `solvers.py:342`) |
| the validation sweep (Sections 8.5 to 8.9, figures `A_`, `C_`, `D_`, `H_`, `I_`) | `validation/common_val.py::lsqr_traj` | **LSQR** (hand-written Paige-Saunders) |

The method works with either, since neither is modified and both consume the same whitened operator $B$. Section 4.2 defines both and gives their complete update rules. Two stale label strings remain: the docstring at `solvers.py:313` and the registry labels at `solvers.py:115-118` still say "LSQR" for the `blockqr_*` ids, which is wrong and affects figure legends only.

---

# 1. Notation

Every symbol used in this document, in one place. Nothing below is used before it appears here.

| symbol | type | meaning |
|---|---|---|
| $N$ | integer | network width parameter; the number of grid intervals on $[-1,1]$ |
| $h$ | real | grid spacing, $h = 2/N$ |
| $\gamma$ | real | inner-layer weight (the bandwidth), $\gamma = \lambda^\star/h$ |
| $\lambda^\star$ | real | dimensionless bandwidth $\gamma h$, fixed at $0.25$ here |
| $c_k$ | real | center of hidden unit $k$, i.e. its position on $[-1,1]$ |
| $x_i$ | real | the $i$-th training input |
| $y \in \mathbb{R}^n$ | vector | the target values, $y_i = f(x_i)$ |
| $n$ | integer | number of training rows; here $n = 4d$ |
| $d$ | integer | number of columns of the feature matrix, one per hidden unit plus one bias |
| $\Phi \in \mathbb{R}^{n \times d}$ | matrix | the feature matrix (defined in Section 2) |
| $w \in \mathbb{R}^d$ | vector | the readout weights, the unknown being solved for |
| $r \in \mathbb{R}^n$ | vector | the residual, $r = \Phi w - y$ |
| $\sigma_1 \ge \sigma_2 \ge \dots$ | reals | the singular values of $\Phi$ |
| $\kappa$ | real | condition number, $\kappa(\Phi) = \sigma_1/\sigma_r$ (Section 4.1) |
| $r$ (context) | integer | numerical rank of $\Phi$; disambiguated by context from the residual |
| $\varepsilon$ | real | fp64 unit roundoff, $\approx 2.2 \times 10^{-16}$ |
| $C$ | index set | a *block*: a set of $k$ column indices treated together |
| $k$ | integer | block size, the method's one tuning knob |
| $Q_C, R_C$ | matrices | the QR factors of block $C$ (Section 4.3) |
| $B \in \mathbb{R}^{n \times d}$ | matrix | the *whitened* feature matrix (Section 5) |
| $z \in \mathbb{R}^d$ | vector | the solution in whitened coordinates, $\Phi w = Bz$ |
| $b$ | integer | batch size, when rows are subsampled (Section 8.4) |

Two derived quantities are used to report accuracy, both evaluated on a held-out grid of 4001 points that shares no point with the training grid:

$$\text{rel } L_2 \;=\; \frac{\|\hat f - f\|_2}{\|f\|_2}, \qquad L_\infty \;=\; \max_x |\hat f(x) - f(x)|.$$

A **pass** means one traversal of the data, costing $O(nd)$ floating-point operations. It is the unit of cost throughout, chosen so the method can be compared against a training step: one forward-plus-backward pass through a network of this size is two passes in this accounting.

---

# 2. The problem

A one-hidden-layer $\tanh$ network trained end to end on a smooth 1-D target stalls somewhere between $10^{-3}$ and $10^{-10}$ relative error. The same network, with weights written down by hand from quasi-interpolant theory, reaches $10^{-15}$. That gap is what this repository exists to close.

Part of the gap is a question about optimization, and part is a question about pure linear algebra. This document answers the linear-algebra part, in isolation, on a frozen network.

## The object

Fix the hidden layer. Every hidden unit $k$ computes $\tanh(\gamma(x - c_k))$, so evaluating all of them at all $n$ training inputs produces a matrix:

$$\Phi \in \mathbb{R}^{n \times d}, \qquad \Phi_{ik} = \tanh\big(\gamma\,(x_i - c_k)\big),$$

with one final column of all ones for the bias. Column $k$ of $\Phi$ is hidden unit $k$; row $i$ is training point $i$. The centers $c_k = -1 + kh$ are uniformly spaced, and they run past the ends of $[-1,1]$ on both sides. Those extra units outside the data range are the **halo**, and the quasi-interpolant theory requires them: they absorb boundary effects that would otherwise pollute the interior.

With the hidden layer frozen, the network's output is linear in the readout weights $w$, so training the readout is a least-squares problem:

$$\boxed{\;\min_{w \in \mathbb{R}^d} \;\|\Phi w - y\|_2\;}$$

That is the entire problem. No nonlinearity, no gradient descent on a nonconvex landscape, no stochasticity. Just: solve a linear least-squares system to fifteen digits.

## Why it is hard

If this were an ordinary least-squares problem, one call to `numpy.linalg.lstsq` would end the matter. Three properties of this particular $\Phi$ make it hard, and every design decision in Section 6 traces back to one of them.

**Property 1: it is extremely ill-conditioned.** $\kappa(\Phi) \approx 7 \times 10^{14}$, which is within a factor of a few hundred of $1/\varepsilon$. Section 4.1 defines $\kappa$ and explains what that number costs. In plain terms: the matrix is at the edge of what fp64 can represent as invertible at all.

**Property 2: it is numerically rank deficient.** The numerical rank is roughly $0.6d$. The reason is the halo. A center $c_k$ sitting well outside $[-1,1]$ makes $\gamma(x_i - c_k)$ large in magnitude for every training point, so $\tanh$ of it saturates to $\pm 1$. That column is numerically a constant, and a collection of constant columns is numerically rank one. Those columns carry no independent information and must not be inverted.

**Property 3: the target's energy is spread across all directions, so truncation is not available.** This is the property that rules out the standard remedy. Write the target in the basis of $\Phi$'s right singular vectors. The coefficient along direction $i$ scales like $\sigma_i$: the small directions carry proportionally small but genuinely needed content. Reaching rel $L_2 \le 10^{-13}$ requires resolving directions down to $\sigma_i/\sigma_1 \sim 10^{-11}$.

The usual fix for an ill-conditioned system is to throw away the small singular directions (truncation, or equivalently ridge regularization). Property 3 says those directions are load-bearing. Throwing them away caps the accuracy at exactly the level you truncated.

## What had been achieved before this recipe

Best result at $O(d)$ memory: about $10^{-9}$. Plain CGLS, the textbook iterative least-squares method, reached $2.4 \times 10^{-6}$. The reference floor, a truncated SVD costing $O(d^2)$ memory, sits at $8.7 \times 10^{-14}$.

---

# 3. The running example

Every concrete number below refers to one configuration, so the trace stays followable.

Take $N = 128$. Then $h = 2/128 = 0.015625$ and $\gamma = 0.25/h = 16$. The default halo puts 70 extra centers on each side, giving 269 centers plus one bias column, so $d = 270$. Rows are $n = 4d = 1080$ equispaced points on $[-1,1]$.

Measured on that matrix:

- $\kappa(\Phi) = 6.8 \times 10^{14}$
- numerical rank $= 144$ out of 270 columns
- the leftmost 70 and rightmost 70 columns are halo, so live centers occupy columns 70 through 198

Adjacent columns are nearly identical, which is the geometric face of the ill-conditioning. Normalizing each column to unit length and taking inner products from a column in the middle:

| center offset | 1 | 2 | 4 | 8 | 16 | 64 |
|---|---|---|---|---|---|---|
| correlation | 0.999 | 0.995 | 0.979 | 0.928 | 0.80 | 0.03 |

Neighbouring hidden units are near-duplicates. Units 64 apart are nearly independent. Hold onto that table: it is the reason the method works, and Section 5.4 uses it directly.

---

# 4. Background

Four tools. Each gets a definition, an intuition, and the one property that matters here.

## 4.1 The condition number, and why it governs everything

For a matrix $A$ with singular values $\sigma_1 \ge \dots \ge \sigma_r > 0$, the condition number is

$$\kappa(A) \;=\; \frac{\sigma_1}{\sigma_r}.$$

It measures how much $A$ stretches the most-stretched direction relative to the least-stretched one. Its practical meaning: if you perturb the data by a relative amount $\delta$, the solution can move by a relative amount $\kappa \delta$. Since floating-point arithmetic perturbs everything by at least $\varepsilon$, the *parameter* error in $w$ is at best around $\varepsilon \kappa$.

At $\kappa = 7 \times 10^{14}$ that product is $\varepsilon\kappa \approx 0.15$. The computed $w$ has no correct digits at all.

**This is not fatal, and understanding why is the key to the whole method.** We do not care about $w$. We care about the function the network computes, which is $\Phi w$. The directions in which $w$ is wrong are precisely the directions $\Phi$ squashes by $\sigma_r$, so multiplying by $\Phi$ undoes the amplification. A wildly wrong $w$ can produce a function accurate to $10^{-14}$.

The trap is what happens when the ill-conditioning is applied *repeatedly*. Section 7 makes this precise.

**For iterative solvers, $\kappa$ has a second, separate meaning: it sets the iteration count.** The Krylov methods of Section 4.2 need roughly $O(\kappa(A))$ iterations to converge on a least-squares problem. At $\kappa = 10^{14}$ that is not a real algorithm.

So $\kappa$ does two jobs, and it is worth keeping them apart:

| what $\kappa$ controls | consequence here |
|---|---|
| error amplification through a single solve | harmless, because $\Phi$ cancels it on the way out |
| iteration count of a Krylov method | fatal at $10^{14}$; this is what must be fixed |

## 4.2 Krylov solvers: LSQR and LSMR

An **iterative solver** for $\min_w \|Aw - y\|$ builds a sequence $w_0, w_1, w_2, \dots$, using the matrix only through two operations: multiply a vector by $A$, and multiply a vector by $A^\top$. It never stores or factorizes $A$. That is why it can run at $O(d)$ memory.

**Krylov methods** are the good iterative solvers. The defining property, stated before any machinery:

> At step $k$, the iterate $w_k$ is the **exact minimizer** of the objective over a $k$-dimensional subspace $\mathcal{K}_k$, and each iteration enlarges that subspace by one dimension.

The iterate does not step toward the answer. It is re-solved exactly, on more room each time. Because $\mathcal{K}_k \subset \mathcal{K}_{k+1}$, the error is monotone for free. The subspace is

$$\mathcal{K}_k \;=\; \mathrm{span}\big\{A^\top y,\;(A^\top A)A^\top y,\;\dots,\;(A^\top A)^{k-1}A^\top y\big\},$$

which is the only subspace reachable using nothing but products with $A$ and $A^\top$. The two methods used here, LSQR and LSMR, share a common engine for building an orthonormal basis of it.

### The engine: Golub-Kahan bidiagonalization

Both methods build orthonormal bases for the row and column spaces, one vector at a time. Start with

$$\beta_1 = \|y\|_2,\quad u_1 = y/\beta_1, \qquad \alpha_1 = \|A^\top u_1\|_2,\quad v_1 = A^\top u_1/\alpha_1,$$

then iterate, for $i = 1, 2, \dots$:

$$\beta_{i+1}\,u_{i+1} \;=\; A v_i - \alpha_i u_i, \qquad \alpha_{i+1}\,v_{i+1} \;=\; A^\top u_{i+1} - \beta_{i+1} v_i,$$

where $\alpha_{i+1}, \beta_{i+1} \ge 0$ are whatever normalizing constants make $\|u_{i+1}\|_2 = \|v_{i+1}\|_2 = 1$. **Each iteration costs exactly one multiply by $A$ and one by $A^\top$, which is one forward-backward pass.** That is the entire per-iteration cost of both methods.

Collect the vectors into $U_{k+1} = [u_1 \dots u_{k+1}]$ and $V_k = [v_1 \dots v_k]$, and the coefficients into the lower-bidiagonal matrix

$$\mathcal{B}_k \;=\; \begin{pmatrix} \alpha_1 & & & \\ \beta_2 & \alpha_2 & & \\ & \beta_3 & \ddots & \\ & & \ddots & \alpha_k \\ & & & \beta_{k+1} \end{pmatrix} \in \mathbb{R}^{(k+1)\times k}.$$

The construction guarantees the identity $A V_k = U_{k+1} \mathcal{B}_k$. Searching for $w_k = V_k s$ with $s \in \mathbb{R}^k$, the residual collapses to something tiny and bidiagonal:

$$y - A w_k \;=\; U_{k+1}\big(\beta_1 e_1 - \mathcal{B}_k s\big).$$

The $n$-dimensional problem has become a $(k{+}1) \times k$ bidiagonal problem, solvable in $O(k)$ work. **The two methods differ only in which quantity they minimize when choosing $s$.**

### LSQR versus LSMR

$$\textbf{LSQR:}\quad s_k = \arg\min_s \|\beta_1 e_1 - \mathcal{B}_k s\|_2 \quad\Longleftrightarrow\quad \text{minimize } \|y - Aw_k\|_2$$

$$\textbf{LSMR:}\quad s_k \text{ chosen to minimize } \|A^\top(y - Aw_k)\|_2$$

LSQR minimizes the residual. LSMR minimizes the *normal-equation* residual, the gradient of the least-squares objective. LSQR is mathematically equivalent to conjugate gradients on the normal equations; LSMR is equivalent to MINRES on them. Neither ever forms $A^\top A$, which is the point: forming it would square the condition number.

Two practical differences:

1. In LSMR, $\|A^\top r_k\|$ decreases monotonically. In LSQR it can oscillate. Since $\|A^\top r_k\|$ is the natural stopping signal, LSMR is easier to stop.
2. On this problem LSMR measured better. Section 12 records the earlier comparison: swapping CGLS for LSMR under an identical preconditioner moved the geometric-mean error from $2.0 \times 10^{-8}$ to $1.1 \times 10^{-9}$, for one changed line.

### How the iterate actually gets closer

The recurrence above builds a basis. It does not obviously move $w$. Two things close that gap.

**What one more dimension buys.** Every $w \in \mathcal{K}_k$ can be written $w = p(A^\top A)A^\top y$ for a polynomial $p$ of degree at most $k-1$. Substitute the SVD $A = U\Sigma V^\top$ and write $g_i = u_i^\top y$ for the target's energy along singular direction $i$. Then the residual and the parameter error are both the *same* polynomial filter applied per direction:

$$\text{residual along } u_i \;=\; -\,q(\sigma_i^2)\,g_i, \qquad w_\star - w_k \;=\; \sum_i q(\sigma_i^2)\,\frac{g_i}{\sigma_i}\,v_i, \qquad q(t) := 1 - t\,p(t).$$

$q$ has degree at most $k$ and is pinned at $q(0) = 1$. Everything else about it is free. **So each iteration buys exactly one more root to place, and the method spends it wherever it kills the most error.** The pin at $q(0)=1$ is why null-space directions are never recovered by any number of iterations.

This also proves the clustering claim used in Section 7. If the $\sigma_i^2$ take only $m$ distinct values $t_1,\dots,t_m$, then $q(t) = \prod_{j=1}^m (1 - t/t_j)$ has degree $m$, satisfies $q(0)=1$, and vanishes at all of them. **The method terminates at iteration $m$ regardless of how many directions exist**, because one root annihilates an entire cluster.

**Why storing $\mathcal{K}_k$ is not required.** Writing $w_k = V_k s_k$ suggests keeping all $k$ basis vectors. Paige & Saunders (1982) avoid that by maintaining the QR factorization of $\mathcal{B}_k$ incrementally with Givens rotations, which collapses the update to a short recurrence

$$w_k = w_{k-1} + \tau_k\,d_k, \qquad d_k = v_k - \omega_k\,d_{k-1},$$

with scalars $\tau_k, \omega_k$ falling out of the rotations. Only $w$, $d$, and the current $u, v$ are resident, so memory is $O(d)$. The update resembles gradient descent with an unusual direction, and that resemblance is misleading: $d_k$ is not chosen by a descent heuristic, it is whatever makes the running total land on the exact subspace minimizer.

### The complete update rules

Everything above is the derivation. This subsection is the reference: both algorithms in full, as you would implement them. Solving $\min_x \|Ax - b\|_2$ from $x_0 = 0$. Every line is one assignment, executed in the order written. $\mathrm{hyp}(a,c) := \sqrt{a^2+c^2}$.

**Shared initialization** (both algorithms):

$$\beta_1 = \|b\|_2,\quad u_1 = b/\beta_1, \qquad \alpha_1 = \|A^\top u_1\|_2,\quad v_1 = A^\top u_1/\alpha_1.$$

**Shared bidiagonalization step** (both algorithms, at the top of iteration $j$). This is the only place $A$ is touched, and it costs one matvec each way:

$$\begin{aligned}
\beta_{j+1} &= \|A v_j - \alpha_j u_j\|_2, &\qquad u_{j+1} &= (A v_j - \alpha_j u_j)/\beta_{j+1},\\
\alpha_{j+1} &= \|A^\top u_{j+1} - \beta_{j+1} v_j\|_2, &\qquad v_{j+1} &= (A^\top u_{j+1} - \beta_{j+1} v_j)/\alpha_{j+1}.
\end{aligned}$$

---

**LSQR** (Paige & Saunders 1982). Minimizes $\|b - Ax_j\|_2$. Extra state: $w \in \mathbb{R}^d$ and three scalars.

Initialize $\;w_1 = v_1,\; x_0 = 0,\; \bar\phi_1 = \beta_1,\; \bar\rho_1 = \alpha_1$. Then for $j = 1, 2, \dots$: run the bidiagonalization step, then

$$\begin{aligned}
\rho_j &= \mathrm{hyp}(\bar\rho_j,\ \beta_{j+1}) & c_j &= \bar\rho_j/\rho_j & s_j &= \beta_{j+1}/\rho_j &&\text{(Givens rotation)}\\
\theta_{j+1} &= s_j\,\alpha_{j+1} & \bar\rho_{j+1} &= -c_j\,\alpha_{j+1} &&&&\text{(propagate }R\text{)}\\
\phi_j &= c_j\,\bar\phi_j & \bar\phi_{j+1} &= s_j\,\bar\phi_j &&&&\text{(propagate rhs)}\\[4pt]
\end{aligned}$$

$$\boxed{\;x_j = x_{j-1} + \frac{\phi_j}{\rho_j}\,w_j\;}\qquad w_{j+1} = v_{j+1} - \frac{\theta_{j+1}}{\rho_j}\,w_j$$

Free monitoring quantities, needing no extra work: $\;\|b - Ax_j\|_2 = \bar\phi_{j+1}\;$ and $\;\|A^\top(b - Ax_j)\|_2 = \bar\phi_{j+1}\,\alpha_{j+1}\,|c_j|$.

---

**LSMR** (Fong & Saunders 2011). Minimizes $\|A^\top(b - Ax_j)\|_2$. Needs a second rotation because it applies QR twice, once to the bidiagonal matrix and once to the result. Extra state: $h, \bar h \in \mathbb{R}^d$ and seven scalars.

Initialize $\;h_1 = v_1,\; \bar h_0 = 0,\; x_0 = 0,\; \bar\alpha_1 = \alpha_1,\; \bar\zeta_1 = \alpha_1\beta_1,\; \rho_0 = \bar\rho_0 = \bar c_0 = 1,\; \bar s_0 = 0$. Then for $j = 1, 2, \dots$: run the bidiagonalization step, then

$$\begin{aligned}
\rho_j &= \mathrm{hyp}(\bar\alpha_j,\ \beta_{j+1}) & c_j &= \bar\alpha_j/\rho_j & s_j &= \beta_{j+1}/\rho_j &&\text{(rotation 1)}\\
\theta_{j+1} &= s_j\,\alpha_{j+1} & \bar\alpha_{j+1} &= c_j\,\alpha_{j+1}\\[4pt]
\bar\theta_j &= \bar s_{j-1}\,\rho_j & \bar\rho_j &= \mathrm{hyp}(\bar c_{j-1}\rho_j,\ \theta_{j+1}) &&&&\text{(rotation 2)}\\
\bar c_j &= \bar c_{j-1}\rho_j/\bar\rho_j & \bar s_j &= \theta_{j+1}/\bar\rho_j\\[4pt]
\zeta_j &= \bar c_j\,\bar\zeta_j & \bar\zeta_{j+1} &= -\bar s_j\,\bar\zeta_j &&&&\text{(propagate rhs)}
\end{aligned}$$

$$\bar h_j = h_j - \frac{\bar\theta_j\,\rho_j}{\rho_{j-1}\,\bar\rho_{j-1}}\,\bar h_{j-1} \qquad\boxed{\;x_j = x_{j-1} + \frac{\zeta_j}{\rho_j\,\bar\rho_j}\,\bar h_j\;}\qquad h_{j+1} = v_{j+1} - \frac{\theta_{j+1}}{\rho_j}\,h_j$$

Free monitoring quantity: $\;\|A^\top(b - Ax_j)\|_2 = |\bar\zeta_{j+1}|$, and this one is monotone decreasing, which is why LSMR is the easier of the two to stop.

---

**Side by side.**

| | LSQR | LSMR |
|---|---|---|
| minimizes | $\|b - Ax\|$ | $\|A^\top(b-Ax)\|$ |
| equivalent to | CG on the normal equations | MINRES on the normal equations |
| rotations per step | 1 | 2 |
| vectors held besides $x, u, v$ | $w$ | $h$, $\bar h$ |
| monotone quantity | $\|b-Ax_j\|$ | both $\|b-Ax_j\|$ and $\|A^\top(b-Ax_j)\|$ |
| used in this repo | `validation/common_val.py::lsqr_traj` | `solvers.py::_blockqr` |

Both were implemented from the above and checked against SciPy on a well-conditioned test matrix: iterates agree to $1.4\times10^{-15}$ relative at every iteration count, and both converge to the `lstsq` solution. On the whitened $B$ of Section 3 they agree with SciPy to machine precision for the first ten iterations and then drift, which is fp64 loss of orthogonality in the bidiagonalization, not a difference in the algorithm.

**Both are used as black boxes in the shipped code.** The method in this document does not modify the solver; it changes the matrix handed to it. SciPy's `lsmr` is called with `atol=0, btol=0, conlim=0` so that no internal stopping heuristic fires, and `maxiter` set explicitly.

## 4.3 QR factorization, and why Householder is special

Any $A \in \mathbb{R}^{n \times m}$ with $n \ge m$ factors as

$$A = QR, \qquad Q \in \mathbb{R}^{n \times m} \text{ with } Q^\top Q = I_m, \qquad R \in \mathbb{R}^{m \times m} \text{ upper triangular}.$$

$Q$'s columns are an orthonormal basis for $A$'s column space. $R$ records how to rebuild $A$'s columns from that basis.

The algorithm that matters is **Householder QR**. It works by applying a sequence of reflections

$$H = I - \frac{2 v v^\top}{v^\top v},$$

each chosen to zero out everything below the diagonal in one column. The property that earns Householder its place in this method: **every $H$ is exactly orthogonal, so $\|Hx\|_2 = \|x\|_2$ for every $x$.** A reflection never stretches anything. Rounding errors are therefore never amplified, no matter how ill-conditioned $A$ is.

The resulting $Q$ satisfies $\|Q^\top Q - I\| \approx \varepsilon$ **by construction, independent of $\kappa(A)$**. That guarantee is why Section 8.1 says "whiten by QR, never via the Gram", and it is worth $7$ orders of accuracy.

Tempting to think you could get the same $Q$ more cheaply by computing $M = A^\top A$, taking its eigendecomposition, and forming $A M^{-1/2}$. Mathematically identical, numerically catastrophic. Forming $A^\top A$ squares the condition number, so anything below $\sqrt{\varepsilon} \approx 10^{-8}$ of the top singular value becomes noise, and $M^{-1/2}$ then has condition number $10^{14}$ in its own right.

## 4.4 Column pivoting and numerical rank

Plain QR processes columns left to right. **Column-pivoted QR** instead picks, at each step, the remaining column with the largest norm orthogonal to what has already been chosen. It produces

$$A\,\Pi = QR$$

with $\Pi$ a permutation matrix, and the diagonal of $R$ ordered so that $|R_{11}| \gtrsim |R_{22}| \gtrsim \dots$.

That ordering is the payoff. **The diagonal of $R$ reveals the numerical rank.** Columns that are linear combinations of earlier ones produce tiny $|R_{jj}|$. Counting

$$n_C = \#\{\,j : |R_{jj}| > \texttt{rcond} \cdot \max_j |R_{jj}|\,\}, \qquad \texttt{rcond} = 10^{-13},$$

gives the number of genuinely independent columns. The rest are discarded rather than inverted, which is exactly what Property 2 of Section 2 demands. SciPy returns $\Pi$ as an index array `piv` with the meaning `A[:, piv] = Q @ R`.

---

# 5. The idea

## 5.1 Preconditioning, stated generally

Section 4.1 established the bind: the solver needs $O(\kappa)$ iterations and $\kappa = 10^{14}$.

**Preconditioning** is the standard escape. Pick an invertible $M \in \mathbb{R}^{d \times d}$ and change variables, $w = Mz$. Then

$$\|\Phi w - y\|_2 = \|\underbrace{\Phi M}_{=:\,B}\,z - y\|_2,$$

so solving $\min_z \|Bz - y\|$ and setting $w = Mz$ recovers the same $w$. The two problems have identical solutions, but the solver sees $B$, and iterates according to $\kappa(B)$, not $\kappa(\Phi)$.

A good $M$ makes $\kappa(B)$ small while being cheap to store and apply. The ideal $M$ is $\Phi^{-1}$, which would give $\kappa = 1$ and requires already having solved the problem.

## 5.2 What "whitening" means

A matrix is **white** when its columns are orthonormal, so $B^\top B = I$ and $\kappa(B) = 1$. Whitening a matrix means finding the $M$ that achieves this.

For a full matrix, $M$ is exactly the inverse $R$ factor from its QR factorization: if $\Phi = QR$ then $\Phi R^{-1} = Q$. But $R$ is $d \times d$, which costs $O(d^2)$ storage. That is the memory budget this method exists to avoid.

## 5.3 Block whitening: the compromise

Whiten *within* groups of columns rather than across all of them.

Partition the $d$ columns into blocks $C_1, C_2, \dots$ of size $k$. Factor each block on its own:

$$\Phi_{:,C}\,\Pi_C \;=\; Q_C R_C,$$

and define $B$ by pasting the $Q$ factors in:

$$B_{:,C} \;=\; Q_C.$$

**$B$ is the whitened feature matrix: a matrix of the same shape as $\Phi$, whose columns within each block are exactly orthonormal.** Cross-block correlations survive untouched, so $\kappa(B) > 1$ in general, but everything internal to a block is gone.

This is the change of variables $w = Mz$ with $M$ block-diagonal, blocks $\Pi_C R_C^{-1}$. Recovering $w$ from $z$ is one triangular solve per block:

$$R_C\,c \;=\; z_C, \qquad w_{C}\big[\Pi_C\big] \;=\; c.$$

**The stored state is the set of $R_C$ factors: $k \times k$ per block, over $d/k$ blocks, so $d \cdot k$ floats in total.** With $k$ held fixed as $d$ grows, that is $O(d)$. Adam, for comparison, stores $2d$. This is the whole reason for blocking.

## 5.4 Why the blocks must be contiguous

A block boundary is a decision about which correlations you are allowed to delete. Whitening removes correlation *inside* a block and does nothing to correlation *between* blocks.

Look back at the correlation table in Section 3. Correlation between two columns is a function of how far apart their centers sit: neighbours are at $0.999$, columns 64 apart are at $0.03$. **Contiguous blocks therefore trap nearly all of the correlation inside blocks, where whitening can destroy it.** A random partition scatters each strongly-correlated pair across different blocks, where whitening cannot touch it.

Measured at $N = 128$, $k = 128$, identical matrix, only the partition changed:

| partition | $\kappa(B)$ | distinct singular-value clusters | columns kept |
|---|---|---|---|
| contiguous | $2.8 \times 10^{7}$ | 13 | 142 (true rank is 144) |
| random | $1.3 \times 10^{13}$ | 83 | 153 |

$\kappa(\Phi)$ itself is $6.8 \times 10^{14}$. Contiguity buys seven orders; a random partition buys one.

There is a second, independent reason. **The rank deficiency is spatially localized: it is the halo.** At $N=128$, $k=128$, the third contiguous block is columns 256 to 269, which are thirteen centers sitting at $x \approx 1.91$ to $2.09$ (entirely outside the data) plus the bias column. Every one of them is numerically constant, so that whole block has rank 1. Pivoted QR sees it immediately, keeps one column, drops thirteen. A random partition smears those saturated columns across all three blocks, where each looks like a small perturbation and no block registers as rank deficient. It keeps 153 columns against a true rank of 144, meaning it inverts nine directions that are numerical noise.

**Contiguity is doing geometric work.** It is meaningful only because column index equals grid position in this 1-D problem. Section 13 flags what that implies for generalizing.

## 5.5 The one remaining trap

Given $M$, the obvious implementation is to hand the solver a matrix-free operator that applies $\Phi$ then $M$ on every matvec. That is how preconditioners are normally implemented, and here it destroys the method completely.

$M$ contains $R_C^{-1}$, whose condition number is around $10^{14}$. Applying it injects relative error $\varepsilon \kappa(R) \approx 10^{-2}$ into *every* matvec, and the Golub-Kahan recurrence compounds that error across iterations.

**So $B$ is formed explicitly, once, and stored.** The iteration touches only $B$. Each $R_C^{-1}$ is applied exactly one time, at the very end. Section 8.2 has the measurement: implicit application fails outright at relative error $1.0$, materialized $B$ reaches $7.5 \times 10^{-15}$.

The cost of that choice is honest and worth stating: $B$ is a second $n \times d$ array. It is whitened data, the same size as the data, not optimizer state. Section 13 lists it as a real limitation.

---

# 6. The algorithm

## 6.1 Statement

Given $\Phi \in \mathbb{R}^{n\times d}$, $y \in \mathbb{R}^n$, block size $k$, and threshold $\texttt{rcond} = 10^{-13}$.

**Stage 1, whiten.** Cost $O(ndk)$, which is $k$ passes. Run once.

$$\begin{aligned}
&\textbf{for } C = \{0..k{-}1\},\ \{k..2k{-}1\},\ \dots \ \textbf{(contiguous)}:\\[2pt]
&\qquad Q_C,\,R_C,\,\Pi_C \;\leftarrow\; \texttt{pivoted Householder QR}\big(\Phi_{:,C}\big) \quad\text{so that}\quad \Phi_{:,C}\Pi_C = Q_C R_C\\[2pt]
&\qquad n_C \;\leftarrow\; \#\{\,j : |(R_C)_{jj}| > \texttt{rcond}\cdot\max_j |(R_C)_{jj}|\,\}\\[2pt]
&\qquad B_{:,\,C[0:n_C]} \;\leftarrow\; (Q_C)_{:,\,0:n_C}\\[2pt]
&\qquad \textbf{store } (R_C)_{0:n_C,\,:},\ \Pi_C,\ n_C \quad\text{(the only persistent state)}
\end{aligned}$$

**Stage 2, solve.** Cost one forward-backward pass per iteration.

$$z \;\leftarrow\; \texttt{LSMR}(B,\,y), \quad\text{unpreconditioned, taking the minimum-error iterate (Section 8.6)}$$

Note that the iterates live in $z$, the whitened coordinates. There is no sequence of $w$'s: $w$ is produced once, in Stage 3, from the final $z$.

**Stage 3, unwhiten.** Cost $O(dk)$, run once.

$$\begin{aligned}
&\textbf{for each block } C:\\
&\qquad c \;\leftarrow\; \texttt{triangular\_solve}\big((R_C)_{0:n_C,\,0:n_C},\; z_{C[0:n_C]}\big)\\
&\qquad w_{C}\big[\Pi_C[0{:}n_C]\big] \;\leftarrow\; c
\end{aligned}$$

Implementation: `solvers.py::_blockqr`. Registry ids `blockqr_k128`, `blockqr_k64`. Run with `run.py --only blockqr_k128 --sweeps 800`.

## 6.2 A worked trace

The running example, $N = 128$, $k = 128$. Here $d = 270$, so there are three blocks.

| block | columns | what they are | rank found | columns kept |
|---|---|---|---|---|
| $C_1$ | 0 to 127 | 70 left-halo centers, then 58 live | 64 | 64 |
| $C_2$ | 128 to 255 | live centers plus right halo | 77 | 77 |
| $C_3$ | 256 to 269 | 13 right-halo centers + bias | 1 | 1 |

Total kept: 142, against a true numerical rank of 144. The pivoting recovers essentially the whole usable rank and discards 128 columns that are numerical noise.

$B$ then has 142 nonzero orthonormal columns and $\kappa(B) = 2.8 \times 10^{7}$, down from $6.8 \times 10^{14}$. LSMR converges on that in tens to low hundreds of iterations. The three triangular solves at the end recover $w$.

Persistent state: three $R$ factors, $d \cdot k = 270 \times 128 \approx 35{,}000$ floats.

## 6.3 One implementation detail worth knowing

SciPy's `lsmr` has no per-iteration callback, so the trajectory cannot be sampled from inside a single run. The code instead re-runs `lsmr` from scratch at 40 logarithmically spaced iteration counts, each with `maxiter` set to that count. Every "trajectory" in the figures is 40 independent runs, not one instrumented run. The results are identical; the cost is 40 times higher than a deployed version would pay.

---

# 7. The governing principle

> **One application of an ill-conditioned operator is harmless. Iterating with one is fatal.**

This single sentence explains every design decision in Section 6, and it is worth stating why it is true rather than taking it on faith.

Applying an operator with condition number $\kappa$ once introduces relative error around $\varepsilon\kappa$ into the result. That is a fixed, one-time cost, and here it lands in a place where it does not matter, because $\Phi$ cancels the amplification when the function is evaluated. Applying it inside a recurrence introduces $\varepsilon\kappa$ into *every step*, and the recurrence compounds it. There is no cancellation available, because the error enters the search directions themselves.

**The receipt that this is the real mechanism:** the truncated SVD applies $\Sigma^{-1}$, condition number $10^{14}$, exactly once, and lands at $10^{-14}$ *function* error. Its *parameter* error genuinely is around $\varepsilon\kappa$, and you can measure that it is. $\Phi$ then re-multiplies by $\sigma$ on the way out and cancels the amplification. This method copies that structure exactly: all the ill-conditioning is confined to the single triangular solve in Stage 3.

**The corollary that explains the whole tuning surface:** $\kappa(B)$ controls *iteration count*, not attainable accuracy. Accuracy comes from every stage being backward stable, which Householder QR, LSMR, and triangular solve all are. This is why Section 8.5 finds that $k$ trades cost against cost, not cost against precision.

## Why the convergence curves are staircases

Whitening makes each block's columns exactly orthonormal, so within a block every singular value is exactly 1, and only cross-block coupling survives. **$B$ therefore has very few *distinct* singular values, with high multiplicities.** At $d = 270$: 140 distinct values at $k = 16$, 18 at $k = 128$, and 2 at $k = 256$ (two blocks).

A Krylov method annihilates a cluster of identical singular values with a single polynomial root, so one iteration clears an entire multiplicity at once. Section 4.2 derives this. The result is roughly one stair per distinct cluster rather than one per direction. The vertical drop is the polynomial factor being acquired; the flat plateau is the build-up to the next one.

That also explains something visible in `C_noise_3x3.png`: the step *locations* are identical across noise levels. They are a property of the operator, not of the right-hand side.

This is a sharper statement than "$\kappa(B)$ sets the iteration count". Whitening does not merely shrink $\kappa$. It collapses the spectrum into a handful of exactly degenerate clusters, which is the most favourable structure a Krylov method can be handed.

## Verification receipt

Compute $\|Q_C^\top Q_C - I\|$ for each block. Expect around $10^{-16}$.

The Gram-based whitening described in Section 4.3 gives about $10^{-2}$ on the same blocks, and that single number is the entire difference between $10^{-8}$ and $10^{-15}$ in the final answer. **Check this first if the method ever underperforms.**

---

# 8. Results and the four load-bearing details

## 8.0 The headline

Nine cells: three targets (`sine`, `sine_8pi`, `runge`) crossed with three widths ($N = 64, 128, 256$). fp64 throughout, single solve, no iterative refinement. Reported as eval rel $L_2$.

| cell | truncated-SVD floor | block-QR, $k=128$ |
|---|---|---|
| sine $N{=}64$ | $2.45\times10^{-15}$ | $1.66\times10^{-15}$ |
| sine $N{=}128$ | $3.43\times10^{-14}$ | $7.74\times10^{-15}$ |
| sine $N{=}256$ | $1.22\times10^{-14}$ | $7.31\times10^{-15}$ |
| sine_8pi $N{=}64$ | $1.40\times10^{-13}$ | $4.25\times10^{-13}$ |
| sine_8pi $N{=}128$ | $4.12\times10^{-14}$ | $7.54\times10^{-14}$ |
| sine_8pi $N{=}256$ | $3.56\times10^{-14}$ | $1.99\times10^{-14}$ |
| runge $N{=}64$ | $3.30\times10^{-9}$ | $3.30\times10^{-9}$ (the target's own floor) |
| runge $N{=}128$ | $2.26\times10^{-14}$ | $9.59\times10^{-16}$ |
| runge $N{=}256$ | $1.80\times10^{-14}$ | $1.77\times10^{-15}$ |

**Geometric mean $4.1 \times 10^{-14}$, against an SVD floor of $8.7 \times 10^{-14}$.** At or below the floor everywhere. Iteration counts: median 54, maximum 342.

The "floor" here is the truncated SVD's own error, which is the best any method can do on this data. Beating it slightly is not a paradox: the SVD truncates at $\texttt{rcond} = 10^{-15}$ and this method's per-block truncation makes a marginally different choice about which directions to keep.

Block-size frontier on the same nine cells (geometric mean): $k{=}8$: $7.5\times10^{-6}$; $k{=}16$: $1.5\times10^{-6}$; $k{=}32$: $1.1\times10^{-8}$; $k{=}64$: $4.8\times10^{-12}$; $k{=}128$: $4.1\times10^{-14}$.

## 8.1 Detail one: whiten by QR, never via the Gram

Householder QR gives $\Phi_C = Q_C R_C$ with $Q_C$ orthonormal to $\varepsilon$ by construction, at any $\kappa(\Phi_C)$, for the reason given in Section 4.3: reflections are orthogonal and never amplify.

Building the same object as $\Phi_C M_C^{-1/2}$, with $M_C$ obtained from `eigh`$(\Phi_C^\top \Phi_C)$, fails twice over. It squares the block condition number, so the small eigenvalues become fp64 noise, and it is catastrophic cancellation, since $\kappa(M^{-1/2})$ reaches $10^{14}$.

> **Gram route $1.6\times10^{-8}$ against QR route $1.7\times10^{-15}$.**

## 8.2 Detail two: materialize $B$; never apply $R^{-1}$ inside the iteration

The mechanism is Section 5.5. An operator that applies $R^{-1}$ per matvec injects relative error $\varepsilon\kappa(R) \approx 10^{-2}$ into every matvec, and the recurrence compounds it.

> **Implicit $1.0\times10^{0}$ (total failure) against materialized $7.5\times10^{-15}$.**

## 8.3 Detail three: contiguous blocks, not random

Mechanism in Section 5.4. Note that this is the *opposite* of what holds for block-Jacobi *preconditioning*, where random and contiguous partitions perform about the same. Blocking for whitening and blocking for preconditioning are not the same operation and do not share intuitions.

> **At $k{=}128$: contiguous $7.5\times10^{-15}$, random $9.6\times10^{-7}$.**

## 8.4 Detail four: pivot and drop

Column-pivoted QR exposes the rank-deficient halo blocks in $|\mathrm{diag}(R_C)|$, as traced in Section 6.2. Dependent columns are dropped, never inverted.

## 8.5 Block size $k$: memory is uncoupled from $d$, cost is not

**Memory.** Run to convergence with a metric that does not saturate. The saturating trap is explained in Section 11; the non-saturating choice is disagreement with the direct solve, $\|A_{ev}(w - w_{svd})\|/\|y_{ev}\|$. Target `abs_cubed`:

| $d$ | $k{=}16$ | $k{=}32$ | $k{=}64$ | $k{=}128$ | $k{=}256$ |
|---|---|---|---|---|---|
| 462 | $6.1\times10^{-12}$ | $1.3\times10^{-12}$ | $4.0\times10^{-14}$ | $7.9\times10^{-14}$ | $3.9\times10^{-14}$ |
| 692 | $9.5\times10^{-12}$ (cap) | $5.9\times10^{-14}$ | $6.8\times10^{-14}$ | $5.4\times10^{-14}$ | $2.3\times10^{-14}$ |
| 922 | $2.1\times10^{-9}$ | $1.9\times10^{-13}$ | $9.6\times10^{-14}$ | $4.0\times10^{-14}$ | $4.2\times10^{-14}$ |

**Every $k \ge 32$ reaches $10^{-13}$ to $10^{-14}$ regardless of $d$. Only $k = 16$ fails.** So the minimum viable state is $32d$, and it does not grow with $d$. (Figure `H_dk_iteration_law.png`.)

**Cost.** The iterations are where $d$ enters. Let $t(\tau, k, d)$ be the first iteration reaching disagreement $\tau$:

$$t \;\approx\; g(\tau)\, f(k/d)\, d^{-0.3}, \qquad g(\tau) \sim \tau^{-1/3}.$$

Three measured facts behind that form (figures `I_isolines.png`, `I2_law_tests.png`):

- **$t$ is a power law in $\tau$, not logarithmic.** Going from $10^{-4}$ to $10^{-12}$ costs $336\times$ the iterations, not $3\times$. The explanation: $\kappa$ is not fixed, it grows with the accuracy demanded. Reaching $\tau$ requires resolving down to $\sigma/\sigma_1 \approx \tau$, giving an effective $\kappa_{\mathrm{eff}} = 1/\tau$ and $t \propto \sqrt{\kappa_{\mathrm{eff}}} = \tau^{-1/2}$. Measured $\tau^{-1/3}$ is shallower than that worst-case bound, as expected for a clustered spectrum.
- **$k$ enters as the ratio $k/d$, not as $k$.** Plotting $t$ against $k/d$ collapses all eight widths ($d = 206$ to $692$) onto one curve over five orders of magnitude. $R^2$ using $k/d$ is $0.95$ against $0.84$ using $k$ alone (at $\tau = 10^{-12}$). Whitening flattens a *window* of $k$ adjacent directions, so the fraction of the matrix that window covers is what matters.
- **A separable power law is the wrong functional form.** The fitted $k$-exponent drifts with $d$ (from $3.3$ to $2.6$ for sine), and the global fit $t \approx 10^{0.03}\tau^{-0.22}k^{-1.68}d^{1.37}$ has $R^2 = 0.849$. Use it as a local approximation only.

> **Practical consequence: the budget that matters is the fraction $k/d$. Holding $k$ fixed as $d$ grows does not hold difficulty fixed.**

**Choosing $k$.** Use $k = 32$ for minimum state ($32d$). Use $k = 128$ for minimum total cost. Below 32 it fails outright.

## 8.6 Semiconvergence: take the minimum, always

**The method overshoots and then drifts back up, with no noise present.** At `abs_cubed`, $d = 922$, $k = 64$: disagreement reaches $9.6\times10^{-14}$ at iteration 6900, then degrades to $8.0\times10^{-9}$ by iteration 60000. Five orders lost. Final-versus-best ratios reach $13\times$ on sine.

Three consequences:

- **Every accuracy number in this document is the minimum over the trajectory.**
- A deployed version must take the minimum, or stop early.
- The obvious stopping rule, watching for the $\|B^\top r\|$ plateau, was measured to fire far too early. It stopped $k{=}32$ at 6475 iterations at $10^{-9}$, when 42400 iterations reach $1.9\times10^{-13}$. **The stopping rule is unsolved**, and this is the largest practical gap in the method.

## 8.7 Noise: it lands exactly on the statistical floor

With Gaussian noise of relative scale $\sigma_{rel}$ added to $y$, the achieved error is $0.252,\ 0.251,\ 0.250,\ 0.249 \times \sigma_{rel}$ at $\sigma_{rel} = 10^{-8}, 10^{-6}, 10^{-4}, 10^{-2}$. The statistical floor for rank-$r$ regression on $n$ rows is $\sigma\sqrt{r/n} = 0.272\,\sigma_{rel}$ here.

**Flat across six decades, and slightly below the floor coefficient**, because early stopping acts as regularization. Statistically optimal, not merely stable. (`C_noise_3x3.png`, `C2_noise_summary.png`.)

Semiconvergence is visible and mild under noise: each curve dips, then rises as the solver begins fitting noise. This is the one place a stopping rule is unambiguously load-bearing.

## 8.8 Batching: sets the floor, not the rate

Whitening *and* solving on the same random row batch of size $b$, drawn from $n = 8d$ rows:

| batch | geo-mean best | against floor $1.08\times10^{-13}$ |
|---|---|---|
| $b = 8d$ | $3.1\times10^{-14}$ | below |
| $b = 4d$ | $6.9\times10^{-14}$ | below |
| $b = 2d$ | $7.9\times10^{-13}$ | $7\times$ |
| $b = d$ | $3.5\times10^{-10}$ | $3000\times$ |

**The curves overlay during descent and separate only at their plateaus.** Batch size sets the achievable floor and never the convergence rate. Degradation is smooth. The only hard failure is $b = d$, where the batch system is square and the whitening has no redundancy to work with. **Use $b \ge 4d$.** (`D1_batch_trajectories_3x3.png`.)

## 8.9 Rows needed for the QR factorization

There is **no threshold**. Accuracy improves smoothly with the row count $s$ used for the QR, and saturates only near the full row count. At $k = 128$, $n = 8d$: $s = 2k \to 10^{-6}$; $4k \to 2.2\times10^{-10}$; $8k \to 2.5\times10^{-12}$; $16k \to 4.8\times10^{-13}$; all rows $\to 5.3\times10^{-14}$.

An earlier claim that "$s = 8k$ matches full accuracy" was measured at $n = 4d$ and is **withdrawn**. Worse, the redrawn figure shows the required $s$ grows with $d$ and not only with $k$, so normalizing in multiples of $k$ was the wrong choice of axis. (`D2_qr_rows.png`.)

## 8.10 Cost accounting

| item | cost |
|---|---|
| persistent state | the $R$ factors, $d\cdot k$ floats |
| whitening (the QR, which produces $B$ directly) | $O(ndk)$, i.e. $k$ passes |
| per LSMR iteration | one matvec with $B$, one with $B^\top$, i.e. one forward-backward pass |
| total to floor ($k{=}128$, $d{=}462$) | about 530 pass-equivalents |

A direct QR of the whole of $\Phi$ costs $O(nd^2)$, which is $d$ passes. At $d = 462$ that is 462, so **at small $d$ this method is not cheaper than a direct solve. Its advantage there is purely memory** ($dk$ against $d^2$). Iteration growth is sublinear, so at $d = 3688$ the method costs about 2180 passes against about 3688 for the direct solve, winning on flops as well. The crossover lies somewhere between $d \approx 900$ and $d \approx 3700$.

One factor is unmeasured and favours the method: the whitening is a matmul ($n \times k$ by $k \times k$) that runs near peak FLOPs, while a matvec is memory-bandwidth-bound. Wall-clock is likely well below what the flop count implies. Worth measuring before optimizing anything.

---

# 9. How to integrate it into training

## The binding fact: the whitening cannot be stale

Suppose the geometry moves, so $\Phi$ becomes $\Phi + \Delta\Phi$, and you reuse the old $R$ factors. Then

$$B \;=\; (\Phi + \Delta\Phi)R^{-1} \;=\; Q + \Delta\Phi\,R^{-1},$$

and $\|R^{-1}\| \sim 1/\sigma_{\min}$ amplifies the drift by $\kappa \approx 10^{14}$. Measured, with $\eta$ the perturbation applied to the centers and bandwidth:

| geometry drift $\eta$ | $\|\Delta\Phi\|/\|\Phi\|$ | stale $R$ | fresh $R$ |
|---|---|---|---|
| $0$ | $0$ | $2.26\times10^{-14}$ | $2.26\times10^{-14}$ |
| $10^{-8}$ | $3.4\times10^{-8}$ | $2.04\times10^{-6}$ | $2.04\times10^{-14}$ |
| $10^{-3}$ (Adam scale) | $3.5\times10^{-3}$ | $7.6\times10^{-2}$ | $1.68\times10^{-10}$ |

A $10^{-8}$ perturbation costs eight orders. **So this is a solve-occasionally method, not a solve-every-step method.** Amortized over a refresh every $T$ base steps, the cost is about $530/T$ passes per step; $T = 200$ gives $2.7$, roughly $3\times$ Adam.

## Recommended architecture: two tiers

The argument against chasing machine precision at every step comes from this repository's own measurement. expD08 iteration 11 established the **coupling law**: after an exact readout solve, one ordinary Adam step re-injects about $\|v\|\eta$ of error, where $v$ is the readout weight vector and $\eta$ the step size. That was measured across five decades of $\|v\|$ and six of $\eta$; at $\|v\| = 0.5$, $\eta = 10^{-3}$, it is $5\times10^{-4}$.

A $10^{-14}$ readout is therefore destroyed by the very next base step. **The precision only banks once the geometry stops moving.** Hence:

- **Tier 1, every step, cheap.** A few LSMR iterations with a block-Jacobi preconditioner, no whitening. Lands $10^{-6}$ to $10^{-9}$ in about 3 passes. That is enough while the geometry is still moving, which is all the coupling law lets you keep anyway.
- **Tier 2, rarely, or once at the end.** Full block-QR whitening plus LSMR to the floor, once the geometry has settled. Gauge settling on the drift measure that expD08 iteration 11 already computes. This is where $10^{-14}$ survives.

Tier 2 runs at roughly $3\%$ of the naive every-step cost, for the same final precision.

## If every-step machine precision is genuinely required

One untested lever: **localized mixed precision.** Apply $R^{-1}$ implicitly in double-double arithmetic instead of materializing $B$. Double-double represents each number as an unevaluated sum of two float64 words (Dekker/Knuth), roughly doubling the mantissa at a few times the arithmetic cost. Confined to a $k \times k$ triangular solve, that costs about $10k/n$ of a pass rather than $k$ passes, which is nearly free when $n \gg k$. This is materially different from running everything in extended precision.

Supporting evidence: LSQR run entirely in double-double reached $2.1\times10^{-16}$ at iteration 100 with only a block-Jacobi preconditioner. It also needs a *deliberately* ill-conditioned preconditioner ($\kappa(M) = 2.6\times10^{14}$); weakening it to $10^{6}$ made the method diverge. Reference implementation in `dd.py`, `ddpc.py`, `ddlsqr.py`.

---

# 10. Do not re-try these

All measured, all dead.

| idea | outcome |
|---|---|
| reuse a stale $R$ across steps | a $10^{-8}$ drift costs 8 orders |
| apply $R^{-1}$ implicitly in fp64 | total failure, $1.0\times10^{0}$ |
| random blocks | not better conditioned; 8 orders worse |
| shrink $k$ below 32 | fails outright ($k{=}16$: $2.1\times10^{-9}$ at $d{=}922$) |
| diagonal preconditioning | a no-op; all $\tanh$ column norms are equal to within $1.6\%$ |
| banded Cholesky | impossible; Gram off-diagonals do not decay (0.99 down to only 0.74) |
| Toeplitz / circulant (FFT) | $G - T$ is full rank at every halo width |
| DCT / DFT as a fixed diagonalizer | 47 to 66% of off-diagonal mass remains |
| Nystrom rank-$\ell$ preconditioner | worse than block-Jacobi at equal state; no spectral gap to exploit |
| polynomial preconditioner | dead by theory: required degree $\sim\sqrt\kappa \approx 10^{7}$ |
| extended-precision residuals only | changed results by under 1%; residual rounding was never the barrier |
| full Krylov reorthogonalization | $7\times$ speedup, **zero** accuracy gain; orthogonality was never the barrier |
| exact block coordinate descent as a standalone solver | contraction $\rho\approx0.997$ per sweep, about $10^4$ sweeps to the floor |
| two-level / deflation (block plus coarse space) | 4 attempts, none correct; spectrally it looks right ($\kappa$ falls $10^9 \to 30$ at $c{=}17$) but no working solver was produced |

---

# 11. Methodology warnings

The $k$-versus-$d$ question was answered **wrongly four times** before it was answered correctly. Every one of these failures is easy to repeat.

| failure | mechanism |
|---|---|
| fixed iteration budget across $k$ | starves small $k$, which manufactures a false coupling |
| budget scaled as $1/k$ | *still* under-feeds small $k$ (gave $k{=}32$ 8436 iterations when 42400 were needed) |
| ratio-to-floor as the metric | **saturates at 1.0** once the solver beats the approximation error, so it cannot resolve anything below that |
| best-over-trajectory reported per-metric | `min(eval)` and `min(disagreement)` land at *different* iterations, producing algebraically impossible pairs |
| fixed recording stride | quantized $t$ to 1 or 2 steps at large $k$, exactly where the $k$-exponent is measured |
| patience-based stopping | fired far too early on slowly-converging cells; the same confound as a short budget |

**The general rule: any per-cell budget or stopping rule that varies with the swept variable will manufacture a coupling.** Use a fixed generous cap, take the minimum, flag cells that hit the cap, and use a metric that does not saturate.

One further trap, separate from the above. Measure $\kappa$ as `svd(A @ Minv)` computed from $A$ **directly**. Deriving it from an explicitly formed $A^\top A$ floors the small eigenvalues at $\varepsilon\|A\|^2$, which is $10^{-10}$ here against a true $\sigma_r^2 \approx 10^{-24}$. That error put $\kappa$ off by six orders and inverted the contiguous-versus-random ordering in an earlier round.

---

# 12. Where this sits against everything else tried

From `SUBPROBLEM.md`, best method per memory class, geometric mean over the nine cells:

| state | method | geo-mean eval rel $L_2$ |
|---|---|---|
| $3d$ | CGLS | $2.4\times10^{-6}$ |
| $O(k^2) = 8.9d$ | LSMR + block-Jacobi preconditioner | $1.1\times10^{-9}$ |
| $\mathbf{128d}$ | **block-QR whitening + LSMR** | $\mathbf{4.1\times10^{-14}}$ |
| $d\cdot r$ ($270d$) | SPIR (sketch-and-precondition + iterative refinement) | $3.7\times10^{-15}$ |
| $d^2$ | truncated SVD (reference floor) | $8.7\times10^{-14}$ |

SPIR is more accurate but needs $O(d\cdot r)$ memory, which for these problems is about $270d$ against this method's $128d$ (and $32d$ if minimum state is the priority). SPIR is Epperly, Meier & Nakatsukasa 2024, arXiv:2406.03468.

---

# 13. Known limits and open questions

- Verified on a **frozen** $\Phi$, 1-D toy geometry, noise-free unless stated otherwise.
- **The stopping rule is unsolved** (Section 8.6). This is the largest practical gap.
- $B$ is a second data-sized array, $n \times d$. Not optimizer state, but not free either.
- **Load-bearing detail three (contiguous $\gg$ random) may not survive** a deep or high-input-dimension $\Phi$, where column ordering carries no spatial meaning and `idx[i:i+k]` degenerates into an arbitrary partition, which is the random arm. The correct generalization is *group columns whose centers are close in the lattice metric*, which reduces to consecutive indices only in 1-D. **Re-measure this first** when moving to the real architecture; it is the assumption most likely to break.
- The $k/d$ collapse is not exact. A residual $d^{-0.3}$ survives, worth a factor of about 2 over the tested range, against the $\sim10^5$ spanned by $k/d$.
- Untested: row subsampling under noise simultaneously, column masking, deeper $\Phi$, and whether the $k \in (16, 32]$ lower bound drifts beyond $d \approx 3700$.

---

# 14. Where everything lives

| what | where |
|---|---|
| solver | `solvers.py::_blockqr`, ids `blockqr_k128` / `blockqr_k64` |
| problem statement and full history | `SUBPROBLEM.md` |
| validation suite | `validation/` (`run_sweep.py`, `run_law.py`, `run_isolines.py`, `plots*.py`) |
| validation writeup | `validation/README.md` |
| double-double reference | `dd.py`, `ddpc.py`, `ddlsqr.py` |
| tests (7, all passing) | `tests/test_expD09.py` |
| figures | `results/checkpoint_D_optimizers/expD09_2nd_order_regime/{figures,validation/figures}/` |

Key figures: `blockqr_k128.png` (the headline result), `H_dk_iteration_law.png` ($k$ uncoupled from $d$), `I_isolines.png` and `I2_law_tests.png` (the cost law), `C_noise_3x3.png` and `C2_noise_summary.png` (noise), `D1_batch_trajectories_3x3.png` (batching), `D2_qr_rows.png` (QR rows), `A_trajectories_3x3.png` (convergence trajectories).
