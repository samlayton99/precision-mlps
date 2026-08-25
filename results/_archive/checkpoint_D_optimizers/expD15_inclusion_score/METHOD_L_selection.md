# Discovering the least-squares block: methods, tradeoffs, and what to try next

This document is self-contained. It assumes you know what a neural network and a least-squares problem are, and nothing else about this project. It gives the theory, the shared measurement primitives, four selection mechanisms that work, one that does not, the measured tradeoffs between them, and enough implementation detail to build and evaluate any of them from scratch.

There is no single winner. Four mechanisms are viable, they fail in different places, and the two most promising have never been combined. Sections 6 to 10 are the catalogue; Section 11 is the comparison table; Section 12 is what to build next.

## 1. The problem

Train a one-hidden-layer network on a regression target and the error stalls around $10^{-3}$. Build the same network by hand from quasi-interpolant theory and you get $10^{-15}$. Training loses twelve digits the architecture provably holds.

The cause is not tuning. Split the parameters in two. The network is *exactly linear* in some of them and curved in the rest. For the linear ones, fitting is a linear least-squares problem with a closed-form solution that a direct solver reaches to machine precision in one shot. Gradient descent on that same subproblem needs about $(\sigma_1/\sigma_j)^2$ steps to fix the direction with singular value $\sigma_j$, and on these feature matrices $\sigma_{400}/\sigma_1 \approx 3\times10^{-19}$. That is arithmetic, not patience.

So the goal is an optimizer that runs ordinary gradient training on the curved parameters, solves the linear ones exactly, and **discovers which is which by measurement**. Hard-coding "solve the last layer" works on the toy and fails on any architecture where the linear block sits elsewhere. Section 12's measurements show it fails on a ResNet and on a transformer, where the best block is not the head.

This document covers the discovery problem only. The optimizer that consumes the answer is separate and still open.

## 2. Setup and notation

The running model is a one-hidden-layer network with activation $\sigma$:

$$f(x;\theta) \;=\; \sum_{k=1}^{W} v_k\,\sigma(a_k\!\cdot\!x + b_k) \;+\; c_0 ,\qquad x\in\mathbb R^{d},$$

with parameters collected into one vector

$$\theta \;=\; (\underbrace{A}_{W\times d}\,|\,\underbrace{b}_{W}\,|\,\underbrace{v}_{W}\,|\,c_0) \;\in\;\mathbb R^{m},\qquad m = W(d{+}2)+1,$$

where $a_k$ is row $k$ of $A$. Given inputs $x_1,\dots,x_n$ and targets $y_1,\dots,y_n$, the **residual** and the loss are

$$r(\theta)\in\mathbb R^{n},\quad r_i(\theta) = f(x_i;\theta)-y_i,\qquad \mathcal L(\theta) = \tfrac1n\|r(\theta)\|_2^2 .$$

The **Jacobian** $J(\theta)\in\mathbb R^{n\times m}$ has entries $J_{ik} = \partial f(x_i;\theta)/\partial\theta_k$. Write $J_k$ for column $k$, a vector in $\mathbb R^n$. Differentiating gives all four column types:

$$\frac{\partial f}{\partial v_k} = \sigma(a_k\!\cdot\!x+b_k), \qquad \frac{\partial f}{\partial c_0} = 1,$$

$$\frac{\partial f}{\partial b_k} = v_k\,\sigma'(a_k\!\cdot\!x+b_k), \qquad \frac{\partial f}{\partial A_{kj}} = v_k\,\sigma'(a_k\!\cdot\!x+b_k)\,x_j .$$

One asymmetry in those lines drives everything: **the readout columns contain no $v$, and the geometry columns are proportional to $v$.** Nothing else about $\sigma$ matters.

For an index set $S\subseteq\{1,\dots,m\}$, let $P_S:\mathbb R^{|S|}\to\mathbb R^m$ embed a vector on the coordinates of $S$ and zero elsewhere, and let $J_S = J P_S$ be the corresponding column submatrix.

## 3. What belongs in $L$, and why the answer is not unique

**Definition (affine block).** $S$ is an *affine block at $\theta$* if $\delta\mapsto f(\theta+P_S\delta)$ is affine in $\delta$ for every $\delta$. Equivalently, every second derivative within $S$ vanishes:

$$\frac{\partial^2 f}{\partial\theta_i\,\partial\theta_j} = 0 \qquad\text{for all } i,j\in S. \tag{3.1}$$

If $S$ is an affine block then $f(\theta + P_S\delta) = f(\theta) + J_S\delta$ exactly, so minimizing the loss over $S$ becomes

$$\min_{\delta}\;\bigl\|J_S\delta + r(\theta)\bigr\|_2^2 , \tag{3.2}$$

a linear least-squares problem whose minimizer is the **global** minimum over that subspace, not a local step.

### Affine blocks conflict, and the maximum is not a whole tensor

Condition (3.1) is a joint condition, so two sets can each be affine while their union is not. This is the normal case in real architectures, not an edge case.

Take a residual block whose output feeds a linear head: $\text{out} = v\cdot(h + W_2 z) + c_0$, where $h$ and $z$ do not depend on $v$ or $W_2$. Measured curvature on a 5-block ResNet, where $\approx 10^{-14}$ means affine and $10^{-3}$ means not:

| set | curvature |
|---|---|
| $\{v, c_0\}$ | $9.6\times10^{-15}$ |
| $\{W_2^{(4)}, b_2^{(4)}, c_0\}$ | $4.2\times10^{-14}$ |
| **their union** | $6.4\times10^{-3}$ |
| $\{W_2^{(3)}, b_2^{(3)}, c_0\}$ (earlier block) | $4.5\times10^{-4}$ |

Both are affine; together they are not, because the output contains $v\cdot W_2 z$, a product of two unknowns. The earlier block is correctly excluded, blocked by the next activation.

**The conflict is per-index, not per-tensor.** The product term is $v_i\,W_2[i,j]$, so $v_i$ conflicts only with *row $i$* of $W_2$. A set can therefore take some $v$ entries and some $W_2$ rows as long as the indices are disjoint. The transformer measurements in Section 12 show discovery finding exactly such a mixture, and Section 9 shows it is worth up to $40\times$.

This has a consequence for evaluation that cost me a false result. Any "oracle" defined as *the largest tensor-aligned affine set* is not the best affine set, so scoring against it overstates performance. Use it only with that caveat stated.

### Purity is everything; recall is cheap but not free

Two kinds of error are possible and they are wildly asymmetric. From a starting point where solving the readout gives eval relative $L_2$ of $8.127\times10^{-3}$:

| perturbation of the set | resulting error |
|---|---|
| drop 2% of it | $8.127\times10^{-3}$ |
| drop 20% of it | $8.100\times10^{-3}$ |
| add 2% wrong | $4.4\times10^{-2}$ |
| add 5% wrong | $7.1\times10^{1}$ |
| add 20% wrong | $1.1\times10^{3}$ |

A wrong member is not merely useless. Equation (3.2) has no remainder term, so the solver takes the step the linear model calls optimal, and for a curved coordinate that step can be arbitrarily large. The least-squares system is coupled, so the damage spreads through the shared solve. Solving 90% of a coupled system is not 90% as good.

**Design consequence: when uncertain, exclude.** A smaller clean set beats a larger contaminated one by orders of magnitude.

One caveat found late. The table above was measured where the achievable error is $10^{-3}$. In a high-precision regime the tolerance shrinks: on a 1-D case reaching $10^{-8}$, losing 3% of the readout cost a factor of 47. Recall gets less cheap as the precision target tightens.

## 4. The characterization every method uses

Condition (3.1) is a statement about second derivatives, which are expensive and, as Section 10 shows, impossible to estimate at the accuracy needed. Rewriting it in terms of the Jacobian gives something measurable.

Since $J_i = \partial f/\partial\theta_i$, the second derivative $\partial^2 f/\partial\theta_i\partial\theta_j$ is exactly $\partial J_i/\partial\theta_j$. So (3.1) says:

> **$S$ is an affine block if and only if every column $J_i$ with $i\in S$ is unchanged by any movement of the parameters in $S$.**

$$J_i(\theta + P_S\delta) \;=\; J_i(\theta) \qquad\text{for all } i\in S,\ \text{all } \delta. \tag{4.1}$$

Applied to the running model: for $i$ in the readout, $J_{v_k}=\sigma(a_k\!\cdot\!x+b_k)$ contains no $v$, so moving the readout leaves it unchanged and the readout is an affine block. For a geometry coordinate, $\partial J_{b_k}/\partial v_k = \sigma'(a_k\!\cdot\!x+b_k)\neq 0$, so moving $v_k$ moves that column and $b_k$ cannot join. Discovery has to reach that conclusion without the algebra.

## 5. Shared measurement primitives

Every method in the catalogue uses the same three pieces. Build these once.

### The probe

Draw a random sign vector $z\in\{-1,+1\}^{|S|}$, set a per-coordinate scale $\varsigma_i = \max(|\theta_i|,\varsigma_{\min})$ with $\varsigma_{\min}=10^{-2}$, and form

$$s \;=\; P_S\bigl(\delta\,\varsigma_S\odot z\bigr), \qquad \delta = 10^{-2}. \tag{5.1}$$

The observable is relative column motion:

$$\text{mv}_i \;=\; \frac{\bigl\|J_i(\theta+s) - J_i(\theta)\bigr\|_2}{\bigl\|J_i(\theta)\bigr\|_2}, \qquad \text{admit when } \text{mv}_i \le \texttt{tol}. \tag{5.2}$$

Use $\texttt{tol}=10^{-10}$ with an analytic Jacobian, $10^{-6}$ with a finite-difference one. The gap between affine and non-affine is nine orders, so the threshold is not delicate; set it above the Jacobian's own noise floor.

**The probe must not be the optimizer's solve step.** Near convergence that step shrinks toward zero, no column moves, and every coordinate reads as affine. Condition (4.1) quantifies over all $\delta$, so the probe is a fixed-size step chosen independently of the current fit. This mistake silently inflated an earlier round of results.

**Normalize per column, not globally.** Dividing by a global scale instead of $\|J_i\|$ collapses recall to 1–45% and breaks precision. Measured, both ways.

### The one-backward sketch

Equation (5.2) needs the Jacobian, an $n\times m$ object that cannot be formed at scale. It is not needed: the question is only whether column $i$ *changed*, and a random contraction answers it. Fix $u\in\mathbb R^{n}$ once and define

$$q(\theta) \;=\; J(\theta)^{\!\top} u \;\in\;\mathbb R^{m}, \tag{5.3}$$

one reverse-mode pass with $u$ as the output cotangent, producing a full length-$m$ vector. Then

$$\widetilde{\text{mv}}_i \;=\; \frac{\bigl|q_i(\theta+s) - q_i(\theta)\bigr|}{|q_i(\theta)| + \epsilon},\qquad q_i(\theta+s)-q_i(\theta) = \bigl(J_i(\theta+s)-J_i(\theta)\bigr)\!\cdot\! u . \tag{5.4}$$

Zeros survive the contraction: if $J_i$ is unchanged the difference vector is identically zero, so its inner product with any $u$ is zero. Only nonzeros are at risk, and only when $u$ happens to be orthogonal to the change.

Measured against the full-Jacobian version on four architectures, the sketch matched it exactly on three and lost one coordinate on the fourth. **Keep $u$ fixed.** A fresh $u$ per probe costs twice the passes and performs worse on deep models.

### The value score

Several methods need to rank candidates. Use the residual reduction from solving parameter $i$ alone:

$$\text{val}_i \;=\; \frac{(r\cdot J_i)^2}{\|J_i\|^2}, \tag{5.5}$$

free given the gradient and the column norms. This is what makes conflicts resolve correctly: when two parameters are bilinearly coupled, the higher-value one is admitted first and its partner is then refused.

## 6. Method A: top-down pruning to a fixed point

**Idea.** Start with everything and repeatedly remove whatever moves. Define

$$M(S) \;=\; \Bigl\{\, i \;:\; \tfrac{\partial J_i}{\partial \theta_j} = 0 \ \text{ for all } j\in S \,\Bigr\},$$

the coordinates unmoved by motion inside $S$. Then $S$ is an affine block exactly when $S\subseteq M(S)$.

**Why the iteration must be gentle.** $M$ is **antitone**: enlarging $S$ imposes more conditions, so $S\subseteq S'$ implies $M(S)\supseteq M(S')$. Iterating $S\leftarrow M(S)$ oscillates. Use the monotone form $S_{t+1} = S_t\cap M(S_t)$, which decreases and converges.

But that over-prunes by construction. Start from everything: $J_{v_k}$ moves because the geometry moved, so the entire readout is evicted alongside it. Applying a hard threshold removes essentially all coordinates in one step and returns nothing. So prune by a **quota**: rank by $\text{mv}$ and keep the smallest $\lceil\kappa|L|\rceil$, with $\kappa\in[0.96,0.99]$, never evicting anything already below `tol`. Gradual eviction lets genuine offenders leave first.

**Measured.** 100% precision in every configuration. Recall 96–99% on the geometry regimes at 28–34 passes; 33% on a ResNet; 2% on a conv net whose head sits behind two conv stages.

**Failure mode.** Eviction is permanent. When the affine block's Jacobian depends on many upstream parameters, the block looks unstable until the upstream quiets, and by then it is gone. Adaptive gap-based pruning does not fix this and is unstable (recall 60–68%, error ratios to $10^8$); the fixed quota is what works.

## 7. Method B: bottom-up growth in value order

**Idea.** Start from the empty set and only add. This inverts Method A's failure: $L$ is clean at every step, so nothing is ever committed wrongly.

**The safety lemma, which makes it cheap.** If $L$ is affine and we perturb $L\cup C$ for a candidate batch $C$, then any $c\in C$ that does not move satisfies $\partial J_c/\partial\theta_j=0$ for every $j\in L\cup C$, including $j=c$. So $c$ is affine with respect to $L\cup\{c\}$ and is safe to admit. Two non-movers are mutually non-disturbing by the same argument, so admitting all of them at once is also safe.

**A large batch therefore costs only false negatives**, never false positives. One nonlinear member disturbs the other candidates, so good candidates get rejected and must be retried, but nothing wrong is ever let in.

**Batch size is the tuning knob and it matters a lot.** At batch 1 the test is exact and depth becomes irrelevant: perturbing $L\cup\{c\}$ moves only affine parameters, so a candidate whose column depends solely on upstream weights (which are not in $L$) cannot move. Larger batches are cheaper but admit a deep block only when the batch happens to contain no upstream weight. Measured on a 2-hidden-layer net at $m=257$ with 9 affine coordinates, as precision/recall against step count:

| batch | 300 steps | 1000 | 2000 |
|---|---|---|---|
| 6 | 100 / 11 | 100 / 11 | 100 / 11 |
| 2 | 100 / 78 | **100 / 100** | 100 / 100 |
| 1 | 100 / 56 | 100 / 89 | **100 / 100** |

**Make the batch adaptive in the direction that grows.** Start at 1, double after a round where every candidate was admitted, halve after a round where none was. Clean regions clear at exponentially growing batch; contested regions fall back to the exact single test. This is the opposite of the natural first guess, and getting it backwards leaves coverage at 20–60%.

**Stopping.** Walk down the value-sorted list and stop when the remaining candidates' total value falls below $\varepsilon_{\text{val}}$ times the value already captured, with $\varepsilon_{\text{val}}=10^{-3}$. One scalar comparison per round. Certifying true maximality instead would need a batch-1 pass over every remaining parameter, which is $O(P)$ and unusable at scale.

**Measured.** Coverage of the largest available affine set: transformer 84%, ResNet 90%, skip-plus-basis 97%, conv head 29%. Across 15 depth cells: median error ratio 1.00, worst 11.45, 26–200 passes.

**Failure mode.** Cost is $\approx 1$ pass per parameter. High-value *nonlinear* parameters sit at the top of the value order and each must be tested and rejected individually, so the batch collapses to 1 in contested regions. Deferring failed batches instead of bisecting them does not help: cost stays at $\approx 1.2m$ and the worst ratio degrades from 11 to 45.

## 8. Method C: per-tensor sampling with verification

**Idea.** The cost of Methods A and B is $\Omega(P)$ for an information reason: classifying $P$ items with a per-item test needs $\Omega(P)$ tests unless the items are correlated. The correlation is the tensor grouping, and it is the one piece of structure every framework already provides. This does not require assuming an architecture, only that parameters arrive in named tensors.

**The rule.** Visit tensors in descending total value $\sum_{i\in T}\text{val}_i$. Sample $n_{\text{samp}}=2$ parameters from tensor $T$ and probe them against the current $L$. If the samples are clean, tentatively admit all of $T$ and **verify once** with a full-tensor probe; keep it only if the verification passes. Stop on the same value rule as Method B.

Two probes per tensor. A wrong generalization from the samples is caught by the verification rather than trusted, so precision stays exact.

**Measured.** This is the cheapest and most reliable mechanism available.

| case | $m$ | tensors | coverage | ratio | passes |
|---|---|---|---|---|---|
| 8 geometry and data regimes | 289–430 | 4 | 100% | 1.00 | **7** |
| transformer, 2 pre-norm blocks | 261 | 22 | 100% | 1.00 | 25 |
| ResNet, 5 blocks | 213 | 24 | 100% | 1.00 | 28 |
| conv, last layer unactivated | 33 | 5 | 100% | 1.00 | 9 |
| skip plus fixed basis | 77 | 6 | 100% | 1.00 | 11 |
| conv, head behind two conv stages | 49 | 6 | 35% | 1.00 | 9 |
| mlp / resnet / skip, depths 1–5 | 33–213 | 6–24 | 100% | 1.00 | 9–28 |

Passes track the tensor count, not $m$. A billion-parameter model with a few hundred tensors costs a few hundred probes, and widening the layers does not change that.

**Failure mode, and it is the important one.** Whole-tensor admission cannot express a mixed set, so it cannot find the per-index threading of Section 3. Measured cost of that limitation: up to $40\times$ (Section 9).

**Sparse fallback.** On a tensor that fails verification, keep only the members that stayed clean in the full-tensor probe. This is still safe by the lemma in Section 7. It changed nothing on the cases tested, because those tensors failed at the sample stage rather than the verification stage.

## 9. Method D: the amortized belief

**Idea.** Discovery does not have to finish before the optimizer starts. Keep a Beta posterior per parameter and update it from one probe per optimizer step, so cost rides along with training instead of being paid up front.

State is $(\alpha_i,\beta_i)$, initialized to $(1,1)$: two floats per parameter, the same order as Adam's two moments. Each probe gives a binary observation for the coordinates it tested, and the conjugate update with decay $\rho$ toward the prior $\alpha_0$ is

$$\alpha_i \leftarrow \rho\,\alpha_i + (1-\rho)\alpha_0 + \mathbb 1[\text{not moved}], \qquad \beta_i \leftarrow \rho\,\beta_i + (1-\rho)\alpha_0 + \mathbb 1[\text{moved}]. \tag{9.1}$$

Membership is $L = \{i : \alpha_i/(\alpha_i+\beta_i)\ge\tfrac12\}$. The decay matters beyond smoothing: the geometry is still training, so the affine block is a property of a moving $\theta$, and decay makes staleness a rate rather than a re-run.

**Explore and exploit must be different probes.** If every probe includes outside candidates, those candidates are usually nonlinear, they move, and their motion disturbs the genuinely affine members, which then accumulate negative evidence forever. On a 2-hidden-layer net this pinned recall at 11%. So:

- **Exploit probe.** Perturb only the selected set $L$. If $L$ is clean nothing moves and every member gets positive evidence. Update all of $L$.
- **Explore probe.** Perturb $L\cup C$ with $|C|\in\{1,2\}$ and update **only** $C$. The selected set must not be punished for a disturbance the probe injected. Run with probability $p_{\text{ex}}\approx0.4$.

**Seed it asymmetrically.** Method A has perfect precision and imperfect recall, so its *accepts* are trustworthy and its *rejects* are not. Seed a strong prior on what it accepted and **no prior at all** on what it rejected, so one clean observation can re-admit. Seeding both sides equally goes nowhere: a rejected parameter then needs eight positive votes to climb back.

**Measured.** Precision 100% at every checkpoint from step 10 onward, with recall climbing to 100%. On the 8 geometry regimes: seeded by Method A, all cells reach ratio 1.00 at roughly 330 total passes; one cell needs 800. From scratch it takes about 2000 probes.

**Use it when** the discovery must track a moving geometry, or when the last few percent of recall matters (the high-precision regime of Section 3), or when you want the per-step cost rather than a solve-event cost.

## 10. What does not work, and why

Recording these matters: each is a natural first idea, and each fails for a reason that constrains the design.

**Per-parameter curvature scores.** Estimate $\partial^2 f/\partial\theta_i^2$ by Hutchinson probes and rank. The target value for an affine coordinate is *exactly zero*, and every stochastic estimator has a noise floor, so the score reports its own variance. Worse, that variance scales with Gram row mass, which is largest for the readout, so the exactly-affine block scored as the *most* nonlinear. Fixing the estimator with common random numbers repairs the inversion but not the floor. Best purity reached 64–68% in 2-D, which by Section 3 is worthless.

**Soft per-coordinate damping.** Skip selection: solve everything with $\mu_i$ proportional to estimated curvature, so wrong estimates degrade gracefully. With *exact* curvature this matches the oracle in all eight cells. With a cheap estimate at 4, 16, or 64 probes it is no better than random damping, for the same reason: an affine coordinate's true curvature is zero, so its damping is set by the estimator's noise, not the truth. Soft or hard, the problem is detecting an exact zero.

**Global uniform damping.** The no-selection baseline. Sweeping one global $\mu$ over fourteen orders reaches $2.7\times10^{-3}$ where the oracle reaches $1.8\times10^{-8}$. The blocks require different treatment.

**Exact set-level verification.** For $s$ supported on $S$, the second difference $f(\theta+s)-2f(\theta)+f(\theta-s)$ vanishes identically iff $S$ is affine. Exact, three forward passes independent of $|S|$ and $m$, with a measured separation of $10^{13}$ at probe step $\epsilon=0.1$. It works, matching the oracle in all eight cells. Abandoned on cost: it is a *group* test reporting only that $S$ contains an offender, so isolating them costs $O(\#\text{offenders})$ tests, roughly a thousand passes. The $O(\log m)$ variant (binary search on the longest clean prefix of a ranking) collapses to 3–8% recall, because one early offender caps everything behind it.

**A "rescue" pass** that re-admits anything not moving under an $L$-step. Wrong in principle, not mistuned: it admits nonlinear parameters whose columns merely do not *depend* on $L$. Precision fell to 8%.

**The depth hypothesis.** I predicted coverage would degrade with depth between the affine block and the input. It does not. Across three families at five depths the pattern is non-monotone (mlp 100/80/60/20/80, ResNet 100/95/100/76/100), and ResNet shows no trend at all because its best block sits one layer from the output regardless of total depth. What looked like a depth effect is contamination probability in the probe batch: deeper nets have more parameters, so a fixed-size batch more often picks up a contaminant. Fixing the batch rule (Section 7) removes it.

## 11. Choosing between the methods

| | A: top-down prune | B: bottom-up value grow | C: per-tensor | D: amortized belief |
|---|---|---|---|---|
| cost | $O(m)$ passes | $\approx 1$ pass/param | $O(\#\text{tensors})$ | 1–2 passes/step |
| measured passes | 28–34 | 26–200 | **7–28** | 330–2000 probes |
| precision | 100% | 100% | 100% | 100% |
| recall | 96–99% shallow, 2–33% deep | 84–97%, 29% conv head | 100% except conv head | 100% |
| finds mixed sets | no | **yes** | no | no |
| terminates on its own | yes | yes (value rule) | yes (value rule) | no, runs with training |
| tracks a moving $\theta$ | no | no | no | **yes** |

Precision is 100% in all four. Every set returned by every method verified exactly affine, on every architecture tested, including transformer attention, layer-norm parameters, and earlier-block weights, none of which was ever admitted.

**Default recommendation: Method C.** Cheapest, terminates cleanly, 100% coverage of the tensor-aligned answer nearly everywhere.

**Use B when the mixed sets matter** and $m$ is small enough to afford $O(m)$.

**Add D when** the geometry is still moving, or the last few percent of recall matters.

## 12. The measured case for going beyond whole tensors

This is the most promising untested direction, and the evidence for it is specific.

On a 5-block ResNet, Method C returns the 21-parameter tensor-aligned set $\{W_2^{(4)}, b_2^{(4)}, c_0\}$ and Method B returns a 13-parameter mixed set. They overlap in only 11 members: the mixed set contains parameters the whole-tensor rule can never admit. Solving each and evaluating:

| target | 21-parameter tensor-aligned set | 13-parameter mixed set | ratio |
|---|---|---|---|
| $\sin(4x)+0.3x$ | $1.28\times10^{-1}$ | $3.19\times10^{-3}$ | **0.025** |
| $\exp(x)$ | $9.69\times10^{-3}$ | $2.20\times10^{-4}$ | **0.023** |
| $\sin(7x)$ | $9.14\times10^{-1}$ | $3.49\times10^{-1}$ | 0.38 |
| $1/(1+25x^2)$ | $3.58\times10^{-1}$ | $2.34\times10^{-1}$ | 0.65 |

Verified on two independent eval grids per target, so it is not an eval-draw artifact. The mixed set includes $v$ alongside compatible $W_2^{(4)}$ entries, threading the per-index conflict of Section 3.

A sparsity probe that drops the least-determined members *from the tensor-aligned set* finds nothing (gain 1.00–1.01). The better set is not a subset of it, which is why that probe could not see it.

**The proposed hybrid, not yet built.** In every architecture tested, everything admitted lives in the last two or three tensors and all upstream tensors are empty. So run Method C first to locate the candidate region at 7–28 passes, then run Method B's per-parameter refinement **only inside those two or three tensors**. Cost is $O(\text{size of the last few tensors})$ rather than $O(P)$, and that is where the $40\times$ lives. The cost argument is sound; the result is unmeasured.

## 13. Why this works in any precision

The zeros are exact, not small, and that survives finite precision.

If $J_i$ does not depend on any coordinate in $L$, then $J_i(\theta+s)$ and $J_i(\theta)$ are computed by the same operations on the same inputs, because $s$ is supported on $L$ and none of those inputs changed. The two results are bitwise identical, so their difference is exactly $0$ in fp64, fp32, or bf16 alike. The contraction with $u$ in (5.4) preserves it.

Non-affine coordinates give relative motion of order 1, far above any dtype's unit roundoff. The separation is between exact zero and $O(1)$ and does not narrow as precision drops.

Contrast the second-derivative probes of Section 10, whose perturbation is a *relative* step of $10^{-3}$. In bf16, unit roundoff is $3.9\times10^{-3}$, so that perturbation rounds away, every curvature reads as zero, and the test admits the whole network.

The solve that consumes the set does not inherit this and needs compensated arithmetic to reach a low-precision floor.

## 14. GPU implementation

Analysis of the computation's shape. Only the pass counts have been measured.

**The dominant cost is sequential dependent passes**, not arithmetic and not bandwidth. Probe passes cannot overlap each other or the training step. Count sequential passes; counting FLOPs will mislead. One saving is available: the probe's backward is independent of the training backward, so both can run as a single backward with two cotangents.

**Index sets must become masks.** The reference implementation uses `nonzero`, `union1d`, `setdiff1d`, and random choice over index arrays. On a GPU these are uncoalesced gathers *and* host-side set operations. Each should be a length-$m$ mask with multiplication in place of gathering: the probe step is `s = delta * scale * z * mask`, the posterior update is masked arithmetic, and the candidate batch is a sparse random mask. This is a rewrite of bookkeeping, not of the method.

**Host-device synchronization is the hidden cost in the control logic.** Any branch on a device scalar (a tolerance comparison, a convergence check, the value-exhaustion test) forces a synchronization that stalls the pipeline. Make them masked arithmetic on device, or run them on a lagged schedule so the transfer overlaps compute.

**Elementwise work is free.** The comparison, the posterior update, the mask, and the decay are bandwidth-bound over length-$m$ arrays and fuse into one or two kernels. Adam already touches five such arrays per step.

**Reductions in the solver are the largest scaling risk in the whole program**, larger than discovery. Krylov methods need four to six inner products per iteration, each a device-wide reduction and, across devices, a collective. At hundreds of iterations per solve event that is thousands of latency-bound collectives, the standard reason Krylov underperforms on GPUs. Communication-avoiding variants cut the count by trading numerical stability, which is the wrong trade for a method whose purpose is machine precision.

**Memory.** Forming $J$ is impossible at scale ($n=10^6$, $m=10^9$ is $10^{15}$ entries), so only the sketch of Section 5 exists. Persistent state is $\theta$, the Beta parameters, and a mask: three length-$m$ arrays beyond the base optimizer, and the Beta parameters tolerate low precision. Nothing adds host-device traffic beyond ordinary training.

## 15. Status

**Measured and reproducible.** The purity-versus-recall asymmetry. The failure of every estimator-based score and of soft damping, with the exact-zero explanation. Precision at 100% for all four methods on every architecture tested. Method C at 100% coverage and ratio 1.00 across 8 geometry regimes, a transformer, a ResNet, two conv nets, and 15 depth cells, at 7–28 passes. The one-backward sketch. The candidate-batch threshold. The mixed-set advantage of up to $40\times$, on four targets and two eval grids. The death of the depth hypothesis.

**Not established.** The hybrid of Section 12 is unbuilt. Random initialization remains the weak regime for Methods A and B (57–67% recall). Batching and label noise are untested throughout. The GPU analysis is structural reasoning, not a profile. Every "ratio 1.00" is measured against the best *tensor-aligned* set, which Section 12 shows is not the best affine set. And all of this concerns discovering the set: whether an in-training solve helps or harms feature learning is a separate open question, currently confounded (see `results/checkpoint_D_optimizers/expD14_lobotomy/`).

## 16. Reproducing this

Code is in `experiments/expD15_inclusion_score/`:

| file | contents |
|---|---|
| `core15.py` | geometry and data cases, reference solver, exact resolution |
| `signals.py` | the estimator-based scores that failed (Section 10) |
| `verify.py` | the abandoned set-level verifier |
| `extended.py` | skip, fixed-basis, and depth-2 architectures |
| `archs.py` | ResNet, convolution, the exact affineness test, Method A |
| `transformer.py` | the small pre-norm transformer |
| `depth.py` | the three depth families, Method B, solve-and-evaluate |
| `pertensor.py` | Method C |
| `amortized.py` | Method D |

Build the four geometry regimes by taking a correct geometry with uniform data and perturbing exactly one thing: push the units radially away from the middle for a geometry defect, remove 96% of the training samples inside a central radius for a data defect, randomize the first layer for the fourth. The two middle cases are the discriminating pair, because any criterion measuring only linearity must score them identically.

Score against an oracle, not an eyeball. Compute the error of solving the analytically known affine block, then the discovered set at the same solver settings, and report the ratio. Report precision and recall separately, since Section 3 shows they are not interchangeable. Verify every returned set with the exact second-difference test rather than trusting it. And when data coverage is non-uniform, report the error split by region: a global average hides the effect entirely.

Two figures are worth regenerating: `figures/L_membership_map.png` (which readout weights enter $L$, by centre position, against the data density) and `figures/L_membership_arch.png` (per-tensor membership for the ResNet and transformer, showing the whole-tensor and mixed sets side by side).
