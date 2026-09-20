# expD11 — Why no $O(d\cdot k)$ iterative solver reaches the fp64 floor on QI systems

**Status:** draft — pending Sam. The Conclusions section states only what the plots show; the step-3 implications are explicitly marked as not yet signed off.

## TL;DR

- The step-2 requirement set is **not satisfiable on QI feature matrices**. Reaching the fp64 floor requires stored orthogonality of size $c \approx r$, i.e. $\Theta(d\cdot r) = \Theta(d^2)$ state — which is requirement 2's disqualified class.
- The binding constraint is **not** batching, drift, sampling, residual precision, or solver tuning. All five were eliminated on a frozen operator with a fixed buffer before the wall was located.
- The window you must store is set by **how many distinct scales the spectrum has**, not by $d$ or $\kappa$. A 2- or 4-level spectrum reaches $10^{-15}$ with $c=0$ — zero stored state. QI spectra are gapless (median $\sigma_i/\sigma_{i+1} = 1.03$ at $N{=}768$) and need the full $c=r$.
- Preconditioning cannot substitute: every $M$ tested caps accuracy at $\sim10^{-6}$ and **no window rescues it**. Loosening the truncation makes it worse, not better.

## Question

Does there exist an iterative least-squares solver that simultaneously (1) reaches machine epsilon, (2) holds $O(d\cdot k)$ state with $k$ fixed, (3) needs no $O(d^3)$ setup, (4) costs one forward-backward per step, (5) tolerates batches down to $b=d/64$, (6) front-loads progress, and (7) tolerates drift in $\Phi$? This experiment answers it for QI feature matrices by eliminating every candidate explanation for the observed $10^{-6}$ ceiling until one survives.

## Experiment design

Every measurement runs on a **frozen operator with a fixed accumulated buffer**: rows are drawn once from the pool into $A_a \in \mathbb{R}^{n_{acc}\times d}$ with $n_{acc}=4d$, and never redrawn. This deliberately removes batching, geometry drift, and resampling from the picture, so none of them can be blamed for what remains.

**The systems.** QI-1D uses the construction's own geometry: centers $x_n = -1 + nh$ for $n \in [-\text{halo}, N+\text{halo}]$ with $h = 2/N$, bandwidth $\gamma = \lambda/h$, and the augmented feature matrix $\Phi = [\tanh(\gamma(x_i - c_k))\;,\;\mathbf{1}]$, so $d = W+1$ with $W = N + 2\,\text{halo} + 1$. Target $g(x)=\sin(4x)$, evaluation on 4001 equispaced points. Halo from `default_halo(N, lambda_star)`. Numerical rank is $r = \#\{\sigma_i > 10^{-15}\sigma_1\}$; note $r \approx 0.57d$, so $r$ grows linearly in $d$.

**The solver knob.** LSQR (Paige–Saunders Golub–Kahan bidiagonalization, verified against SciPy to $5.9\times10^{-16}$ and against `lstsq` to $1.8\times10^{-15}$) with one modification: a **sliding reorthogonalization window** of $c$ vectors. At each step the new $u$ and $v$ are re-orthogonalized by two-pass Gram–Schmidt against the last $c$ stored vectors. $c=0$ is the shipped short-recurrence solver holding $O(d)$ state; $c=\infty$ is full reorthogonalization holding $O(d\cdot r)$. $c$ is therefore a direct dial on stored state, and it is the *only* thing varied in D1 and D2.

**The preconditioners.** Block-Jacobi $M^{-1/2}$ built from each block's own SVD (never the Gram), state $d\cdot k$, applied as $V\,\mathrm{diag}\,V^{\!\top}$; tested at $k \in \{64, 256\}$ and block-eigenvalue truncation $\in \{10^{-13}, 10^{-16}, 0\}$. Also tested: rank-revealing pivoted-QR blocks with $R^{-1}$ applied by backward-stable triangular solve, and the QI **cardinal-coefficient** operator $C$ (banded Toeplitz, $C_{nm}=c_{n-m}$, from the target-independent Toeplitz solve, state $O(K_c)$ independent of $d$).

**Elimination sequence.** Before the window sweep, four candidate causes were ruled out on the same frozen buffer:
- *The buffer*: truncated-SVD on the identical $4d$ accumulated rows reaches $2.4\times10^{-14}$ (QI-1D $N{=}128$) and $3.1\times10^{-15}$ (twin-unstruct $d{=}512$), against $1.5\times10^{-7}$ for the solver. The buffer supports the floor; the solver leaves seven orders on the table.
- *Residual cancellation*: refinement rounds were rerun with an algebraically identical but cancellation-free residual $r = A_a(a_{ex}-w) + r_{ex}$ in place of $y_a - A_a w$. Result unchanged ($7.4\times10^{-8}$ vs $7.3\times10^{-8}$). Extra-precision residuals and mixed-precision iterative refinement therefore cannot help.
- *Refinement*: rounds plateau after round 2 and never move again, over 8 rounds.
- *Conditioning by rescaling*: column scaling moves $\kappa$ from $8.9\times10^{14}$ to $1.9\times10^{13}$ in 1-D and does nothing in 2-D ($9.0\times10^{14} \to 7.6\times10^{14}$).

**Metrics.** All accuracies are eval relative $L_2 = \|\Phi_{ev}w - y_{ev}\|/\|y_{ev}\|$ on the held-out grid, never a training residual. Iteration budget is $2r$ throughout, so no curve is iteration-starved.

**Code & data.** Solver and window: `experiments/expD11_batching/window_law.py`. Panels, data and figures: `experiments/expD11_batching/disproof.py` → `results/checkpoint_D_optimizers/expD11_batching/disproof.jsonl` and `figures/D1_two_regimes.png`, `D2_spectral_law.png`, `D3_scaling_verdict.png`. Problem builders and reference floors: `experiments/expD11_batching/core11.py`. Frozen-episode battery (batch invariance): `experiments/expD11_batching/run_frozen.py` → `frozen.jsonl`.

## Results

**The preconditioner caps accuracy, and orthogonality is what actually buys the floor.** This is the reverse of what expD10 round 4 recorded. With block-Jacobi in place, accuracy sits at $\sim10^{-6}$ and is completely insensitive to $c$ — even at full reorthogonalization. Without it, accuracy falls monotonically with $c$ and reaches $4\times10^{-15}$. Loosening the block truncation from $10^{-13}$ to $10^{-16}$ degrades the result to $5\times10^{-2}$, so there is no truncation setting that trades cap for stability. Pivoted-QR blocks with a backward-stable triangular solve barely precondition at all ($\kappa(AM)=4.6\times10^{13}$ against $\kappa(A)=7.3\times10^{14}$) and diverge. The cardinal-coefficient operator does not whiten $\Phi$ either: $\kappa(\Phi C) \approx \kappa(\Phi)$.

**The required window is a property of the spectrum.** Holding $d=400$, $r=240$, $\kappa=10^{14}$ and the target energy profile fixed, and varying *only* the number of distinct singular-value scales, $c^\ast$ tracks that number: 2- and 4-level spectra reach $10^{-15}$ at $c=0$; a gapless exponential spectrum needs $c=r$.

**No fixed window survives width scaling, at any $\lambda$.** At $\lambda=0.05$ with $c=32$: $8.9\times10^{-15} \to 9.4\times10^{-11} \to 4.3\times10^{-9} \to 4.2\times10^{-8}$ as $d$ goes $798\to1470$. The same monotone degradation holds at $\lambda=0.10$ and $0.25$ and at every $c \in \{0,16,32,64\}$. Meanwhile $c=r$ holds $\sim10^{-14}$ throughout. Smaller $\lambda$ buys a gappier spectrum and a smaller rank — and so genuinely delays the onset — but does not change the asymptotics: the median consecutive singular-value ratio falls to $1.03$ at every $\lambda$ tested.

**What does still work, at reduced accuracy.** The frozen-episode design is batch-invariant to the limit tested: twin-unstruct $d{=}2048$ gives $8.8/9.2/9.3/9.2/9.3 \times 10^{-6}$ across $b$ from $8d$ down to $d/64$, a $512\times$ span. Under noise it sits on the $0.27\sigma$ statistical floor. On the real QI problems its accuracy is $\sim10^{-7}$ and, contrary to what was reported before the width sweep, does **not** degrade with $N$ (QI-1D: $4.1\times10^{-8}$ at $N{=}64$, $5.3\times10^{-8}$ at $N{=}1024$). The earlier "degrades as $d^{2.5}$" claim came from the synthetic twins only.

### Figures

- **D1 — `D1_two_regimes.png`.** Four panels, QI-1D at $N \in \{128,256,384,512\}$, shared log $y$ (eval rel $L_2$, $10^{-16}$ to $10^{-3}$); $x$ is the window $c$ in vectors stored. Blue circles = no preconditioner, red squares = block-Jacobi $k{=}64$; the dashed vertical line marks $c=r$ in each panel. Look for: blue descends to $10^{-15}$ but only as it approaches the dashed line, and the dashed line moves right with $N$; red is flat across the entire $x$ range in all four panels.
- **D2 — `D2_spectral_law.png`.** Single axes, log $y$, $x$ = window $c$. Five curves for five synthetic spectra with identical $d$, $r$, $\kappa$ and target energy, differing only in the number of distinct scales (2, 4, 16, 64, gapless). Dashed vertical line at $c=r=240$. Look for: the blue and orange curves are flat along the bottom from $c=0$, and the knee of each curve moves right as the level count rises, with the gapless curve reaching the floor only at the dashed line.
- **D3 — `D3_scaling_verdict.png`.** Three left panels, one per $\lambda \in \{0.05, 0.10, 0.25\}$: log $y$ = eval rel $L_2$, linear $x$ = $d$ with one tick per measured width. Four coloured lines are fixed windows $c \in \{0,16,32,64\}$; the black dashed triangles are $c=r$, whose state grows with $d$. Right panel: median $\sigma_i/\sigma_{i+1}$ against $d$, one line per $\lambda$, with a dotted reference at $1.0$. Look for: every coloured line rises left-to-right in all three panels while the black line stays flat at $10^{-14}$; and in the right panel all three $\lambda$ curves converge onto the dotted line.

## Additional details

**Why the preconditioner route is blocked, stated conservatively.** For any right preconditioner, $\kappa(\Phi M) \ge \kappa(\Phi)/\kappa(M)$, so driving $\kappa(\Phi M)$ to $O(1)$ forces $\kappa(M) \gtrsim 10^{14}$. That inequality is exact and assumption-free, and it is why the structured cardinal-coefficient operator — which is well-conditioned — cannot help. It does **not** by itself prove that a high-$\kappa$ $M$ must lose accuracy: a bound of $u\cdot\kappa(M)$ predicts a $1.9\times10^{-3}$ cap where $2.1\times10^{-6}$ is measured, so that mechanism is *not* verified and is not claimed here. The empirical statement is the defensible one: across explicit-SVD block-Jacobi at three truncation levels and two block sizes, pivoted-QR with triangular solve, and the structured Toeplitz operator, every $M$ capped and no window rescued any of them.

**Scope of the negative result.** The wall is specific to feature matrices whose spectrum is gapless down to the working-precision floor, and that condition is measurable on any problem from a cheap sketch. D2 shows directly that a clustered spectrum is solvable at $c=0$. The result is therefore not "iterative solvers cannot reach machine epsilon"; it is "they cannot do so at $O(d\cdot k)$ *on this spectrum*."

**Corrections to earlier records.** expD10 round 4 recorded "orthogonality is NOT the barrier — full reorth buys zero accuracy"; that measurement was made with the preconditioner active, which is exactly the regime where the window does nothing (D1, red curves). Unpreconditioned, full reorth buys nine orders. `batching_test.md` trick T1 ("steering preconditioners degrade gracefully under staleness") was separately measured false.

## Conclusions

On QI feature matrices the seven step-2 requirements cannot be met simultaneously: reaching the fp64 floor requires $c \approx r$ stored vectors, and $r$ grows linearly with $d$, so the state is $\Theta(d^2)$ — while every $O(d\cdot k)$ configuration tested caps between $10^{-6}$ and $10^{-7}$. The mechanism is the spectrum, not the solver: $c^\ast$ is set by the number of distinct spectral scales, and QI spectra are gapless at every bandwidth tested.

*Pending Sam's sign-off:* whether this redirects step 3 toward accepting a $10^{-7}$ $O(dk)$ solver, or toward changing the parameterization so that $\Phi$'s spectrum is clustered rather than gapless.

## Open questions

- Does any admissible geometry or reparameterization produce a clustered $\Phi$ spectrum at fixed $N$? D2 says such a system would solve to the floor with zero stored state, which makes this the highest-value follow-up.
- Is $\lambda$-driven delay useful in practice? Small $\lambda$ raises $W$ for the same accuracy but keeps a fixed $c$ viable to larger $N$; the exchange rate has not been measured.
- Does the same window law hold in 2-D? D1–D3 are 1-D; the 2-D spectrum was measured gapless (median ratio $1.08$) but the window sweep was not run there.
