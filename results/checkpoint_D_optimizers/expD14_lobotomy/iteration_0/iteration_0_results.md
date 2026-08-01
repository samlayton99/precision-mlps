# expD14 iteration 0 -- the lobotomy: Adam and a least-squares solve sharing one step budget

**Status: draft, pending Sam. All numbers final (T1-T4). Conclusions unsigned.**

## TL;DR

- The assembled optimizer works. From QI init, with the linear block *discovered* rather than given, **all 18 cells (6 targets $\times$ 3 widths) land at or below the frozen-geometry floor and 10 of them sit at $\sim10^{-16}$** -- every digit fp64 has, two orders below iteration 11 and 8-13 orders below stock Adam. Solving *every* step rather than every 200 turns the exact solve into iterative refinement, which is where those two orders come from.
- **Two of my three control mechanisms are wrong, and the ablations say so cleanly.** The trust-region clip on the $L$ step costs 5-8 orders; drift-matched damping costs 4 orders. Both were protecting against a staleness that does not exist when you re-solve every step.
- **One mechanism is load-bearing and new**: replacing Adam's scale-free step length on the $A$ block with the exact-line-search length along Adam's own direction. It costs one JVP, it is what stops the post-solve sawtooth, and without it three of four cells end the run *diverging* (final error $3\times10^{2}$ on one).
- **The honest failure is feature learning.** From a random init, solving $L$ exactly in-training suppresses geometry learning so badly that plain Adam followed by one terminal solve beats it on 3 of 4 targets. Deliberately under-solving ($\alpha=\max(\text{drift},\sqrt{1-\rho})$) recovers it on `runge` -- a geometry $16000\times$ better than the one it started with, and $100\times$ better than Adam's -- but not on `sine_8pi`.

## Question

Step 3 of the program: lobotomize Adam and stitch the step-2 least-squares solver in, so every iteration updates every parameter -- some by modified Adam, some by a least-squares update. Sam's framing for this first attempt: one fixed step *energy* per iteration, split between the two; the damped Gauss-Newton step multiplied by $\mu$ to put it back in raw gradient coordinates and then scaled to the size the allocation says; low $\mu$ reachable fast when linearity is obvious (QI init) and a long smoothing phase when it is not (random init). Does that assemble into something that beats Adam everywhere and hits the floor on QI init?

## Experiment design

**The model and the split.** One hidden layer, $f(x;\theta)=\sum_k v_k\tanh(w_kx+b_k)+c_0$, $\theta=(w\,|\,b\,|\,v\,|\,c_0)\in\mathbb R^m$, $m=3W+1$; $n=2003$ equispaced train points, eval rel $L_2$ on a disjoint 4001-point grid. A certificate splits $\theta$ into $L$ (the block the model is linear in) and $A$ (everything else). $L$ is either **given** (the oracle $L=(v,c_0)$, used in T1-T3 so the control logic is what is under test) or **discovered** by iteration 11's certificate -- the second-difference nonlinearity probe $q_i=s_i^2\|\partial^2 f/\partial\theta_i^2\|$ evaluated at $\theta$ and at one perturbed point, normalized to $\hat q$, admitted at $\hat q<10^{-8}$ and only when Adam's second moment is nonzero (T4).

**The step.** Per iteration, one fused forward-backward gives the fresh residual $r$ and gradient $g$; Adam's moments are updated on **all** $m$ coordinates whether or not that coordinate spends any energy, so a parameter crossing $A\leftrightarrow L$ never meets a cold optimizer. Then

$$d_\mu=\arg\min_d\ \|J_L d+r\|^2+\mu\|d\|^2,\qquad \mu=(\alpha\sigma_1(J_L))^2,$$

$$\Delta_A=(1-\rho)\,\ell_A\,p_A,\qquad \Delta_L=d_\mu ,$$

with $p_A$ the unit Adam direction restricted to $A$. $\alpha=\sqrt\mu/\sigma_1$ is exactly the inverse effective condition number, which is what makes the damping transferable across matrices (expD12 phase 0). $\sigma_1$ comes from three power iterations on $J_L^\top J_L$, refreshed on the certificate cadence, carrying one $O(d)$ vector.

**The two allocations, both measured rather than scheduled.**

- $\rho$, the energy share, from predicted decrease. For a unit direction $p$ the exact-line-search decrease of $\|r\|^2$ is $(r\cdot Jp)^2/\|Jp\|^2$ -- one JVP each for $p_A$ and $p_L$. Allocate $\rho=g_L/(g_A+g_L)$. No constant.
- $\ell_A$, the $A$ step length, either Adam's own $\|\Delta^{\rm Adam}_A\|$ or $\min(\|\Delta^{\rm Adam}_A\|,t^\star_A)$ with $t^\star_A=-(r\cdot Jp_A)/\|Jp_A\|^2$ the exact-line-search length, which falls out of the JVP already computed for $\rho$. $t^\star_A$ is **linear in the residual**, so unlike Adam's scale-free step it vanishes as the problem is solved -- the escape iteration 11 §9.3 asked for.

**The $\mu$ rules compared.** `fixed` ($\alpha$ pinned); `drift`, $\alpha=\mathrm{EMA}(\|J\Delta_A\|/\|y\|)$ bias-corrected -- "never resolve below the scale my own features are moving at", which is expD13's matching rule; and `drift_gain`, $\alpha=\max(\text{drift},\sqrt{1-\rho})$ -- since gains are quadratic in the residual and $g_A/(g_A+g_L)=1-\rho$, its square root is the residual level $A$ still owns, so the rule reads *never resolve $L$ below the part of the error $A$ can still remove*. Also no constant.

**The solvers.** `direct` is a damped truncated-SVD solve on a materialised $J_L$ (reference only; $J_L$ comes from a one-pass analytic Jacobian verified against per-column JVPs to $1\times10^{-17}$, because a per-column JVP loop costs $|L|$ passes *per step*). `lsqr` is the shipped path: matrix-free damped LSQR, $J_Lz$ by one JVP and $J_L^\top u$ by one VJP, $\tau$ iterations per step restarted from scratch on a fresh residual, Fong-Saunders stopping, $O(d)$ state. **No block preconditioner anywhere** -- scalar (Adam-diagonal) only, per Sam.

**Metrics.** Every run reports the error it *reached*, and separately `floor_final` -- the truncated-SVD solve on the geometry it finished with. That second number is exactly the error one terminal exact solve would give, so every arm is scored twice: as-is, and with a finisher. The pair separates "did the readout get solved" from "did the geometry get learned", which is the only way to see the failure below.

**Grid.** T1 piecewise checks at $N{=}128$. T2 ablation: 10 arms $\times$ 2 inits $\times$ \{`sine`,`runge`\}, 2500 steps. T2b damping rules: 4 arms $\times$ 2 inits $\times$ 4 targets, 4000 steps. T3 solver: `direct` vs `lsqr` at $\tau\in\{1,3,10,30,100\}$ and three fixed $\alpha$. T4: certificate-discovered $L$, 6 targets $\times$ $N\in\{64,128,256\}$ $\times$ 2 inits $\times$ 3 arms, 3000 steps. `lr` $=10^{-3}$, fp64, one seed.

**Code & data.** Optimizer and case builder: `experiments/expD14_lobotomy/iteration_0/core0.py`. Tests: `t1_pieces.py`, `t2_assembly.py`, `t2b_alpha.py`, `t3_solver.py`, `t4_grid.py`. Figures: `figs0.py`. Data: `results/checkpoint_D_optimizers/expD14_lobotomy/iteration_0/{t2_assembly,t2b_alpha,t3_solver,t4_grid}.jsonl`; figures in `figures/`.

## Results

**The pieces (T1).** $\mu d_\mu$ agrees with the raw gradient $J_L^\top(-r)$ to a cosine of $1-8\times10^{-11}$ and a length ratio of $0.99998$ at $\alpha=10^2$, and $d_\mu$ walks continuously to the exact solve as $\alpha\to0$ -- Sam's "multiply by $\mu$, then scale" is exactly the geometry of the step. The useful law is the error a single damped solve lands on: **eval rel $L_2\approx0.3\,\alpha$, holding over ten decades** ($1.5\times10^{-2}$ at $\alpha=10^{-2}$ down to $9.5\times10^{-15}$ at $\alpha=10^{-16}$, against this geometry's floor of $2.3\times10^{-14}$). So $\alpha$ is not an abstract dial: it *is* the accuracy you are choosing to bank. The matrix-free LSQR reproduces the direct solve to $10^{-13}$ where it converges, needing 8 iterations at $\alpha=1$ and 70 at $\alpha=10^{-2}$, and **fails to converge at all at $\alpha\le10^{-4}$ within 400 iterations** -- expD11's wall, present and correct. $\sigma_1$ from three power iterations is within $0.25\%$ of the SVD; the gain estimator matches the actual line-search decrease to $1.00000$ on the $L$ direction and $0.989$ on a random one; and with $L$ empty the optimizer reproduces stock Adam to $8\times10^{-17}$ relative in $\theta$.

**The ablation (T2, $N{=}128$, 2500 steps, oracle $L$, direct solve).** Best eval rel $L_2$:

| arm | qi/sine | qi/runge | rand/sine | rand/runge |
|---|---|---|---|---|
| stock Adam | $8.6\times10^{-4}$ | $2.3\times10^{-4}$ | $1.6\times10^{-2}$ | $1.0\times10^{-1}$ |
| **undamped, $\rho$ allocated** | $\mathbf{2.0\times10^{-16}}$ | $\mathbf{1.2\times10^{-16}}$ | $\mathbf{5.7\times10^{-11}}$ | $\mathbf{5.8\times10^{-3}}$ |
| drift-matched $\mu$ (my proposal) | $1.1\times10^{-12}$ | $2.5\times10^{-14}$ | $6.4\times10^{-11}$ | $9.5\times10^{-3}$ |
| trust-region clip on | $9.3\times10^{-7}$ | $4.7\times10^{-12}$ | $6.2\times10^{-1}$ | $2.4\times10^{-1}$ |
| Adam's own step length on $A$ | $7.7\times10^{-12}$ | $7.0\times10^{-13}$ | $1.1\times10^{-10}$ | $2.8\times10^{-2}$ |
| $\alpha$ pinned at 1 | $6.8\times10^{-3}$ | $2.2\times10^{-3}$ | $1.8\times10^{-1}$ | $1.7\times10^{-1}$ |
| $\rho=0$ (no allocation) | $2.7\times10^{-12}$ | $2.4\times10^{-13}$ | $1.6\times10^{-1}$ | $2.1\times10^{-1}$ |

Reading it: the clip and the damping are both **costs, not protections**, and for the same reason -- they exist to stop you banking accuracy you cannot keep, and when the solve is repeated every step there is nothing to keep, you just re-solve. `resetmom` (zeroing Adam's moments on $L$ after a solve) changed nothing to three digits; `adamL` (Adam also stepping $L$, iteration 11's rule) was neutral to mildly positive ($1.6\times10^{-14}$ vs $2.5\times10^{-14}$ on qi/runge). The `adamstep` row hides its real failure in a "best" column: its *final* errors are $2.2\times10^{-6}$, $7.5\times10^{-10}$, $1.26$ and $3.2\times10^{2}$ -- it oscillates and then diverges. That is the $\|v\|\eta$ re-injection, and the line search is what removes it.

**The tension (T2b, 4000 steps).** Scoring every arm twice exposes what the error column alone hides. From a random init, on `runge`: Adam reaches $4.2\times10^{-2}$ but leaves a geometry worth $5.7\times10^{-5}$; the undamped lobotomy reaches $5.9\times10^{-3}$ and leaves a geometry worth $9.3\times10^{-3}$ -- i.e. **it never improved the geometry at all**. On `sine_8pi` the gap is starker: Adam's final geometry is worth $4.3\times10^{-5}$, four orders better than the $0.43$ it started from, while every lobotomy arm leaves it at $0.43$. Solving $L$ exactly makes the residual orthogonal to the features, and the geometry gradient that remains is too weak to learn from -- iteration 11 §9.4, now self-reinforcing, because a weak signal shortens the line-searched $A$ step, which quiets the drift, which deepens the solve.

`drift_gain` is the one arm that breaks the loop, and on `runge` it works: it leaves a geometry worth $5.8\times10^{-7}$, $16000\times$ better than the one it started with and $100\times$ better than Adam's. It does not work on `sine_8pi`. And because it deliberately under-solves, its *reached* error is poor ($7\times10^{-2}$) -- it is only a win if a finisher cashes the geometry in.

**On QI init the lobotomy also beats Adam-plus-a-finisher**, which is worth stating separately because that baseline is expD02's winner. Adam damages the QI geometry: `sine`'s floor degrades from $2.3\times10^{-14}$ to $1.4\times10^{-11}$ over 4000 steps, `sine_8pi`'s from $5.5\times10^{-14}$ to $5.6\times10^{-9}$. The undamped lobotomy holds the geometry at its initial floor and reaches $10^{-16}$.

**The $O(d)$ solver (T3) fails at the job it was given and turns out to be good at the other one.** Swapping the reference solve for the matrix-free damped LSQR, everything else held:

| arm | passes/step | qi/sine reached | geometry left | rand/runge reached | geometry left |
|---|---|---|---|---|---|
| direct (reference) | 274 | $2.0\times10^{-16}$ | $2.3\times10^{-14}$ | $5.9\times10^{-3}$ | $9.4\times10^{-3}$ |
| lsqr $\tau{=}1$ | 7 | $3.2\times10^{-3}$ | $1.2\times10^{-9}$ | $1.1\times10^{-1}$ | $\mathbf{4.4\times10^{-6}}$ |
| lsqr $\tau{=}3$ | 11 | $3.6\times10^{-4}$ | $1.5\times10^{-11}$ | $7.1\times10^{-2}$ | $2.6\times10^{-4}$ |
| lsqr $\tau{=}30$ | 65 | $8.2\times10^{-6}$ | $9.8\times10^{-13}$ | $5.2\times10^{-2}$ | $9.3\times10^{-3}$ |

As an error-reaching solver it is 10 orders short of the reference on QI init, exactly as T1/P2 predicted -- carrying the warm start in $\theta$ across steps does *not* rescue a short-recurrence Krylov solve on a gapless spectrum, so expD11's wall stands. But read the "geometry left" columns: **the shallower the solve, the better the geometry it leaves**, monotonically, and $\tau{=}1$ on `rand/runge` leaves a geometry worth $4.4\times10^{-6}$ -- $2000\times$ better than the reference solve's, at $1/40$ of the reference's passes -- and $13\times$ better than the geometry stock Adam leaves (T2b, $5.7\times10^{-5}$, at a longer 4000-step budget, so that particular comparison is generous to Adam). The same holds on `rand/sine` ($1.3\times10^{-12}$, $56\times$ better than the reference). Sweeping $\alpha$ at fixed $\tau$ changes almost nothing ($8.9$, $8.7$, $8.7\times10^{-6}$ across four decades), which confirms it is the iteration budget, not the damping, doing the under-solving here.

So the two under-solving mechanisms -- damping and a truncated iterative solver -- produce the same effect, and the cheap one produces it for free. The direction of the rule is consistent across every measurement: **when the geometry is already right, solve as hard as you can; when it still has to be learned, solve softly and cash in at the end.**

**The grid with $L$ discovered rather than given (T4).** The certificate admits the readout plus a tail of saturated geometry coordinates ($B=583$ against a 462-coordinate readout at $N{=}256$ -- iteration 11's 121 extra, to the unit), and the observability filter drops every one of them at solve time, so the block that does arithmetic is the readout. Nothing here depended on being told that.

*QI init, error reached (eval rel $L_2$), all 18 cells:*

| target | $N{=}64$ | $128$ | $256$ | | floor $N{=}64$ | $128$ | $256$ |
|---|---|---|---|---|---|---|---|
| `sine` | $2.4\times10^{-16}$ | $2.0\times10^{-16}$ | $2.1\times10^{-16}$ | | $5.7\times10^{-15}$ | $2.3\times10^{-14}$ | $1.6\times10^{-14}$ |
| `sine_8pi` | $1.3\times10^{-13}$ | $1.4\times10^{-15}$ | $8.7\times10^{-16}$ | | $1.9\times10^{-13}$ | $5.5\times10^{-14}$ | $3.8\times10^{-14}$ |
| `runge` | $1.3\times10^{-9}$ | $1.2\times10^{-16}$ | $1.1\times10^{-16}$ | | $3.3\times10^{-9}$ | $2.8\times10^{-14}$ | $1.7\times10^{-14}$ |
| `sine_mixture` | $4.7\times10^{-11}$ | $8.7\times10^{-15}$ | $4.6\times10^{-16}$ | | $4.7\times10^{-11}$ | $6.8\times10^{-14}$ | $2.8\times10^{-14}$ |
| `exp` | $1.0\times10^{-16}$ | $1.0\times10^{-16}$ | $1.0\times10^{-16}$ | | $2.8\times10^{-14}$ | $8.3\times10^{-15}$ | $1.1\times10^{-14}$ |
| `abs_cubed` | $4.6\times10^{-7}$ | $4.5\times10^{-8}$ | $4.2\times10^{-9}$ | | $5.3\times10^{-7}$ | $4.7\times10^{-8}$ | $4.2\times10^{-9}$ |

**All 18 cells land at or below the frozen-geometry floor, and 10 of them sit at $\sim10^{-16}$** -- the rounding floor of evaluating the model on the held-out grid, i.e. every digit fp64 has. Stock Adam on the same cells ranges from $4\times10^{-5}$ to $2\times10^{-1}$, so the margin is 8 to 13 orders. `abs_cubed` is the control: it is only twice differentiable and its own floor is $10^{-7}$ to $10^{-9}$, and the optimizer sits exactly on it at every width rather than overfitting past it.

*Random init, after one terminal exact solve* (the only fair column, since as-is every arm is pinned at its own geometry's floor):

| target | Adam + finisher | lobo + finisher | lobo_soft + finisher |
|---|---|---|---|
| `sine` ($N{=}256$) | $4.6\times10^{-12}$ | $5.6\times10^{-11}$ | $\mathbf{9.0\times10^{-13}}$ |
| `sine_8pi` ($N{=}256$) | $2.5\times10^{-4}$ | $4.2\times10^{-1}$ | $\mathbf{1.0\times10^{-5}}$ |
| `runge` ($N{=}128$) | $8.9\times10^{-5}$ | $9.4\times10^{-3}$ | $\mathbf{2.8\times10^{-6}}$ |
| `sine_mixture` ($N{=}256$) | $\mathbf{5.3\times10^{-3}}$ | $2.1\times10^{-1}$ | $2.2\times10^{-2}$ |
| `exp` ($N{=}256$) | $1.0\times10^{-13}$ | $1.9\times10^{-13}$ | $1.9\times10^{-13}$ |
| `abs_cubed` ($N{=}256$) | $2.8\times10^{-5}$ | $7.7\times10^{-5}$ | $\mathbf{1.1\times10^{-5}}$ |

The hard-solving arm (`lobo`) never improves the geometry -- its finisher column is its `floor0` column to two digits, in all 18 cells. The under-solving arm beats Adam-plus-finisher on four of six targets, by $13\times$ to $32\times$. **But it is not reliable**: `sine_8pi` at $N{=}128$ returns $4.3\times10^{-1}$ (a total failure) between $1.8\times10^{-4}$ at $N{=}64$ and $1.0\times10^{-5}$ at $N{=}256$. A rule that swings six orders on a width change is a rule that is not yet right.

### Figures

- **`T1_mu_interpolation.png`** -- two panels, $x$ is $\alpha$ on a log axis *reversed* (large damping on the left, so the eye reads left-to-right as "solving harder"). Left: cosine similarity of $\mu d_\mu$ to the raw gradient (circles) and of $d_\mu$ to the exact solve (squares). Right: the same two quantities as relative lengths. Look for the circles pinned at 1 on the left edge -- that is the claim that $\mu d_\mu$ *is* the gradient at heavy damping -- and the squares rising to 1 on the right.
- **`T2_ablation.png`** -- 2$\times$2 over (init $\times$ target), eval rel $L_2$ against iteration, one colour per ablation arm, shared log $y$ fixed to $[10^{-16},3]$ so panels are comparable; the dashed line is the floor of the initial geometry. Look for: purple and orange (undamped, $\rho{=}1$) flat along $10^{-16}$ in both QI panels; green (`adamstep`) oscillating over six decades and rising; grey (Adam) barely moving; and in the bottom-right panel every line collapsing onto the dashed floor, which is the geometry never improving.
- **`T2_signals.png`** -- three stacked rows, two columns (QI and random init, `sine`): the error, then $\alpha$ as the drift gauge asks for it with the error overlaid dotted for scale, then $\rho$. Look for: $\alpha$ tracking the error curve about two decades below it (the gauge is measuring the right thing), and $\rho$ chattering across the full $[0,1]$ range once converged -- both gains are then at rounding level, so the allocator is reading noise. Harmless here, but it is why the rule needs a floor.
- **`T2b_tension.png`** -- four panels: rows are init, columns are "reached during the run" and "after one terminal exact solve", grouped bars per target, dashed line the initial geometry's floor. This is the figure that carries the negative result: compare the two columns in the bottom row and watch Adam (grey) go from worst to best on `sine_8pi` and `runge`, while the solving arms barely move because their bars were already at their geometry's floor.
- **`T3_solver.png`** -- 2$\times$2 over (init $\times$ target), the reference direct solve dashed black against the matrix-free LSQR at five values of $\tau$, error against iteration. Look for the clean monotone ladder in $\tau$ and the ten-order gap to the black line: this is the price of the $O(d)$ path, and expD11's wall is why it is that large.
- **`T3b_undersolving.png`** -- the other half of T3, and the one that matters: $x$ is the per-step solver budget $\tau$ (the star at the right edge is the exact solve), $y$ is what the geometry each arm finished with is *worth*, dashed lines the geometry each started from. Look for the two right-hand curves rising as $\tau$ grows -- solving harder leaves a worse geometry -- and for `runge`'s $\tau{=}1$ point sitting three orders below its own dashed line while the exact solve sits on it.
- **`T4_grid_{qi,rand}.png`** -- one panel per target, eval rel $L_2$ against width, solid lines the error reached and dotted the same arm after a finisher, with the certificate discovering $L$ rather than being told it.

## Additional details

**Why every-step solving is different in kind, not degree.** Iteration 11 solved every 200 steps and sawtoothed: each solve reached the floor, the following 200 Adam steps walked it back up. Solving every step removes the walk-up *and* makes the repeated truncated solve an iterative refinement, which converges past the accuracy of any single truncated solve -- that is where $2\times10^{-16}$ comes from against a single-solve floor of $2.3\times10^{-14}$. The cost is real but bounded: the shipped path is 1 fused forward-backward, $2\tau$ passes for the solver, and 2 passes for the allocation.

**What the memory story actually is.** Persistent state is $\theta$, Adam's two moments, the membership mask and one power-iteration vector -- all $O(d)$, Adam's class. The reference `direct` arm materialises $J_L$ and is a measurement instrument, not a candidate; every claim about the deployable form has to come from the `lsqr` rows (T3), and T1/P2 already says those rows cannot reach $\alpha\le10^{-4}$ from cold. Whether the warm start carried in $\theta$ rescues that across steps is exactly what T3 measures, and it is the single most important number for whether this is practical.

**Confounds worth naming.** One seed throughout. `floor_final` is a truncated-SVD solve at `rcond` $10^{-15}$, so it is a *lower bound proxy* for what a finisher would deliver, not a measured finisher run. The `direct` arm's analytic Jacobian is architecture-specific; it is verified equal to the blind JVP construction to $10^{-17}$, but it means the T2/T2b/T4 timings are not the timings of a blind implementation. And "random init" here is PyTorch's stock `nn.Linear` initialisation, which for a $1\to W$ layer gives $w\sim U(-1,1)$ -- a bandwidth regime far from the QI one, so it is a hard case by construction rather than a representative one.

## Conclusions

*Unsigned, pending Sam.* What the data supports so far:

1. The lobotomy assembles and reduces exactly to stock Adam when the linear block is empty. On a good geometry it holds or beats the frozen-geometry floor in all 18 cells and reaches $\sim10^{-16}$ in ten of them, two orders below iteration 11, because solving every step is iterative refinement rather than a periodic reset.
2. Of the three control mechanisms proposed, one survives. The exact-line-search length on the $A$ block is load-bearing and cheap. The trust-region clip and drift-matched damping are both net costs at this cadence; $\mu$'s remaining job is solver tractability, not staleness insurance.
3. The binding obstacle is no longer precision, it is feature learning. The hard-solving arm never improves a random geometry -- in all 18 random-init cells its post-finisher error equals the floor it started from. Under-solving restores it and beats Adam-plus-finisher on four of six targets by $13$-$32\times$, but swings six orders across widths on `sine_8pi`, so the mechanism is real and the rule is not.
4. How hard to solve should be governed by how much the geometry still has to learn. Two independent knobs -- the damping $\alpha$ and the iteration budget $\tau$ -- produce the same trade in the same direction, and the cheap $O(d)$ solver produces it for free.

## Open questions

- The $O(d)$ solver cannot reach the floor even warm-started through $\theta$ (T3), so the deep solve still needs a finisher. Is the right architecture explicitly two-tier -- cheap under-solving throughout, one $O(dr)$ finisher when the drift gauge quiets -- which is expD10's tier-2/tier-3 split arriving from the other direction?
- The greedy allocation is myopic: $L$'s one-step decrease essentially always beats $A$'s, so $\rho\to1$ and the geometry freezes. $A$'s value is the *floor* it will reach, not the decrease it makes this step. Is there an $O(d)$ observable for that?
- `lobo_soft` swings from $1.8\times10^{-4}$ to $4.3\times10^{-1}$ to $1.0\times10^{-5}$ across $N\in\{64,128,256\}$ on `sine_8pi`. What makes that cell bistable -- is $\sqrt{1-\rho}$ latching, or is it the target?
- $\rho$ chatters between 0 and 1 once both gains are at rounding level. Floor it, or drive it from a smoothed observable.
