# The hardened $\lambda$ rule: what survives, what was withdrawn, and what to do

**Status:** consolidated 2026-09-08 from expC08, expC09 and expC10 after an independent adversarial review (`SKEPTIC_REVIEW_2026-09-08.md`, same folder). Prior work: see `related_work.md` beside this file; the closest precedent is Adcock, Huybrechs and Piret (SIAM J. Numer. Anal. 2024), which proves the linear scaling law, the halo, the oversampled truncated-SVD least squares and an explicit constant for the Gaussian RBF, and whose frame-theoretic error bound is the rigorous form of the two-wall idea. This file is the only statement of the $\lambda$ theory and its evidence that should be cited. The original writeups are in `archive/writeups/` with a SUPERSEDED header listing what each one got wrong. The theorem itself is in `docs/lambda_rule_theory.md`, which the review checked and did not break.

## 1. The question

A frozen uniform-grid network on $[-1,1]$ has centers $c_k=-1+kh$, $h=2/N$, a halo of extra centers beyond each end, one shared inner weight $\gamma$, and neurons $\psi(\gamma(x-c_k))$. Its readout is solved by least squares. The dimensionless bandwidth $\lambda=\gamma h$ is the one continuous knob; the error-vs-$\lambda$ curve is a floor between a right wall (aliasing) and a left wall (finite precision). The question is how to choose $\lambda$ before seeing the target, from the activation, the width $N$, and the working precision, with at most an order-of-magnitude idea of the target's highest frequency.

## 2. What is proven and independently verified

### 2.1 The fiber theorem (the result)

In grid units $u=x/h$ the network spans integer translates of one function, so its Fourier transform is $\hat g(\theta)=\hat a(\theta)\,w_r(\theta)$ with $\hat a$ the $2\pi$-periodic transform of the readout weights and

$$w_r(\theta)=\frac{\lambda^{r-1}\,\widehat K(\theta/\lambda)}{(i\theta)^r},$$

where $K=\psi^{(r)}$ is the activation's kernel ($r=1$, $K=\mathrm{sech}^2$ for tanh; $r=2$ for gelu and swish) and $\widehat K$ its transform normalized to $\widehat K(0)=1$. Periodicity of $\hat a$ ties the output at $\theta$ to ghost outputs at every $\theta+2\pi m$ with the fixed ratio

$$t_m(\theta)=\frac{\widehat K((\theta+2\pi m)/\lambda)}{\widehat K(\theta/\lambda)}\Big(\frac{\theta}{\theta+2\pi m}\Big)^{r},$$

the kernel ratio times the attenuation from integrating $r$ times. Least squares picks the signal amplitude that minimizes $|1-y|^2+S|y|^2$, $S=\sum_{m\ne0}t_m^2$, so for a tone at grid frequency $\theta$ the best possible relative $L^2$ error is exactly

$$E_{\text{alias}}(\theta,\lambda)=\sqrt{\frac{S}{1+S}}=:\Lambda_r(\theta;\lambda)^{1/2},$$

and for a general target the squared errors add frequency by frequency: $\operatorname{dist}^2/\|f\|^2=\int|\hat f|^2\Lambda_r\,d\theta\big/\int|\hat f|^2$ (Theorems 1 and 2 of the theory doc; the periodic case is an equality, Corollary 2.1). There is no constant in it.

**Independent verification (the review).** A dense periodic least squares built from scratch, with no repo code, matches $\Lambda_r^{1/2}$ to ratio $1.0000$ in every cell above its own arithmetic floor, for $\mathrm{sech}^2$, tanh, gelu and swish. The three kernel transforms (tanh $\tfrac{\pi\xi/2}{\sinh(\pi\xi/2)}$, gelu $(1+\xi^2)e^{-\xi^2/2}$, swish $\tfrac{\pi^2\xi^2\cosh\pi\xi}{\sinh^2\pi\xi}$) agree with 30-digit quadrature to 12 digits. The theory doc's numerical tables reproduce to four digits.

### 2.2 The right wall of real targets (the evidence)

- **Tones** (ex-expC08; `figures/c08_tones_grid_*.png`): six targets from $\sin\pi x$ to $\sin100\pi x$, $N=16$ to $512$, three activations. The measured wall, defined as the largest $\lambda$ within $10\times$ the cell's floor, equals the $\lambda$ at which $\Lambda_r^{1/2}$ crosses $10\times$ that floor: median ratio $1.00$, 10-90% band $0.96$-$1.03$ over 80 cells. This is a 1-3% statement, not a four-digit one; the wall is floor-referenced, so it does not by itself separate the aliasing law from the floor level.
- **Bounded analytic targets** (ex-expC09; `figures/c09_general_grid_*.png`, `figures/c09_wall_ratio.png`): $1/(1+25x^2)$, $1/(1+100x^2)$, $1/((x-0.5)^2+0.04)$, $\tanh10x$, $e^{-20x^2}$, on tanh, gelu, swish, $N=64$ to $512$. With the target extended past the interval by its own formula, projected on the whole line, and the error scored on the interval, measured/predicted on the wall (points above $30\times$ the floor and below $10^{-3}$) is $1.0000$-$1.0032$ in all 20 group-A cells. The review rebuilt the prediction for Runge from its exact analytic transform with no extension, taper or cap and got the same numbers to four to six digits. The mask uses only the measured curve, so it cannot manufacture agreement.
- **Growing and non-analytic targets:** $e^x$, $x^4-x^2$: $0.86$-$1.00$, rising to $1.00$ at $N=512$; $e^{\sin3\pi x}$: $0.99$-$1.00$; $|x|^3$: $0.93$-$0.99$ including its algebraic resolution floor at every width. Runge's and $\tanh10x$'s flat resolution floors at small $N$ and the kink into the wall are also predicted.
- **The one failure:** the chirp $\sin(10\pi x^2)$ at $N\le128$, where the prediction depends on how the target is extended past the interval; any extension either crosses Nyquist or has a junction that leaks at $10^{-7}$. Predicted at $N\ge256$.

What makes this work, and where it stops: on the right wall the error is pure approximation error, an identity; the interval with a halo behaves like the infinite lattice; dense sampling makes discrete least squares the continuous projection. The left side of the curve is finite-precision arithmetic and is not covered by the theorem.

### 2.3 The activation constant

For a resolved target ($\theta_B\ll\pi$) the wall's location is nearly target-independent. The constant

$$A_K(\lambda_{\text{act}})=\frac{|\widehat K(2\pi/\lambda)|}{|\widehat K(0)|}=\varepsilon$$

drops the $\sqrt2(\theta/2\pi)^r$ factor of the exact floor and therefore sits left of the measured wall by construction. Values (fp64, $\varepsilon=2^{-52}$): tanh $0.244$, sigmoid $0.488$, gelu $0.699$, swish $0.445$, $\mathrm{sech}^2$ $0.244$, gaussian $0.523$; expC07's anchored version ($\varepsilon^\ast=5.65\times10^{-16}$) gives $0.250/0.500/0.707/0.455$. Measured walls of resolved targets sit above these by $1.35\times$ (tanh), $1.27\times$ (gelu), $1.62\times$ (swish). expC07's constant was therefore not lucky: it is conservative by a factor the theorem names.

## 3. What was tested and what the tests showed

### 3.1 The frequency-resolved rule (Sam's tables) does not survive

The rule budgets $\varepsilon$ against the target's own two ghosts without the attenuation factor and without any left wall. At $q=\theta_B/\pi=0.5$ it halves $\lambda$ for tanh; the measured optimum drops by about a quarter. In those cells the error at its $\lambda$ is $10^2$-$10^4$ times worse than at the constant, for every halo in the ladder, so this is not a halo artifact, and the $h=2/(N-1)$ vs $2/N$ convention moves the rule by one unit in the third decimal. For gelu the rule tracks the optimum, because gelu's left wall is gentle. (`figures/c08_tones_grid_*.png`, `figures/c08_wall_vs_q.png`.)

### 3.2 The two-wall rule: a location rule, not a bound

Adding a computational term $E_C=c\varepsilon(1+W\theta_B^r/(\lambda^{r-1}\widehat K(\theta_B/\lambda)))$ ($c=10$, declared) to $E_A=\Lambda_r(\theta_B;\lambda)^{1/2}$ and taking $\lambda^\star=\arg\min(E_A+E_C)$ gives a rule that needs no solve and no target beyond its band edge. What the data support and what they refute:

- **Withdrawn: "$E\le E_A+E_C$" and "$E^\star$ is the guaranteed accuracy."** On the pre-registered test the level is violated in 70% of fp64 cells and 75% of fp32 cells, by up to $2\times10^7$. Six of the twelve held-out targets are not band-limited, which is a hypothesis of the aliasing bound; the 95%-energy band edge used for them is a proxy that takes the rule outside its own theorem. The sum of a bound on approximation error and a bound on linear-algebra error does not bound the total when the target's own content is unresolved.
- **Withdrawn: the $(1-q/2)$ law and the "decades lost $=16q/(2-q)$" formula.** The frozen rule's own $\lambda^\star(q)$ is non-monotone in $q$ (tanh, $N=128$: $0.224$ at $q\to0$, $0.271$ at $q=0.1$, $0.232$ at $q=0.5$); the three-digit agreement quoted in the superseded note could not be regenerated under any normalization.
- **Survives, as a consistency check:** on the expC08 tone cells with $q>0.25$ (15 cells, the data $c$ was set on) the rule's median regret is $1.01$ against the constant's $2.42$; gelu $1.00$ against $59.8$ (max $150$). On the six held-out under-resolved cells of expC10 the rule is $1.0$-$1.1$ and the constant $1.4$-$2$ (tanh), $13$-$33$ (gelu), $1.0$ (swish). The resolved limit reduces to the constant of Section 2.3, and $c$ between $1$ and $100$ moves $\lambda^\star$ by about $\pm8\%$.

### 3.3 The pre-registered test (ex-expC10)

Frozen rule, hashed predictions for 288 cells before any solve (the hashes still match byte for byte; the scoring script was not hashed, which the review rightly flags), 12 seeded random targets, three activations, $N=64$ to $512$, fp64 and fp32; regret $=E(\lambda^\star)/E_{\min}$ on a 5-point running median of each curve; criteria fixed in advance.

- **fp64: passes, and the criterion has teeth.** Resolved cells median regret $1.38$, 90th percentile $6.3$ (criteria $2$ and $10$); under-resolved 90th percentile $1.10$. No fixed $\lambda$ on the 60-point grid passes ($0/60$). The constant $A_K=\varepsilon$ **fails** the registered under-resolved criterion at fp64 (p90 $22.9$).
- **fp32: uninformative, not passed.** Seven fixed $\lambda$'s pass with no theory, including $\lambda=1.0$, as do the constant $\times1.5$ and $\times2$; 18 predictions sit on the rule grid's boundary $\lambda=1.5$. Nothing about fp32 is established.
- **Withdrawn: "the dominant failure mode is a tilted floor."** Over all low-$q$ fp64 cells the median curve gets worse to the left, by $6.6\times$ (tanh) and by three to four decades (gelu, swish). A genuine $25\times$ dip at $\lambda\approx0.13$ exists in exactly two cells (tanh, $N=128$, the two $K=2$ polynomials), and those cells produce the maximum regret in both precisions. The fp32 argmin ranges quoted in the superseded writeup are wrong for gelu and swish.
- **Effective sample size.** 288 cells rest on 12 band edges from two generators and 30 distinct $q$ values; the entire under-resolved arm is two targets. Treating cells as independent overstates the evidence by roughly an order of magnitude.

### 3.4 The halo

A ladder $\{0,4,8,16,32,64,\max(70,0.4N)\}$ on the tone cells (`figures/c08_halo_ladder_*.png`). What holds: halo $0$ loses five to ten decades, $4$ and $8$ lose $10^2$-$10^3$; from $16$ up the curves coincide with the default across the basin and the wall for $\lambda\ge0.15$, so the requirement is a kernel reach in cells, independent of $N$, not $0.4N$. What does not hold: "halo 32 is the rule" was produced by a tolerance placed 5% above halo 32's worst 90th-percentile penalty; halo 16 has the better typical penalty at every width. The defensible statement is **at least 16 cells per side, independent of $N$**, with 32 a margin choice. Below $\lambda=0.15$ no halo in the ladder is adequate for gelu and swish (halo-32 vs halo-256 error ratio up to $148\times$), so nothing measured at $\lambda<0.15$ with halo 32 is a statement about the target.

## 4. The hardened recommendation

For fp64 least-squares readouts on a uniform grid:

1. **Choose $\lambda$ from the activation alone:** the largest $\lambda$ with $|\widehat K(2\pi/\lambda)|/|\widehat K(0)|=\varepsilon$, i.e. tanh $0.244$, gelu $0.699$, swish $0.445$ (or expC07's $0.25/0.707/0.455$; the difference is immaterial). Do not add a buffer: moving 20% left is worse by more than $1.5\times$ in 12% of tanh cells and 24% of swish cells.
2. **Put the frequency knowledge into the width.** With at least 8 grid cells per shortest wavelength ($q\le0.25$; $N\ge8k$ for content up to $\sin k\pi x$) the constant's regret has median $1.4$-$1.6$, 90th percentile $5$-$11$; one cell in five pays more than $3\times$, mostly targets with sharp features that the band edge under-counts. A factor-3 error in the band edge does not change these statistics.
3. **If the width cannot be raised** ($q>0.25$), use the two-wall $\lambda^\star=\arg\min(E_A+E_C)$ with $c=10$ as a location rule. It beats the constant wherever it has been checked (21 cells, 15 of them calibration data), especially for gelu. Do not use its $E^\star$ as an accuracy estimate.
4. **Halo:** at least 16 cells per side, independent of $N$.
5. **fp32:** untested. Do not carry any of the above over.

## 5. What to test next, in order

1. A held-out set with $q\in[0.3,0.7]$ at several widths and activations: the only regime where the two rules differ, sampled by two targets so far. The fp64 criteria are demonstrably discriminating.
2. The left half of the basin at a halo that is adequate there (256 cells or a $1/\lambda$-scaled halo, stated as a confound), before any statement about $\lambda<0.15$ or about fp32.
3. The two tanh $N=128$ cells with the $25\times$ dip: a floor anomaly at one width, not a mechanism.
4. Hash the scorer and commit the pre-registration before the sweep next time; report $E_{\min}$ with an uncertainty (the floor has 1-2 decades of jitter, so regrets of $1.4$ and $3$ are the same measurement).

## 7. fp32, one quick look (added after the consolidation, 2026-09-08)

One run, pure tones only, the fp32 pipeline (features, right-hand side, solve and forward pass in float32), halo 32, three activations, $N=64$ to $512$, 60 $\lambda$'s: `figures/fp32_quick_tones.png`, `data/fp32_quick_{rows,summary}.json`, script `fp32_quick.py`. Read as data, not as a rule.

- **The fp32 floor is far above $\varepsilon_{32}$ and depends on the target and the width.** Relative $L^2$ floors: $\sin\pi x$ $10^{-6}$ to $10^{-5}$, $\cos4\pi x$ $10^{-5}$ to $5\times10^{-4}$, $\sin16\pi x$ $10^{-4}$ to $4\times10^{-3}$ (unresolved at $N=64$ for gelu and swish). In units of $\varepsilon_{32}=1.2\times10^{-7}$: median $190\times$ (tanh), $1400\times$ (gelu), $350\times$ (swish), range $8$ to $36{,}000$. The floor rises with the tone's frequency and, for a fixed tone, with $N$.
- **Aliasing is irrelevant below $\lambda\approx1$.** The fiber floor stays under the fp32 floor until $\lambda\approx0.9$ to $1.5$ (tanh) and beyond $1.5$ (gelu, swish), and that is where the measured tanh curves turn up; the measured $10\times$ shoulder ($0.88$ to $1.5$) matches the fiber curve's crossing of $10\times$ the floor in every tanh cell. So the fp32 constants ($0.50/1.00/0.86$) are left of the wall, as designed, but the wall is not what limits fp32.
- **The floor is not flat and is noisy.** A decade of jitter across the basin; for gelu and swish the error rises mildly with $\lambda$ across the basin, and for $\sin\pi x$ on tanh at $N\le128$ the minimum is at $\lambda\approx0.03$-$0.05$, $16$ to $50\times$ below the value at the fp32 constant.
- **Regret of three fixed choices** (smoothed curves, 31 scored cells): fp32 constant median $2.9/6.8/9.1$ (tanh/gelu/swish), max $52$; fp64 constant median $2.7/3.7/2.3$, max $651$ (one $\sin16\pi x$ cell at $N=64$ where $0.24$ is on the left wall); $\lambda=0.15$ median $4.4/3.9/2.5$, max $1366$. No fixed $\lambda$ is good everywhere, and the differences between them are within the jitter for most cells.

**Other precisions (added later the same day).** Native formats on this machine are fp32 and fp64 only. An emulated ladder ($p=11$ to $53$ mantissa bits: inputs, coefficients and forward-pass partial sums rounded to $p$ bits, SVD cutoff $2^{-p}$; `precision_ladder.py`, `figures/precision_ladder_*.png`) reproduces the wall at every $p$ (fiber curve at $10\times$ the floor, ratio $1.00$) but **fails validation against native fp32**: at $p=24$ its floors are $5$ to $1000\times$ below the native ones, so the solver's internal arithmetic, which the emulation does not touch, sets the real fp32 floor. The ladder is therefore a statement about a model, not about real precisions (the literature's `chop`-style emulators round after every operation for this reason; see `related_work.md`). Beyond fp64, one mpmath run at $N=32$, tanh, $\sin\pi x$, halo 32, normal equations solved at 265 bits (`mp_beyond_fp64.py`, `figures/mp_beyond_fp64.png`): the measured error lies on the fiber curve from $\lambda=0.14$ to $0.40$, over seventeen decades ($2.8\times10^{-16}$ vs $3.2\times10^{-16}$ at $0.26$; $1.7\times10^{-29}$ vs $2.0\times10^{-29}$ at $0.14$), with a floor of $2\times10^{-31}$ and a wall at $\lambda\approx0.145$, where the constant $A_K=2^{-100}$ gives $0.143$. The same run at 100 bits through the normal equations floors at $10^{-6}$, because the normal equations square a condition number of $10^{15}$; that is a solver artifact, not a property of 100-bit least squares.

What this supports: in fp32 the error is roundoff-dominated at every $\lambda$ up to about $1$, the theory's wall is real but sits where nobody would operate, and the $\lambda$ rule neither helps nor hurts much; the achievable precision is set by roundoff growth with the width and the frequency, which the theory does not model. What it does not support: any fp32 $\lambda$ recommendation from this theory.

## 8. File map

- `hardened_rule.md` (this file); `SKEPTIC_REVIEW_2026-09-08.md` (the adversarial review with its own recomputations).
- `figures/`: `c08_tones_grid_{tanh,gelu,swish}.png`, `c08_wall_vs_q.png`, `c08_halo_ladder_*.png`; `c09_general_grid_{act}_{A,B}.png`, `c09_wall_ratio.png`; `c10_prereg_regret.png`, `c10_prereg_curves_{act}_{fp64,fp32}.png`.
- `data/`: the raw rows, predictions and scores of all three sweeps (`c08_*.json`, `c09_*.json`, `c10_*.json`).
- `archive/writeups/`: the three superseded writeups and the superseded two-wall note, each with a header listing its withdrawn claims.
- `figures/fp32_quick_tones.png`, `data/fp32_quick_*.json`, `figures/precision_ladder_*.png`, `data/precision_ladder_*.json`, `figures/mp_beyond_fp64.png`, `data/mp_beyond_fp64.json` (Section 7); `related_work.md` (prior work).
- **expC08_anchor_rule** (promoted 2026-09-10): Sam's output-ratio anchor rule $B\,\mathcal R_{K,r}\le\varepsilon_p$, exploratory sweeps and a pre-registered test on ten fresh targets; `results/checkpoint_C_geometry/expC08_anchor_rule/expC08_results.md`. Not yet folded into the recommendation above.
- Code, in `experiments/expC07_lambda_energy_rule/lambda_rule/`: `fp32_quick.py`, `precision_ladder.py`, `mp_beyond_fp64.py`, `run_c08_tones.py`, `predictions_c08.py`, `halo_ladder_c08.py`, `run_c09_general.py` (the general fiber projection, `fiber_prediction`), `rule.py` (the frozen two-wall rule), `targets.py`, `predict.py`, `run_c10_prereg.py`, `PREREGISTRATION_c10.md`.
