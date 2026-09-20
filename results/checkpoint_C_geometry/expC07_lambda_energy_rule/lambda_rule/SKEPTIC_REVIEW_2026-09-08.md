# Adversarial review of the 2026-09-08 $\lambda$-rule work (expC08 / expC09 / expC10 + the two theory notes)

**Reviewer:** independent agent, instructed to break the work. **Scope:** `docs/lambda_rule_theory.md`, `docs/lambda_two_wall_rule.md`, `experiments/expC08_lambda_frequency_rule/`, `experiments/expC09_fiber_floor_general/`, `experiments/expC10_two_wall_test/` and all their stored data and figures. **Nothing in the repo was modified**; every number below was recomputed from the stored JSON or from scratch in the project environment. Scratch scripts: `/private/tmp/claude-501/-Users-sam-my-repos-research-collaborations-precisionMLPs/014888b9-334a-40a2-8c01-eba25029defd/scratchpad/skeptic/`.

**Headline.** The mathematics is right and the central empirical claim is stronger than the writeups need it to be: I reproduced the fiber floor from an independent brute-force least squares and from an independent analytic-transform prediction, and it holds. What does not survive is a layer of secondary claims stacked on top of it — the two-wall *bound* is violated in 70 % of its own test cells, the $(1-q/2)$ law does not reproduce from the frozen code, the fp32 conclusions are contradicted by the fp32 figures, the "tilted floor" is a two-cell phenomenon generalised to a law, and the halo-32 rule is a threshold placed 5 % above the number it had to clear.

---

## 1. Verdict table

| # | Claim | Verdict | One line of evidence |
|---|---|---|---|
| 1 | The fiber projection theorem: $\mathrm{dist}(f,W)/\|f\|=\Lambda_r(\theta_0;\lambda)^{1/2}$ for a tone (Thm 1/2, Cor 2.1) | **SURVIVES** | My own periodic dense least squares (basis built from $r$-th differences of periodised $\psi$, no repo code) matches $\Lambda_r$ to ratio $1.0000$ in 21/28 cells; the 7 misses are all cells where the prediction is below the fp64 floor of my construction ($10^{-12}$). |
| 2 | The kernel transforms (tanh $\frac{\pi\xi/2}{\sinh(\pi\xi/2)}$, gelu $(1+\xi^2)e^{-\xi^2/2}$, swish $\frac{\pi^2\xi^2\cosh\pi\xi}{\sinh^2\pi\xi}$) | **SURVIVES** | 30-digit quadrature of $\psi^{(r)}$ against each closed form: ratio $1.000000000000$ at $\xi=0.5,1,2,4,8$ for all three. |
| 3 | The theory doc's Section 9 numerical tables (Tests A, B, C) | **SURVIVES** | All 10 Test-A predicted values reproduce to 4 digits; Test-B measured/predicted recomputed from `expC03/full_sweep.json` gives $0.968$–$0.998$ against the quoted $0.97$–$1.00$; Test-C $\lvert1-M\rvert$ column reproduces to 4 digits; $4A_{\tanh}(0.25)=2.2604\times10^{-15}$ as quoted. |
| 4 | expC09: the fiber floor predicts the measured right wall of general bounded targets "to 3–4 digits, no fitting" | **SURVIVES** — and is not an artifact of the extension machinery | I rebuilt the prediction for $1/(1+25x^2)$ from its **exact analytic** transform $\pi a e^{-a\lvert\theta\rvert}$, $a=N/10$ (no extension, no taper, no cap). It reproduces expC09's stored prediction to 4–6 digits and the measurement to $1.0000$–$1.0002$ (tanh, $N=128,256$, $\lambda=0.35\ldots0.88$). Stored `expC09_wall_ratios.json`: all 20 group-A tanh/gelu/swish cells have median ratio $1.0000$–$1.0032$. |
| 5 | expC08: the measured right wall of the tone cells sits on the fiber curve | **SURVIVES WITH CORRECTION** | Quantitatively true: predicting the wall as "largest $\lambda$ with $\Lambda_r^{1/2}\le10\times$ the measured floor" reproduces the measured `wall10` in 76 cells with median ratio $0.99$ and 10–90 % band $0.96$–$1.02$. Correction: the "wall" is *floor-referenced*, so wall-vs-$q$ plots confound the aliasing curve with the achievable floor. expC08 reports no ratio; the four-digit language belongs to expC09. |
| 6 | expC09: the left wall is predicted "within a factor 1–3" by the truncation model | **DOES NOT SURVIVE as stated** | Recomputed pooled medians over group A: tanh $2.81$, gelu $1.74$, swish $2.97$ — but the 10–90 % range is $0.75$–$13.9$ (tanh), $1.26$–$18.1$ (swish). The quoted trio $1.28/2.31/2.44$ reproduces **only** over group A $\cup\{e^{\sin3\pi x},\lvert x\rvert^3\}$, while the sentence attributes it to "the rational, step and bump targets" (which give $1.74/2.81/2.97$). |
| 7 | Two-wall Theorem C: $E(\lambda)\le E_A+E_C$ with $c=10$; "the bound at the minimizer is the guaranteed accuracy" | **DOES NOT SURVIVE** | On expC10's own scored cells the bound is violated in **70 % (fp64) and 75 % (fp32)** of cells; gelu 95 %/100 %; worst case $E^\star=2.0\times10^{-14}$ against a measured $4.5\times10^{-7}$ (rat_J3_2, gelu, $N=128$) — a factor $2\times10^{7}$. |
| 8 | The $(1-q/2)$ law for tanh, "reproduced to three digits by the numeric minimizer of $B$" | **DOES NOT SURVIVE** | Recomputing $\arg\min_\lambda B$ from the frozen `rule.py`: $\lambda^\star(q)$ is *non-monotone* in $q$ (tanh $N=128$: $0.224$ at $q=0.001$, peak $0.271$ at $q=0.1$, $0.232$ at $q=0.5$). Ratios to the plateau are $0.97/0.86/0.73$ at $q=0.25/0.5/0.75$, not the quoted $0.873/0.745/0.618$. I could not reproduce those three numbers under any normalisation I tried (argmin, wall crossing, $\theta_B\to0$). |
| 9 | expC10 H1 passes at fp64 | **SURVIVES, and the criterion has teeth at fp64** | Recomputed verdict identical to the stored one ($1.38$/$6.29$/$33.6$; under-resolved p90 $1.10$). Null test: **no** fixed $\lambda$ on the 60-point grid passes at fp64 (0/60); per-activation best-fixed-$\lambda$ passes at only 5–6 of 60 values. The constant $A_K=\varepsilon$ **fails** the registered under-resolved criterion at fp64 (p90 $22.9$) — the writeup's "statistically indistinguishable" framing understates this. |
| 10 | expC10 H1 passes at fp32 | **SURVIVES BUT IS UNINFORMATIVE** | At fp32 the criteria are nearly vacuous: fixed $\lambda=0.7,0.72,0.77,0.83,0.88,0.94,1.01$ all pass (7/60), as do "constant $\times1.5$" and "constant $\times2$"; for swish alone 15/60 fixed values pass. 18 of 288 cells have $\lambda^\star$ pinned at the rule's grid maximum $1.5$ (boundary minimum, all swish/fp32) and the measured sweep also stops at $1.5$. |
| 11 | expC10: the failure mode is "a tilted floor" — the error keeps falling toward small $\lambda$ in low-$q$ cells | **DOES NOT SURVIVE as a general statement** | Band medians of the raw curve, $q\le0.1$ fp64, $E[0.04,0.08]/E[0.20,0.30]$: tanh $6.6$, gelu $3.6\times10^{3}$, swish $4.7\times10^{3}$ — going left is *worse*, by 3–4 decades for the $r=2$ activations. $E[0.10,0.16]/E[0.20,0.30]$: tanh $1.03$ (flat), gelu $29$, swish $5.1$. The genuine dip is confined to **tanh, $N=128$, trig_K2** (and it is $\sim25\times$ there); those are exactly the max-regret cells. Median $\log_{10}$ jitter of the raw curve in that band is $1.06$–$1.80$ decades, so single-point argmins are noise. |
| 12 | expC10: "for every resolved target the fp32 minimum sits far left of both rules (tanh argmin 0.04–0.16, gelu/swish 0.07–0.16)" | **DOES NOT SURVIVE** | Actual fp32 resolved argmins: tanh $0.030$–$0.773$ (median $0.205$), gelu $0.030$–$1.500$ (median $0.593$), swish $0.066$–$1.500$ (median $0.826$) against constants $0.501/1.001/0.861$. Only the trig_K2 family sits at $0.03$–$0.16$, and its fp32 curve (figure `expC10_curves_tanh_fp32.png`, rows 1–2) is flat to within its own jitter. |
| 13 | expC08: halo $=32$ per side at every width is the right rule for the lstsq readout | **DOES NOT SURVIVE as a derived rule** (the practical advice is harmless) | The selection is "smallest ladder halo whose pooled p90 penalty $\le$ TOL at every width". Halo 32's worst p90 is $1.52$ ($N=512$); TOL $=1.6$ (`halo_rule.py:33`). TOL $\le1.5\Rightarrow$ nothing passes $\Rightarrow$ "default"; TOL $\ge2.76\Rightarrow$ halo 16 wins. The answer "32" exists only for TOL $\in[1.52,2.76)$ and the chosen value is 5 % above the number it had to clear. Halo 16's *geometric-mean* penalty ($0.47$–$0.93$) is better than halo 32's ($0.59$–$0.90$) at every width. |
| 14 | The pre-registration was honoured (hashes, no post-hoc tuning) | **SURVIVES WITH CORRECTION** | The three SHA-256 values in `PREREGISTRATION.md` match the files on disk exactly, and I recomputed 17 sampled cells of `predictions.json` from `rule.py` with 0 mismatches. Corrections: (i) nothing is committed to git, so the ordering rests entirely on mtimes (rule/targets/predict 19:18:19, prereg+predictions 19:18:24, rows/scores 19:21:08 — I timed the sweep and $\sim160$ s for 17 280 solves on 10 cores is feasible, so the timeline is at least self-consistent); (ii) `run.py`, which contains **all of the scoring** (smoothing, interpolation, resolved/under-resolved split, verdict), was last modified at 19:19:39, *after* the hashes were written, and is not hashed. |
| 15 | The theory doc's activation constants and asymptotics ($\lambda^\star$ table, Prop. 5, Prop. 9) | **SURVIVES** | Recomputed: tanh $0.2500$, sigmoid $0.5000$, gelu $0.7070$, swish $0.4554$, gaussian $0.5302$ at $\varepsilon^\ast=5.65\times10^{-16}$; $B(0.25)=1.056\times10^{-7}$; Prop. 9(iv) $0.281\to0.247$ vs exact $0.250$; fp32 $0.4852$ at $2^{-24}$. All as printed. |

---

## 2. The strongest problems, ranked

### P1. The two-wall "bound" is not a bound, and the writeups do not say so

`docs/lambda_two_wall_rule.md` §4 states Theorem C as $E(\lambda)\le B(\lambda)=E_A+E_C$ and then: *"the bound at the minimizer is the guaranteed accuracy: $E^\star=B(\lambda^\star)$."* On expC10's own 251 scored cells:

| precision | $E^\star/E_{\text{measured}}(\lambda^\star)$, median | fraction with $E^\star<E_{\text{measured}}$ (bound violated) | worst |
|---|---|---|---|
| fp64 | 0.36 | **70 %** (tanh 28 %, gelu 95 %, swish 86 %) | $4.6\times10^{-8}$ |
| fp32 | 0.20 | **75 %** (tanh 33 %, gelu 100 %, swish 100 %) | $8.3\times10^{-3}$ |

Worst fp64 cells: `rat_J3_2` gelu $N=128$, $E^\star=2.04\times10^{-14}$ vs measured $4.46\times10^{-7}$; `rat_J2_3` gelu $N=128$, $4.99\times10^{-14}$ vs $3.77\times10^{-7}$; the swish twins are the same. You can *see* this in `expC10_curves_gelu_fp64.png`: in the `rat_J3_2`, `rat_J2_3` and `rat_J2_5` rows at $N=64$ and $N=128$ the red dot sits 7–9 decades below the blue curve. The expC10 writeup's sentence *"its $E^\star$ dot lies on or near the measured curve (curves figures)"* is contradicted by the figure it cites.

Two structural reasons, both worth fixing rather than hiding:

1. **Six of the twelve held-out targets are not band-limited.** Corollary A4 ($E_{\text{alias}}\le\Lambda_r(\theta_B)^{1/2}$) has "band-limited to $\theta_B$" as a hypothesis. The F2 rationals have $\hat f\propto e^{-s\lvert\omega\rvert}$ with no band edge at all; the pre-registration substitutes a 95 %-energy proxy (`targets.py:band_edge_rational`). $E_A$ then bounds nothing, and the cells where it fails worst are precisely the F2 cells at small $N$. This is a hypothesis violation, not a calibration issue, and it is not flagged anywhere.
2. $E_C$ is a bound on a *different* error (the linear-algebra residual), added to a bound on the *approximation* error; nothing makes the sum bound the total in the regime where the true limitation is that the target's own content is unresolved at that $N$.

The regret result (claim 9) is unaffected — an argmin can be right while the level is wrong by seven decades — but the two must be stated separately. As it stands, $E^\star$ is quoted as a guarantee and reported in S1 as if the only issue were 30 % of cells falling outside a factor 10.

### P2. The $(1-q/2)$ law does not reproduce from the frozen rule

`lambda_two_wall_rule.md` §5: *"The numeric minimizer of $B$ reproduces this to three digits for tanh ($q=0.25$: $0.873$ vs $0.875$; $q=0.5$: $0.745$ vs $0.750$; $q=0.75$: $0.618$ vs $0.625$)."* Recomputing $\arg\min_\lambda[E_A+E_C]$ with `rule.py` (its own $c=10$, $W=N+65$, $\varepsilon=2^{-52}$, its own 600-point grid):

| $q$ | 0.001 | 0.01 | 0.05 | 0.1 | 0.25 | 0.5 | 0.75 |
|---|---|---|---|---|---|---|---|
| tanh $N=128$, $\lambda^\star$ | 0.224 | 0.251 | 0.269 | **0.271** | 0.262 | 0.232 | 0.199 |
| ratio to the $q=0.1$ peak | 0.83 | 0.93 | 0.99 | 1.00 | 0.97 | 0.86 | 0.73 |

The rule's own $\lambda^\star$ is non-monotone in $q$ and its high-$q$ drift is closer to $1-q/3$ than $1-q/2$. I tried three normalisations (ratio to $\lambda^\star$ at $\theta_B\to0$; ratio to the plateau; ratio of the $E_A=E_C$ crossing) and none produces $0.873/0.745/0.618$. Either the doc used a different closed form than the one that was frozen, or the numbers are the closed form $\pi^2(1-q/2)/L'$ evaluated against itself. The gelu/swish sentence in the same paragraph ("the numeric rule falls faster, $0.69$ and $0.65$ at $q=0.5$") does reproduce: I get $0.855$ (gelu) and $1.006$ (swish) relative to $A_K=\varepsilon$, i.e. $0.597$ and $0.448$ absolute — the doc's $0.69/0.65$ appear to be ratios to the $q\to0$ plateau, a third normalisation again.

This matters because the $(1-q/2)$ law is the doc's only closed-form deliverable and the "decades lost $=16q/(2-q)$" formula is derived from it.

### P3. The fp32 section of expC10 is contradicted by its own data

Three separate problems.

- **The argmin claim is wrong** (verdict 12). "For every resolved target the fp32 minimum sits far to the left of both rules" is false for 2 of 3 activations at the median; gelu's median fp32 argmin is $0.593$ and swish's $0.826$, against constants $1.001$ and $0.861$.
- **The criteria are vacuous at fp32.** Seven of the 60 grid $\lambda$'s pass H1 as *fixed constants with no theory*, including $\lambda=1.0$; "constant $\times1.5$" and "constant $\times2.0$" pass. Reporting "H1 passes at both precisions" as the first TL;DR bullet gives fp32 the same evidentiary weight as fp64, where 0/60 fixed $\lambda$'s pass. These are not the same result.
- **18 of 288 predictions are boundary artifacts**: `predictions.json` has $\lambda^\star=1.5$ (the rule grid's right endpoint) for all 18 swish/fp32 cells; the measured sweep also ends at $1.5$, so both the prediction and the interpolated $E(\lambda^\star)$ are evaluated at the edge. Those cells still count toward the fp32 pass.

The writeup's own open question ("the fp32 regime as its own problem") is right; the TL;DR should say *fp32 is untested*, not *H1 passes at fp32*.

### P4. The "tilted floor" is two cells promoted to a mechanism

expC10 TL;DR: *"The dominant failure mode, in both precisions, is a tilted floor... In the low-$q$ cells the measured error keeps falling as $\lambda$ decreases below the rule's value, by a factor 10–30 (fp64)."* Band medians of the **raw** (unsmoothed) curves over all fp64 cells with $q\le0.1$ ($n=33$ per activation):

| activation | $E[0.04,0.08]/E[0.20,0.30]$ | $E[0.10,0.16]/E[0.20,0.30]$ | median $\log_{10}$ jitter over $[0.04,0.30]$ |
|---|---|---|---|
| tanh | 6.60 | 1.03 | 1.06 dec |
| gelu | $3.6\times10^{3}$ | 29.0 | 1.80 dec |
| swish | $4.7\times10^{3}$ | 5.11 | 1.50 dec |

The median low-$q$ cell gets *worse* to the left, catastrophically so for $r=2$. Per-cell, the genuine dip lives in `trig_K2_{0,1}` + tanh + $N=128$ only: there $E[0.10,0.16]=3.5\times10^{-15}$ / $2.0\times10^{-15}$ against $E[0.20,0.30]=8.7\times10^{-14}$ / $6.2\times10^{-14}$. At $N=64$ and $N=256$ the same targets show no dip ($1.3$ vs $1.4$, $2.7$ vs $5.1$, in units of $10^{-14}$). What is actually anomalous is that the $\lambda\approx0.25$ floor for those two targets is $4$–$6\times$ higher at $N=128$ than at $N=64$ or $N=256$ — a floor-level anomaly at one width, not a monotone tilt and not a law. Calling it "the dominant failure mode... which neither rule models" writes a mechanism into two cells.

Related: $E_{\min}$ is the minimum of a 5-point running median over a curve with 1–1.8 decades of jitter (`run.py:65,85`). On a genuinely flat noisy curve that estimator is biased low, which inflates every regret. The direction is conservative for the rule, but it means the reported regret distribution is partly a noise-floor measurement.

### P5. The halo-32 rule is a threshold set to produce the answer

`halo_rule.py:33` `TOL = 1.6`, `halo_rule.py:81` `ok = [h for h in fixed if all(stats[N][h]["p90"] <= TOL ...)]`. Recomputed p90 penalties vs the default halo, over $\lambda\ge0.15$:

| halo | $N{=}16$ | 32 | 64 | 128 | 256 | 512 | geometric mean range |
|---|---|---|---|---|---|---|---|
| 8 | 2.34 | 3.28 | 7.97 | 22.1 | 32.1 | 37.4 | 0.67–2.36 |
| 16 | 1.01 | 1.12 | 1.54 | 1.85 | 1.65 | **2.76** | 0.47–0.93 |
| 32 | 1.25 | 1.37 | 1.25 | 1.39 | 1.50 | **1.52** | 0.59–0.90 |
| 64 | **2.06** | 1.60 | 1.30 | 1.40 | 1.40 | 1.34 | 0.65–1.07 |

Sweeping TOL: $\le1.5\to$ nothing passes $\to$ "default"; $1.6\to32$; $2.0\to32$; $3.0\to16$. The rule "32" is the output of a threshold placed $5\%$ above halo 32's worst value. Two further problems: (i) halo 16 has a *better* geometric-mean penalty at every width, so on the typical cell it is the better choice and the p90 is measuring floor jitter rather than halo inadequacy — the metric that decides is noise; (ii) the band $\lambda\ge0.15$ (`halo_rule.py:34`) excludes exactly the region where the kernel reach makes the halo bind, and expC10 then reports its fp32 optima and its "tilt" argmins at $\lambda=0.03$–$0.16$, i.e. inside the excluded region. From expC09's two stored halos, at $\lambda<0.10$ the halo-32/halo-256 error ratio has p90 $148$ (gelu) and $19.8$ (swish). Nothing at $\lambda<0.15$ in expC10 is halo-safe.

The figure caption is also wrong: *"Look for: halo 32 (green) staying under the $1.6\times$ line for $\lambda\ge0.15$ at every $N$."* The per-$\lambda$ p90 curve for halo 32 exceeds $1.6$ at $14\%$–$34\%$ of the band's $\lambda$ points depending on $N$, with maxima $3.4$–$5.5$. The quoted $1.25$–$1.52$ is the p90 of the pooled (cell, $\lambda$) sample; the figure plots the p90 *per $\lambda$*. Two different statistics, one caption.

*What is fine here:* halo 0/4/8 are genuinely inadequate (5–10 decades, $10^2$–$10^3\times$), and any halo in $\{16,32,64\}$ is defensible. The defensible statement is "$\ge16$ cells per side, independent of $N$", with the 32 as a margin choice, not a measured optimum.

### P6. expC09's left-wall numbers are pooled over a different target set than the sentence says

The quoted trio *"median measured/predicted is $1.28$ (gelu), $2.31$ (tanh), $2.44$ (swish)"* for "the rational, step and bump targets" reproduces exactly only when the pool is group A **plus** $e^{\sin3\pi x}$ and $\lvert x\rvert^3$. Over the named targets (group A) it is $1.74/2.81/2.97$; over all ten it is $1.15/1.83/2.09$. More importantly the TL;DR's "within a factor 1–3" hides a 10–90 % range of $0.75$–$13.9$ (tanh) and $1.26$–$18.1$ (swish) over the same points, and per-cell medians up to $14.7$ for the green backward-stability line (`expC09_wall_ratios.json`, gelu/runge100/$N{=}64$). The $N$-drift claim ("Runge on tanh: $1.4$ at $N=64$, $6.1$ at $N=256$") is exactly right ($1.36$, $6.10$) — that one I could not break.

### P7. Smaller but checkable errors

- **fp32 constants are inconsistent between the doc and the frozen code.** `lambda_two_wall_rule.md` §5 tabulates fp32 tanh $0.485$ / gelu $0.984$ / swish $0.834$; those solve $A_K(\lambda)=2^{-24}$. `rule.py:11` uses `np.finfo(np.float32).eps` $=2^{-23}$, giving $0.503/1.003/0.862$ — and the fp64 row of the same table uses $2^{-52}$, not $2^{-53}$. The doc's two rows use different unit-roundoff conventions, and its "sharpest untested prediction" is not the number the pre-registered code tested.
- **"The measured walls of resolved targets sit at $1.3\times$ these values"** (§5). Recomputed over $q\le0.06$ cells: tanh $1.35$, gelu $1.27$, swish **$1.62$** (range $1.52$–$1.74$). Swish is not $1.3$.
- **"$\lambda^\star$ lies between the measured argmin and the measured wall in every tone cell tested today (46 cells)"** (§4). Applying the frozen rule to every expC08 cell with $E_{\min}\le10^{-6}$: **54 of 76** cells satisfy it. Of the 22 violations, 15 are within 10 % of the interval and 7 are material (worst: gelu `f8` $N=128$, $\lambda^\star=0.722$ against a wall at $0.593$, and gelu `f8` $N=256$, $0.756$ vs $0.677$).
- **"verified to four digits in expC08 on 36 cells per activation"** (§2). expC08 computes and reports no measured/predicted ratio at all; its evidence is visual overlay plus the `wall10` summary. The four-digit ratios exist in expC09, for non-tone targets. (I supplied the missing quantitative tone check below; it comes out at 1–3 %, not four digits.)
- **expC10 quoted example mixes two cells**: "tanh $K=2$ at $N=128$ (argmin $0.13$, $E_{\min}=2\times10^{-15}$, $E(\lambda^\star)=7\times10^{-14}$, regret 34)". `trig_K2_0`: $0.129$, $2.56\times10^{-15}$, $8.61\times10^{-14}$, $33.6$. `trig_K2_1`: $0.121$, $1.96\times10^{-15}$, $6.51\times10^{-14}$, $33.2$. Neither is the quoted row.
- **Figure defect**: `expC08_halo_adequacy.png` — the two-line $y$-label of the bottom-left panel overprints the top-left panel's label; unreadable in the left margin.

### P8. What I attacked and could not break

Worth saying, because it changes what the PI should do next.

- **Circularity in the "no fitting" claim for the right wall.** The `on_wall` mask (`expC09/run.py:242`) selects points from the *measured* curve only (above $30\times$ the cell floor, below $10^{-3}$, right of the argmin); the prediction never enters. Comparison is raw measured against raw predicted (no smoothing on either side; smoothing appears only in expC08's wall and expC10's regret). Widening or narrowing the mask cannot manufacture $1.0000$.
- **Extension/taper as a hidden fit.** Rebuilt from the exact analytic transform with no extension at all: agreement unchanged to 4–6 digits. `REACH_CELLS=800`, `TAPER=200`, the `X_MAX` caps and the chirp's 60-cell taper are load-bearing only for the growing targets and the chirp, exactly as the writeup says.
- **$m_{\max}=8$ truncation.** $E_A$ at $m_{\max}=8$ and $m_{\max}=20$ agree to $10^{-6}$ relative for all three kernels at $\lambda=0.25,0.7,1.5$.
- **Lemma A3 / the band-edge bound.** $\Lambda_r(\theta;\lambda)$ is numerically monotone increasing on $(0,\pi)$ for all three kernels at $\lambda\in\{0.25,0.45,0.7,1.0,1.5\}$, including gelu whose $\widehat K$ has an interior mode. The band-edge choice is genuinely conservative *for band-limited targets*.
- **Fairness of the expC08 "frequency-resolved rule failed" verdict (halo confound).** At the new rule's $\lambda$ ($0.12$–$0.13$ tanh, $0.24$–$0.25$ swish) the error is $10^{-6}$–$10^{-8}$ for **every** halo in $\{4,8,16,32,64,\text{default}\}$, against $10^{-9}$–$10^{-10}$ at the constant. The verdict is not a halo artifact. The $h=2/(N-1)$ vs $2/N$ convention moves the rule's $\lambda$ by $\le1$ unit in the third decimal, as claimed.
- **Pre-registration hashes.** All three match byte-for-byte, and 17 sampled prediction cells recompute exactly from `rule.py`.
- **Sweep feasibility.** I timed `solve_cell`: $0.008$–$0.104$ s per cell, so 17 280 solves on 10 cores in the $\sim164$ s between the `predictions.json` and `expC10_rows.json` mtimes is comfortably possible. The claimed ordering is at least physically consistent.

---

## 3. What I recomputed, and what I got

All scripts in the scratchpad directory named above.

**(a) Kernel transforms** (`khat_check.py`). 30-digit `mpmath` quadrature of $\int K(u)e^{-i\xi u}du$ for $K=\mathrm{sech}^2$, $\mathrm{gelu}''$, $\mathrm{swish}''$ against the three closed forms. Ratio $=1.000000000000$ at $\xi=0.5,1,2,4,8$ for all three; $\widehat K(0)=2,1,1$ respectively, as the normalisations assume.

**(b) The fiber formula from scratch** (`fiber_bruteforce.py`). Periodic torus with $P$ lattice points; basis = $r$-th finite differences of the periodised activation (so the $r\ge1$ cases are in $L^2$); dense `gelsd` least squares of $\cos(\theta_0u)$ on 4096 torus points; relative $L^2$ of the residual against $\Lambda_r(\theta_0;\lambda)^{1/2}$. No repo code used.

| activation ($r$) | cells with prediction above my $10^{-12}$ floor | measured/predicted |
|---|---|---|
| $\mathrm{sech}^2$ (0) | 7/7 | $1.0000$ every cell ($7.4\times10^{-6}$ down to $2.1\times10^{-7}$) |
| tanh (1) | 7/7 | $1.0000$ every cell ($1.0\times10^{-3}$ down to $6.8\times10^{-9}$) |
| gelu (2) | 3/7 | $1.0030$, $1.0000$, $1.1088$ |
| swish (2) | 4/7 | $1.0031$, $1.0000$, $1.0000$, $1.6403$ |

The gelu/swish misses are the cells where $\Lambda_r^{1/2}<10^{-12}$, i.e. below the floor of my own second-difference construction. **Theorem 2 and Corollary 2.1 are correct.**

**(c) Independent prediction for a general target** (`indep_pred.py`). $1/(1+25x^2)$ has $\hat g(\theta)=\pi a e^{-a\lvert\theta\rvert}$, $a=1/(5h)=N/10$, in grid units. I sampled that exact transform on the box frequencies and ran the fiber projection — no extension, no taper, no cap, no FFT of the target. Against the stored halo-256 measurement, tanh: $1.0001, 1.0000, 1.0000, 1.0000, 1.0000$ at $N=128$ and $1.0062, 1.0002, 1.0002, 1.0002, 1.0002$ at $N=256$ over $\lambda=0.35\ldots0.88$; against expC09's own stored prediction, agreement to 4–6 digits everywhere the prediction is above $10^{-14}$. **The four-digit right-wall claim is real and the extension machinery is not doing the work for bounded analytic targets.**

**(d) The missing quantitative tone check for expC08.** For each of the 76 scorable cells with $E_{\min}\le10^{-6}$ I computed $\lambda_{\text{pred}}=\max\{\lambda:\Lambda_r(\theta_0;\lambda)^{1/2}\le10E_{\min}\}$, the exact analogue of the measured $10\times$-shoulder wall. Median measured/predicted $=1.001$ (tanh), $0.987$ (gelu), $0.995$ (swish); pooled 10–90 % band $0.957$–$1.024$. The only outliers are `f8`/gelu at $N=128,256$ ($0.72,0.78$), gelu `sin_100pi` at $q=0.78$ ($1.13$), and one tanh cell at $0.41$ — the `sin_1pi` cell whose wall expC08 itself flags as unreliable because its floor slopes downward to the left. So the wall is *exactly* "the fiber curve crossing ten times the achievable floor" — which is also why wall-vs-$q$ plots cannot separate the aliasing law from the floor level.

**(e) The two-wall rule on the expC08 tone set — the discriminating test expC10 did not run.** expC10's held-out set has $q>0.25$ in only 24 of 288 cells, essentially two targets (`trig_K32_{0,1}`). expC08's tones give a real $q$ spread. Applying the frozen `rule.py` (fp64, halo 32, $W=N+65$) to all 75 scorable expC08 cells:

| split | $n$ | two-wall: median / p90 / max regret | constant $A_K=\varepsilon$: median / p90 / max |
|---|---|---|---|
| $q\le0.25$ | 60 | **1.30** / 6.32 / 52.4 | 1.44 / 4.77 / 48.7 |
| $q>0.25$ | 15 | **1.01** / 1.54 / 8.7 | 2.42 / 65.2 / 149.9 |
| $q>0.25$, gelu only | 5 | **1.00** / — / 1.0 | 59.8 / — / 149.9 |

The under-resolved advantage is real and much better evidenced here than on the six expC10 cells — **with the caveat that $c=10$ was calibrated on this very data**, so this is a consistency check, not a held-out test. It is nonetheless the first place the two rules visibly separate on more than two targets.

**(f) Null tests on the pre-registered criteria** (`c10_null.py`). Regret of trivially bad rules under expC10's own scoring:

| rule | fp64 (resolved median / p90; under p90) | verdict | fp32 | verdict |
|---|---|---|---|---|
| two-wall (registered) | 1.38 / 6.29; 1.10 | PASS | 1.36 / 7.27; 1.15 | PASS |
| constant $A_K=\varepsilon$ | 1.38 / 6.71; **22.9** | **FAIL** | 1.76 / 6.49; 1.62 | PASS |
| fixed $\lambda=0.25$ | 2.00 / 530; $2\times10^{6}$ | FAIL | 3.17 / 121; 122 | FAIL |
| fixed $\lambda=0.5$ | 1.85 / $2.6\times10^{5}$ | FAIL | 1.94 / 12.6; 1.62 | FAIL |
| fixed $\lambda=0.8$ | 81 / $2\times10^{8}$ | FAIL | **1.63 / 4.90; 1.83** | **PASS** |
| fixed $\lambda=1.0$ | $1.4\times10^{4}$ / … | FAIL | **1.83 / 6.30; 6.66** | **PASS** |
| constant $\times1.5$ | 60 / $1.4\times10^{4}$ | FAIL | **1.38 / 6.58; 1.40** | **PASS** |

Scan over the whole 60-point grid: **0/60** fixed $\lambda$'s pass at fp64; **7/60** pass at fp32. Per-activation fixed $\lambda$ (a three-number rule fitted on the test data itself) passes at 5/60 (tanh), 6/60 (gelu), 5/60 (swish) values at fp64, and 3/13/15 at fp32.

**(g) Bound violation** (§P1 table), **(h) tilt band medians** (§P4 table), **(i) halo tolerance sweep** (§P5 table), **(j) $\Lambda_r$ monotonicity and $m_{\max}$ sensitivity** (§P8) — all as tabulated above.

**(k) Effective sample size.** expC10's 288 cells are 12 targets $\times$ 3 activations $\times$ 4 widths $\times$ 2 precisions, but the targets carry only 12 band edges $\Omega/\pi\in\{1.47,2.0,2.0,2.15,2.39,3.66,5.99,6.1,8.0,8.0,32.0,32.0\}$ from two generators, and $q$ is a deterministic function of $(\Omega,N)$: 30 distinct $q$ values, and the entire under-resolved arm is `trig_K32_{0,1}`. Median regret by target ranges $1.02$–$2.17$ and by activation $1.17$–$1.82$; the width axis is nearly inert ($1.28$–$1.50$). Treating 123 fp64 resolved cells as 123 independent trials overstates the evidence by roughly an order of magnitude in $n$.

---

## 4. What to keep, discard, and re-test

### Keep (publishable as it stands)

1. **Theorem 1 / Theorem 2 / Corollary 2.1** and the $\Lambda_r$ formula. Independently verified from scratch, exact to five digits wherever it is above the arithmetic floor. This is the real result of the whole $\lambda$ program.
2. **The kernel table** (transforms, $r$, tail classes, $\lambda^\star$, $B(\lambda^\star)$, Prop. 5, Prop. 9). All constants check out to the digits printed.
3. **expC09's right wall for bounded analytic targets.** $1.0000$ on 20/20 group-A cells, corroborated by an independent analytic-transform reconstruction. The extension is not doing the work. State the mask ($30\times$ floor to $10^{-3}$) explicitly in the writeup — it is legitimate but it is a mask.
4. **expC08's negative result** on the frequency-resolved rule for tanh and swish. Robust to the halo (every ladder entry) and to the $h$ convention.
5. **The measured wall $=$ fiber curve at $10\times$ the floor** (my item (d)) — put this number in expC08 in place of the visual claim, and drop "four digits" from `lambda_two_wall_rule.md` §2.

### Discard, or demote to "observed once"

6. **"$E^\star$ is the guaranteed accuracy."** Delete. Replace with: $E_A+E_C$ is a *level model*, violated in 70 % of cells by up to $10^7$; its value is the *location* of the minimum, not the level. Simultaneously restrict Corollary A4's use to band-limited targets, or state explicitly that the 95 %-energy proxy takes the rule outside the theorem.
7. **The $(1-q/2)$ law and everything derived from it** (including $16q/(2-q)$ decades) until it reproduces from `rule.py`. Right now the doc's three numbers cannot be regenerated by any normalisation I tried and the rule's own $\lambda^\star(q)$ is non-monotone.
8. **"H1 passes at fp32."** Demote to "the fp32 criteria are not discriminating: 7 of 60 fixed $\lambda$'s pass, including $\lambda=1.0$, and 18 predictions sit on the grid boundary." Remove the fp32 argmin ranges from the TL;DR; they are wrong for gelu and swish.
9. **"The dominant failure mode is a tilted floor."** Demote to: "two tanh $N=128$ cells show a $25\times$ dip at $\lambda\approx0.13$; elsewhere the low-$q$ curve is flat (tanh) or rises by 3–4 decades to the left (gelu, swish)." The general statement is contradicted by the same data.
10. **"halo 32 at every width is the right rule."** Replace with "$\ge16$ cells per side, independent of $N$; 32 chosen for margin", and say plainly that TOL was set above the observed p90. Also fix the figure caption: halo 32's per-$\lambda$ p90 exceeds $1.6$ at $14$–$34\%$ of the band.
11. **expC09's "within a factor 1–3" for the left wall.** Report the 10–90 % range ($0.75$–$18$) and name the actual pool.
12. **The fp32 constants in `lambda_two_wall_rule.md` §5.** Recompute at whichever $\varepsilon$ the code uses, or state both conventions. As printed the doc and `rule.py` disagree by 3–4 %.

### Re-test (in priority order)

13. **A held-out set with $q\in[0.25,0.8]$ at several widths and several activations.** This is the *only* regime where the two-wall rule differs from the constant, and expC10 sampled it with two targets. My item (e) shows the separation is large (constant p90 regret $65$, max $150$; gelu median $60$) but on calibration data. Twelve new band-limited targets with $\Omega/\pi$ chosen so that $q\in[0.3,0.7]$ at $N\in\{64,\ldots,512\}$ would settle it in one afternoon, and the criteria at fp64 are demonstrably discriminating (0/60 fixed $\lambda$'s pass).
14. **Re-run the low-$\lambda$ half of expC10 at halo 256.** Every conclusion about $\lambda<0.15$ — the tilt, the fp32 optima, the "far-left minima" — is drawn in a region the halo rule explicitly excluded, where expC09's own two halos differ by up to $148\times$ (gelu, p90). Until this is done, no statement about the left of the basin is safe.
15. **Diagnose the $N=128$ tanh floor anomaly** (`trig_K2_{0,1}`: floor $4$–$6\times$ higher at $N=128$ than at $N=64,256$ near $\lambda=0.25$). It generates the max regret in both precisions and is currently explained by a mechanism ("tilt") that the aggregate data contradicts.
16. **Hash the scorer, not just the rule.** `run.py` — which contains the smoothing, the interpolation, the resolved/under-resolved split and the verdict logic — was last written *after* the pre-registration hashes and is not covered by any of them. Commit the pre-registration and its hashes to git before the sweep next time; mtimes are not a timestamp.
17. **Report $E_{\min}$ with an uncertainty.** With 1–1.8 decades of jitter on the floor, "regret $=1.4$" and "regret $=3$" are the same measurement. A bootstrap over the $\lambda$ grid, or a floor estimated as a band median rather than a running-median minimum, would make the regret numbers mean something.
