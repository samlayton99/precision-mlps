> **SUPERSEDED (2026-09-08).** Kept as the record of what was claimed. Do not cite. Withdrawn or corrected per the skeptic review: 'H1 passes at fp32' (7 of 60 fixed lambdas pass; 18 boundary predictions); the fp32 argmin ranges (wrong for gelu and swish); 'the dominant failure mode is a tilted floor' (two tanh N=128 cells); 'E* lies on the curve' (the level model is violated in 70% of cells); the quoted example mixes two cells. Survives: H1 passes at fp64 with discriminating criteria (0 of 60 fixed lambdas pass; the constant fails the under-resolved criterion, p90 22.9); the hashes; the six under-resolved cells. The hardened statement is `../../hardened_rule.md`.

# expC10 -- Pre-registered test of the two-wall rule on held-out targets, fp64 and fp32

**Status:** draft-pending-Sam. Pre-registration: `experiments/expC10_two_wall_test/PREREGISTRATION.md` (hashes of the rule, the target generator and the prediction file are recorded there before any solve). The rule (`docs/lambda_two_wall_rule.md`) was not modified during or after this experiment, and is not modified by this writeup.

## TL;DR

- **H1 passes at both precisions under the criteria fixed in advance.** Resolved held-out cells: median regret $1.38$ (fp64), $1.36$ (fp32), 90th percentile $6.3$ and $7.3$ (criteria $\le2$ and $\le10$). Under-resolved cells: 90th percentile $1.1$ and $1.2$ (criterion $\le10$).
- **The pass does not show an advantage over the activation-only constant on this set.** On resolved cells the two rules are statistically indistinguishable (median ratio of regrets $1.03$ fp64, $0.91$ fp32; each wins about half the cells), because the held-out set is almost entirely resolved, where the theory itself says the rule reduces to the constant. The only cells where they differ are the six fp64 under-resolved cells, where the rule's regret is $1.0$-$1.1$ against the constant's $1.4$-$2.0$ (tanh), $13$-$33$ (gelu), $1.0$ (swish), the pattern the theory predicts, on too few cells to be more than consistent.
- **The dominant failure mode, in both precisions, is a tilted floor the theory does not model.** In the low-$q$ cells the measured error keeps falling as $\lambda$ decreases below the rule's value, by a factor $10$-$30$ (fp64) and up to $80$ (fp32), with the argmin at $\lambda\approx0.04$-$0.15$. The rule and the constant sit at the high end of that tilt and pay the same regret. The theory's computational term is flat or decreasing toward small $\lambda$ at low $q$, so it cannot see this.
- **fp32 in particular:** the measured minima sit far left of every rule for every resolved target (argmin $0.04$-$0.16$ against the rule's $0.58$-$0.68$ for tanh), and the fp32 wall predictions are not where the fp32 optima are. The criteria still pass because the tilt is modest in most cells.
- S1: the rule's predicted error $E^\star$ is within a factor 10 of the measured minimum in $70\%$ (fp64) and $72\%$ (fp32) of scored cells; tanh's $E^\star$ runs high (median $+0.8$ decades), gelu's and swish's low ($-0.4$ to $-0.8$). S2: misstating the band edge by a factor 3 either way changes the regret statistics by less than the cell-to-cell spread.

## Question / hypothesis

H1, as registered: given only the activation, $N$, $\varepsilon$ and the target's band edge, does $\lambda^\star=\arg\min[E_A+E_C]$ of the two-wall rule achieve error within the fixed criteria of the best attainable on that cell, for targets never used in developing the rule, at fp64 and fp32?

## Experiment design

Everything below was fixed in the pre-registration before any solve; see that file for the exact statements.

**Rule.** $E_A=\Lambda_r(\theta_B;\lambda)^{1/2}$ (the fiber floor at the band edge) and $E_C=c\varepsilon(1+W\theta_B^r/(\lambda^{r-1}\widehat K(\theta_B/\lambda)))$ with $c=10$, $W=N+65$; $\lambda^\star$ the minimizer over $\mathrm{geomspace}(0.03,1.5,600)$; $\varepsilon=2^{-52}$ (fp64), $2^{-23}$ (fp32); $\theta_B=2\Omega/N$. Predictions for all 288 cells written to `predictions.json` (SHA-256 `e4c6d854...bbfe7a`) before the sweep.

**Held-out targets** (seed 20260908, generated and not inspected): six random trigonometric polynomials $\sum_{k\le K}a_k\sin(k\pi x+\phi_k)$ with $K\in\{2,8,32\}$, band edge $K\pi$ exactly; six random rational functions $\sum_jc_js_j/((x-p_j)^2+s_j^2)$ with one to three poles at random positions and widths $s_j\in[0.05,0.5]$, band edge the 95%-energy frequency of the closed-form transform ($\Omega/\pi$ between $1.5$ and $6.1$).

**Measurement.** expC09's solver: uniform grid, halo 32, $\gamma=\lambda/h$, augmented $[\Phi,\mathbf1]$ by `gelsd` with default cutoff, features, right-hand side, solve and forward pass all in the stated dtype, residual against the fp64 target in fp64, rel $L_2$ on 4001 points. 3 activations $\times$ 4 widths ($64$-$512$) $\times$ 2 precisions $\times$ 12 targets $\times$ 60 $\lambda$; 17280 solves.

**Scoring.** Per cell the curve is smoothed (5-point running median in log), $E_{\min}$ is its minimum, regret $\rho=E(\lambda^\star)/E_{\min}$. Cells with $q=\theta_B/\pi\le0.25$ and $E_{\min}$ below $10^{-6}$ (fp64) or $10^{-3}$ (fp32) are *resolved*; $q>0.25$ with $E_{\min}$ below threshold are *under-resolved*; the rest are unscored. Criteria: resolved median $\rho\le2$ and 90th percentile $\le10$; under-resolved 90th percentile $\le10$. Secondary: S1 $E^\star$ vs $E_{\min}$; S2 the rule at $3\Omega$ and $\Omega/3$; S3 the constant $A_K(\lambda)=\varepsilon$.

**Code & data.** `experiments/expC10_two_wall_test/` (`PREREGISTRATION.md`, `rule.py`, `targets.py`, `predict.py`, `run.py`). `results/checkpoint_C_geometry/expC10_two_wall_test/`: `predictions.json`, `expC10_rows.json`, `expC10_scores.json` (per-cell regrets and the verdict block), `figures/expC10_regret.png`, `figures/expC10_curves_{tanh,gelu,swish}_{fp64,fp32}.png`.

## Results

| | fp64 | fp32 |
|---|---|---|
| scored resolved / under-resolved / unscored cells | 123 / 6 / 15 | 120 / 2 / 22 |
| resolved: median regret, 90th pct, max | $1.38$, $6.3$, $34$ | $1.36$, $7.3$, $82$ |
| under-resolved: median, 90th pct, max | $1.0$, $1.1$, $1.1$ | $1.08$, $1.15$, $1.17$ |
| **H1** | **pass** | **pass** |
| S1: $E^\star$ within $10\times$ of $E_{\min}$ | $70\%$ | $72\%$ |
| S2: rule at $3\Omega$, median / 90th | $1.43$ / $6.4$ | $1.36$ / $7.7$ |
| S2: rule at $\Omega/3$, median / 90th | $1.56$ / $8.4$ | $1.42$ / $7.3$ |
| S3: constant $A_K=\varepsilon$, median / 90th / max (resolved) | $1.40$ / $7.1$ / $34$ | $1.74$ / $6.4$ / $68$ |

**Where the rule sits.** In every resolved fp64 cell the rule's $\lambda^\star$ is on the floor, just left of the aliasing wall, as designed, and its $E^\star$ dot lies on or near the measured curve (curves figures). The under-resolved fp64 cells (the $K=32$ polynomials at $N=128$, $q=0.5$) are the only cells where the two rules separate: the rule's $\lambda^\star$ ($0.23$ tanh, $0.60$ gelu, $0.45$ swish) lands on the measured argmin ($0.19$-$0.21$, $0.56$, $0.43$) with regret $1.0$-$1.1$, while the constant is $1.4$-$2\times$ off for tanh and $13$-$33\times$ off for gelu.

**The tilted floor.** The worst regrets, in both precisions, are all low-$q$ cells ($q\le0.06$) whose measured curve has no flat floor: it keeps descending toward small $\lambda$ until the halo or the far-left conditioning wall stops it, with the argmin at $\lambda=0.04$-$0.15$ and $E_{\min}$ a factor $10$-$30$ (fp64) or up to $80$ (fp32) below the value at $\lambda^\star$. Examples: fp64 tanh $K=2$ polynomials at $N=128$ (argmin $0.13$, $E_{\min}=2\times10^{-15}$, $E(\lambda^\star)=7\times10^{-14}$, regret $34$); fp32 gelu $K=2$ at $N=64$ (argmin $0.16$, regret $82$). The constant rule pays the same regret in the same cells. The theory's $E_C$ is flat or decreasing toward small $\lambda$ when $q$ is small, so it places the floor level and cannot produce this tilt; nothing in the registered rule was allowed to react to it.

**fp32.** For every resolved target the fp32 minimum sits far to the left of both rules (tanh argmin $0.04$-$0.16$ vs $\lambda^\star=0.58$-$0.68$ and constant $0.50$; gelu and swish argmin $0.07$-$0.16$ vs $\lambda^\star=1.26$-$1.36$), and the curves show no distinct aliasing wall within the measured range for the low-$q$ targets: the fp32 error is computation-dominated everywhere, at $10^{-6}$ to $10^{-5}$, and slowly improving as $\lambda$ decreases. The fp32 wall predictions of the theory are therefore not where the fp32 optima are. The criteria pass because the tilt costs less than a factor 10 in 90% of cells. The $K=32$ polynomials are unscored at every fp32 width (minimum above $10^{-3}$).

**S1.** Tanh's predicted $E^\star$ is systematically above the measured minimum (median $+0.8$ decades), gelu's and swish's below ($-0.4$ to $-0.8$); the 10-90% range of $\log_{10}(E^\star/E_{\min})$ is $-4.1$ to $+1.1$ at fp64 (the low tail is the tilted-floor cells, where the measured minimum is much lower than any level the theory sets).

**S2.** A factor 3 misstatement of the band edge in either direction moves the median regret by at most $0.2$ and the 90th percentile by at most $2$: the rule is as insensitive to the band edge as the theory's sensitivity section says.

**Figures.**
- `expC10_regret.png` -- regret per scored cell (red: the rule; gray: the constant), split by activation and by resolved / under-resolved, one panel per precision, with the two criteria as horizontal lines. Look for: the red and gray clouds overlapping on resolved cells, the six fp64 under-resolved cells where red sits at 1 and gray (gelu) at $13$-$33$.
- `expC10_curves_{act}_{prec}.png` -- 12 targets $\times$ 4 widths, measured curve with the locked $\lambda^\star$ (red dashed), the predicted $(\lambda^\star,E^\star)$ (red dot), the rule at $3\Omega$ and $\Omega/3$ (orange triangles) and the constant (black dotted). Look for: red dots on the curve at fp64; the fp64 gelu $K=32$ panels at $N=128$ where the red line is on the sharp minimum and the black line is up the wall; the fp32 tanh panels where every curve slopes down to the left past both rules.

## Additional details

- **Why the held-out set is weak on the question that matters.** With $N\ge64$ and band edges at $\Omega/\pi\le8$ (except the $K=32$ polynomials), $q\le0.25$ in $114$ of $129$ scored fp64 cells. The test therefore mostly exercises the regime where the rule and the constant coincide by construction. This was a design choice made before the run and is reported as such; a discriminating test needs a target set with $q$ spread over $0.25$-$0.8$ at several widths.
- **Nothing was tuned.** The only inputs to the rule were the activation, $N$, $\varepsilon$ and the band edge from the generator; the smoothing, thresholds and criteria are the registered ones; the halo is the expC08 rule. The rows file and the scores file are the raw record.

## Conclusions

Pending Sam's review. As registered: H1 passes at fp64 and fp32. As data: on resolved held-out targets the two-wall rule and the activation constant are equivalent; on the six under-resolved fp64 cells the rule is on the optimum and the constant is not (strongly so for gelu); in both precisions the measured floor tilts downward toward small $\lambda$ in the low-$q$ cells by one to two decades, which neither rule models and which sets the worst regrets; in fp32 the optima of all resolved targets lie far left of both rules.

## Open questions

- The tilted floor: what sets the $\lambda$-dependence of the roundoff floor at low $q$ (it is not the coefficient amplification, which is flat there), and why is it steeper in fp32?
- A discriminating held-out set with $q$ between $0.25$ and $0.8$, to test the under-resolved prediction on more than six cells.
- The fp32 regime as its own problem: the measured optimum is at $\lambda\approx0.05$-$0.15$ for resolved targets, where the kernel reach exceeds the halo; the halo rule and the left wall need their own fp32 study before any $\lambda$ rule is claimed there.
