# Pre-registration: lambda anchor rule v1 on ten fresh targets

Written 2026-09-10T00:36:23.532732+00:00 by `anchor_v1_prereg.py --predict`, BEFORE any least-squares solve of this test.

## Hashes
- `anchor_v1_prereg.py` (rule call, targets, arms, sweeps, scoring, criteria): `4796a5103c9646468391fc54e045f5ce537719d7c6c2279229fdee6076afbb63`
- `anchor_rule.py` (the frozen rule implementation `lambda_anchor`, unchanged since 2026-09-09): `71ba5c51d274c5346c5b53f34996c0afc067fc0c06812bc36b74896218e54e2e`
- `data/anchor_v1_prereg_predictions.json` (every lambda for every arm and cell): `f449add0f89445dbac47f7f36b862d42e64fac337def5670a7baf210c8caf9ec`

## Rule under test
$\lambda_{v1}=\sup\{\lambda\in G: B\,\mathcal R_{K,r}(\lambda,N,\bar\omega)\le 2^{1-p}\}$, $G$ the 600-point log grid on $[0.03,1.5]$, $B=\sum|b_j|$ in raw target units, $\bar\omega=\sum|b_j||\omega_j|/B$, first-pair ratio with denominator $\min\{1,H(\theta/\lambda)\}$ and factor $1/(1-\rho_K)$, exactly as in `anchor_rule_v1.md`.

## Arms
v1 (primary), const ($H(2\pi/\lambda)=2^{1-p}$, baseline), v1n ($B\to B/\|f\|_\infty$, declared amplitude fix), max ($\bar\omega\to\omega_{\max}$, line targets only, diagnostic).

## Targets (none used before) and their rule inputs
| target | kind | B | $\bar\omega/\pi$ | $\omega_{\max}/\pi$ | $\|f\|_\infty$ | $B/\|f\|_\infty$ |
|---|---|---:|---:|---:|---:|---:|
| $3\sin3\pi x+0.3\sin9\pi x$ | lines | 3.3 | 3.545 | 9 | 2.7 | 1.22 |
| $0.01(\sin2\pi x+\sin4\pi x)$ | lines | 0.02 | 3.000 | 4 | 0.0176 | 1.14 |
| $\cos\pi x+0.1\cos7\pi x+0.01\cos13\pi x+0.001\cos19\pi x$ | lines | 1.111 | 1.664 | 19 | 1.111 | 1 |
| $\sin5\pi x+\sin7\pi x+\sin11\pi x$ | lines | 3 | 7.667 | 11 | 2.655 | 1.13 |
| $1/(1+9x^2)$ | line | 1 | 0.955 | nan | 1 | 1 |
| $e^{-8x^2}$ | line | 1 | 1.016 | nan | 1 | 1 |
| $1/(2+\cos3\pi x)$ | periodic | 1 | 1.732 | nan | 1 | 1 |
| $1/((x+0.3)^2+0.09)$ | line | 11.11 | 1.061 | nan | 11.11 | 1 |
| $\mathrm{sech}(6x)$ | line | 1 | 1.418 | nan | 1 | 1 |
| $x\,e^{-4x^2}$ | line | 0.2821 | 1.128 | nan | 0.2144 | 1.32 |

Numerical-spectrum routine checked on known inputs: runge25 B=1.000000 (1), omega_bar=5.00000 (5); gauss20 B=1.000000 (1), omega_bar=5.04626 (5.04627); expsin B=2.718282 (e), omega_bar=6.3492 (6.349).

## Sweeps
Width: native fp64, $N\in\{48,96,192,384\}$. Precision: $N=96$, $p\in\{11,16,24,32,40,48,53\}$ emulated (inputs, coefficients, partial sums rounded to $p$ bits; SVD cutoff $2^{1-p}$; solver internals fp64) plus native float32. Halo 32, 40 $\lambda$ points on $[0.03,1.5]$, rel $L_2$ on 4001 points. Four activations: tanh ($r=1$), gelu ($r=2$), swish ($r=2$), gaussian $e^{-x^2}$ ($r=0$).

Status counts for v1 before the run: upper_search_limit width 0, precision 50; undefined ($\theta\ge\pi$ or no feasible point) width 0, precision 0. Upper-limit cells are scored but flagged; they carry no threshold claim.

## Scoring and criteria (fixed now)
- $S$ = 5-point running median of $\log$ error; $E_{\min}=\min S$; floor = median of $S$ over $\{S\le5E_{\min}\}$.
- Cell excluded if $E_{\min}>10^{-3}$ or the arm's $\lambda$ is undefined.
- regret(arm) $=S(\lambda_{\rm arm})/E_{\min}$, log-interpolated on the same smoothed curve.
- on_wall(arm): exact fiber floor at $\lambda_{\rm arm}$ (exact line sum for line targets, expC09 whole-line projection otherwise) divided by the floor $>2$.
- **C1** median regret $\le3$; **C2** 90th percentile regret $\le30$; **C3** on-wall fraction $\le0.10$; each per activation and per sweep. **Verdict per activation: PASS iff C1 to C3 hold in both sweeps.**
- **C4** v1 vs const: per-cell $\log(\text{regret}_{v1}/\text{regret}_{\rm const})$ pooled over both sweeps per activation; median ratio and two-sided exact sign test on nonzero differences. "v1 better" iff median $<0.8$ and $p<0.05$; "worse" iff median $>1.25$ and $p<0.05$; else indistinguishable.
- **C5** amplitude: on cells whose raw $B$ is outside $[0.5,2]$ (four targets: $B=3.3,\ 0.02,\ 3,\ 11.1$), median regret of v1n against v1. (Amended before the run: the first draft keyed on $B/\|f\|_\infty$, which is within $[0.5,2]$ for every target and selected nothing.)

## Priors (stated now)
tanh, gelu, swish: PASS. gaussian: fails C3 (on-wall fraction about 0.15 in the 2026-09-09 mean-frequency run). C4: indistinguishable for $r\ge1$; if anything v1 better on gaussian. C5: v1n no worse than v1.

No criterion, arm, target, width or precision will be changed after `--run`. Results: `anchor_v1_prereg_results.md`.
