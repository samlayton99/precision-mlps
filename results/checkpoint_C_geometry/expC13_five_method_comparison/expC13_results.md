# expC13 -- Five constructions with every construction and inference operation at p bits

**Status: the tap-out comparison (QUILLS against ChebNet) was reviewed and approved by Sam on 2026-09-25; the parameter-budget sweep to $p=128$ is a draft pending review.**

## TL;DR

- At every precision from 21 to 53 bits, on the three harder targets, QUILLS reaches its precision floor with fewer hidden neurons (summed over layers) than ChebNet, the only other construction that reaches its floor within 1024 neurons. It needs 1.7 to 2.4 times fewer on the chirp, 2 to 3.4 on $\sin4\pi x$ and 4 to 5.7 on Runge; on $e^x$ it is at its floor by 32 neurons while ChebNet needs up to 64. The same ordering holds without any tap-out definition: at every $p$ from 15 to 53, any error at least 4 times above QUILLS' floor is reached by QUILLS with no more neurons than ChebNet.
- QUILLS does this with one hidden tanh layer. At the FP64 tap-out ChebNet uses 4 to 8 hidden ReQU layers and 4 to 14 times as many parameters.
- ChebNet's floor sits lower, by a median of 2.4 bits on Runge, 3.8 on $\sin4\pi x$ and 4.5 on the chirp; on $e^x$ the two are within about a bit of each other. Both floors follow the unit roundoff, $E^*\approx C\,2^{-p}$.
- The other three constructions never reach a precision-limited floor within 1024 neurons from $p=11$ to $23$ on, depending on target and method. The staircase and Costarelli-Spigler are limited by width (errors of $10^{-6}$ to $10^{-2}$ that more bits do not change). Mhaskar cannot use width at these precisions: its best error comes at 32 neurons on $e^x$ and Runge, and it fails on $\sin4\pi x$ and the chirp.

## Question

The comparison table gives QUILLS and ChebNet the same asymptotic width, $O(\log 1/\varepsilon)$, so it cannot separate them. This experiment measures the constants. At a fixed precision $p$, how many hidden neurons does each construction need before its error stops improving (the tap-out width, a rate), and how low is the error there (the floor, an intercept)? It uses a protocol in which every value in construction and inference has $p$ significant bits, and every method gets the same exponent range and the same target information. The same protocol also ran a first sweep under a common parameter budget to $p=128$.

## Experiment design

The spec (SPEC.md, see Code & data) defines the protocol; this section summarizes it.

**Precision and arithmetic.**
- $p$ counts significant bits, including the implicit one (binary64 has $p=53$).
- One format serves every method and every $p$: $(p,e_{\min},e_{\max})=(p,-958,959)$, IEEE-style with gradual underflow and overflow to infinity.
- All model arithmetic is [pfloat](https://github.com/samlayton99/pfloat) 0.1:
  - each $+,-,\times,\div,\sqrt{\ }$ is correctly rounded (round to nearest, ties to even);
  - negation, $|\cdot|$, comparison, $\max(z,0)$ and power-of-two scaling are exact;
  - there is no fused multiply-add and no compensated or extra-precise summation.
- For $p\le53$ pfloat emulates the format exactly on binary64; above 53 it runs on MPFR records, so the sweep continues past 53 with the same code.
- tanh is pfloat's $\tanh_p$, built from those operations only (within 3 ulp). It is used in construction and inference alike, for every method.
- Least squares is pfloat's `lstsq`: reference LAPACK `DGELSS` ported operation by operation, bit-identical to netlib in binary32 and binary64.

**Constants and inputs.**
- Universal constants are rounded once into the format: Chebyshev node values $\cos(\pi(k+\frac12)/M)$, $\ln 2/2$, exact rational finite-difference weights, tanh's own constants, and LAPACK's machine parameters.
- External hyperparameters are exact numbers rounded once: QUILLS' bandwidth $\lambda$, Mhaskar's step $h$, and the steepness factors.
- Everything computed from these, including every target-dependent coefficient, is computed at $p$ bits.
- Each method chooses its sampling locations (format values) and receives the target there, correctly rounded once: $Q(f(\tilde x))$. The value $f$ is evaluated by mpmath at $p+96$ bits, with a Ziv check that both ends of its uncertainty interval round the same way.

**Networks.**
- One hidden tanh layer (QUILLS, Mhaskar, staircase, Costarelli-Spigler): $y=c+\sum_j a_j\tanh(w_jx+b_j)$, with $z_j=\mathrm{add}(\mathrm{mul}(w_j,x),b_j)$ and the readout summed sequentially from $c$.
- ChebNet: deep ReQU, $\sigma(z)=\max(z,0)^2$, with every pre-activation summed sequentially from its bias.

**Targets** on $[-1,1]$: $e^x$ (entire), $\sin(4\pi x)$ (entire, oscillatory), $1/(1+25x^2)$ (poles at $\pm i/5$) and the chirp $\sin(8\pi(x+1)^2)$ of expC09 to expC12.

**Measurement.**
- **Validation.** Candidates are scored on 1021 validation midpoints $-1+(2i+1)/1021$. The count is prime, so the points are not commensurate with any method's grid (see Additional details).
- **Reporting.** Winners are evaluated on the 8001-point reporting grid $-1+2i/8000$.
- **Inputs.** Networks are evaluated at $Q(x)$ and compared with $f(x)$ at the exact point, so input rounding counts as error equally for every method.
- **Error.** Relative $L^2$ error $\|y-f\|_2/\|f\|_2$ is computed in 320-bit arithmetic against $f$ at 384 bits. Every reported value is recomputed at 640 bits, and the two agree to all printed digits in every row. The meter reads outputs only.
- **Replay.** Each winner is saved exactly and replayed; the replay must be bit-for-bit identical.

**Methods.** Each is the construction in the paper's comparison appendix, with the choices its source leaves free selected on validation. Improvements that keep the method's defining feature are validated options, labelled as the literature's or ours. The faithful configuration is also reported on its own.

- **QUILLS.**
  - Uniform centers with a halo, and $\gamma=\lambda/h$ with $\lambda$ from expC09's refined rule at $e_{\rm tol}=2^{1-p}$.
  - Readout by $p$-bit `DGELSS` (RCOND $2^{1-p}$) on 4801 points.
  - This removes expC12's FP64-SVD exception.
- **Mhaskar** (1996).
  - Construction: a $p$-bit discrete Chebyshev projection from 4096 nodes, then conversion to monomials. Each $x^k$ is realized by finite differences of $\tanh(b_0+wx)$ in the slope $w$, with $b_0=\ln 2/2$ and $c_k=\tanh^{(k)}(b_0)/k!$ from the $y'=1-y^2$ recurrence.
  - Faithful (`mhaskar_appendix`): per-monomial centered stencils, $2d+1$ neurons for degree $d$. This is Mhaskar's Lemma 3.2 (eq. 3.19), which the appendix reproduces; it is bit-identical to expC12's kernels.
  - Option (our variant, not in the paper): one slope set $b_0+jh$, $j=-m..m$, realizes every monomial up to degree $2m$ from all $2m+1$ neurons by maximal-order differences (exact Lagrange weights).
  - Degree $\le128$ and step $h\in[10^{-4},4]$ (65 values) are selected on validation.
- **Classical staircase.**
  - Network: $f(-1)+\sum_j[f(x_j)-f(x_{j-1})]S(\kappa N(x-t_j))$ with $S=(1+\tanh)/2$; the offset is formed by exact telescoping.
  - Faithful (`staircase_classical`): $\kappa=1$, jumps at the samples.
  - Options:
    - $\kappa\in\{\frac18,\dots,32\}$;
    - jumps at cell midpoints;
    - with midpoints, a halo of 8 samples per side, extrapolated from in-domain samples by quadratic Lagrange weights.
- **Costarelli-Spigler** (2015, Theorem 5.4, sampled).
  - Network: $\sum_{|k|\le K}F(u_k)\psi(wx-k)$ with $\psi(t)=\sigma(t+1)-\sigma(t)$ and a linear-taper extension, merged into $2K+2$ neurons with offset 0 by telescoping.
  - Faithful (`costarelli_appendix`): $u_k=k/w$, $K=2w$.
  - Option: truncation $K=w+\lceil p\ln2/2\rceil+1$, the point where $\psi$'s tail falls below unit roundoff (the paper allows any $K>w$).
  - Option: centred samples $u_k=(k-\frac12)/w$, which remove the half-cell shift behind the $O(1/w)$ interior rate.
- **ChebNet** (Tang-Li-Yu, Theorem 3, with the layer algorithm of Li-Tang-Yu 2020).
  - Construction: the same $p$-bit projection, the hierarchical coefficient split, and the exact ReQU identity, product and square primitives, collapsed into a standard MLP. Degrees not of the form $2^L-1$ use a pruned tree.
  - A degree-$n$ network has $\lceil\log_2(n+1)\rceil$ hidden layers.
  - Faithful (`chebnet_paper`): as published.
  - Option: exact power-of-two normalization of the coefficients.

**The tap-out analysis** (the headline).
- **Grids.**
  - Hidden neurons: $N\in\{32\cdot2^{k/4}:k=0,\dots,20\}$, i.e. 32 to 1024, four per doubling.
  - Precision: $p=9,11,\dots,51$ in the $p$-bit format, and $p=53$ in native binary64.
- **Size.** $N$ counts hidden neurons summed over all hidden layers.
- **QUILLS.**
  - Halo $\min(24,\lfloor W/4\rfloor)$, so the interior stays populated at small $W$, and $\lambda$ from the rule at each $(W,p)$.
  - At $p\le51$: every grid width up to 512, plus the budget sweep's 96, 192, 384, 768 and 1024. Widths 609, 724 and 861 were not built, which can only cost QUILLS.
  - At $p=53$: every grid width, with numpy tanh and scipy `lstsq` (`gelsd`, cond $2^{-52}$).
- **ChebNet.**
  - 52 degrees from 1 to 250 are built (every degree to 16, then steps of 4 to 16; 250 is the largest within 1024 neurons), with and without normalization.
  - At $p=53$: coefficients from a DCT at 4096 Chebyshev nodes, weights formed in binary64, and a dense numpy forward pass.
- **The other three.** They enter as swept below, with no extra tuning. Mhaskar's candidates stop at 257 neurons because of the parameter budget; the staircase and Costarelli-Spigler reach 1024. At $p=53$ their values come from the 53-bit sweep: that is the same correctly rounded binary64 arithmetic as native FP64, differing only in tanh.
- **Definitions.**
  - $E(N,p)$: the validation error of the validation-best candidate with at most $N$ neurons, which makes it non-increasing in $N$.
  - Floor: $E^*(p)=E(1024,p)$.
  - Tap-out width: $N^*(p)$, the smallest $N$ with $E(N,p)\le10\,E^*(p)$.
  - Width-limited (no tap-out within the cap): two more bits improve $E(1024,p)$ by less than $2\times$, while the last half-doubling of width ($724\to1024$) still improves it by at least $1.5\times$. At $p=53$ there are no two more bits, so the $51\to53$ verdict is reused.
  - No tap-out is reported where $E^*>0.1$, since the method does not approximate the target there.
- **Definition-free check.** For every $p$ and every error level $\varepsilon$ between $4E^*_{\rm QUILLS}(p)$ and $0.1$ (80 log-spaced levels), compare the smallest $N$ at which each method reaches $\varepsilon$.

**The parameter-budget sweep** (to $p=128$).
- Size is the number of nonzero weights and biases. The budget is 3073, the size of a 1024-neuron one-hidden-layer network.
- Each method's validation-best candidate within the budget is reported on the reporting grid.
- QUILLS' width is selected from $W\in\{64,\dots,1024\}$ with halo 24; `quills_w1024` is the fixed expC11 configuration.
- ChebNet's degree is selected within the budget ($n\le95$). A sensitivity arm, `chebnet_neurons`, uses a budget of 1024 hidden neurons instead; this is the ChebNet of the tap-out analysis.
- $p=8..53$ in steps of 1, then $56,64,72,80,96,113,128$.

**Checks** (77 tests in the two expC13 test files, plus pfloat's own suite for the arithmetic).
- The oracle equals MPFR's correct rounding of $f$. This includes a sample within $2^{-230}$ of a rounding midpoint, resolved by re-evaluation at higher precision. The meter agrees at 320 and 640 bits.
- The pfloat QUILLS path (expC11's model) reproduces `src/precision`, an independent C implementation of the model and the `DGELSS` port, bit for bit at $p=20,32,53$.
- The Mhaskar port reproduces expC12's MPFR kernels bit for bit (projection, monomials, Taylor recurrence, stencils) at $p=16,24,53$.
- The tanh-network forward pass equals an independent gmpy2 replay written from the spec, and saved models replay bit for bit. An audit replays all 396 selected models and passes.
- At 200 bits each network equals its defining formula:
  - ChebNet equals $\sum c_jT_j$ to $2^{-180}$, with the paper's hierarchical transform $S_m$ and the derived neuron and parameter counts;
  - the staircase and Costarelli-Spigler networks equal their series (Lemma 5.1's partition of unity is checked);
  - the halo extrapolation is exact for quadratics;
  - both Mhaskar grids converge to their polynomial as $h\to0$.
- At 53 bits the faithful staircase and Costarelli-Spigler converge at rate $1/N$ ($0.9<$ order $<1.1$).

**Code & data.**
- Experiment folder `experiments/expC13_five_method_comparison/`:
  - `SPEC.md` (normative);
  - runner `run.py` (`--sweep` for $p=8..53$, `--extended` for $p>53$);
  - shared protocol `common.py`, `selection.py`;
  - constructions `methods/{quills,mhaskar,chebyshev,staircase,costarelli,chebnet}.py`;
  - tap-out data `tradeoff_data.py` (`--quills-pbit` for the widths the sweep lacks, `--fp64` for $p=53$);
  - tap-out figures and table `tapout_plot.py`;
  - budget-sweep figures `plot.py`, `size_curves.py`;
  - checks `audit.py`, `tanh_policy_check.py`, and the fixed-bandwidth QUILLS test `quills_min_test.py` / `quills_min_analysis.py`.
- Tests: `tests/test_expC13_protocol.py`, `tests/test_expC13_constructions.py`.
- Data in `results/checkpoint_C_geometry/expC13_five_method_comparison/data/`:
  - `config.json` (source hashes per run);
  - `summary.jsonl` (one row per target, method and $p$);
  - `candidates/*.json` (every candidate's validation error);
  - `tradeoff_rows.jsonl` (the tap-out analysis' extra QUILLS widths and its FP64 rows);
  - `tapout.json` (status, $N^*$ and $E^*$ per target, method and $p$);
  - `models/*.npz` (exact winners);
  - `tanh_policy.json`;
  - `quills_min_test.jsonl`, `quills_min_fit.json`.
- Figures in `figures/`: `tapout_curves.png`, `tapout_summary.png`, `precision.png`, `size_p53.png`, `quills_min_structure.png`, and `detail/`.

## Results

### Tap-out width: QUILLS needs fewer neurons at every precision

At a fixed $p$ each method's error falls with width until it meets its precision floor, then stops. QUILLS meets its floor at a smaller width than ChebNet at every $p$ on the three harder targets, and on $e^x$ at no larger width (both are at their floor by 32 neurons up to $p=31$):

| Target | $p=21$ | $p=33$ | $p=45$ | $p=53$ (FP64) |
|---|---|---|---|---|
| $e^x$ | $\le32$ / $\le32$ | $\le32$ / 45 | $\le32$ / 54 | $\le32$ / 64 |
| $\sin4\pi x$ | 38 / 108 | 54 / 128 | 64 / 181 | 91 / 181 |
| Runge | 54 / 215 | 108 / 431 | 128 / 609 | 152 / 861 |
| chirp | 181 / 362 | 181 / 431 | 256 / 512 | 304 / 512 |

*Tap-out width $N^*$, QUILLS / ChebNet, in hidden neurons.*

Over all $p$ from 21 to 53, ChebNet's tap-out width is 1.7 to 2.4 times QUILLS' on the chirp, 2 to 3.4 on $\sin4\pi x$ and 4 to 5.7 on Runge. Both grow roughly linearly in $p$, as $O(\log1/\varepsilon)$ width predicts; ChebNet's slope on Runge is about 17 neurons per bit against QUILLS' 3. The ratios are at the resolution of the width grid (a factor $2^{1/4}$).

The ordering does not depend on the tap-out definition down to near QUILLS' floor. At every $p$ from 15 to 53 and every error level at least 4 times above QUILLS' floor, QUILLS reaches the level with no more neurons than ChebNet (no exceptions on any target). Within a factor 4 of QUILLS' floor, ChebNet often gets there first: QUILLS gains that last factor only slowly with width (Additional details), while ChebNet's floor is lower.

ChebNet's neurons also come in more layers and with more parameters per neuron:

| Target | QUILLS: neurons, parameters, hidden layers | ChebNet: neurons, parameters, hidden layers (degree) |
|---|---|---|
| $e^x$ | $\le32$, 97, 1 | 62, 418, 4 (13) |
| $\sin4\pi x$ | 91, 274, 1 | 178, 1310, 6 (40) |
| Runge | 152, 457, 1 | 786, 6166, 8 (192) |
| chirp | 304, 913, 1 | 496, 3856, 7 (120) |

*The network at the FP64 tap-out ($p=53$). ChebNet's network is the best one with at most $N^*$ neurons, so its neuron count can be below $N^*$.*

### Floor: ChebNet rests lower

Both methods' floors run parallel to the unit roundoff. Measured as $\log_2(E^*/2^{-p})$, the bits of error above unit roundoff (median over $p\ge21$, QUILLS / ChebNet):

- $e^x$: 3.6 / 4.5;
- $\sin4\pi x$: 7.6 / 4.1;
- Runge: 6.6 / 4.1;
- chirp: 10.0 / 5.5.

So ChebNet's floor is lower by 2 to 4.5 bits, a factor of 4 to 20 in error, on the three harder targets. At FP64 the floors are:

- $e^x$: $2.0\times10^{-15}$ / $2.7\times10^{-16}$;
- $\sin4\pi x$: $6.7\times10^{-15}$ / $1.2\times10^{-15}$;
- Runge: $1.9\times10^{-14}$ / $9.0\times10^{-16}$;
- chirp: $1.0\times10^{-13}$ / $5.1\times10^{-15}$.

QUILLS' constant is plausibly its least-squares solver's (Additional details).

### The other three constructions

- **Staircase and Costarelli-Spigler.** At low precision they tap out early, but from $p=11$ (chirp) to $23$ ($e^x$) on they are width-limited. Their error at 1024 neurons is approximation error, $2\times10^{-6}$ to $10^{-2}$ depending on target and method, and more bits do not change it.
- **Mhaskar.** It cannot use width at these precisions. Higher degrees need more bits than $p\le53$ provides, so its best error comes at 32 neurons on $e^x$ ($5\times10^{-12}$ at $p=53$) and Runge ($2.5\times10^{-2}$), and it fails on $\sin4\pi x$ and the chirp (error about 1).

### The parameter-budget sweep to $p=128$

- **Low precision ($p\lesssim16$ to $20$).** The tuned staircase and Costarelli-Spigler are the most accurate constructions on every target. Their readouts are sample differences whose total absolute size is about the target's variation, so rounding is not amplified. The first $p$ at which QUILLS or ChebNet is best is 21 ($e^x$), 16 ($\sin4\pi x$), 19 (Runge) and 14 (chirp).
- **The sampled-difference plateau.** Above that crossover the staircase and Costarelli-Spigler stop improving, because their error is approximation error at the budget. For the tuned staircase it is:
  - $1.8\times10^{-6}$ on $e^x$;
  - $2.8\times10^{-4}$ on $\sin4\pi x$;
  - $5.3\times10^{-5}$ on Runge;
  - $3.2\times10^{-3}$ on the chirp.

  Costarelli-Spigler is 1.1 to 35 times worse. The faithful forms are first order and sit another 13 to 530 times higher.
- **QUILLS and ChebNet follow the unit roundoff to $p=128$** wherever their budget allows the accuracy. Under the parameter budget ChebNet's degree is capped at 95 (about 32 parameters per degree), so its Runge and chirp errors stop at the truncation error of degree 92: $8.7\times10^{-9}$ and $4.7\times10^{-7}$ at every $p\ge34$. QUILLS has no such floor within the sweep: $1.9\times10^{-37}$ (Runge) and $1.4\times10^{-35}$ (chirp) at $p=128$.
- **Mhaskar needs many more bits.**
  - With the common grid, $e^x$ goes from $5.2\times10^{-12}$ at $p=53$ to $9.9\times10^{-28}$ at $p=128$. That is one decimal digit per 4.8 bits, against one per 3.3 bits for QUILLS and ChebNet.
  - $\sin4\pi x$ needs $p\approx64$ to get below $0.4$.
  - Runge and the chirp stay at $2.3\times10^{-2}$ and $0.93$ at every $p\le128$.
  - The appendix stencils are far worse: $3.2\times10^{-6}$ on $e^x$ at $p=53$.
- **Selections.** ChebNet's normalization option is selected in 38 of 212 cases and changes the error by at most 5%, so its curves coincide with the published construction's.

### Figures

- `tapout_curves.png` (the headline):
  - Layout: a $4\times4$ grid, with columns for the targets and rows for $p=21,33,45$ and 53 (FP64). Each panel plots the best error with at most $N$ neurons against $N$ (log-2 axis, 32 to 1024), one line per method; the grey dotted line is the unit roundoff $2^{-p}$, and a large open circle marks each tap-out $N^*$.
  - Look for:
    - QUILLS' curve falling and flattening well to the left of ChebNet's in every panel;
    - ChebNet's flat shelf sitting slightly below QUILLS';
    - the staircase and Costarelli-Spigler still sloping down at 1024;
    - Mhaskar flat from 32 neurons.
- `tapout_summary.png`:
  - Layout: two rows by four targets, with bits $p$ on the x-axis.
  - Top row: tap-out width $N^*$ on a linear axis, so the slope is neurons per bit. Open markers in the shaded band above 1024 are width-limited points, and the shaded strip below 32 means "at its floor by the smallest width tested."
  - Bottom row: the floor $E^*$, with the unit roundoff dotted. Filled markers are floors; open markers are errors that are not floors (width-limited, or no useful fit).
  - Look for QUILLS below ChebNet in the top row at every $p$, and both parallel to $2^{-p}$ in the bottom row, ChebNet slightly lower.
- `precision.png`:
  - Layout: four panels, relative $L^2$ error on the reporting grid against $p$ (8 to 128), one line per method: its validation-selected best configuration within the 3073-parameter budget. Grey dotted: the unit roundoff. The light vertical line marks $p=53$.
  - Look for:
    - QUILLS and ChebNet running parallel to the unit roundoff;
    - the staircase and Costarelli-Spigler going flat;
    - ChebNet's degree-cap floors on Runge and the chirp;
    - Mhaskar's shallower slope.
- `size_p53.png`:
  - Layout: the same four targets at $p=53$, error on the reporting grid against the size allowed (nonzero weights and biases). Where each line meets the dotted budget line equals `precision.png` at $p=53$ (`size_curves.py` asserts it). Dashed orange: ChebNet allowed more parameters than the budget.
  - Look for how few parameters QUILLS needs on Runge and the chirp.
- `quills_min_structure.png`:
  - Layout: QUILLS alone on the chirp and Runge at widths 96 to 384, error against $p$, one line per width. Top row: the bandwidth rule's $\lambda$. Bottom row: the best of four $\lambda$ values.
  - Look for each width following the unit roundoff until it flattens at its own approximation floor (the min structure behind the tap-out).
- `detail/`: the earlier, busier budget-sweep figures (all published and improved variants; selected sizes and largest weights; validation error against size).

## Additional details

- **Why the tap-out uses $10\times$, and QUILLS' creep.**
  - The min structure: at a fixed bandwidth, each QUILLS width follows the unit roundoff in $p$ until it meets that width's approximation floor (`quills_min_structure.png`).
  - The creep: past its knee in width, QUILLS' error keeps falling slowly. On the chirp at $p=45$ it goes from $5\times10^{-10}$ at 181 neurons to $2.5\times10^{-11}$ at 1024, about 1.7 bits per doubling. ChebNet's plateau is a flat shelf.
  - Effect of a $2\times$ rule instead of $10\times$ ("within one bit of the best at 1024"):
    - ChebNet's $N^*$ moves by at most one grid step on the three harder targets;
    - QUILLS' $N^*$ moves right by 3 to 4 steps on the chirp (to 304 to 512, comparable to ChebNet's 431 to 512);
    - QUILLS' $N^*$ on $\sin4\pi x$ at $p\le29$ exceeds ChebNet's.
  - The $10\times$ rule measures where the steep fall ends, which is what "starts to tap out" means. The definition-free check shows the ordering holds for every error above 4 times QUILLS' floor.
- **Validation grid alignment.** A first validation grid of 1024 dyadic midpoints coincided exactly with the midpoint-jump locations of the staircase at $N=1024$. At those points each sigmoid equals $\frac12$ and the network returns the average of neighboring samples, so the grid scored the midpoint staircase at $4.8\times10^{-7}$ on $e^x$ against its true $2.4\times10^{-5}$. The validation grid is therefore the midpoints of 1021 cells (a prime count), and a check confirms that neither grid has fewer than 20 distinct phases against any method's sample grid.
- **Where the method details come from.**
  - **ChebNet.** Its algorithm is taken from arXiv:1911.05467 v3 (v1 has errors fixed in v2, including a wrong ordering in the transform $S_m$) and its companion Li-Tang-Yu (CiCP 2020). The network sizes were derived and checked against the paper's $n=3$ network.
  - **Costarelli-Spigler (2015)** was read in full. The linear taper, the sampled coefficients and the single scale parameter are the appendix's adaptations.
  - **Mhaskar (1996)** was read in full.
    - Lemma 3.2 (eq. 3.19) realizes each monomial by its own centered difference in the slope at one bias, exactly the appendix's construction. Three later papers describe it as a single maximal-order grid; that description does not match the paper.
    - Its Theorem 2.3 uses Chebyshev interpolation. Using that instead of the 4096-node projection changes the faithful errors by at most 8%.
  - **The staircase's classical form** is Costarelli-Spigler's $G_N$ (Anal. Theory Appl. 29 (2013), Theorem 2.1, read in full). Its logistic slope condition corresponds to $\kappa\approx1.7$ against the appendix's $\kappa=1$. The classical staircase's error at $N=1024$ is $9.4$ to $9.8\times10^{-4}$ on $e^x$ for every $\kappa$ from $\frac18$ to 8, so this does not matter.
- **A better-conditioned Mhaskar exists.** Realizing all monomials to maximal order from the same $2m+1$ slopes allows about 10 times larger steps and far smaller readouts: $5.2\times10^{-12}$ versus $3.2\times10^{-6}$ on $e^x$ at $p=53$. This is our variant, so the table's Mhaskar row correctly describes Mhaskar's construction. It still cannot use width at $p\le53$ (the tap-out analysis uses it).
- **ChebNet and the size unit.**
  - The tap-out analysis counts hidden neurons over all layers.
  - Counting parameters would widen QUILLS' lead: ChebNet spends about 8 nonzero parameters per neuron (32 per polynomial degree), against 3 per neuron for a one-hidden-layer network.
  - Counting only ChebNet's widest layer (about half its neurons: 30, 82, 386 and 242 at the FP64 tap-outs) would make the two comparable on $e^x$, $\sin4\pi x$ and the chirp (0.8 to 0.9 of QUILLS' width), and leave QUILLS 2.5 times narrower on Runge.
  - Under the 3073-parameter budget ChebNet's degree is at most 95, which is where the budget sweep's Runge and chirp floors come from.
- **QUILLS' constant is plausibly the solver's.** QUILLS' $C$ (up to about $10^3$ on the chirp) matches the $p$-bit `DGELSS` constant already seen in expC11. There, on the same matrices in native FP32 and FP64, the pivoted-QR solver `DGELSY` was 17 to 49 times more accurate than the SVD solvers. A gain of that size is comparable to the 4 to 20 times floor gap to ChebNet; it has not been measured at $p$ bits.
- **The tanh policy.** Rebuilding and evaluating QUILLS and the tuned staircase with a correctly rounded tanh (MPFR, same format) instead of $\tanh_p$ changes:
  - QUILLS' error by factors 0.89 to 1.26 (not systematically better);
  - the staircase's error by at most 0.4%.

  The check covers $p=16,24,53$ on $e^x$ and the chirp.
- **A hard rounding case.** At $p=113$ the training point $Q(1/5)$ puts $1/(1+25x^2)$ within about $2^{-230}$ of a rounding midpoint. The oracle's interval check caught it; the oracle now re-evaluates such entries at doubled precision (Ziv's strategy), and a regression test pins the case.
- **Cost.**
  - The budget sweep is about 25 CPU-hours.
  - The tap-out analysis adds 1740 runs: the missing QUILLS widths at $p\le51$ and every FP64 row.

## Conclusions

At every precision up to FP64, QUILLS reaches its precision floor with fewer neurons than ChebNet (1.7 to 5.7 times fewer on the three harder targets), with one hidden layer against ChebNet's 4 to 8. ChebNet's floor rests 2 to 4.5 bits lower. QUILLS is also the simpler construction: one layer and one least-squares solve. It is not tied to tanh (expC07 gives the bandwidth rule for GELU and swish), although this comparison used tanh only. The staircase, Costarelli-Spigler and Mhaskar constructions do not reach a precision-limited floor within 1024 neurons beyond low precision.

## Open questions

- Port `DGELSY` (pivoted QR) to $p$ bits and measure whether it closes QUILLS' 2 to 4.5-bit floor gap to ChebNet.
- Whether the table should mention that a maximal-order variant of Mhaskar's construction is far better conditioned (it keeps one hidden layer and small slopes).
