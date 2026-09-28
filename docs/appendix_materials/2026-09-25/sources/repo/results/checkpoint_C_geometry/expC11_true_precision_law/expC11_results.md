# expC11 -- The precision law with the model and the solve both at p bits

**Status: draft, pending Sam's review.**

## TL;DR

- With every operation of the model and of the least-squares solve rounded to $p$ bits, the chirp error follows $E\approx C\,2^{-p}$ with $C\approx1.1\times10^{3}$ from $p=16$ to $53$; the intercept fit gives $C=1386$, against $C=19.5$ when only the forward pass was at $p$ bits.
- There is no separate high-precision floor: at $p=53$ the error, $1.04\times10^{-13}$, sits on the same line. The plateau near $10^{-13}$ in the forward-only panel is the FP64 solve's own $C\,2^{-53}$.
- The solve is reference LAPACK `DGELSS` ported operation by operation; in exact FP32 and FP64 it reproduces netlib `SGELSS`/`DGELSS` bit for bit on the full $4801\times1025$ problem.
- Below $p\approx13$ the $W=1024$ solve breaks down (error $0.25$ to $1.5$, rank collapsing to 100 at $p=8$).

## Question

If nothing in the model or the solve is allowed more than $p$ significand bits, does the error still fall one bit per bit of precision, and with what constant?

## Experiment design

**Target and geometry** (as in the expC09 panel): $f(x)=\sin(8\pi(x+1)^2)$ on $[-1,1]$, total width $W=1024$ with 24 halo centers per side ($N=975$ interior intervals, $h=2/N$), centers $c_j=-1+jh$, $j=-24,\dots,999$. The bandwidth for each $p$ is the expC09 refined-rule value $\lambda(p)$ (tolerance $2^{1-p}$, $\bar\omega=16\pi$), chosen offline and rounded into the format. Fitting uses 4,801 equispaced points; the error uses 8,001.

**Arithmetic.** A format $(p,e_{\min},e_{\max})$ has a $p$-bit significand and gradual underflow. Each $+,-,\times,\div,\sqrt{\ }$ returns the correctly rounded result in the format (round to nearest, ties to even), with no fused multiply-add. Constants are stored after one correct rounding. The sweep uses $(p,-958,959)$ for $p=8,\dots,53$: binary64's exponent field pulled in 64 binades at each end so the binary64 carrier of the emulator holds every value exactly. At $p=53$ this is bit-identical to IEEE binary64 (checked).

**Model.** $h=\mathrm{div}(2,N)$, $c_j=\mathrm{add}(-1,\mathrm{mul}(j,h))$, $\gamma=\mathrm{div}(\lambda,h)$, features $\Phi_{ij}=\tanh_p(\mathrm{mul}(\gamma,\mathrm{sub}(x_i,c_j)))$, readout $s=w_{\text{bias}}$, then $s=\mathrm{add}(s,\mathrm{mul}(\Phi_{ij},w_j))$ in order. $\tanh_p$ is built from the five operations only (no library tanh, no extra internal bits): $\tanh|z|=-E/(E+2)$ with $E=\mathrm{expm1}(-2|z|)$ from Cody-Waite reduction by $\ln2$ and a Horner Taylor polynomial whose degree is set by $p$. It is at most 2.5 ulp at every $p$.

**Solve.** Reference LAPACK 3.12.1 `DGELSS` (the SVD least-squares driver) on $[\Phi,\mathbf1]\,w=y$, ported statement by statement from the netlib source with each Fortran floating-point operation mapped to one rounded operation: Householder QR (`DGEQR2`, `DLARFG` with Blue's `DNRM2`, `DLARF1F`), bidiagonalization (`DGEBD2`), $P^T$ generation (`DORGL2`), implicit-shift QR on the bidiagonal (`DBDSQR` with `DLARTG`, `DLAS2`, `DLASV2`, `DLASR`), and the truncated back-substitution (`DRSCL`, `DGEMV`). Block size is 1, the unblocked configuration; blocking only reorders the same operations. The machine parameters (`DLAMCH`, `LA_CONSTANTS`, the `DNRM2` thresholds, $\mathrm{EPS}^{-1/8}$, $0.01$) are those of the format. $\mathrm{RCOND}=2^{1-p}$, the expC09 cutoff. The cutoffs $\kappa\,2^{-p}$, $\kappa\in\{1,8,32\}$, are applied to the same decomposition as a sensitivity check.

**Metric.** Relative $L_2=\|s-f\|_2/\|f\|_2$ on the evaluation grid, computed in binary64 against binary64 $f$ at the unrounded points (the observer, outside the model). The reference line fixes slope $-1$ in $\log_2 E$ and fits only the intercept on $p=16..40$, as in expC09.

**Checks** (`tests/test_pbit.py`, 202 tests):
- Every primitive matches MPFR, subnormals included, for $p=2..53$ and binary32. Binary32 and binary64 also match numpy and the native builds.
- The machine parameters equal what gfortran computes from the reference source.
- tanh, features and readout equal an independent MPFR replay written from the spec alone.
- The ported `DGELSS` equals netlib `SGELSS`/`DGELSS`, built from source with gfortran and `-ffp-contract=off`. It was tested on tanh design matrices and on random, rank-deficient and graded matrices.
- At full size, `run.py --anchors` repeats the reference-LAPACK and native-build comparisons in exact FP32 and FP64.
- A static no-leakage audit (`src/precision/audit.py`) walks clang's syntax tree of the emulator build, with every macro expanded and every branch included. Outside the rounding primitives it finds no floating $+,-,\times,\div$, no unrounded conversion, no math call other than exact ones, and no raw literal. The one exception is LAPACK's integer crossover $\mathrm{MNTHR}=\mathrm{INT}(\mathrm{REAL}(\min(M,N))\cdot1.6)$, computed in single precision as reference LAPACK does, which only selects whether to QR-factor first. Negative controls confirm the audit catches a planted multiply, `sqrt` call and unrounded conversion.
- The emulator's remainders are exact at every magnitude: under $2^{-900}$ they are formed on operands scaled by an exact power of two. MPFR checks cover products, quotients and roots down to $2^{-1100}$.

**Code & data.** Library `src/precision/` (`pbit_algo.h` model, `lapack_gelss.h` the port, `pbit_emul.c` emulator, `pbit_native.c`, `pbit.py`, `reference_lapack.py`); spec `experiments/expC11_true_precision_law/SPEC.md`; runner `experiments/expC11_true_precision_law/run.py` (`--sweep`, `--anchors`, `--standard`) and `plot.py`; tests `tests/test_pbit.py`, `tests/pbit_replay_reference.py`, `tests/test_pbit_replay_selfcheck.py` (`uv run --extra dev --extra precision python -m pytest ...`). Data in `results/checkpoint_C_geometry/expC11_true_precision_law/data/`: `config.json` (config and source hashes), `measurements.jsonl`, `summary.json`, `models/p*.npz`, `anchors.json`, `standard.json`. Figures in `figures/`: `all_targets_chirp_precision_law_true.png`, `diagnostic_true_vs_forward_only.png`.

## Results

The fully $p$-bit error falls one bit per bit from $p\approx13$ to $p=53$, with no floor. Over $p=16..53$ the constant $E\,2^{p}$ has median $1115$; from $p=18$ on it stays between 857 and 1980 except for three single-point spikes ($p=30$: 5402; $p=38$: 2319; $p=52$: 2214), and it is 2833 at $p=16$ where the curve is still settling. The larger cutoffs $8\cdot2^{-p}$ and $32\cdot2^{-p}$ remove the spikes (median 1042, maximum 2588 and 2304), and $1\cdot2^{-p}$ adds more. At $p=53$ the error is $1.04\times10^{-13}$, the same as the forward-only experiment's FP64 plateau ($9.3\times10^{-14}$).

In exact IEEE FP32 the error is $6.39\times10^{-5}$, bit-identical to netlib `SGELSS` on the same matrix. The sweep's $p=24$ point, which differs only in exponent range, is $6.81\times10^{-5}$. The standard numpy-tanh plus scipy pipelines in native FP32 give $7.0\times10^{-5}$ (`gelsd`), $5.5\times10^{-5}$ (`gelss`) and $1.45\times10^{-6}$ (`gelsy`); in FP64 they give $8.2\times10^{-14}$, $7.8\times10^{-14}$ and $4.7\times10^{-15}$.

At low precision the large solve fails: the error is $1.35$, $1.46$, $0.84$, $0.39$, $0.29$ and $0.25$ at $p=8..13$, and the kept rank is 100, 232, 299, 575, 871 and 943 (it is 977 to 989 from $p=16$ up).

### Figures

- `all_targets_chirp_precision_law_true.png`: the expC09 three-panel figure with panel (c) replaced, same layout and styling. Panels (a) and (b) are unchanged FP64 measurements. Panel (c) plots relative $L_2$ error against working precision $p$ ($8..53$) for the fully $p$-bit pipeline at the predicted $\lambda$, with the dashed $C\,2^{-p}$ line ($C=1386$, intercept fit on $p=16..40$). Look for the curve following the dashed line to $p=53$ without leveling off.
- `diagnostic_true_vs_forward_only.png`, left: error against $p$ for the fully $p$-bit pipeline (teal) and the expC09 forward-only pipeline (grey), each with its own dashed $C\,2^{-p}$ line. Red open markers are the numpy/scipy FP32 ($p=24$) and FP64 ($p=53$) pipelines; the blue star is exact IEEE FP32. Look for the constant gap of about $6$ bits between the two curves, the grey curve's floor from $p\approx48$, and `gelsy` sitting well below the SVD solvers.
- `diagnostic_true_vs_forward_only.png`, right: $E\,2^{p}$ (the constant $C$) against $p$ on a log axis. The thick teal line is the panel's cutoff, the thin lines are the other cutoffs and grey is forward-only. The shaded band is the intercept-fit range. Look for teal holding near $10^3$ across the whole range, the spikes that only the smallest cutoffs show, and grey holding near 20 until it rises into the FP64-solve floor.

## Additional details

- **Why the exponent range matters at $p=24$.** Exact FP32 produced $1.8\times10^{8}$ subnormal intermediates, mostly deep in the solve. With binary64's range these are ordinary numbers, and the result moves by 6% ($6.39$ vs $6.81\times10^{-5}$). The sweep isolates significand bits. The FP32 point is shown separately.
- **The solver matters for $C$.** On the identical problem in native FP32 and FP64, LAPACK's `gelsy` (column-pivoted QR, complete orthogonal decomposition) is 38 to 49 times (FP32) and 17 times (FP64) more accurate than the SVD drivers `gelss`/`gelsd`. This comparison uses numpy's tanh features and OpenBLAS, so it is not $p$-bit faithful. The $C\approx10^3$ here belongs to `DGELSS`, not to least squares in general.
- **Cost.** One $p$ takes 1 to 2.5 minutes single-threaded in the emulator; the sweep took 1.5 CPU-hours.

## Conclusions

Pending Sam's review. What the data plainly shows: with the model and the `DGELSS` solve both at $p$ bits, the error follows $C\,2^{-p}$ with $C\approx10^{3}$ from $p\approx16$ through $53$, including the top of the range, where the forward-only experiment had a floor.

## Open questions

- Does $C$ grow with the width $W$ (for example like $W$, or like $\sqrt{MN}$, as backward-error bounds for Householder least squares suggest), and does the breakdown at low $p$ move with $W$?
- Would a $p$-bit port of `DGELSY` bring $C$ down toward the forward-only value of about 20, as the native FP32/FP64 comparison suggests?
