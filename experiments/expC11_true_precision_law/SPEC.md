# expC11 specification: the chirp precision law with every model and solver operation at p bits

This is the normative definition of the computation. It is implemented once in C (`src/precision/pbit_algo.h` for the model, `src/precision/lapack_gelss.h` for the solve) and compiled as a p-bit emulator and as native binary32/binary64 code (`src/precision/pbit.py`). `tests/test_pbit.py` checks it against MPFR, against an independent MPFR replay of the model written from this spec alone (`tests/pbit_replay_reference.py`), and against netlib reference LAPACK compiled from source.

## Arithmetic

A *format* $(p, e_{\min}, e_{\max})$ is a binary floating-point format with a $p$-bit significand, normal exponents $e_{\min}\le e\le e_{\max}$, and gradual underflow below $2^{e_{\min}}$ (IEEE 754 semantics). $Q$ is round-to-nearest, ties-to-even, into the format. The only arithmetic operations are

$$\mathrm{add}(a,b)=Q(a+b),\ \mathrm{sub}(a,b)=Q(a-b),\ \mathrm{mul}(a,b)=Q(ab),\ \mathrm{div}(a,b)=Q(a/b),\ \mathrm{sqrt}(a)=Q(\sqrt a),$$

each applied to format values, with no fused multiply-add. Exact operations need no rounding: negation, absolute value, comparison, copying the sign bit, multiplication by a power of two that stays in range (`ldexp`), and $\mathrm{rint}$ (nearest integer, ties to even) of a small value. An integer enters arithmetic as $Q(n)$, like Fortran `DBLE(N)`.

Formats used:

- **Sweep (the panel):** $(p, -958, 959)$ for $p = 8,\dots,53$. This is binary64's exponent field pulled in by 64 binades at each end, so that the binary64 carrier of the emulator holds every value of the format, subnormals included, as a normal number. At $p=53$ the results are bit-identical to exact binary64 (checked).
- **Anchors:** binary32 $(24,-126,127)$ and binary64 $(53,-1022,1023)$.

## Constants (each rounded once into the format, stored)

- tanh: $\mathrm{INVLN2}=Q(1/\ln 2)$; with $k_b$ the bit length of $p+4$, $\mathrm{LN2HI}=Q_{\max(p-k_b,1)}(\ln 2)$ and $\mathrm{LN2LO}=Q(\ln 2-\mathrm{LN2HI})$; $\mathrm{SAT}=Q((p+3)\ln 2/2)$; Taylor degree $d$ = the smallest integer $\ge1$ with $0.35^d/(d+1)!\le2^{-(p+1)}$, coefficients $a_j=Q(1/(j+1)!)$, $j=0,\dots,d-1$.
- LAPACK machine parameters, as reference LAPACK computes them for a Fortran real kind with `DIGITS` $=p$, `MINEXPONENT` $=e_{\min}+1$, `MAXEXPONENT` $=e_{\max}+1$: `DLAMCH` ('E' $=2^{-p}$, 'P' $=2^{1-p}$, 'S', 'O'), `LA_CONSTANTS` (`safmin`, `safmax`), the `DNRM2` thresholds (`tsml`, `tbig`, `ssml`, `sbig`), $\mathrm{EPS}^{-1/8}$ of `DBDSQR` as a correctly rounded power, and $Q(0.01)$. For binary32 and binary64 these equal what gfortran computes from the reference source (checked).

## Inputs (data, rounded into the format outside the model)

- Training inputs $\tilde x_i=Q(x_i)$, $x_i$ the binary64 `np.linspace(-1,1,M)`; labels $\tilde y_i=Q(\mathrm{fl}_{64}(f(x_i)))$, $f(x)=\sin(8\pi(x+1)^2)$.
- Evaluation inputs $\tilde x^e_i=Q(x^e_i)$ on `np.linspace(-1,1,M_e)`.
- Geometry integers $\tilde N=Q(N)$ and $\tilde\jmath=Q(j)$ for $j=-H,\dots,N+H$.
- Bandwidth $\tilde\lambda=Q(\lambda)$, $\lambda$ from the refined rule, chosen offline (not a model computation).

## Model

1. Geometry: $h=\mathrm{div}(2,\tilde N)$; $c_j=\mathrm{add}(-1,\mathrm{mul}(\tilde\jmath,h))$; $\gamma=\mathrm{div}(\tilde\lambda,h)$.
2. Feature: $\phi_j(x)=\mathrm{tanh}_p(\mathrm{mul}(\gamma,\mathrm{sub}(x,c_j)))$.
3. $\mathrm{tanh}_p(z)$: with $a=|z|$, return $\mathrm{sign}(z)$ if $a\ge\mathrm{SAT}$. Otherwise $u=-2a$ (exact); $k=\mathrm{rint}(\mathrm{mul}(u,\mathrm{INVLN2}))$; $r=u$ if $k=0$, else $r=\mathrm{sub}(\mathrm{sub}(u,\mathrm{mul}(k,\mathrm{LN2HI})),\mathrm{mul}(k,\mathrm{LN2LO}))$; Horner $q=a_{d-1}$, $q=\mathrm{add}(a_j,\mathrm{mul}(r,q))$ for $j=d-2,\dots,0$, $P=\mathrm{mul}(r,q)$; $E=P$ if $k=0$, else $E=\mathrm{add}(\mathrm{ldexp}(P,k),\mathrm{sub}(2^k,1))$; $t=-\mathrm{div}(E,\mathrm{add}(E,2))$; the result is $t$ for $z\ge0$ and $-t$ otherwise.
4. Design matrix $A\in\mathbb R^{M\times n}$, $n=W+1$: column $j<W$ is $\phi_j(\tilde x_i)$, the last column is all ones.
5. Output: $s=w_{\text{bias}}$, then $s=\mathrm{add}(s,\mathrm{mul}(\phi_j(\tilde x),w_j))$ for $j=0,\dots,W-1$.

## Least-squares solve

Reference LAPACK 3.12.1 `DGELSS` (`SRC/dgelss.f`, the $M\ge N$ branch, `NRHS` $=1$) and every routine it reaches, ported statement by statement with each Fortran floating-point operation mapped to one operation above, in Fortran evaluation order: `DLANGE`, `DGEQRF`→`DGEQR2`, `DLARFG`, `DNRM2` (Blue's algorithm), `DLAPY2`, `DLARF1F`, `DGEMV`, `DGER`, `DAXPY`, `DSCAL`, `DORMQR`→`DORM2R`, `DGEBRD`→`DGEBD2`, `DORMBR`, `DORGBR`→`DORGLQ`→`DORGL2`, `DBDSQR` with `DLARTG` (the Fortran 90 version), `DLAS2`, `DLASV2`, `DLASR`, `DROT`, `DSWAP`, and `DRSCL`. Configuration: block size 1 (`ILAENV` returns 1), so the level-2 kernels run; blocking only reorders the same operations. $\mathrm{RCOND}=2^{1-p}$ (the expC09 panel's cutoff); singular values $\le\mathrm{RCOND}\cdot\sigma_1$ are treated as zero. Further cutoffs $\kappa\,2^{-p}$, $\kappa\in\{1,8,32\}$, are applied to the same decomposition for a sensitivity check, exactly as separate `DGELSS` calls would from the state after `DBDSQR`. Branches `DGELSS` cannot reach here (lower bidiagonal, `DLASCL` rescaling, `M < N`) are not ported and return an error if reached.

## Evaluation

The error is the observer, in binary64 and outside the model: $\|s-f(x^e)\|_2/\|f(x^e)\|_2$ against binary64 $f$ at the unrounded evaluation grid, so input quantization counts as error.
