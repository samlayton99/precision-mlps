# expD10 hardening ledger -- what the step-2 part was taken through

**Status: draft-pending-Sam. This is the explicit record of the final step-2 gate: every claim tested, every failure with its mechanism, and the fix status of each.** Numbers regenerable from `hardening.json`, `hardening_extra.json`, `evidence.json`, `floors.json`, `warm.json`. Figures in `figures/` (`F1`..`F5`). Method details in `expD10_results.md`.

The part under test: **hardened block-QR** -- agglomerative cluster-matched blocking (`cluster_blocks`) from an $s$-row sample, pivoted Householder QR whitening at $k{=}128$ (drop at `rcond`), one uninterrupted LSMR run with Fong-Saunders stopping (`atol=btol=1e-15`), single unwhiten; pre-solve applicability gate `kappa_gate`; materialized-$B$ and implicit-compensated memory modes.

---

## List 1 -- every claim, assumption, and experiment it was taken through

| # | claim / assumption | experiment | outcome |
|---|---|---|---|
| 1 | works across the target family, not 3 targets | H1: 6 targets $\times$ $N\in\{64..512\}$, 24 cells | **pass**: at/below SVD floor everywhere |
| 2 | correct behavior when the *target's* floor is high | H1: abs_cubed etc. | **pass**: lands exactly on the target's own floor at every width, no overfit |
| 3 | observable stopping $\approx$ oracle, incl. semiconvergence stress | E4 + H1 | **pass**: $1.6\times$ geo off oracle; stress cell stops on-floor at it 4011 |
| 4 | tolerance knobs are plateaus, not cliffs | H5: rcond $\times$ atol $3\times3$ | **pass**: one-order total spread; rcond $10^{-15}$ marginally best |
| 5 | recovers a destroyed column order | E1 + band check | **pass**: $7.7\times10^{-9}\to5.4\times10^{-14}$ |
| 6 | one blocking algorithm for band AND cluster structure | agglomerative vs Fiedler, both cases | **pass**: agglomerative wins both; shipped + tested |
| 7 | "premium in iterations, not accuracy"; caps not binding | H6: $4\times$ cap | **pass** |
| 8 | statistical noise floor, six decades | E5 | **pass**: $1.06$--$1.15\times$ of $0.272\sigma$; stopping self-regularizes |
| 9 | batching at $\sigma=0$ | E5 | **pass**: $b\ge4d$ at floor |
| 10 | batching under noise (jointly) | H4 | **pass at $b=4d$** ($\approx1.1\times$ batch floor); $b=2d$ fails, see F2 |
| 11 | floor survives growing $d$ on the QI band structure | H2: $d\to1844$ | **pass**: $1.2\times$ floor at $d=1844$ |
| 12 | random $\Phi$ with the QI spectral fingerprint, growing $d$ | twin d-series $270\to1844$, structured + structure-free | structured: **pass at flat cost** (38$\to$131 it); structure-free: fails, worsening -- F1 |
| 13 | precision-agnostic (drives to the dtype floor) | full fp32 pipeline, 9 cells | **pass**: geo $1.26\times10^{-6}$ vs fp32 floor $1.10\times10^{-6}$ |
| 14 | $B$-free implicit mode parity | E2 (3 cells) + H2 ($d=1844$) | **pass to $d\approx500$**; breaks at $d=1844$ -- F6 |
| 15 | warm start + anytime budgets | E6 | **pass**: no loss from adversarial $w_0$; smooth budget curve |
| 16 | applicability detectable pre-solve | `kappa_gate` on 4 case types | **pass**: 3-orders separation from a 384-row sample ($\rho$-mass detector failed, withdrawn) |
| 17 | min-norm output (coupling armor) | E6 audit | measured: cold near min-norm; warm inflates $5$--$20\times$ (step-3 watch item) |

## List 2 -- what it cannot pass (mechanism, meaning, fix status)

**F1. Structure-free matrices at $k/d\ll1$ -- fundamental, worsens with $d$.**
$7\times10^{-11}\to2.7\times10^{-8}\to1.8\times10^{-6}$ over $d=462\to1844$ (floors $\sim5\times10^{-15}$; SPIR passes at every $d$). Mechanism: round-4 information argument -- Haar singular vectors admit no blocking, so whitening removes only a $k/d$ fraction of the conditioning. **Not fixable within $O(d)$ state; this is the method's scope boundary.** Fix status: `kappa_gate` detects it from a sample pre-solve ($10^{9}$ = go, $10^{13}$ = no-go); adaptive-$k$ escalation policy quantifies what budget would be needed; fallback is SPIR ($O(dr)$) or accepting $\sim10^{-6}$. Step-3 question this forwards: do real trained-network blocks sit on the structured side? (QI does; Haar does not; the gate answers per-instance.)

**F2. Small batches under noise ($b=2d$, $\sigma>0$).**
Blows up to $6\times10^{-2}$ ($10^{3}\times$ above the batch statistical floor) on some cells; the $\sigma=0$ finding "2d mostly fine" does not transfer. Mechanism: near-square noisy batch -- the batch null space rotates into noise directions and the whitening inverts them. Fix status: see fix log below.

**F3. Isolated mild floor gap (sine $N{=}512$: $6\times$ above SVD floor at any budget).**
Not a stopping artifact (oracle-confirmed); non-monotone in $d$ ($N{=}1024$: $1.2\times$). Softens "at or below the floor everywhere" to "within a small multiple". Fix status: see fix log.

**F4. Superlinear iteration growth in $d$ on band structure** ($62\to1930$ over $d=270\to1844$ at $k{=}128$).
Flops-competitiveness with direct QR is marginal below $d\sim2000$; memory remains the win. The clustered case does NOT show this (131 it at $d=1844$): the growth is the band's long-range coupling chain, not the method. Fix status: see fix log.

**F5. Unmeasured (honest residue).** 2-D geometry (fix log -- now being measured), $\Phi$-as-Jacobian spectra from real training, bf16 (PyTorch port), TSQR streaming whitening, warm-start from real Adam iterates.

**F6. Implicit-compensated mode breaks parity at $d=1844$** ($1.16\times10^{-11}$ vs materialized $8.0\times10^{-14}$).
Mechanism (measured): the implicit operator is $A_{\rm kept}R^{-1}=Q+ER^{-1}$ with a fixed, consistent $\varepsilon\kappa(R)\approx3\times10^{-6}$ per-call deviation from $Q$ at *every* $d$ -- the QR factorization's own backward error, distinct from the application error dd removes. Benign to $d\approx500$; at $d{=}1844$ it plausibly splits the degenerate singular-value clusters Krylov speed depends on. Fix status: see fix log.

---

## Where we stand: the step-3 package (re-anchored to motivation.md)

**Cost model, stated precisely (this was getting conflated):** persistent state is the $R$ factors, $d\cdot k$ floats, $O(d)$ -- always. Whitening costs $k$ passes **once per solve event, never per step**. Each LSMR iteration is one forward-backward pass at $O(d)$ memory. $B$ is a batch-sized transient that exists only during a solve event. Amortized over a $T{=}200$-step event cadence the whole thing is $\sim$2--3 extra passes per step. The per-step $O(d)$ requirement is met everywhere, on every geometry, including 2-D ridges; the 2-D finding was never a cost failure.

**The re-anchoring:** motivation.md asks the recurring tier for *meaningful least-squares progress during training* past Adam's $10^{-3}$; machine epsilon is demanded only on settled geometry (the QI 1-D/2-D goalposts), i.e. at a **one-time final solve** -- and a one-time solve is not bound by the per-step memory constraint ("Memory $O(d)$ **per step**", SUBPROBLEM.md). Graded against the right bars, the hardest measured cell (2-D radon, $\kappa=10^{15}$, floor $5\times10^{-15}$) gives:

| tier | cadence | cost | banks (hard 2-D cell) |
|---|---|---|---|
| 1: LSMR + block-Jacobi | every step | 3--30 passes, $O(k^2)$ state | $3\times10^{-3}$..$6\times10^{-4}$ |
| 2: block-QR whitened solve | per event ($T\sim200$ steps) | $k$-pass setup + 100--1000 it | $3.7\times10^{-9}$..$1.3\times10^{-10}$ |
| 3: finisher (SPIR) | **once**, geometry settled | $O(dr)$ transient, CPU-parkable | $1.1\times10^{-16}$ |

All three tiers share the batching rules, Fong-Saunders stopping, and precision-agnosticism; `plan_blocks`/`kappa_gate` picks tier-2's config and predicts its ceiling pre-solve. In 1-D/clustered structure tier 2 alone reaches machine epsilon (the goalpost cells); on global-support features (ridges = MLP neurons in high input dimension) tier 2 banks $10^{-9}$--$10^{-12}$ and the finisher closes to machine epsilon. **Under the coupling law ($\|v\|\eta$ re-injection) nothing beyond tier-2 precision is bankable while the geometry moves anyway, so the division of labor matches the physics of the training loop, not just the memory budget.**

**Revised F1/2-D disposition:** "$O(d)$ is dead on unstructured/ridge features" applies only to the last $\sim$3 orders (from $10^{-12}$ to $10^{-15}$) of a *recurring* solve -- exactly the orders the coupling law says cannot be banked mid-training and the finisher supplies at the end. The recurring tiers beat Adam's barrier by 3--9 orders on every geometry ever measured, structured or not.

## Fix log

Each entry: the mechanism-level fix, its cost against the constraints ($O(d)$ state, one pass/iter, precision-agnostic, battle-tested), and the measurement.

- **F6 -- PARTIALLY FIXED: CholeskyQR-style refinement in the right metric recovers $9\times$; a $\sim10^{-12}$ deviation floor remains.** Two false starts are part of the record: (i) forming $W=A_{\rm kept}R^{-1}$ in fp64 and Cholesky-correcting it does nothing (deviation $2.3\times10^{-4}$ before and after) -- the fp64 *application* error is the same size as the *factorization* error being corrected; (ii) composing $S\,R$ in fp64 would re-inject $\varepsilon\kappa(R)$. The correct step: measure $W$'s Gram through a **dd multi-rhs trisolve** (`dd_trisolve_lower_multi`), Cholesky in fp64 (harmless, $G\approx I$), keep $S$ **separate** (apply $R^{-1}$ then $S^{-1}$). Measured at $d{=}1844$: deviation $7.0\times10^{-5}\to1.4\times10^{-12}$ (5.5 s one-time); implicit solve $1.16\times10^{-11}\to1.31\times10^{-12}$, against materialized $8.0\times10^{-14}$. **The achieved error tracks the deviation 1:1**, and a second refinement step does NOT square it ($\to7\times10^{-13}$ only): an unidentified mechanism floors the block orthonormality near $10^{-12}$ at this scale. Disposition: implicit mode improved $9\times$ but not to parity; materialized $B$ remains the primary mode; the deviation floor is the identified thread to pull if $B$-free at large $d$ ever becomes binding.
- **F4/F3 -- NEGATIVE: offset second-stage whitening does not work.** Whitening $B$ again with $k/2$-offset blocks left accuracy unchanged and *increased* iterations ($284\to314$, $564\to710$, $1930\to2813$): the band's cross-block coupling is not boundary-localized, and the second rotation destroys stage 1's exact within-block orthonormality. Do not re-try. F4 disposition: intrinsic to band-like long-range coupling; mitigation is the known $k$ tradeoff ($k{=}256$ cuts sine $N{=}512$ iterations $564\to215$). F3 disposition: no knob closes the $6\times$ gap ($k{=}256$: $1.8\times10^{-13}$; rcond $10^{-15}$ *hurts* this cell, $1.4\times10^{-10}$ -- the $10^{-13}$ default stands); accepted as a bounded, non-growing limit.
- **F2 -- FIXED: noise-scaled truncation.** At $b{=}2d$, $\sigma{=}10^{-4}$: default rcond gives $4.3\times10^{-4}$; rcond $\in[\sigma^2/..,\sigma]$ ($10^{-6}$..$10^{-4}$) lands both tested cells on the batch statistical floor ($\approx5\times10^{-5}$). Classical truncated-SVD regularization, zero new state. Auto-rule for the API: solve at default; if the train residual floors at $\tau\gg\varepsilon$, re-whiten once with $\mathrm{rcond}=\tau$. Recommended spec stays $b\ge4d$; the fix rescues the $b{=}2d$ corner when $\sigma$ is known/estimated.
- **F1 -- POLICY SHIPPED, plus a gate bug found and fixed.** `plan_blocks`: gate at $k$, escalate $2k,4k,\dots$ on the sample only, return go/no-go + predicted config; never a silent stall. Measured: QI $d{=}922$ GO at $k{=}128$ ($\kappa=10^{8}$); unstructured twin NO-GO with $\kappa\sim10^{13}$ at every $k$ to 512. **Gate bug:** with $s<d$ sample rows the whitened sample is rank-deficient and $\kappa$ is silently *under*estimated -- measured false-GO at $d{=}922$, $s{=}384$ (read $6\times10^{8}$ vs true $7.5\times10^{12}$). `kappa_gate` now requires $s\ge d$ (raises otherwise); at $s\ge d$ separation is four orders. Earlier gate numbers taken at $s{=}384$, $d\ge462$ should be read with this caveat.
- **2-D -- MEASURED in two regimes, and the hard regime is a genuine failure at practical $k$.** First pass (`twod.json`, `twod_hard.json`): 16 cells, solver exactly on the SVD floor in every one, but the floors were $10^{-3}$-scale -- an artifact of a mis-built test (no collar/halo: radius 1.0 vs the zoo's 2.5; $\lambda\in[1,4]$ vs the zoo's tuned $[0.05,1.2]$; harder target), **not of 2-D**: expE01 itself reaches $\sim6\times10^{-14}$ on gauss_bump under the Radon geometries. Corrected run at zoo conventions (`twod_proper.json`; radon_tensor, radius 2.5, gauss_bump, $n{=}8000$): the machine-precision regime is real ($\kappa\approx10^{15}$, rank $\approx0.44$--$0.63d$, floors $5\times10^{-15}$--$2\times10^{-14}$), and in it **block-QR at $k{=}128$/$256$ stalls 2.5--3 orders above the floor** ($2.6\times10^{-12}$..$9\times10^{-10}$ at a 40k cap, $\lambda\in\{0.12,.18,.26\}$). $k{=}512$ ($k/d=0.9$, essentially a full QR) reaches $9.6\times10^{-14}$; **SPIR reaches $1.1\times10^{-16}$**; the **gate correctly reads NO-GO ($\sim10^{13}$) at every $k$, pre-solve.** **The discriminating test settles it: the required $k$ tracks $d$, not the dimension.** At $d{=}1021$ (floor $2.2\times10^{-15}$), $k{=}512$ ($k/d{=}0.50$) and $k{=}768$ ($0.75$) both stall at $\sim5\times10^{-11}$ at a 60k cap, while $k/d{=}0.90$ had sufficed at $d{=}571$. So in the 2-D machine-precision regime block-QR degenerates toward a full QR: **2-D ridge systems at machine precision belong to SPIR ($O(dr)$; measured $1.1\times10^{-16}$), and the gate detects this pre-solve at every $k$.** Mechanism (sharper than the boundary-fraction guess): ridge features have **global spatial support** -- every pair of lines in the disk intersects -- so the Gram never localizes and there is no blocking to find; the gate reading ($1.4\times10^{13}$) is numerically identical to the structure-free twin's, because structurally it is the same case. Design consequence for step 3: if block-QR is wanted in higher input dimension, the features must be **localized** (bumps/patches, not ridges); with localized features the Gram is patch-banded and the 1-D story should transfer. That is a testable prediction, not a measured fact.
