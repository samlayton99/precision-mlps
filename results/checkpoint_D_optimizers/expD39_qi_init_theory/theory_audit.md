# Which part of the QI theory transfers to initialization?

The current tabular initializer imports the **ridge-bank geometry**, not the complete constructive approximation theorem. That distinction changes both how to interpret the gamma histograms and which modifications deserve experiments.

## Gamma is a physical scale; lambda is the invariant

A bank has features $\tanh(\gamma_r(u_r^T x-c_{rj}))$, with $\|u_r\|_2=1$. Thus the row norm requested by Sam is exactly $\gamma_r$. For equally spaced centers separated by $h_r$, the dimensionless quantity is $\lambda_r=\gamma_rh_r$.

If a projected interval is $[m_r-A_r,m_r+A_r]$ and contains $P_r$ endpoint-inclusive centers, then $h_r=2A_r/(P_r-1)$ and $\gamma_r=\lambda(P_r-1)/(2A_r)$. In normalized coordinates $s_r=(u_r^T x-m_r)/A_r$, the slope is $\Gamma_r=\gamma_r A_r=\lambda(P_r-1)/2$. Equal-sized banks can therefore have identical normalized slopes while their physical row norms differ considerably. A broad gamma histogram alone does not falsify the construction.

Common gamma is a legitimate alternative when center spacing is also common. D39 tests a shared spacing large enough to cover every estimated projection interval. That necessarily spends more centers outside the narrower intervals; equal norms are not a free improvement.

## Concrete discrepancies in D38

1. **The actual spacing differs from the one used to compute gamma.** F04 divides the projected span by the center count, then uses an endpoint-inclusive grid. In D38, 22-center banks have actual $\lambda=0.25(22/21)$, and the final six-center bank has actual $\lambda=0.25(6/5)$. The isolated correction uses the number of gaps. It lowers gamma; correctness does not imply better validation MSE.
2. **The remainder is undersized.** Width 512 was allocated as 23 banks of 22 plus one of six. Balancing across the same 24 directions gives eight banks of 22 and sixteen of 21. This removes the exceptional under-resolved bank while retaining the direction count.
3. **The projected interval is centered at zero.** Train-standardized raw inputs have zero mean, but their projections can be skewed, and hidden activations are generally not centered. D39 tests actual lower/upper projection quantiles and shifts biases with the interval midpoint. This remains a data-adaptive heuristic, not a theorem about the unknown target.
4. **No boundary halo is present.** The source construction distinguishes $N$ interior intervals, $N+1$ interior nodes and $R$ extra centers on each side: $W=N+2R+1$. A bank of 21–22 neurons cannot retain that many interior centers and also afford the high-precision halo. A fixed-width 25% collar is tested as a practical coverage adjustment; it is not the full theoretical halo construction.
5. **The direction allocation is not derived from the target.** At initialization each repeated-direction weight matrix has rank at most 24. This is a real linear input bottleneck, particularly for the higher-dimensional tasks. It does not mean the nonlinear feature matrix has rank 24. Training is unconstrained and can separate the rows. The ridge theory assigns separate errors to direction coverage and scalar resolution; it does not prescribe $M\approx P\approx\sqrt W$.
6. **The readout is random and both hidden layers train.** The original constructive weights depend on the target derivative; the alternative span argument depends on fitting a suitable readout. Neither is the random paired readout used here. The depth theory additionally assumes useful scalar channels, controlled ranges and downstream sensitivities. Applying random projection banks twice does not establish these conditions.
7. **The meaning of $\lambda=.25$ changed.** Its justification comes from near-fp64 approximation with a frozen scalar grid and a suitable readout. Noisy generalization and jointly trained Adam optimize a different objective. Larger lambda trades alias suppression for narrower transitions and different conditioning; the .5 and 1 tests are empirical hypotheses, not theorem-guaranteed improvements.

## Older manuscript qualifications

The Section 3 rewrite is more precise about center counts than the main workshop PDF, but several displayed arguments are not valid as written:

- Appendix C.2 adds a fixed aliasing contribution $\delta$ to exponentially decreasing terms, then absorbs it into $Ae^{-\alpha W}$. That argument instead gives $Ae^{-\alpha W}+\delta$. It cannot establish pure exponential convergence for arbitrarily large width at fixed positive delta.
- Appendix B.3 reindexes a doubly truncated convolution using $m=k+j$ but keeps the original outer support. The support actually extends by the stencil radius, and coefficients must account for the original $k$ restriction. The implementation's same-width convolution with extended derivative samples is a separately defined truncation.
- With $T_{rj}=hK_\gamma((r-j)h)$, the cardinal condition requires $Tc=he_0$. The rewrite's Algorithm 1/B.2 omit the factor $h$; the practical notes and implementation include it.

The supplied September manuscript *Boundary-corrected tanh networks: representation, sampling, and numerics* avoids the first overclaim: it retains an explicit replica term, gives geometric convergence above a declared floor, and distinguishes a bounded-readout recovery theorem from a posteriori certification of an arbitrary computed LS network. Its requirements still do not certify D38's two-layer ordinary training. These manuscript observations are local audit findings, not edits to the authors' papers.

## What the experiments can establish

The geometry tests verify the initializer actually implements its stated equations, coordinate shifts, bank allocation and shared-readout control. The scalar analytic check verifies a resolved tanh dictionary and the real derivative-convolution construction separately. The tabular screen asks which geometry helps ordinary training under fixed recipes; three-seed confirmation checks whether the selected effect persists. Neither a successful scalar span check nor a favorable tabular score establishes the missing multidimensional, noisy, jointly trained theorem.

## Sources and evidence

- Original problem and precision setting: `papers/QIs_workshop.pdf`.
- Center counts and original error terms: `papers/Section_3_Rewrite.pdf`, Sections 3.1–3.4 and Appendices B/C.
- Correct implemented normalization and halo: `papers/practical_implementation.tex`; `src/construction/qi_mpmath.py`, especially lines 120–134.
- Historical tabular initializer: `experiments/expF04_qi_init_real_data/model.py`, lines 84–158; D38 invocation in `experiments/expD38_init_readout_baseline/run.py`.
- Qualified frozen-grid bandwidth guidance: `results/checkpoint_C_geometry/expC07_lambda_energy_rule/lambda_rule/hardened_rule.md`.
- Direction versus scalar-resolution allocation: `docs/ridge_quadrature_theory.md`.
- Conditional depth construction: `results/checkpoint_I_depth_theory/compositional_qi_theory.md`, Sections 6–10.
- Revised approximation/recovery/numerical statements: `docs/appendix_materials/2026-09-25/supplied/fixed_lambda_tanh_three_theorems.pdf`, pages 1–2.
- D39 measurements: `geometry.json`, `analytic_checks.json`, and the matching figures in this output directory.

Independent read-only theory and implementation reviews agreed with the main audit. Seventeen focused checks passed before screening; the completed D38/D39 focused suite contains 18 passing tests after adding the direction/slope controls. The completed experiment report separates the small, mixed effect of correcting spacing from the larger task-specific bandwidth and layer-placement results.
