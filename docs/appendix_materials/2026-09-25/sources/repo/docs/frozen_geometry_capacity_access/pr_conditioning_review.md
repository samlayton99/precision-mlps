# PR #2 conditioning implementation review

Completed by the delegated explorer `pr_conditioning_review` and incorporated by the coordinating task. Inspected source matches merged commit `d7d26e6`; PR source head is `b71a03396990`. No training or code changes were performed for this review.

## Main finding

The revised note's five readout maps match the PR implementations. Its independent alpha-Xavier physical initialization is new: existing individual-scale runs re-encoded the earlier collective physical start. A small new frozen-gamma runner is required; the existing training entry points do not implement the proposed audit unchanged.

## Geometry, allowances, and initialization

[core.py:47](/Users/sam/my-repos/research/collaborations/precisionMLPs/experiments/expD06_fixed_center_scales/core.py:47) constructs h=2/N, ceil(sqrt(N)) halo centers per side, and the uniform grid. At N512 this gives 559 neurons. The construction allowances use the fixed `LAMBDA_REF=.25`; they do not depend on the gamma being swept. Ordinary allowances are

$$\alpha_{\rm ordinary}=\frac{h}{2(0.25-\pi h/(2\lambda_{\rm ref}))}.$$

Corrected outer halo slots receive additional construction terms; the bias allowance is $1+\sum_j\alpha_j$. `g.d` denotes $\sqrt\alpha$. At N512 the ordinary allowance is approximately .008663, largest corrected-halo allowance 1.04521, and bias allowance 10.7639. There is no clipping or interpolation toward one. The report describes ordinary and corrected-halo allowances as asymptotically O(h) at fixed reference bandwidth, with different constants; the bias allowance is O(1). See [parameter-scale report](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD06_fixed_center_scales/parameter_scale_results.md:97).

[core.py:81](/Users/sam/my-repos/research/collaborations/precisionMLPs/experiments/expD06_fixed_center_scales/core.py:81) distinguishes:

- `xavier`: physical Gaussian Xavier readouts.
- `xavier_a_uniform`: physical readouts sqrt(h) times the Xavier draw.
- `xavier_a_reference`: physical readouts sqrt(alpha_j) times the Xavier draw.
- `envelope`: alpha_j times the draw's sign; not a Gaussian initialization.

The Xavier standard deviation is sqrt(2/(W+1)). Thus the collective ordinary physical start is O(h): its sqrt(alpha) multiplier and Xavier draw are both O(sqrt(h)). Multiplying Xavier by alpha=O(h), as the revised note separately proposes, gives O(h^(3/2)) ordinary coefficients. These are different physical starts.

The paired individual-coordinate campaigns start with `xavier_a_reference` and re-encode it. To make the new alpha-Xavier start with the same random draw, obtain the collective physical vector and multiply by `g.d` once more; keep bias zero. Then encode each physical start into every coordinate map, and reuse it across gamma.

## Exact readout maps

Write the physical readout as c=Tz. Let L be the square lower-bidiagonal difference map and s_j=sum_{l<=j} alpha_l over neurons, excluding bias.

| Coordinates | Hidden-readout map | Bias scale | Existing label |
|---|---|---|---|
| Raw | I | 1 | `physical` in higher-order code; `raw` elsewhere |
| Collective | diag(sqrt(alpha)) | sqrt(alpha_bias) | `scaled` |
| Individual | diag(alpha) | alpha_bias | `parameter_scale` |
| Collective neighboring | L diag(sqrt(s)) | sqrt(alpha_bias) | `scaled_differences` |
| Individual neighboring | L diag(s) | alpha_bias | `parameter_differences` |

[difference_training.py:25](/Users/sam/my-repos/research/collaborations/precisionMLPs/experiments/expD06_fixed_center_scales/difference_training.py:25) provides `decode`, `encode`, and gradient `pullback` for the four nonraw maps. Encoding neighboring coefficients uses cumulative physical readouts divided by the appropriate cumulative scales.

Neighbor features are scaled differences phi_j-phi_{j+1}, plus the final scaled phi_W anchor. With that anchor and the output bias, the change of coordinates is invertible and preserves the function space. Scale cumulative coefficients before differencing; unequal scaling after differencing breaks the intended tail cancellation. Do not apply individual scales a second time or draw independent native neighbor coefficients when claiming matched physical starts.

## What to reuse, and what not to call unchanged

- [readout_solvers.py:18](/Users/sam/my-repos/research/collaborations/precisionMLPs/experiments/expD06_fixed_center_scales/readout_solvers.py:18): `coordinate_map`, `from_physical`, and `dictionary` support collective and collective-neighbor readouts. Frozen learned-slope signs are absorbed in the map; uniform positive gamma needs no sign correction.
- Its `linear_chunk(..., stable_armijo=True)` has readout-only GD/momentum/Adam machinery, but its GD uses Armijo. The CLI hardcodes the uniform control gamma=.25/h and accepts a time budget. It is not the proposed constant-rate multi-gamma run.
- The joint `difference_training.chunk` updates slopes. It cannot serve as a frozen-geometry control unchanged.
- [difference_analysis.py:138](/Users/sam/my-repos/research/collaborations/precisionMLPs/experiments/expD06_fixed_center_scales/difference_analysis.py:138): `mapped_features` supports only `scaled` and `scaled_differences`; other strings silently take its difference branch. Do not supply individual-map labels to this helper without extending it.
- [diagnostics.py:148](/Users/sam/my-repos/research/collaborations/precisionMLPs/experiments/expD06_fixed_center_scales/diagnostics.py:148): use the existing `rel_l2` output directly. Historical MSE results must not be labeled relative L2.
- Build the frozen comparison around fixed geometry allowances, fixed physical starts, and the normalized B=A(gamma)T. At N512, gamma64 means lambda .25; gamma32 means .125.

## Optimizer-coordinate consequences

Native GD induces physical movement

$$\Delta c=-\eta TT^T\nabla_c L.$$

Diagonal scales are squared in that physical GD multiplier. Neighboring introduces a coupled preconditioner. Stable step selection must include the entire normalized B, including bias and halos.

For positive diagonal c_j=t_j z_j, Adam has a physical outer step factor eta*t_j, with native epsilon translating to physical gradient threshold epsilon/t_j. The PR explicitly matched those thresholds in its collective-versus-individual campaign. Neighboring Adam mixes gradients before coordinatewise normalization, so it is not equivalent to GD's TT^T rule.

For the revised single fixed-damping readout GN probe, reuse [higher_order.py:92](/Users/sam/my-repos/research/collaborations/precisionMLPs/experiments/expD06_fixed_center_scales/higher_order.py:92) `augmented_qr(B,r,damping)` or the SVD gain s/(s²+damping) in [joint_mechanism_probes.py:142](/Users/sam/my-repos/research/collaborations/precisionMLPs/experiments/expD06_fixed_center_scales/joint_mechanism_probes.py:142). Restart from the same residual for each damping. Do not call joint outer GN, which moves slopes and adapts damping, or add a learning-rate scaling after the solve.

Identity damping in native coordinates penalizes the physical step by damping*||T^(-1) Delta c||². Map changes therefore change the regularizer. A same-physical-penalty QR control already exists at `joint_mechanism_probes.py:150`; the report records agreement of matched physical/function steps at roughly 1e-12–1e-14.

## Existing evidence and its scope

The directly relevant frozen uniform-lambda=.25 study used zero readouts, three targets, two widths, and up to 140k updates. Neighbor coordinates improved Adam's final-window MSE in all six cases. For N512 sine, the reported MSE improved from 1.06e-8 to 3.38e-11. GD/momentum improved sine and mixed sine but worsened quadratic. These are coordinate effects at identical geometry, not gamma-sweep results. See [relative-rate report](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD06_fixed_center_scales/relative_rate_results.md:255).

Some frozen plateaus were numerical line-search stalls. A repaired audit restored updates but gave only modest extra Adam improvement, leaving broader slow convergence. Record actual representable updates and distinguish these failures from mathematical floors.

The detached uniform sweep covered lambda .125/.20/.25/.35/.50/1, two maps, and two widths. It measured spectra and fits, not actual training across gamma. Its `uniform_conditioning.png` was inspected: retained conditioning, retained rank, and sine fit error behave differently; its condition numbers exclude discarded modes.

The [neighboring theorem](/Users/sam/my-repos/research/collaborations/precisionMLPs/docs/neighbor_difference_conditioning.md) removes a width-dependent cumulative-coordinate penalty for an ideal uniform whole-line block while retaining small-lambda smoothing. It is not automatically a bound for the complete finite-window matrix including bias, final anchor, halos, and unequal learned slopes. Sharpening can improve interior localization while strengthening halo/bias near-cancellation.

The PR supplies useful controls and implementation pieces. It has not measured the requested fraction of finite-budget optimization error attributable to bounded gamma after those controls.
