# What not to import, and what is still missing

## Already covered or superseded

| Material | Decision |
|---|---|
| `docs/quill_table_bounds_appendix.tex` | Same lines as the supplied comparison source; do not count twice. Keep the bibliography as an addition. |
| `docs/quill_comparison_table_appendix.tex` and earlier table drafts | Superseded by the supplied five-row comparison derivation. Earlier author attributions and precision wording changed. |
| `papers/Section_3_Rewrite.pdf`, `papers/QIs_workshop.pdf`, and `papers/practical_implementation.tex` | Earlier construction/cardinal-stencil framework. Do not import wholesale into the newer finite-contour, corrected-halo representation proof. |
| `theorem_for_sam.pdf` and `high_precision_tanh_implementation_note.pdf` | Earlier reference theorem and implementation sketch. The supplied fixed-lambda and FP64 recovery notes control the current statement. The implementation audit remains useful. |
| Earlier compact bandwidth bundles v1–v6 and `docs/theory_lambda_rule.md` | Prefer the supplied new tanh proof. Preserve v7 only for additions such as GELU and its documented recipe. |
| Old two-wall model and universal floor/halo selection claims | Several were explicitly withdrawn in the September 8 audit. Do not revive them by copying an older report. |
| Gamma ratio notes v2/v3 and anchor-dependent lower-ratio plots | Replaced by Revision 4's direct two-sided intervals. A lower ratio bound alone does not give a necessary training-time bound. |
| Earlier collaborator smoothing/periodic notes | The later `gamma_optimization_full_note.pdf` explicitly separates its simplified theorem from its full spectral forecasts and evaluates the theorem's slack. Prefer that account. |
| Earlier polynomial accessibility notes | Already covered by the supplied bounded-slope source. They are not an additional contribution and are not substituted for the selected finite-kernel argument. |
| Older C09 precision panels | Retain for their actual restricted rounding experiments, not as evidence of an entirely p-bit solve. C11 is the completed true-p-bit source. |
| Initial C12 parameter/output-only comparison | Superseded by the strict and rescue controls; even the strict comparison retains the stated FP64-solve exception. C13 is the newest shared-contract comparison, but was incomplete. |
| Early “geometry never helps” or “large scale rates must be unstable” summaries | Later interventions and audits qualify or contradict these universal interpretations. Use the evidence ledger and later reports. |
| Higher-dimensional ridge, tensor, PDE, compositional-depth, arithmetic-circuit, and LLM-installation work | Outside the requested Section 3 scope. It was not copied simply to make the packet larger. |

## Specific gaps against the current draft

### 1. Section 3.5 is not backed by a finished joint-training theorem in the located sources

There are exact identities, conditional drift bounds, and useful diagnostics. They do not yet establish that every sufficiently accurate admissible representation requires leaving the region reached by the specified joint-training algorithm, or that readout adaptation persistently prevents that escape. This is the main mathematical gap in the collected Section 3 material. It is not repaired by relabeling a frozen-readout-matrix result as a result about learned geometry.

### 2. The logarithmic-width statement needs its bandwidth dependence reconciled

Theorem 3.1 fixes lambda and allows its constants to depend on lambda. Corollary 3.1.1 and the QUILL comparison row discuss choosing lambda with the requested tolerance. The supplied comparison source explicitly identifies the need for suitable uniform control. The supplied fixed-design recovery discussion also has a finite allowance regime. During assembly, identify which existing statement supplies the uniform or fixed-regime guarantee being claimed. This inventory did not locate a separate completed proof that silently makes every fixed-lambda constant uniform in that changing parameter.

### 3. A theorem's numerical contract is not the same as the experimental baseline

The fixed-lambda constrained-recovery note, the all-binary64 scaled-SVD note, the ordinary practical least-squares experiments, and the true-p-bit DGELSS experiment have different contracts. Match the proof and protocol explicitly. A prescribed geometry plus an arbitrary library least-squares call is not automatically an implementation of the certified recovery theorem.

### 4. The current Figure 3 needs its own provenance

The paper specifies width 512, a normalized 2π/6π/14π mixed sine, and five million executed updates. Our collected gamma interval example uses 153 neurons, 263 samples, a 2π/6π/10π mixture, and spectral calculations. The collaborator's full-note primary example uses 559 hidden features and 8,193 samples. These sources must not be used interchangeably. The exact current-paper Figure 3 data/script/manifest was not located in this sweep.

The caption's phrase “executed frozen-readout GD” also needs checking: the stated experiment/theorem freezes geometry and trains the readout. This is a wording issue to resolve when the original figure protocol is identified.

### 5. Exact inequalities and certified numerical endpoints have different status

Our Revision 4 supplies exact mathematical inequalities. Its displayed evaluations have quadrature/SVD checks and an empirical numerical allowance, not a full rounding certificate. The collaborator's later note separately reports ball-arithmetic checks for specific timing endpoints; its checker artifacts were not located here. Do not transfer that certification to our ratio figures or the current draft's different experiment.

### 6. The latest comparison is still a partial result

C13 has a useful current protocol and saved output, but the inspected sweep was incomplete. No final table of all targets, precisions, and variants should be inferred from its provisional figures. C12's complete results remain usable within their different arithmetic contract.

These are assembly checks and missing-source findings. No new proof or experiment was undertaken to fill them.
