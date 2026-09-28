# Reading map: scale acquisition, population theory, and optimizer evidence

Updated September 26, 2026. This is an index for assembling the paper, not a
new theory note. Start with the five documents below. The remaining tables
locate detailed evidence and earlier arguments without treating every version
as the current conclusion.

## 1. The five main documents

| Read | Document | What to take into the paper |
|---|---|---|
| 1 | [Consolidated theory and evidence](d34_scale_acquisition_consolidated.md) | The current complete narrative: exact ODE, force decomposition, GD population proof, output-error consequence, Adam controls, and remaining questions. Sections 3–6 contain the GD argument; Sections 7–9 contain Adam; Section 10 suggests the paper claim. |
| 2 | [Population concentration and target moments](d34_population_concentration.md) | The principal detailed GD source: sensitivity lemmas, projection and target-moment refinements, accumulated concentration, relative growth advantage, dispersion closure, causal redistribution, and broad validation. Sections 9–14 develop the most recent population mechanism. |
| 3 | [Adam population study](../results/checkpoint_D_optimizers/expD34_readout_race/population_output/ADAM_POPULATION.md) | The full Adam empirical sequence: energy concentration, signed motion, coherence, momentum interventions, adaptive sensitivity, denominator restriction, and why the GD bound does not transfer directly. |
| 4 | [Coarse feedback and adaptive scaling](d34_adam_coarse_feedback.md) | The latest Adam/GD comparison, exact tracking-renewal identities, fixed-preconditioner mobility proof, reduced adaptive-cycle proof, and completed causal controls. Section 7 includes the late-window qualification and failed branches. |
| 5 | [Population dynamics and output-error study](../results/checkpoint_D_optimizers/expD34_readout_race/population_output/README.md) | The empirical bridge from population movement to raw output error, larger geometry interventions, collective conditions, and adaptive diagnostics. Read its earlier “latest” language in conjunction with the consolidated note. |

The current GD claim is conditional on observable population structure and
tracking allowances. The proof derives weak sensitivity and an output-error
floor. The Adam controls support a coarse adaptive stability constraint with
target-dependent persistence; they do not supply the same quantitative theorem.

## 2. Detailed experiments, organized by the question they answer

| Question | Source | Scope and reading caution |
|---|---|---|
| Does the effective fine force explain motion across more than degree nine and sine? | [Matched-feedback campaign](../results/checkpoint_D_optimizers/expD34_readout_race/effective_feedback/README.md) | 23 target instances, 333 starting checkpoints, and 999 primary branches. Separates changing residuals from changing sensitivity and records forecast failures at longer horizons. |
| How does that decomposition connect to a finite-time bound? | [Cross-function audit](d34_cross_function_audit.md) | Connects the same force to tracking, sensitivity evolution, and approximation defects. This is an earlier audit, not the final concentration theorem. |
| Does changing energy concentration cause different acquisition? | [Completed energy-redistribution campaign](../results/checkpoint_D_optimizers/expD34_readout_race/population_output/evidence/energy_final_20260925/README.md) | Matched Gram-preserving concentration doses, native GD/effective-flow pairs, and refinement checks. Concentration limits available sensitivity but does not uniquely determine expansion. |
| What happens after finite outward slope perturbations? | [Dilation experiment](../results/checkpoint_D_optimizers/expD34_readout_race/mechanism_dilation/README.md) | The earlier twofold perturbation study. Use the population-output study above for the later larger-perturbation work; do not describe this earlier study as an order-of-magnitude intervention. |
| Which coupled models predict motion, and which fail? | [Mechanism refinement](../results/checkpoint_D_optimizers/expD34_readout_race/mechanism_refinement/README.md) and [persistence experiments](../results/checkpoint_D_optimizers/expD34_readout_race/mechanism_persistence/README.md) | Forecasts, coupled polynomial models, physical perturbations, and failed scalar or restoration closures. These explain why we moved beyond frozen-force and equilibrium descriptions. |
| Do fitted readouts at large slopes pursue the construction's $O(h)$ scale? | [Readout-scale experiment](../results/checkpoint_D_optimizers/expD34_readout_race/readout_scale/README.md) | Frozen sine dictionaries: GD and Adam readout magnitudes are consistent with that scale. This is a separate geometry-fixed test, not a joint-learning theorem. |
| Can enlarging Adam's learned slopes repair its geometry? | [Frozen-scale gap](../results/checkpoint_D_optimizers/expD34_readout_race/adam_force_extension/frozen_scale_gap/README.md) | Common-gamma construction dictionaries versus center-preserving dilation of learned features. Larger slopes at learned centers do not uniformly close the gap. |
| What do frozen-readout assays say about learned features? | [Useful-slope diagnostics](../results/checkpoint_D_optimizers/expD34_readout_race/useful_slopes/README.md) | Earlier fixed-budget readout assays and crossed slope/bias diagnostics. Some language describes relative improvement as “useful geometry”; do not substitute that for attaining the paper's precision objective. |
| What did the earlier GD/Adam force comparison show? | [Force-extension study](../results/checkpoint_D_optimizers/expD34_readout_race/adam_force_extension/README.md) | Earlier multi-target optimizer comparison and decomposition. The new wide Adam studies above refine its conclusions. |
| Did the aligned-population theorem's assumptions hold? | [Population coverage audit](../results/checkpoint_D_optimizers/expD34_readout_race/population_coverage/README.md) | They did not: none of 223 archived states lies in the exact aligned sector. This is a reason not to use that theorem as the main explanation. |
| Were the initial-state moment certificates numerically useful? | [Moment-persistence audit](../results/checkpoint_D_optimizers/expD34_readout_race/moment_persistence/README.md) | Tests the conservative initial-region and GD certificates. Distinguish these short guarantees from the later useful conditional population bounds. |
| What were the original race and signal-recovery observations? | [D34 report](../results/checkpoint_D_optimizers/expD34_readout_race/REPORT.md), [signal recovery](../results/checkpoint_D_optimizers/expD34_readout_race/signal_recovery/README.md), and [transport evidence](../results/checkpoint_D_optimizers/expD34_readout_race/transport_barrier/README.md) | Early evidence for depletion, recovery, and insufficient population acquisition. Historical motivation rather than the latest persistence explanation. |

For the final coarse-feedback campaign, the argument and figures are in the
coarse-feedback note. Its machine-readable supporting records are
[coverage and comparisons](../results/checkpoint_D_optimizers/expD34_readout_race/population_output/evidence/coarse_feedback_20260926/release_analysis/highlights.json)
and [late-window controls and failures](../results/checkpoint_D_optimizers/expD34_readout_race/population_output/evidence/coarse_feedback_20260926/release_analysis/completion_details.json).

## 3. Other theory notes: what they contribute and what superseded them

| Note | Contribution | Status for paper assembly |
|---|---|---|
| [Population output persistence](d34_population_output_persistence.md) | Output-error connection and Theorems 14–17: accumulated feedback, GD transfer, initial-data refinement, and movement-dependent reinforcement. | Important predecessor. The concentration argument now supplies more interpretable structure than assuming an allowance on future reinforcement. |
| [Signed population balances](d34_population_balance_mechanism.md) | Exact tanh balance, signed target/generated-error effects, collective comparisons, and short-window refinements. | Useful supporting mechanism and derivations; later concentration work sharpens the main assumption set. |
| [Coarse-balance exposition](d34_coarse_balance_stagnation.md) | The earlier central account of effective-fine-force dominance and coupled residual/geometry feedback. | Good historical bridge; use the consolidated note for current conclusions. |
| [Coarse-balance details](d34_coarse_balance_stagnation_details.md) | Detailed low-disequilibrium derivations and supporting evidence. | Appendix source, with notation reconciled to the current note. |
| [Conditional effective-force acquisition theorems](d34_effective_force_acquisition_theorems.md) | Signed defect accounting, coupled forecasts, tracking allowances, neighborhood conditions, and GD transfer. | Earlier theorem family; its uniformly useful constants were not established across long intervals. |
| [Mechanism and rate theorems](d34_mechanism_rate_theorems.md) | Conditional mechanisms, discrete rate bounds, and small-parameter estimates. | Supporting mathematics; several proposed low-dimensional closures were not validated generally. |
| [State-dependent persistence](d34_state_dependent_persistence.md) | Signed force evolution and conditional persistence through loaded amplification. | Explains reinforcement bookkeeping; the regional amplification premise still needs justification. |
| [Coupled ODE mechanisms](d34_coupled_ode_mechanisms.md) | Proved reduced examples of compensation, generated-error correction, passage delays, and algebraic slowing. | Illustrative mechanisms, not established reductions of every trained network. |
| [Polynomial effective-force surrogate](d34_polynomial_surrogate_theorem.md) | Evolving polynomial approximation and conditional transfer to exact tanh. | Useful technical appendix material; approximation validity is distinct from structural persistence. |
| [Rescaled transport model](d34_rescaled_transport_model.md) | Coupled population transport with changing geometry under an approximately fixed error load; conservation and dissipation identities. | The clearest detailed link back to the original transport-PDE viewpoint. |
| [Transport action bound](d34_transport_action_bound.md) | A generated-error energy budget bounds the fraction that can ever acquire distant scales in the reduced flow. | Reduced-model theorem; native tanh-GD transfer remains conditional. |
| [Target-gap persistence](d34_target_gap_persistence.md) | Exact tanh persistence for targets with little low-degree fine loading; finite-width energy proof. | Proved, but its checkpoint-only durations remain too short to replace the current conditional theorem. |
| [Aligned population persistence](d34_population_persistence_theorem.md) | Heterogeneous maximum principle and preserved readout–slope alignment. | The proof is for a restrictive sector that fails the archived population audit. Keep as an example, not the main paper theorem. |
| [Initial-data moment theorem](d34_moment_persistence_theorem.md) and [GD companion](d34_moment_persistence_gd.md) | Persistence with mixed signs and biases, tracking control, and discrete certificates. | Earlier initial-data route with conservative numerical usefulness. Do not confuse it with the later accumulated-concentration result. |
| [Certified instances](d34_certified_instance.md) | Rounding-controlled ordinary-GD exclusion windows: 20k for one degree-nine state and 13k for one mixed-sine state. | Genuine instance certificates with narrow scope; distinct from broad empirical validation. |
| [Energy baseline](d34_energy_baseline.md) | Generic energy-based finite-time scale exclusion. | Baseline for judging how much structure the sharper theory retains. |
| [Barrier walkthrough](d34_barrier_theorem_walkthrough.md), [transport mechanisms](d34_transport_mechanisms.md), and [transport scale barrier](d34_transport_scale_barrier.md) | Original formulation, deterministic transport intuition, residual modes, and finite-time acquisition framing. | Historical starting point; not the current theorem statement. |
| [Signal-depletion theory](d34_scale_acquisition_theory.md) | Early depletion and subsequent recovery under D34's actual random-readout initialization. | Motivation and earlier rate accounting. |
| [Zero-readout signal depletion](readout_driven_slope_signal_depletion.md) | An early-window mechanism under different readout initialization. | Do not transfer its initialization assumptions to the later D34 regime. |
| [Earlier Adam moment results](d34_adam_moment_results.md) | Moment interventions at width 177 and failure of a general persistent two-phase closure. | Useful negative result; the wide population and coarse-feedback studies are newer. |
| [Adam extension plan](d34_adam_force_plan.md) | Original experiment design and implementation record. | Protocol, not an additional source of findings. |

The older [gamma-barrier synthesis](gamma_barrier_synthesis/synthesis.md) and
[theory framework](gamma_barrier_synthesis/theory_framework.md) connect earlier
conditioning, geometry, readout fitting, and drift arguments. Their September
14 status predates the current population and adaptive-control results.

## 4. Reader-facing exposition and paper sources

| Artifact | Intended use |
|---|---|
| [PI technical note](d34_scale_acquisition_pi_note.md) | Earlier self-contained exposition with force-versus-scale plots, motivations, width comparisons, and perturbations. Useful for figure ideas and explanation, but predates the latest concentration and Adam conclusions. |
| [Earlier paper-section Markdown](d34_scale_acquisition_paper_draft.md) | The compressed narrative and observation/force figures for an earlier feedback-theorem version. |
| [Earlier main LaTeX](d34_scale_acquisition_paper_main.tex), [appendix](d34_scale_acquisition_paper_appendix.tex), [figures](d34_scale_acquisition_paper_figures.tex), and [integration memo](d34_scale_acquisition_paper_integration.md) | Reusable layout and technical material. These are not synchronized to the concentration-based Theorem 3.3 in manuscript version 27. |
| [Manuscript version 27](drafts/QI_MLPs___ICLR_2027_Submission%20(27).pdf) | The manuscript version used in the latest review. Its Section 3.5 and Theorem 3.3 are the paper baseline; its GD theorem should be read with the current concentration note. |
| [New Adam main-text paragraph](d34_adam_feedback_paper_main.tex) and [new Adam appendix](d34_adam_feedback_paper_appendix.tex) | Current insert and detailed derivations incorporating the completed controls. |
| [Compiled Adam review](../output/latex_d34_adam_feedback/d34_adam_feedback_paper_review.pdf) | Rendered proposed paragraph, appendix, and control figure. This is an insert review, not a rebuilt complete submission. |

## 5. Related frozen-feature work in the neighboring checkout

These sources concern the preceding conditioning/readout part of the paper.
They are separate from the moving-feature population theorem above.

| Source | Content |
|---|---|
| [Section 3.4 spectrum note](../../precision-mlps-frozen-gamma-probe/docs/section34_spectrum_note.tex) and [PDF](../../precision-mlps-frozen-gamma-probe/output/pdf/section34_spectrum_note.pdf) | Frozen-feature spectral argument and the joint/fixed optimizer comparison plots discussed during paper assembly. |
| [Complete gamma optimization note](../../precision-mlps-frozen-gamma-probe/docs/gamma_optimization_full_note.md) | The broader conditioning and target-direction account in that checkout. |
| [Readout paper note](../../precision-mlps-frozen-gamma-probe/docs/gamma_readout_paper_note.md) | Readout-optimization argument intended for the paper. |
| [Adam feature-probe results](../../precision-mlps-frozen-gamma-probe/docs/adam_feature_probe_results.md) | Matched readout restarts, uniform-scale comparisons, and target-weighted spectrum diagnostics. A different campaign and budget from the D34 controls. |

When combining these studies, preserve the distinction between actual trained
readouts and diagnostic refits, raw relative error and relative MSE, physical
slopes and normalized RMS scale, and current-endpoint versus best-within-budget
selection. These differences matter more than matching the wording of older
notes. Relative improvement alone does not establish the high-precision
accuracy required by the paper.
