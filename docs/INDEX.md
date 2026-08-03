# Where everything lives

Every surviving document in the repo, what it is for, and when to read it. Documents live **next to the experiment's writeup under `results/`** (Sam reads `results/`, not `experiments/`); `experiments/` holds only code; `docs/` holds only program-level material that no single experiment owns.

## Read in this order

| # | document | why |
|---|---|---|
| 1 | **`docs/REQUIREMENTS.md`** | **The gate.** Requirements, the complexity budget, GPU cost model, the kill list of measured-dead designs, and a pre-build checklist. Any proposed mechanism is checked against this *before* implementation. Most bad ideas die here in ten minutes instead of a week. |
| 2 | `docs/ORIENTATION.md` | Current state of checkpoint D: what is settled, what is open, what failed, which written claims are wrong. The roadmap for the next phase. |
| 3 | `docs/motivation.md` | Why the program exists. The twelve-digit gap, why gradient descent cannot close it, the three steps. |
| 4 | `CLAUDE.md` | Repo conventions, architecture, the QI construction's precision regimes, writeup format. |

If a document is not linked from `ORIENTATION.md` or from this index, treat it as historical.

## The optimizer program (checkpoint D)

| document | contents |
|---|---|
| `docs/REQUIREMENTS.md` | the gate (above) |
| `docs/requirements_and_lessons.md` | the evidence behind the gate: five requirements as originally written, three litmus tests, eight measured lessons with their measurements |
| `results/checkpoint_D_optimizers/expD09_2nd_order_regime/DAMPED_GAUSS_NEWTON.md` | the $\mu$ math written out: $d_\mu = \arg\min\|J_Ld+r\|^2 + \mu\|d\|^2$, the $\alpha = \sqrt\mu/\sigma_1$ correspondence, $\kappa_\mu = 1/\alpha$, three measured laws |
| `results/checkpoint_D_optimizers/expD12_mu_ladder/STEP2_SOLVER_SPEC.md` | **the step-2 solver in full.** The $\mu$ control rules (IV.5-IV.7), the damping floor $\alpha \ge r_\text{entry}$, the terminal-solve laws, the APPROVED row-separation result. Also the **only** writeup of expD12 and expD13, which have no results file. Covers expD09 through expD13. |
| `results/checkpoint_D_optimizers/expD15_inclusion_score/METHOD_L_selection.md` | **which parameters to solve.** Four working mechanisms with measured tradeoffs, costs, and enough implementation detail to rebuild them. The characterization: a parameter is in $L$ iff its Jacobian column does not move when $L$ is perturbed. |
| `results/checkpoint_D_optimizers/expD11_batching/SAM_SPEC_superseded.md` | Sam's original $O(m{+}n)$ rank-deficient least-squares spec. Answered negatively by expD11; kept for framing. |
| `results/checkpoint_D_optimizers/expD09_2nd_order_regime/SUBPROBLEM.md` | the frozen-$\Phi$ subproblem definition |
| `results/checkpoint_D_optimizers/expD10_step2_hardening/batching_test.md` | the batching test plan. T1 is corrected in place from expD11's negative result. |

**Experiment map, D07 onward.** Each writeup is at `results/checkpoint_D_optimizers/<exp>/`.

| experiment | question | writeup |
|---|---|---|
| expD07 | can standard optimizers reach the floor on real multilayer data | yes |
| expD08 | QI-init nonlinear CG, the tether, batching | yes |
| expD09 | the second-order regime, the recipe that hits machine epsilon at $O(d)$ state | yes |
| expD10 | hardening across 6 targets, 4 widths, noise, batching, fp32, 2-D | yes |
| expD11 | can any $O(dk)$ iterative solver reach the fp64 floor | yes, negative result |
| expD12 | the $\mu$ ladder on a frozen $\Phi$ | **only** in `STEP2_SOLVER_SPEC.md` |
| expD13 | the $\mu$ ladder on a drifting $\Phi$ | **only** in `STEP2_SOLVER_SPEC.md` |
| expD14 | first stitch of Adam and the solve (iteration 0) | yes, with a correction header listing four wrong claims |
| expD15 | which parameters enter $L$ | yes, plus `METHOD_L_selection.md` |

## Construction and theory (checkpoints A through C)

| document | contents |
|---|---|
| `docs/explanation.md` | conceptual walkthrough mapping the paper's equations onto `src/construction/qi_mpmath.py`. The Toeplitz solve versus the $\Phi$ least-squares solve, and why both exist. |
| `docs/theory_lambda_rule.md` | the aliasing rule predicting the optimal dimensionless bandwidth $\lambda^*$ from the activation's Fourier tail |
| `docs/theory_magnitude_rule.md` | the readout-norm law, same Fourier tail, one spectral integral |
| `docs/thoughts.md` | Sam's working notes on lstsq versus QI and the $\Phi$ null space |

## Applications (checkpoints E through G)

| document | contents |
|---|---|
| `results/checkpoint_F_applications/expF01_linear_de_zoo/PINN_FEASIBILITY.md` | can the QI construction power a PINN. Analysis plus the reasoning behind expF01. |
| `experiments/expF01_linear_de_zoo/pinn_poc.py` | the throwaway numpy proof-of-concept behind that analysis |
| `docs/future_experiments.md` | the design spec for the non-optimizer checkpoints (E, F, G) |

## Results

`results/results.md` is the global cross-experiment synthesis. Per-experiment writeups are at `results/checkpoint_<A..G>_<name>/exp<X>NN_<name>/exp<X>NN_results.md`.

**All `*.md` under `results/` is tracked** (the ignore exception was widened in August 2026 from `*_results.md`-only -- the narrow rule is how four load-bearing documents were nearly lost in the July cleanup). Data, caches and gif directories under `results/` remain ignored; figures and HTML explainers are tracked.

## Archive

`results/_archive/expD08_qi_init_nlcg/` holds 311 MB of superseded data from the expD08 campaign: iterations 1 through 10, the tether study, the batching runs, and the hardening runs. Iteration 11 supersedes all of it and stays live at `results/checkpoint_D_optimizers/expD08_qi_init_nlcg/iteration_11/`. The archive is local only, ignored by git, and safe to delete. Its reusable content was promoted into `docs/REQUIREMENTS.md` and `docs/requirements_and_lessons.md` before archiving.
