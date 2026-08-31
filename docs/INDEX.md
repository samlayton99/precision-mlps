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

**expD07 through expD15 are archived (2026-08-12).** The salvage record — what was achieved, what failed, what is still worth revisiting, with pointers into the archive — is **`results/checkpoint_D_optimizers/D07-D15_SALVAGE.md`**. Read that instead of the experiments. Code: `experiments/_archive/expD07..expD15`; results and method docs: `results/_archive/checkpoint_D_optimizers/expD07..expD15`; tests: `tests/_archive/` (excluded from collection via `pyproject.toml` addopts). Note `docs/ORIENTATION.md` predates expD14 iterations 1-4 and is stale where they disagree.

| document | contents |
|---|---|
| `docs/REQUIREMENTS.md` | the gate (above) |
| `docs/requirements_and_lessons.md` | the evidence behind the gate: five requirements as originally written, three litmus tests, eight measured lessons with their measurements |
| `results/checkpoint_D_optimizers/PROGRAM_FRAMING.md` | **the program-level context (Sam, 2026-08-12): the geometry/readout split, the four experiment axes, the three ways to win, the moonshot stated properly. Read before designing any optimizer or init experiment.** |
| `results/checkpoint_D_optimizers/D07-D15_SALVAGE.md` | **the salvage record for the archived campaign** |
| `results/checkpoint_D_optimizers/expD22_cdrge/expD22_results.md` | CD-RGE zero-order (Chaubard `zero_order_rnn`) on the expD16 suite: tuned best is a ZO Adam clone at ~50x the passes; the class is not precision-competitive. Collection canceled early by Sam. |
| `results/_archive/checkpoint_D_optimizers/expD12_mu_ladder/STEP2_SOLVER_SPEC.md` | the step-2 solver in full: $\mu$ control rules, the damping floor $\alpha \ge r_\text{entry}$, terminal-solve laws, the APPROVED row-separation result. Only writeup of expD12/expD13. |
| `results/_archive/checkpoint_D_optimizers/expD15_inclusion_score/METHOD_L_selection.md` | which parameters to solve: four mechanisms, tradeoffs, costs |
| `results/_archive/checkpoint_D_optimizers/expD09_2nd_order_regime/{expD09_recipe_results.md, DAMPED_GAUSS_NEWTON.md, SUBPROBLEM.md}` | the block-QR + LSMR recipe, the $\mu$ math, the frozen-$\Phi$ subproblem |

## Construction and theory (checkpoints A through C)

| document | contents |
|---|---|
| `docs/explanation.md` | conceptual walkthrough mapping the paper's equations onto `src/construction/qi_mpmath.py`. The Toeplitz solve versus the $\Phi$ least-squares solve, and why both exist. |
| `docs/theory_lambda_rule.md` | the aliasing rule predicting the optimal dimensionless bandwidth $\lambda^*$ from the activation's Fourier tail |
| `docs/theory_magnitude_rule.md` | the readout-norm law, same Fourier tail, one spectral integral |
| `docs/thoughts.md` | Sam's working notes on lstsq versus QI and the $\Phi$ null space |

## Applications (checkpoints E through H)

| document | contents |
|---|---|
| `results/checkpoint_F_applications/expF01_linear_de_zoo/PINN_FEASIBILITY.md` | can the QI construction power a PINN. Analysis plus the reasoning behind expF01. |
| `experiments/expF01_linear_de_zoo/pinn_poc.py` | the throwaway numpy proof-of-concept behind that analysis |
| `docs/future_experiments.md` | the design spec for the non-optimizer checkpoints (E, F, G) |
| `results/checkpoint_H_highdim/expH01_highdim_suite/SUITE_SPEC.md` | Sam's specification of the checkpoint-H high-dimensional benchmark: the 80-task suite with genuinely multivariate targets (Version 3, the one built), plus the superseded 60-task suite (Version 2) and the 138-task factorial framing (Version 1, context) |
| `results/checkpoint_H_highdim/expH01_highdim_suite/expH01_results.md` | the suite as built: the twelve function families, the common scaling, the data geometries, the three test sets, the predicted center density, the gallery figures, and the even-geometry reference that exercises it |
| `results/checkpoint_H_highdim/expH02_nonuniform_spacing_1d/expH02_results.md` | smoothly non-uniform 1-D center spacing at constant lambda: works with the locally right gamma; the widest gap sets the width; a spacing jump that does not shrink with N stalls the error |
| [Ridge Cascade](https://claude.ai/code/artifact/6e225ab0-8eb9-4a72-b8ff-a423b9caafe9) | interactive 3-D explorer of one two-hidden-layer ridge basis function (direction wheel, per-layer spreads, five second-layer sheets, tanh/swish/gelu with expC07 $\lambda$) |
| `docs/ridge_quadrature_theory.md` | the operating theory of checkpoint H, formalized: Fourier-polar representation, the direction-snapping certificate $\|F-F_V\|\le r\int\|\xi\|\theta(\hat\xi,V)d|\mu|$, the $M$-free 1-D leg, certificate-to-projection, the max-of-two-floors bracket; each claim labeled theorem / exact-in-2D / measured / open |
| `docs/highdim_open_questions.md` | Sam's eight open questions for checkpoint H (the 2-D direction cliff, the optimal center distribution, their interaction, global+local neurons, depth, gated MLPs, 3-D to the floor, whether Radon is the right picture) plus the bookmarked hole experiment |
| `results/checkpoint_H_highdim/expH04_mesh_finding/expH04_results.md` | the mesh-finding ladder: what placement theory says (spectral rule for centers, the ridge direction tax, active subspaces), and the measured rungs from data-only monitors to the iterated active subspace in d = 3 and 5 |
| `results/checkpoint_H_highdim/expH05_direction_cliff_2d/expH05_results.md` | the 2-D direction cliff on nine targets: error on a data ball versus the direction count at fixed along-direction resolution -- a plateau, then a 5-9 order collapse in two or three steps of $M$, then a floor near $10^{-14}$; the threshold rises with the ball radius and with the target's difficulty. Plus the budget-split follow-up: at fixed $MN$ the error is the worse of a direction-limited and an offset-limited floor, the best split is where they cross, and the optimal path doubles $M$ and $N$ in alternation |
| `results/checkpoint_H_highdim/expH06_ridge_hierarchy/expH06_results.md` | the hierarchical ridge mesh in $d=3,4$: hidden-ridge recovery to the floor (projection pursuit + Gauss-Newton polish of directions), the greedy hierarchy (nested background + atoms, refine-vs-open by trial fits) against the even mesh, the 3-D floor curves and the push to the floor |

## Results

`results/results.md` is the global cross-experiment synthesis. Per-experiment writeups are at `results/checkpoint_<A..H>_<name>/exp<X>NN_<name>/exp<X>NN_results.md`.

**All `*.md` under `results/` is tracked** (the ignore exception was widened in August 2026 from `*_results.md`-only -- the narrow rule is how four load-bearing documents were nearly lost in the July cleanup). Data, caches and gif directories under `results/` remain ignored; figures and HTML explainers are tracked.

## Archive

`results/_archive/checkpoint_D_optimizers/expD07..expD15` holds the archived optimizer campaign (writeups, method docs and figures stay git-tracked; data does not). `experiments/_archive/` holds its code, `tests/_archive/` its tests. Entry point: `results/checkpoint_D_optimizers/D07-D15_SALVAGE.md`.

`results/_archive/expD08_qi_init_nlcg/` holds 311 MB of superseded data from the expD08 campaign: iterations 1 through 10, the tether study, the batching runs, and the hardening runs. Iteration 11 supersedes all of it (now at `results/_archive/checkpoint_D_optimizers/expD08_qi_init_nlcg/iteration_11/`). That data dump is local only, ignored by git, and safe to delete. Its reusable content was promoted into `docs/REQUIREMENTS.md` and `docs/requirements_and_lessons.md` before archiving.
