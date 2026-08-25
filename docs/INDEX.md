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

`results/_archive/checkpoint_D_optimizers/expD07..expD15` holds the archived optimizer campaign (writeups, method docs and figures stay git-tracked; data does not). `experiments/_archive/` holds its code, `tests/_archive/` its tests. Entry point: `results/checkpoint_D_optimizers/D07-D15_SALVAGE.md`.

`results/_archive/expD08_qi_init_nlcg/` holds 311 MB of superseded data from the expD08 campaign: iterations 1 through 10, the tether study, the batching runs, and the hardening runs. Iteration 11 supersedes all of it (now at `results/_archive/checkpoint_D_optimizers/expD08_qi_init_nlcg/iteration_11/`). That data dump is local only, ignored by git, and safe to delete. Its reusable content was promoted into `docs/REQUIREMENTS.md` and `docs/requirements_and_lessons.md` before archiving.
