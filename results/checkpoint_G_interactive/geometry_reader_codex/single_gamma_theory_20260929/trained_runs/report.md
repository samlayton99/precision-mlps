# Gamma and readout branches in saved trained networks

The saved VarPro floor solution supports a qualified version of Sam's prediction: its visible coefficient bands separate inverse-width **regimes**, not necessarily nearly identical gamma values. The near-horizontal band around readout −0.05 mostly contains broad kernels with gamma below 1. The almost-zero band contains the sharp kernels with gamma at least 8. Intermediate gamma values carry the larger positive and negative coefficients. One checkpoint is insufficient to establish a general branch law.

## Saved floor case

Source: `results/checkpoint_D_optimizers/expD04_varpro/varpro_corrected/06_sine_xavier_seeds/geometry_sine_xavier_N256_s0.npz`, adjacent `.json`, and `experiments/expD04_varpro/varpro_corrected/inspect_geometry.py` / `varpro_gn_probe.py`.

- Target sin(2πx), Xavier initialization, VarPro Adam warmup then Gauss–Newton. The saved summary reports relative L2 6.654869951238788e-15.
- 461 neurons; 242 centers inside [−1,1]. Canonicalization is gamma=|a|, center=−b/a, oriented readout=sign(a)·v. This removes the trivial tanh sign ambiguity.
- Original training grid: 2003 uniformly spaced points, including endpoints. Original readout is a truncated-SVD minimum-norm solve with relative singular cutoff 1e-13.
- Output bias was omitted from the checkpoint. Its least-squares optimum was recovered while holding the saved neuron coefficients fixed. On 8193 evaluation points, the recovered saved model has relative L2 6.5735e-15. A fresh solve gives 6.5694e-15 with numerical rank 81 of 462 columns. These confirm the saved near-machine-precision fit; the last digits are evaluation-grid/backend dependent.
- Interior gamma minimum/median/maximum: 0.0684 / 2.1126 / 109.4406; quartiles 1.1656 and 3.9776.
- All 54 interior neurons with gamma<1 have negative oriented readouts, ranging from −0.0582 to −0.0154 (median −0.04825).
- All 26 interior neurons with gamma≥8 have |v|≤1.47e-4; most are substantially smaller. This near-zero branch spans a factor of 13 in gamma, directly showing that a common branch need not imply equal gamma.
- The 70 interior neurons with gamma in [2,4) have readouts from −0.7073 to 0.8851: sharing a coarse gamma range does not force a single readout branch.
- Exploratory band v∈[−0.06,−0.04] contains 47 interior neurons, of which 40 (85%) have gamma<1. These bounds were chosen after looking at the figure; this is descriptive evidence, not a preregistered hypothesis test.

Two descriptive readout cohorts, both selected after viewing and restricted to centers in [−1,1]:

| Readout cohort | Neurons | Median gamma | Gamma IQR | Fraction gamma<1 | Fraction gamma>8 |
|---|---:|---:|---:|---:|---:|
| abs(v)<0.001 | 37 | 9.927 | 7.413–13.629 | 0% | 70.3% |
| abs(v+0.05)<0.01 | 47 | 0.584 | 0.451–0.846 | 85.1% | 0% |

Figures: `varpro_floor.png` gives all readout amplitudes in the displayed center range; `varpro_floor_zoom.png` shows the central coefficient bands; `varpro_floor_gamma_cohorts.png` splits the same checkpoint into fixed gamma bins. Both zoom figures explicitly identify omitted amplitude outliers. The color bounds are not shared with other runs.

## Ordinary joint Adam comparisons

Used all three sine initialization arms from `results/checkpoint_D_optimizers/expD31_split_adam/data/{xavier,scaled_xavier,qi_zero}__sine__0.npz`. The source `experiments/expD31_split_adam/run.py` explicitly makes mu=0 ordinary joint Adam on a,b,v. These are 177-neuron, 500-step, seed-0 runs, learning rate .002, not long-converged Adam examples. No new training was run. All readouts include a saved output bias.

| Initial geometry | Saved Adam relative L2 | Fresh readout solve relative L2 (1e-13) | Interpretation |
|---|---:|---:|---|
| Xavier | 0.9212 | 0.005281 | Little target structure learned in 500 steps; not evidence about successful converged networks. |
| Scaled Xavier | 0.08502 | 4.025e-6 | Raw Adam coefficients are scattered; fresh solve uses large compensating coefficients. |
| QI, zero readout | 0.003396 | 3.639e-11 | Raw Adam readout follows a clear smooth derivative-like curve; interior gamma stays 15.916–16.139. Geometry was initially nearly uniform, so this does not show spontaneous gamma grouping. |

Plots show the original Adam readout and freshly solved readout separately. A solved readout is not the readout Adam found. Fresh solves use the saved 1024 midpoint training grid and the original relative singular cutoff 1e-13. Evaluation uses 8193 uniform points including endpoints for all cases.

## Transparent local descriptor

For every interior neuron, choose its 12 nearest centers. Among these choose the 3 closest in oriented readout. Compare the mean absolute log-gamma difference in that selected group to the mean over all 12 candidate neighbors. A ratio below 1 means readout-nearby points within a center neighborhood tend to have more similar gamma. It is not a branch detector or a causal test; neighborhoods overlap and center dependence is only controlled approximately.

| Run | Saved readout ratio | Fresh solved readout ratio |
|---|---:|---:|
| VarPro floor | 0.670 | 0.670 |
| Adam Xavier | 0.979 | 0.742 |
| Adam scaled Xavier | 1.008 | 0.672 |
| Adam QI | 0.677 | 0.719 |

This descriptor supports association for the floor case and solved coefficients, but does not support a universal association in the raw Adam coefficients. For QI the absolute gamma variation is tiny: colors exaggerate that variation unless the colorbar is read.

## Limits and reproducibility

Coefficients are not uniquely determined by good function values in these ill-conditioned systems. The TSVD cutoff chooses one solution convention. On the floor geometry, changing cutoff from 1e-13 to 1e-10 changes numerical rank 81→69 and relative L2 6.57e-15→9.59e-10. On the scaled-Xavier Adam geometry, cutoff1e-13 yields coefficient norm1.50e6; cutoff1e-10 yields3.64e3 with a different error. Coefficient-branch claims must specify the solver convention.

No seed replication is claimed: the floor geometry is the one available saved seed-0 checkpoint. The analysis does not establish that gamma alone predicts readouts, or that it causes the observed bands independently of center distribution and the global least-squares solve.

A bounded search for a longer fully joint Adam snapshot found only aggregate metrics in expD02 and expD17; the expD31 `ordinary_refits` files also contain only metrics. The long saved expD06 Adam states use fixed centers and a special parameterization and were not mislabeled as unconstrained joint Adam. Thus the ordinary-Adam evidence here is deliberately limited to the three 500-step controls, with QI the only one already fitting reasonably closely.

Numerical rank81 does not invalidate the algebraic dual identity for TSVD: if A=UΣVᵀ and retained modes are r, then wτ=Aᵀqτ with qτ=UrΣr⁻²Urᵀy exactly (in exact arithmetic for that decomposition). Full column rank is unnecessary for this statement. The dual field depends on the entire geometry and cutoff, however, and bias handling matters: here the constant column is included in the same minimum-norm solve, so qτᵀ1 equals the solved output bias; it is not constrained to zero. A derivation that first removes an unpenalized intercept or integrates by parts must use that distinct convention and retain boundary terms where appropriate. The identity alone does not prove v is a local sample of f′.

`analyze.py` reads only existing checkpoints, produces PNGs and diagnostic NPZs, and saves source hashes and exact metrics to `summary.json`. It uses NumPy/SciPy and Matplotlib (Agg) from `/Users/sam/venv/pfloat/bin/python`. Reproduce from the repo root with `OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/precisionmlps-trained-gamma-mpl /Users/sam/venv/pfloat/bin/python results/checkpoint_G_interactive/geometry_reader_codex/single_gamma_theory_20260929/trained_runs/analyze.py`. No application code, experiment source, checkpoints, or training state was modified.
