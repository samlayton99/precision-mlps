# Does one shared learning rate with the prescribed scales sustain useful geometry learning?

**Scaled training helps acquire useful geometry, but sustained useful geometry learning and high-precision joint convergence are not established.** At the shared Adam rate $\eta=10^{-3}$, accurate features are already present by 20,000 updates. Bandwidths keep growing through 320,000 updates, but their checkpoint gradients are dominated by error the readout could fit. Final detached readout errors are $6.3\times10^{-12}$ and $5.1\times10^{-11}$, while live training-window errors remain $2.1\times10^{-3}$ and $1.1\times10^{-3}$. Paired unscaled training learns much worse geometry. This is partial support for the proposed intervention on one target, one width, and two seeds.

**Definitions used below.**

| Term | Meaning |
|---|---|
| Unscaled training | Update the network's readout weights, output bias, and slopes directly, all with the same base rate $\eta$ |
| Scaled training | Update their rescaled versions, all with the same base rate $\eta$; the fixed scales from the note determine the resulting changes to the network parameters |
| $\eta$ | One shared learning rate for all trained coordinates, including the output bias |
| $w_j,\gamma_j$ | A neuron's output weight and slope, appearing in $w_j\tanh(\gamma_j(x-x_j))$ |
| $a_j=w_j/d_j$, $\lambda_j=h\gamma_j$ | The rescaled variables updated in scaled training; $d_j$ is the prescribed readout scale and $h$ is center spacing. Reported bandwidths use $\lvert\lambda_j\rvert$. |
| Training-window RMS | RMS over every training sample and every update in the preceding 20,000 steps |
| Detached refit | Validation RMS after fitting only the saved geometry's readout by truncated SVD; never fed into training |

## The single prescription being tested

Both methods use the same network, initial predictions, data, loss, and fixed centers. The difference is which variables the optimizer updates. **Unscaled training** updates $w_j$ and $\gamma_j$ directly. **Scaled training** updates $a_j=w_j/d_j$ and $\lambda_j=h\gamma_j$, then obtains the network parameters through $w_j=d_ja_j$ and $\gamma_j=\lambda_j/h$. These are rescaled versions of the same parameters, not additional neurons or parameters. A change $\Delta a_j$ produces $\Delta w_j=d_j\Delta a_j$; a change $\Delta\lambda_j$ produces $\Delta\gamma_j=\Delta\lambda_j/h$.

Earlier descriptions called scaled training “reparametrized” or “Both,” and unscaled training “Raw” or “the control.” Those were aliases for these two methods. This report uses **scaled training** and **unscaled training** throughout; the formulas specify the scaling precisely.

Use center spacing $h=2/N$, halo radius $R=\lceil\sqrt N\rceil$, and independent signed slopes. In scaled training, collect the readouts and output bias into $\mathbf c$ and train jointly using

$$
\mathbf c=D\mathbf a,\qquad \boldsymbol\gamma=\boldsymbol\lambda/h,
\qquad \boxed{\eta_a=\eta_\lambda=\eta}.
$$

Here $\mathbf c$ includes the output bias. The fixed diagonal $D$ comes from the reference envelopes in the [scale note](../../../docs/correcting_scales.md), evaluated once at $\lambda_{\rm ref}=0.25$, including bias and halo allowances. Ordinary diagonal entries scale as $\sqrt h$. Adam in these coordinates therefore supplies physical update prefactors $\eta d_j$ and $\eta/h$, with physical epsilons $10^{-8}/d_j$ and $h\,10^{-8}$. These transformations are applied automatically through the parametrization. There is no independent multiplier for either block.

Hold initialization fixed: physical slopes use tanh Xavier, $\widetilde\gamma_{j,0}=(5/3)\sqrt{2/(W+1)}\,g_j$; physical readouts use the note's reference-envelope law $\widetilde w_{j,0}=\alpha_j\operatorname{sign}(\xi_j)$ with $d_j=\sqrt{\alpha_j}$ and independent standard normal draws $g_j,\xi_j$; output bias starts at zero. Initial slope signs are absorbed into readouts without changing the function. Unscaled training starts from exactly the same physical network and trains $(\mathbf c,\boldsymbol\gamma)$ with one shared $\eta$. Both methods use full-batch FP64 Adam, moments $(0.9,0.999)$, and trained-coordinate epsilon $10^{-8}$ throughout.

The baseline evidence uses existing pilot runs satisfying that contract: 10 runs of scaled training at $\eta\in\{10^{-5},10^{-4},10^{-3},10^{-2},10^{-1}\}$ and six runs of unscaled training at $\eta\in\{10^{-4},10^{-3},10^{-2}\}$, each with seeds 0 and 1. All completed 320,000 updates and have verified complete traces. The user's shared-rate clarification supersedes the earlier independent-rate search in the protocol. Unequal-rate results appear only in the explicitly identified historical comparison below, to explain the earlier reported errors.

The target is $\sqrt2\sin(2\pi x)$ on $[-1,1]$, with $N=512$, $R=23$, and $W=559$. Training uses 8,193 endpoint-inclusive samples; diagnostic validation uses 32,768 midpoint samples. The final test grid remains unused. Detached solves use the same $D$ and relative SVD cutoff $10^{-12}$ in both arms. This is an optimization study, not a final held-out evaluation.

### Effective learning rates in physical parameters

At $N=512$, $h=1/256$, $d_{\rm ordinary}=0.09307517$, and $d_{\rm bias}=3.28083978$. The shared coordinate rate $\eta=10^{-3}$ therefore gives the following physical Adam prefactors. These multiply Adam's normalized moments; they are not measured parameter displacements or rates of error reduction.

**Table 1. Physical Adam prefactors at the same shared base rate $\eta=10^{-3}$.**

| Physical parameter | Unscaled training | Scaled training | Scaled / unscaled |
|---|---:|---:|---:|
| Ordinary readout weights, including uncorrected halo | $10^{-3}$ | $9.3075\times10^{-5}$ | 0.0931 |
| Corrected halo readout weights | $10^{-3}$ | $9.3075\times10^{-5}$ to $1.0224\times10^{-3}$ | 0.0931 to 1.0224 |
| Output bias | $10^{-3}$ | $3.2808\times10^{-3}$ | 3.2808 |
| Physical slopes $\gamma$ | $10^{-3}$ | 0.256 | 256 |

Thus ordinary readout prefactors are 10.74 times smaller, slope prefactors are 256 times larger, and the output-bias prefactor is 3.28 times larger. The corrected halo is not uniformly slowed. Setting the shared rate to $10^{-2}$ multiplies every scaled prefactor in this table by ten without changing their ratios.

The physical epsilon thresholds also follow the coordinate map: $1.0744\times10^{-7}$ for ordinary readouts, $3.0480\times10^{-9}$ for the bias, and $3.90625\times10^{-11}$ for slopes, versus $10^{-8}$ throughout unscaled training. Hence the prefactor ratios alone are not exact ratios of actual updates. The [computed rate record](shared_rate_analysis/effective_rates.json) stores all 560 readout/bias prefactors and epsilons, slope settings, and source cases, calculated using `core.geometry` and `run.case_settings`.

## What the shared-rate runs show

**Table 2. Results at 320,000 updates; entries show seed 0 / seed 1. Training RMS uses updates 300k–320k.**

| Setup | Shared $\eta$ | Training-window RMS | Median $\lvert\lambda\rvert$ | Detached refit RMS |
|---|---|---|---|---|
| Unscaled training | $10^{-3}$ | 0.0312 / 0.0130 | 0.000507 / 0.000236 | $7.52\times10^{-4}$ / $2.41\times10^{-5}$ |
| Scaled training | $10^{-3}$ | 0.00212 / 0.00111 | 0.0329 / 0.0808 | $6.32\times10^{-12}$ / $5.15\times10^{-11}$ |
| Scaled training | $10^{-2}$ | 0.01098 / 0.01098 | 0.321 / 0.329 | $4.00\times10^{-8}$ / $1.42\times10^{-8}$ |

Among the five sampled shared rates, $10^{-3}$ has the lowest paired geometric mean of late-window training RMS. This is a descriptive choice within the existing grid, not a converged optimum or a final validation selection. Smaller rates leave large error and small median bandwidths; $10^{-1}$ has window RMS above 0.1. The complete [filtered rate table](shared_rate_analysis/summary.csv) retains every candidate.

At $\eta=10^{-3}$, scaled median bandwidths grow from 0.00882 / 0.0160 at 20k to 0.0231 / 0.0658 at 160k and 0.0329 / 0.0808 at 320k. The corresponding unscaled medians at 20k are 0.000492 / 0.000653 and remain small through 320k. Scaling therefore sustains substantially more geometry movement after the early fitting phase. The final scaled refits require readout $\ell_1$ norms of 16.6 / 16.2, so their accuracy is not obtained only with enormous coefficients.

This demonstrates accurate approximation with the learned features, not good conditioning of the full readout problem. The SVD retains only 204 / 313 of the 560 feature columns' singular directions at the stated cutoff. A highly accurate truncated solve does not imply that Adam can quickly find its readout coefficients.

<figure>
  <img src="shared_rate_analysis/figures/shared_rate_trajectories.png" alt="Two-seed trajectories of bandwidth, complete training-window error, and detached readout error for scaled training and unscaled training" style="max-width: 100%;">
  <figcaption>One initialization and one shared rate per run. Blue is scaled training at the lowest-error sampled shared rate; orange is scaled training with that shared rate increased tenfold; gray is unscaled training at the same base rate as blue. Solid and dashed lines show the two seeds separately. The dotted bandwidth reference is 0.25, not a required median for a heterogeneous learned dictionary. The first training window includes initialization.</figcaption>
</figure>

Useful features do not yet translate into a precise live fit. At $\eta=10^{-3}$, the final three training windows are 0.00246, 0.00230, 0.00212 for seed 0 and 0.001128, 0.001115, 0.001108 for seed 1, with substantial within-window oscillations. Error is still improving, especially in seed 0; these runs must not be called converged. Raising the shared rate to $10^{-2}$ brings median bandwidths above 0.25 but worsens both sustained training error and the final detached fit. Merely reaching that median does not establish the intended geometry.

### Why earlier errors reached $10^{-4}$

There are two separate comparisons. First, endpoint error can be much smaller than sustained error: the current shared-$10^{-3}$ runs have final validation RMS $5.58\times10^{-4}$ / $6.27\times10^{-5}$, but training-window RMS $2.12\times10^{-3}$ / $1.11\times10^{-3}$. A favorable oscillation phase is not a sustained $10^{-4}$ result.

Second, earlier scaled training with unequal rates did achieve sustained RMS $1.48\times10^{-4}$ / $1.52\times10^{-4}$ on the same target, width, initialization, seeds, and 300k–320k window. It used $(\eta_a,\eta_\lambda)=(10^{-4},10^{-2})$. Its physical ordinary-readout, bias, and slope prefactors were respectively $9.3075\times10^{-6}$, $3.2808\times10^{-4}$, and 2.56. Relative to scaled training with shared $\eta=10^{-3}$, every readout/bias prefactor was ten times smaller and the slope prefactor ten times larger. This extra ratio is outside the shared-rate prescription. Earlier unscaled training with unequal physical rates, $9.3075\times10^{-6}$ and 2.56, likewise reached approximately $10^{-4}$; it did not use one shared rate.

The lower live error did not reflect better features in the comparison of scaled training at different rates: those earlier detached errors were about $2\times10^{-7}$, with refitted coefficient norms above 20,000. At the same geometry coordinate rate $10^{-2}$, reducing the readout coordinate rate from $10^{-2}$ to $10^{-4}$ lowered sustained error from about 0.011 to 0.00015, even though the final refit worsened. This supports sensitivity to readout step size and joint optimizer dynamics; it does not isolate a particular oscillation mechanism or establish the shared-rate prescription. The [rate record](shared_rate_analysis/effective_rates.json) preserves the historical configurations and comparable error measurements.

## What the residual and gradient evidence says

At $\eta=10^{-3}$, detached errors are already $2.8\times10^{-11}$ / $1.4\times10^{-11}$ at 20k. Later bandwidth growth therefore does not demonstrate escape from a persistently inaccurate feature space. At 320k, the norms of the bandwidth-gradient component shared with readout fitting are $2.3\times10^{-3}$ / $1.5\times10^{-6}$; the components outside the retained readout span are only about $10^{-16}$ / $10^{-17}$. At these checkpoints, geometry is responding overwhelmingly to readout-accessible error. Continued parameter movement is not evidence that the out-of-reach residual is driving productive geometry improvement.

The largest out-of-span residual Fourier band shifts from DFT indices 4–7 initially to 32–63 / 128–255 at 320k, while total out-of-span residual RMS falls to approximately $7\times10^{-12}$ / $5\times10^{-11}$. This is consistent with depletion of coarse residual, but does not establish an exponential frequency-dependent gradient law. The tiny perpendicular gradients are numerically sensitive; their exact signs are not interpreted. Changing the SVD cutoff from $10^{-10}$ to $10^{-14}$ gives final refit errors between roughly $10^{-9}$ and $10^{-13}$, preserving the large gap from the live error. The [signed spectral histories](shared_rate_analysis/spectral_history.json), [gradient diagnostics](shared_rate_analysis/mechanism.json), and [cutoff checks](shared_rate_analysis/cutoff_sensitivity.json) retain both seeds.

Increasing the same shared rate to $10^{-2}$ also acquires accurate features by 20k, but subsequently loses accuracy as bandwidths grow. Thus the evidence supports early geometry acquisition; it does not yet demonstrate the intended sustained useful geometry learning. Convergence, other widths and targets, and tuning the single shared rate between the sampled decades remain unresolved.

## Evidence and reproduction

The [selection record](shared_rate_analysis/selection.json), [window records](shared_rate_analysis/windows.json), and [checkpoint metrics](shared_rate_analysis/checkpoint_metrics.json) identify the included cases and measurements. The source is the saved pilot checkpoints, not the later crossed study. All new work for this report is detached analysis; no training updates were added or readouts replaced.

Run `python -m experiments.expD06_fixed_center_scales.shared_rate_analysis --root /workspace/junmiaoh/experiments/precision-mlps/runs/pilot` inside an eight-CPU Slurm allocation with `JAX_PLATFORMS=cpu`, `CUDA_VISIBLE_DEVICES=''`, and two BLAS threads. The script verifies the exact 16-case subset, reuses the existing diagnostics, and emits data and the figure. CPU Slurm job 305 completed the export after a path-type correction to job 304; neither used GPUs. All 27 focused experiment tests pass. The [numerical audit](shared_rate_analysis/numerical_audit.json) records reconstruction and CPU/GPU prediction checks. The report is authored directly from those inspected artifacts. Earlier broad analyses remain in the evidence directories and Git history; they are not used to establish the shared-rate result.
