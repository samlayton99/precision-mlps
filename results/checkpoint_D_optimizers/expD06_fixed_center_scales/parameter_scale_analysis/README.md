# Parameter-scale normalization evidence

The [report](../parameter_scale_results.md) interprets this study. It compares
collective normalization, $c=Da$, with parameter-scale normalization, $c=D^2u$,
using GD and Adam with one constant shared rate. There are 64 distinct trials:
57 finite 100k runs and seven recorded nonfinite failures. Sixteen selected
trajectories reach 2.3 million updates. Total allocation use is 7075 GPU-seconds.

**Evidence files and their roles.** Training MSE selects rates; midpoint and
doubled-grid evaluations are diagnostic, not held-out model-selection data.

| Location | Contents |
|---|---|
| `selected.json`, `confirmation.json`, `continuation.json` | The exact selected rates and subsequent case matrices. |
| `transfer/sweep_summary.csv` | Every 100k comparison, including failed cases and bandwidths at matched rates. |
| `physical_scales.json` | The ordinary, bias, halo, and physical-slope update factors at both widths. |
| `allocation_ledger.json`, `slurm_accounting.txt` | Actual allocation charges, the common horizon, and the reason for stopping. |
| `source_hashes.json`, `environment_*.json` | Source identities and the training environment recorded inside Slurm. |
| `states/<case>/` | Case configuration, reference geometry, and complete initial/final checkpoints and resumable optimizer states. |
| `final/summary.json` | Per-case histories of spectral, projection, refit, motion, epsilon, and numerical diagnostics. |
| `final/mechanism_summary.json` | Final-window Fourier percentages, modal fractions, counterfactual changes, and motion summaries for all 16 cases. |
| `final/<case>/parameter_history.npz` | Physical readouts and slopes at every saved checkpoint, centers, allowances, errors, and residual band energies. |
| `final/<case>/window_mse.npz` | Means and quantiles for every complete 20k training window. |
| `final/<case>/spectrum_*.npz` | Native/reference spectra, modal coefficients, Fourier gradients, and actual proposed updates at saved diagnostic checkpoints. |
| `final/<case>/dense_*.npz` | Exact finite-step budgets at 16 stratified states from each early/final 2048-update window. |
| `final/<case>/late_movie_parameters.npz` | The final 256 consecutive physical parameter states used by the width-512 late movies. |
| `final/*.mp4`, `final/*_animation.json` | Separate optimizer/seed movies and exact displayed steps and playback rates. |
| `checkpoint_verification.json`, `final/verification.json`, `pytest_complete.txt` | State/initialization checks, numerical reconstruction audits, and the 237-test regression result. |

The compact parameter histories store $\gamma$; recover $\lambda=(2/N)\gamma$
using the case configuration. The bias is column zero of `c`, followed by
readouts in increasing center order. `band_bounds` uses half-open DFT-index
intervals; each band includes both frequency signs, and index zero is DC.
`dense_*.npz` contains sampled mechanism calculations, not every raw dense state.
The accompanying JSON motion and loss-drift summaries use all 2048 updates.

The full final analysis was downloaded locally, including `history.npz` and
both full `dense_parameters_*.npz` archives per case. These larger archives,
along with intermediate pilot/one-million exports, are left outside Git.
The tracked export retains compact physical histories, the late movie inputs,
selected complete states, and all report-facing diagnostics. The artifact
manifest records hashes of both the compact export and the larger local inputs.

All raw per-update traces and dense gradient/update records remain at
`/workspace/junmiaoh/experiments/precision-mlps/runs/parameter_scale/` on the
training host. The final CPU export is `runs/parameter_scale_final_analysis/`
under the same experiment directory. Use the
[runner and analysis instructions](../../../../experiments/expD06_fixed_center_scales/README.md#parameter-scale-normalization-paired-gd-and-adam)
to reproduce the campaign. Regenerating movies with the existing entry point
uses the full local histories and dense parameter archives and requires FFmpeg.
