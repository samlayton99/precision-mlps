# D40 mode diagnostics

`gamma_diagnostics_initial.json` verifies the seed-0 centered reference and the original 30:1 mixtures for all four tasks (12 reconstructed models). Every mixed layer has 256 rows at `0.1g` and 256 at `3g`, where `g` is the paired reference layer's median row norm. Row directions, zero-crossing centers, and the random readout match the paired reference. Unmixed layers retain their weights and biases exactly.

The two `gamma_initial_*.png` figures use common logarithmic bins and physical row-norm limits. Heights are percentages of all 512 rows. The first layer of `mix_last` is shown in gray as one group; its stored high/low mask is bookkeeping, not a gamma mixture. The figures explicitly identify the original 30:1 range and do not describe the later 10:1 follow-up.

On the first at most 2,048 fitting inputs, 63–70% of the original high group's activations satisfy `|tanh(z)| > 0.99`, versus none in the low group. This measures the operating range at initialization; it does not establish that the high neurons cannot learn or that saturation causes subsequent performance differences. Available pilot traces were checked against reconstructed initial fitting predictions. No validation or test predictions were used in this initial audit.

`figures/group_initial_saturation.png` plots these saved measurements on common 0–100% axes. It uses no new model evaluation. Recreate it with `python -m experiments.expD40_bimodal_gamma.diagnostics saturation`; `gamma_diagnostics_saturation.json` records the source JSON hash and exact plotted values.

The final diagnostic supports both original and mild mixtures using each saved run's configuration and the follow-up dispatcher. It checks the locked source manifests, screening hashes, complete checkpoint identity, data metadata, and initial/final fitting and validation predictions before writing outputs. It retains original group membership and records physical gammas, own-row direction cosines, norm ratios, and fitting-input activation diagnostics. Layer-2 cosine is measured in the fixed hidden-neuron coordinate system, whose upstream representation may also move.

The final audit ran after all 45 confirmation JSON/checkpoint pairs were complete. To reproduce the selected seed-0 diagnostics, run from the repository root:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m experiments.expD40_bimodal_gamma.diagnostics final --ablate
```

This produces per-task 2×2 gamma histograms, 1×2 cosine histograms, group movement and absolute/relative displacement summaries, and `gamma_diagnostics_final.json`. The optional `--ablate` replaces each final hidden-layer group's activations by its fitting-data means, without readout refitting. Its errors measure sensitivity to a representation shift, not causal importance. It uses fitting and validation data only. All four selected seed-0 reconstructions passed the initial/final prediction, source, selection, data, and checkpoint checks; no verification issue was found.

## Selected-mixture pilot endpoints

The separate `pilot --ablate` mode ran on the four locked seed-0, 10,000-step pilot checkpoints. It uses their saved original/mild configurations, verifies source and selection hashes, and reproduces initial and endpoint fitting/validation predictions. Outputs are `gamma_diagnostics_pilot.json` and figures ending in `_pilot.png`. These describe the 10k endpoints, not each run's validation-selected checkpoint or the separate 20k confirmation. No test predictions or new training choices were made.

Every originally low row still has a smaller physical gamma than every originally high row in each mixed layer. The initial peaks broaden, and low first-layer rows can move substantially without closing the gap:

| Task and mixed layer | Median norm multiplier, low / high | Median cosine to own initial row, low / high |
|---|---:|---:|
| Airfoil, layer 2 | 1.196 / 1.081 | 0.835 / 0.948 |
| Kin8nm, layer 1 | 1.758 / 1.019 | 0.723 / 0.989 |
| Kin8nm, layer 2 | 1.028 / 1.029 | 0.976 / 0.983 |
| SARCOS, layer 2 | 1.168 / 1.032 | 0.840 / 0.982 |
| Superconductivity, layer 1 | 4.195 / 1.054 | 0.278 / 0.975 |
| Superconductivity, layer 2 | 1.043 / 1.005 | 0.797 / 0.988 |

These relative changes must be separated from absolute weight movement. `group_displacement_pilot.png` compares `||w_end-w_0||₂` and that quantity divided by `||w_0||₂`, with per-row values and group quantiles saved in the JSON. In all six mixed layers, the high group's median absolute displacement is larger, while the low group's median relative displacement is larger. For example, Superconductivity layer 1 has absolute displacements 0.330 / 0.595 for low / high, but relative displacements 4.013 / 0.241. Kin8nm layer 1 has absolute displacements 0.115 / 0.142 and relative displacements 1.273 / 0.157. A small starting norm allows a large rotation or relative change with less absolute movement; none of these metrics alone establishes which group learns more.

Airfoil and SARCOS have an unmixed first layer; Airfoil, Kin8nm, and SARCOS use the mild 10:1 mixture, while Superconductivity uses 30:1. In layer 2, the high group's saturated-activation fraction rises from 16.5% to 66.6% on Airfoil, 17.7% to 31.8% on Kin8nm, 15.9% to 55.4% on SARCOS, and 67.4% to 85.1% on Superconductivity. The low group's final fractions are 0.66%, 0%, 0.20%, and 2.38%, respectively. Gamma movement alone cannot explain these changes: biases, row directions, and upstream representations also train.

Replacing the final layer-2 low group by fitting means raises validation MSE by 3.04×, 2.48×, 4.90×, and 1.06× on those four tasks. Replacing the high group raises it by 15.37×, 16.80×, 51.48×, and 10.76×. Both groups affect these fixed fitted predictors, with the weakest low-group sensitivity on Superconductivity. This is an ablation with a shifted representation, not a refitted capacity comparison or proof of causal importance. Selection and confirmation are unchanged by the audit.

## Selected-mixture confirmation endpoints: 20,000 steps

These results describe one model seed per task, at the fixed 20k endpoint. They are separate from the three-seed performance comparison and from validation-selected checkpoints. The 20k runs start from the paired initialization with a schedule set for 20k; they are not continuations of the 10k pilots. Outputs without `_pilot` are the confirmation diagnostics.

The original low/high norm ranges remain completely separated in all six mixed layers: every originally low row has a smaller final gamma than every originally high row. The largest ratio of a low-group maximum to a high-group minimum is 0.447. Median high/low gamma ratios are 7.79 for Airfoil layer 2, 5.08 and 10.06 for Kin8nm layers 1 and 2, 7.13 for SARCOS layer 2, and 4.98 and 27.11 for Superconductivity layers 1 and 2. This records persistent separation by original assignment; it does not require that the evolved distribution has exactly two peaks.

| Task and mixed layer | Median norm multiplier, low / high | Median cosine, low / high | Median absolute displacement, low / high | Median relative displacement, low / high |
|---|---:|---:|---:|---:|
| Airfoil, layer 2 | 1.480 / 1.153 | 0.689 / 0.902 | 0.212 / 0.988 | 1.065 / 0.496 |
| Kin8nm, layer 1 | 2.076 / 1.054 | 0.686 / 0.982 | 0.141 / 0.194 | 1.555 / 0.215 |
| Kin8nm, layer 2 | 1.044 / 1.050 | 0.962 / 0.972 | 0.065 / 0.565 | 0.285 / 0.246 |
| SARCOS, layer 2 | 1.558 / 1.111 | 0.645 / 0.939 | 0.307 / 0.940 | 1.258 / 0.385 |
| Superconductivity, layer 1 | 6.700 / 1.112 | 0.201 / 0.950 | 0.538 / 0.881 | 6.549 / 0.358 |
| Superconductivity, layer 2 | 1.128 / 1.019 | 0.595 / 0.972 | 0.233 / 1.788 | 0.941 / 0.241 |

The high group moves farther in absolute weight space in every mixed layer, despite retaining its directions better and having smaller relative displacement. The low group starts closer to zero, so rotation and relative displacement alone would give an incomplete account. Airfoil and SARCOS layer 1 retain their unmixed label; their median norm multipliers are 1.187 and 1.340, and their median direction cosines are 0.966 and 0.860. These layers train normally after initialization.

Low initial gamma also does not guarantee an unsaturated final activation. On the same fitting-input probe, layer-2 saturated fractions at 20k are:

| Task | Originally low | Originally high |
|---|---:|---:|
| Airfoil | 35.5% | 77.5% |
| Kin8nm | 0.0% | 41.0% |
| SARCOS | 46.4% | 70.3% |
| Superconductivity | 34.9% | 89.6% |

These use `|tanh(z)| > 0.99` over all probed inputs and rows in each group. They reflect the trained biases and upstream representations as well as row norms and directions.

At the 20k endpoint, replacing layer-2 groups by fitting means gives the following validation-MSE multipliers:

| Task | Replace low group | Replace high group |
|---|---:|---:|
| Airfoil | 1.534× | 16.540× |
| Kin8nm | 2.455× | 18.299× |
| SARCOS | 1.293× | 82.829× |
| Superconductivity | 1.009× | 10.631× |

The fixed predictors are more sensitive to replacing the high group, and Superconductivity is only weakly sensitive to replacing its low group. No readout is refitted, so this is not a comparison of representational capacity after removing neurons. These movement and ablation observations do not establish a generalization benefit from a mixture; that question belongs to the locked mixture/control performance comparison. No candidate was changed or selected using this audit.
