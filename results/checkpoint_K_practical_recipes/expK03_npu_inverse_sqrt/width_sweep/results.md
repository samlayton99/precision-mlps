# NPU and RealNPU capacity sweep

All runs use the same bounded C17 extension, fixed QUILL halo dataset, float64 precision, seed 0, batch 100 and 20,000 Adam updates at lr=0.001. Only model capacity changes. No extra seeds or learning-rate search.

- Best NPU by validation: **0.067307562** relative L2 (6.7308%), 2,048 units / 6,145 parameters, step 18,000.
- Best RealNPU by validation: **0.25553421** relative L2 (25.553%), 6 units / 13 parameters, step 750.

QUILL reaches about 5e-14 at 4,815 tanh neurons / 14,446 parameters. The fixed-width Adam MLP reference has the same architecture/parameter count, with best trained error 2.55313. These are different construction/training procedures, not identical access to derivative information.

| Model | Units | Parameters | Best trained test relative L2 | Selected step | Final relative L2 |
|---|---:|---:|---:|---:|---:|
| NPU | 6 | 19 | 0.58503814 | 16,750 | 0.68781584 |
| NPU | 32 | 97 | 0.60594801 | 6,000 | 0.93419463 |
| NPU | 128 | 385 | 0.37657954 | 5,250 | 0.8912841 |
| NPU | 512 | 1,537 | 0.41186512 | 19,500 | 0.67679427 |
| NPU | 2,048 | 6,145 | 0.067307562 | 18,000 | 0.18071973 |
| NPU | 4,815 | 14,446 | 0.19757252 | 12,500 | 0.23321049 |
| RealNPU | 6 | 13 | 0.25553421 | 750 | 1.1575537 |
| RealNPU | 32 | 65 | 1.5270547 | 5,000 | 2.7365391 |
| RealNPU | 128 | 257 | 1.861354 | 12,250 | 1.9253008 |
| RealNPU | 512 | 1,025 | 1.7832013 | 11,750 | 2.0559424 |
| RealNPU | 2,048 | 4,097 | 2.5967808 | 19,000 | 2.9420196 |
| RealNPU | 4,815 | 9,631 | 4.482062 | 3,250 | 7.6795475 |
| RealNPU | 7,222 | 14,445 | 4.7109445 | 3,250 | 8.1934582 |

## Protocol and validation

The halo samples exactly match the selected 4,815-neuron QUILL construction (15 blocks, N=256, R=32), including repeated physical locations and both halos. All widths receive the same target values and sample-index RNG sequence. The extension-node range is [-0.4096,3.6864]; evaluation is physical dx relative L2 on [1e-4,1].

Checkpoints are selected by minimum positive-step order-32 validation error, logged every 250 updates. Independent order-256/512 quadrature evaluates every selected checkpoint. Every model was reloaded and its score, parameter count and selection step verified. Ten implementation tests passed.

The widest NPU has 14,446 parameters and the widest RealNPU has 14,445, matching QUILL to within one parameter. Width-6 data are reused from the earlier run; all other points are newly trained. The parameter-axis figure shows measured unsmoothed curves, with white centers marking sampled capacities.

This is one seed and one training recipe per width. Nonmonotonic curves reflect the measured runs; they do not prove a capacity limit or establish the best possible optimizer/hyperparameter configuration. Wider models may need different training budgets. QUILL uses derivative samples and an anchor; NPU/RealNPU use function values. Other QUILL sweep widths retain their original 32-cell halo recipe; the shared halo reference matches exactly at 4,815 neurons.

Artifacts: `npu_capacity_sweep.png` / `.pdf`, `summary.csv`, `summary.json`, `findings.json`, `validation.json`, `sweep_specification.json`, `figure_provenance.json`, and per-width histories, prediction arrays, initial models, optimizer checkpoints, and selected model checkpoints.
