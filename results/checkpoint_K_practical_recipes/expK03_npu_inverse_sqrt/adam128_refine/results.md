# Adam refinement at 128 neurons

**Archived exploratory result, excluded from the current baseline.** The user
subsequently disallowed the multiscale initializer and required traditional
initialization only. The current figure uses `../adam_standard/`; do not present
the result below as a default-initialization Adam baseline.

The selected model reaches **3.49908947e-4 physical-domain relative L2 (0.03499%)** after 100,000 steps. Width remains 128, with 385 trainable parameters. No capacity expansion was run after the user accepted this result.

## Winning recipe

- Random multiscale hidden initialization: 112 tanh features with stratified, randomly jittered log centers across [epsilon/2, twice the outer extension scale], plus 16 broad features across the full signed halo.
- Local physical slopes scale inversely with the center and log spacing, with multiplicative random jitter. Initial centered-coordinate slopes span approximately 0.073 to 101,251. The original trained model had maximum slope about 25.
- The readout starts at the normal PyTorch random initialization. All hidden weights, hidden biases, readout weights and readout bias remain trainable by Adam.
- Batch size increased from 100 to 512; sampling remains uniform over the same halo-center indices.
- Adam LR 0.003 for the first 50k updates, cosine decay to 1e-5 over the next 50k. Betas=(0.9,0.999), epsilon=1e-8, seed 0, float64, no warmup or weight decay.
- No output normalization in the winning run (output_scale=1). Output-normalized variants were screened and retained in the table.

The target, C17 extension, fixed N=256/R=32 halo grid, label values, relative sample weights, affine input centering, and physical dx evaluation metric are unchanged. This is a domain-scale-informed initializer: it uses the location scales and halo extent, not QI coefficients or a coefficient solve. It should be described as **Adam with random multiscale initialization**, not default PyTorch initialization.

At x=1e-4 the improved model predicts 99.9895654 for target 100; the prior 128-unit trained baseline predicted about 58.7361. The prior best 128-unit error was 1.03422, about 2,956 times larger. Larger batches/decay alone reached 0.98307. Multiple hyperparameters differ between trials, so the study is a practical screen, not a perfectly isolated causal ablation.

| Trial | LR | Output scale | Selected step | Test relative L2 |
|---|---:|---:|---:|---:|
| standard_output100_decay | 0.003 | 100 | 97,500 | 1.71545646 |
| multiscale_gain1 | 0.003 | 1 | 100,000 | 0.000349908947 |
| standard_batch512_decay | 0.01 | 1 | 92,000 | 0.983073801 |
| multiscale_gain4 | 0.003 | 1 | 98,000 | 0.00332990586 |
| multiscale_gain1_output100 | 0.001 | 100 | 99,500 | 0.000357533853 |
| multiscale_gain4_output100 | 0.001 | 100 | 98,500 | 0.00513797992 |

## Validation

Selection used order-32 validation error over steps 50k–100k. Each selected checkpoint was independently scored with order-256 and order-512 quadrature. The selected errors are 0.00034990894694916466 and 0.00034990894694918965. All six halo target arrays match the original saved inputs/targets bitwise. Each variant folds into an ordinary Linear/Tanh/Linear model without adding trainable parameters. Fourteen implementation tests passed, including schedule endpoints and preservation of the initial physical function under output normalization.

All six 100k runs, full checkpoints every 25k, selected optimizer/model/RNG state, histories and predictions are retained. See summary.csv, validation.json, selected_reference.json, and multiscale_gain1/exported_mlp.pt.

Training is unweighted MSE over the halo-center dataset, whereas the reported score is physical-domain relative L2 on the original interval. The result establishes a much stronger Adam baseline for this protocol, not an accuracy lower bound for Adam or tanh networks. NPU/RealNPU curves retain their previous training recipe.
