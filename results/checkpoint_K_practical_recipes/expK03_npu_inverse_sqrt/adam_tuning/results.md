# Adam baseline: width/rate pilots and nine 100k runs

Selected baseline: **width 16, lr=0.01, checkpoint 85,250**, test relative L2 **0.971605074**. Selection uses validation scores, never the reported test quadrature.

Pilot widths: 8, 16, 32, 64, 128, 256, 512, 1,000. Each is trained for 10k steps at learning rates 1e-2, 1e-3, 1e-4 (24 pilots). Rank each width by its lowest validation error in steps 5k–10k over those three rates.

The resulting candidate widths are **[8, 16, 128]**. All nine width/rate combinations are trained from initialization to 100,000 updates. The plotted checkpoint is selected only from steps 50k–100k, so it is not a short-run or initialization result.

| Width | LR | Selected step | Test relative L2 | Final relative L2 |
|---:|---:|---:|---:|---:|
| 8 | 0.01 | 97,250 | 1.6329077 | 1.6547869 |
| 8 | 0.001 | 95,250 | 2.1307954 | 2.1682354 |
| 8 | 0.0001 | 73,500 | 4.0407526 | 4.1364044 |
| 16 | 0.01 | 85,250 | 0.97160507 | 1.0379933 |
| 16 | 0.001 | 95,000 | 1.7300817 | 1.770831 |
| 16 | 0.0001 | 100,000 | 4.3270346 | 4.3270346 |
| 128 | 0.01 | 96,500 | 1.0342163 | 1.1207966 |
| 128 | 0.001 | 93,000 | 1.6339161 | 1.639024 |
| 128 | 0.0001 | 100,000 | 3.0757644 | 3.0757644 |

## Shared protocol

Ordinary centered tanh MLP, all weights/biases trained, PyTorch default initialization, seed 0, float64. The C17 target, N=256/R=32 halo-center dataset (4,815 nodes), data seed 10000, batch 100, Adam betas=(0.9,0.999), epsilon=1e-8, and unweighted MSE are unchanged from the previous Adam reference. No warmup, LR decay or target rescaling. Evaluation is physical dx relative L2 on [1e-4,1].

The halo training loss and original-interval evaluation metric have different domains/weights. This is the specified halo-fitting protocol, not an assertion of globally optimal Adam training. NPU/RealNPU curves retain their earlier 20k budgets; the Adam baseline now receives this larger tuning/training budget.

## Validation and saved artifacts

Every selected checkpoint was reloaded and scored with independent quadrature orders 256 and 512. Each long run completed all 100k steps. Its first 10k updates match the corresponding pilot model and loss history bitwise. Target arrays match the saved C17 halo data bitwise. The eleven implementation tests include selection-window enforcement and validation-only width selection.

See `specification.json`, `width_selection.json`, `pilot_table.csv`, `long_run_table.csv`, `selected_reference.json`, `validation.json`, and per-run configs, model/optimizer checkpoints, histories and predictions. All previous 20k Adam references and both NPU sweeps are preserved.
