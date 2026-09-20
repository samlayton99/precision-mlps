# Published conditioning evidence

This directory accompanies the [constant-rate GD report](../conditioning_results.md). It contains the completed search at 100k updates and the selected continuations at one million and 13 million updates. The report defines the measurements and their limits.

The committed evidence includes:

- Search and continuation summaries in JSON and CSV, case lists, paired-initialization checks, and 20k-window MSE arrays.
- Per-case mechanism records, checkpoint spectral arrays, projection audits, and the uniform-bandwidth reference arrays.
- Figures and MP4 animations. The videos contain the same frames as the larger local HTML players; `continuation_final/animation_verification.json` records frame counts and update ranges.
- The four selected networks' initial and 13-million-update checkpoints, reference geometry, case settings, and resumable states in `final_states/`.
- Source hashes, test results, and Slurm allocation accounting.

Full parameter histories, dense per-update arrays, and embedded-frame HTML players remain outside Git. Original training data remains at `/workspace/junmiaoh/experiments/precision-mlps/runs/conditioning/` on the experiment host; its detached export is under `runs/conditioning_analysis/`. Local copies of the large exported arrays remain beside this bundle. No training data was deleted to prepare publication.

The [experiment instructions](../../../../experiments/expD06_fixed_center_scales/README.md#constant-shared-rate-sweep-with-neighbor-differences) describe reproduction. Rebuilding complete histories or dense-window diagnostics requires the original training records. Pass `--end 13000000` for the final comparison. Regenerating all plots and animations requires those exported histories and dense arrays in addition to this committed subset.
