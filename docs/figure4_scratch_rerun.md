# Figure 4 rerun with verified off-pod backups

The rerun bypasses the unresponsive `/workspace` mount. Training and immutable backup segments use RAM scratch at `/dev/shm/figure4_scratch_20260926`. A laptop collector copies each segment over direct SSH, verifies SHA-256 hashes and checkpoint consistency, flushes the copied data to disk, and acknowledges receipt. Training waits for that acknowledgment every 250,000 updates and stops if it does not arrive within five minutes.

## Storage and protocol

Repeating widths 128, 256, 512, and 1024 with five seeds, eight optimizer/schedule/rate recipes, and five million updates requires 12.8 GiB for raw FP64 error/RMS traces and parameter checkpoints. Retaining both the ordinary arrays and immutable export segments needs approximately 26 GiB on the pod; allow 30 GiB. The laptop needs approximately 13 GiB for the exports, or 26 GiB if all ordinary arrays are reconstructed alongside them. The preflight found 352 GiB free in RAM scratch and 56 GiB free on the laptop.

The training target, observations, seed convention, power-of-two total widths including halos, candidate grid, and full-horizon cosine schedules remain those of the previous Figure 4 study. Rerunning width 512 supplies its missing population trajectories and keeps the new error and bandwidth panels from the same executions. Select one complete recipe per width and optimizer by median final validation error across all five seeds.

## Recovery trial

Both Adam and GD completed 30,000-update trials at width 128, preserving the intended five-million-update schedule. Exports were made every 10,000 updates for this trial. All six segments reached the laptop, passed their checksums, and reproduced saved output errors with an independent NumPy evaluation to within $2.3\times10^{-16}$. A deliberately corrupted copy was rejected. A separate missing-acknowledgment check stopped the exporter while retaining its files.

The initial restoration test exposed an incomplete checkpoint: the runner retained a cached gradient between updates but saved only parameters, optimizer moments, and the update count. Recomputing that gradient introduced small numerical differences that grew during continued Adam training. Checkpoints now include the cached gradient. This changes checkpoint contents, not the update equations.

After moving the original trial outputs aside, the laptop copies were uploaded into a separate restoration directory. For both Adam and GD, restoring update 10,000 and continuing to update 20,000 reproduced the uninterrupted execution **bit-for-bit**: parameters, both moment arrays, cached gradient, and output errors were identical. This verifies recovery in the tested environment; it does not assert bitwise reproducibility on different hardware or compiler versions.

- [Trial verification](../output/diagnostics/figure4_scratch_20260926/trial_v2_verification.json).
- [Restored continuation verification](../output/diagnostics/figure4_scratch_20260926/restore_v2_verification.json).
- [Backup exporter and collector](../experiments/expD36_frozen_gamma_probe/figure4_durable.py).

The environment uses Python 3.12, JAX/JAXlib 0.11.1, NumPy 2.5.1, and SciPy 1.18.0 with CUDA 12 packages. Direct SSH uses `root@213.181.111.129`, port `16865`, and the user's existing Ed25519 key. The user authorized direct execution without Slurm. GPU concurrency remains capped at two.

RAM scratch is temporary. The protection is the verified laptop copy, not the remote export directory. A pod failure can still lose work since the most recent acknowledged segment. The collector runs under `caffeinate` to prevent idle sleep; a network interruption or unavailable laptop causes training to wait and then stop rather than continue without backups.

## Full sweep

The full rerun launched at **2026-09-26 05:18:33 UTC**. GPU 0 runs width 1024. GPU 1 runs widths 512, 128, and 256 in that order. Each width executes Adam cosine, Adam constant, and the combined GD candidate batch. These are 160 seed/recipe trajectories, each with five million planned updates. GPU 2 is unused.

Each worker is enclosed by a 10,400-second timeout with a five-second process-group kill grace. Together these allow at most 20,810 GPU-seconds; reserving another 790 seconds for the completed trials and restoration checks keeps the total within six GPU-hours. Both workers have an outer deadline of **08:11:58 UTC**. The expected completion is earlier, but only complete exported groups count as completed experiments.

- Remote run root: `/dev/shm/figure4_scratch_20260926/production`.
- [Local inputs and durable exports](../results/checkpoint_D_optimizers/expD36_frozen_gamma_probe/figure4_scratch_20260926/production/).
- [Launch record](../output/diagnostics/figure4_scratch_20260926/launch.json), including worker PIDs, commands, resource bounds, and the hashes of all four executed source files.
- [Input and geometry checks](../output/diagnostics/figure4_scratch_20260926/preflight.json).
- [Local backup process](../output/diagnostics/figure4_scratch_20260926/backup_process.json) and [live backup log](../output/diagnostics/figure4_scratch_20260926/backup.log).

An export contains the contiguous raw error/RMS segment, every parameter snapshot in that segment, the complete terminal optimizer state, and a manifest with configuration and input hash. Concatenating segments in update order reconstructs the original arrays without interpolation. `step05000000/ack_sent` indicates a verified final segment; the collector exits after all twelve final groups have been acknowledged. The early trial directories are separate and must not be used as paper evidence.

## Finishing the interrupted batch

Eleven groups completed within the initial worker timeouts. Width-256 GD reached update 4,426,000 before its worker deadline; its most recent acknowledged export was update 4,250,000. The continuation starts from that verified export, preserving the original input, configuration, cached gradient, optimizer state, and global update counter. The interrupted ordinary arrays remain untouched. A fresh directory reconstructs the verified prefix, then appends updates through five million using the original schedule.

The resume implementation was checked on the archived Adam and GD trials: restoring update 10,000 and continuing through 20,000 reproduced the full output-error trace and terminal state bit-for-bit. The RMS diagnostic differed by at most $3.24\times10^{-16}$ relative, consistent with floating-point reduction rounding; it does not enter the updates. Figure 4's population means and standard deviations are calculated from saved parameters.

The finishing job uses GPU 0 alone with a 900-second timeout and the same off-pod acknowledgment gate. [Resume validation](../output/diagnostics/figure4_scratch_20260926/resume_cli_verification.json) and the [launch record](../output/diagnostics/figure4_scratch_20260926/resume_launch.json) record the executed checks, code hashes, and resource limits. [Array reconstruction](../experiments/expD36_frozen_gamma_probe/figure4_restore.py) verifies checksums, contiguous update intervals, input/configuration agreement, and checkpoint alignment before restoring the standard analysis layout.

The continuation completed in 222 seconds, and all 240 segments are now verified locally. All 160 seed/recipe trajectories reached five million updates. Conservative campaign accounting is below 5.13 GPU-hours. The [completed analysis and Figure 4](figure4_completed_sweep.md) report the results, population statistics, and interpretation; [completion metadata](../output/diagnostics/figure4_scratch_20260926/completion.json) records the resource calculation.
