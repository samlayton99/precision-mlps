# Figure 4 width sweep: launch record

Launched on 2026-09-25 at 08:02:14 UTC. Both workers passed the independent Adam/GD update checks, executed training, and saved parameter/optimizer checkpoints before handoff. This records a verified launch, not completed experiments.

The sweep adds total widths 128, 256, and 1024 to the completed width-512 comparison. Each width uses five seeds, the same normalized mixed-sine target, 2,048 training midpoints, FP64, and five million updates. Halos count within the total width: $(N,R)=(105,11),(225,15),(961,31)$, respectively, with $W=N+1+2R$ and $h=2/N$. Initialization follows the existing affine-Xavier convention keyed by seed and the budget-derived number of intervals.

## Candidates and evidence

The final width-512 candidate grids are reused exactly:

- Adam, full-horizon cosine: initial rates 0.05 and 0.02.
- Adam, constant: rates 0.02 and 0.002.
- GD, constant and full-horizon cosine: rates 0.2 and 0.5.

This gives 120 new seed/recipe trajectories. It does not repeat the earlier broad learning-rate screen at each width. After completion, choose one recipe per width and optimizer by median final error on the 4,096-point validation grid, requiring finite endpoints for all five seeds. Retain failed candidates and do not splice schedules or select separate recipes per seed. The 8,192-point grid is a resolution check.

Raw error and slope RMS are stored at every update; full parameters and optimizer state are saved every 10,000 updates. Those parameters support arithmetic mean bandwidth and neuron-population standard deviation for the revised figure. The target arrays are copied directly from the completed comparison. The preparation check reproduces its initialization to floating-point rounding tolerance.

## Execution and limits

Slurm is absent on the restored pod. The user explicitly authorized direct root execution in that case. Two detached workers expose one fixed GPU each; GPU 2 is unused.

| Worker | Physical GPU | Widths, in order | Timeout process |
| --- | --- | --- | --- |
| 0 | 0 | 128, 256 | 10004 |
| 1 | 1 | 1024 | 10005 |

Each width runs Adam cosine, Adam constant, then GD. Each worker is enclosed by GNU `timeout` for 10,790 seconds, with a five-second kill grace applying to its process group. The maximum combined worker time is 21,590 GPU-seconds, below six GPU-hours, including startup, compilation, checks, and I/O. The initial throughput suggests roughly two hours to complete the longer worker; this is an estimate, not a completion guarantee.

At the 08:03:33 UTC launch check, width 128 had reached 568,000 updates with a durable checkpoint at 560,000; width 1024 had reached 159,000 with a durable checkpoint at 150,000. All ten active seed/recipe cases in each batch had finite errors. GPU utilizations were 89%, 97%, and 0%. This status is a snapshot, not a live monitor.

## Locations and follow-up

Remote root:

```text
/workspace/junmiaoh/experiments/precision-mlps/runs/figure4_width_sweep_20260925
```

`lane0.log` and `lane1.log` record dispatch and completion. Per-width `joint_*.log` files record progress; `lane*_completed.json` records finished groups and exit codes. Require a successful `summary.json` with `completed_updates=5000000` before treating a run as complete: the raw trace files are preallocated to the planned horizon.

The [launcher](../experiments/expD36_frozen_gamma_probe/figure4_width_sweep.py) is committed as `83f17c2`. Its remote copy and the unchanged preparation/training scripts are hashed in the [local launch record](../output/diagnostics/figure4_width_sweep_20260925/launch.json), which also retains configurations, geometry, process IDs, GPU UUIDs, and the verified progress snapshot. The restored SSH gateway is the one supplied by the user.

After the runs finish, select complete recipes, recover the existing width-512 parameter trajectories, and assemble the three panels specified in [the Figure 4 revision note](section34_pending_figure_updates.md). The Section 3.5 writing package is already complete; these new measurements will determine the width-scaling claim.
