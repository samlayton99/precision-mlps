# Modal execution of the force-plateau campaign

The numerical source is archived from `0f3385d`. The runtime overlays add
allocation checks, Volume commits, a two-worker scheduler, and a verification
pilot. Targets, seeds, FP64, learning rates, update counts, optimizer histories,
interventions, and checkpoint forecasts retain the queued protocol.

Use Modal client 1.5.5 and profile `kinematic-pretrain`. Prepare an isolated
source directory with `scripts/prepare_d34_modal.py`; supply the SHA-256 of the
input tarball measured on Runpod. The packager verifies the transferred archive,
extracts the queued source, applies only named runtime overlays, and records
source and input hashes. It performs no numerical work. Set
`D34_MODAL_SOURCE_DIR` to that directory for all Modal commands.

The input Volume is `d34-plateau-inputs-0f3385d`; workers mount it read-only.
The output Volume is `d34-plateau-outputs`, with one campaign directory and
exclusive ownership of each bundle. Every completed save and issued forecast
is explicitly committed; readers reload before accessing dependent bundles.
Input hashes are checked remotely before execution.

Run these phases in order with the same campaign identifier:

```bash
modal run --profile kinematic-pretrain experiments/expD34_readout_race/modal_campaign.py \
  --phase verify --campaign modal-20260922 --input-dir /path/to/prepared/inputs
modal run --profile kinematic-pretrain experiments/expD34_readout_race/modal_campaign.py \
  --phase pilot --campaign modal-20260922 --input-dir /path/to/prepared/inputs
modal run --detach --profile kinematic-pretrain experiments/expD34_readout_race/modal_campaign.py \
  --phase campaign --campaign modal-20260922 --input-dir /path/to/prepared/inputs
```

Before the pilot, inspect Runpod accounting and hold the pending GPU arrays
1025, 1036, 1038, 1040, and 1041. Save the scheduler output. The prepared inputs
must contain `cutover.json`, with `jobs` equal to `[1025, 1036, 1037, 1038, 1039,
1040, 1041, 1042, 1043]`, `gpu_seconds: 0`, and `state: "held"`. Nonzero Runpod
usage stops this launcher for explicit budget reconciliation. After the pilot
passes, cancel only those nine campaign jobs, save final accounting, and change
the cutover state to `"cancelled"` before launching the campaign. If the pilot
fails, stop its Modal app and release the held Runpod arrays. Do not issue a
fresh campaign identifier to bypass a failed GPU attempt's budget ledger.
For a manually reconciled pilot retry, record `prior_modal_campaign` in the
cutover file. The coordinator requires that pilot to be stopped and carries
its entire reservation into the new ledger; no failed allocation is refunded.

The sole GPU function requests H100, four CPU cores and 48 GiB, with hard CPU
and memory limits, serial inputs, at most two containers, and single-use
containers. The actual GPU is recorded because Modal may supply H200 at the
H100 price. No application retries are enabled. A repeated platform attempt
fails its persistent ownership claim instead of obtaining a new deadline.

Reservations charge each attempt's entire submission-to-stop interval, including
startup and a 30-second shutdown allowance. Successful early completion does
not return the reservation to the budget. The pilot reserves 900 GPU-seconds;
the unchanged campaign stages reserve 30,000. The 10-hour ceiling leaves 5,100
seconds for explicitly reconciled interruptions or resumes; this launcher does
not automatically spend that reserve. A CPU watchdog cancels calls at their
original deadlines and enforces an independent campaign deadline. Unexpected
failure stops all active campaign calls. Incomplete scientific runs are retained;
the existing completion gates prevent dependent stages from reading them.
The pilot coordinator has a 15-minute wall deadline to allow CPU dispatch and
storage RPCs; its two GPU inputs each retain their own 450-second reservation.

The spending calculation uses USD 0.001097 per H100-second, USD 0.0000131 per CPU
core-second, and USD 0.00000222 per GiB-second, plus a USD 4 CPU/build allowance.
At the full 10-hour GPU ceiling this is USD 49.21456 before credits. These are
conservative resource reservations, not a provider invoice. The launcher also
checks the USD 50 stop before reserving each attempt. Rates were checked against
[Modal pricing](https://modal.com/pricing) on 2026-09-22.

The pilot runs representative degree-9 and mixed-sine archived GD/Adam states
on two GPUs. It checks gradients and force reconstruction, saves full optimizer
and movement histories, terminates each producer process with SIGTERM, and
resumes its checkpoint on the other worker after a Volume commit/reload.
It compares all state fields with uninterrupted execution and records numerical
errors, device identities, and throughput. Existing D34 tests, including
forecast immutability and rejection of incomplete inputs, run on remote CPUs.

Create the local destination directory before downloading:

```bash
mkdir -p /path/to/local/results
modal volume get --profile kinematic-pretrain d34-plateau-outputs /modal-20260922 /path/to/local/results
```

The campaign directory contains CPU-test logs,
verification and pilot evidence, immutable attempt provenance, conservative
accounting, checkpoints, and the CPU endpoint audit. Modal's app page supplies
live worker logs. Interpret completed scientific evidence in the owning report;
launch success alone supports no new scientific conclusion.

## Runpod late-checkpoint follow-up

The supplemental Runpod work retains the combined 10 GPU-hour ceiling and
USD 50 stop. Its separately funded pod adds no campaign spending. Reserve the
entire Modal plan, including both pilot attempts: 31,350 GPU-seconds. Runpod
reserves 300 seconds for a one-GPU pilot and three 1,200-second seed bundles,
leaving 750 seconds for explicitly accounted recovery. Reservations are not
refunded on early completion. Permit at most two Runpod GPUs across the user's
jobs, alongside the existing two Modal workers; use only Slurm-assigned GPUs.

`plateau_followup prepare` adapts the completed Modal GD checkpoints at six
million updates into per-seed inputs, retaining every state field. It writes
the unchanged constant-force and frozen-Jacobian forecasts before GPU work.
`pilot` checks representative GD and Adam states, including an on-disk 13+19
update resume against 32 uninterrupted updates. `probe --seed 0|1|2` invokes
the existing 37-case intervention matrix for 500,000 additional updates.
These are exploratory late interventions; they do not replace the original
independent-seed confirmation. Numerical kernels and forecast rules are unchanged.

`plateau_followup.sbatch` accepts a campaign root and phase. Submit `prepare`
and `analysis` as CPU-only jobs with `--gres=none --mem=32G --time=01:00:00`.
Submit `pilot` with `--gres=gpu:1 --time=00:05:00` after successful preparation;
submit `probe` as `--array=0-2%2 --gres=gpu:1 --time=00:20:00` after the pilot.
Disable requeues, retain exclusive bundle owners, and record every submission
in the combined accounting before allocating GPUs. The campaign root contains
the pinned `code`, immutable downloaded `modal` bundles and their
`transfer_hashes.json`, separate `late` inputs/outputs, and job logs.

The CPU-only `plateau_campaign_analysis` entrypoint has `summary`, `figures`,
`dense`, and `late` phases. Supply `--analysis` for existing endpoint CSVs,
`--source` for the relevant campaign root, and `--output` for new artifacts.
The dense phase retains all 13 targets, both optimizers, and seeds 0–2 at 600k,
1.1m, 3m, and 6m, using the original 512-update observer and complete optimizer
states. `plateau_dense --starts ...` also accepts explicit checkpoint steps;
its default remains 100k and 600k. The late phase evaluates available bundles
and scores the immutable 6m forecasts only when the 6.5m endpoint is complete.
Half-step and grid controls remain separate in summary tables. These commands
produce evidence artifacts only; reports are authored directly in Markdown.
