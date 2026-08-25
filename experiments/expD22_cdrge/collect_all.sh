#!/bin/bash
# expD22 final collection: one worker per (init, target, N) shard, all seeds.
# Shards append to results/.../data/trajectories_<init>__<target>_<N>.jsonl;
# plotting globs the shards. Run from the repo root:
#   bash experiments/expD22_cdrge/collect_all.sh [n_workers]
set -u
WORKERS=${1:-8}
cd "$(dirname "$0")/../.."

CMDS=()
for init in qi xavier; do
  for target in sine exp runge sine_8pi; do
    for N in 64 128 256; do
      CMDS+=("OMP_NUM_THREADS=1 uv run --extra dev python experiments/expD22_cdrge/run.py \
        --collect --init $init --targets $target --widths $N --seeds 0,1,2 \
        --tag __${target}_${N}")
    done
  done
done

printf '%s\n' "${CMDS[@]}" | xargs -P "$WORKERS" -I{} bash -c '{}'
echo "ALL SHARDS DONE"
uv run --extra dev python experiments/expD22_cdrge/run.py --plot
