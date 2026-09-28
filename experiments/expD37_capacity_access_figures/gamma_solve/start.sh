#!/bin/bash
# Start only this viewer, leave existing unrelated servers untouched.
set -euo pipefail
task_dir="$(cd "$(dirname "$0")" && pwd)"
repo_dir="$(cd "$task_dir/../../.." && pwd)"
result_dir="$repo_dir/results/checkpoint_D_optimizers/expD37_capacity_access_figures/gamma_solve"
if /usr/bin/curl -fsS --max-time 2 http://127.0.0.1:8067/api/info >/dev/null 2>&1; then
  echo 'Gamma Solve is running at http://127.0.0.1:8067'
  exit 0
fi
mkdir -p "$result_dir/data"
cd "$repo_dir"
nohup "$repo_dir/.venv/bin/python" "$task_dir/server.py" >"$result_dir/data/server.log" 2>&1 </dev/null &
for attempt in {1..30}; do
  if /usr/bin/curl -fsS --max-time 1 http://127.0.0.1:8067/api/info >/dev/null 2>&1; then
    echo 'Gamma Solve is running at http://127.0.0.1:8067'
    exit 0
  fi
  sleep .2
done
echo "Gamma Solve did not start; see $result_dir/data/server.log" >&2
exit 1
