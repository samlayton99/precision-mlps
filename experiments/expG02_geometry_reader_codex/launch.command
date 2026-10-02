#!/bin/bash
set -euo pipefail
APP_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$APP_DIR/../.." && pwd)"
OUTPUT_DIR="$REPO_DIR/results/checkpoint_G_interactive/geometry_reader_codex"
PYTHON_BIN="${GEOMETRY_READER_PYTHON:-$HOME/venv/precisionMLPs/bin/python}"
if [ ! -x "$PYTHON_BIN" ]; then
  printf 'Python environment not found: %s\nUse GEOMETRY_READER_PYTHON to select a Python with NumPy and SciPy.\n' "$PYTHON_BIN"
  exit 1
fi
mkdir -p "$OUTPUT_DIR"
if ! /usr/bin/curl -fsS --max-time 2 http://127.0.0.1:8068/api/state >/dev/null 2>&1; then
  cd "$REPO_DIR"
  nohup "$PYTHON_BIN" "$APP_DIR/run.py" >"$OUTPUT_DIR/server.log" 2>&1 </dev/null &
  printf '%s\n' "$!" > "$OUTPUT_DIR/server.pid"
  for attempt in {1..40}; do
    if /usr/bin/curl -fsS --max-time 1 http://127.0.0.1:8068/api/state >/dev/null 2>&1; then break; fi
    sleep 0.25
  done
fi
if ! /usr/bin/curl -fsS --max-time 2 http://127.0.0.1:8068/api/state >/dev/null 2>&1; then
  printf 'The geometry reader did not start. See %s/server.log\n' "$OUTPUT_DIR"
  exit 1
fi
/usr/bin/open http://127.0.0.1:8068
