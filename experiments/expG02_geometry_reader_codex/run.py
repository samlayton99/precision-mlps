"""Start the laptop-local Codex geometry reader, without Dash or Torch."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

# A small interactive SVD should not consume all BLAS threads.
for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"

from server import serve


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8068)
    parser.add_argument("--runs", type=Path, help="Override the recording folder (default: this repo's results folder)")
    parser.add_argument("--restore-session", type=Path, help="Restore a saved preview and committed baseline after an update")
    args = parser.parse_args()
    print(f"Geometry Reader · Codex: http://127.0.0.1:{args.port}", flush=True)
    serve(host="127.0.0.1", port=args.port, run_root=args.runs, session_path=args.restore_session)
