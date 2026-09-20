"""expA06 -- launch the interactive readout-structure explorer.

Run:  uv run python experiments/expA06_readout_structure/run.py
Then open http://127.0.0.1:8051
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from app import build_app   # noqa: E402

if __name__ == "__main__":
    build_app().run(debug=False, port=8051)
