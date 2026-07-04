"""Launch the expG01 interactive explorer.

    ~/.venvs/precisionMLPs/bin/python experiments/expG01_interactive_explorer/run.py

Then open http://127.0.0.1:8050 in a browser.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
for p in (str(REPO_ROOT), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

from app import app  # noqa: E402

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8050, debug=False)
