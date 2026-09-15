"""Write the rule's predictions for every expC10 cell BEFORE the sweep, and hash them into the pre-registration."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from rule import two_wall, constant_rule  # noqa: E402
from targets import make_targets  # noqa: E402

ACTS = ["tanh", "gelu", "swish"]
WIDTHS = [64, 128, 256, 512]
PRECISIONS = ["fp64", "fp32"]
OUT = HERE.parents[2] / "results" / "checkpoint_C_geometry" / "expC07_lambda_energy_rule" / "lambda_rule" / "data"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    targets = make_targets()
    preds = {"targets": [{"name": t["name"], "family": t["family"], "omega": t["omega"], "params": t["params"]} for t in targets],
             "cells": []}
    for t in targets:
        for act in ACTS:
            for N in WIDTHS:
                for prec in PRECISIONS:
                    cell = {"target": t["name"], "act": act, "N": N, "precision": prec}
                    for tag, om in (("exact", t["omega"]), ("x3", 3 * t["omega"]), ("div3", t["omega"] / 3)):
                        lam, E, th = two_wall(act, N, om, prec)
                        cell[tag] = {"lambda_star": lam, "E_star": E, "theta_b": th, "q": th / 3.141592653589793}
                    cell["constant"] = constant_rule(act, prec)
                    preds["cells"].append(cell)
    p = OUT / "c10_predictions.json"
    p.write_text(json.dumps(preds, indent=1))
    h_pred = hashlib.sha256(p.read_bytes()).hexdigest()
    h_rule = hashlib.sha256((HERE / "rule.py").read_bytes()).hexdigest()
    h_tgt = hashlib.sha256((HERE / "targets.py").read_bytes()).hexdigest()
    prereg = HERE / "PREREGISTRATION_c10.md"
    prereg.write_text(prereg.read_text().rstrip("\n") + f"\n\n- `predictions.json` SHA-256: `{h_pred}`\n- `rule.py` SHA-256: `{h_rule}`\n- `targets.py` SHA-256: `{h_tgt}`\n")
    print(f"wrote {p} ({len(preds['cells'])} cells)\npredictions {h_pred}\nrule {h_rule}\ntargets {h_tgt}")
    print("\nband edges (omega/pi) and q at N=64..512:")
    for t in targets:
        print(f"  {t['name']:12s} omega/pi={t['omega']/3.141592653589793:7.2f}  q=" + " ".join(f"{2*t['omega']/N/3.141592653589793:.3f}" for N in WIDTHS))
    print("\nconstant-rule baselines:", {(a, p): round(constant_rule(a, p), 3) for a in ACTS for p in PRECISIONS})


if __name__ == "__main__":
    main()
