"""expD23 -- print the cell-A results table (markdown) from the data rows.

    uv run --extra dev python experiments/expD23_zo_ceiling/summarize.py
"""
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA = REPO_ROOT / "results" / "checkpoint_D_optimizers" / "expD23_zo_ceiling" / "data"

ORDER = ["adam", "fdgrad_adam", "cdrge_adam", "cdrge_adam_3000", "cdrge_lr_eq_eps", "spsa15_asrun",
         "spsa15", "lrsearch_1spsa", "lrsearch_spsa15", "spsa2", "sanger", "bandit", "bandit_active",
         "upstream_rmsprop", "probe_precond"]
LABEL = {
    "adam": "Adam, autograd gradient (reference)",
    "fdgrad_adam": "Adam on the coordinate-FD gradient ($2m$ evals/step; the $n\\to\\infty$ oracle)",
    "cdrge_adam": "CD-RGE Adam-style, $n{=}100$ (expD22 headline)",
    "cdrge_adam_3000": "CD-RGE Adam-style, $n{=}100$, 3000 iterations",
    "cdrge_lr_eq_eps": "CD-RGE, $\\text{lr}=\\epsilon=3\\times10^{-3}$ (author tie)",
    "spsa15_asrun": "`1.5-SPSA` as upstream executes it ($\\equiv$ lr $=\\epsilon$)",
    "spsa15": "`1.5-SPSA` as intended, curvature-normalised ($\\alpha{=}0.5$, $\\epsilon{=}0.1$, tuned)",
    "lrsearch_1spsa": "CD-RGE + upstream binary LR search (lr $=\\epsilon$ retied)",
    "lrsearch_spsa15": "`1.5-SPSA` (intended) + binary LR search",
    "spsa2": "`2SPSA` upstream (6 evals/probe, $n{=}33$, $\\epsilon{=}10^{-2}$, tuned)",
    "sanger": "`Sanger-SPSA` (rank 8, lr $3\\times10^{-3}$, tuned)",
    "bandit": "`BanditSPSA` (upstream reservoir threshold; $\\epsilon{=}3\\times10^{-5}$, tuned)",
    "bandit_active": "`BanditSPSA`, reservoir active (threshold 0)",
    "upstream_rmsprop": "CD-RGE upstream $\\beta_1/\\beta_2$ path ($v$ init ones; lr $10^{-2}$, tuned)",
    "probe_precond": "same + probe preconditioning ($z/\\sqrt v$)",
}


def main():
    rows = []
    for p in sorted(DATA.glob("arms_*.jsonl")) + sorted(DATA.glob("long_*.jsonl")):
        rows += [json.loads(l) for l in open(p)]
    rows = [r for r in rows if (r["target"], r["N"], r["init"], r["seed"]) == ("sine", 64, "qi", 0)]
    latest = {}
    for r in rows:
        latest[r["opt"]] = r
    adam = latest.get("adam")
    print("| arm | rel $L_2$ at its budget | steps | evaluations | rel $L_2$ of Adam at the same step |")
    print("|---|---:|---:|---:|---:|")
    for name in ORDER:
        r = latest.get(name)
        if r is None:
            continue
        tr = r["trace"]
        it = tr["iter"][-1]
        same = ""
        if adam is not None:
            ai = np.array(adam["trace"]["iter"]); ae = np.array(adam["trace"]["rel_l2"])
            same = f"${np.interp(min(it, ai[-1]), ai, ae):.1e}$"
        div = " (diverged)" if r.get("info", {}).get("diverged") else ""
        extra = ""
        if "n_lr_searches" in r.get("info", {}) and r["info"]["n_lr_searches"]:
            extra = f"; {r['info']['n_lr_searches']} LR searches, final lr {r['info']['final_lr']:.1e}"
        if name == "fdgrad_adam" and r["info"].get("fd_grad_rel_err"):
            errs = [e for _, e in r["info"]["fd_grad_rel_err"]]
            extra = f"; FD-vs-autograd gradient rel err ${min(errs):.0e}$ to ${max(errs):.0e}$"
        print(f"| {LABEL.get(name, name)}{extra} | ${r['final_rel_l2']:.1e}${div} | {it} | {r['evals']:,} | {same} |")


if __name__ == "__main__":
    main()
