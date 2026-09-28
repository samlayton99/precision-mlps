"""Does QUILLS' error follow the min structure E(W, p) = max(A(W), C 2^-p) at a fixed bandwidth?

The sweep chooses lambda by the bandwidth rule at every (W, p), which couples the width floor to p
(the rule lowers lambda as p grows). The theory's min structure is for a fixed lambda. This runs
QUILLS on a dense (W, p) grid at fixed lambda values and at the rule's lambda, everything at p bits
exactly as in the sweep (methods/quills.py: build), measured on the reporting grid.

    .venv/bin/python experiments/expC13_five_method_comparison/quills_min_test.py      # resumable
Writes data/quills_min_test.jsonl.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pfloat  # noqa: E402

from experiments.expC13_five_method_comparison import common as C  # noqa: E402
from experiments.expC13_five_method_comparison.methods import quills  # noqa: E402
from experiments.expC13_five_method_comparison.selection import report_meter  # noqa: E402

OUT = C.OUT / "data" / "quills_min_test.jsonl"
TARGETS = ["chirp", "runge"]
WIDTHS = [96, 128, 160, 192, 224, 256, 320, 384]   # the narrow widths, where the rule-lambda fit misses
PRECISIONS = list(range(12, 53, 2)) + [53]
LAMBDAS = [0.25, 0.35, 0.5, "rule"]


def job(target, W, p, lam):
    pfloat.set_num_threads(1)
    t = time.monotonic()
    try:
        value = quills.rule_lambda(target, p, W) if lam == "rule" else lam
    except ValueError:
        return {"target": target, "width": W, "p": p, "lambda": lam, "status": "rule_inadmissible"}
    net, meta = quills.build(target, p, width=W, lam=value)
    xr = C.inputs(C.linspace_points(C.REPORT_POINTS), C.fmt(p))
    e = report_meter(target).rel_l2(net.forward(xr))
    return {"target": target, "width": W, "p": p, "lambda": lam, "lambda_value": value, "rel_l2": e,
            "rank": meta["rank"], "params": net.params(), "status": "ok", "seconds": time.monotonic() - t}


def main():
    done = set()
    if OUT.exists():
        for s in OUT.read_text().splitlines():
            r = json.loads(s)
            done.add((r["target"], r["width"], r["p"], r["lambda"]))
    jobs = [(t, W, p, lam) for t in TARGETS for lam in LAMBDAS for W in WIDTHS for p in PRECISIONS
            if (t, W, p, lam) not in done]
    jobs.sort(key=lambda j: -j[1])
    print(f"{len(jobs)} fits", flush=True)
    start = time.monotonic()
    with ProcessPoolExecutor(9) as ex, OUT.open("a") as fh:
        futs = [ex.submit(job, *j) for j in jobs]
        for i, fut in enumerate(as_completed(futs)):
            fh.write(json.dumps(fut.result()) + "\n")
            fh.flush()
            if i % 200 == 0:
                print(f"{i}/{len(jobs)} [{time.monotonic() - start:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
