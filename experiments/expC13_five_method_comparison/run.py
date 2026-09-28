"""expC13: five constructions (QUILLS, Mhaskar, classical staircase, Costarelli-Spigler, ChebNet) with
every construction and inference operation at p bits. SPEC.md is normative.

Run from the repository root (resumable; finished (target, method, p) rows are skipped):
    .venv/bin/python experiments/expC13_five_method_comparison/run.py --sweep            # p = 8..53
    .venv/bin/python experiments/expC13_five_method_comparison/run.py --extended         # p > 53
    .venv/bin/python experiments/expC13_five_method_comparison/plot.py
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import importlib
import json
import os
from pathlib import Path
import platform
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pfloat  # noqa: E402

from experiments.expC13_five_method_comparison import common as C  # noqa: E402
from experiments.expC13_five_method_comparison.selection import Context, report_meter  # noqa: E402

HERE = Path(__file__).resolve().parent
DATA = C.OUT / "data"
METHODS = ["quills", "quills_w1024", "mhaskar", "staircase", "costarelli", "chebnet",
           "mhaskar_appendix", "staircase_classical", "costarelli_appendix", "chebnet_paper",
           "chebnet_neurons"]
# faithful configurations: the appendix's networks with no implementation variants
VARIANTS = {"quills_w1024": ("quills", {"widths": [1024]}),        # the expC11 configuration
            "mhaskar_appendix": ("mhaskar", {"faithful_only": True}),
            "staircase_classical": ("staircase", {"faithful_only": True}),
            "costarelli_appendix": ("costarelli", {"faithful_only": True}),
            "chebnet_paper": ("chebnet", {"faithful_only": True}),
            # sensitivity arm: budget of 1024 hidden neurons instead of 3073 parameters
            "chebnet_neurons": ("chebnet", {"neuron_budget": True})}
TARGETS = ["exp", "sine", "runge", "chirp"]
SWEEP = list(range(8, 54))
EXTENDED = [56, 64, 72, 80, 96, 113, 128]


def sources() -> dict:
    files = sorted(HERE.rglob("*.py")) + [HERE / "SPEC.md"]
    out = {str(f.relative_to(ROOT)): hashlib.sha256(f.read_bytes()).hexdigest() for f in files if f.exists()}
    out["pfloat"] = pfloat.__version__
    return out


def job(target: str, method: str, p: int, threads: int) -> dict:
    pfloat.set_num_threads(threads)
    t0 = time.monotonic()
    module, kwargs = VARIANTS.get(method, (method, {}))
    mod = importlib.import_module(f"experiments.expC13_five_method_comparison.methods.{module}")
    ctx = Context(target, p)
    sel = mod.select(ctx, **kwargs)
    table_path = DATA / "candidates" / f"{target}_{method}_p{p}.json"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    table_path.write_text(json.dumps(sel.table, default=float) + "\n")
    row = {"target": target, "method": method, "p": p, "candidates": len(sel.table),
           "valid_candidates": sum(r.get("status") == "ok" for r in sel.table)}
    if sel.best_net is None:
        row.update(status="no_valid_candidate", seconds=time.monotonic() - t0)
        return row
    net = sel.best_net
    xr = C.inputs(C.linspace_points(C.REPORT_POINTS), ctx.F)
    out = net.forward(xr)
    rep = report_meter(target).errors(out)
    check = C.Meter(target, C.linspace_points(C.REPORT_POINTS), C.CHECK_BITS).errors(out)
    # exact storage and bit-for-bit replay of the selected model
    path = DATA / "models" / f"{target}_{method}_p{p}.npz"
    C.save_model(path, net, p, {"hp": sel.best_hp})
    replay = C.load_model(path).forward(xr)
    assert C.same_bits(replay, out), "replay differs"
    row.update(status="ok", hp=sel.best_hp, val_rel_l2=sel.best_val, rel_l2=rep["rel_l2"], rel_linf=rep["rel_linf"],
               rel_l2_check=check["rel_l2"],
               meter_stable=bool(abs(rep["rel_l2"] - check["rel_l2"]) <= 1e-12 * max(check["rel_l2"], 1e-300)),
               params=net.params(), neurons=net.neurons(), depth=net.depth(), width=net.width,
               max_abs=net.max_abs(), seconds=time.monotonic() - t0)
    return row


def run(precisions, targets, methods, workers: int, threads: int):
    DATA.mkdir(parents=True, exist_ok=True)
    man_path = DATA / "config.json"
    man = {"sources": sources(), "exponent_range": [C.EMIN, C.EMAX], "ref_bits": C.REF_BITS,
           "check_bits": C.CHECK_BITS, "param_budget": C.PARAM_BUDGET, "machine": platform.platform()}
    runs = json.loads(man_path.read_text()) if man_path.exists() else []
    runs.append({**man, "started": time.strftime("%Y-%m-%d %H:%M:%S"), "precisions": list(precisions),
                 "targets": list(targets), "methods": list(methods)})
    man_path.write_text(json.dumps(runs, indent=1) + "\n")
    rows_path = DATA / "summary.jsonl"
    done = set()
    if rows_path.exists():
        for s in rows_path.read_text().splitlines():
            r = json.loads(s)
            done.add((r["target"], r["method"], r["p"]))
    jobs = [(t, m, p) for p in precisions for t in targets for m in methods if (t, m, p) not in done]
    # expensive jobs first (high p, QUILLS' lstsq, Mhaskar's search)
    weight = {"quills": 5, "quills_w1024": 3, "mhaskar": 4, "mhaskar_appendix": 3, "chebnet": 2}
    jobs.sort(key=lambda j: (-j[2], -weight.get(j[1], 1)))
    start = time.monotonic()
    print(f"{len(jobs)} jobs, {workers} workers x {threads} threads", flush=True)
    with ProcessPoolExecutor(workers) as ex, rows_path.open("a") as fh:
        futs = {ex.submit(job, t, m, p, threads): (t, m, p) for t, m, p in jobs}
        for fut in as_completed(futs):
            t, m, p = futs[fut]
            try:
                row = fut.result()
            except Exception:
                print(f"FAILED {t} {m} p={p}\n{traceback.format_exc()}", flush=True)
                continue
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            err = row.get("rel_l2", float("nan"))
            print(f"[{time.monotonic() - start:7.0f}s] {t:6s} {m:10s} p={p:3d} rel_l2={err:.3g} "
                  f"params={row.get('params')} {row.get('hp')} ({row['seconds']:.0f}s)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--extended", action="store_true")
    ap.add_argument("--p", type=int, nargs="*")
    ap.add_argument("--targets", nargs="*", default=TARGETS)
    ap.add_argument("--methods", nargs="*", default=METHODS)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--threads", type=int, default=1)
    a = ap.parse_args()
    precisions = a.p or (SWEEP if a.sweep else []) + (EXTENDED if a.extended else [])
    if not precisions:
        ap.error("give --sweep, --extended or --p")
    run(precisions, a.targets, a.methods, a.workers, a.threads)


if __name__ == "__main__":
    main()
