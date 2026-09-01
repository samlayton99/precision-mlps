"""expF17 -- the clean cross-method scaling suite. Runner.

Protocol: results/checkpoint_F_applications/expF17_method_scaling_suite/SPEC.md
(sections 13-17 are binding). Everything here is resumable: cells land in
cells.jsonl keyed by (task, method, C, seed, regime, variant); a re-run skips
landed cells; plots regenerate as landings arrive.

Usage:
    uv run --extra dev python experiments/expF17_method_scaling_suite/run.py --gate
    ... --run [--extras] [--max-cells N] [--only task=wave,method=qi_radon]
    ... --plot | --status
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from f17 import dicts as fd, solve as sv, protocol as pr, store
from f17.tasks import TASKS


def frozen_config(task, method, C, tuning):
    """The tuned config for this (task, method, C); runs the tuning sweep on
    demand (oracle regime, seed 0, select on the scored metric -- SPEC 17.1).
    The dynamic regime and the poly variant inherit this frozen config."""
    tkey = f"{task}|{method}|{C}|oracle"
    if tkey in tuning:
        return tuning[tkey]["chosen"]
    if task.startswith("dysts_"):
        from f17 import dicts1d as f1, solve1d as s1
        from f17.tasks1d import dysts_task
        t = dysts_task(task[len("dysts_"):])
        knob, grid = pr.knob_grid_1d(method, C)
        size, is_1d = C, True

        def evaluate(cfg):
            d = f1.build_dictionary_1d(method, C, cfg, fd.dict_rng(method, C, 0))
            A, sigma, _ = s1.oracle_fit_1d(t, d, 0)
            m = s1.score_1d(t, d, A, sigma)
            return dict(config=cfg, rel_l2=m["rel_l2"], linf=m["linf"])
    else:
        n_ax = fd.n_ax_for(C)
        knob, grid = pr.knob_grid(method, n_ax)
        t = TASKS[task]
        size, is_1d = n_ax, False

        def evaluate(cfg):
            d = fd.build_dictionary(method, n_ax, cfg, fd.dict_rng(method, n_ax, 0),
                                    fourier_x=t["periodic_fourier"])
            if task == "darcy_orig":
                a, _ = sv.darcy_orig_fit(t, d, 0)
            else:
                a, _ = sv.oracle_fit(t, d, 0)
            m = sv.score(t, d, a)
            return dict(config=cfg, rel_l2=m["rel_l2"], linf=m["linf"])

    rows = [evaluate(cfg) for cfg in grid]
    walked = 0
    while knob is not None and len(rows) >= 2 and walked < 2:
        best = min(rows, key=lambda r: r["rel_l2"])
        vals = sorted(r["config"][knob] for r in rows)
        v = best["config"][knob]
        if v not in (vals[0], vals[-1]):
            break  # interior argmin: the walk is done
        nxt = pr.knob_step(method, knob, v, +1 if v == vals[-1] else -1,
                           size, is_1d=is_1d)
        if nxt is None or any(r["config"][knob] == nxt for r in rows):
            break
        rows.append(evaluate({**best["config"], knob: nxt}))
        walked += 1
    chosen = min(rows, key=lambda r: r["rel_l2"])["config"]
    rec = dict(task=task, method=method, C=C, regime="oracle", knob=knob,
               grid=rows, chosen=chosen, edge_walked=walked)
    store.save_tuning(rec)
    tuning[tkey] = rec
    return chosen


def run_cell_1d(entry, tuning):
    from f17 import dicts1d as f1, solve1d as s1
    from f17.tasks1d import dysts_task
    task, method, W = entry["task"], entry["method"], entry["C"]
    seed, regime = entry["seed"], entry["regime"]
    t = dysts_task(task[len("dysts_"):])
    cfg = frozen_config(task, method, W, tuning)
    w_mult = None
    t0 = time.time()
    if regime == "oracle":
        d = f1.build_dictionary_1d(method, W, cfg, fd.dict_rng(method, W, seed))
        A, sigma, info = s1.oracle_fit_1d(t, d, seed)
    else:
        cfg, w_mult = dynamic_config(task, method, W, tuning)
        d, A, sigma, info = s1.dynamic_solve_1d(t, method, W, cfg, seed,
                                                w_mult=w_mult)
    m = s1.score_1d(t, d, A, sigma)
    rec = dict(task=task, method=method, C=W, seed=seed, regime=regime,
               variant="base", cols=d.cols, config=cfg, meta=d.meta,
               w_mult=(None if regime == "oracle" else w_mult),
               t_total=round(time.time() - t0, 2), **m)
    for k in ("iters", "init", "diverged", "res", "n_data"):
        if k in info:
            rec[k] = info[k]
    store.save_cell(rec)
    return rec


W_MULT_GRID = [0.3, 1.0, 3.0]


def dynamic_config(task, method, C, tuning):
    """Dictionary knobs are INHERITED from the oracle tuning; the dynamic solve
    additionally owns w_mult (the BC/IC block-weight multiplier, the solver-side
    knob expF03 measured as load-bearing). Swept {0.3, 1, 3} at seed 0 in the
    dynamic regime, selected on the scored metric (17.1), frozen, identical
    declared effort for every method."""
    tkey = f"{task}|{method}|{C}|dynamic"
    if tkey in tuning:
        return tuning[tkey]["chosen_dict"], tuning[tkey]["chosen"]["w_mult"]
    cfg = frozen_config(task, method, C, tuning)
    rows = []
    for w in W_MULT_GRID:
        if task.startswith("dysts_"):
            from f17 import solve1d as s1
            from f17.tasks1d import dysts_task
            t = dysts_task(task[len("dysts_"):])
            d, A, sigma, info = s1.dynamic_solve_1d(t, method, C, cfg, 0, w_mult=w)
            m = s1.score_1d(t, d, A, sigma)
        else:
            t = TASKS[task]
            d, a, info = sv.dynamic_solve(t, method, fd.n_ax_for(C), cfg, 0,
                                          fourier_x=t["periodic_fourier"],
                                          w_mult=w)
            m = sv.score(t, d, a)
        rows.append(dict(w_mult=w, rel_l2=m["rel_l2"]))
    chosen = min(rows, key=lambda r: r["rel_l2"])
    rec = dict(task=task, method=method, C=C, regime="dynamic", knob="w_mult",
               grid=rows, chosen={"w_mult": chosen["w_mult"]}, chosen_dict=cfg)
    store.save_tuning(rec)
    tuning[tkey] = rec
    return cfg, chosen["w_mult"]


def run_cell(entry, tuning):
    if entry["task"].startswith("dysts_"):
        return run_cell_1d(entry, tuning)
    task, method, C = entry["task"], entry["method"], entry["C"]
    seed, regime, variant = entry["seed"], entry["regime"], entry["variant"]
    t = TASKS[task]
    n_ax = fd.n_ax_for(C)
    cfg = frozen_config(task, method, C, tuning)
    w_mult = None
    t0 = time.time()
    if regime == "oracle":
        d = fd.build_dictionary(method, n_ax, cfg, fd.dict_rng(method, n_ax, seed),
                                fourier_x=t["periodic_fourier"],
                                poly=(variant == "poly"))
        if task == "darcy_orig":
            a, info = sv.darcy_orig_fit(t, d, seed)
        else:
            a, info = sv.oracle_fit(t, d, seed)
    else:
        cfg, w_mult = dynamic_config(task, method, C, tuning)
        d, a, info = sv.dynamic_solve(t, method, n_ax, cfg, seed,
                                      fourier_x=t["periodic_fourier"],
                                      poly=(variant == "poly"), w_mult=w_mult)
    m = sv.score(t, d, a)
    rec = dict(task=task, method=method, C=C, seed=seed, regime=regime,
               variant=variant, n_ax=n_ax, cols=d.cols, config=cfg,
               meta=d.meta if not hasattr(d, "base") else d.meta, w_mult=w_mult,
               t_total=round(time.time() - t0, 2), **m)
    for k in ("iters", "init", "rungs", "rank", "n_data", "res"):
        if k in info:
            rec[k] = info[k]
    store.save_cell(rec)
    return rec


SCALING_SET = set(pr.SCALING_METHODS) | set(pr.SCALING_1D)


def run_queue(include_extras=False, max_cells=None, only=None, replot_every=15,
              scaling_only=False):
    from f17 import plots
    cells = store.load()
    tuning = store.load_tuning()
    queue = pr.build_queue(include_extras=include_extras)
    if scaling_only:
        queue = [e for e in queue if e["method"] in SCALING_SET]
    if only:
        queue = [e for e in queue
                 if all(str(e[k]) == v for k, v in only.items())]
    pending = [e for e in queue
               if pr.cell_key(e["task"], e["method"], e["C"], e["seed"],
                              e["regime"], e["variant"]) not in cells]
    print(f"queue: {len(queue)} cells, {len(queue) - len(pending)} landed, "
          f"{len(pending)} pending", flush=True)
    n_done = 0
    t_start = time.time()
    prev_task = None
    for entry in pending:
        rec = run_cell(entry, tuning)
        n_done += 1
        print(f"[{n_done}/{len(pending)}] {rec['key']:52s} rel_l2={rec['rel_l2']:.2e} "
              f"cfg={rec['config']} ({rec['t_total']}s)", flush=True)
        if prev_task is not None and entry["task"] != prev_task:
            plots.residual_figs(store.load(), store.load_tuning())
        prev_task = entry["task"]
        if n_done % replot_every == 0:
            plots.replot()
        if max_cells and n_done >= max_cells:
            break
    plots.replot()
    plots.residual_figs(store.load(), store.load_tuning())
    print(f"done: {n_done} new cells in {(time.time() - t_start) / 60:.1f} min",
          flush=True)


def main():
    args = sys.argv[1:]
    if "--gate" in args:
        from f17.tasks import verify_oracles
        verify_oracles()
        return
    if "--status" in args or "--plot" in args:
        from f17 import plots
        plots.replot()
        return
    only = None
    for a in args:
        if a.startswith("--only"):
            kv = a.split("=", 1)[1] if "=" in a else args[args.index(a) + 1]
            only = dict(p.split("=") for p in kv.split(","))
    max_cells = None
    if "--max-cells" in args:
        max_cells = int(args[args.index("--max-cells") + 1])
    if "--run" in args or "--smoke" in args:
        if "--smoke" in args and max_cells is None:
            max_cells = 12
        run_queue(include_extras=("--extras" in args), max_cells=max_cells,
                  only=only, scaling_only=("--scaling-only" in args))
        return
    print(__doc__)


if __name__ == "__main__":
    main()
