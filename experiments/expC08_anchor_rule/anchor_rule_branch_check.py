"""expC08: does the denominator branch min{Khat(0), Khat(theta/lambda)} predict landing on the wall? Run from the repo root
after anchor_rule.py (both versions). Output: data/anchor_rule_branch_vs_wall.json (figure made inline on 2026-09-09:
figures/anchor_rule_branch_vs_wall.png)."""
import sys, json, numpy as np, importlib.util
sys.argv.append("--omega-mean")
sys.path.insert(0, "experiments/expC08_anchor_rule")
import anchor_rule as ar
sp = importlib.util.spec_from_file_location("c9", "experiments/expC07_lambda_energy_rule/lambda_rule/run_c09_general.py"); c9 = importlib.util.module_from_spec(sp); sp.loader.exec_module(c9)
c9.log_khat = ar.log_khat; c9.KERNEL_ORDER = ar.ORDER
D = "results/checkpoint_C_geometry/expC08_anchor_rule/data/"
rows_w = json.load(open(D + "anchor_rule_rows_width.json")); rows_p = json.load(open(D + "anchor_rule_rows_precision.json"))
LAM = np.array(ar.LAM)
fib_cache = {}
def fiber(act, fn, N):
    k = (act, fn, N)
    if k not in fib_cache:
        f = ar.fiber_floor(act, fn, N, ar.LAM)
        fib_cache[k] = f if f is not None else c9.fiber_prediction(act, fn, N, ar.LAM, reach="main", halo=ar.HALO)["err"]
    return fib_cache[k]
def interp(l0, lam, y): return float(np.exp(np.interp(np.log(l0), np.log(lam), np.log(np.maximum(y, 1e-300)))))
out = []
for mode in ("max", "mean"):
    ar.OMEGA_MODE = mode
    preds = json.load(open(D + f"anchor_rule_predictions{'_mean' if mode=='mean' else ''}.json"))
    for sweep, rows in (("width", rows_w), ("precision", rows_p)):
        for key, v in preds[sweep].items():
            act, fn, third = key.split("|"); N = int(third) if sweep == "width" else ar.N_PREC; p = 53 if sweep == "width" else int(third)
            la = v["anchor"]
            if not np.isfinite(la): continue
            lam, rel = ar.curve(rows, act=act, N=N, fn=fn, p=p); sm = ar.smooth(rel)
            floor = float(np.median(sm[sm <= 5 * sm.min()]))
            if floor > 1e-3: continue                                   # unresolvable cell, no valley
            ff = fiber(act, fn, N); fa = interp(la, LAM, ff)
            xi = v["theta_M"] / la; k0 = float(np.exp(ar.log_khat(act, xi))) >= 1.0
            # local slope of the smoothed measured curve at the anchor (decades per decade), +-1 grid point
            i = int(np.argmin(abs(np.log(lam) - np.log(la)))); i0, i1 = max(0, i - 1), min(len(lam) - 1, i + 1)
            slope = float((np.log10(sm[i1]) - np.log10(sm[i0])) / (np.log10(lam[i1]) - np.log10(lam[i0])))
            out.append(dict(mode=mode, sweep=sweep, act=act, fn=fn, N=N, p=p, theta=v["theta_M"], xi=xi, K0=bool(k0), anchor=la,
                            fiber_over_floor=fa / floor, meas_over_floor=interp(la, lam, sm) / floor, slope=slope))
json.dump(out, open("results/checkpoint_C_geometry/expC08_anchor_rule/data/anchor_rule_branch_vs_wall.json", "w"))
def med(x): return np.median(x) if len(x) else float("nan")
for mode in ("max", "mean"):
    print(f"\n=== {mode} frequency.  'on wall' := exact fiber floor at the anchor > 2x the measured floor.  Resolved := theta <= 1")
    print(f"{'act':9s}{'branch':10s}{'n':>4s} {'on wall':>8s} {'med fib/floor':>14s} {'med meas/floor':>15s} {'med slope':>10s} | resolved only: n  on wall  med fib/floor")
    for act in ar.ACTS:
        for k0 in (True, False):
            S = [o for o in out if o["mode"] == mode and o["act"] == act and o["K0"] == k0]
            if not S: continue
            R = [o for o in S if o["theta"] <= 1.0]
            print(f"{act:9s}{'Khat(0)' if k0 else 'Khat(th/l)':10s}{len(S):4d} {sum(o['fiber_over_floor']>2 for o in S)/len(S):8.2f} {med([o['fiber_over_floor'] for o in S]):14.2f} {med([o['meas_over_floor'] for o in S]):15.2f} {med([o['slope'] for o in S]):10.2f} | {len(R):3d} {sum(o['fiber_over_floor']>2 for o in R)/max(1,len(R)):8.2f} {med([o['fiber_over_floor'] for o in R]):10.2f}")
    print("  cells on the wall by >10x (fiber floor at anchor / measured floor):")
    for o in sorted([o for o in out if o["mode"] == mode and o["fiber_over_floor"] > 10], key=lambda o: -o["fiber_over_floor"]):
        print(f"    {o['act']:8s} {o['fn']:12s} {o['sweep']:9s} N={o['N']:3d} p={o['p']:2d} theta={o['theta']:.2f} xi={o['xi']:5.2f} {'Khat(0)' if o['K0'] else 'Khat(th/l)':10s} fib/floor={o['fiber_over_floor']:7.1f} meas/floor={o['meas_over_floor']:6.1f} slope={o['slope']:5.1f}")
