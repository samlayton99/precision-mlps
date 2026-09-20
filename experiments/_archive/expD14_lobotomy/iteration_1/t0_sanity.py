"""iteration_1 sanity: everything that must be true before the grid runs.

  P1  lset="none" reproduces stock Adam to the last bit
  P2  Method C discovers exactly {v, c0} on all five cases (oracle match)
  P3  the qi case's floor0 is the known QI-geometry floor (~1e-14)
  P4  datagap keeps ~4% of the samples inside |x| < 0.4
  P5  coherent travel: 1.0 on a straight line, ~1/sqrt(K) on a random walk
  P6  corrected SNR: ~1 on a persistent gradient, ~0 on pure noise
  P7  the floor trajectory at step 0 equals floor0
"""
import importlib.util
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
_s = importlib.util.spec_from_file_location("d14_core1", HERE / "core1.py")
c1 = importlib.util.module_from_spec(_s)
_s.loader.exec_module(c1)

torch.set_default_dtype(torch.float64)
ok = True


def check(name, cond, detail=""):
    global ok
    ok &= bool(cond)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}  {detail}")


# P1 -- stock Adam equivalence
env = c1.build_case("sine", 32, case="rand", seed=0)
res = c1.train_lobo1(env, 50, lset="none", record=50)
th = env["theta0"].clone()
ma = torch.zeros(env["m"])
va = torch.zeros(env["m"])
for t in range(1, 51):
    _, g = env["f_residgrad"](th)
    ma = 0.9 * ma + 0.1 * g
    va = 0.999 * va + 0.001 * g * g
    th = th - 1e-3 * (ma / (1 - 0.9 ** t)) / ((va / (1 - 0.999 ** t)).sqrt() + 1e-8)
d = float((res["theta"] - th).abs().max())
check("P1 stock-Adam bit equivalence", d < 1e-14, f"max|dtheta| = {d:.2e}")

# P2 -- Method C: 100% precision always; recall may drop only by the value
# rule's design (it stops before a tensor whose captured value is negligible,
# e.g. c0 on datagap). Assert precision == 100% and recall >= 99%.
for case in ("qi", "clustered", "datagap", "rand", "rand_scale"):
    e = c1.build_case("sine", 32, case=case, seed=0)
    mask, ev = c1.discover_pertensor(e, e["theta0"], seed=0)
    got = set(torch.nonzero(mask).flatten().tolist())
    want = set(e["idx_readout"].tolist())
    pure = got <= want
    recall = len(got & want) / len(want)
    check(f"P2 discovery pure and >=99% on {case}", pure and recall >= 0.99,
          f"precision {'100%' if pure else 'BROKEN'}, "
          f"recall {recall:.3f}, J-evals = {ev}")

# P3 -- qi floor0 is the QI floor
e = c1.build_case("sine", 128, case="qi", seed=0)
check("P3 qi floor0 ~ 1e-14", e["floor0"] < 1e-12, f"floor0 = {e['floor0']:.2e}")

# P4 -- datagap keeps ~4% inside
e = c1.build_case("sine", 128, case="datagap", seed=0)
x = e["x_tr"]
n_in = int((np.abs(x) < c1.GAP_RADIUS).sum())
n_in_full = int((np.abs(np.linspace(-1, 1, c1.N_TRAIN)) < c1.GAP_RADIUS).sum())
frac = n_in / n_in_full
check("P4 datagap keeps ~4% inside", abs(frac - 0.04) < 0.01,
      f"kept {n_in}/{n_in_full} = {frac:.3f}")

# P5 -- coherent travel
v = torch.ones(10)
sv, sl = torch.zeros(10), 0.0
for _ in range(100):
    sv += v
    sl += float(torch.linalg.norm(v))
check("P5a travel = 1 on a line", abs(c1.coherent_travel(sv, sl) - 1.0) < 1e-12)
rng = np.random.default_rng(0)
sv, sl = torch.zeros(10), 0.0
K = 400
for _ in range(K):
    s = torch.tensor(rng.standard_normal(10))
    sv += s
    sl += float(torch.linalg.norm(s))
D = c1.coherent_travel(sv, sl)
check("P5b travel ~ 1/sqrt(K) on a walk", D < 3.0 / np.sqrt(K),
      f"D = {D:.3f}, 1/sqrt(K) = {1/np.sqrt(K):.3f}")

# P6 -- corrected SNR
m = torch.ones(50)
vv = torch.ones(50)
mask = torch.ones(50, dtype=torch.bool)
check("P6a SNR ~ 1 on persistent gradient",
      c1.snr_corrected(m, vv, mask, 0.9) > 0.99)
b1, b2 = 0.9, 0.999
ma = torch.zeros(2000)
va = torch.zeros(2000)
for t in range(1, 3001):
    g = torch.tensor(rng.standard_normal(2000))
    ma = b1 * ma + (1 - b1) * g
    va = b2 * va + (1 - b2) * g * g
mh = ma / (1 - b1 ** 3000)
vh = va / (1 - b2 ** 3000)
s = c1.snr_corrected(mh, vh, torch.ones(2000, dtype=torch.bool), b1)
check("P6b SNR ~ 0 on pure noise", s < 0.05, f"shat = {s:.4f}")

# P7 -- floor trajectory starts at floor0
e = c1.build_case("sine", 32, case="clustered", seed=0)
r = c1.train_lobo1(e, 10, lset="none", record=10, floor_every=5)
check("P7 fhist[0] == floor0", abs(r["fhist"]["floor"][0] - e["floor0"]) < 1e-16,
      f"{r['fhist']['floor'][0]:.3e} vs {e['floor0']:.3e}")

print("\nALL PASS" if ok else "\nFAILURES PRESENT")
raise SystemExit(0 if ok else 1)
