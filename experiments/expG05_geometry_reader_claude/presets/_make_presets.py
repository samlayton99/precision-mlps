"""Build 1-D geometry presets for the geometry editor. Run with the repo venv from the repo root."""
import json, math, sys, importlib.util
from pathlib import Path
import numpy as np
import torch

REPO = Path("/Users/sam/my-repos/research/collaborations/precisionMLPs")
sys.path.insert(0, str(REPO))
from src.construction.qi_mpmath import default_halo

OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)
N = 64
LAM = 0.25


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def refit(centers, gammas, f, n=4001, rcond=1e-13):
    """Truncated-SVD readout with bias; relative L2 on a midpoint eval grid."""
    xt = np.linspace(-1, 1, max(16 * len(centers), 2003))
    xe = -1 + (np.arange(n) + 0.5) * 2 / n
    P = lambda x: np.hstack([np.tanh(gammas[None] * (x[:, None] - centers[None])), np.ones((len(x), 1))])
    A = P(xt); U, s, Vt = np.linalg.svd(A, full_matrices=False)
    k = s > rcond * s[0]
    w = Vt.T[:, k] @ ((U.T[k] @ f(xt)) / s[k])
    e = P(xe) @ w - f(xe)
    return float(np.linalg.norm(e) / np.linalg.norm(f(xe)))


sine = lambda x: np.sin(2 * np.pi * x)


def write(name, description, source, centers, gammas, h, h_convention, extra=None, check=True):
    centers = np.asarray(centers, float); gammas = np.asarray(gammas, float)
    h = np.broadcast_to(np.asarray(h, float), centers.shape)
    d = dict(name=name, description=description, source=source, N=int(extra.pop("N")) if extra and "N" in extra else N,
             W=int(len(centers)), h_convention=h_convention,
             centers=centers.tolist(), lambdas=(gammas * h).tolist(), gammas=gammas.tolist())
    if extra:
        d.update(extra)
    if check:
        d["check_refit_rel_l2_sin2pix"] = refit(centers, gammas, sine)
    (OUT / f"{name}.json").write_text(json.dumps(d))
    print(f"{name:32s} W={len(centers):4d} refit(sin2pix)={d.get('check_refit_rel_l2_sin2pix', float('nan')):.2e}")


# 1. standard QI grid ---------------------------------------------------------
h = 2 / N
R = default_halo(N, lambda_star=LAM)
c_qi = -1 + np.arange(-R, N + R + 1) * h
write("qi_uniform", "Standard QI geometry: uniform grid + halo, constant lambda = 0.25. Reaches the fp64 floor under lstsq.",
      "src/construction/qi_mpmath.py (x_n=-1+n h, h=2/N, n in [-halo, N+halo], halo=default_halo(N,0.25)=max(70, floor(0.4N)))",
      c_qi, np.full_like(c_qi, LAM / h), h, "h = 2/N (grid spacing)",
      extra=dict(formula="c_n=-1+n*h, h=2/N, n=-R..N+R, R=max(ceil(35/(2*0.25)), floor(0.4*N)); gamma=0.25/h"))

# 1b. halo-sqrt variant used by expD06_fixed_center_scales / expD24-37
R2 = math.ceil(math.sqrt(N))
c_sq = -1 + np.arange(-R2, N + R2 + 1) * h
write("qi_uniform_sqrt_halo", "Same uniform lambda=0.25 grid with the lighter halo R=ceil(sqrt N) used in the expD06-fixed-center and expD24-D37 campaigns.",
      "results/checkpoint_D_optimizers/expD06_fixed_center_scales/*.md (N,h,R,W table)", c_sq, np.full_like(c_sq, LAM / h), h,
      "h = 2/N", extra=dict(formula="c_n=-1+n*h, h=2/N, n=-R..N+R, R=ceil(sqrt N); gamma=0.25/h"))

# 2. Xavier (expD16/expD02 init), seed 0, at the QI width W for N=64 ---------
W = len(c_qi)
g = torch.Generator().manual_seed(0)
bound = math.sqrt(6.0 / (1.0 + W))
w = ((torch.rand(W, 1, generator=g) * 2 - 1) * bound).numpy().ravel()
b = ((torch.rand(W, generator=g) * 2 - 1) * bound).numpy().ravel()
c_x = -b / w; g_x = np.abs(w)
write("xavier_standard", "Standard Glorot-uniform inner layer (w,b ~ U(+-sqrt(6/(1+W)))), read as centers c=-b/w and gamma=|w|. Centers are a ratio of uniforms (heavy tails, many far outside [-1,1]); lambda ~ 1e-3.",
      "experiments/expD16_optimizer_zoo/run.py build_model(init='xavier') == expD02 TanhMLP init; center reading as in expD05 canonical_semantic (c=-b/w, gamma=|w|)",
      c_x, g_x, h, "h = 2/N of the matching QI grid (N=64), used only to express lambda",
      extra=dict(formula="torch.Generator().manual_seed(seed); w=(rand(W,1)*2-1)*sqrt(6/(1+W)); b=(rand(W)*2-1)*sqrt(6/(1+W)); c=-b/w; gamma=|w|; W=N+1+2*default_halo(N,0.25)",
                 seed=0, frac_centers_in_unit_interval=float(np.mean(np.abs(c_x) <= 1))))

# 3-5. trained-from-Xavier endpoints, expD06_fixed_center_scales (N=512, sine) ---
base = REPO / "results/checkpoint_D_optimizers/expD06_fixed_center_scales/newton_handoffs_analysis/final"
t_sine = lambda x: math.sqrt(2) * np.sin(2 * np.pi * x)
runs = [
    ("xavier_ssbroyden_trained", "ssbroyden_N512_s1_parameter_scale_a71fab637b",
     "Pure Xavier start (QI centers fixed, Xavier slopes) -> SSBroyden with individual parameter scales, small curvature guard 1e-30, seed 1; 8,028 accepted updates before line-search failure. Trained MSE 2.48e-19 (L2RE 5.0e-10). Median |lambda| 0.18; spread [0.10,0.30] 10-90%.",
     "results/checkpoint_D_optimizers/expD06_fixed_center_scales/newton_handoffs_results.md (small-guard SSBroyden table)"),
    ("xavier_adam_gn_best", "gn_N512_s1_parameter_scale_dcd1258ff2",
     "Best trained non-construction endpoint on sine in the repo: Xavier -> Adam (5.3M updates, eta 0.003, individual scales, eps 1e-15) -> Gauss-Newton 90,666 updates, seed 1. Trained MSE 1.02e-21, L2RE 3.2e-11. Median |lambda| 0.13, NOT uniform 0.25.",
     "results/checkpoint_D_optimizers/expD06_fixed_center_scales/newton_handoffs_results.md ('What longer training changes')"),
    ("xavier_newton_collapsed", "newton_N512_s0_parameter_scale_ff140666ec",
     "Contrast: Xavier -> full-Hessian trust-region Newton, 100k updates, seed 0. Trained MSE 6.3e-10; most slopes stay tiny (median |lambda| 3e-4) with a few grown ones.",
     "results/checkpoint_D_optimizers/expD06_fixed_center_scales/newton_handoffs_results.md"),
]
for name, key, desc, src in runs:
    H = np.load(base / key / "history.npz")
    cen = H["centers"]; gam = H["gamma"][-1]; cc = H["c"][-1]; hN = 2 / 512
    x = np.linspace(-1, 1, 8193)
    pred = cc[0] + np.tanh(gam[None] * (x[:, None] - cen[None])) @ cc[1:]
    mse = float(np.mean((pred - t_sine(x)) ** 2))
    write(name, desc, src + f" ; arrays from {base.relative_to(REPO)}/{key}/history.npz (keys centers, gamma[step], c[step]=(bias, weights), step, train_mse)",
          cen, gam, hN, "h = 2/N, N=512 (fixed centers -1+n h, n=-23..535, halo ceil(sqrt N)=23)",
          extra=dict(N=512, target="sqrt(2) sin(2 pi x)", step=int(H["step"][-1]),
                     recorded_train_mse=float(H["train_mse"][-1]), recomputed_train_mse=mse,
                     readout_bias=float(cc[0]), readout_weights=cc[1:].tolist(),
                     init_gammas=H["gamma"][0].tolist(), init_step=int(H["step"][0]),
                     signed="gammas/lambdas are signed; the editor should plot |lambda|"))

# 6-7. expH02 non-uniform spacing, gamma_j = 0.25/h_j --------------------------
h02 = load("expH02_run", REPO / "experiments/expH02_nonuniform_spacing_1d/run.py")
for dist, desc in [("halfgauss", "Smooth graded spacing (3x across the interval, neighbor ratio 1.00), gamma_j=0.25/h_j: reaches the fp64 floor like the uniform grid."),
                   ("bimodal", "Two dense clusters at +-0.5 (spacing varies 10x, neighbor ratio <=1.12), gamma_j=0.25/h_j: floor reached but needs ~4x the width of uniform (price = widest gap)."),
                   ("beta", "Beta(2,5) spacing, zero density at x=1: last gap stays ~4x its neighbor at every N, error stalls (3e-7 on sin 2pi x). The failure case.")]:
    c, hj, gm, Rh = h02.nonuniform_geometry(h02.DISTS[dist], 1.0, N)
    write(f"graded_{dist}", desc,
          "results/checkpoint_H_highdim/expH02_nonuniform_spacing_1d/expH02_results.md ; experiments/expH02_nonuniform_spacing_1d/run.py nonuniform_geometry(DISTS[name], s=1, N)",
          c, gm, hj, "h_j = (c_{j+1}-c_{j-1})/2 local spacing (one-sided at the outer halo ends); lambda_j = gamma_j h_j = 0.25 exactly",
          extra=dict(formula="c_j=Q_s^{-1}(j/N), j=0..N, q_s=(1-s)/2+s q; halo R=default_halo(N,0.25) continuing end spacing; gamma_j=0.25/h_j",
                     density=dict(halfgauss="q ~ exp(-t^2/2), t=-1.5+0.75(x+1)", bimodal="0.1*U + 0.9*(N(-0.5,0.2^2)+N(0.5,0.2^2))/2",
                                  beta="Beta(2,5) on (x+1)/2")[dist], s=1.0))

# 8. expG04 cascade (sharp grid + two soft coarser bands) ---------------------
casc = load("expG04_cascade", REPO / "experiments/expG04_cascade_multiband/cascade.py")
c, gv, band = casc.cascade_geometry(N, [0.25, 0.10, 0.05], 2)
hk = np.array([2 / max(4, N // 2 ** k) for k in band])
write("cascade_3band", "Multi-band cascade: the lambda=0.25 grid at N plus lambda=0.10 at N/2 and lambda=0.05 at N/4 (overlapping centers). Keeps the fp64 floor in-domain and removes the Runge extrapolation blowup of a single soft band.",
      "results/checkpoint_G_generalization/expG04_cascade_multiband/expG04_results.md ; experiments/expG04_cascade_multiband/cascade.py cascade_geometry(N,[0.25,0.10,0.05],2)",
      c, gv, hk, "per-band h_k = 2/(N/2^k); lambda = band's lambda", extra=dict(band=band.tolist(),
      formula="band k=0,1,2: N_k=max(4,N//2^k), h_k=2/N_k, centers -1+n h_k for n=-R_k..N_k+R_k, R_k=default_halo(N_k,0.25), gamma=lambda_k/h_k, lambda=(0.25,0.10,0.05)"))

# 9. expC04 random / clustered placement at the QI gamma ----------------------
from src.construction.center_geometry import random_centers, clustered_centers
span = (float(c_qi[0]), float(c_qi[-1]))
for nm, cc, desc in [("random_centers", random_centers(W, span, seed=0), "Same width, span and constant gamma=0.25/h as the QI grid, centers i.i.d. uniform: plateaus 2-3 orders above the floor at every width."),
                     ("clustered_centers", clustered_centers(W, span, seed=0), "Same width/span/gamma, centers drawn around W/8 uniform meta-centers (sigma=0.75 meta spacing): plateaus above the floor.")]:
    write(nm, desc, "results/checkpoint_C_geometry/expC04_center_geometry/expC04_results.md ; src/construction/center_geometry.py",
          cc, np.full(W, LAM / h), h, "h = 2/N of the matching QI grid (lambda = 0.25 nominal)",
          extra=dict(formula=f"{nm}(W, span=(c_qi[0], c_qi[-1]), seed=0), gamma=0.25/h"))

# 10. expH04-style monitor mesh for a narrow spike -----------------------------
sys.path.insert(0, str(REPO / "experiments/expH01_highdim_suite")); sys.path.insert(0, str(REPO / "experiments/expH04_mesh_finding"))
mesh = load("expH04_mesh", REPO / "experiments/expH04_mesh_finding/mesh.py")
x0, sig = 0.3, 0.03
spike = lambda x: np.exp(-((x - x0) / sig) ** 2)
grid = np.linspace(-1, 1, 20001)
d1 = np.abs(np.gradient(spike(grid), grid))
bw = mesh.BW_MULT * 2 / N
ker = np.exp(-0.5 * ((grid - grid[len(grid) // 2]) / bw) ** 2); ker /= ker.sum()
mon = np.convolve(d1, ker, mode="same")
ci, hi, info = mesh.place_by_density(grid, mon, N + 1, 0.6)
hl, hr = ci[1] - ci[0], ci[-1] - ci[-2]
cfull = np.concatenate([ci[0] - hl * np.arange(R, 0, -1), ci, ci[-1] + hr * np.arange(1, R + 1)])
hf = np.empty_like(cfull); hf[1:-1] = 0.5 * (cfull[2:] - cfull[:-2]); hf[0] = cfull[1] - cfull[0]; hf[-1] = cfull[-1] - cfull[-2]
d = dict()
write("spike_monitor_mesh", "Mesh-finding (expH04 machinery): N+1 centers placed by a slope monitor |f'| smoothed at 5.8 gaps, uniform floor 1-s=0.4, spacing graded |dh/dt|<=0.15, gamma_j=0.25/h_j; concentrates centers on a spike at x=0.3.",
      "results/checkpoint_H_highdim/expH04_mesh_finding/expH04_results.md ; experiments/expH04_mesh_finding/mesh.py place_by_density. Illustrative: target, s, and monitor chosen here, not an exact H04 cell.",
      cfull, 0.25 / hf, hf, "h_j local central-difference spacing; lambda_j = 0.25",
      extra=dict(formula="monitor m=|f'| of f=exp(-((x-0.3)/0.03)^2), Gaussian-smoothed at 5.8*(2/N); place_by_density(grid, m, N+1, s=0.6, g=0.15); halo R=default_halo(N,0.25) continuing end spacing; gamma_j=0.25/h_j",
                 max_neighbor_ratio=info["max_neighbor_ratio"],
                 check_refit_rel_l2_spike=refit(cfull, 0.25 / hf, spike),
                 check_refit_rel_l2_spike_uniform_qi=refit(c_qi, np.full_like(c_qi, LAM / h), spike)))

# 11. Chebyshev centers with local gamma (prediction case) ---------------------
j = np.arange(N + 1)
ci = -np.cos(np.pi * j / N)
hl, hr = ci[1] - ci[0], ci[-1] - ci[-2]
cfull = np.concatenate([ci[0] - hl * np.arange(R, 0, -1), ci, ci[-1] + hr * np.arange(1, R + 1)])
hf = np.empty_like(cfull); hf[1:-1] = 0.5 * (cfull[2:] - cfull[:-2]); hf[0] = cfull[1] - cfull[0]; hf[-1] = cfull[-1] - cfull[-2]
write("chebyshev_local_lambda", "Chebyshev-Lobatto centers -cos(pi j/N) with gamma_j=0.25/h_j: the end gaps jump ~3x between neighbors at every N (like expH02's Beta case), so expH02 predicts a stall. Not run in the repo; the refit number here is this script's check.",
      "formula only (no repo experiment); rule from expH02", cfull, 0.25 / hf, hf, "h_j local central-difference spacing; lambda_j = 0.25",
      extra=dict(formula="c_j=-cos(pi j/N), j=0..N; halo R=default_halo(N,0.25) continuing end spacing; gamma_j=0.25/h_j"))
