"""Export six traceable geometry examples without running training.

Run from the repository root with a NumPy installation. The numerical source
checkpoints stay untouched; the small JSON travels with the laptop application.
"""
from __future__ import annotations

import hashlib
import json
import runpy
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).with_name("catalog.json")
FLOOR = "results/checkpoint_D_optimizers/expD04_varpro/varpro_corrected/06_sine_xavier_seeds/geometry_sine_xavier_N256_s0"
SCALED = "results/checkpoint_D_optimizers/expD28_loss_gradient_decomposition/data/sine__scaled_xavier.npz"


def array(value):
    return np.asarray(value, dtype=np.float64).tolist()


def digest(relative):
    return hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()


def settings(n=128, halo=70, sampling="uniform", n_train=2003, n_eval=4001):
    return dict(N=n, N_convention="intervals", n_interior=n+1, h=2 / n, halo=halo, target="sine", activation="tanh",
                domain=[-1., 1.], n_train=n_train, n_eval=n_eval,
                sampling=sampling, rcond=1e-13)


def evaluate(a, b, setup):
    n = setup["n_train"]
    x = (-1 + (np.arange(n) + .5) * 2 / n if setup["sampling"] == "midpoint"
         else np.linspace(-1, 1, n))
    xe = np.linspace(-1, 1, setup["n_eval"])
    y, ye = np.sin(2 * np.pi * x), np.sin(2 * np.pi * xe)
    matrix = np.c_[np.tanh(x[:, None] * a + b), np.ones(n)]
    u, s, vh = np.linalg.svd(matrix, full_matrices=False)
    keep = s > setup["rcond"] * s[0]
    coeff = vh[keep].T @ ((u[:, keep].T @ y) / s[keep])
    prediction = np.tanh(xe[:, None] * a + b) @ coeff[:-1] + coeff[-1]
    error = float(np.linalg.norm(prediction - ye) / np.linalg.norm(ye))
    return coeff, error, int(keep.sum())


def entry(key, label, description, a, b, setup, provenance,
          initial=None, initial_bias=0., saved=None, saved_bias=None,
          saved_kind=None):
    a, b = np.asarray(a), np.asarray(b)
    assert len(a) == len(b) and np.all(np.isfinite(a)) and np.all(a != 0)
    c = -b / a
    solved, error, rank = evaluate(a, b, setup)
    result = dict(
        id=key, label=label, description=description, settings=setup,
        provenance=provenance,
        geometry=dict(centers=array(c), lambdas=array(abs(a) * setup["h"]),
                      signs=array(np.sign(a)), slopes=array(a), biases=array(b)),
        readout=dict(solved=array(solved[:-1]), solved_bias=float(solved[-1]),
                     initial=None if initial is None else array(initial),
                     initial_bias=None if initial is None else float(initial_bias),
                     saved=None if saved is None else array(saved),
                     saved_bias=saved_bias, saved_kind=saved_kind),
        verification=dict(eval_relative_l2=error, retained_rank=rank,
                          evaluated_on="2026-09-26", numpy_version=np.__version__,
                          solver="numpy.linalg.svd; relative cutoff 1e-13",
                          note="Fresh readout solve on the exported raw affine geometry; last digits depend on LAPACK."))
    print(f"{key}: W={len(a)}, refit={error:.6g}, rank={rank}")
    return result


def build():
    n, halo = 128, 70
    h = 2 / n
    centers = -1 + np.arange(-halo, n + halo + 1) * h
    w = len(centers)
    gamma = .25 / h
    setup = settings()
    clean_n = 144
    clean_halo = max(10, int(np.sqrt(clean_n)))
    clean_h = 2 / (clean_n-1)
    clean_gamma = .25 / clean_h
    clean_centers = -1 + np.arange(-clean_halo, clean_n + clean_halo) * clean_h
    clean_setup = settings(clean_n-1, clean_halo)
    clean_setup["auto_halo"] = True
    clean = entry(
        "clean_qi", "Clean QI · λ 0.25",
        "Uniform centers, λ=0.25, and the requested R=max(10, sqrt(N)) halo (R=12 at N=144 interior points; 168 total).",
        np.full(clean_centers.size, clean_gamma), -clean_gamma * clean_centers, clean_setup,
        dict(kind="reconstructed_source_formula",
             sources=["experiments/expC05_geometry_interpolation/common.py",
                      "src/construction/qi_mpmath.py"],
             formula="N interior points; c=-1+k*2/(N-1), k=-R..N-1+R; a=.25/(2/(N-1)); b=-a*c",
             note="Reference formula with Sam’s square-root halo with minimum R=10; not a saved optimizer checkpoint."))

    original = np.load(ROOT / (FLOOR + ".npz"))
    summary = json.loads((ROOT / (FLOOR + ".json")).read_text())
    a0, b0 = np.split(original["theta0"], 2)
    a, b = np.split(original["theta"], 2)
    floor_setup = settings(256, 102)
    rng = np.random.default_rng(0)
    bound = np.sqrt(6 / (len(a0) + 1))
    draw = rng.uniform(-bound, bound, len(a0))
    assert np.array_equal(draw, a0), "Original Xavier recipe no longer matches saved theta0"
    v0 = rng.uniform(-bound, bound, len(a0))
    base_source = dict(sources=[FLOOR + ".npz", FLOOR + ".json",
                               "experiments/expD04_varpro/varpro_corrected/inspect_geometry.py"],
                       source_sha256=digest(FLOOR + ".npz"), seed=0)
    xavier = entry(
        "xavier_original", "Xavier · original floor-run seed",
        "The exact untrained seed of the sine floor run: Xavier slopes, zero hidden biases, and all centers at zero.",
        a0, b0, floor_setup,
        dict(base_source, kind="saved_checkpoint_initial_geometry",
             recorded_eval_relative_l2=summary["init_refit"],
             note="Xavier here means the expD05 zero-bias affine convention, not random hidden biases."),
        initial=v0)
    learned = entry(
        "xavier_sine_floor", "Xavier → sine floor · saved geometry",
        "The recovered sine geometry from 200 VarPro Adam steps followed by Gauss–Newton. Recorded refit error 6.65e-15.",
        a, b, floor_setup,
        dict(base_source, kind="saved_checkpoint_final_geometry",
             recorded_initial_eval_relative_l2=summary["init_refit"],
             recorded_eval_relative_l2=summary["final"],
             method="VarPro: 200 Adam warmup steps then Gauss–Newton (cap 40; stopped at 28)",
             saved_arrays={"slopes": "theta[:461]", "biases": "theta[461:]", "readout": "v"},
             note="The saved v is a solved VarPro readout, not an Adam-trained readout. Original output bias was not saved; use the fresh solved_bias."),
        saved=original["v"], saved_kind="least_squares")

    scaled_data = np.load(ROOT / SCALED)
    assert int(scaled_data["steps"][0]) == 0
    scaled = entry(
        "scaled_xavier", "Scaled Xavier · mean λ 0.25",
        "Saved center-preserving Xavier initialization with the slopes and hidden biases scaled together to mean λ=0.25.",
        scaled_data["a"][0], scaled_data["b"][0],
        settings(128, 24, "midpoint", 1024, 8192),
        dict(kind="saved_checkpoint_initial_geometry", sources=[SCALED,
             "experiments/expD28_loss_gradient_decomposition/run.py"],
             source_sha256=digest(SCALED), seed=0, source_frame=0, source_step=0,
             note="Snapshot zero precedes GD. The preserved initial readout is the original random draw."),
        initial=scaled_data["v"][0, :-1], initial_bias=scaled_data["v"][0, -1])

    rng = np.random.default_rng([0, n])
    bound = np.sqrt(6 / (w + 1))
    raw_a = rng.uniform(-bound, bound, w)
    raw_b = rng.uniform(-bound, bound, w)
    order = np.argsort(-raw_b / raw_a, kind="stable")
    raw_a = raw_a[order]
    base = abs(raw_a) / abs(raw_a).mean()
    protected = gamma * base < .75
    mag = np.empty(w)
    mag[protected] = gamma * base[protected]
    mag[~protected] = (gamma * w - mag[protected].sum()) / (~protected).sum()
    soft_a = mag * np.sign(raw_a)
    soft = entry(
        "soft_protected", "Uniform + six soft neurons",
        "Uniform centers with six soft slopes kept from a Xavier pattern; the remaining slopes share the same magnitude while mean λ stays 0.25.",
        soft_a, -soft_a * centers, setup,
        dict(kind="reconstructed_source_formula",
             sources=["experiments/expC06_soft_neuron_interp/run_threshold.py",
                      "experiments/expC05_geometry_interpolation/common.py"],
             seed=0, center_mode="uniform", threshold_kind="const", threshold=.75,
             interpolation_s=1., protected_neurons=int(protected.sum()),
             note="Exact endpoint recipe from expC06. Freshly reconstructed; no optimizer training performed."))

    namespace = runpy.run_path(str(ROOT / "src/construction/center_geometry.py"))
    clustered_c = namespace["regular_clustered_centers"](
        w, (float(centers[0]), float(centers[-1])), cluster_size=8, ratio=.75)
    clustered = entry(
        "regular_clusters", "Regular clusters · groups of eight",
        "Eight-neuron groups compressed by ratio 0.75, exposing the gaps between clusters at uniform λ=0.25.",
        np.full(w, gamma), -gamma * clustered_c, setup,
        dict(kind="reconstructed_source_formula",
             sources=["src/construction/center_geometry.py",
                      "experiments/expC04_center_geometry/config.yaml"],
             cluster_size=8, cluster_ratio=.75,
             note="Source placement routine with expC04 settings; fresh sine readout measured on the catalog grid."))

    catalog = dict(schema_version=1, created="2026-09-26", model="sum_j v_j*tanh(a_j*x+b_j)+bias",
                   lambda_definition="lambda_j=abs(a_j)*(2/N); h is nominal spacing, not each local center gap",
                   raw_parameter_policy="Preserve signed slopes, hidden biases, and matching signed readouts for Adam continuity.",
                   presets=[clean, xavier, learned, scaled, soft, clustered])
    OUT.write_text(json.dumps(catalog, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    build()
