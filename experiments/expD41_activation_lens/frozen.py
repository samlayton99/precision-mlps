"""Adam readout learning at the saved, fixed QI geometries from expD41.

The random arm changes only the readout; it is not a full Xavier network.
Feature matrices are cached, but all updates use explicit residuals, not normal
equations. Least-squares readouts remain observational and are never injected.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import scipy
import torch
from torch import nn

from experiments.expD41_activation_lens import run as base

OUT = base.OUT / "frozen_geometry"


class FrozenReadout(nn.Module):
    def __init__(self, v, bias):
        super().__init__()
        self.v = nn.Parameter(torch.tensor(v, dtype=torch.float64))
        self.bias = nn.Parameter(torch.tensor(bias, dtype=torch.float64))

    def forward(self, phi):
        return phi @ self.v + self.bias


def fixed_features(activation, w, b, x):
    """Use exactly the activation implementation used by the trainable model."""
    with torch.no_grad():
        z = torch.tensor(x[:, None] * w[None, :] + b[None, :], dtype=torch.float64)
        return base.activation_torch(activation, z).detach()


def load_initial(activation, target):
    path = base.OUT / "runs" / f"{target}__{activation}__qi__s0.npz"
    with np.load(path) as data:
        assert data["step"][0] == 0
        params = tuple(data[k][0].copy() for k in ("w", "b", "v", "bias"))
    return params, path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def frozen_config():
    cfg = json.loads((base.OUT / "config.json").read_text())
    cfg["frozen_geometry"] = {
        "geometry_source": "saved parent QI step zero, identical for both readout starts",
        "trainable_parameters": ["v", "output_bias"],
        "initializations": {"zero_readout": "all readout weights and output bias exactly zero",
                            "xavier_readout": "parent paired Xavier readout draw; zero output bias"},
        "feature_evaluation": "cached torch float64 activation values; explicit residual gradients",
        "bandwidth_selection": "reuse parent QI values; no new sweep",
        "least_squares_injection": False,
    }
    return cfg


def run_geometry(activation, target, cfg, out=OUT):
    torch.set_num_threads(1)
    params, parent = load_initial(activation, target)
    w, b, vqi, dqi = params
    phi = fixed_features(activation, w, b, base.grid(cfg["n_train"]))
    phie = fixed_features(activation, w, b, base.grid(cfg["n_eval"]))
    fn = base.get_target(target).fn_numpy
    yt = fn(base.grid(cfg["n_train"]))
    ye = fn(base.grid(cfg["n_eval"]))
    y = torch.tensor(yt, dtype=torch.float64)
    solved, solved_bias, lsinfo = base.readout_solve(phi.numpy(), yt, cfg["readout_rcond"])
    floor = base.relative(phie.numpy() @ solved + solved_bias, ye)
    floor_train = base.relative(phi.numpy() @ solved + solved_bias, yt)
    geometry_digest = hashlib.sha256(w.tobytes() + b.tobytes()).hexdigest()
    feature_digest = hashlib.sha256(phi.numpy().tobytes()).hexdigest()
    for initialization, seeds in (("zero_readout", [0]), ("xavier_readout", cfg["xavier_seeds"])):
        for seed in seeds:
            name = f"{target}__{activation}__{initialization}__s{seed}"
            path = out / "runs" / f"{name}.json"
            source_paths = [Path(__file__), base.HERE / "run.py", base.HERE / "activations.py",
                            base.HERE / "local_activation.py", base.OUT / "local_design.json", parent]
            signature = {"config": cfg, "activation": activation, "target": target,
                         "initialization": initialization, "seed": seed,
                         "sources": {str(p.relative_to(ROOT)): sha(p) for p in source_paths}}
            fingerprint = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
            if path.exists():
                previous = json.loads(path.read_text())
                if previous["fingerprint"] != fingerprint:
                    raise ValueError(f"Preserve changed-protocol data and use a new output: {path}")
                if previous["complete"] and path.with_suffix(".npz").exists() and path.with_suffix(".pt").exists():
                    print(f"Already complete {name}", flush=True)
                    continue
            if initialization == "zero_readout":
                v, bias = np.zeros_like(vqi), 0.0
            else:
                _, _, v, bias = base.xavier(len(w), seed, cfg)
            model = FrozenReadout(v, bias)
            opt = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"], betas=tuple(cfg["adam_betas"]),
                                   eps=cfg["adam_epsilon"], weight_decay=0)
            history, vs, biases, steps = [], [], [], []
            start = time.perf_counter()
            record = set(base.recording_steps(cfg["steps"]))
            result = {"activation": activation, "target": target, "initialization": initialization,
                      "seed": seed, "complete": False, "fingerprint": fingerprint, "signature": signature,
                      "initialization_info": {"hidden_geometry": "parent QI step zero", "geometry_sha256": geometry_digest,
                                              "feature_sha256": feature_digest, "readout_start": initialization},
                      "least_squares_info": lsinfo, "history": history}
            for step in range(cfg["steps"] + 1):
                if step:
                    for group in opt.param_groups:
                        group["lr"] = base.learning_rate(step, cfg)
                    loss = base.train_step(model, opt, phi, y, base.mse)
                    if not np.isfinite(loss):
                        raise FloatingPointError(name)
                if step in record:
                    v = model.v.detach().numpy().copy()
                    bias = float(model.bias.detach())
                    with torch.no_grad():
                        actual = base.relative(model(phie).numpy(), ye)
                        train_actual = base.relative(model(phi).numpy(), yt)
                    history.append({"step": step, "lr": base.learning_rate(max(step, 1), cfg),
                                    "actual_rel_l2": actual, "train_actual_rel_l2": train_actual,
                                    "floor_rel_l2": floor, "train_floor_rel_l2": floor_train,
                                    "readout_norm": float(np.linalg.norm(v))})
                    steps.append(step); vs.append(v); biases.append(bias)
                    result["wall_seconds"] = time.perf_counter() - start
                    base.save_json(path, result)
                    if step in (0, cfg["steps"]):
                        print(f"{name} step={step} actual={actual:.4e} LS={floor:.4e}", flush=True)
            assert hashlib.sha256(phi.numpy().tobytes()).hexdigest() == feature_digest
            assert hashlib.sha256(w.tobytes() + b.tobytes()).hexdigest() == geometry_digest
            np.savez_compressed(path.with_suffix(".npz"), step=steps, w=w, b=b, v=vs, bias=biases,
                                solved_v=solved, solved_bias=solved_bias)
            torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(), "w": w, "b": b,
                        "step": cfg["steps"], "config": cfg}, path.with_suffix(".pt"))
            result.update(complete=True, geometry_unchanged=True, features_unchanged=True,
                          wall_seconds=time.perf_counter() - start)
            base.save_json(path, result)
            print(f"COMPLETE {name} {result['wall_seconds']:.1f}s", flush=True)


def validate(cfg, out=OUT):
    rows = []
    for activation in base.all_activations(cfg):
        for target in cfg["targets"]:
            (w, b, _, _), _ = load_initial(activation, target)
            x = base.grid(cfg["n_verify"])
            phi = fixed_features(activation, w, b, x).numpy()
            y = base.get_target(target).fn_numpy(x)
            xt = base.grid(cfg["n_train"])
            phit = fixed_features(activation, w, b, xt).numpy()
            yt = base.get_target(target).fn_numpy(xt)
            cutoffs = {}
            for cutoff in (1e-12, 1e-13, 1e-14):
                v, d, info = base.readout_solve(phit, yt, cutoff)
                cutoffs[str(cutoff)] = {"relative_l2": base.relative(phi @ v + d, y), **info}
            for initialization, seeds in (("zero_readout", [0]), ("xavier_readout", cfg["xavier_seeds"])):
                for seed in seeds:
                    path = out / "runs" / f"{target}__{activation}__{initialization}__s{seed}.json"
                    r = json.loads(path.read_text())
                    assert r["complete"] and r["history"][-1]["step"] == cfg["steps"]
                    assert r["geometry_unchanged"] and r["features_unchanged"]
                    for source, digest in r["signature"]["sources"].items():
                        assert sha(ROOT / source) == digest, source
                    with np.load(path.with_suffix(".npz")) as data:
                        assert np.array_equal(data["w"], w) and np.array_equal(data["b"], b)
                        assert data["step"].tolist() == [h["step"] for h in r["history"]]
                        actual = base.relative(phi @ data["v"][-1] + data["bias"][-1], y)
                        floor = base.relative(phi @ data["solved_v"] + data["solved_bias"], y)
                    rows.append({"run": path.stem, "actual_rel_l2_dense": actual, "floor_rel_l2_dense": floor,
                                 "actual_grid_ratio": actual/r["history"][-1]["actual_rel_l2"],
                                 "floor_grid_ratio": floor/r["history"][-1]["floor_rel_l2"], "cutoffs": cutoffs})
    base.save_json(out / "validation.json", {"grid_size": cfg["n_verify"], "rows": rows})
    print(f"Validated {len(rows)} frozen-geometry endpoints", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["train", "validate", "plot", "all"])
    parser.add_argument("--activation", choices=["tanh", "notch", "local", "sinc"])
    args = parser.parse_args()
    cfg = frozen_config()
    torch.set_num_threads(1)
    OUT.mkdir(parents=True, exist_ok=True)
    if args.stage in ("train", "all"):
        base.save_json(OUT / "config.json", cfg)
        base.save_json(OUT / "bandwidth.json", json.loads((base.OUT / "bandwidth.json").read_text()))
        base.save_json(OUT / "environment.json", {"python": sys.version, "torch": torch.__version__,
            "numpy": np.__version__, "scipy": scipy.__version__, "script_sha256": sha(__file__)})
        for activation in ([args.activation] if args.activation else base.all_activations(cfg)):
            for target in cfg["targets"]:
                run_geometry(activation, target, cfg)
    if args.stage in ("validate", "all"):
        validate(cfg)
    if args.stage in ("plot", "all"):
        from experiments.expD41_activation_lens.plot_frozen import render
        render(OUT)


if __name__ == "__main__":
    main()
