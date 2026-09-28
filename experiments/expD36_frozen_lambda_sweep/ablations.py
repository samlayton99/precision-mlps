"""One independently selected readout-coordinate ablation per invocation."""
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import math

import numpy as np
import scipy.linalg as sla
import torch
from threadpoolctl import threadpool_limits

import run as baseline
from plot import plot_all

DESCRIPTIONS = {
    "neighbor_unscaled": "Unscaled neighboring features",
    "sqrt_allowance": "PR square-root coefficient scaling (no neighboring)",
}


def reference_allowances(n, halo_per_side, lambda_reference=.25):
    """PR core.geometry allowance formula, retaining this campaign's halo count.

    Pure NumPy extraction of expD06_fixed_center_scales/core.py:47-78;
    no JAX training dependency and no change to the physical center grid.
    """
    h = 2/n
    delta = .25
    pole_distance = math.pi*h/(2*lambda_reference)
    if pole_distance >= delta:
        raise ValueError("Reference envelopes require pole distance < delta")
    alpha = np.full(n+1+2*halo_per_side, h/(2*(delta-pole_distance)))
    order = (halo_per_side+1)//2
    zeta = math.exp(-2*lambda_reference)
    products = np.r_[1., np.cumprod(1-zeta**np.arange(1, order+1))]
    d_lambda = math.pi/(2*lambda_reference)+4*math.log(2)/math.pi
    for i in range(1, order+1):
        li = zeta**((i*(i+1)-1)/2)/(products[i-1]*products[order-i])
        li *= math.prod(1+zeta**(j-.5) for j in range(1, order+1) if j != i)
        for slot in (i-1, len(alpha)-i):
            alpha[slot] += h*d_lambda*li/(2*delta)
    return np.r_[1+alpha.sum(), alpha]


def transform_problem(p, arm):
    """Keep physical A/E and geometry; train in the new coordinate matrix AM."""
    d = p.A.shape[2]
    M = torch.eye(d, dtype=torch.float64)
    extra_reference = {}
    if arm == "neighbor_unscaled":
        rows = torch.arange(2, d)
        M[rows, rows-1] = -1

        def mapped(a):
            return torch.cat((a[:, :, :1], a[:, :, 1:-1]-a[:, :, 2:], a[:, :, -1:]), dim=2)
    elif arm == "sqrt_allowance":
        alpha = reference_allowances(p.cfg["n"], p.cfg["halo_per_side"], p.cfg["lambda_reference"])
        scales = torch.from_numpy(np.sqrt(alpha))
        M = torch.diag(scales)
        extra_reference["alpha"] = alpha

        def mapped(a):
            return a*scales
    else:
        raise ValueError(arm)
    train_matrix, eval_matrix = mapped(p.A), mapped(p.E)
    B = train_matrix / math.sqrt(len(p.x_train))
    spectrum = np.stack([sla.svdvals(b.numpy(), check_finite=False) for b in B])
    rates = p.cfg["gd_rate_multiplier"] / spectrum[:, 0]**2
    cfg = dict(p.cfg, coordinates=arm)
    reference = dict(p.reference, optimizer_singular_values=spectrum, **extra_reference)
    return replace(p, cfg=cfg, B=B, BT=B.transpose(1, 2).contiguous(),
                   gd_rates=torch.from_numpy(rates)[:, None, None], reference=reference,
                   readout_map=M, mapped_A=train_matrix, mapped_E=eval_matrix)


def prepare(arm, source):
    source_metadata = json.loads((source / "data/metadata.json").read_text())
    p = baseline.make_problem(source_metadata["config"])
    with np.load(source / "data/reference.npz") as ref:
        for name, value in (("lambdas", p.lambdas), ("gammas", p.gammas),
                            ("centers", p.centers.numpy()), ("x_train", p.x_train.numpy()),
                            ("x_eval", p.x_eval.numpy()), ("y_train", p.y_train.numpy()),
                            ("y_eval", p.y_eval.numpy())):
            np.testing.assert_array_equal(value, ref[name])
        # Reuse the actual saved baseline reference, with its original rank policy.
        p.reference = {key: ref[key].copy() for key in p.reference}
    return transform_problem(p, arm), source_metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("arm", choices=DESCRIPTIONS)
    parser.add_argument("--baseline", type=Path, default=baseline.RESULTS)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()
    output = args.baseline / "ablations" / args.arm
    if args.plot_only:
        plot_all(output, baseline_output=args.baseline)
        return
    source = json.loads((args.baseline / "data/metadata.json").read_text())
    torch.set_num_threads(source["config"]["torch_threads"])
    torch.set_num_interop_threads(1)
    with threadpool_limits(limits=source["config"]["torch_threads"]):
        p, source_metadata = prepare(args.arm, args.baseline)
        if args.steps is not None:
            p.cfg["steps"] = args.steps
        state_path = output / "data/trajectory.npz"
        if state_path.exists():
            with np.load(state_path) as saved:
                if str(saved["config_sha256"]) != baseline.fingerprint(p.cfg):
                    raise ValueError("Ablation configuration changed")
        baseline.record_problem(p, output)
        meta_path = output / "data/metadata.json"
        meta = json.loads(meta_path.read_text())
        meta.update(display_name=DESCRIPTIONS[args.arm], baseline=str(args.baseline),
                    baseline_config_sha256=source_metadata["config_sha256"],
                    baseline_trajectory_sha256=hashlib.sha256((args.baseline / "data/trajectory.npz").read_bytes()).hexdigest(),
                    one_changed_factor="readout coordinate map M; same GD rate policy, same native Adam settings",
                    comparison="baseline curves and numerical least-squares references reused, not retrained",
                    readout_map=("c=M theta; unscaled adjacent differences, bias and final tanh anchor retained"
                                 if args.arm == "neighbor_unscaled" else
                                 "c=diag(sqrt(alpha)) theta; PR allowance formula with baseline halo count; no neighboring"),
                    reference_scaling=("none: unscaled neighboring" if args.arm == "neighbor_unscaled" else
                                       "alpha fixed at lambda_ref=.25 across all swept lambdas; current geometry unchanged"),
                    gd_rate_policy="1/sigma_max(AM/sqrt(m))**2, fixed during each run",
                    adam_epsilon_policy="fixed native epsilon; no compensating physical-threshold retuning")
        baseline.save_json(meta_path, meta)
        state, history = baseline.train(p, output, p.cfg["steps"])
        # Check the native prediction against the unchanged physical dictionary.
        physical = baseline.physical_coefficients(p, state)
        direct, native = p.E @ physical, p.mapped_E @ state["c"]
        discrepancy = torch.linalg.vector_norm(direct-native, dim=1)
        normalized = discrepancy / torch.linalg.vector_norm(p.y_eval_pair, dim=0)
        baseline.save_json(output / "data/map_validation.json", {
            "maximum_final_prediction_discrepancy_relative_to_target": float(normalized.max()),
            "map_rank": int(torch.linalg.matrix_rank(p.readout_map)),
            "map_dimension": len(p.readout_map),
            "all_step_zero_errors_equal_one": bool(np.all(history["eval_rel_l2"][0] == 1)),
        })
        plot_all(output, baseline_output=args.baseline)


if __name__ == "__main__":
    main()
