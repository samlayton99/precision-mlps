"""Detached evidence for the constant-rate joint experiment; never trains a model."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import svd

from . import core, diagnostics, difference_training as training, ratio, ratio_analysis, run

LABELS = {"scaled": "Scaled", "scaled_differences": "Scaled neighbor differences"}
CUTOFFS = (1e-10, 1e-12, 1e-14)


def projected_forces(j, r, u):
    parallel=u@(u.T@r)
    perpendicular=r-parallel
    naive=j.T@perpendicular
    # A second projection removes leakage of the much larger in-span residual.
    # In exact arithmetic (I-P)^2=I-P. This avoids forming a dense projected J.
    perpendicular-=u@(u.T@perpendicular)
    gp,gn=j.T@parallel,j.T@perpendicular
    return gp,gn,np.linalg.norm(naive-gn),np.linalg.norm(j.T@r-gp-gn)


def read_trace(folder, end):
    steps, traces = [], []
    for path in sorted(folder.glob("trace_*.npz")):
        _, lo, hi = path.stem.split("_")
        if int(lo) >= end:
            continue
        with np.load(path) as a:
            np.testing.assert_array_equal(a["columns"], training.TRACE_COLUMNS)
            assert len(a["trace"]) == int(hi)-int(lo)
            steps.append(np.arange(int(lo), int(hi)))
            traces.append(a["trace"])
    step, trace = np.concatenate(steps), np.concatenate(traces)
    # Resume can replace a deterministic overlap. Keep the last copy.
    _, indices = np.unique(step[::-1], return_index=True)
    indices = len(step)-1-indices
    indices = indices[np.argsort(step[indices])]
    keep = indices[step[indices] < end]
    np.testing.assert_array_equal(step[keep], np.arange(end))
    return trace[keep]


def history(folder, end):
    meta = json.loads((folder/"case.json").read_text())
    y = core.target(np.linspace(-1, 1, meta["n"]*meta["samples_per_cell"]+1), "sine", np)
    rows = []
    for path in sorted(folder.glob("checkpoint_*.npz")):
        step = int(path.stem.split("_")[-1])
        if step > end:
            continue
        with np.load(path) as a:
            bounds, rb, _ = diagnostics.band_residuals((a["prediction_train"]-y)/np.sqrt(len(y)))
            rows.append(dict(step=step, **{k:a[k] for k in ("c", "gamma", "lambda", "train_mse", "validation_mse",
                                                         "travel_c", "travel_lambda", "alternate_eval_max")},
                             band_mse=np.sum(rb**2, axis=1), eta_a=meta["eta"], eta_lambda=meta["eta"]))
    return {k:np.stack([r[k] for r in rows]) for k in rows[0]}, bounds


def summarize(root, output, end=100000):
    rows = []
    for folder in sorted(root.glob("N*_eta*")):
        case = json.loads((folder/"case.json").read_text())
        latest = json.loads((folder/"latest.json").read_text())
        row = {k:case[k] for k in ("n", "seed", "coordinates", "eta")}
        row["comparison_step"]=end
        row.update(key=folder.name, completed_updates=latest["completed_updates"], failed_update=latest["failed_update"])
        row["eligible"] = latest["completed_updates"] >= end and not latest["failed_update"]
        if row["eligible"]:
            trace = read_trace(folder, end)
            assert np.all(np.isfinite(trace))
            np.testing.assert_array_equal(trace[:, -1], np.full(end, case["eta"]))
            window = 2*trace[end-20000:end, 0]
            row.update(window_mean_mse=float(window.mean()), window_std_mse=float(window.std()),
                       window_min_mse=float(window.min()), window_max_mse=float(window.max()),
                       previous_20k_mse=float(2*trace[end-40000:end-20000, 0].mean()),
                       half_window_ratio=float(window[10000:].mean()/window[:10000].mean()),
                       zero_native_steps=int(trace[:, 8].sum()), zero_physical_steps=int(trace[:, 9].sum()),
                       sign_crossings=int(trace[:, 10].sum()))
            with np.load(folder/f"checkpoint_{end:09d}.npz") as cp:
                g = core.geometry(case["n"])
                row.update(endpoint_mse=float(cp["train_mse"]), validation_mse=float(cp["validation_mse"]),
                           lambda_core_abs_quantiles=np.quantile(np.abs(cp["lambda"][g.core]), [0,.1,.5,.9,1]).tolist(),
                           lambda_halo_abs_quantiles=np.quantile(np.abs(cp["lambda"][~g.core]), [0,.1,.5,.9,1]).tolist(),
                           negative_slopes=int(np.sum(cp["lambda"] < 0)),
                           mean_late_delta_c_rms=float(trace[-20000:, 3].mean()),
                           mean_late_delta_gamma_rms=float(trace[-20000:, 5].mean()),
                           physical_gamma_rate=case["eta"]/g.h**2,
                           physical_scaled_ordinary_readout_rate=case["eta"]*g.ordinary_alpha)
            h, bounds = history(folder, end)
            dest = output/folder.name
            dest.mkdir(exist_ok=True, parents=True)
            run.save_arrays(dest/"history.npz", **h, band_bounds=bounds, centers=g.centers)
            blocks=2*trace[:end//20000*20000,0].reshape(-1,20000)
            run.save_arrays(dest/"window_mse.npz",step=20000*(np.arange(len(blocks))+1),
                            mean=blocks.mean(axis=1),quantiles=np.quantile(blocks,[0,.1,.5,.9,1],axis=1))
        rows.append(row)
    run.write_json(output/"summary.json", rows)
    pairs=[]
    for n,seed in sorted({(r["n"],r["seed"]) for r in rows}):
        paths=[]
        for coord in training.COORDINATES:
            group=[r for r in rows if (r["n"],r["seed"],r["coordinates"])==(n,seed,coord)]
            if group: paths.append(root/group[0]["key"]/"checkpoint_000000000.npz")
        if len(paths)==2:
            with np.load(paths[0]) as a, np.load(paths[1]) as b:
                errors={k:float(np.max(np.abs(a[k]-b[k]))) for k in ("c","gamma","prediction_train")}
                for k in errors: np.testing.assert_allclose(a[k],b[k],rtol=1e-12,atol=1e-12)
            pairs.append(dict(n=n,seed=seed,maximum_initial_difference=errors))
    run.write_json(output/"initial_pairing.json",pairs)
    if rows:
        fields = list(dict.fromkeys(k for r in rows for k in r))
        with (output/"summary.csv").open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader(); writer.writerows(rows)
    eligible = [r for r in rows if r["eligible"] and r["n"] == 512 and r["seed"] == 0]
    selected = []
    for coord in training.COORDINATES:
        candidates = [r for r in eligible if r["coordinates"] == coord]
        if candidates:
            selected.append(min(candidates, key=lambda r: (r["window_mean_mse"], r["eta"])))
    run.write_json(output/"selected.json", [{k:r[k] for k in ("n", "seed", "coordinates", "eta")} for r in selected])
    print(json.dumps({"total":len(rows), "eligible":sum(r["eligible"] for r in rows),
                      "selected":[{k:r[k] for k in ("key", "window_mean_mse", "half_window_ratio")} for r in selected]}), flush=True)
    return rows


def mapped_features(a, g, coord):
    if coord == "scaled":
        return a*g.d
    s = np.sqrt(np.cumsum(g.alpha[1:]))
    return np.column_stack((a[:, 0]*g.d[0], (a[:, 1:-1]-a[:, 2:])*s[:-1], a[:, -1]*s[-1]))


def halo_cancellation_bound(g, lam, coord):
    """Analytic near-null bias/outer-anchor direction on the observation interval."""
    scale=g.d[-1] if coord=="scaled" else np.sqrt(g.alpha[1:].sum())
    db=g.d[0];vector_norm=np.sqrt(1+(scale/db)**2)
    upper=2*scale*np.exp(-2*lam*g.radius)/vector_norm
    x=np.linspace(-1,1,16*g.n+1)
    u=(lam/g.h)*(x-g.centers[-1])
    e=np.exp(2*u)
    # Compute 1+tanh(u) without subtracting two almost equal numbers.
    pair=scale*(2*e/(1+e))/vector_norm
    return dict(n=g.n,uniform_lambda=lam,coordinates=coord,
                sigma_min_upper_bound=upper,condition_lower_bound=g.d[0]/upper,
                normalized_pair_image_norm=float(np.sqrt(np.mean(pair**2))),
                naive_tanh_plus_one_max=float(np.max(1+np.tanh(u))))


def spectral_probe(g, c, gamma, coord, eta, samples=16):
    x = np.linspace(-1, 1, samples*g.n+1)
    y = core.target(x, "sine", np)
    root_m = np.sqrt(len(x))
    a = diagnostics.features(x, g.centers, gamma)/root_m
    b = mapped_features(a, g, coord)
    r = a@c-y/root_m
    u, s, vh = svd(b, full_matrices=False, lapack_driver="gesdd")
    coefficients = u.T@r
    bounds, rb, _ = diagnostics.band_residuals(r)
    distances = x[:, None]-g.centers
    e = np.exp(-2*np.abs(distances*gamma))
    j = c[1:]*distances*(4*e/(1+e)**2)/(g.h*root_m)
    retained = s > 1e-12*s[0]
    parallel = u[:, retained]@coefficients[retained]
    glp,gln,leakage,split_error=projected_forces(j,r,u[:,retained])
    grad = b.T@r
    frozen_delta = -eta*(b@grad)
    expected_coefficients = (1-eta*s**2)*coefficients
    actual_coefficients = u.T@(r+frozen_delta)
    refits = []
    cv = {}
    xv = diagnostics.midpoint_grid(32768)
    for cutoff in CUTOFFS:
        keep = s > cutoff*s[0]
        z = vh[keep].T@((u[:, keep].T@(y/root_m))/s[keep])
        cf = training.decode(z, g, coord, np)
        residual = a@cf-y/root_m
        validation = diagnostics.prediction(xv, g.centers, cf, gamma)-core.target(xv, "sine", np)
        gp,gn,leak,closure=projected_forces(j,r,u[:,keep])
        refits.append(dict(cutoff=cutoff, rank=int(keep.sum()), train_mse=float(residual@residual),
                           validation_mse=float(np.mean(validation**2)), physical_coefficient_norm=float(np.linalg.norm(cf)),
                           gradient_lambda_parallel_norm=float(np.linalg.norm(gp)),
                           gradient_lambda_perpendicular_norm=float(np.linalg.norm(gn)),
                           projection_leakage_norm=float(leak),gradient_split_closure_norm=float(closure)))
        cv[f"refit_c_{cutoff:g}"] = cf
    record = dict(coordinates=coord, eta=eta, train_mse=float(r@r), sigma_max=float(s[0]),
                  smallest_singular=float(s[-1]), resolved_condition_1e12=float(s[0]/s[retained][-1]),
                  retained_rank_1e12=int(retained.sum()), total_columns=len(s),
                  adjacent_sign_disagreement=float(np.mean(gamma[:-1]*gamma[1:]<0)),
                  adjacent_absolute_gamma_ratio_quantiles=np.quantile(
                      np.maximum(np.abs(gamma[:-1]),np.abs(gamma[1:]))/
                      np.maximum(np.minimum(np.abs(gamma[:-1]),np.abs(gamma[1:])),1e-300),[.5,.9,.99]).tolist(),
                  readout_stability_limit=float(2/s[0]**2), eta_sigma_max_squared=float(eta*s[0]**2),
                  refits=refits, residual_outside_retained_span_mse=float(np.sum((r-parallel)**2)),
                  gradient_lambda_parallel_norm=float(np.linalg.norm(glp)),
                  gradient_lambda_perpendicular_norm=float(np.linalg.norm(gln)),
                  projection_leakage_norm=float(leakage),gradient_split_closure_norm=float(split_error),
                  readout_gradient_norm=float(np.linalg.norm(grad)),
                  frozen_readout_mse_change=float(2*r@frozen_delta+frozen_delta@frozen_delta),
                  frozen_local_mse_timescale=(float((r@r)/(2*eta*(grad@grad))) if eta and grad@grad else None),
                  modal_prediction_max_error=float(np.max(np.abs(actual_coefficients-expected_coefficients))),
                  fourier_parseval_error=float(abs(np.sum(rb**2)-r@r)))
    forces={}
    for region,mask in g.masks.items():
        direction=np.where(mask,np.sign(gamma),0.)
        direction/=max(np.linalg.norm(direction),1.)
        tangent=j@direction
        perp=tangent-u[:,retained]@(u[:,retained].T@tangent)
        forces[region]=dict(parallel_force=float(-direction@glp),perpendicular_force=float(-direction@gln),
                            tangent_norm=float(np.linalg.norm(tangent)),perpendicular_tangent_norm=float(np.linalg.norm(perp)))
    record["regional_scale_forces"]=forces
    arrays = dict(singular_values=s, residual_coefficients=coefficients, residual=r*root_m,
                  modal_gradient=s*coefficients, frozen_modal_delta=-eta*s**2*coefficients,
                  band_bounds=bounds, band_mse=np.sum(rb**2, axis=1),
                  band_gradient_native=rb@b, band_gradient_lambda=rb@j,
                  gradient_lambda_parallel=glp, gradient_lambda_perpendicular=gln,
                  c=c, gamma=gamma, **cv)
    return record, arrays, (u, s, vh)


def dense_probe(folder, g, case, end, output):
    dense = ratio_analysis.read_dense(folder, end)
    first, _, (u, singular, vh) = spectral_probe(g, dense["c"][0], dense["gamma"][0], case["coordinates"], case["eta"])
    x = np.linspace(-1, 1, 16*g.n+1)
    y = core.target(x, "sine", np)
    root_m = np.sqrt(len(x))
    rows = []
    for index in ratio.dense_sample_indices():
        c, gamma, dc, dl = (dense[k][index] for k in ("c", "gamma", "delta_c", "delta_lambda"))
        r, pieces, budget = ratio_analysis.update_budget(x, y, g.centers, g.h, c, gamma, dc, dl)
        normalized = r/root_m
        a = diagnostics.features(x, g.centers, gamma)/root_m
        b = mapped_features(a, g, case["coordinates"])
        distances = x[:, None]-g.centers
        e = np.exp(-2*np.abs(distances*gamma))
        j = c[1:]*distances*4*e/(1+e)**2/(g.h*root_m)
        bounds, rb, _ = diagnostics.band_residuals(normalized)
        gradient_c, gradient_l = rb@a, rb@j
        gradient_native = np.asarray(training.pullback(dense["gradient_c"][index], g, case["coordinates"]))
        coefficients = u.T@normalized
        keep = singular > 1e-12*singular[0]
        parallel = u[:, keep]@coefficients[keep]
        gp,gn,leak,split_error=projected_forces(j,normalized,u[:,keep])
        rows.append(dict(step=dense["step"][index], **budget, residual_mse=normalized@normalized,
                         singular_residual_coefficients=coefficients,
                         singular_update_coefficients=(pieces/root_m)@u,
                         singular_native_gradient_coefficients=vh@gradient_native,
                         band_mse=np.sum(rb**2, axis=1), band_gradient_native=rb@b,
                         band_gradient_lambda=gradient_l,
                         band_readout_linear_mse_change=2*gradient_c@dc,
                         band_geometry_linear_mse_change=2*gradient_l@dl,
                         gradient_lambda_parallel=gp,gradient_lambda_perpendicular=gn,
                         projection_leakage_norm=leak,gradient_split_closure_norm=split_error,
                         gradient_closure=np.array([np.linalg.norm(gradient_c.sum(axis=0)-dense["gradient_c"][index]),
                                                    np.linalg.norm(gradient_l.sum(axis=0)-dense["gradient_lambda"][index])]),
                         update_identity_error=np.array([np.max(np.abs(dc-training.decode(-case["eta"]*gradient_native,g,case["coordinates"],np))),
                                                          np.max(np.abs(dl+case["eta"]*dense["gradient_lambda"][index]))]),
                         readout_geometry_cosine=pieces[0]@pieces[1]/max(np.linalg.norm(pieces[0])*np.linalg.norm(pieces[1]), 1e-300)))
    arrays = {k:np.stack([r[k] for r in rows]) for k in rows[0]}
    run.save_arrays(output/f"dense_{end}.npz", **arrays, singular_values=singular, band_bounds=bounds)
    run.save_arrays(output/f"dense_parameters_{end}.npz", **{k:dense[k] for k in ("step", "c", "gamma")})
    result = dict(end=end, basis_step=end-2048, saved_states=2048, sampled_states=64,
                  mean_mse_change=arrays["mse_change"].mean(axis=0).tolist(),
                  mean_quadratic_cost=arrays["quadratic_mse_cost"].mean(axis=0).tolist(),
                  maximum_prediction_closure=float(arrays["closure_max"].max()),
                  maximum_gradient_closure=arrays["gradient_closure"].max(axis=0).tolist(),
                  maximum_update_identity_error=arrays["update_identity_error"].max(axis=0).tolist(),
                  mean_readout_geometry_cosine=float(arrays["readout_geometry_cosine"].mean()),
                  geometry_perpendicular_to_parallel_norm=float(np.linalg.norm(arrays["gradient_lambda_perpendicular"])/max(np.linalg.norm(arrays["gradient_lambda_parallel"]),1e-300)))
    with np.load(folder/f"checkpoint_{end:09d}.npz") as cp:
        result["all_2048_mean_mse_change"]=float((cp["train_mse"]-first["train_mse"])/2048)
    result["modal_fractions"] = [dict(relative_singular_threshold=t,
        residual_below=float(np.mean(np.sum(arrays["singular_residual_coefficients"][:, singular<t*singular[0]]**2, axis=1))/arrays["residual_mse"].mean()),
        update_below=(np.mean(np.sum(arrays["singular_update_coefficients"][:, :, singular<t*singular[0]]**2, axis=2), axis=0)/
                      np.maximum(arrays["quadratic_mse_cost"][:, :3].mean(axis=0), 1e-300)).tolist()) for t in (.1,.001,1e-4,1e-6)]
    return result


def analyze_case(root, output, case, end):
    key = training.case_key(case)
    folder, dest = root/key, output/key
    dest.mkdir(parents=True, exist_ok=True)
    g = core.geometry(case["n"])
    record = dict(case=case, end=end, checkpoints={}, dense=[])
    for step in sorted({0,20000,60000,end}):
        if step > end:
            continue
        checkpoint = folder/f"checkpoint_{step:09d}.npz"
        with np.load(checkpoint) as cp:
            c, gamma = cp["c"], cp["gamma"]
        probes = {}
        for coord in training.COORDINATES:
            stats, arrays, _ = spectral_probe(g, c, gamma, coord, case["eta"])
            probes[coord] = stats
            run.save_arrays(dest/f"spectrum_{step}_{coord}.npz", **arrays)
        record["checkpoints"][str(step)] = probes
        record.setdefault("checkpoint_sha256", {})[str(step)] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        run.write_json(dest/"mechanism.json", record)
        print(json.dumps(dict(case=key, spectral_step=step)), flush=True)
    for frontier in sorted({20000,60000,end}):
        if frontier <= end:
            record["dense"].append(dense_probe(folder, g, case, frontier, dest))
            run.write_json(dest/"mechanism.json", record)
            print(json.dumps(dict(case=key, dense_end=frontier)), flush=True)
    with np.load(folder/f"checkpoint_{end:09d}.npz") as cp:
        doubled, arrays, _ = spectral_probe(g, cp["c"], cp["gamma"], case["coordinates"], case["eta"], samples=32)
        # Detached post-hoc check: sign canonicalization preserves the function but
        # changes the coupled readout metric. It is never fed back into training.
        canonical_c=cp["c"].copy()
        canonical_c[1:]*=np.sign(cp["gamma"])
        canonical, canonical_arrays, _ = spectral_probe(g, canonical_c, np.abs(cp["gamma"]),
                                                        "scaled_differences", case["eta"])
    record["doubled_training_grid"] = doubled
    record["canonical_sign_diagnostic"] = canonical
    run.save_arrays(dest/"spectrum_doubled_grid.npz", **arrays)
    run.save_arrays(dest/"spectrum_canonical_signs.npz", **canonical_arrays)
    run.write_json(dest/"mechanism.json", record)


def uniform_references(output):
    rows = []
    dest = output/"uniform"
    dest.mkdir(exist_ok=True, parents=True)
    for n in (512,1024):
        g = core.geometry(n)
        for lam in (.125,.20,.25,.35,.50,1.):
            for coord in training.COORDINATES:
                record, arrays, _ = spectral_probe(g, np.zeros(g.width+1), np.full(g.width,lam/g.h), coord, 0.)
                record.update(n=n, uniform_lambda=lam)
                rows.append(record)
                run.save_arrays(dest/f"N{n}_lambda{lam:g}_{coord}.npz", **arrays)
                run.write_json(dest/"summary.json", rows)
            print(json.dumps(dict(uniform_n=n, uniform_lambda=lam)), flush=True)


def audit_endpoint_projection(root,output,case,end):
    key=training.case_key(case);dest=output/key;dest.mkdir(parents=True,exist_ok=True)
    with np.load(root/key/f"checkpoint_{end:09d}.npz") as cp:
        stats,arrays,_=spectral_probe(core.geometry(case["n"]),cp["c"],cp["gamma"],case["coordinates"],case["eta"])
    run.write_json(dest/"projection_audit.json",stats)
    run.save_arrays(dest/f'spectrum_{end}_{case["coordinates"]}.npz',**arrays)
    if (dest/"mechanism.json").exists():
        record=json.loads((dest/"mechanism.json").read_text())
        record["checkpoints"][str(end)][case["coordinates"]]=stats
        record["endpoint_projection_audit"]="Perpendicular residual reprojected to suppress floating-point leakage"
        run.write_json(dest/"mechanism.json",record)
    print(json.dumps(dict(projection_audit=key,end=end)),flush=True)


def figures(output):
    rows = json.loads((output/"summary.json").read_text())
    end=rows[0].get("comparison_step",100000)
    fig, axes = plt.subplots(1, 2, figsize=(12,4), layout="constrained")
    for ci,coord in enumerate(training.COORDINATES):
        group = sorted([r for r in rows if r["eligible"] and r["n"] == 512 and r["seed"] == 0 and r["coordinates"] == coord], key=lambda r:r["eta"])
        if not group:
            continue
        axes[0].loglog([r["eta"] for r in group], [r["window_mean_mse"] for r in group], "o-", label=LABELS[coord])
        failed=[r["eta"] for r in rows if r["failed_update"] and r["n"]==512 and r["seed"]==0 and r["coordinates"]==coord]
        axes[0].scatter(failed,np.full(len(failed),1.5+.5*ci),marker="x",color=f"C{ci}",s=50)
        selected = min(group, key=lambda r:(r["window_mean_mse"], r["eta"]))
        with np.load(output/selected["key"]/"history.npz") as h:
            axes[1].loglog(np.maximum(h["step"],1), h["train_mse"], label=f'{LABELS[coord]}, eta={selected["eta"]:g}')
    axes[0].set(xlabel="Constant shared eta", ylabel=f"Mean training MSE, updates {end-20000:,}–{end:,}",
                title="Matched scalar-rate comparison; N=512, seed 0" if end==100000 else "Selected continuations; no new rate selection")
    axes[0].text(.02,.02,"× = nonfinite update; symbol height is not MSE",transform=axes[0].transAxes,fontsize=8)
    axes[1].set(xlabel="Updates", ylabel="Checkpoint training MSE", title="Each arm's best tested scalar rate")
    for ax in axes:
        ax.grid(alpha=.2); ax.legend(fontsize=8)
    fig.savefig(output/"rate_sweep.png", dpi=160); plt.close(fig)
    if end>100000:
        fig,ax=plt.subplots(figsize=(9,5),layout="constrained")
        for r in rows:
            if not r["eligible"] or r["n"]!=512: continue
            with np.load(output/r["key"]/"window_mse.npz") as a:
                color="C0" if r["coordinates"]=="scaled" else "C1"
                ax.loglog(a["step"],a["mean"],color=color,ls="-" if r["seed"]==0 else "--",
                          label=f'{LABELS[r["coordinates"]]}, seed {r["seed"]}, eta={r["eta"]:g}')
                ax.fill_between(a["step"],a["quantiles"][1],a["quantiles"][3],color=color,alpha=.09)
        ax.set(xlabel="Updates (window end)",ylabel="Training MSE",title="Constant-rate continuation: 20k means and 10–90% ranges")
        ax.grid(alpha=.2);ax.legend(fontsize=8)
        fig.savefig(output/"continuation_progress.png",dpi=160);plt.close(fig)
    for path in sorted(output.glob("N*/mechanism.json")):
        record = json.loads(path.read_text())
        coord, end = record["case"]["coordinates"], record["end"]
        required=[path.parent/f"spectrum_{end}_{c}.npz" for c in training.COORDINATES]
        required += [path.parent/f"dense_{end}.npz",path.parent/"history.npz"]
        if not all(p.exists() for p in required):
            print(json.dumps(dict(incomplete_figure_export=path.parent.name)),flush=True)
            continue
        fig, axes = plt.subplots(2, 2, figsize=(13,8), layout="constrained")
        for other in training.COORDINATES:
            with np.load(path.parent/f"spectrum_{end}_{other}.npz") as a:
                axes[0,0].semilogy(np.arange(1,len(a["singular_values"])+1), a["singular_values"]/a["singular_values"][0],
                                  label=f'{LABELS[other]} (sigma max {a["singular_values"][0]:.3g})')
        with np.load(path.parent/f"dense_{end}.npz") as a:
            bounds, energy = a["band_bounds"], a["band_mse"].mean(axis=0)
            labels = ["DC" if lo==0 else str(lo) if hi-lo==1 else f"{lo}–{hi-1}" for lo,hi in bounds]
            axes[0,1].bar(np.arange(len(labels)), energy)
            percent = 100*energy/energy.sum()
            for i in np.flatnonzero(percent>=1):
                axes[0,1].text(i, energy[i]*1.3, f"{percent[i]:.1f}%", ha="center", fontsize=8)
            axes[0,1].set_yscale("log")
            for field,label in [("band_readout_linear_mse_change","Readout"),("band_geometry_linear_mse_change","Geometry")]:
                axes[1,1].plot(np.arange(len(labels)), a[field].mean(axis=0), "o-", label=label)
            for ax in (axes[0,1], axes[1,1]):
                ax.set_xticks(np.arange(len(labels)), labels, rotation=55, ha="right")
            s = a["singular_values"]/a["singular_values"][0]
            resolved = s >= 1e-14
            axes[1,0].loglog(s[resolved], np.mean(a["singular_residual_coefficients"]**2, axis=0)[resolved], ".", label="Residual MSE")
            axes[1,0].loglog(s[resolved], np.mean(a["singular_update_coefficients"][:,0]**2, axis=0)[resolved], ".", label="Readout update energy")
        axes[0,0].axhline(1e-12,color=".5",ls="--",lw=1)
        axes[0,0].set(xlabel="Singular-value index", ylabel="Relative singular value", title="Same learned geometry, two coordinate maps")
        axes[0,1].set(xlabel="DFT index band (both signs)", ylabel="Residual MSE", title="Mean over 64 sampled late-window states; labels are fractions")
        axes[1,0].set(xlabel="Relative singular value, fixed window-start basis", ylabel="Mean squared modal coefficient", title="Residual and readout motion; values below 1e-14 omitted")
        axes[1,1].set(xlabel="DFT index band (both signs)", ylabel="Signed linear MSE change per update", title="Negative values reduce MSE; mean of 64 states")
        axes[1,1].set_yscale("symlog", linthresh=1e-18)
        for ax in axes.flat:
            ax.grid(alpha=.18)
        for ax in (axes[0,0], axes[1,0], axes[1,1]):
            ax.legend(fontsize=8)
        fig.suptitle(f'{LABELS[coord]}, N={record["case"]["n"]}, seed={record["case"]["seed"]}, eta={record["case"]["eta"]:g}; update {end:,}')
        fig.savefig(path.parent/"mechanism.png", dpi=150); plt.close(fig)
        with np.load(path.parent/f"spectrum_{end}_{coord}.npz") as a:
            residual=a["residual"]; x=np.linspace(-1,1,len(residual))
            taper=np.hanning(len(x)); tapered=residual*taper/np.sqrt(np.mean(taper*taper))
            bounds,bands,_=diagnostics.band_residuals(tapered/np.sqrt(len(x)))
            audit=dict(endpoint_step=end,outer_tenth_mse_fraction=float(np.sum(residual[np.abs(x)>.9]**2)/np.sum(residual**2)),
                       endpoint_residual_jump=float(residual[-1]-residual[0]),
                       tapered_over_original_mse=float(np.sum(tapered**2)/np.sum(residual**2)),
                       band_bounds=bounds.tolist(),tapered_band_mse=np.sum(bands**2,axis=1).tolist())
            with np.load(path.parent/f"dense_{end}.npz") as dense_a:
                audit["all_2048_mean_mse_change"]=float((np.mean(residual**2)-dense_a["residual_mse"][0])/2048)
            run.write_json(path.parent/"spatial_audit.json",audit)
            fig,axes=plt.subplots(1,2,figsize=(12,4),layout="constrained")
            axes[0].plot(x,residual,lw=1);axes[0].set(xlabel="Physical x",ylabel="Prediction minus target",title=f"Actual residual at update {end:,}")
            axes[0].axvspan(-1,-.9,color=".93");axes[0].axvspan(.9,1,color=".93")
            labels=["DC" if lo==0 else str(lo) if hi-lo==1 else f"{lo}–{hi-1}" for lo,hi in bounds]
            axes[1].semilogy(np.arange(len(bounds)),a["band_mse"],"o-",label="Original residual")
            axes[1].semilogy(np.arange(len(bounds)),np.sum(bands**2,axis=1),"o-",label="Hann weighted; unit mean-square window")
            axes[1].set_xticks(np.arange(len(bounds)),labels,rotation=55,ha="right")
            axes[1].set(ylabel="Band MSE",xlabel="DFT index band (both signs)",title="Boundary sensitivity; weighting changes the measured norm")
            axes[1].legend(fontsize=8)
            for ax in axes: ax.grid(alpha=.2)
            fig.savefig(path.parent/"residual_shape.png",dpi=150);plt.close(fig)
        with np.load(path.parent/"history.npz") as h:
            fig, axes = plt.subplots(2,2,figsize=(12,7),layout="constrained")
            centers = h["centers"]
            mask = np.abs(centers)<=1
            for region, color in ((mask,"C0"),(~mask,"C1")):
                q = np.quantile(np.abs(h["lambda"][:,region]), [.1,.5,.9],axis=1)
                axes[0,0].plot(h["step"],q[1],color=color,label="Core" if region is mask else "Halo")
                axes[0,0].fill_between(h["step"],q[0],q[2],color=color,alpha=.15)
            axes[0,0].axhline(.25,color=".4",ls="--",label="Reference 0.25")
            axes[0,0].set(xlabel="Updates",ylabel="Absolute lambda",title="Median and 10–90% range")
            for field,ax,label in [("c",axes[0,1],"Physical readout w"),("gamma",axes[1,1],"Physical gamma")]:
                for i in (0,len(h["step"])//2,len(h["step"])-1):
                    values = h[field][i,1:] if field=="c" else h[field][i]
                    ax.plot(centers,values,lw=.8,label=f'Update {int(h["step"][i]):,}')
                ax.set(xlabel="Fixed physical center",ylabel=label)
                ax.set_yscale("symlog",linthresh=.01 if field=="c" else .1)
            for field,travel,color in [("c","travel_c","C0"),("lambda","travel_lambda","C1")]:
                net=np.linalg.norm(h[field]-h[field][0],axis=1)
                path_length=np.linalg.norm(h[travel],axis=1)
                axes[1,0].plot(h["step"],net,color=color,label=f"{field}: net displacement")
                axes[1,0].plot(h["step"],path_length,color=color,ls="--",label=f"{field}: accumulated absolute movement")
            axes[1,0].set(xlabel="Updates",ylabel="Euclidean norm",title="Movement; different physical units are labeled")
            axes[1,0].set_yscale("symlog",linthresh=1e-5)
            for ax in axes.flat:
                ax.grid(alpha=.18); ax.legend(fontsize=8)
            fig.savefig(path.parent/"parameters.png",dpi=150); plt.close(fig)
    uniform=output/"uniform"/"summary.json"
    if uniform.exists():
        rows=json.loads(uniform.read_text())
        run.write_json(output/"uniform"/"halo_cancellation_bounds.json",
                       [halo_cancellation_bound(core.geometry(r["n"]),r["uniform_lambda"],r["coordinates"]) for r in rows])
        fig,axes=plt.subplots(1,3,figsize=(16,4),layout="constrained")
        for n,style in ((512,"-"),(1024,"--")):
            for coord,color in zip(training.COORDINATES,("C0","C1")):
                group=[r for r in rows if r["n"]==n and r["coordinates"]==coord]
                axes[0].semilogy([r["uniform_lambda"] for r in group],[r["resolved_condition_1e12"] for r in group],"o"+style,color=color,label=f"{LABELS[coord]}, N={n}")
                axes[1].plot([r["uniform_lambda"] for r in group],[r["retained_rank_1e12"]/r["total_columns"] for r in group],"o"+style,color=color)
                axes[2].semilogy([r["uniform_lambda"] for r in group],[r["refits"][1]["train_mse"] for r in group],"o"+style,color=color)
        axes[0].set(xlabel="Prescribed uniform lambda",ylabel="Condition number on retained modes",title="Full scaled matrix: bias, anchor, and halos included")
        axes[1].set(xlabel="Prescribed uniform lambda",ylabel="Retained rank / number of columns",title="Relative cutoff 1e-12; unresolved modes are excluded")
        axes[2].set(xlabel="Prescribed uniform lambda",ylabel="Detached readout-fit training MSE",title="Sine approximation and conditioning differ")
        axes[0].legend(fontsize=7)
        for ax in axes: ax.grid(alpha=.2)
        fig.savefig(output/"uniform_conditioning.png",dpi=160);plt.close(fig)


def animations(output, late=False):
    from matplotlib.animation import FuncAnimation, FFMpegWriter
    from .animate_parameters import player_html
    matplotlib.rcParams["animation.embed_limit"]=256
    rows=json.loads((output/"summary.json").read_text())
    for seed in (0,1):
        selected=[]
        for coord in training.COORDINATES:
            group=[r for r in rows if r["eligible"] and r["n"]==512 and r["seed"]==seed and r["coordinates"]==coord]
            if group: selected.append(min(group,key=lambda r:(r["window_mean_mse"],r["eta"])))
        if len(selected)!=2: continue
        histories=[]
        for row in selected:
            with np.load(output/row["key"]/"history.npz") as a: h=dict(a)
            if late:
                with np.load(output/row["key"]/f'dense_parameters_{int(h["step"][-1])}.npz') as a:
                    h={**dict(a),"centers":h["centers"]}
            histories.append(h)
        common=sorted(set(histories[0]["step"])&set(histories[1]["step"]))
        if late:
            common=common[::8]+([common[-1]] if common[-1] not in common[::8] else [])
            frames=common;fps=20
        elif common[-1]>300000:
            common=[s for s in common if s<1000 or s<=300000 and s%10000==0 or s>300000 and s%100000==0 or s==common[-1]]
            frames=np.repeat(common,[6 if s<=300000 else 1 for s in common]);fps=6
        else:
            frames=common;fps=1
        fig,axes=plt.subplots(2,2,figsize=(13,7),layout="constrained",sharex=True)
        lines=[];bias_labels=[]
        for j,(row,h) in enumerate(zip(selected,histories)):
            for i,field in enumerate(("c","gamma")):
                values=h[field][:,1:] if field=="c" else h[field]
                if late: values=values-values[0]
                bound=max(float(np.max(np.abs(values)))*1.05,1e-8)
                line,=axes[i,j].plot(h["centers"],values[0],".",ms=3)
                lines.append((line,i,j,field))
                axes[i,j].set(ylim=(-bound,bound),ylabel=("Change in " if late else "")+("physical w" if i==0 else "physical gamma"))
                axes[i,j].set_yscale("symlog",linthresh=max(bound/1000,1e-15) if late else .01 if i==0 else .1)
                axes[i,j].grid(alpha=.15)
                axes[i,j].axvspan(h["centers"][0],-1,color=".93")
                axes[i,j].axvspan(1,h["centers"][-1],color=".93")
            axes[0,j].set_title(f'{LABELS[row["coordinates"]]}; constant shared eta={row["eta"]:g}')
            axes[1,j].set_xlabel("Fixed physical center")
            bias_labels.append(axes[0,j].text(.02,.97,"",transform=axes[0,j].transAxes,va="top",fontsize=9))
        heading=fig.suptitle("")
        def update(frame):
            step=frames[frame]
            for line,i,j,field in lines:
                h=histories[j]; index=int(np.searchsorted(h["step"],step))
                value=h[field][index]-h[field][0] if late else h[field][index]
                line.set_ydata(value[1:] if field=="c" else value)
                if field=="c": bias_labels[j].set_text(f'Bias b={h["c"][index,0]:+.5g}')
            cadence=(f'Changes since {histories[0]["step"][0]:,}; every 8 updates at 20 frames/s' if late else
                     "First 300k: one checkpoint/s; later: six/s" if frames[-1]>300000 else "One checkpoint per second")
            heading.set_text(f"Seed {seed}; update {step:,}; actual saved states, no interpolation\n{cadence}")
        movie=FuncAnimation(fig,update,frames=len(frames),interval=1000/fps,repeat=False)
        suffix="_late" if late else ""
        movie.save(output/f"seed_{seed}{suffix}.mp4",writer=FFMpegWriter(fps=fps,codec="libx264",bitrate=1500),dpi=100)
        (output/f"seed_{seed}{suffix}.html").write_text(player_html(movie,fps))
        update(len(frames)-1);fig.savefig(output/f"seed_{seed}{suffix}_last.png",dpi=120);plt.close(fig)
        run.write_json(output/f"animation_seed_{seed}{suffix}.json",dict(cases=[r["key"] for r in selected],steps=[int(s) for s in frames],fps=fps))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--end", type=int, default=100000)
    parser.add_argument("--cases", type=Path)
    parser.add_argument("--uniform", action="store_true")
    parser.add_argument("--figures", action="store_true")
    parser.add_argument("--animations", action="store_true")
    parser.add_argument("--late",action="store_true",help="Animate the final dense window as changes from its first state")
    parser.add_argument("--projection-only",action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.animations:
        animations(args.output,args.late)
    elif args.cases:
        for case in json.loads(args.cases.read_text()):
            if args.projection_only: audit_endpoint_projection(args.root,args.output,case,args.end)
            else: analyze_case(args.root, args.output, case, args.end)
    elif args.uniform:
        uniform_references(args.output)
    elif args.figures:
        figures(args.output)
    else:
        summarize(args.root, args.output, args.end)


if __name__ == "__main__":
    main()
