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


def read_trace(folder, end):
    steps, traces = [], []
    for path in sorted(folder.glob("trace_*.npz")):
        _, lo, hi = path.stem.split("_")
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
        rows.append(row)
    run.write_json(output/"summary.json", rows)
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
    glp, gln = j.T@parallel, j.T@(r-parallel)
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
        refits.append(dict(cutoff=cutoff, rank=int(keep.sum()), train_mse=float(residual@residual),
                           validation_mse=float(np.mean(validation**2)), physical_coefficient_norm=float(np.linalg.norm(cf))))
        cv[f"refit_c_{cutoff:g}"] = cf
    record = dict(coordinates=coord, eta=eta, train_mse=float(r@r), sigma_max=float(s[0]),
                  smallest_singular=float(s[-1]), resolved_condition_1e12=float(s[0]/s[retained][-1]),
                  retained_rank_1e12=int(retained.sum()), total_columns=len(s),
                  readout_stability_limit=float(2/s[0]**2), eta_sigma_max_squared=float(eta*s[0]**2),
                  refits=refits, residual_outside_retained_span_mse=float(np.sum((r-parallel)**2)),
                  gradient_lambda_parallel_norm=float(np.linalg.norm(glp)),
                  gradient_lambda_perpendicular_norm=float(np.linalg.norm(gln)),
                  modal_prediction_max_error=float(np.max(np.abs(actual_coefficients-expected_coefficients))),
                  fourier_parseval_error=float(abs(np.sum(rb**2)-r@r)))
    arrays = dict(singular_values=s, residual_coefficients=coefficients, residual=r*root_m,
                  modal_gradient=s*coefficients, frozen_modal_delta=-eta*s**2*coefficients,
                  band_bounds=bounds, band_mse=np.sum(rb**2, axis=1),
                  band_gradient_native=rb@b, band_gradient_lambda=rb@j,
                  gradient_lambda_parallel=glp, gradient_lambda_perpendicular=gln,
                  c=c, gamma=gamma, **cv)
    return record, arrays, (u, s, vh)


def dense_probe(folder, g, case, end, output):
    dense = ratio_analysis.read_dense(folder, end)
    _, _, (u, singular, vh) = spectral_probe(g, dense["c"][0], dense["gamma"][0], case["coordinates"], case["eta"])
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
        gn = np.asarray(training.pullback(dense["gradient_c"][index], g, case["coordinates"]))
        coefficients = u.T@normalized
        keep = singular > 1e-12*singular[0]
        parallel = u[:, keep]@coefficients[keep]
        rows.append(dict(step=dense["step"][index], **budget, residual_mse=normalized@normalized,
                         singular_residual_coefficients=coefficients,
                         singular_update_coefficients=(pieces/root_m)@u,
                         singular_native_gradient_coefficients=vh@gn,
                         band_mse=np.sum(rb**2, axis=1), band_gradient_native=rb@b,
                         band_gradient_lambda=gradient_l,
                         band_readout_linear_mse_change=2*gradient_c@dc,
                         band_geometry_linear_mse_change=2*gradient_l@dl,
                         gradient_lambda_parallel=j.T@parallel,
                         gradient_lambda_perpendicular=j.T@(normalized-parallel),
                         gradient_closure=np.array([np.linalg.norm(gradient_c.sum(axis=0)-dense["gradient_c"][index]),
                                                    np.linalg.norm(gradient_l.sum(axis=0)-dense["gradient_lambda"][index])]),
                         readout_geometry_cosine=pieces[0]@pieces[1]/max(np.linalg.norm(pieces[0])*np.linalg.norm(pieces[1]), 1e-300)))
    arrays = {k:np.stack([r[k] for r in rows]) for k in rows[0]}
    run.save_arrays(output/f"dense_{end}.npz", **arrays, singular_values=singular, band_bounds=bounds)
    run.save_arrays(output/f"dense_parameters_{end}.npz", **{k:dense[k] for k in ("step", "c", "gamma")})
    result = dict(end=end, basis_step=end-2048, saved_states=2048, sampled_states=64,
                  mean_mse_change=arrays["mse_change"].mean(axis=0).tolist(),
                  mean_quadratic_cost=arrays["quadratic_mse_cost"].mean(axis=0).tolist(),
                  maximum_prediction_closure=float(arrays["closure_max"].max()),
                  maximum_gradient_closure=arrays["gradient_closure"].max(axis=0).tolist(),
                  mean_readout_geometry_cosine=float(arrays["readout_geometry_cosine"].mean()),
                  geometry_perpendicular_to_parallel_norm=float(np.linalg.norm(arrays["gradient_lambda_perpendicular"])/max(np.linalg.norm(arrays["gradient_lambda_parallel"]),1e-300)))
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
    record["doubled_training_grid"] = doubled
    run.save_arrays(dest/"spectrum_doubled_grid.npz", **arrays)
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


def figures(output):
    rows = json.loads((output/"summary.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(12,4), layout="constrained")
    for coord in training.COORDINATES:
        group = sorted([r for r in rows if r["eligible"] and r["n"] == 512 and r["seed"] == 0 and r["coordinates"] == coord], key=lambda r:r["eta"])
        if not group:
            continue
        axes[0].loglog([r["eta"] for r in group], [r["window_mean_mse"] for r in group], "o-", label=LABELS[coord])
        selected = min(group, key=lambda r:(r["window_mean_mse"], r["eta"]))
        with np.load(output/selected["key"]/"history.npz") as h:
            axes[1].loglog(np.maximum(h["step"],1), h["train_mse"], label=f'{LABELS[coord]}, eta={selected["eta"]:g}')
    axes[0].set(xlabel="Constant shared eta", ylabel="Mean training MSE, updates 80k–100k", title="Matched scalar-rate comparison; N=512, seed 0")
    axes[1].set(xlabel="Updates", ylabel="Checkpoint training MSE", title="Each arm's best tested scalar rate")
    for ax in axes:
        ax.grid(alpha=.2); ax.legend(fontsize=8)
    fig.savefig(output/"rate_sweep.png", dpi=160); plt.close(fig)
    for path in sorted(output.glob("N*/mechanism.json")):
        record = json.loads(path.read_text())
        coord, end = record["case"]["coordinates"], record["end"]
        fig, axes = plt.subplots(2, 2, figsize=(13,8), layout="constrained")
        for other in training.COORDINATES:
            with np.load(path.parent/f"spectrum_{end}_{other}.npz") as a:
                axes[0,0].semilogy(np.arange(1,len(a["singular_values"])+1), a["singular_values"]/a["singular_values"][0], label=LABELS[other])
        with np.load(path.parent/f"dense_{end}.npz") as a:
            bounds, energy = a["band_bounds"], a["band_mse"].mean(axis=0)
            labels = ["DC" if lo==0 else f"{lo}–{hi-1}" for lo,hi in bounds]
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
            axes[1,0].loglog(s, np.mean(a["singular_residual_coefficients"]**2, axis=0), ".", label="Residual MSE")
            axes[1,0].loglog(s, np.mean(a["singular_update_coefficients"][:,0]**2, axis=0), ".", label="Readout update energy")
        axes[0,0].axhline(1e-12,color=".5",ls="--",lw=1)
        axes[0,0].set(xlabel="Singular-value index", ylabel="Relative singular value", title="Same learned geometry, two coordinate maps")
        axes[0,1].set(xlabel="DFT index band (both signs)", ylabel="Residual MSE", title="Mean over 64 sampled late-window states; labels are fractions")
        axes[1,0].set(xlabel="Relative singular value, fixed window-start basis", ylabel="Mean squared modal coefficient", title="Residual occupancy and actual readout motion")
        axes[1,1].set(xlabel="DFT index band (both signs)", ylabel="Signed linear MSE change per update", title="Negative values reduce MSE; mean of 64 states")
        axes[1,1].set_yscale("symlog", linthresh=1e-18)
        for ax in axes.flat:
            ax.grid(alpha=.18)
        for ax in (axes[0,0], axes[1,0], axes[1,1]):
            ax.legend(fontsize=8)
        fig.suptitle(f'{LABELS[coord]}, N={record["case"]["n"]}, seed={record["case"]["seed"]}, eta={record["case"]["eta"]:g}; update {end:,}')
        fig.savefig(path.parent/"mechanism.png", dpi=150); plt.close(fig)
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
        fig,axes=plt.subplots(1,2,figsize=(12,4),layout="constrained")
        for n,style in ((512,"-"),(1024,"--")):
            for coord,color in zip(training.COORDINATES,("C0","C1")):
                group=[r for r in rows if r["n"]==n and r["coordinates"]==coord]
                axes[0].semilogy([r["uniform_lambda"] for r in group],[r["resolved_condition_1e12"] for r in group],"o"+style,color=color,label=f"{LABELS[coord]}, N={n}")
                axes[1].plot([r["uniform_lambda"] for r in group],[r["retained_rank_1e12"]/r["total_columns"] for r in group],"o"+style,color=color)
        axes[0].set(xlabel="Prescribed uniform lambda",ylabel="Condition number on retained modes",title="Full scaled matrix: bias, anchor, and halos included")
        axes[1].set(xlabel="Prescribed uniform lambda",ylabel="Retained rank / number of columns",title="Relative cutoff 1e-12; unresolved modes are excluded")
        axes[0].legend(fontsize=7)
        for ax in axes: ax.grid(alpha=.2)
        fig.savefig(output/"uniform_conditioning.png",dpi=160);plt.close(fig)


def animations(output):
    from matplotlib.animation import FuncAnimation, FFMpegWriter
    from .animate_parameters import player_html
    rows=json.loads((output/"summary.json").read_text())
    for seed in (0,1):
        selected=[]
        for coord in training.COORDINATES:
            group=[r for r in rows if r["eligible"] and r["n"]==512 and r["seed"]==seed and r["coordinates"]==coord]
            if group: selected.append(min(group,key=lambda r:(r["window_mean_mse"],r["eta"])))
        if len(selected)!=2: continue
        histories=[]
        for row in selected:
            with np.load(output/row["key"]/"history.npz") as a: histories.append(dict(a))
        common=sorted(set(histories[0]["step"])&set(histories[1]["step"]))
        fig,axes=plt.subplots(2,2,figsize=(13,7),layout="constrained",sharex=True)
        lines=[]
        for j,(row,h) in enumerate(zip(selected,histories)):
            for i,field in enumerate(("c","gamma")):
                values=h[field][:,1:] if field=="c" else h[field]
                bound=max(float(np.max(np.abs(values)))*1.05,1e-8)
                line,=axes[i,j].plot(h["centers"],values[0],".",ms=3)
                lines.append((line,i,j,field))
                axes[i,j].set(ylim=(-bound,bound),ylabel="Physical w" if i==0 else "Physical gamma")
                axes[i,j].set_yscale("symlog",linthresh=.01 if i==0 else .1)
                axes[i,j].grid(alpha=.15)
                axes[i,j].axvspan(h["centers"][0],-1,color=".93")
                axes[i,j].axvspan(1,h["centers"][-1],color=".93")
            axes[0,j].set_title(f'{LABELS[row["coordinates"]]}; constant shared eta={row["eta"]:g}')
            axes[1,j].set_xlabel("Fixed physical center")
        heading=fig.suptitle("")
        def update(frame):
            step=common[frame]
            for line,i,j,field in lines:
                h=histories[j]; index=int(np.searchsorted(h["step"],step))
                line.set_ydata(h[field][index,1:] if field=="c" else h[field][index])
            heading.set_text(f"Seed {seed}; update {step:,}; actual checkpoints, no interpolation\nPhysical readouts above physical slopes; one checkpoint per second")
        movie=FuncAnimation(fig,update,frames=len(common),interval=1000,repeat=False)
        movie.save(output/f"seed_{seed}.mp4",writer=FFMpegWriter(fps=1,codec="libx264",bitrate=1500),dpi=100)
        (output/f"seed_{seed}.html").write_text(player_html(movie,1))
        update(len(common)-1);fig.savefig(output/f"seed_{seed}_last.png",dpi=120);plt.close(fig)
        run.write_json(output/f"animation_seed_{seed}.json",dict(cases=[r["key"] for r in selected],steps=[int(s) for s in common],fps=1))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--end", type=int, default=100000)
    parser.add_argument("--cases", type=Path)
    parser.add_argument("--uniform", action="store_true")
    parser.add_argument("--figures", action="store_true")
    parser.add_argument("--animations", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.animations:
        animations(args.output)
    elif args.cases:
        for case in json.loads(args.cases.read_text()):
            analyze_case(args.root, args.output, case, args.end)
    elif args.uniform:
        uniform_references(args.output)
    elif args.figures:
        figures(args.output)
    else:
        summarize(args.root, args.output, args.end)


if __name__ == "__main__":
    main()
