"""p-bit construction and inference, with an FP64 linear-solve exception."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("MPLCONFIGDIR", "/tmp/precisionmlps-mpl")
import gmpy2 as g
import numpy as np
from numpy.polynomial.chebyshev import chebval
from scipy.linalg import lstsq,norm
from threadpoolctl import threadpool_limits
import yaml

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from experiments.expC12_mhaskar_comparison import pbit
from experiments.expC12_mhaskar_comparison.construction import chirp,round_bits,TanhNetwork
from experiments.expC12_mhaskar_comparison.run import relative_l2,json_write

HERE=Path(__file__).resolve().parent
OUT=pbit.OUT
PROTOCOL={
    "construction":"All arithmetic at p bits, including QUILL geometry/bandwidth and Mhaskar projection, polynomial conversion, Taylor recurrence, stencil construction and merging",
    "solve_exception":"QUILL SVD in FP64 on p-bit features and labels; returned readout immediately rounded to p bits",
    "inference":"p-bit inputs, stored affine parameters, separate products/additions, correctly rounded tanh, sequential readout; no FMA",
    "data_boundary":"Sampling coordinates, hyperparameter search grids, and reference target observations are external FP64 data, rounded when entering model calculations",
    "metric":"FP64 relative L2 evaluation against original unrounded target observations; FP64 metrics also used for model selection",
    "exponent":"MPFR significand-p arithmetic; completed model parameters must be exactly storable in FP64 arrays",
    "emulation":"MPFR provides correctly rounded p-bit operations; not native hardware p-bit arithmetic",
}


def run():
    cfg=yaml.safe_load((HERE/"config.yaml").read_text())
    cfg["rounding"]="all_p_bit_except_linear_solve_and_reference_metrics"
    cfg["activation_bias"]="log(2)/2 computed at p bits"
    cfg["validation_screen_stride"]=8
    cfg["selection"]="minimum full-validation error; safely prune only using partial residual norm lower bounds"
    data,models=OUT/"data",OUT/"models"
    data.mkdir(parents=True,exist_ok=True);models.mkdir(exist_ok=True)
    sources=[HERE/"pbit.py",HERE/"pbit_kernels.c",HERE/"strict.py",HERE/"config.yaml",
             HERE/"construction.py",HERE/"run.py",
             HERE.parent/"expC09_bandwidth_figures/strict_arithmetic.c",
             HERE.parent/"expC09_bandwidth_figures/strict_precision.py"]
    json_write(data/"config.json",{"config":cfg,"precision_model":PROTOCOL,
                                  "mpfr_version":g.mpfr_version(),
                                  "sources":{str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in sources}})
    bits=np.arange(cfg["precision_min"],cfg["precision_max"]+1)
    degrees=np.array(cfg["degrees"])
    steps=np.geomspace(cfg["step_min"],cfg["step_max"],cfg["step_count"])
    xv=-1+(np.arange(cfg["validation_points"])+.5)*2/cfg["validation_points"]
    xs=xv[::cfg["validation_screen_stride"]]
    xe=np.linspace(-1.,1.,cfg["evaluation_points"])
    xt=np.linspace(-1.,1.,cfg["train_points"])
    nodes=np.cos(np.pi*(np.arange(cfg["chebyshev_points"])+.5)/cfg["chebyshev_points"])
    yv,ys,ye,yt,yn=[chirp(x) for x in (xv,xs,xe,xt,nodes)]
    assert np.intersect1d(xv,xe).size==0
    assert np.array_equal(xs,xv[::8])
    validation=np.full((len(steps),len(degrees),len(bits)),np.nan)
    screen=np.full_like(validation,np.inf)
    lower=np.full_like(validation,np.inf)
    pruned=np.zeros_like(validation,dtype=bool)
    finite=np.zeros_like(validation,dtype=bool)
    rows=[]
    np.savez(data/"grids.npz",validation=xv,screen=xs,evaluation=xe,truth=ye)
    start=time.monotonic()
    last_update=start
    for pi,p in enumerate(bits):
        p=int(p)
        coeff,poly,taylor,bias=pbit.polynomial_data(nodes,yn,int(degrees[-1]),p)
        np.savez(data/f"construction_p{p}.npz",chebyshev=coeff,monomial=poly,taylor=taylor,bias=bias)
        best=None
        for hi,step in enumerate(steps):
            candidates=[]
            for di,degree in enumerate(degrees):
                try:
                    model=pbit.construct(poly[degree],taylor,int(degree),float(step),bias,p)
                except FloatingPointError:
                    continue
                finite[hi,di,pi]=True
                candidates.append((di,int(degree),model))
            if not candidates:continue
            # The construction uses common slopes k*(h/2), rounded operation by operation.
            largest=max(candidates,key=lambda item:item[1])
            common=pbit.features(xs,largest[2].slope,largest[2].bias,p)
            for di,degree,model in candidates:
                offset=largest[1]-degree
                phi=np.ascontiguousarray(common[:,offset:offset+model.width])
                estimate=pbit.readout(phi,model.readout,model.offset,p)
                screen[hi,di,pi]=relative_l2(estimate,ys)
                bound=relative_l2(estimate,ys)*(norm(ys)/norm(yv))
                lower[hi,di,pi]=bound
                # The sum of squared residuals on a subset cannot exceed the
                # full sum. This rejects only candidates that cannot win.
                if best is not None and bound>best["error"]*(1+1e-12):
                    pruned[hi,di,pi]=True
                    continue
                full=pbit.evaluate(model,xv,p)
                np.testing.assert_array_equal(full[::8],estimate)
                error=relative_l2(full,yv)
                validation[hi,di,pi]=error
                if best is None or error<best["error"]:
                    best={"error":error,"degree":degree,"step":float(step),"model":model,"hi":hi,"di":di}
            if time.monotonic()-last_update>20:
                print(f"p={p}, Mhaskar steps {hi+1}/{len(steps)}; elapsed {time.monotonic()-start:.0f}s",flush=True)
                last_update=time.monotonic()
        assert best is not None and np.isfinite(best["error"])
        assert np.all(lower[:,:,pi][pruned[:,:,pi]]>best["error"])
        # QUILL: p-bit geometry, features and labels; FP64 SVD only.
        slope,b,meta=pbit.quill_geometry(cfg["width_budget"],cfg["halo_per_side"],p)
        phi=pbit.features(xt,slope,b,p)
        np.testing.assert_array_equal(phi,round_bits(phi,p))
        a=np.column_stack((phi,np.ones(len(xt))))
        rhs=round_bits(yt,p)
        w,_,rank,_=lstsq(a,rhs,cond=2.**(1-p),lapack_driver="gelsd")
        w=round_bits(w,p)
        quill=TanhNetwork(slope,b,w[:-1],w[-1])
        mhaskar=best["model"]
        qout=pbit.evaluate(quill,xe,p);mout=pbit.evaluate(mhaskar,xe,p)
        for name,model,predicted in [("quill",quill,qout),("mhaskar",mhaskar,mout)]:
            for arr in (model.slope,model.bias,model.readout,model.offset,predicted):
                np.testing.assert_array_equal(arr,round_bits(arr,p))
            model.save(models/f"{name}_p{p}.npz")
            saved=np.load(models/f"{name}_p{p}.npz")
            reloaded=TanhNetwork(saved["slope"],saved["bias"],saved["readout"],saved["offset"])
            np.testing.assert_array_equal(pbit.evaluate(reloaded,xe,p),predicted)
        row={"p":p,"width_budget":cfg["width_budget"],"quill_width":quill.width,
             "quill_lambda":meta["lambda"],"quill_geometry":meta,"quill_rank":int(rank),
             "quill_error":relative_l2(qout,ye),"mhaskar_width":mhaskar.width,
             "mhaskar_degree":best["degree"],"mhaskar_step":best["step"],
             "mhaskar_validation_error":best["error"],"mhaskar_error":relative_l2(mout,ye),
             "mhaskar_readout_l1":float(norm(mhaskar.readout,1)),
             "mhaskar_step_boundary":best["hi"] in (0,len(steps)-1),
             "finite_candidates":int(finite[:,:,pi].sum()),"pruned_candidates":int(pruned[:,:,pi].sum())}
        rows.append(row)
        json_write(data/"summary.json",rows)
        print(f"p={p} complete: QUILL={row['quill_error']:.4g}, Mhaskar={row['mhaskar_error']:.4g}; {time.monotonic()-start:.0f}s",flush=True)
        last_update=time.monotonic()
    polynomial_errors=[relative_l2(chebval(xe,coeff[:d+1]),ye) for d in degrees]
    np.savez_compressed(data/"mhaskar_search.npz",degrees=degrees,steps=steps,bits=bits,
                        validation_error=validation,screen_error=screen,partial_norm_lower_bound=lower,
                        safely_pruned=pruned,finite_models=finite,polynomial_error=polynomial_errors)
    diag_degrees=[4,8,16,32,64,128]
    np.savez(data/"diagnostics.npz",degrees=diag_degrees,steps=steps,
             network_error=np.array([screen[:,list(degrees).index(d),-1] for d in diag_degrees]))
    json_write(data/"validation.json",{
        "complete":True,"precisions":len(rows),"candidate_count_per_precision":len(steps)*len(degrees),
        "saved_models_verified":2*len(rows),"all_saved_parameters_and_outputs_exactly_p_bit":True,
        "saved_model_replay_bitwise_identical":True,"screen_is_subset_of_validation":True,
        "screen_outputs_match_full_validation_bitwise":True,"all_pruned_lower_bounds_exceed_winner_error":True,
        "selection_uses_reporting_grid":False,"pbit_features_and_labels_before_fp64_svd":True,
        "all_construction_and_inference_ops_pbit_except_svd":True,"elapsed_seconds":time.monotonic()-start})
    return rows


if __name__=="__main__":
    with threadpool_limits(limits=1):run()
    from experiments.expC12_mhaskar_comparison.plot import main
    main(OUT)
