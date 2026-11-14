"""Overlay the 16 original and eight extra-hard matched tanh-GD targets."""
from __future__ import annotations
import os
os.environ.setdefault("MPLCONFIGDIR","/private/tmp/precisionmlps-mpl")
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from experiments.expD56_scale_acquisition.plot import apply_style,measured_data,aggregate_figure


def main():
    base=ROOT/"results/checkpoint_D_optimizers"
    roots=[base/"expD56_scale_acquisition/sweep",base/"expD58_extra_hard_scale_acquisition/sweep"]
    manifests=[json.loads((r/"manifest.json").read_text()) for r in roots]
    configs=[m["config"] for m in manifests]
    for key in ("width","steps","learning_rates","initialization_multipliers","seed",
                "train_points","eval_points","loss","dtype","quill_reference","optimizer"):
        assert configs[0][key]==configs[1][key],f"Mismatched {key}"
    assert all(c.get("activation","tanh")=="tanh" for c in configs)
    # Verify the actual saved sample grids, not just their requested sizes.
    starts=[next((r/"runs").glob("*/initial.npz")) for r in roots]
    with np.load(starts[0]) as a,np.load(starts[1]) as b:
        for key in ("train_x","eval_x"):np.testing.assert_array_equal(a[key],b[key])
    targets={**configs[0]["targets"],**configs[1]["targets"]}
    names=[n for group in targets.values() for n in group]
    assert len(names)==len(set(names))==24
    data,hashes={},{}
    for root in roots:
        rows,source_hashes=measured_data(root)
        assert not (data.keys() & rows.keys()),"Overlapping runs"
        for status in (root/"runs").glob("*/status.json"):
            s=json.loads(status.read_text())
            assert s["state"]=="complete" and s["step"]==200000
        data.update(rows);hashes.update(source_hashes)
    assert len(data)==216
    assert all(row["step"][0]==0 and row["step"][-1]==200000 for row in data.values())
    config={**configs[0],"targets":targets,"figure_title":"Scale Acquisition Ablation: 24 functions",
            "show_median_legend":False}
    output=base/"expD58_extra_hard_scale_acquisition/combined_24_functions"
    output.mkdir(exist_ok=True)
    apply_style()
    aggregate_figure(config,data,output,False,"gamma_distribution",False,"linear")
    sources=[Path(__file__),ROOT/"experiments/expD56_scale_acquisition/plot.py"]
    (output/"figure_provenance.json").write_text(json.dumps({
        "source_sweeps":[str(r.relative_to(ROOT)) for r in roots],"functions":names,"runs":216,
        "source_sha256":hashes,"config":config,
        "renderer_sha256":{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        "line":"Per-function median absolute neuron slope",
        "band":"Per-function central 98% interval across 196 neurons",
        "median_annotation":"Median of the 72 final run medians in each initialization panel",
        "training":"All 216 runs are tanh with full-batch GD, 200000 updates"},indent=2)+"\n")
    print(output/"scale_acquisition.png")


if __name__=="__main__":main()
