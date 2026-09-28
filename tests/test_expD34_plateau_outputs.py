import json
from types import SimpleNamespace
import numpy as np
from experiments.expD34_readout_race import plateau_probes as pp, plateau_results as results


def test_probe_disk_resume_and_paired_endpoint_report(tmp_path,monkeypatch):
    monkeypatch.setattr(pp,'verify_gpu',lambda *args:None)
    source=tmp_path/'source'/'primary_0';source.mkdir(parents=True)
    cases=[dict(target=t,optimizer='gd') for t in pp.ANCHORS]
    (source/'manifest.json').write_text(json.dumps(dict(cases=cases)))
    p=np.random.default_rng(1).normal(size=532)*.1
    np.savez(source/'snapshots.npz',steps=np.array([100000]),p=np.tile(p,(7,1,1)))
    folder=tmp_path/'campaign'/'probes'/'seed0_100000'
    args=SimpleNamespace(source=source.parent,output=folder,seed=0,start=100000,half=False,
        samples=64,horizon=10,max_seconds=-1.,runtime='slurm')
    pp.run(args)
    assert json.loads((folder/'status.json').read_text())['offset']==1
    args.max_seconds=1000;pp.run(args)
    final=np.load(folder/'state.npz');expected={k:final[k].copy() for k in final.files}
    assert json.loads((folder/'status.json').read_text())['complete']
    args.output=tmp_path/'whole';pp.run(args)
    with np.load(args.output/'state.npz') as whole:
        for k,v in expected.items():np.testing.assert_array_equal(v,whole[k])
    results.run(tmp_path/'campaign',tmp_path/'analysis')
    import csv
    with open(tmp_path/'analysis'/'endpoints.csv') as handle:rows=list(csv.DictReader(handle))
    assert len(rows)==37
    assert max(float(r['motion_identity']) for r in rows)<1e-14
    assert all(float(r['eval_relative_mse'])>0 for r in rows)
    with open(tmp_path/'analysis'/'contrasts.csv') as handle:rows=list(csv.DictReader(handle))
    assert len(rows)==37
