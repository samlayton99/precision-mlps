import json
import numpy as np
from experiments.expD36_ssb_geometry_switching.analyze import collect


def test_terminal_failure_is_not_an_accepted_update(tmp_path):
    folder=tmp_path/'failed_case';folder.mkdir()
    (folder/'case.json').write_text('{}')
    (folder/'latest.json').write_text(json.dumps(dict(step=2,status='search_failed')))
    trace=np.zeros((3,18));trace[:,0]=[4.,2.,1.];trace[:,1]=[1,2,2]
    trace[:,2]=[1,1,0];trace[-1,14]=1
    np.savez(folder/'ssb_trace_0_2.npz',trace=trace)
    row=collect(tmp_path)[0]
    assert row['accepted_trace_rows']==2
    assert row['rejected_trace_rows']==1
    assert row['last_2048_mean_mse']==3.
    assert row['emergency_resets']==1
