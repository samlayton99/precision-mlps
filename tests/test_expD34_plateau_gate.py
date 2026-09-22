import json
import pytest
from experiments.expD34_readout_race import plateau_gate as gate


def test_partial_and_unresolved_bundles_stop_dependent_stage(tmp_path):
    paths=[]
    for seed in range(3):
        for step in (100000,600000):
            path=tmp_path/'probes'/f'seed{seed}_{step}'/'status.json';path.parent.mkdir(parents=True)
            path.write_text(json.dumps(dict(complete=True,failed=0,unresolved_steps=0,motion_identity=1e-14,channel_identity=1e-14)))
            paths.append(path)
    gate.check(tmp_path,'discovery')
    for broken in (dict(complete=False,failed=0),dict(complete=True,failed=1),dict(complete=True,failed=0,unresolved_steps=1)):
        paths[-1].write_text(json.dumps(broken))
        with pytest.raises(ValueError,match='Incomplete or invalid'):gate.check(tmp_path,'discovery')
