import numpy as np
from experiments.expD34_readout_race import plateau_run as pr, adam_run as ar, targets


def test_confirmation_forecast_is_immutable_and_checkpoint_only(tmp_path):
    cases=pr.confirmation_cases(20)
    assert len(cases)==10 and all(c['seed']==20 for c in cases)
    z,d=targets.initial(128,24,20);p=np.r_[z.ravel(),d]
    state={'p':p[None]};cases=[next(c for c in cases if c['optimizer']=='gd' and c['target']=='moment9')]
    pr.issue_forecast(tmp_path,state,cases,100000)
    path=tmp_path/'forecast_100000.npz';original=path.read_bytes()
    pr.issue_forecast(tmp_path,state,cases,100000)
    assert path.read_bytes()==original
    with np.load(path) as f:
        assert int(f['issued_step'])==100000 and int(f['end_step'])==600000
        np.testing.assert_array_equal(f['initial_p'],p[None])
        assert np.all(np.isfinite(f['frozen_p']))
    import pytest
    with pytest.raises(ValueError,match='checkpoint changed'):
        pr.issue_forecast(tmp_path,{'p':(p+.01)[None]},cases,100000)
