import numpy as np
import pytest

pytest.importorskip('flint')
from experiments.expD36_frozen_gamma_probe import cap_resolvent as r


def test_resolvent_improves_conversion_with_an_interval_certificate():
    cert = dict(status='interval_certified', beta=.00042, delta=.19, label='control')
    result = r.improve(cert, 204)
    assert result['status'] == 'interval_certified'
    assert result['bound'] > 700
    assert result['proof']['error_squared_lower'] > .01**2
    # This abstract operator satisfies the same Rayleigh and angle information.
    rate = cert['beta']/cert['delta']**2
    hit = int(np.ceil(np.log(.01)/np.log1p(-.5*rate)))
    assert result['bound'] <= hit


def test_resolvent_enclosure_against_random_noncommuting_geometry():
    rng = np.random.default_rng(42)
    for _ in range(5):
        q, _ = np.linalg.qr(rng.normal(size=(7, 7)))
        rates = np.geomspace(.0001, 1., 7)
        h = (q*rates)@q.T
        y = rng.normal(size=7); y /= np.linalg.norm(y)
        v = y+.2*rng.normal(size=7); v /= np.linalg.norm(v)
        beta = float(v@h@v); delta = abs(float(v@y)); steps = 13
        candidate = r.scalar_candidate(steps, beta, delta)
        proof = r.enclose(steps, beta, delta, candidate['t'], candidate['z'],
                          epsilon=.9, max_intervals=256)
        actual = np.sum((q.T@y)**2*(1-.5*rates)**(2*steps))
        assert proof['error_squared_lower'] <= actual+1e-13
