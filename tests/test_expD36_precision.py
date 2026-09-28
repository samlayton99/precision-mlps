import numpy as np

from experiments.expD36_frozen_gamma_probe import core, precision


def test_full_grid_precision_witness_matches_independent_qr():
    g = core.geometry(64)
    x = np.linspace(-1, 1, 129)
    j = core.design(x, g.centers, 4)
    y = core.target(x, 'sine_mix_2_6_10')[:, None]/np.sqrt(len(x))
    qr = core.polynomial_transform(x, 3)
    e, mu, _ = core.access(core.transform(qr, y), core.transform(qr, j), 3)
    high = precision.witness(64, 129, 4, 3, 50)
    np.testing.assert_allclose(float(high['E']), e[3, 0], rtol=1e-13)
    np.testing.assert_allclose(float(high['mu']), mu[3, 0], rtol=1e-12)
    assert float(high['polynomial_norm_neighbor_error']) < 1e-45


def test_full_precision_targets_and_neighbor_map():
    import time
    from experiments.expD36_frozen_gamma_probe import full_core as f, full_precision as p
    n=64; m=129; g=core.geometry(n); x=np.linspace(-1,1,m); qr=core.polynomial_transform(x,3)
    for name in ['raw','collective_neighbor']:
        for target in f.config()['targets']:
            j=core.design(x,g.centers,4)@f.map_matrix(g,name)
            y=f.target(x,target)[:,None]/np.sqrt(m)
            e,mu,_=core.access(core.transform(qr,y),core.transform(qr,j),3)
            k=1 if target=='quadratic' else 3
            high=p.witness((n,m,4,k,50,target,name,time.monotonic()+60))
            np.testing.assert_allclose(float(high['E']),e[k,0],rtol=1e-12)
            np.testing.assert_allclose(float(high['mu']),mu[k,0],rtol=1e-11)
