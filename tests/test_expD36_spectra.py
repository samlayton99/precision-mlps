import numpy as np
from experiments.expD36_ssb_geometry_switching.spectra import summarize


def test_frequency_accounting_preserves_mse_and_descent():
    m=1024;x=np.arange(m)/m
    residual=(.2+np.sin(2*np.pi*64*x)+.3*np.cos(2*np.pi*128*x))/np.sqrt(m)
    readout=np.cos(2*np.pi*128*x)/np.sqrt(m)
    geometry=-np.sin(2*np.pi*64*x)/np.sqrt(m)
    rows=summarize(residual,readout,geometry)
    np.testing.assert_allclose(sum(q['residual_mse'] for q in rows),residual@residual,rtol=1e-14)
    np.testing.assert_allclose(sum(q['residual_percent'] for q in rows),100.)
    np.testing.assert_allclose(sum(q['readout_descent'] for q in rows),-2*residual@readout,rtol=1e-14)
    np.testing.assert_allclose(sum(q['geometry_descent'] for q in rows),-2*residual@geometry,rtol=1e-14)
    band=next(q for q in rows if q['indices']==[64,127])
    np.testing.assert_allclose(band['residual_mse'],.5,rtol=1e-14)
