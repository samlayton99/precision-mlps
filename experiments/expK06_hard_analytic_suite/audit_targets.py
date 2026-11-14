"""Independent high-precision value/derivative and complete real halo audit."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
import mpmath as mp
from experiments.expK06_hard_analytic_suite.targets import *
from experiments.expK06_hard_analytic_suite.common import *

def main():
    rows=[]
    with mp.workdps(70):
        for name in NAMES:
            x=np.array([-.973,-.71,-.317,-.031,0,.071,.134,.217,.538,.913,1.])
            y=target(name,x);dy=derivative(name,x)
            exact=np.array([float(mp_target(name,mp.mpf(float(z)),mp)) for z in x])
            if name=='kepler':dexact=np.array([float(mp.pi/(1-mp.mpf('.9')*mp.cos(mp_target(name,mp.mpf(float(z)),mp)))) for z in x])
            else:dexact=np.array([float(mp.diff(lambda t:mp_target(name,t,mp),mp.mpf(float(z)))) for z in x])
            value_error=float(np.max(abs(y-exact))/max(1,np.max(abs(exact))))
            derivative_error=float(np.max(abs(dy-dexact))/max(1,np.max(abs(dexact))))
            assert value_error<2e-13,(name,value_error)
            assert derivative_error<2e-12,(name,derivative_error)
            halo_checks=[]
            for n in NS:
                xx=-1+2/n*np.arange(-halo(n),n+halo(n)+1)
                yy=target(name,xx);dd=derivative(name,xx)
                assert np.isfinite(yy).all() and np.isfinite(dd).all()
                halo_checks.append(dict(N=n,sampling_interval=[float(xx.min()),float(xx.max())],max_value=float(abs(yy).max()),max_derivative=float(abs(dd).max())))
            rows.append(dict(target=name,entire=name in ENTIRE,value_relative_scale_error=value_error,
                             derivative_relative_scale_error=derivative_error,halos=halo_checks))
            print(name,value_error,derivative_error,flush=True)
    save(OUT/'target_audit.json',rows)
if __name__=='__main__':main()
