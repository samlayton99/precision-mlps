"""Compensated p-bit algorithms; correction registers also have p bits."""
import ctypes
import hashlib
from pathlib import Path
import subprocess
import sys
import gmpy2 as g
import numpy as np
from experiments.expC12_mhaskar_comparison import pbit
from experiments.expC12_mhaskar_comparison.construction import TanhNetwork,round_bits

OUT=pbit.OUT.parent/"rescue"
MODES=("sequential","sorted","pairwise","neumaier","dot2","sorted_dot2")
_LIB=None


def library():
    global _LIB
    if _LIB is not None:return _LIB
    source=Path(__file__).with_name("robust_kernels.c")
    base=source.with_name("pbit_kernels.c")
    package=Path(g.__file__).parent
    libs=package.parent/"gmpy2.libs"
    mpfr=next(iter(list(libs.glob("libmpfr*.dylib"))+list(libs.glob("libmpfr*.so*"))))
    cache=OUT/"cache";cache.mkdir(parents=True,exist_ok=True)
    digest=hashlib.sha256(source.read_bytes()+base.read_bytes()+g.mpfr_version().encode()).hexdigest()[:16]
    compiled=cache/f"robust_{digest}.so"
    if not compiled.exists():
        subprocess.run(["cc","-O3","-shared","-fPIC","-I",str(package),str(source),str(mpfr),"-o",str(compiled)],check=True)
        if sys.platform=="darwin":
            name=subprocess.check_output(["otool","-D",str(mpfr)],text=True).splitlines()[1].strip()
            subprocess.run(["install_name_tool","-change",name,str(mpfr),str(compiled)],check=True)
    lib=ctypes.CDLL(str(compiled))
    a=np.ctypeslib.ndpointer(dtype=np.float64,flags="C_CONTIGUOUS")
    lib.rescued_readout.argtypes=[ctypes.c_size_t,ctypes.c_size_t,a,a,ctypes.c_double,ctypes.c_uint,ctypes.c_uint,a]
    lib.rescued_stencil.argtypes=[ctypes.c_uint,a,a,ctypes.c_double,ctypes.c_uint,a,a]
    lib.rescued_polynomial_data.argtypes=[ctypes.c_size_t,ctypes.c_uint,a,a,ctypes.c_double,ctypes.c_uint,a,a,a]
    for name in ("rescued_readout","rescued_stencil","rescued_polynomial_data"):getattr(lib,name).restype=ctypes.c_int
    _LIB=lib
    return lib


def readout(phi,weights,bias,p,mode="dot2"):
    phi,weights=[np.ascontiguousarray(a,dtype=np.float64) for a in (phi,weights)]
    out=np.empty(len(phi))
    if library().rescued_readout(*phi.shape,phi,weights,float(bias),p,MODES.index(mode),out):raise MemoryError()
    return out


def evaluate(model,x,p,mode="dot2"):
    return readout(pbit.features(x,model.slope,model.bias,p),model.readout,model.offset,p,mode)


def construct(poly,taylor,degree,step,bias,p):
    a=np.ascontiguousarray(poly[:degree+1],dtype=np.float64)
    t=np.ascontiguousarray(taylor[:degree+1],dtype=np.float64)
    slope,weights=np.empty(2*degree+1),np.empty(2*degree+1)
    status=library().rescued_stencil(degree,a,t,step,p,slope,weights)
    if status:raise FloatingPointError(f"rescued construction status {status}")
    return TanhNetwork(slope,np.full(len(slope),float(round_bits(bias,p))),weights,0.)


def polynomial_data(nodes,labels,degree,p):
    with g.context(precision=p,round=g.RoundToNearest):bias=float(g.log(g.mpfr(2))/g.mpfr(2))
    c,poly,t=np.empty(degree+1),np.empty((degree+1,degree+1)),np.empty(degree+1)
    nodes,labels=[np.ascontiguousarray(a,dtype=np.float64) for a in (nodes,labels)]
    if library().rescued_polynomial_data(len(nodes),degree,nodes,labels,bias,p,c,poly,t):raise MemoryError()
    return c,poly,t,bias
