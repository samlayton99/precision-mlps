"""Experimental fp64 local-window ridge operator with a matching transpose.

The only approximation is truncating tanh-minus-step and derivative tails.
Uniform centers, fixed common slope, and derivatives 0..2 are supported.
Projections and outputs use O(MK); coefficient/work arrays use O(MN).
"""
from __future__ import annotations
import ctypes
import hashlib
import os
from pathlib import Path
import subprocess
import tempfile
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'results/checkpoint_F_applications/expF20_local_ridge_pde'


def library():
    source=Path(__file__).with_name('local_kernel.c')
    key=hashlib.sha256(source.read_bytes()).hexdigest()[:16]
    OUT.mkdir(parents=True,exist_ok=True)
    target=OUT/f'kernel_{key}.so'
    if not target.exists():
        descriptor,name=tempfile.mkstemp(prefix='kernel_build_',suffix='.so',dir=OUT)
        os.close(descriptor)
        try:
            subprocess.run(['clang','-O3','-shared','-fPIC',str(source),'-o',name],check=True)
            os.replace(name,target)
        finally:
            if os.path.exists(name):os.unlink(name)
    lib=ctypes.CDLL(str(target))
    arr=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS')
    args=[ctypes.c_int]*3+[arr,arr]+[ctypes.c_double]*4+[ctypes.c_int,arr]
    for name in ('profiles','transpose'):
        getattr(lib,name).argtypes=args
        getattr(lib,name).restype=None
    return lib


class LocalBank:
    def __init__(self,directions,centers,gamma,cutoff=18.):
        self.directions=np.asarray(directions,dtype=float)
        self.centers=np.asarray(centers,dtype=float)
        self.gamma,self.cutoff=float(gamma),float(cutoff)
        if self.directions.ndim!=2 or not np.all(np.isfinite(self.directions)):
            raise ValueError('directions must be a finite M-by-d matrix')
        if self.centers.ndim!=1 or len(self.centers)<2 or not np.all(np.isfinite(self.centers)):
            raise ValueError('centers must be a finite vector of length >=2')
        self.h=float((self.centers[-1]-self.centers[0])/(len(self.centers)-1))
        tolerance=64*np.finfo(float).eps*max(1.,np.max(abs(self.centers)))
        if self.h<=0 or not np.allclose(self.centers,self.centers[0]+self.h*np.arange(len(self.centers)),rtol=0,atol=tolerance):
            raise ValueError('centers must be uniformly increasing')
        if not 0<self.gamma<np.inf or not 0<self.cutoff<=100:
            raise ValueError('positive finite gamma and cutoff <=100 required')
        self.m,self.n=len(self.directions),len(self.centers)
        self.size=self.m*self.n
        self.lib=library()

    def bind(self,points):
        return LocalPlan(self,points)


class LocalPlan:
    def __init__(self,bank,points):
        self.bank=bank
        self.points=np.asarray(points,dtype=float)
        if self.points.ndim!=2 or self.points.shape[1]!=bank.directions.shape[1] or not np.all(np.isfinite(self.points)):
            raise ValueError('points must be a finite K-by-d matrix')
        self.projection=np.ascontiguousarray(bank.directions@self.points.T)
        self.k=len(self.points)

    def _call(self,name,values,order):
        if order not in (0,1,2):raise ValueError('derivative order must be 0, 1 or 2')
        b=self.bank
        shape=(b.m,b.n) if name=='profiles' else (b.m,self.k)
        v=np.asarray(values,dtype=float)
        if v.size!=np.prod(shape):raise ValueError(f'expected input shape {shape}')
        v=np.ascontiguousarray(v.reshape(shape))
        out=np.empty((b.m,self.k) if name=='profiles' else (b.m,b.n))
        getattr(b.lib,name)(b.m,b.n,self.k,self.projection,v,b.centers[0],b.h,b.gamma,b.cutoff,order,out)
        return out

    def profiles(self,weights,order=0):
        return self._call('profiles',weights,order)

    def forward(self,weights,order=0,factors=None):
        profiles=self.profiles(weights,order)
        return profiles.sum(axis=0) if factors is None else np.asarray(factors)@profiles

    def adjoint_profiles(self,residual,order=0):
        return self._call('transpose',residual,order)

    def adjoint(self,residual,order=0,factors=None):
        b=self.bank
        factors=np.ones(b.m) if factors is None else np.asarray(factors)
        return self.adjoint_profiles(factors[:,None]*np.asarray(residual)[None,:],order)
