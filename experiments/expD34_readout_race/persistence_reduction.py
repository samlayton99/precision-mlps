"""Frozen coarse-balanced quadratic/cubic residuals with a fixed hard drive."""
from __future__ import annotations
import numpy as np
from . import persistence_theory as pt


def reduced_state(p, x, y, q):
    state = pt.tensors(p, x, y)
    modal = q.T @ state['J']/len(x)
    residual = q.T @ state['r']/len(x)
    coarse = modal[:2]
    tangent = modal[[2, 3, 9]].T
    tangent -= coarse.T @ np.linalg.solve(coarse @ coarse.T, coarse @ tangent)
    generated, hard = tangent[:, :2], tangent[:, 2]*residual[9]
    kernel = generated.T @ generated
    values, vectors = np.linalg.eigh(kernel)
    if values.min() <= 0: raise ValueError('Requires independent generated-mode tangents')
    equilibrium = -np.linalg.solve(kernel, generated.T @ hard)
    return dict(p0=p.copy(), generated=generated, hard=hard, initial=residual[2:4],
                equilibrium=equilibrium, values=values, vectors=vectors)


def reduced_at(state, n, eta=.002, driven=True):
    values, vectors = state['values'], state['vectors']
    if eta*values.max() >= 1: raise ValueError('Requires nonoscillating reduced GD')
    equilibrium = state['equilibrium'] if driven else np.zeros(2)
    floor = state['hard']+state['generated'] @ equilibrium if driven else np.zeros_like(state['hard'])
    loading = vectors.T @ (state['initial']-equilibrium)
    directions = state['generated'] @ vectors
    exponent = n*np.log1p(-eta*values)
    integral = -np.expm1(exponent)/values
    p = state['p0']-directions @ (integral*loading)-eta*n*floor
    force = directions @ (np.exp(exponent)*loading)+floor
    residual = equilibrium+vectors @ (np.exp(exponent)*loading)
    w = (len(p)-1)//3
    transient_budget = np.linalg.norm(directions[:w], axis=0) @ (abs(loading)/values)
    finite_path = np.linalg.norm(directions[:w], axis=0) @ (abs(loading)*integral)+eta*n*np.linalg.norm(floor[:w])
    return dict(p=p, force=force, residual=residual, transient_budget=transient_budget,
                force_floor=np.linalg.norm(floor[:w]), slope_path_bound=finite_path)
