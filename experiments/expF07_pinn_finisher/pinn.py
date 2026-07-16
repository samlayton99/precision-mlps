"""Vanilla torch tanh-MLP PINN for the expF06 Burgers problem, plus a numpy
fields adapter so the frozen net can serve as base_fields in expF06's Newton
loop. Everything float64 on CPU."""
from __future__ import annotations

import numpy as np
import torch

import burgers as bp  # expF06's burgers problem (sys.path order set by caller)

torch.set_default_dtype(torch.float64)


class PINN(torch.nn.Module):
    def __init__(self, width=64, depth=4):
        super().__init__()
        layers = [torch.nn.Linear(2, width), torch.nn.Tanh()]
        for _ in range(depth - 1):
            layers += [torch.nn.Linear(width, width), torch.nn.Tanh()]
        layers += [torch.nn.Linear(width, 2)]
        self.net = torch.nn.Sequential(*layers)

    def forward(self, X):
        return self.net(X)


def _derivs(net, X):
    """u, v and first/second derivatives at X [n,2] (requires_grad)."""
    out = net(X)
    u, v = out[:, 0], out[:, 1]
    d = {}
    for name, w in (("u", u), ("v", v)):
        g = torch.autograd.grad(w.sum(), X, create_graph=True)[0]
        wx, wy = g[:, 0], g[:, 1]
        gxx = torch.autograd.grad(wx.sum(), X, create_graph=True)[0][:, 0]
        gyy = torch.autograd.grad(wy.sum(), X, create_graph=True)[0][:, 1]
        d[name], d[name + "x"], d[name + "y"] = w, wx, wy
        d["lap_" + name] = gxx + gyy
    return d


def pde_loss(net, X, nu):
    d = _derivs(net, X)
    P = X.detach().numpy()
    fu = torch.from_numpy(bp.f_u(P, nu))
    fv = torch.from_numpy(bp.f_v(P, nu))
    Fu = d["u"] * d["ux"] + d["v"] * d["uy"] - nu * d["lap_u"] - fu
    Fv = d["u"] * d["vx"] + d["v"] * d["vy"] - nu * d["lap_v"] - fv
    return (Fu**2).mean() + (Fv**2).mean()


def bc_loss(net, Xb):
    out = net(Xb)
    P = Xb.detach().numpy()
    gu = torch.from_numpy(bp.u_exact(P))
    gv = torch.from_numpy(bp.v_exact(P))
    return ((out[:, 0] - gu)**2).mean() + ((out[:, 1] - gv)**2).mean()


def _eval_rel_l2(net, n=100):
    g = np.linspace(-0.995, 0.995, n)
    P = np.stack(np.meshgrid(g, g, indexing="ij"), -1).reshape(-1, 2)
    with torch.no_grad():
        out = net(torch.from_numpy(P)).numpy()
    ue = bp.u_exact(P)
    return float(np.linalg.norm(out[:, 0] - ue) / np.linalg.norm(ue))


def train_pinn(nu, steps=50000, batch=1024, n_bc=256, lr=1e-3, bc_weight=10.0,
               eval_every=500, seed=0):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net = PINN()
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    history = []
    for step in range(steps):
        X = torch.from_numpy(rng.uniform(-1, 1, (batch, 2))).requires_grad_(True)
        s = rng.uniform(-1, 1, n_bc)
        edge = rng.integers(0, 4, n_bc)
        Pb = np.empty((n_bc, 2))
        Pb[:, 0] = np.where(edge < 2, s, np.where(edge == 2, -1.0, 1.0))
        Pb[:, 1] = np.where(edge >= 2, s, np.where(edge == 0, -1.0, 1.0))
        Xb = torch.from_numpy(Pb)
        loss = pde_loss(net, X, nu) + bc_weight * bc_loss(net, Xb)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if step % eval_every == 0 or step == steps - 1:
            rec = dict(step=step, loss=float(loss.item()),
                       rel_l2_u=_eval_rel_l2(net))
            history.append(rec)
            print(rec, flush=True)
    return net, history


def pinn_fields(net):
    """Frozen-net numpy fields adapter: P [n,2] -> dict of numpy [n] arrays
    (u, ux, uy, lap_u, v, vx, vy, lap_v) — the base_fields contract of
    expF06 newton.newton_burgers."""
    def fields(P, chunk=2048):
        out = {k: np.empty(len(P)) for k in
               ["u", "ux", "uy", "lap_u", "v", "vx", "vy", "lap_v"]}
        for i in range(0, len(P), chunk):
            X = torch.from_numpy(np.ascontiguousarray(P[i:i + chunk]))
            X.requires_grad_(True)
            d = _derivs(net, X)
            for k in out:
                out[k][i:i + chunk] = d[k].detach().numpy()
        return out
    return fields
