"""expD20 -- dataset loaders: the incumbent expF04 six, plus candidates.

Every loader returns (Xtr, ytr, Xte, yte) as float64 numpy, already split.
Downloads are cached under <repo>/data/cache_expD20/ as .npz, so a rerun is
free. A loader that cannot reach its source raises; the driver catches and
records the failure rather than aborting the sweep.

Candidate selection is biased toward SIMULATION / PHYSICAL-SURROGATE data:
this project approximates smooth functions, and a dataset that is really a
sampled deterministic map has little label noise, which is the only regime
where high precision is even conceivable.
"""
from __future__ import annotations

import io
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
CACHE = REPO_ROOT / "data" / "cache_expD20"
CACHE.mkdir(parents=True, exist_ok=True)
SEED = 0


def _get(url, timeout=180):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 expD20"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def _split(X, y, frac=0.8, seed=SEED):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    k = int(frac * len(X))
    return X[idx[:k]], y[idx[:k]], X[idx[k:]], y[idx[k:]]


def _cached(name):
    """Decorator: cache a loader's four arrays to one npz."""
    def deco(fn):
        def wrapped(*a, **kw):
            p = CACHE / f"{name}.npz"
            if p.exists():
                z = np.load(p)
                return z["Xtr"], z["ytr"], z["Xte"], z["yte"]
            Xtr, ytr, Xte, yte = fn(*a, **kw)
            np.savez_compressed(p, Xtr=Xtr, ytr=ytr, Xte=Xte, yte=yte)
            return Xtr, ytr, Xte, yte
        wrapped.__name__ = fn.__name__
        return wrapped
    return deco


def _cap(X, y, n_max):
    """Subsample to keep experiments cheap; stated in the writeup where used."""
    if len(X) <= n_max:
        return X, y
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(X), n_max, replace=False)
    return X[idx], y[idx]


# ==========================================================================
# incumbents -- read from the expF04 cache so the comparison is like for like
# ==========================================================================

def load_incumbent(name):
    import torch
    p = REPO_ROOT / "data" / "cache_all20" / f"{name}.pt"
    d = torch.load(p, weights_only=False)
    return (np.asarray(d["xtr"], dtype=np.float64), np.asarray(d["ytr"], dtype=np.float64),
            np.asarray(d["xte"], dtype=np.float64), np.asarray(d["yte"], dtype=np.float64))


INCUMBENTS = ["superconductivity", "sarcos", "airfoil", "parkinsons",
              "bike_sharing", "beijing_pm25"]


# ==========================================================================
# candidates -- UCI physical / simulation surrogates
# ==========================================================================

@_cached("ccpp")
def load_ccpp():
    """Combined Cycle Power Plant: 4 sensor inputs -> net electrical output.
    Smooth thermodynamic response, near-deterministic. n=9568, d=4."""
    raw = _get("https://archive.ics.uci.edu/static/public/294/combined+cycle+power+plant.zip")
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        name = [n for n in z.namelist() if n.lower().endswith(".xlsx")][0]
        blob = z.read(name)
    # xlsx without pandas/openpyxl: sheet1 is a flat table of numbers
    import xml.etree.ElementTree as ET
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        shared = []
        if "xl/sharedStrings.xml" in z.namelist():
            root = ET.fromstring(z.read("xl/sharedStrings.xml"))
            shared = ["".join(t.text or "" for t in si.iter() if t.tag.endswith("}t"))
                      for si in root if si.tag.endswith("}si")]
        sheet = ET.fromstring(z.read("xl/worksheets/sheet1.xml"))
        rows = []
        for row in sheet.iter():
            if not row.tag.endswith("}row"):
                continue
            vals = []
            for c in row:
                if not c.tag.endswith("}c"):
                    continue
                v = None
                for ch in c:
                    if ch.tag.endswith("}v"):
                        v = ch.text
                if v is None:
                    continue
                if c.get("t") == "s":
                    vals.append(shared[int(v)])
                else:
                    vals.append(v)
            if vals:
                rows.append(vals)
    data = []
    for r in rows:
        try:
            data.append([float(x) for x in r])
        except (ValueError, TypeError):
            continue  # header
    arr = np.array(data, dtype=np.float64)
    X, y = arr[:, :4], arr[:, 4]
    return _split(X, y)


@_cached("naval")
def load_naval():
    """Condition-based maintenance of naval propulsion: a GAS-TURBINE SIMULATOR.
    16 features -> compressor decay coefficient. Numerically simulated, so
    essentially NOISELESS. n=11934, d=16."""
    raw = _get("https://archive.ics.uci.edu/static/public/316/condition+based+maintenance+of+naval+propulsion+plants.zip")
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        names = [n for n in z.namelist() if n.endswith("data.txt")]
        inner = [n for n in z.namelist() if n.endswith(".zip")]
        if inner and not names:
            with zipfile.ZipFile(io.BytesIO(z.read(inner[0]))) as z2:
                names = [n for n in z2.namelist() if n.endswith("data.txt")]
                txt = z2.read(names[0]).decode()
        else:
            txt = z.read(names[0]).decode()
    arr = np.array([[float(v) for v in ln.split()] for ln in txt.strip().splitlines()],
                   dtype=np.float64)
    # last two columns are the two decay coefficients; predict the compressor one
    X, y = arr[:, :16], arr[:, 16]
    return _split(X, y)


@_cached("gasturbine")
def load_gasturbine():
    """Gas turbine CO/NOx emissions, 5 years of sensor data -> turbine energy yield.
    n~36k, d=9. Real sensors, so genuinely noisy (contrast with naval)."""
    raw = _get("https://archive.ics.uci.edu/static/public/551/gas+turbine+co+and+nox+emission+data+set.zip")
    frames = []
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        for n in sorted(z.namelist()):
            if not n.lower().endswith(".csv"):
                continue
            txt = z.read(n).decode()
            lines = txt.strip().splitlines()
            rows = [[float(v) for v in ln.split(",")] for ln in lines[1:]]
            frames.append(np.array(rows, dtype=np.float64))
    arr = np.vstack(frames)
    # columns: AT AP AH AFDP GTEP TIT TAT TEY CDP CO NOX  -> predict TEY (index 7)
    y = arr[:, 7]
    X = np.delete(arr, 7, axis=1)
    return _split(X, y)


@_cached("casp")
def load_casp():
    """CASP protein tertiary structure: 9 physicochemical features -> RMSD.
    n=45730, d=9. A classic hard-nonlinear regression benchmark."""
    raw = _get("https://archive.ics.uci.edu/static/public/265/physicochemical+properties+of+protein+tertiary+structure.zip")
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        name = [n for n in z.namelist() if n.lower().endswith(".csv")][0]
        txt = z.read(name).decode()
    lines = txt.strip().splitlines()
    arr = np.array([[float(v) for v in ln.split(",")] for ln in lines[1:]], dtype=np.float64)
    X, y = arr[:, 1:], arr[:, 0]
    return _split(X, y)


# ==========================================================================
# candidates -- OpenML (sklearn fetch_openml)
# ==========================================================================

def _openml(data_id, target=None, n_max=60000, drop=()):
    """`drop` removes columns that leak the target (e.g. a binarized copy)."""
    from sklearn.datasets import fetch_openml
    ds = fetch_openml(data_id=data_id, as_frame=True, parser="auto")
    df = ds.frame
    tgt = target or ds.target_names[0]
    y = df[tgt].to_numpy(dtype=np.float64)
    Xdf = df.drop(columns=[tgt, *drop])
    # numeric only; one-hot small categoricals
    num, cat = [], []
    for c in Xdf.columns:
        if str(Xdf[c].dtype) in ("category", "object", "bool"):
            if Xdf[c].nunique() <= 20:
                cat.append(c)
        else:
            num.append(c)
    parts = [Xdf[num].to_numpy(dtype=np.float64)] if num else []
    for c in cat:
        d = Xdf[c].astype("category").cat.codes.to_numpy()
        k = d.max() + 1
        if k > 1:
            parts.append(np.eye(k)[d])
    X = np.hstack(parts)
    ok = np.isfinite(X).all(1) & np.isfinite(y)
    X, y = X[ok], y[ok]
    X, y = _cap(X, y, n_max)
    return _split(X, y)


@_cached("elevators")
def load_elevators():
    """F16 aircraft control simulation -> elevator action. Simulated, smooth. n~16.6k d=18."""
    return _openml(216)


@_cached("ailerons")
def load_ailerons():
    """F16 aileron control simulation. Sister task to elevators. n~13.75k d=40."""
    return _openml(296)


@_cached("pol")
def load_pol():
    """Telecom pole positioning; known to be strongly nonlinear (trees crush linear)."""
    return _openml(201)


@_cached("wind")
def load_wind():
    """Wind speed at Irish stations -- smooth spatiotemporal field. n~6.5k d=14."""
    return _openml(503)


@_cached("cpu_act")
def load_cpu_act():
    """Computer activity: system telemetry -> user CPU time. Nonlinear, n=8192 d=21."""
    return _openml(197)


@_cached("kin8nm")
def load_kin8nm():
    """8-link ROBOT ARM FORWARD KINEMATICS simulation. Nonlinear, near-noiseless,
    a sampled deterministic map -- the ideal shape for this project. n=8192 d=8."""
    return _openml(189)


@_cached("puma32h")
def load_puma32h():
    """PUMA 560 robot arm dynamics simulation, high nonlinearity variant. n=8192 d=32."""
    return _openml(308)


@_cached("bank8fm")
def load_bank8fm():
    """Simulated bank queue -> rejection rate (OpenML id 572 is bank8FM, d=8,
    not the 32-feature variant). Nonlinear simulation. n=8192."""
    return _openml(572)


@_cached("grid_stability")
def load_grid_stability():
    """Electrical grid stability SIMULATION: 12 params -> max real part of the
    characteristic root. A sampled deterministic map from a differential-equation
    model, so essentially noiseless. n=10000, d=12.

    `stabf` is a binarized copy of the target and MUST be dropped or it leaks."""
    return _openml(43007, target="stab", drop=("stabf",))


@_cached("house_16h")
def load_house_16h():
    """House prices from 16 demographic features. Real, noisy, strongly nonlinear.
    n=22784, d=16."""
    return _openml(574)


@_cached("fried")
def load_fried():
    """Friedman's synthetic benchmark: y = 10 sin(pi x1 x2) + 20(x3-.5)^2 + 10 x4 + 5 x5 + noise.
    A KNOWN smooth nonlinear function with only mild noise -- the cleanest possible
    test that a method can capture nonlinearity. n=40768 d=10."""
    return _openml(564)


CANDIDATES = {
    # UCI physical/simulation
    "ccpp": load_ccpp,
    "naval": load_naval,
    "gasturbine": load_gasturbine,
    "casp": load_casp,
    # OpenML simulation surrogates
    "kin8nm": load_kin8nm,
    "elevators": load_elevators,
    "ailerons": load_ailerons,
    "puma32h": load_puma32h,
    "bank8fm": load_bank8fm,
    "grid_stability": load_grid_stability,
    "pol": load_pol,
    "cpu_act": load_cpu_act,
    "wind": load_wind,
    "house_16h": load_house_16h,
    "fried": load_fried,
}

# Candidates that are sampled deterministic simulations, i.e. little or no label
# noise. This matters beyond convenience: a noiseless target is the only regime
# where high precision is even conceivable (expB01: y-noise sets a hard
# sigma*n^{-1/2} floor), so these are the tasks where the method's headline
# capability could ever be demonstrated on real-shaped data.
NEAR_NOISELESS = {"naval", "kin8nm", "elevators", "ailerons",
                  "puma32h", "bank8fm", "grid_stability", "pol", "ccpp", "fried"}
