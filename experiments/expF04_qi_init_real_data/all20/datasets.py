"""Dataset registry for the 20-task sweep (expF04/all20).

Each task reduces to a fixed-length feature vector, min-max normalized to [-1,1]
(regression targets too). The 4 already-fetched tasks delegate to the parent loaders;
the rest are best-effort parsers over files the fetcher pulls into <repo>/data/<name>/.

Design goals:
  - robust-to-missing: available(data_dir) reports only tasks whose files parse.
  - no heavy deps: numpy-only featurization (text -> hashing bag-of-words; categorical
    -> one-hot; sequence -> one-hot), so no sklearn/librosa/rdkit needed.
  - a task that fails to parse is skipped with a logged reason, never crashes the sweep.

Tasks needing auth or heavy deps are NOT registered here (mit_bih ECG / sdss redshift =
Kaggle auth; qm9 = rdkit; fsdd audio = many wav files). Flagged in the fetcher.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent
sys.path.insert(0, str(PARENT))
import data as PD   # parent data.py (the 4 fetched tasks)   # noqa: E402


@dataclass
class Task:
    name: str
    kind: str
    d_in: int
    d_out: int
    x_train: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray


PRETTY = {
    "fashion_mnist": "Fashion-MNIST", "uci_har": "UCI HAR",
    "superconductivity": "Superconductivity", "sarcos": "SARCOS",
    "airfoil": "Airfoil Self-Noise", "appliances_energy": "Appliances Energy",
    "parkinsons": "Parkinsons UPDRS", "online_news": "Online News Popularity",
    "bike_sharing": "Bike Sharing", "landsat": "Statlog Landsat",
    "covertype": "Covertype", "splice_dna": "Splice-junction DNA",
    "ag_news": "AG News", "gas_sensor": "Gas Sensor Drift",
    "nsl_kdd": "NSL-KDD", "beijing_pm25": "Beijing PM2.5",
}

# order: 4 anchors, then regression, then classification
TASK_ORDER = [
    "superconductivity", "sarcos", "airfoil", "appliances_energy", "parkinsons",
    "online_news", "bike_sharing", "beijing_pm25",
    "fashion_mnist", "uci_har", "landsat", "covertype", "splice_dna",
    "ag_news", "gas_sensor", "nsl_kdd",
]


# ----------------------------- featurization -----------------------------

def _minmax(xtr, xte):
    lo = xtr.min(0); hi = xtr.max(0); sc = hi - lo; sc[sc == 0] = 1.0
    return (np.clip(2 * (xtr - lo) / sc - 1, -1, 1),
            np.clip(2 * (xte - lo) / sc - 1, -1, 1))


def _onehot(col):
    vals = sorted(set(col.tolist()))
    idx = {v: i for i, v in enumerate(vals)}
    M = np.zeros((len(col), len(vals)), np.float32)
    for i, v in enumerate(col):
        M[i, idx[v]] = 1.0
    return M


def _split(X, y, seed=0, frac=0.2):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(X))
    n_te = int(frac * len(X))
    te, tr = perm[:n_te], perm[n_te:]
    return X[tr], y[tr], X[te], y[te]


def _pack(name, kind, xtr, ytr, xte, yte) -> Task:
    xtr, xte = np.asarray(xtr, np.float32), np.asarray(xte, np.float32)
    xtr, xte = _minmax(xtr, xte)
    if kind == "regr":
        ytr = np.asarray(ytr, np.float32).reshape(-1, 1)
        yte = np.asarray(yte, np.float32).reshape(-1, 1)
        lo = ytr.min(0); hi = ytr.max(0); sc = hi - lo; sc[sc == 0] = 1.0
        ytr = (2 * (ytr - lo) / sc - 1).ravel()
        yte = (2 * (yte - lo) / sc - 1).ravel()
        d_out = 1
    else:
        ytr = np.asarray(ytr, np.int64).ravel()
        yte = np.asarray(yte, np.int64).ravel()
        d_out = int(max(ytr.max(), yte.max())) + 1
    return Task(name, kind, xtr.shape[1], d_out, xtr, ytr, xte, yte)


def _text_hash(strings, dim=512):
    """Hashing bag-of-words: token -> hash bucket, log-count. numpy-only, no sklearn."""
    X = np.zeros((len(strings), dim), np.float32)
    for i, s in enumerate(strings):
        for tok in str(s).lower().split():
            X[i, hash(tok) % dim] += 1.0
    return np.log1p(X)


# ----------------------------- loaders -----------------------------

def _find(root: Path, pattern: str) -> Path:
    hits = sorted(root.rglob(pattern))
    if not hits:
        raise FileNotFoundError(f"{pattern} under {root}")
    return hits[0]


def _load_parent(name):
    t = PD.load_task(name, PARENT.parents[1] / "data")
    return Task(name, t.kind, t.d_in, t.d_out,
                t.x_train.numpy(), t.y_train.numpy(), t.x_test.numpy(), t.y_test.numpy())


def _load_airfoil(d):
    a = np.loadtxt(_find(d / "airfoil", "airfoil_self_noise.dat"))
    return _pack("airfoil", "regr", *_split(a[:, :-1], a[:, -1]))


def _load_appliances(d):
    import csv
    p = _find(d / "appliances_energy", "energydata_complete.csv")
    rows = list(csv.reader(open(p)))[1:]           # drop header; col0=date (string)
    data = []
    for r in rows:
        try:
            data.append([float(v) for v in r[1:]])  # Appliances .. rv2, all numeric
        except ValueError:
            continue
    a = np.array(data, np.float32)
    return _pack("appliances_energy", "regr", *_split(a[:, 1:], a[:, 0]))     # col0 (of a) = Appliances


def _load_parkinsons(d):
    p = _find(d / "parkinsons", "parkinsons_updrs.data")
    a = np.genfromtxt(p, delimiter=",", skip_header=1)
    # cols: subject#,age,sex,test_time,motor_UPDRS,total_UPDRS,then 16 voice features
    y = a[:, 5]                       # total_UPDRS
    X = np.delete(a, [0, 4, 5], axis=1)
    return _pack("parkinsons", "regr", *_split(X, y))


def _load_online_news(d):
    p = _find(d / "online_news", "OnlineNewsPopularity.csv")
    a = np.genfromtxt(p, delimiter=",", skip_header=1)[:, 2:]  # drop url,timedelta
    y = np.log1p(a[:, -1])            # log(shares)
    return _pack("online_news", "regr", *_split(a[:, :-1], y))


def _load_bike(d):
    import csv
    p = _find(d / "bike_sharing", "hour.csv")
    rows = list(csv.reader(open(p)))
    hdr = rows[0]
    drop = {"instant", "dteday", "casual", "registered", "cnt"}
    fcols = [i for i, h in enumerate(hdr) if h not in drop]
    ycol = hdr.index("cnt")
    data = np.array([[float(r[i]) for i in fcols] + [float(r[ycol])] for r in rows[1:]], np.float32)
    return _pack("bike_sharing", "regr", *_split(data[:, :-1], data[:, -1]))


def _load_beijing(d):
    import csv
    p = _find(d / "beijing_pm25", "*.csv")
    rows = list(csv.reader(open(p)))
    hdr = rows[0]
    yi = hdr.index("pm2.5")
    cbwd = hdr.index("cbwd")
    num = [i for i, h in enumerate(hdr) if h not in ("No", "pm2.5", "cbwd", "year")]
    X, y, wind = [], [], []
    for r in rows[1:]:
        if r[yi] == "NA" or r[yi] == "":
            continue
        X.append([float(r[i]) for i in num]); y.append(float(r[yi])); wind.append(r[cbwd])
    X = np.hstack([np.array(X, np.float32), _onehot(np.array(wind))])
    return _pack("beijing_pm25", "regr", *_split(X, np.array(y, np.float32)))


def _load_landsat(d):
    tr = np.loadtxt(_find(d / "landsat", "sat.trn"))
    te = np.loadtxt(_find(d / "landsat", "sat.tst"))
    ymap = {v: i for i, v in enumerate(sorted(set(tr[:, -1]).union(te[:, -1])))}
    ytr = np.array([ymap[v] for v in tr[:, -1]])
    yte = np.array([ymap[v] for v in te[:, -1]])
    return _pack("landsat", "classif", tr[:, :-1], ytr, te[:, :-1], yte)


def _load_covertype(d):
    p = _find(d / "covertype", "covtype.data*")
    a = np.genfromtxt(p, delimiter=",")
    rng = np.random.default_rng(0)
    a = a[rng.permutation(len(a))[:40000]]        # subsample the 581k
    return _pack("covertype", "classif", *_split(a[:, :-1], (a[:, -1] - 1).astype(int)))


def _load_splice(d):
    p = _find(d / "splice_dna", "splice.data")
    cls, seqs = [], []
    for line in open(p):
        parts = [s.strip() for s in line.split(",")]
        if len(parts) < 3:
            continue
        cls.append(parts[0]); seqs.append(parts[2])
    base = {"A": 0, "C": 1, "G": 2, "T": 3}
    L = min(len(s) for s in seqs)
    X = np.zeros((len(seqs), L * 4), np.float32)
    for i, s in enumerate(seqs):
        for j, ch in enumerate(s[:L]):
            X[i, j * 4 + base.get(ch, 0)] = 1.0
    ymap = {v: i for i, v in enumerate(sorted(set(cls)))}
    y = np.array([ymap[c] for c in cls])
    return _pack("splice_dna", "classif", *_split(X, y))


def _load_ag_news(d):
    import csv
    tr = _find(d / "ag_news", "train.csv")
    rows = list(csv.reader(open(tr)))
    y = np.array([int(r[0]) - 1 for r in rows])
    txt = [r[1] + " " + r[2] for r in rows]
    X = _text_hash(txt, 512)
    return _pack("ag_news", "classif", *_split(X, y))


def _load_gas_sensor(d):
    root = d / "gas_sensor"
    feats, labs = [], []
    for bf in sorted(root.rglob("batch*.dat")):
        for line in open(bf):
            parts = line.split()
            if not parts:
                continue
            labs.append(int(float(parts[0].split(";")[0])) - 1)
            feats.append([float(p.split(":")[1]) for p in parts[1:]])
    n = min(len(f) for f in feats)
    X = np.array([f[:n] for f in feats], np.float32)
    return _pack("gas_sensor", "classif", *_split(X, np.array(labs)))


def _load_nsl_kdd(d):
    import csv
    p = _find(d / "nsl_kdd", "KDDTrain+.txt")
    rows = [r for r in csv.reader(open(p)) if len(r) > 40]
    cat = [1, 2, 3]                      # protocol, service, flag
    num = [i for i in range(41) if i not in cat]
    Xnum = np.array([[float(r[i]) for i in num] for r in rows], np.float32)
    Xcat = np.hstack([_onehot(np.array([r[i] for r in rows])) for i in cat])
    y = np.array([0 if r[41] == "normal" else 1 for r in rows])   # binary normal/attack
    X = np.hstack([Xnum, Xcat])
    return _pack("nsl_kdd", "classif", *_split(X, y))


REGISTRY = {
    "fashion_mnist": lambda d: _load_parent("fashion_mnist"),
    "uci_har": lambda d: _load_parent("uci_har"),
    "superconductivity": lambda d: _load_parent("superconductivity"),
    "sarcos": lambda d: _load_parent("sarcos"),
    "airfoil": _load_airfoil,
    "appliances_energy": _load_appliances,
    "parkinsons": _load_parkinsons,
    "online_news": _load_online_news,
    "bike_sharing": _load_bike,
    "beijing_pm25": _load_beijing,
    "landsat": _load_landsat,
    "covertype": _load_covertype,
    "splice_dna": _load_splice,
    "ag_news": _load_ag_news,
    "gas_sensor": _load_gas_sensor,
    "nsl_kdd": _load_nsl_kdd,
}


def load(name: str, data_dir: Path) -> Task:
    return REGISTRY[name](Path(data_dir))


def available(data_dir: Path, verbose=True):
    """Return names that load cleanly; log why others are skipped."""
    ok = []
    for name in TASK_ORDER:
        try:
            t = load(name, data_dir)
            assert t.x_train.shape[0] > 10 and t.x_train.shape[1] > 0
            ok.append(name)
            if verbose:
                print(f"  OK   {name:18s} d_in={t.d_in:<5d} d_out={t.d_out:<3d} "
                      f"kind={t.kind} n_train={t.x_train.shape[0]}")
        except Exception as e:
            if verbose:
                print(f"  skip {name:18s} ({type(e).__name__}: {str(e)[:60]})")
    return ok
