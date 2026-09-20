"""Download the four expF04 datasets into <repo>/data/. RUN THIS YOURSELF (the agent
sandbox can't reach the network):

    uv run python experiments/expF04_qi_init_real_data/fetch_data.py

Plain urllib, no extra deps. Idempotent (skips files already present). If a URL has
rotted, edit the URLS block below -- each dataset is independent, failures are isolated
so a partial run still makes progress. Total ~150 MB.
"""
from __future__ import annotations

import io
import sys
import urllib.request
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA = REPO_ROOT / "data"

FMNIST_BASE = "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion"
FMNIST_FILES = [
    "train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz",
]
HAR_ZIP = "https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip"
SUPER_ZIP = "https://archive.ics.uci.edu/static/public/464/superconductivty+data.zip"
SARCOS_FILES = {
    "sarcos_inv.mat": "http://www.gaussianprocess.org/gpml/data/sarcos_inv.mat",
    "sarcos_inv_test.mat": "http://www.gaussianprocess.org/gpml/data/sarcos_inv_test.mat",
}


def _get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 expF04-fetch"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def _download_to(url: str, dest: Path):
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  skip (exists) {dest.name}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  GET {url}")
    dest.write_bytes(_get(url))
    print(f"      -> {dest} ({dest.stat().st_size/1e6:.1f} MB)")


def _extract_zip(data: bytes, out: Path, recurse=True):
    out.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        z.extractall(out)
    if recurse:  # UCI HAR ships a zip inside the zip
        for inner in list(out.rglob("*.zip")):
            with zipfile.ZipFile(inner) as z:
                z.extractall(inner.parent)


def fetch_fashion_mnist():
    print("[fashion_mnist]")
    for f in FMNIST_FILES:
        _download_to(f"{FMNIST_BASE}/{f}", DATA / "fashion_mnist" / f)


def fetch_uci_har():
    print("[uci_har]")
    root = DATA / "uci_har"
    if list(root.rglob("X_train.txt")):
        print("  skip (already extracted)")
        return
    _extract_zip(_get(HAR_ZIP), root)
    print(f"  extracted -> {root}")


def fetch_superconductivity():
    print("[superconductivity]")
    root = DATA / "superconductivity"
    if list(root.rglob("train.csv")):
        print("  skip (already extracted)")
        return
    _extract_zip(_get(SUPER_ZIP), root)
    print(f"  extracted -> {root}")


def fetch_sarcos():
    print("[sarcos]")
    for name, url in SARCOS_FILES.items():
        _download_to(url, DATA / "sarcos" / name)


def main():
    print(f"Downloading into {DATA}\n")
    ok, fail = [], []
    for fn in (fetch_fashion_mnist, fetch_uci_har, fetch_superconductivity, fetch_sarcos):
        try:
            fn()
            ok.append(fn.__name__)
        except Exception as e:
            print(f"  FAILED: {type(e).__name__}: {e}")
            fail.append(fn.__name__)
        print()
    print(f"done. ok={ok} failed={fail}")
    if fail:
        print("Edit the URLS at the top of this file for any failed dataset and re-run.")
        sys.exit(1)


if __name__ == "__main__":
    main()
