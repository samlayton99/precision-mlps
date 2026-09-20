"""Fetch the no-auth datasets for the 20-task sweep into <repo>/data/<name>/.

RUN YOURSELF (agent sandbox can't download):
    uv run python experiments/expF04_qi_init_real_data/all20/fetch_data.py

Best-effort URLs -- UCI occasionally renames zip slugs and github mirrors move. Each
dataset is isolated: a failure is reported, not fatal. Edit the URLS block for anything
that 404s. The sweep runs on whatever parses (datasets.available()), so partial success
is fine.

NOT fetched here (need auth or heavy deps -- get them yourself if you want them):
    mit_bih ECG   Kaggle auth
    sdss redshift Kaggle / astroquery
    qm9           rdkit descriptors
    fsdd audio    many wav files + MFCC
"""
from __future__ import annotations

import gzip
import io
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA = REPO_ROOT / "data"

# name -> (url, type)  where type in {zip, gz->target, file}
URLS = {
    "airfoil": ("https://archive.ics.uci.edu/static/public/291/airfoil+self+noise.zip", "zip"),
    "appliances_energy": ("https://archive.ics.uci.edu/static/public/374/appliances+energy+prediction.zip", "zip"),
    "parkinsons": ("https://archive.ics.uci.edu/static/public/189/parkinsons+telemonitoring.zip", "zip"),
    "online_news": ("https://archive.ics.uci.edu/static/public/332/online+news+popularity.zip", "zip"),
    "bike_sharing": ("https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip", "zip"),
    "beijing_pm25": ("https://archive.ics.uci.edu/static/public/381/beijing+pm2+5+data.zip", "zip"),
    "landsat": ("https://archive.ics.uci.edu/static/public/146/statlog+landsat+satellite.zip", "zip"),
    "covertype": ("https://archive.ics.uci.edu/static/public/31/covertype.zip", "zip"),
    "splice_dna": ("https://archive.ics.uci.edu/static/public/69/molecular+biology+splice+junction+gene+sequences.zip", "zip"),
    "gas_sensor": ("https://archive.ics.uci.edu/static/public/224/gas+sensor+array+drift+dataset.zip", "zip"),
    "ag_news": ("https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/train.csv",
                "file:train.csv"),
    "nsl_kdd": ("https://raw.githubusercontent.com/jmnwong/NSL-KDD-Dataset/master/KDDTrain%2B.txt",
                "file:KDDTrain+.txt"),
}


def _get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 expF04-all20"})
    with urllib.request.urlopen(req, timeout=180) as r:
        return r.read()


def _extract_zip(data, out):
    out.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        z.extractall(out)
    for inner in list(out.rglob("*.zip")):      # nested zips (UCI habit)
        with zipfile.ZipFile(inner) as z:
            z.extractall(inner.parent)
    for gz in list(out.rglob("*.gz")):          # e.g. covtype.data.gz
        tgt = gz.with_suffix("")
        if not tgt.exists():
            with gzip.open(gz, "rb") as f_in, open(tgt, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)


def fetch(name, url, typ):
    root = DATA / name
    if root.exists() and any(root.rglob("*")):
        print(f"[{name}] skip (present)")
        return
    root.mkdir(parents=True, exist_ok=True)
    if typ == "zip":
        _extract_zip(_get(url), root)
    elif typ.startswith("file:"):
        (root / typ.split(":", 1)[1]).write_bytes(_get(url))
    print(f"[{name}] fetched -> {root}")


def main():
    print(f"Downloading into {DATA}\n")
    ok, fail = [], []
    for name, (url, typ) in URLS.items():
        try:
            fetch(name, url, typ)
            ok.append(name)
        except Exception as e:
            print(f"[{name}] FAILED: {type(e).__name__}: {str(e)[:80]}")
            fail.append(name)
    print(f"\nok={ok}\nfailed={fail}")
    if fail:
        print("Edit URLS at the top for any failed dataset and re-run; the sweep uses "
              "whatever parses.")
        sys.exit(1)


if __name__ == "__main__":
    main()
