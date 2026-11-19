"""
PhysioNet/CinC 2012 with the extraction and train/val/test split of GP-VAE (Fortuin et al.,
2020): downloads data/physionet/physionet.npz and writes data/physionet/{train,val,test}/.
label.csv: unit, t = step / 47, category = in-hospital mortality.

usage:
    python -m data_loader.physionet [--data DIR]
"""
import argparse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_io import write_data

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "physionet"
URL = "https://www.dropbox.com/s/651d86winb4cy9n/physionet.npz?dl=1"
SPLITS = ("train", "val", "test")


def fetch(data_dir):
    """data_dir/physionet.npz, downloaded if missing."""
    npz = Path(data_dir).expanduser() / "physionet.npz"
    if not npz.exists():
        npz.parent.mkdir(parents=True, exist_ok=True)
        print(f"  downloading {URL} to {npz}")
        tmp = npz.with_suffix(".part")
        urllib.request.urlretrieve(URL, tmp)
        tmp.rename(npz)
    return np.load(npz)


def build(z, split):
    """(Y, mask, label_frame) for one split, rows ordered unit-major then step."""
    x = z[f"x_{split}_miss"]
    m = z[f"m_{split}_miss"]
    y = z[f"y_{split}"]
    N, T, K = x.shape

    obs = (m == 0).astype(np.float32)          # npz m == 1 means missing
    Y = x.reshape(N * T, K).astype(np.float32)
    M = obs.reshape(N * T, K)
    lab = pd.DataFrame({
        "unit": np.repeat(np.arange(N), T),
        "t": np.tile(np.arange(T) / (T - 1), N),
        "category": np.repeat(y.astype(int), T),
    })
    return Y, M, lab


def main():
    ap = argparse.ArgumentParser(description="Download and format PhysioNet-2012 (GP-VAE split).")
    ap.add_argument("--data", default=str(DATA), help="output directory")
    args = ap.parse_args()
    z = fetch(args.data)
    for split in SPLITS:
        Y, M, lab = build(z, split)
        write_data(str(Path(args.data) / split), Y, M, lab)
        print(f"{split}: {lab.unit.nunique()} stays -> {Path(args.data) / split}")


if __name__ == "__main__":
    main()
