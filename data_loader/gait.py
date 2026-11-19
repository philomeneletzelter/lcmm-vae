"""
Multivariate Gait Data (Helwig and Hsiao-Wecksler, 2016; UCI id 760): 300 walks
(10 subjects x 3 brace conditions x 10 replications) x 101 timepoints x 6 joint angles,
written to data/gait/.
label.csv: unit, t = time / 100, person, category (0 none, 1 knee, 2 ankle brace).

usage:
    python -m data_loader.gait [--out DIR]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo

from src.data_io import write_data

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "gait"
UCI_ID = 760

N_TIME = 101
# left ankle, knee, hip, then right; braces were worn on the right leg
MARKERS = [(leg, joint) for leg in (1, 2) for joint in (1, 2, 3)]


def build():
    """(Y, label_frame) from the UCI long table. Y is (30300, 6), rows unit-major then time."""
    g = fetch_ucirepo(id=UCI_ID).data.features

    keys = ["subject", "condition", "replication"]
    units = g[keys].drop_duplicates().sort_values(keys).reset_index(drop=True)
    units["unit"] = np.arange(len(units))
    g = g.merge(units, on=keys)

    wide = g.pivot_table(index=["unit", "time"], columns=["leg", "joint"], values="angle")
    wide = wide.reindex(columns=pd.MultiIndex.from_tuples(MARKERS)).sort_index()
    Y = wide.to_numpy(np.float64)

    idx = wide.index.to_frame(index=False)
    lab = pd.DataFrame({
        "unit": idx["unit"].astype(int),
        "t": idx["time"].to_numpy() / 100.0,
    })
    meta = units.set_index("unit")
    lab["person"] = meta.loc[lab.unit, "subject"].to_numpy() - 1
    lab["category"] = meta.loc[lab.unit, "condition"].to_numpy() - 1
    return Y, lab


def write(d, Y, lab):
    """Write the four files, mask all ones."""
    # round as float32, the model's dtype;
    Y32 = Y.astype(np.float32).astype(np.float64)
    # mask is all ones because missingness is applied later on 
    write_data(str(d), Y32, np.ones_like(Y, dtype=np.int8), lab, float_format="%.6f")
    return d


def main():
    ap = argparse.ArgumentParser(description="Download and format the UCI gait data.")
    ap.add_argument("--out", default=str(OUT), help="output directory")
    args = ap.parse_args()
    Y, lab = build()
    write(Path(args.out), Y, lab)
    print(f"{lab.unit.nunique()} walks, {Y.shape[1]} markers -> {args.out}")


if __name__ == "__main__":
    main()
