"""
Gait conditions: missingness mechanism, rate, and the held-out pair of persons of each seed.
A channel-missing walk loses Binomial(K - 1, rate / 100) whole markers (at least one is
kept), so about (K - 1)/K of rate% of cells are hidden; the realised share is recorded per run. Masks and pairs are deterministic in (mech, rate, seed).

usage:
    with split("channel", 33, seed=4) as (train_dir, test_dir):
        ...  # train on train_dir, evaluate on test_dir
"""
import itertools
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from src.data_io import read_data, write_data

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "data" / "gait"

# mech -> rates, the grid every gait runner and table iterates
CONDITIONS = {"complete": (0,), "mcar": (20, 40, 60), "channel": (17, 33, 50)}
# digit of each mechanism in seed_for; part of every mask's seed
MECH_ID = {"mcar": 0, "channel": 3}

# seed s holds out PAIRS[s]: the 45 pairs of the 10 persons, shuffled once
PAIRS = list(itertools.combinations(range(10), 2))
np.random.default_rng(0).shuffle(PAIRS)


# each mechanism returns a (S*T, K) mask

def m_mcar(Y, S, T, K, rate, rng):
    """Cells dropped independently."""
    return (rng.random(Y.shape) >= rate / 100).astype(np.float32)


def m_channel(Y, S, T, K, rate, rng):
    """Whole markers absent from a walk; every marker stays observed in at least one walk."""
    keep = np.ones((S, K), bool)
    n_drop = rng.binomial(K - 1, rate / 100, size=S)    # K-1: a walk never loses all K
    for i in range(S):
        if n_drop[i]:
            keep[i, rng.choice(K, n_drop[i], replace=False)] = False
    for k in range(K):
        if not keep[:, k].any():
            keep[rng.integers(S), k] = True
    return np.repeat(keep[:, None, :], T, axis=1).reshape(S * T, K).astype(np.float32)


MECHANISMS = {"mcar": m_mcar, "channel": m_channel}


def seed_for(mech, rate, seed):
    """Collision-free seed per condition: channel, 33, seed 4 -> 7_333_004."""
    return 7_000_000 + MECH_ID[mech] * 100_000 + rate * 1_000 + seed


def draw(mech, rate, seed):
    """(Y, mask, label_frame). rate 0 is the complete panel, mask all ones."""
    if rate and mech not in MECHANISMS:
        raise KeyError(f"unknown mechanism {mech!r}; have {sorted(MECHANISMS)}")
    Y, _, lab = read_data(str(SRC))
    S = lab.unit.nunique()
    T = len(Y) // S
    K = Y.shape[1]
    if rate == 0:
        return Y, np.ones_like(Y, np.float32), lab
    rng = np.random.default_rng(seed_for(mech, rate, seed))
    M = MECHANISMS[mech](Y, S, T, K, rate, rng).reshape(len(Y), K)
    return Y, M, lab


def held_out_pair(seed):
    """The two persons (label.csv `person`) held out at this seed."""
    return PAIRS[seed]


def write_split(root, mech, rate, seed):
    """Draw one condition and write it as root/train (eight persons) and root/test (the
    held-out pair), row order kept."""
    Y, M, lab = draw(mech, rate, seed)
    test = lab.person.isin(held_out_pair(seed)).to_numpy()
    root = Path(root)
    return tuple(Path(write_data(str(root / name), Y[rows], M[rows],
                                  lab[rows].reset_index(drop=True)))
                 for name, rows in (("train", ~test), ("test", test)))


@contextmanager
def split(mech, rate, seed, keep=False):
    """write_split into a temp directory; yields (train_dir, test_dir). Removed on exit
    unless `keep`."""
    tmp = Path(tempfile.mkdtemp(prefix=f"gait_{mech}{rate}_s{seed}_"))
    try:
        yield write_split(tmp, mech, rate, seed)
    finally:
        if not keep:
            shutil.rmtree(tmp, ignore_errors=True)
