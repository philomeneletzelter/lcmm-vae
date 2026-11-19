"""
Simulated cohorts: G=3 latent-class trajectories in an L=3 latent space, decoded by a frozen
random MLP to K markers (K_SIG informative), observed through an MCAR or MAR visit
process. Each cohort holds data.csv, mask.csv, label.csv, truth.csv and truth_dropped.csv;
experiments/simulation/run.py writes the cohorts it needs.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src.data_io import write_data

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "simulation"

L, G = 3, 3                     # latent width, classes
T = 100                         # visits on the shared grid
PI = np.array([0.5, 0.3, 0.2])
K, K_SIG = 50, 10               # markers, of which informative
TAU = (0.35, 0.50, 0.65)        # class centre of the rise and of the transient
WIDTH = 0.20                    # width of the transient
# scale of the class deviations around the mean curve
AMP = 4.346821
D0 = np.diag([0.5, 1.5])        # D_l of the random intercept and slope, R=2
NOISE_SD = 0.5                  # noise-to-signal sd on the informative markers
GP_SCALE = 0.15                 # RBF length-scale of the noise state and the nuisance
RETENTION = 0.20                # target fraction of the grid observed
BETA = 2.0                      # MAR-signed: logit slope on the last observed driving marker
MAX_GAP = 20                    # longest gap between visits, in grid steps
MECHS = ('mcar', 'mar')
# fixed offsets added to the cohort seed, one independent random stream each
NOISE_STATE_SEED, NOISE_MLP_SEED, NUISANCE_SEED = 4321, 1236, 1234
VISIT_SEED, FIRST_VISIT_SEED = 900_000, 900_500

t = np.linspace(0, 1, T)
XR = np.stack([np.ones_like(t), t], 1)                                  # (T, 2)


def _class_trajectories():
    sig = lambda k, tau: 1 / (1 + np.exp(-k * (t - tau)))
    mu0 = np.zeros((G, L, T))
    for g, tau in enumerate(TAU):
        mu0[g, 0] = sig(18, tau)                                        # timing of a rise
        mu0[g, 1] = np.exp(-((t - tau) / WIDTH) ** 2)                   # shape, a transient
        mu0[g, 2] = 1 - 0.8 * sig(6, 0.5) - 0.3 * (g - 1) * t           # rate of a decline
    mubar = np.einsum("g,glt->lt", PI, mu0)
    return mubar + AMP * (mu0 - mubar)


MU = _class_trajectories()                                                    # (G, L, T)


def _frozen(n_in, n_out, seed, hidden=32):
    """A random tanh MLP, sampled once and never trained."""
    torch.manual_seed(seed)
    f = nn.Sequential(nn.Linear(n_in, hidden), nn.Tanh(), nn.Linear(hidden, n_out)).eval()
    for q in f.parameters():
        q.requires_grad_(False)
    return f


def _gp_draws(rng, N, n):
    Kg = np.exp(-0.5 * np.subtract.outer(t, t) ** 2 / GP_SCALE ** 2)
    Lg = np.linalg.cholesky(Kg + 1e-6 * np.eye(T))
    return np.einsum("ab,nkb->nak", Lg, rng.standard_normal((N, n, T)))  # (N, T, n)


def _zscore(x):
    return (x - x.mean(0)) / (x.std(0) + 1e-8)


def class_sizes(N):
    n = np.round(PI * N).astype(int)
    n[0] += N - n.sum()
    return n


def simulate(seed, N=500):
    """
    One cohort on the full grid, before any visit is dropped.

    Returns:
    - c (ndarray): class of each subject. [N]
    - Y (ndarray): markers. [N, T, K]
    - kstar (int): the MAR driving marker, the informative marker with the largest
      between-class variance of the noiseless class curves.
    """
    rng = np.random.default_rng(seed)
    n = class_sizes(N)
    c = np.repeat(np.arange(G), n)
    c = c[rng.permutation(N)]
    Z = np.transpose(MU[c], (0, 2, 1))                                  # (N, T, L)
    A = rng.standard_normal((N, L, 2)) @ np.linalg.cholesky(D0).T
    Z = Z + np.einsum("tr,nlr->ntl", XR, A) + rng.standard_normal((N, T, L))

    f = _frozen(L, K_SIG, seed)
    with torch.no_grad():
        sg = f(torch.tensor(Z.reshape(-1, L), dtype=torch.float32)).numpy()
        cur = f(torch.tensor(np.transpose(MU, (0, 2, 1)).reshape(-1, L), dtype=torch.float32)).numpy()
    m1, s1 = sg.mean(0), sg.std(0) + 1e-8
    sg, cur = (sg - m1) / s1, (cur - m1) / s1

    # observation noise: a rank-3 smooth state per subject through a second frozen MLP
    V = _gp_draws(np.random.default_rng(NOISE_STATE_SEED + seed), N, 3)
    with torch.no_grad():
        E = _frozen(3, K_SIG, NOISE_MLP_SEED + seed)(
            torch.tensor(V.reshape(-1, 3), dtype=torch.float32)).numpy()
    sg = sg + NOISE_SD * (E - E.mean(0)) / (E.std(0) + 1e-8)
    m2, s2 = sg.mean(0), sg.std(0) + 1e-8
    sg, cur = (sg - m2) / s2, (cur - m2) / s2

    U = _gp_draws(np.random.default_rng(NUISANCE_SEED + seed), N, K - K_SIG).reshape(-1, K - K_SIG)
    Y = np.concatenate([sg, _zscore(U)], 1).reshape(N, T, K)

    pi = n / N
    truth = cur.reshape(G, T, K_SIG)
    cbar = np.einsum("g,gtk->tk", pi, truth)
    kstar = int(np.argmax(np.einsum("g,gtk->k", pi, (truth - cbar) ** 2)))
    return c, Y, kstar


def visit_mask(s, mech, alpha, seed):
    """
    Visit mask [N, T] from a first visit in 0..4; each later point is a visit with logit
    alpha (MCAR) or alpha + BETA times the driving marker s [N, T] at the last visit (MAR).
    """
    N = len(s)
    U = np.random.default_rng(VISIT_SEED + seed).random((N, T))
    j0 = np.random.default_rng(FIRST_VISIT_SEED + seed).integers(0, 5, N)
    M = np.zeros((N, T), bool)
    M[np.arange(N), j0] = True
    last = j0.copy()
    for j in range(1, T):
        eta = alpha + (BETA * s[np.arange(N), last] if mech == 'mar' else 0.0)
        hit = (j > j0) & ((U[:, j] < 1 / (1 + np.exp(-eta))) | (j - last >= MAX_GAP))
        M[hit, j] = True
        last[hit] = j
    return M


def calibrate(s, mech, seed):
    """Visit-logit intercept alpha giving a RETENTION observed fraction (bisection)."""
    lo, hi = -10.0, 5.0
    for _ in range(40):
        a = (lo + hi) / 2
        r = visit_mask(s, mech, a, seed).mean()
        if abs(r - RETENTION) < 0.003:
            return a
        lo, hi = (a, hi) if r < RETENTION else (lo, a)
    raise RuntimeError(f"retention calibration failed for {mech}, seed {seed}")


def cohort(data_dir, mech, seed, N=500):
    """Path of the cohort under data_dir, written if missing."""
    d = Path(data_dir).resolve() / f"{mech}_N{N}_s{seed}"
    if (d / "truth_dropped.csv").exists():
        return d
    c, Y, kstar = simulate(seed, N)
    s = Y[:, :, kstar]
    M = visit_mask(s, mech, calibrate(s, mech, seed), seed)
    keep = M.reshape(-1)
    Yo = Y.reshape(N * T, K)[keep]
    grid = pd.DataFrame({"unit": np.repeat(np.arange(N), T), "t": np.tile(t, N)})
    lab = grid[keep]
    write_data(str(d), Yo, np.ones_like(Yo, np.int8), lab)
    pd.DataFrame({"unit": np.arange(N), "category": c}).to_csv(d / "truth.csv", index=False)
    drop = pd.concat([grid.loc[~keep, ["unit", "t"]].reset_index(drop=True),
                      pd.DataFrame(Y.reshape(N * T, K)[~keep])], axis=1)
    drop.to_csv(d / "truth_dropped.csv", index=False)
    return d
