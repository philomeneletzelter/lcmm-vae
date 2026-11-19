"""
The LCMM-VAE rows of the paper's tables: mean ± sd (n) per cell, and median seconds per fit
where the paper reports time.

usage: python -m experiments.tables [--out results] [--seeds 0 1 ...]
"""
import argparse
import os

import pandas as pd

from data_loader import simulation as sim
from experiments import utils
from experiments.gait.missingness import CONDITIONS


def rows(index):
    return [m for m in utils.METHODS if m in index] + sorted(set(index) - set(utils.METHODS))


def fmt(mu, sd, n):
    text = lambda x: x.map(lambda v: f"{v:.3f}")
    return text(mu) + " ± " + text(sd) + " (" + n.astype(str) + ")"


def stats(groups):
    return groups.mean(), groups.std(), groups.count()


def simulation(results, col, N):
    """method x mechanism, at cohort size N; and median secs."""
    results = results[results.N == N]
    table = fmt(*stats(results.groupby(["method", "mech"])[col])).unstack("mech")
    table = table.reindex(index=rows(table.index),
                          columns=[m for m in sim.MECHS if m in table])
    secs = results.groupby(["method", "mech"]).secs.median().unstack("mech").round(1)
    return table, secs.reindex(index=table.index, columns=table.columns)


def gait_by_mechanism(results, col):
    """Per seed, the mean over a mechanism's rates (NaN unless every rate scored)."""
    out = {}
    for mech in CONDITIONS:
        sub = results[results.mech == mech]
        if len(sub):
            per_seed = sub.groupby(["method", "seed"])[col].agg(
                lambda x: x.mean() if x.notna().all() else float("nan"))
            out[mech] = fmt(*stats(per_seed.groupby("method")))
    table = pd.DataFrame(out)
    return table.reindex(rows(table.index))


def gait_by_rate(results, col):
    results = results[results.mech != "complete"]
    table = fmt(*stats(results.groupby(["method", "mech", "rate"])[col]))
    table = table.unstack(["mech", "rate"])
    cols = [(m, r) for m in CONDITIONS for r in CONDITIONS[m] if (m, r) in table]
    return table.reindex(index=rows(table.index), columns=cols)


def physionet(results):
    groups = results.groupby("method")
    table = pd.DataFrame({"AUROC": fmt(*stats(groups.auroc)), "MSE": fmt(*stats(groups.mse)),
                          "secs (median)": groups.secs.median().round(0)})
    return table.reindex(rows(table.index))


def show(title, table):
    if len(table):
        print(f"==== {title} ====\n{table.fillna('-').to_string()}\n")


def main(args):
    pd.set_option("display.width", 200)

    def load(dataset, task):
        path = os.path.join(args.out, dataset, f"{task}.csv")
        if not os.path.exists(path):
            return None
        results = pd.read_csv(path)
        return results if args.seeds is None else results[results.seed.isin(args.seeds)]

    ari, mse = load("simulation", "clustering"), load("simulation", "reconstruction")
    if ari is not None:
        for N in sorted(ari.N.unique()):
            table, secs = simulation(ari, "ari", N)
            show(f"Simulation N={N}: ARI", table.join(secs, rsuffix=" secs (median)"))
            show(f"Simulation N={N}: MSE, {sim.K_SIG} informative markers",
                 simulation(mse, "mse_informative", N)[0])
            show(f"Simulation N={N}: MSE, all {sim.K} markers", simulation(mse, "mse", N)[0])

    ari, mse = load("gait", "clustering"), load("gait", "reconstruction")
    if ari is not None:
        show("Gait: ARI per mechanism (mean over rates)", gait_by_mechanism(ari, "ari"))
        show("Gait: ARI per rate", gait_by_rate(ari, "ari"))
    if mse is not None:
        show("Gait: MSE per mechanism (mean over rates)", gait_by_mechanism(mse, "mse"))
        show("Gait: MSE per rate", gait_by_rate(mse, "mse"))

    results = load("physionet", "mortality")
    if results is not None:
        show("PhysioNet x_test: mortality AUROC and MSE", physionet(results))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LCMM-VAE rows of the paper's tables.")
    parser.add_argument("--out", default=utils.RESULTS, help="output root of the runners")
    parser.add_argument("--seeds", nargs="+", type=int, default=None,
                        help="restrict to these seeds")
    main(parser.parse_args())
