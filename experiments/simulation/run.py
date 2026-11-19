"""
LCMM-VAE, LCMM-VAE (R=0) and LCMM-VAE (G=1) on the simulated cohorts: ARI and prior-path
MSE at the held-out subjects' dropped visits.

usage: python -m experiments.simulation.run [--N 500] [--latent-dim L] [--out results]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import adjusted_rand_score

from data_loader import simulation as sim
from experiments import utils
from src.utils import design_matrix

DATASET = "simulation"


def classes(cohort_dir, units):
    """True class of the units [N]."""
    truth = pd.read_csv(Path(cohort_dir) / "truth.csv").set_index("unit")
    return truth.loc[units, "category"].to_numpy()


def dropped_design(config, design, units):
    """
    Design rows at the dropped visits of units [N] (in subject-code order).

    Returns:
    - rows (DataFrame): unit and t of each dropped visit. [M]
    - x (Tensor): design rows. [M, P]
    - sid (Tensor): subject code of each row. [M]
    """
    rows = pd.read_csv(Path(config['dataset']['data_path']) / "truth_dropped.csv",
                       usecols=["unit", "t"])
    rows = rows[rows['unit'].isin(units)].reset_index(drop=True)
    x, _ = design_matrix(rows, config['dataset']['covariates'], design)
    sid = pd.Index(units).get_indexer(rows['unit'])
    return rows[["unit", "t"]], torch.tensor(x, dtype=torch.float32), torch.tensor(sid)


def recon_mse(config, scaler, rows, Y_hat):
    """Standardised MSE of Y_hat [M, K] at the dropped visits, all and informative markers."""
    truth = pd.read_csv(Path(config['dataset']['data_path']) / "truth_dropped.csv")
    Y = rows.merge(truth, on=["unit", "t"], how="left").drop(columns=["unit", "t"]).to_numpy()
    sq_error = (np.asarray(Y_hat) - (Y - scaler['mean']) / scaler['std']) ** 2
    return float(sq_error.mean()), float(sq_error[:, :sim.K_SIG].mean())


def main(args):
    for seed in args.seeds:
        for mech in args.mech:
            cohort_dir = sim.cohort(args.data, mech, seed, args.N)

            def load(method):
                return utils.load_config("simulation", seed, cohort_dir,
                                         R=0 if method == utils.R0 else 2,
                                         latent_dim=args.latent_dim)

            for name, result in utils.fits(args.methods, load, seed, None):
                units = result['loader'].dataset.subject_ids
                rows, x, sid = dropped_design(result['config'], result['design'], units)
                y_hat = utils.reconstruct(result, x, sid).numpy()
                mse, mse_inf = recon_mse(result['config'], result['scaler'], rows, y_hat)
                row = dict(method=name, mech=mech, N=args.N, seed=seed,
                           epochs=result['epochs'], secs=result['secs'])
                ari = float(adjusted_rand_score(classes(cohort_dir, units), result['labels']))
                utils.append_row(args.out, DATASET, "clustering", dict(row, ari=ari))
                utils.append_row(args.out, DATASET, "reconstruction",
                                 dict(row, mse=mse, mse_informative=mse_inf))
                print(f"{name:16} {mech:4} s{seed:<2} ARI {ari:.3f} MSE {mse:.3f} "
                      f"epochs {result['epochs']} ({result['secs']:.0f}s)", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LCMM-VAE on the simulated cohorts.")
    parser.add_argument('--N', type=int, default=500, help="subjects per cohort")
    parser.add_argument('--mech', nargs='+', choices=sim.MECHS, default=list(sim.MECHS),
                        help="missingness mechanisms")
    parser.add_argument('--methods', nargs='+', choices=utils.METHODS, default=list(utils.METHODS),
                        help="methods to fit")
    parser.add_argument('--seeds', nargs='+', type=int, default=list(range(30)),
                        help="cohort and training seeds")
    parser.add_argument('--latent-dim', type=int, default=None, help="overrides model.latent_dim")
    parser.add_argument('--data', default=str(sim.DATA),
                        help="cohort directory, written when missing")
    parser.add_argument('--out', default=utils.RESULTS, help="output root: result rows")
    main(parser.parse_args())
