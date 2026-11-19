"""
Gait: seed s holds out the pair of persons missingness.held_out_pair(s); LCMM-VAE and
LCMM-VAE (G=1) fit on the other eight persons' walks. ARI against the brace condition of
the held-out walks and, when cells are hidden, prior-path MSE at those cells, on markers
standardised by the training walks' observed cells.

usage: python -m experiments.gait.run [--mech complete mcar channel] [--out results]
"""
import argparse

import numpy as np
from sklearn.metrics import adjusted_rand_score

from experiments import utils
from experiments.gait.missingness import CONDITIONS, split
from src.data_io import marker_scaler, read_data, standardise

DATASET = "gait"


def panel(test_dir, train_dir):
    """
    Held-out walks, unit-major and sorted in t as written by missingness.split.

    Returns:
    - p (dict): units [N], truth [N], Y (complete values on the training scale) and
      M [N, T, K], scaler, realised_rate (% hidden cells, train and test).
    """
    Y_tr, M_tr, _ = read_data(str(train_dir))
    Y, M, lab = read_data(str(test_dir))
    units = lab["unit"].unique()
    N, K = len(units), Y.shape[1]
    T = len(Y) // N
    assert (lab["unit"].to_numpy().reshape(N, T) == units[:, None]).all()
    scaler = marker_scaler(Y_tr, M_tr)
    return dict(units=units, truth=lab.groupby("unit")["category"].first().loc[units].values,
                Y=standardise(Y, np.ones_like(Y), scaler, np.nan).reshape(N, T, K),
                M=M.reshape(N, T, K), scaler=scaler,
                realised_rate=100 * (1 - np.concatenate([M_tr.ravel(), M.ravel()]).mean()))


def main(args):
    for mech in args.mech:
        for rate in CONDITIONS[mech]:
            for seed in args.seeds:
                with split(mech, rate, seed) as (train_dir, test_dir):
                    test = panel(test_dir, train_dir)

                    def load(method):
                        return utils.load_config("gait", seed, train_dir)

                    for name, result in utils.fits(args.methods, load, seed, test_dir):
                        assert np.array_equal(result["subject_ids"], test["units"])
                        raw = utils.to_raw(utils.reconstruct(result).numpy()
                                           .reshape(test["Y"].shape), result["scaler"])
                        Y_hat = standardise(raw, np.ones_like(raw), test["scaler"], np.nan)
                        row = dict(method=name, mech=mech, rate=rate, seed=seed,
                                   realised_rate=test["realised_rate"],
                                   epochs=result["epochs"], secs=result["secs"])
                        ari = adjusted_rand_score(test["truth"], result["labels"])
                        utils.append_row(args.out, DATASET, "clustering", dict(row, ari=ari))
                        mse = float("nan")
                        if rate > 0:
                            mse = float(((test["Y"] - Y_hat)[test["M"] == 0] ** 2).mean())
                            utils.append_row(args.out, DATASET, "reconstruction",
                                             dict(row, mse=mse))
                        print(f"{name:16} {mech:8}/{rate:>2} s{seed:<2} ARI {ari:.4f} "
                              f"MSE {mse:.4f} ({result['secs']:.0f}s)", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LCMM-VAE on gait, by missingness condition.")
    parser.add_argument("--mech", nargs="+", choices=list(CONDITIONS), default=list(CONDITIONS),
                        help="missingness mechanisms, each at its rates")
    parser.add_argument("--methods", nargs="+", choices=(utils.OURS, utils.G1),
                        default=[utils.OURS, utils.G1], help="methods to fit")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(30)),
                        help="held-out pair and training seeds")
    parser.add_argument("--out", default=utils.RESULTS, help="output root: result rows")
    main(parser.parse_args())
