"""
PhysioNet 2012 on GP-VAE's extraction and split (Fortuin et al., 2020): fit on x_train,
impute x_test by the prior path; mortality AUROC of a logistic regression on the
imputation, and MSE at m_test_artificial.

usage: python -m experiments.physionet.run [--seeds 0 1 ...] [--out results]
"""
import argparse
import os
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from experiments import utils
from src.data_io import read_data
from src.utils import ROOT

DATASET, TASK = "physionet", "mortality"


def load_npz(data_dir):
    """
    x_test of physionet.npz (m == 1 means missing).

    Returns:
    - test_set (dict): 'x' x_test_miss [N, T, K], 'obs' its observed cells, 'full' the complete
      x_test, 'art' m_test_artificial, 'y' mortality [N].
    """
    f = Path(data_dir) / "physionet.npz"
    if not f.exists():
        raise SystemExit(f"{f} not found: run `python -m data_loader.physionet`")
    npz = np.load(f)
    return dict(x=npz["x_test_miss"].astype(np.float32), obs=npz["m_test_miss"] == 0,
                full=npz["x_test_full"].astype(np.float64), art=npz["m_test_artificial"],
                y=npz["y_test"].astype(int))


def held_out_mse(imputation, test_set):
    artificial = test_set["art"] == 1
    imputation = np.nan_to_num(np.asarray(imputation, np.float64))
    return float(np.mean((imputation[artificial] - test_set["full"][artificial]) ** 2))


def their_auroc(imputation, test_set):
    """GP-VAE's downstream AUROC: observed cells restored, logistic regression fit on the
    first half of the stays, scored on the second."""
    x = np.asarray(imputation, dtype=np.float64).copy()
    x[test_set["obs"]] = test_set["x"][test_set["obs"]]
    x = np.nan_to_num(x.reshape(len(x), -1))
    half = len(x) // 2
    classifier = LogisticRegression(solver="liblinear", tol=1e-10, max_iter=10000)
    classifier.fit(x[:half], test_set["y"][:half])
    return float(roc_auc_score(test_set["y"][half:],
                               classifier.predict_proba(x[half:])[:, 1]))


def test_panel(config, test_set):
    """The x_test panel next to the training panel; its observed cells must be the npz's
    minus m_test_artificial."""
    test_dir = Path(config["dataset"]["data_path"]).parent / "test"
    if not (test_dir / "mask.csv").exists():
        raise FileNotFoundError(f"{test_dir}: run `python -m data_loader.physionet`")
    T, K = test_set["x"].shape[1:]
    mask = read_data(str(test_dir))[1]
    if len(mask) != len(test_set["y"]) * T:
        raise RuntimeError(f"{test_dir} has {len(mask) // T} stays, "
                           f"the npz {len(test_set['y'])}")
    if (test_set["obs"] != (mask.reshape(-1, T, K) == 1)).any():
        raise RuntimeError(f"{test_dir} and the npz disagree on observed cells: re-export it")
    return str(test_dir)


def main(args):
    test_set = load_npz(args.data)
    T, K = test_set["x"].shape[1:]
    for seed in args.seeds:
        def load(method):
            return utils.load_config("physionet", seed)

        def run_dir(method):
            return os.path.join(args.out, "runs", DATASET, utils.RUN_NAMES[method],
                                f"seed{seed}")

        test_dir = test_panel(load(None), test_set)
        for name, result in utils.fits(args.methods, load, seed, test_dir, run_dir):
            imputation = utils.to_raw(utils.reconstruct(result), result["scaler"])
            imputation = imputation.reshape(-1, T, K)
            row = dict(method=name, seed=seed, auroc=their_auroc(imputation, test_set),
                       mse=held_out_mse(imputation, test_set), epochs=result["epochs"],
                       secs=result["secs"])
            utils.append_row(args.out, DATASET, TASK, row)
            print(f"{name:16} s{seed:<2} AUROC {row['auroc']:.4f} MSE {row['mse']:.4f} "
                  f"({result['secs']:.0f}s)", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LCMM-VAE on PhysioNet 2012, scored on x_test.")
    parser.add_argument("--methods", nargs="+", choices=(utils.OURS, utils.G1),
                        default=[utils.OURS, utils.G1], help="methods to fit")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(10)),
                        help="training seeds")
    parser.add_argument("--data", default=str(ROOT / "data" / "physionet"),
                        help="directory of physionet.npz")
    parser.add_argument("--out", default=utils.RESULTS, help="output root: rows and checkpoints")
    main(parser.parse_args())
