"""
Data I/O for the layout data.csv and mask.csv ([rows, K], no header; hidden cells of
data.csv may hold any value or be empty), label.csv (unit, t, optional columns) and the
optional data_masked.csv; input checks; standardisation.
"""
import os

import numpy as np
import pandas as pd


def read_data(data_dir):
    """
    Args:
    - data_dir (str): data directory.

    Returns:
    - Y (ndarray): marker values, file row order; NaN where data.csv is empty. [rows, K]
    - M (ndarray): 1 where observed. [rows, K]
    - labels (DataFrame): label.csv, row labels equal to row positions.
    """
    labels = pd.read_csv(os.path.join(data_dir, "label.csv"))
    Y = pd.read_csv(os.path.join(data_dir, "data.csv"), header=None).values.astype(np.float64)
    M = pd.read_csv(os.path.join(data_dir, "mask.csv"), header=None).values.astype(np.float64)
    return Y, M, labels


def check_data(Y, M, labels, n_markers=None):
    """
    Check a panel and convert it to the types used downstream.

    Args:
    - Y (array): marker values; hidden cells may hold any value or NaN. [rows, K]
    - M (array): 1 where observed. [rows, K]
    - labels (DataFrame): one row per data row, with unit and t.
    - n_markers (int): required K; None accepts any.

    Returns:
    - Y, M (ndarray): float64. [rows, K]
    - labels (DataFrame): row labels reset to 0..rows-1.

    Raises:
    - ValueError: on a missing unit or t column, a non-numeric or missing t, mismatched
      shapes, a mask other than 0/1, a non-finite observed cell, or K != n_markers.
    """
    Y = np.asarray(Y, dtype=np.float64)
    M = np.asarray(M, dtype=np.float64)
    labels = pd.DataFrame(labels).reset_index(drop=True)
    for c in ('unit', 't'):
        if c not in labels:
            raise ValueError(f"label.csv needs a column {c!r}; it has {list(labels.columns)}")
    if not pd.api.types.is_numeric_dtype(labels['t']) or labels['t'].isna().any():
        raise ValueError("label.csv column 't' must be numeric with no missing values")
    if Y.ndim != 2 or Y.shape != M.shape or len(Y) != len(labels):
        raise ValueError(f"data {Y.shape}, mask {M.shape} and label.csv "
                         f"({len(labels)} rows) must have one row per visit")
    if not np.isin(M, (0, 1)).all():
        raise ValueError("mask.csv must hold only 0 and 1")
    if not np.isfinite(Y[M == 1]).all():
        raise ValueError("data.csv has non-finite values at observed cells (mask 1)")
    if n_markers is not None and Y.shape[1] != n_markers:
        raise ValueError(f"data has {Y.shape[1]} markers; the model was fitted on {n_markers}")
    return Y, M, labels


def write_data(data_dir, Y, M, labels, float_format=None):
    """
    Args:
    - data_dir (str): directory, created if missing.
    - Y (ndarray): marker values; hidden cells may be NaN. [rows, K]
    - M (ndarray): 1 where observed. [rows, K]
    - labels (DataFrame): one row per data row, with unit and t.
    - float_format (str): passed to to_csv for the values.

    Returns:
    - data_dir (str)
    """
    os.makedirs(data_dir, exist_ok=True)
    pd.DataFrame(Y).to_csv(os.path.join(data_dir, "data.csv"), index=False, header=False,
                           float_format=float_format)
    pd.DataFrame(np.where(M == 1, Y, 0.0)).to_csv(os.path.join(data_dir, "data_masked.csv"),
                                                  index=False, header=False,
                                                  float_format=float_format)
    pd.DataFrame(M.astype(np.int8)).to_csv(os.path.join(data_dir, "mask.csv"), index=False,
                                           header=False)
    labels.to_csv(os.path.join(data_dir, "label.csv"), index=False)
    return data_dir


def marker_scaler(Y, M):
    """
    Per-marker mean and sd over the observed cells.

    Args:
    - Y (ndarray): values, markers on the last axis; hidden cells are ignored. [..., K]
    - M (ndarray): 1 where observed. [..., K]

    Returns:
    - scaler (dict): 'mean' and 'std', float64. [K] each
    """
    K = Y.shape[-1]
    Y = np.asarray(Y, dtype=np.float64).reshape(-1, K)
    M = np.asarray(M, dtype=np.float64).reshape(-1, K)
    n = np.clip(M.sum(0), 1, None)
    obs = M == 1
    mean = np.where(obs, Y, 0.0).sum(0) / n
    var = np.where(obs, (Y - mean) ** 2, 0.0).sum(0) / n
    return {"mean": mean, "std": np.sqrt(np.clip(var, 1e-16, None))}


def standardise(Y, M, scaler, fill):
    """
    Args:
    - Y (ndarray): values, markers on the last axis. [..., K]
    - M (ndarray): 1 where observed. [..., K]
    - scaler (dict): from marker_scaler.
    - fill (float): value of a hidden cell.

    Returns:
    - Z (ndarray): standardised observed cells, `fill` at hidden ones. [..., K]
    """
    return np.where(M == 1, (Y - scaler["mean"]) / scaler["std"], fill)
