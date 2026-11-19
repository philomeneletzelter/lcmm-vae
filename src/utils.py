"""Config loading, seeding, optimiser and the design X = [x_R | x_S]."""
import copy
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_config(config, seed=None):
    """
    Args:
    - config (str or dict): config name in configs/ (e.g. 'gait'), YAML path, or dict.
    - seed (int): training seed; None keeps the config's.

    Returns:
    - config (dict): a new dict, with a relative dataset.data_path made absolute from the
      repository root.
    """
    if isinstance(config, dict):
        config = copy.deepcopy(config)
    else:
        name = str(config)
        path = (Path(name) if name.endswith(('.yaml', '.yml'))
                else ROOT / 'configs' / f'{name}.yaml')
        with open(path) as file:
            config = yaml.safe_load(file)
    if seed is not None:
        config['training']['seed'] = seed
    data_path = config['dataset'].get('data_path')
    if data_path is not None and not os.path.isabs(data_path):
        config['dataset']['data_path'] = str(ROOT / data_path)
    return config


def set_seed(seed_value=42):
    """
    Seed torch, numpy and random, and force deterministic cuDNN. On CUDA, index_add_ stays
    non-deterministic.

    Args:
    - seed_value (int): the seed.
    """
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    np.random.seed(seed_value)
    random.seed(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_optimiser(params, lr):
    """
    Adam with default betas, used by both training stages.

    Args:
    - params (iterable): the parameters to optimise.
    - lr (float): learning rate.

    Returns:
    - optimiser (torch.optim.Adam)
    """
    return torch.optim.Adam(params, lr=lr)


def generate_bspline_basis(time_vector, n_basis, degree=3, t_range=None, knots=None):
    """
    B-spline basis on a 1-D time vector, extrapolated linearly outside the knots.

    Args:
    - time_vector (ndarray): the times. [N] or [N, 1]
    - n_basis (int): number of basis functions.
    - degree (int): spline degree; 3 is cubic.
    - t_range (tuple): (lo, hi) of uniform knots. None uses the range of time_vector.
    - knots (list): all knots, boundary included; overrides t_range. [n_basis - degree + 1]

    Returns:
    - basis (ndarray): the evaluated basis. [N, n_basis]
    """
    from sklearn.preprocessing import SplineTransformer

    t = np.asarray(time_vector, dtype=np.float32).reshape(-1, 1)
    n_knots = max(2, n_basis - degree + 1)
    if knots is not None:
        knots = np.asarray(knots, dtype=np.float64).reshape(-1, 1)
        spline = SplineTransformer(degree=degree, knots=knots, include_bias=True,
                                   extrapolation='linear')
        return spline.fit(knots).transform(t).astype(np.float32)
    spline = SplineTransformer(n_knots=n_knots, degree=degree, include_bias=True,
                               knots='uniform', extrapolation='linear')
    span = t if t_range is None else np.asarray(t_range, dtype=np.float32).reshape(-1, 1)
    return spline.fit(span).transform(t).astype(np.float32)


def fit_design(labels, covariates):
    """
    Training-set state of the design: time range, knots and covariate-column encodings.
    Quantile knots: boundary knots at the 1st and 99th percentiles of the training visit
    times, interior knots at equally spaced percentiles in between.

    Args:
    - labels (DataFrame): the training visits' label.csv rows, with t and the covariate
      columns.
    - covariates (dict): the config's dataset.covariates block.

    Returns:
    - design (dict): 'time_range' [lo, hi]; 'knots' (list, or None for uniform knots);
      'columns' {name: {'levels': [...]} (categorical, first is the reference) or
      {'mean', 'std'} (continuous)}.
    """
    t = labels['t']
    design = {'time_range': [float(t.min()), float(t.max())], 'knots': None, 'columns': {}}
    knots = covariates.get('knots', 'uniform')
    if knots == 'quantile':
        n_knots = max(2, covariates['n_basis'] - 3 + 1)
        q = np.percentile(t.to_numpy(np.float64), np.linspace(1, 99, n_knots))
        if np.any(np.diff(q) <= 0):
            raise ValueError("quantile knots are not distinct: too few distinct visit times "
                             "for n_basis; lower n_basis or use knots: uniform")
        design['knots'] = q.tolist()
    elif knots != 'uniform':
        raise ValueError(f"dataset.covariates.knots must be 'uniform' or 'quantile', "
                         f"got {knots!r}")
    for term in covariates['terms']:
        if term['transform'] != 'columns':
            continue
        for column in term['columns']:
            if column not in labels:
                raise ValueError(f"covariate column {column!r} is not in label.csv")
            values = labels[column]
            if values.isna().any():
                raise ValueError(f"covariate column {column!r} has missing values")
            if (column in term.get('categorical', [])
                    or not pd.api.types.is_numeric_dtype(values)):
                levels = values.unique().tolist()
                try:
                    levels = sorted(levels)
                except TypeError:
                    levels = sorted(levels, key=str)
                design['columns'][column] = {'levels': levels}
            else:
                design['columns'][column] = {'mean': float(values.mean()),
                                             'std': float(values.std(ddof=0)) or 1.0}
    return design


def _columns_block(labels, columns, design):
    """Categorical columns one-hot without the reference level; continuous standardised."""
    out = []
    for column in columns:
        encoding = design['columns'][column]
        values = labels[column]
        if 'levels' in encoding:
            unseen = set(values.unique().tolist()) - set(encoding['levels'])
            if unseen:
                raise ValueError(f"covariate column {column!r} has levels unseen in "
                                 f"training: {sorted(unseen, key=str)}")
            dummies = [(values == level).to_numpy(np.float32)
                       for level in encoding['levels'][1:]]
            out.append(np.stack(dummies, axis=1) if dummies
                       else np.empty((len(values), 0), np.float32))
        else:
            z = (values.to_numpy(np.float64) - encoding['mean']) / encoding['std']
            out.append(z.astype(np.float32).reshape(-1, 1))
    return np.hstack(out) if out else np.empty((len(labels), 0), np.float32)


def default_covariates(design, values=None):
    """
    Covariate values for design rows built without a subject: the reference level of each
    categorical column and the training mean of each continuous one, overridden by values.

    Args:
    - design (dict): from fit_design.
    - values (dict): {column: value}.

    Returns:
    - row (dict): {column: value}
    """
    values = dict(values or {})
    unknown = set(values) - set(design['columns'])
    if unknown:
        raise ValueError(f"unknown covariate columns {sorted(unknown)}; the design has "
                         f"{sorted(design['columns'])}")
    row = {column: (encoding['levels'][0] if 'levels' in encoding else encoding['mean'])
           for column, encoding in design['columns'].items()}
    row.update(values)
    return row


def _build_covariate_term(labels, term, time_features, time_raw, design):
    """
    One covariate block from a term of the config. term['transform'] is one of

      'bspline'          the time basis
      'intercept_slope'  (1, t), the x_R of a random intercept and slope (R = 2)
      'columns'          label.csv columns term['columns']: categorical one-hot without the
                         reference level, continuous standardised by the training visits

    Args:
    - labels (DataFrame): one row per observation.
    - term (dict): transform and block.
    - time_features (ndarray): the time basis. [N, n_basis]
    - time_raw (ndarray): the times. [N]
    - design (dict): from fit_design.

    Returns:
    - feats (ndarray): the block's columns. [N, width]
    """
    transform = term['transform']
    if transform == 'bspline':
        return time_features
    if transform == 'intercept_slope':
        t = np.asarray(time_raw, dtype=np.float32).reshape(-1)
        return np.stack([np.ones_like(t), t], axis=1)
    if transform == 'columns':
        return _columns_block(labels, term['columns'], design)
    raise ValueError(f"unknown covariate transform: {transform!r}")


def design_matrix(labels, covariates, design):
    """
    The design X = [x_R | x_S] from the config's covariates block = {n_basis, knots, terms},
    on label.csv's time column t. Terms with block: fixed go to x_S, block: random to x_R.
    x_S is shared by A_S and Delta^(g), so each fixed covariate has class-specific effects.

    Args:
    - labels (DataFrame): one row per observation, with t and the covariate columns.
    - covariates (dict): the config's dataset.covariates block.
    - design (dict): from fit_design on the training visits.

    Returns:
    - X (ndarray): the design, float32. [N, dR + dS]
    - widths (dict): block widths, 'random' (dR) and 'fixed' (dS).
    """
    N = len(labels)
    time_raw = labels['t'].values.astype(np.float32)
    time_features = generate_bspline_basis(time_raw, covariates['n_basis'],
                                           t_range=design['time_range'],
                                           knots=design.get('knots'))

    blocks = {'fixed': [], 'random': []}
    for term in covariates['terms']:
        if term['block'] not in blocks:
            raise ValueError(f"unknown covariate block {term['block']!r}; "
                             f"expected 'fixed' or 'random'")
        blocks[term['block']].append(_build_covariate_term(labels, term, time_features,
                                                           time_raw, design))

    def stack(block):
        return np.hstack(blocks[block]) if blocks[block] else np.empty((N, 0), np.float32)
    X_random, X_fixed = stack('random'), stack('fixed')
    return np.hstack([X_random, X_fixed]), {'random': X_random.shape[1],
                                            'fixed': X_fixed.shape[1]}


def split_covariates(x, dR):
    """
    Split the loader's covariate row [x_R | x_S] into its two designs.

    Args:
    - x (Tensor): the loader's covariate rows. [B, dR + dS]
    - dR (int): width of x_R.

    Returns:
    - xS (Tensor): fixed-effect design. [B, dS]
    - xR (Tensor): random-effect design. [B, dR]
    """
    return x[:, dR:].contiguous(), x[:, :dR]
