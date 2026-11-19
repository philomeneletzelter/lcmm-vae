"""Visit-level dataset, train/held-out split and data loaders."""
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from src.data_io import read_data, marker_scaler
from src.utils import fit_design, design_matrix


class VisitDataset(Dataset):
    """
    One item per visit: masked observations, covariate row, mask and subject code.

    Args:
    - masked_data (ndarray): observations, hidden cells at 0. [n_visits, K]
    - covariates (ndarray): design rows [x_R | x_S]. [n_visits, dR + dS]
    - masks (ndarray): 1 where observed. [n_visits, K]
    - subjects (ndarray): subject code of each visit, contiguous 0..N-1. [n_visits]
    - subject_ids (ndarray): label.csv `unit` of each subject code. [N]
    """

    def __init__(self, masked_data, covariates, masks, subjects, subject_ids):
        self.masked_data = masked_data
        self.covariates = covariates
        self.masks = masks
        self.subjects = subjects
        self.subject_ids = np.asarray(subject_ids)
        self.n_subjects = len(self.subject_ids)
        self.n_markers = masked_data.shape[1]

    def __len__(self):
        return len(self.covariates)

    def __getitem__(self, idx):
        return {'observations': torch.tensor(self.masked_data[idx], dtype=torch.float32),
                'covariates': torch.tensor(self.covariates[idx], dtype=torch.float32),
                'mask': torch.tensor(self.masks[idx], dtype=torch.float32),
                'subject': torch.tensor(self.subjects[idx], dtype=torch.long)}


def _split(labels, heldout_frac, seed):
    """
    Train/held-out row split by subject.

    Args:
    - labels (DataFrame): label.csv.
    - heldout_frac (float): fraction of the subjects to hold out.
    - seed (int): seed of the shuffle.

    Returns:
    - train_idx, heldout_idx (list): row positions for each split.
    """
    rng = np.random.default_rng(seed)
    subjects = labels['unit'].unique().tolist()
    rng.shuffle(subjects)
    n_out = int(heldout_frac * len(subjects))
    heldout, train = subjects[:n_out], subjects[n_out:]

    def rows(keep):
        return labels[labels['unit'].isin(keep)].index.tolist()
    return rows(train), rows(heldout)


def ordered(loader):
    """
    Non-shuffled copy of `loader` with its own generator; iterating it leaves the global
    torch RNG (and training shuffles) untouched.

    Args:
    - loader (DataLoader): any loader.

    Returns:
    - loader (DataLoader): a non-shuffled loader over loader.dataset.
    """
    return DataLoader(loader.dataset, batch_size=loader.batch_size, shuffle=False,
                      num_workers=loader.num_workers, collate_fn=loader.collate_fn,
                      pin_memory=loader.pin_memory, generator=torch.Generator())


def visit_dataset(Y, M, labels, scaler, covariates, design):
    """
    The dataset of one panel: observed cells standardised by `scaler`, hidden cells at 0,
    and the design rows. Arithmetic is in the dtype of Y and scaler.

    Args:
    - Y (ndarray): marker values; hidden cells may hold any value or NaN. [n_visits, K]
    - M (ndarray): 1 where observed. [n_visits, K]
    - labels (DataFrame): label.csv rows of the visits.
    - scaler (dict): 'mean' and 'std'. [K] each
    - covariates (dict): the config's dataset.covariates block.
    - design (dict): from fit_design.

    Returns:
    - dataset (VisitDataset): subjects coded 0..N-1 in order of first visit.
    - widths (dict): 'random' (dR) and 'fixed' (dS).
    """
    obs = np.where(M == 1, (Y - scaler['mean']) / scaler['std'], 0.0)
    X, widths = design_matrix(labels, covariates, design)
    return VisitDataset(obs, X, M, *pd.factorize(labels['unit'].values)), widths


def get_dataloaders(config, data=None):
    """
    Build the training and held-out loaders from a config.

    Args:
    - config (dict): run config; reads the dataset block and training.seed.
    - data (str or tuple): data directory or (Y, M, labels); None reads
      dataset.data_path.

    Returns:
    - train_loader, heldout_loader (DataLoader): heldout_loader is None when
      dataset.heldout_fraction is 0 or absent.
    - widths (dict): 'random' (dR) and 'fixed' (dS).
    - scaler (dict): training 'mean' and 'std', float64. [K] each
    - design (dict): from fit_design on the training visits.
    """
    data_config = config['dataset']
    batch_size = data_config['batch_size']
    seed = config['training']['seed']

    if data is None or isinstance(data, (str, os.PathLike)):
        Y, M, labels = read_data(data_config['data_path'] if data is None else data)
    else:
        Y, M, labels = data

    train_idx, heldout_idx = _split(labels, data_config.get('heldout_fraction', 0), seed)

    # time basis and covariate encodings from the training visits only
    covariates = data_config['covariates']
    design = fit_design(labels.iloc[train_idx], covariates)

    scaler = marker_scaler(Y[train_idx], M[train_idx])

    def dataset(idx):
        return visit_dataset(Y[idx], M[idx], labels.iloc[idx], scaler, covariates, design)

    train_dataset, widths = dataset(train_idx)
    heldout_dataset = dataset(heldout_idx)[0] if heldout_idx else None
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              generator=torch.Generator().manual_seed(seed))
    heldout_loader = (DataLoader(heldout_dataset, batch_size=batch_size, shuffle=False)
                      if heldout_dataset is not None else None)
    return train_loader, heldout_loader, widths, scaler, design
