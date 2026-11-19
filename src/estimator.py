"""LCMMVAE: the estimator (fit, assignment, trajectories, reconstruction, sampling,
save/load)."""
import os
import warnings

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src import trainer
from src.closed_forms import xtx_per_subject, latent_mean
from src.data_io import read_data, check_data
from src.datasets import get_dataloaders, visit_dataset
from src.initialisation import initialise
from src.model import build_model
from src.utils import load_config, set_seed, make_optimiser, design_matrix, default_covariates


class LCMMVAE:
    """
    LCMM-VAE: G=1 pretraining, em-EM initialisation, then variational EM. The number of
    classes G is model.n_clusters of the config.

    `data` is a directory holding data.csv, mask.csv and label.csv (one row per visit;
    label.csv with columns unit, t and any covariate columns), or a tuple (Y, M, labels) as
    returned by src.data_io.read_data. Hidden cells (M == 0) may hold any value or NaN.
    Subject-level outputs are indexed by `unit`; visit-level outputs have index unit and
    columns t, then one column per marker or latent coordinate. Marker values are in the
    units of data.csv.

    Args:
    - config (str or dict): config name in configs/, YAML path, or config dict.
    - random_state (int): training seed; None keeps training.seed of the config.
    - device (str): torch device; None uses CUDA when available.

    Attributes (after fit):
    - labels_ (Series): class of each training subject. [N]
    - responsibilities_ (DataFrame): gamma of the training subjects. [N, G]
    - proportions_ (ndarray): class proportions pi. [G]
    - scaler_ (dict): training 'mean' and 'std' of each marker. [K] each
    - time_range_ (list): [min, max] of the training visit times.
    - knots_ (list): B-spline knots; None for uniform knots.
    - n_features_in_ (int): number of markers K.
    - n_iter_ (int): epochs of the LCMM-VAE phase.
    - history_ (dict): per-epoch metrics of the LCMM-VAE phase: train_loss, train_recon,
      train_kl_latent, train_kl_class, train_kl_re, train_gamma_usage,
      train_gamma_switched.
    - config_ (dict): the resolved config.
    - module_ (LCMMVAENet): the trained network.
    """

    def __init__(self, config, random_state=None, device=None):
        self.config = config
        self.random_state = random_state
        self.device = device

    def fit(self, data):
        """
        Pretrain at G=1, initialise by em-EM and train, on every subject of `data`.

        Args:
        - data (str or tuple): training panel.

        Returns:
        - self
        """
        config = load_config(self.config, self.random_state)
        if config['dataset'].pop('heldout_fraction', 0):
            warnings.warn("dataset.heldout_fraction is ignored by LCMMVAE.fit: every "
                          "subject of `data` is trained on")
        seed, device = config['training']['seed'], self._device()
        Y, M, labels = _read(data)
        train_loader, _, widths, scaler, design = get_dataloaders(config, (Y, M, labels))
        # advances the loader's shuffle generator: reproduces the published runs
        next(iter(train_loader))
        set_seed(seed)

        pretrained_model = build_model(config, Y.shape[1], widths).to(device)
        trainer.pretrain(
            model=pretrained_model,
            optimiser=make_optimiser(pretrained_model.parameters(),
                                     lr=config['pretraining']['lr']),
            loader=train_loader,
            epochs=config['pretraining']['epochs'],
            kl_annealing_epochs=config['pretraining']['kl_annealing_epochs'],
            device=device)

        model = initialise(config=config, pretrained_model=pretrained_model,
                           train_loader=train_loader, widths=widths,
                           G=config['model']['n_clusters'], seed=seed, device=device)
        history = trainer.train(
            model=model,
            optimiser=make_optimiser(model.parameters(), lr=config['training']['lr']),
            loader=train_loader,
            epochs=config['training']['epochs'],
            tol=config['training']['tol'],
            estep_sweeps=config['model']['estep_sweeps'],
            estep_sweeps_infer=config['model']['estep_sweeps_infer'],
            device=device)

        self.config_ = config
        self._set_fitted(model, scaler, design, widths, Y.shape[1])
        self.history_ = history
        self.n_iter_ = len(history['train_loss'])
        gamma = self._posterior(train_loader)[0].cpu().numpy()
        self.responsibilities_ = self._subjects(gamma, train_loader.dataset.subject_ids)
        self.labels_ = self.responsibilities_.idxmax(axis=1).rename('class')
        return self

    def fit_predict(self, data):
        """Fit, then return labels_, the classes of the training subjects. [N]"""
        return self.fit(data).labels_

    def predict_proba(self, data):
        """Responsibilities gamma_ig of the subjects of `data`, a DataFrame. [N, G]"""
        loader = self._loader(*self._panel(data))
        gamma = self._posterior(loader)[0].cpu().numpy()
        return self._subjects(gamma, loader.dataset.subject_ids)

    def predict(self, data):
        """Classes argmax_g gamma_ig of the subjects of `data`, a Series. [N]"""
        return self.predict_proba(data).idxmax(axis=1).rename('class')

    def encode(self, data):
        """Encoder means mu_phi(y_ij) at the visits of `data`. [n_visits, 1 + L]"""
        Y, M, labels = self._panel(data)
        cache = self.module_.encode_loader(self._loader(Y, M, labels))
        return self._visits(labels, cache['mu_phi'].cpu().numpy())

    @torch.no_grad()
    def latent_trajectories(self, data, t=None):
        """
        Latent trajectory of each subject, sum_g gamma_ig (B^(g) x_S + m_i^(g) x_R), after
        an E-step on its observed visits.

        Args:
        - data (str or tuple): observed visits.
        - t (array or DataFrame): times evaluated for every subject, or rows with columns
          unit and t; None uses the visits of `data`. Covariate columns absent from a
          DataFrame take each subject's value at its first visit.

        Returns:
        - z (DataFrame): index unit, columns t and the L latent coordinates. [M, 1 + L]
        """
        rows, z = self._latent(*self._panel(data), t)
        return self._visits(rows, z.cpu().numpy())

    @torch.no_grad()
    def reconstruct(self, data, t=None):
        """
        Reconstruction f_theta(sum_g gamma_ig (B^(g) x_S + m_i^(g) x_R)), the decoded
        latent_trajectories. Outside time_range_ the B-spline basis extends linearly.

        Args:
        - data (str or tuple): observed visits.
        - t (array or DataFrame): as in latent_trajectories.

        Returns:
        - y_hat (DataFrame): index unit, columns t and the K markers. [M, 1 + K]
        """
        rows, z = self._latent(*self._panel(data), t)
        return self._visits(rows, self._to_marker_units(self.module_.decoder(z)))

    @torch.no_grad()
    def impute(self, data):
        """
        Observed cells of `data` kept, hidden cells filled by reconstruct.

        Args:
        - data (str or tuple): observed visits.

        Returns:
        - Y (DataFrame): index unit, columns t and the K markers. [n_visits, 1 + K]
        """
        Y, M, labels = self._panel(data)
        rows, z = self._latent(Y, M, labels, None)
        y_hat = self._to_marker_units(self.module_.decoder(z))
        return self._visits(rows, np.where(M == 1, Y, y_hat))

    def random_effects(self, data):
        """
        Posterior means of the random effects, sum_g gamma_ig m_i^(g).

        Returns:
        - a_R (ndarray): subjects in the order of predict_proba(data).index. [N, L, R]
        """
        gamma, m_g, _ = self._posterior(self._loader(*self._panel(data)))
        return torch.einsum('ng,gnlr->nlr', gamma, m_g).cpu().numpy()

    @torch.no_grad()
    def class_trajectories(self, t, latent=False, covariates=None):
        """
        Class trajectories B^(g) x_S, without subject effects, decoded unless latent.

        Args:
        - t (array): times. [T]
        - latent (bool): return the latent trajectories instead.
        - covariates (dict): values of the covariate columns; default the reference level
          of categorical and the training mean of continuous columns.

        Returns:
        - y (ndarray): [G, T, K], or [G, T, L] if latent.
        """
        self._check_fitted()
        xS, _ = self._design(self._rows_at(t, covariates))
        _, B_g, _ = self.module_.prior.class_coefficients()
        z = torch.einsum('gls,ts->gtl', B_g, xS)
        if latent:
            return z.cpu().numpy()
        return self._to_marker_units(self.module_.decoder(z))

    @torch.no_grad()
    def sample(self, n_samples, t, random_state=None, covariates=None):
        """
        Draw subjects from the generative model: c ~ Cat(pi), a_R,l ~ N(0, D_l),
        z_j = B^(c) x_S,j + A_R x_R,j + eps_j with eps_j ~ N(0, sigma_eps2 I), and
        y_j ~ N(f_theta(z_j), diag(sigma_y^2)).

        Args:
        - n_samples (int): number of subjects.
        - t (array): visit times shared by all subjects. [T]
        - random_state (int): seed of the draws; None uses the global torch RNG.
        - covariates (dict): as in class_trajectories.

        Returns:
        - Y (ndarray): observations. [n_samples, T, K]
        - classes (ndarray): class of each subject. [n_samples]
        """
        self._check_fitted()
        generator = (None if random_state is None
                     else torch.Generator().manual_seed(random_state))
        model, prior = self.module_, self.module_.prior
        device = model.log_vy.device
        xS, xR = self._design(self._rows_at(t, covariates))
        n, T, L, R = n_samples, len(xS), prior.L, prior.dR

        def normal(*shape):
            return torch.randn(*shape, generator=generator).to(device)

        c = torch.multinomial(prior.pi.cpu(), n, replacement=True, generator=generator)
        c = c.to(device)
        a_R = torch.einsum('lrq,nlq->nlr', torch.linalg.cholesky(prior.D) if R else prior.D,
                           normal(n, L, R))
        _, B_g, _ = prior.class_coefficients()
        z = (torch.einsum('nls,ts->ntl', B_g[c], xS)
             + torch.einsum('nlr,tr->ntl', a_R, xR)
             + prior.sigma_eps2.sqrt() * normal(n, T, L))
        y = model.decoder(z) + (0.5 * model.log_vy).exp() * normal(n, T, self.n_features_in_)
        return self._to_marker_units(y), c.cpu().numpy()

    def save(self, path):
        """Write the fitted estimator to one file, loadable with weights_only=True."""
        self._check_fitted()
        torch.save(dict(
            random_state=self.random_state, config=self.config_,
            state_dict=self.module_.state_dict(),
            scaler={k: torch.as_tensor(v) for k, v in self.scaler_.items()},
            design=self._design_info, widths=self._widths, n_features_in=self.n_features_in_,
            n_iter=self.n_iter_, history=self.history_,
            responsibilities=torch.tensor(self.responsibilities_.to_numpy()),
            units=self.responsibilities_.index.tolist()), path)

    @classmethod
    def load(cls, path, device=None):
        """Rebuild an estimator written by save; no training data needed."""
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        config = checkpoint['config']
        estimator = cls(config, random_state=checkpoint['random_state'], device=device)
        estimator.config_ = config
        model = build_model(config, checkpoint['n_features_in'], checkpoint['widths'],
                            config['model']['n_clusters'])
        model.load_state_dict(checkpoint['state_dict'])
        estimator._set_fitted(model.to(estimator._device()),
                              {k: v.numpy() for k, v in checkpoint['scaler'].items()},
                              checkpoint['design'], checkpoint['widths'],
                              checkpoint['n_features_in'])
        estimator.n_iter_, estimator.history_ = checkpoint['n_iter'], checkpoint['history']
        estimator.responsibilities_ = estimator._subjects(
            checkpoint['responsibilities'].numpy(), checkpoint['units'])
        estimator.labels_ = estimator.responsibilities_.idxmax(axis=1).rename('class')
        return estimator

    def _check_fitted(self):
        if not hasattr(self, 'module_'):
            raise RuntimeError("this LCMMVAE is not fitted yet: call fit(data) or "
                               "LCMMVAE.load(path) first")

    def _device(self):
        return self.device or ('cuda' if torch.cuda.is_available() else 'cpu')

    def _set_fitted(self, model, scaler, design, widths, K):
        self.module_ = model.eval()
        self.scaler_, self._design_info, self._widths = scaler, design, widths
        self.n_features_in_ = K
        self.time_range_ = list(design['time_range'])
        self.knots_ = None if design['knots'] is None else list(design['knots'])
        self.proportions_ = model.prior.pi.detach().cpu().numpy()

    def _panel(self, data):
        """(Y, M, labels) of `data`, checked against the fitted model."""
        self._check_fitted()
        return _read(data, self.n_features_in_)

    def _loader(self, Y, M, labels):
        """Loader over the visits (in file order) on the training scale."""
        dataset, _ = visit_dataset(Y, M, labels, self.scaler_,
                                   self.config_['dataset']['covariates'], self._design_info)
        return DataLoader(dataset, batch_size=self.config_['dataset']['batch_size'])

    @torch.no_grad()
    def _posterior(self, loader):
        """
        q(c) and q(a | c) of the subjects of `loader`; mean-field sweeps start at pi.

        Returns:
        - gamma, m_g (Tensor): [N, G], [G, N, L, dR]
        - cache (dict): the encoder pass, from encode_loader.
        """
        model = self.module_
        cache = model.encode_loader(loader)
        N = loader.dataset.n_subjects
        XtX_R = xtx_per_subject(cache['xR'], cache['sid'], N)
        pi_start = model.prior.pi.detach().unsqueeze(0).expand(N, -1).clone()
        gamma, m_g, _, _ = model.e_step(cache, XtX_R, pi_start,
                                        self.config_['model']['estep_sweeps_infer'])
        return gamma, m_g, cache

    def _latent(self, Y, M, labels, t):
        """Rows (unit, t) and the latent trajectory at them. [M, L]"""
        loader = self._loader(Y, M, labels)
        gamma, m_g, cache = self._posterior(loader)
        if t is None:
            rows, xS, xR, sid = labels, cache['xS'], cache['xR'], cache['sid']
        else:
            ids = loader.dataset.subject_ids
            if isinstance(t, pd.DataFrame):
                if not {'unit', 't'} <= set(t.columns):
                    raise ValueError(f"a DataFrame t needs columns 'unit' and 't'; it has "
                                     f"{list(t.columns)}")
                rows = t.reset_index(drop=True)
            else:
                times = np.asarray(t, dtype=float).reshape(-1)
                rows = pd.DataFrame({'unit': np.repeat(ids, len(times)),
                                     't': np.tile(times, len(ids))})
            sid = pd.Index(ids).get_indexer(rows['unit'])
            if (sid < 0).any():
                raise ValueError("t has units with no visit in `data`")
            cols = [c for c in self._design_info['columns'] if c not in rows]
            if cols:
                first = labels.groupby('unit', sort=False)[cols].first()
                rows = pd.concat([rows, first.reindex(rows['unit']).reset_index(drop=True)],
                                 axis=1)
            xS, xR = self._design(rows)
            sid = torch.as_tensor(sid, device=xS.device)
        _, B_g, _ = self.module_.prior.class_coefficients()
        return rows, latent_mean(B_g, gamma, m_g, xS, xR, sid)

    def _rows_at(self, t, covariates=None):
        """Rows at times t with the covariate values of default_covariates(covariates)."""
        rows = pd.DataFrame({'t': np.asarray(t, dtype=float)})
        for column, value in default_covariates(self._design_info, covariates).items():
            rows[column] = value
        return rows

    def _design(self, rows):
        """x_S and x_R of rows with t and the covariate columns."""
        x, _ = design_matrix(rows, self.config_['dataset']['covariates'], self._design_info)
        return self.module_.split_x(torch.as_tensor(x, dtype=torch.float32,
                                                    device=self.module_.log_vy.device))

    def _to_marker_units(self, y):
        return y.detach().cpu().numpy() * self.scaler_['std'] + self.scaler_['mean']

    @staticmethod
    def _subjects(gamma, units):
        return pd.DataFrame(gamma, index=pd.Index(units, name='unit'))

    @staticmethod
    def _visits(rows, values):
        out = pd.DataFrame(values, index=pd.Index(rows['unit'].values, name='unit'))
        out.insert(0, 't', rows['t'].values)
        return out


def _read(data, n_markers=None):
    """(Y, M, labels) of a data directory or tuple, checked by check_data."""
    if isinstance(data, (str, os.PathLike)):
        data = read_data(str(data))
    return check_data(*data, n_markers=n_markers)
