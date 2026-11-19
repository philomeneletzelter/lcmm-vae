"""The LMM-VAE (G=1 pretraining) and LCMM-VAE networks, and build_model."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.closed_forms import e_step, POSTERIORS
from src.datasets import ordered
from src.modules import Encoder, Decoder, LCMMPrior, LMMPrior
from src.utils import split_covariates


class _VAE(nn.Module):
    """
    Shared parts: encoder, decoder, prior and the learned observation noise sigma_y^2.

    Args:
    - encoder (Encoder): q_phi(z_ij | y_ij).
    - decoder (Decoder): f_theta(z_ij).
    - prior (nn.Module): LMMPrior or LCMMPrior.
    - K (int): number of markers.
    - vy_init (float): initial sigma_y,k^2.
    - vy_min (float): floor on sigma_y,k^2.
    """

    def __init__(self, encoder, decoder, prior, K, vy_init=1.0, vy_min=0.0003):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.prior = prior

        self.register_buffer('min_log_vy', torch.tensor(float(np.log(vy_min))))
        # starts sigma_y,k^2 at vy_init
        self._log_vy = nn.Parameter(torch.full((K,), float(np.log(vy_init - vy_min))))

    @property
    def log_vy(self):
        """log sigma_y,k^2 with the softplus floor at min_log_vy applied. [K]"""
        return self.min_log_vy + F.softplus(self._log_vy - self.min_log_vy)

    def sample_latent(self, mu, logvar):
        """
        One reparameterised draw from q_phi.

        Args:
        - mu (Tensor): posterior means. [B, L]
        - logvar (Tensor): posterior log-variances. [B, L]

        Returns:
        - z (Tensor): a single sample. [B, L]
        """
        return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)


class LMMVAENet(_VAE):
    """
    G=1 pretraining network with an LMMPrior; its encoder, decoder and sigma_y^2 initialise
    LCMMVAENet.

    Args: as _VAE.
    """

    def forward(self, y, x):
        """
        One training pass: encode, sample, decode, and evaluate the covariate prior.

        Args:
        - y (Tensor): observations. [B, K]
        - x (Tensor): covariates. [B, dR + dS]

        Returns:
        - out (dict): y_recon, z_mu, z_logvar, log_vy, and the prior moments
          batch_prior_mu, batch_prior_var, global_A_mu, global_A_logvar.
        """
        z_mu, z_logvar = self.encoder(y)
        y_hat = self.decoder(self.sample_latent(z_mu, z_logvar))
        global_A_mu, global_A_logvar, batch_prior_mu, batch_prior_var = self.prior(x)
        return {'y_recon': y_hat, 'z_mu': z_mu, 'z_logvar': z_logvar, 'log_vy': self.log_vy,
                'batch_prior_mu': batch_prior_mu, 'batch_prior_var': batch_prior_var,
                'global_A_mu': global_A_mu, 'global_A_logvar': global_A_logvar}


class LCMMVAENet(_VAE):
    """
    Latent-class mixed-model VAE. Only encoder, decoder and sigma_y^2 are optimised;
    pi, A_S, Delta^(g) and D are prior buffers set by closed-form updates. Per-subject
    gamma, m^(g) and V live outside the model, computed by e_step.

    Args: as _VAE, with prior an LCMMPrior, and
    - posterior (str): 'meanfield' or 'structured'.
    """

    def __init__(self, encoder, decoder, prior, K, posterior, vy_init=1.0, vy_min=0.0003):
        if posterior not in POSTERIORS:
            raise ValueError(f"posterior must be one of {POSTERIORS}, got {posterior!r}")
        super().__init__(encoder, decoder, prior, K, vy_init, vy_min)
        self.posterior = posterior

    def split_x(self, x):
        """
        Args:
        - x (Tensor): the loader's [x_R | x_S] row. [B, dR + dS]

        Returns:
        - xS (Tensor): fixed-effect design. [B, dS]
        - xR (Tensor): random-effect design. [B, dR]
        """
        return split_covariates(x, self.prior.dR)

    def forward(self, y, x, sid, m_g, V):
        """
        One training pass: encode, sample, decode, and the prior moments at q(a | c).

        Args:
        - y (Tensor): observations. [B, K]
        - x (Tensor): covariates, laid out [x_R | x_S]. [B, dR + dS]
        - sid (Tensor): each visit's subject, indexing m_g and V. [B]
        - m_g (Tensor): random-effect means. [G, N, L, dR]
        - V (Tensor): random-effect covariances. [N, L, dR, dR]

        Returns:
        - out (dict): y_recon, z_mu, z_logvar, log_vy; mbar, class mean plus its
          random-effect mean [B, G, L]; v_re, x_R^T V_i x_R [B, L].
        """
        xS, xR = self.split_x(x)

        z_mu, z_logvar = self.encoder(y)
        y_hat = self.decoder(self.sample_latent(z_mu, z_logvar))

        _, B_g, _ = self.prior.class_coefficients()
        m_g, V_i = m_g[:, sid].detach(), V[sid].detach()          # [G, B, L, dR], [B, L, dR, dR]
        mbar = (torch.einsum('gls,bs->bgl', B_g, xS)
                + torch.einsum('gblr,br->bgl', m_g, xR))
        v_re = torch.einsum('br,blrq,bq->bl', xR, V_i, xR)

        return {'y_recon': y_hat, 'z_mu': z_mu, 'z_logvar': z_logvar,
                'mbar': mbar, 'v_re': v_re, 'log_vy': self.log_vy}

    @torch.no_grad()
    def encode_loader(self, loader):
        """
        Encoder pass over every visit of `loader`, in the order of ordered(loader).

        Args:
        - loader (DataLoader): a VisitDataset loader.

        Returns:
        - cache (dict): mu_phi, logvar_phi [n_visits, L], xS, xR and sid, on the model's
          device.
        """
        device = self._log_vy.device
        cache = {k: [] for k in ('mu_phi', 'logvar_phi', 'xS', 'xR', 'sid')}
        for batch in ordered(loader):
            mu_phi, logvar_phi = self.encoder(batch['observations'].to(device))
            xS, xR = self.split_x(batch['covariates'].to(device))
            cache['mu_phi'].append(mu_phi)
            cache['logvar_phi'].append(logvar_phi)
            cache['xS'].append(xS)
            cache['xR'].append(xR)
            cache['sid'].append(batch['subject'].to(device))
        return {k: torch.cat(v) for k, v in cache.items()}

    @torch.no_grad()
    def e_step(self, cache, XtX_R, gamma_start, sweeps):
        """
        closed_forms.e_step at the current prior, under the model's posterior.

        Args:
        - cache (dict): an encoder pass, from encode_loader.
        - XtX_R (Tensor): X_R,i^T X_R,i of the cache's subjects. [N, dR, dR]
        - gamma_start (Tensor): where the mean-field sweeps start. [N, G]
        - sweeps (int): number of mean-field sweeps.

        Returns:
        - gamma, m_g, V, score (Tensor): as closed_forms.e_step.
        """
        _, B_g, _ = self.prior.class_coefficients()
        return e_step(cache, XtX_R, pi=self.prior.pi, B_g=B_g, D=self.prior.D,
                      sigma_eps2=self.prior.sigma_eps2, posterior=self.posterior,
                      gamma_start=gamma_start, sweeps=sweeps)


def build_model(config, K, widths, G=None):
    """
    Args:
    - config (dict): run config.
    - K (int): number of markers.
    - widths (dict): 'random' (dR) and 'fixed' (dS), from get_dataloaders.
    - G (int): number of classes of the LCMMVAENet; None builds the G=1 LMMVAENet.

    Returns:
    - model (LMMVAENet or LCMMVAENet): on the CPU.
    """
    L = config['model']['latent_dim']
    encoder = Encoder(input_dim=K, hidden_dims=config['model']['encoder_hidden_dims'],
                      latent_dim=L)
    decoder = Decoder(input_dim=K, latent_dim=L,
                      hidden_dims=config['model']['decoder_hidden_dims'])
    if G is None:
        prior = LMMPrior(latent_dim=L, condition_dim=widths['random'] + widths['fixed'])
        return LMMVAENet(encoder, decoder, prior, K, vy_init=config['model']['vy_init'])
    prior = LCMMPrior(G=G, L=L, dS=widths['fixed'], dR=widths['random'],
                      sigma_eps2=config['model']['sigma_eps2'],
                      d_init=config['model']['d_init'],
                      min_D_diag=config['model']['min_D_diag'])
    return LCMMVAENet(encoder, decoder, prior, K, posterior=config['model']['posterior'],
                      vy_init=config['model']['vy_init'])
