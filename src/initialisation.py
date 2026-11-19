"""
Initialisation: fit a latent LCMM on the frozen G=1 encoder means by em-EM + EM.
The E-step is the structured one whatever the model's posterior (exact for the LCMM).
"""
import math
import warnings

import numpy as np
import torch

from src.closed_forms import xtx_per_subject, e_step, m_step, B_star
from src.model import build_model
from src.utils import set_seed

# Rmixmod defaults
EM_EM = dict(max_iter=5, eps=1e-3, min_iter=1)
EM = dict(max_iter=200, eps=1e-3, min_iter=2)

G1_EM = dict(max_iter=2000, eps=1e-3, min_iter=2)


def _loglik(codes, pi, B, D, sigma_eps2):
    """
    E-step of the latent LCMM (encoder variance set to sigma_eps^2) and its observed
    log-likelihood.

    Args:
    - codes (dict): mu_phi, xS, xR, sid and XtX_R, float64 on the CPU.
    - pi (Tensor): class proportions. [G]
    - B (Tensor): class coefficients B^(g). [G, L, dS]
    - D (Tensor): random-effect covariance. [L, R, R]
    - sigma_eps2 (float): latent residual variance.

    Returns:
    - gamma (Tensor): responsibilities. [N, G]
    - m_g (Tensor): class-conditional random-effect means. [G, N, L, R]
    - V (Tensor): random-effect covariances. [N, L, R, R]
    - loglik (float): observed log-likelihood.
    """
    logvar = torch.full_like(codes['mu_phi'], math.log(sigma_eps2))
    gamma, m_g, V, score = e_step(dict(codes, logvar_phi=logvar), codes['XtX_R'], pi=pi,
                                  B_g=B, D=D, sigma_eps2=sigma_eps2, posterior='structured')
    loglik = (torch.logsumexp(torch.log(pi.clamp_min(1e-30)) - score, 1).sum()
              - 0.5 * codes['mu_phi'].numel() * math.log(2 * math.pi * sigma_eps2))
    return gamma, m_g, V, float(loglik)


def _em(codes, pi, B, D, sigma_eps2, max_iter, eps, min_iter):
    """
    EM from (pi, B, D), stopped at max_iter or at an absolute log-likelihood change below
    eps after min_iter iterations.

    Args:
    - codes (dict): mu_phi, xS, xR, sid and XtX_R.
    - pi, B, D (Tensor): the starting values. [G], [G, L, dS], [L, R, R]
    - sigma_eps2 (float): latent residual variance.
    - max_iter (int): iteration cap.
    - eps (float): tolerance on the log-likelihood change.
    - min_iter (int): iterations before the tolerance is tested.

    Returns:
    - fit (dict): pi, B, D, gamma, loglik, converged, degenerate (some B^(g) solve failed).
    """
    gamma, m_g, V, ll = _loglik(codes, pi, B, D, sigma_eps2)
    degenerate = False
    for it in range(1, max_iter + 1):
        pi, D, B, bad = m_step(gamma, codes['mu_phi'], codes['xS'], codes['xR'], codes['sid'],
                               m_g, V, B)
        degenerate |= bool(bad.any())
        gamma, m_g, V, ll_new = _loglik(codes, pi, B, D, sigma_eps2)
        step, ll = abs(ll_new - ll), ll_new
        if it > min_iter and step < eps:
            break
    return dict(pi=pi, B=B, D=D, gamma=gamma, loglik=ll, converged=step < eps,
                degenerate=degenerate)


@torch.no_grad()
def initialise_from_pretrained(pretrained, model, train_loader, seed, init_ridge, init_tries):
    """
    Initialisation by em-EM (Biernacki, Celeux & Govaert 2003): init_tries short EM runs
    (EM_EM), the best continued by EM.

        1. Copy encoder, decoder and sigma_y^2; encode every training visit.
        2. Per-subject ridge regression of mu_phi on x_S gives candidate centres B_i.
        3. G=1 latent LMM to convergence; gives D.
        4. em-EM then EM: pi = 1/G, B^(g) = B_i of G random subjects, D from step 3.
        5. Write pi, then B^(g) (centring), then D.

    Args:
    - pretrained (nn.Module): a trained G=1 model (LMMVAENet, or LCMMVAENet at G=1).
    - model (LCMMVAENet): the model to initialise, modified in place.
    - train_loader (DataLoader): the training visits.
    - seed (int): seed for the subjects drawn by em-EM.
    - init_ridge (float): ridge of the per-subject regressions.
    - init_tries (int): number of em-EM runs.

    Returns:
    - model (LCMMVAENet): the same model, initialised.
    """
    model.encoder.load_state_dict(pretrained.encoder.state_dict())
    model.decoder.load_state_dict(pretrained.decoder.state_dict())
    model._log_vy.copy_(pretrained._log_vy)

    float64 = torch.float64
    cache = model.encode_loader(train_loader)
    sid, mu = cache['sid'].cpu(), cache['mu_phi'].cpu().to(float64)
    xS, xR = cache['xS'].cpu().to(float64), cache['xR'].cpu().to(float64)
    N, G = train_loader.dataset.n_subjects, model.prior.G
    L, dS = mu.shape[1], xS.shape[1]
    codes = dict(mu_phi=mu, xS=xS, xR=xR, sid=sid, XtX_R=xtx_per_subject(xR, sid, N))

    W_i = torch.zeros(N, dS, dS, dtype=float64).index_add_(
        0, sid, torch.einsum('bs,bt->bst', xS, xS))                      # [N, dS, dS]
    u_i = torch.zeros(N, dS, L, dtype=float64).index_add_(
        0, sid, torch.einsum('bs,bl->bsl', xS, mu))                      # [N, dS, L]
    B_i = torch.linalg.solve(W_i + init_ridge * torch.eye(dS, dtype=float64), u_i).transpose(1, 2)

    sigma_eps2 = float(model.prior.sigma_eps2)
    # G=1 start: gamma = 1 and m = 0 make the B update pooled least squares
    n_obs, R = sid.shape[0], xR.shape[1]
    B1, _ = B_star(torch.ones(n_obs, 1, dtype=float64), mu, xS, xR,
                   torch.zeros(1, n_obs, L, R, dtype=float64), B_i[:1])
    best = _em(codes, torch.ones(1, dtype=float64), B1, model.prior.D.cpu().to(float64),
               sigma_eps2, **G1_EM)
    if not best['converged']:
        warnings.warn(f"the G=1 latent LMM did not converge in {G1_EM['max_iter']} "
                      f"iterations")
    D0 = best['D']
    if G > 1:
        rng = np.random.default_rng(seed)
        best = None
        for _ in range(init_tries):
            idx = torch.as_tensor(rng.choice(N, G, replace=False))
            run = _em(codes, torch.full((G,), 1.0 / G, dtype=float64), B_i[idx], D0,
                      sigma_eps2, **EM_EM)
            # a run whose class empties or whose solve is singular is dropped
            if not run['degenerate'] and (best is None or run['loglik'] > best['loglik']):
                best = run
        if best is None:
            raise RuntimeError(
                "em-EM: in every run a class had a near-singular fixed-effect design "
                "(x_S^T x_S); usual causes: spline basis functions with few visits (reduce "
                "n_basis, use knots: quantile, or restrict t to the follow-up range) or too "
                "many covariates")
    fit = _em(codes, best['pi'], best['B'], best['D'], sigma_eps2, **EM)
    print(f"  [init] em-EM + EM: loglik {fit['loglik']:.1f}"
          + ("" if fit['converged'] else f", stopped at {EM['max_iter']} iterations")
          + f", subjects per class "
            f"{torch.bincount(fit['gamma'].argmax(1), minlength=G).tolist()}")
    if fit['degenerate']:
        warnings.warn("a class kept its previous B^(g) during the final EM of the initialisation "
                      "(empty class or singular x_S^T x_S)")

    float32 = dict(dtype=torch.float32, device=model._log_vy.device)
    model.prior.set_pi_(fit['pi'].to(**float32))
    model.prior.set_from_class_coefficients_(fit['B'].to(**float32))
    model.prior.set_D_(fit['D'].to(**float32))
    return model


def initialise(config, pretrained_model, train_loader, widths, G, seed, device='cpu'):
    """
    Build a G-class LCMMVAENet and initialise it from a trained G=1 model.

    Args:
    - config (dict): run config.
    - pretrained_model (nn.Module): trained G=1 model (LMMVAENet, or LCMMVAENet at G=1).
    - train_loader (DataLoader): the training visits.
    - widths (dict): from get_dataloaders.
    - G (int): number of classes.
    - seed (int): training seed.
    - device (str): torch device.

    Returns:
    - model (LCMMVAENet): initialised by em-EM.
    """
    set_seed(seed)
    model = build_model(config, train_loader.dataset.n_markers, widths, G).to(device)
    return initialise_from_pretrained(pretrained_model, model, train_loader, seed=seed,
                                      init_ridge=config['model']['init_ridge'],
                                      init_tries=config['model']['init_tries'])
