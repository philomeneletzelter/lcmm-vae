"""
Training loops for the LMM-VAE (G=1 pretraining) and the LCMM-VAE (variational EM). An
LCMM-VAE epoch:

    E-step              one encoder pass -> cache -> V, m, gamma
    M-step, closed      pi, D, B^(g) on the same cache
    M-step, gradient    minibatches, gamma/m/V frozen, Adam on encoder, decoder, sigma_y^2
"""
import os
import warnings

import numpy as np
import torch

from src.datasets import ordered
from src.closed_forms import (recon_loglik, kappa, kl_class, kl_re, V_star, xtx_per_subject,
                              class_posterior, shared_mean, m_step)


def pretrain_loss(model, batch, beta, device):
    """
    Negative ELBO of the LMM-VAE on one batch.

    Args:
    - model (LMMVAENet): the G=1 model.
    - batch (dict): a VisitDataset batch.
    - beta (float): weight of the latent KL.
    - device (str): torch device.

    Returns:
    - metrics (dict): 'loss', 'recon', 'kl_latent', 'kl_A' (per-visit means).
    """
    y = batch['observations'].to(device)
    mask = batch['mask'].to(device)
    out = model(y, batch['covariates'].to(device))
    z_mu, z_logvar = out['z_mu'], out['z_logvar']
    recon_loss = -recon_loglik(y, out['y_recon'], out['log_vy'], mask).mean()

    prior_mu, prior_var = out['batch_prior_mu'], out['batch_prior_var']
    A_mu, A_logvar = out['global_A_mu'], out['global_A_logvar']
    kl_per_dim = -0.5 * (1 + z_logvar - (z_mu - prior_mu)**2 - prior_var - z_logvar.exp())  # [B, L]
    kl_latent = torch.mean(torch.sum(kl_per_dim, dim=1), dim=0)
    kl_A = -(0.5 * (1 + A_logvar - A_mu**2 - A_logvar.exp())).sum()

    loss = recon_loss + beta * kl_latent + kl_A
    return {'loss': loss, 'recon': recon_loss.detach(), 'kl_latent': kl_latent.detach(),
            'kl_A': kl_A.detach()}


def lcmm_loss(model, batch, q, device):
    """
    Negative ELBO of the LCMM-VAE on one batch at the frozen q(c) and q(a | c), as a
    per-visit mean on the scale of the G=1 loss. Terms (iii) and (iv) carry no gradient;
    they are per subject, so each visit carries 1/n_i of them.

    Args:
    - model (LCMMVAENet): the model.
    - batch (dict): a VisitDataset batch.
    - q (dict): gamma [N, G], m_g [G, N, L, dR], V [N, L, dR, dR], n_i [N] and the
      gamma-weighted random-effect KL kl_re_i [N] of the training subjects.
    - device (str): torch device.

    Returns:
    - metrics (dict): 'loss', 'recon', 'kl_latent', 'kl_class', 'kl_re' (per-visit means).
    """
    y = batch['observations'].to(device)
    sid = batch['subject'].to(device)
    mask = batch['mask'].to(device)
    out = model(y, batch['covariates'].to(device), sid, q['m_g'], q['V'])

    ll = recon_loglik(y, out['y_recon'], out['log_vy'], mask)                 # [B]
    recon_loss = -ll.mean()

    kappa_ijgl = kappa(out['z_mu'], out['z_logvar'], out['mbar'], out['v_re'],
                       model.prior.sigma_eps2)                               # [B, G, L]
    gamma_ijg = q['gamma'][sid].detach()
    kl_latent = torch.einsum('bg,bgl->b', gamma_ijg, kappa_ijgl).mean()

    visit_weight = 1.0 / q['n_i'][sid]
    kl_c = (visit_weight * kl_class(gamma_ijg, model.prior.pi)).mean()
    kl_a = (visit_weight * q['kl_re_i'][sid]).mean()

    loss = recon_loss + kl_latent + kl_c + kl_a
    return {'loss': loss, 'recon': recon_loss.detach(), 'kl_latent': kl_latent.detach(),
            'kl_class': kl_c.detach(), 'kl_re': kl_a.detach()}


def _gradient_epoch(optimiser, loader, loss_fn):
    """
    One pass of Adam over the minibatches of `loader`.

    Returns:
    - metrics (dict): batch means of the loss and its terms.
    """
    totals = {}
    for batch in loader:
        optimiser.zero_grad()
        out = loss_fn(batch)
        out['loss'].backward()
        optimiser.step()
        for name, value in out.items():
            totals[name] = totals.get(name, 0.0) + value.item()
    return {name: total / len(loader) for name, total in totals.items()}


def _save(model, history, save_dir):
    if save_dir is None:
        return
    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, 'loss_metrics.npy'), history)
    torch.save(model.state_dict(), os.path.join(save_dir, 'model_final.pt'))


def _log(history, metrics, epoch, epochs, beta):
    for name, value in metrics.items():
        history.setdefault(f'train_{name}', []).append(value)
    row = ", ".join(f"{name} {value:.4f}" for name, value in metrics.items()
                    if not isinstance(value, list))
    print(f"Epoch {epoch}/{epochs}, beta {beta:.3f} | train {row}")


def pretrain(model, optimiser, loader, epochs, kl_annealing_epochs, save_dir=None,
             device='cpu'):
    """
    Train the G=1 LMM-VAE with a linear KL warm-up from 0 (Bowman et al. 2016). Saves
    loss_metrics.npy and model_final.pt to save_dir.

    Args:
    - model (LMMVAENet): the model, moved to device.
    - optimiser (torch.optim.Optimizer): over the model's parameters.
    - loader (DataLoader): the training visits, shuffled.
    - epochs (int): number of epochs.
    - kl_annealing_epochs (int): warm-up length of the latent KL weight; 0 disables it.
    - save_dir (str): output directory; None saves nothing.
    - device (str): torch device.

    Returns:
    - history (dict): per-epoch metrics, keyed 'train_<name>'.
    """
    model.to(device)
    history = {}
    for epoch in range(1, epochs + 1):
        beta = (min(1.0, (epoch - 1) / kl_annealing_epochs) if kl_annealing_epochs > 0
                else 1.0)
        metrics = _gradient_epoch(optimiser, loader,
                                  lambda b: pretrain_loss(model, b, beta, device))
        _log(history, metrics, epoch, epochs, beta)
    _save(model, history, save_dir)
    return history


@torch.no_grad()
def _closed_form_block(model, cache, XtX_R, gamma_prev, sweeps):
    """
    E-step, closed-form M-step, then q(a | c) recomputed at the new prior for the gradient
    step.

    Args:
    - model (LCMMVAENet): the model; its prior is written.
    - cache (dict): one encoder pass, from LCMMVAENet.encode_loader.
    - XtX_R (Tensor): X_R,i^T X_R,i of the training subjects. [N, dR, dR]
    - gamma_prev (Tensor): responsibilities before the block. [N, G]
    - sweeps (int): mean-field E-step sweeps.

    Returns:
    - gamma (Tensor): [N, G]
    - V (Tensor): [N, L, dR, dR]
    - m_g (Tensor): [G, N, L, dR]
    """
    prior = model.prior
    mu, xS, xR, sid = (cache[k] for k in ('mu_phi', 'xS', 'xR', 'sid'))

    gamma, m_g, V, _ = model.e_step(cache, XtX_R, gamma_prev, sweeps)

    _, B_prev, _ = prior.class_coefficients()
    pi, D, B_new, empty = m_step(gamma, mu, xS, xR, sid, m_g, V, B_prev)
    prior.set_pi_(pi)
    prior.set_D_(D)
    if empty.any():
        warnings.warn(f"class(es) {torch.nonzero(empty).flatten().tolist()}: singular "
                      f"x_S^T x_S (empty class or rank-deficient design), B^(g) unchanged")
    prior.set_from_class_coefficients_(B_new)
    _, B_g, _ = prior.class_coefficients()

    V, m_g = class_posterior(mu, xS, xR, sid, XtX_R, B_g, prior.D, prior.sigma_eps2)
    if model.posterior == 'meanfield':
        m_g = shared_mean(gamma, m_g)
    return gamma, V, m_g


def train(model, optimiser, loader, epochs, tol, estep_sweeps, estep_sweeps_infer,
          save_dir=None, device='cpu'):
    """
    Train the LCMM-VAE. Each epoch runs the E-step and closed-form M-step on one cached
    encoder pass, then a gradient epoch with gamma, m and V frozen; KL weight 1. A final
    closed-form block runs at the final encoder. Saves loss_metrics.npy, model_final.pt and
    gamma.pt (training responsibilities, mean-field sweeps from pi) to save_dir.

    Args:
    - model (LCMMVAENet): the initialised model, moved to device.
    - optimiser (torch.optim.Optimizer): over the model's parameters.
    - loader (DataLoader): the training visits, shuffled.
    - epochs (int): number of epochs, or the cap when tol is set.
    - tol (float): stop once fewer than this fraction of subjects change class between
      consecutive E-steps (DEC, Xie et al. 2016); None runs every epoch.
    - estep_sweeps (int): mean-field E-step sweeps per epoch.
    - estep_sweeps_infer (int): mean-field E-step sweeps of the saved gamma.
    - save_dir (str): output directory; None saves nothing.
    - device (str): torch device.

    Returns:
    - history (dict): per-epoch metrics, keyed 'train_<name>'.
    """
    model.to(device)
    prior = model.prior
    N = loader.dataset.n_subjects
    xR, sid = [], []
    for batch in ordered(loader):
        xR.append(model.split_x(batch['covariates'].to(device))[1])
        sid.append(batch['subject'].to(device))
    xR, sid = torch.cat(xR), torch.cat(sid)
    XtX_R = xtx_per_subject(xR, sid, N)                                  # [N, dR, dR]
    n_i = torch.bincount(sid, minlength=N).to(xR.dtype)                  # [N]

    # before the first E-step: gamma at pi, m^(g) = 0, V at the initial D
    gamma = prior.pi.detach().unsqueeze(0).expand(N, -1).clone()        # [N, G]
    V = V_star(XtX_R, prior.D, prior.sigma_eps2)                        # [N, L, dR, dR]
    m_g = V.new_zeros(prior.G, N, prior.L, prior.dR)                    # [G, N, L, dR]

    history, prev_hard = {}, None
    for epoch in range(1, epochs + 1):
        gamma, V, m_g = _closed_form_block(model, model.encode_loader(loader), XtX_R, gamma,
                                           estep_sweeps)
        hard = gamma.argmax(1)
        usage = torch.bincount(hard, minlength=gamma.shape[1]).tolist()
        # stop tested from epoch 2, as DEC's ite > 0
        switched = ((hard != prev_hard).float().mean().item()
                    if prev_hard is not None else 1.0)
        prev_hard = hard.clone()
        print(f"  [gamma] usage={usage}  switched={switched:.3f}")

        q = dict(gamma=gamma, m_g=m_g, V=V, n_i=n_i,
                 kl_re_i=(gamma.detach() * kl_re(m_g, V, prior.D)).sum(1))  # [N]
        metrics = _gradient_epoch(optimiser, loader, lambda b: lcmm_loss(model, b, q, device))
        _log(history, dict(metrics, gamma_usage=usage, gamma_switched=switched), epoch,
             epochs, 1.0)
        if tol is not None and switched < float(tol):
            print(f"  [stop] converged at epoch {epoch} of {epochs}")
            break

    _closed_form_block(model, model.encode_loader(loader), XtX_R, gamma, estep_sweeps)
    _save(model, history, save_dir)
    if save_dir is not None:
        pi_start = prior.pi.detach().unsqueeze(0).expand(N, -1).clone()
        gamma = model.e_step(model.encode_loader(loader), XtX_R, pi_start,
                             estep_sweeps_infer)[0]
        torch.save(gamma.cpu(), os.path.join(save_dir, 'gamma.pt'))
    return history
