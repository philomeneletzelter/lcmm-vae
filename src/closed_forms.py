"""Closed-form E- and M-step updates; pure functions, no model state."""
import math

import torch


# indices: i subject, j visit, k marker, l latent coordinate, g class, r/s columns of x_R/x_S.
# suffixes name the indices carried: gamma_ig [N, G], gamma_ijg [B, G]; sid [B] maps rows
# to subjects. R = 0 needs no special case (random-effect tensors are empty).
# m is always [G, N, L, dR]; under meanfield its G slices are equal.
#   meanfield    q(c_i) q(a_i), one m_i shared by every class
#   structured   q(c_i) q(a_i | c_i), one m^(g)_i per class

POSTERIORS = ('meanfield', 'structured')


def xtx_per_subject(xR, sid, N):
    """
    Per-subject X_R^T X_R.

    Args:
    - xR (Tensor): random-effect design rows. [B, dR]
    - sid (Tensor): subject of each row. [B]
    - N (int): number of subjects.

    Returns:
    - XtX_R (Tensor): [N, dR, dR]
    """
    dR = xR.shape[1]
    return torch.zeros(N, dR, dR, device=xR.device, dtype=xR.dtype).index_add_(
        0, sid, torch.einsum('br,bq->brq', xR, xR))


def recon_loglik(y, y_hat, log_vy, mask):
    """
    Masked per-visit Gaussian reconstruction log-likelihood; empty visits contribute 0.

    Args:
    - y (Tensor): observations. [B, K]
    - y_hat (Tensor): decoder means. [B, K]
    - log_vy (Tensor): log sigma_y,k^2 after the softplus floor. [K]
    - mask (Tensor): 1 where observed. [B, K]

    Returns:
    - ll (Tensor): per-visit log-likelihood; sum over the batch for term (i). [B]
    """
    bracket = (y - y_hat) ** 2 / torch.exp(log_vy) + (math.log(2 * math.pi) + log_vy)
    return -0.5 * (bracket * mask).sum(dim=1)


def kappa(mu_phi, logvar_phi, mbar, v_re, sigma_eps2):
    """
    Latent regularisation kappa^(g)_ijl, before weighting by the responsibilities. The
    encoder-width term has no class index and cancels in the responsibility softmax.

    Args:
    - mu_phi (Tensor): encoder mean. [B, L]
    - logvar_phi (Tensor): encoder log-variance. [B, L]
    - mbar (Tensor): class trajectory with the random-effect mean m^(g)_i,l added. [B, G, L]
    - v_re (Tensor): x_R,ij^T V_i,l x_R,ij, with no class axis. [B, L]
    - sigma_eps2 (Tensor or float): latent residual variance.

    Returns:
    - kappa (Tensor): sum over l and over a subject's visits to get kappa_ig. [B, G, L]
    """
    ratio = logvar_phi.exp() / sigma_eps2                   # [B, L]
    centre = ((mu_phi.unsqueeze(1) - mbar) ** 2 + v_re.unsqueeze(1)) / (2.0 * sigma_eps2)
    width = 0.5 * (ratio - 1.0 - torch.log(ratio))          # [B, L]
    return centre + width.unsqueeze(1)                      # [B, G, L]


def kl_class(gamma, pi):
    """
    KL(gamma_i || pi) per subject.

    Args:
    - gamma (Tensor): responsibilities. [N, G]
    - pi (Tensor): class prevalences. [G]

    Returns:
    - kl (Tensor): [N]
    """
    return (gamma * (torch.log(gamma.clamp_min(1e-30)) - torch.log(pi.clamp_min(1e-30)))).sum(-1)


def kl_re_l(m, V, D):
    """
    KL( N(m_i,l, V_i,l) || N(0, D_l) ) per subject and latent coordinate.

    Args:
    - m (Tensor): random-effect means. [N, L, dR]
    - V (Tensor): random-effect covariances. [N, L, dR, dR]
    - D (Tensor): prior covariances, sym. PD. [L, dR, dR]

    Returns:
    - kl (Tensor): [N, L]
    """
    dR = m.shape[-1]
    Lc = torch.linalg.cholesky(D)                                    # [L, dR, dR]
    Lc_b = Lc.unsqueeze(0).expand(V.shape[0], -1, -1, -1)            # [N, L, dR, dR]
    DiV = torch.cholesky_solve(V, Lc_b)
    tr = DiV.diagonal(dim1=-2, dim2=-1).sum(-1)                      # [N, L]
    sol = torch.cholesky_solve(m.unsqueeze(-1), Lc_b).squeeze(-1)
    quad = (m * sol).sum(-1)                                         # [N, L]
    logdet_D = 2.0 * torch.log(Lc.diagonal(dim1=-2, dim2=-1)).sum(-1)          # [L]

    LV = torch.linalg.cholesky(V)                                              # [N, L, dR, dR]
    logdet_V = 2.0 * torch.log(LV.diagonal(dim1=-2, dim2=-1)).sum(-1)          # [N, L]
    return 0.5 * (tr + quad - dR - (logdet_V - logdet_D.unsqueeze(0)))


def V_star(XtX_R, D, sigma_eps2):
    """
    Posterior covariance V_i,l of the random effects.

    Args:
    - XtX_R (Tensor): X_R,i^T X_R,i. [N, dR, dR]
    - D (Tensor): random-effect covariance per latent coordinate. [L, dR, dR]
    - sigma_eps2 (float): latent residual variance.

    Returns:
    - V (Tensor): [N, L, dR, dR]
    """
    Lc = torch.linalg.cholesky(D)                                     # [L, dR, dR]
    eye = torch.eye(D.shape[-1], device=D.device, dtype=D.dtype)
    D_inv = torch.cholesky_solve(eye.expand_as(D), Lc)                # [L, dR, dR]
    prec = D_inv.unsqueeze(0) + XtX_R.unsqueeze(1) / sigma_eps2       # [N, L, dR, dR]
    prec = 0.5 * (prec + prec.transpose(-1, -2))                      # symmetrise fp drift
    Lp = torch.linalg.cholesky(prec)
    return torch.cholesky_solve(eye.expand_as(prec), Lp)              # [N, L, dR, dR]


def m_star(mu_phi, mfix, xR, sid, V, sigma_eps2):
    """
    Random-effect mean m_i,l: the mixed-model BLUP with the encoder mean as the response.

    Args:
    - mu_phi (Tensor): encoder means. [B, L]
    - mfix (Tensor): trajectory without the random effect. [B, L]
    - xR (Tensor): random-effect design. [B, dR]
    - sid (Tensor): subject of each visit. [B]
    - V (Tensor): from V_star. [N, L, dR, dR]
    - sigma_eps2 (float): latent residual variance.

    Returns:
    - m (Tensor): [N, L, dR]
    """
    L, dR = mu_phi.shape[1], xR.shape[1]
    resid = mu_phi - mfix                                             # [B, L]
    xR_resid = resid.unsqueeze(-1) * xR.unsqueeze(1)                  # [B, L, dR]
    sum_xR_resid = torch.zeros(V.shape[0], L, dR, device=mu_phi.device, dtype=mu_phi.dtype)
    sum_xR_resid.index_add_(0, sid, xR_resid)                         # sum_j, [N, L, dR]
    return torch.einsum('ilrq,ilq->ilr', V, sum_xR_resid) / sigma_eps2


def class_posterior(mu_phi, xS, xR, sid, XtX_R, B_g, D, sigma_eps2):
    """
    Structured posterior q(a_i | c_i = g) = N(m^(g)_i, V_i); shared_mean turns it into
    the mean-field one.

    Args:
    - mu_phi (Tensor): encoder means. [B, L]
    - xS (Tensor): fixed-effect design. [B, dS]
    - xR (Tensor): random-effect design. [B, dR]
    - sid (Tensor): subject of each visit. [B]
    - XtX_R (Tensor): X_R,i^T X_R,i. [N, dR, dR]
    - B_g (Tensor): class coefficients. [G, L, dS]
    - D (Tensor): random-effect covariances. [L, dR, dR]
    - sigma_eps2 (float): latent residual variance.

    Returns:
    - V (Tensor): the same for every class. [N, L, dR, dR]
    - m_g (Tensor): [G, N, L, dR]
    """
    fixed = torch.einsum('gls,bs->bgl', B_g, xS)                        # [B, G, L]
    V = V_star(XtX_R, D, sigma_eps2)
    m_g = torch.stack([m_star(mu_phi, fixed[:, g], xR, sid, V, sigma_eps2)
                       for g in range(B_g.shape[0])])
    return V, m_g


def kappa_kl(mu_phi, logvar_phi, xS, xR, sid, V, m_g, B_g, D, sigma_eps2):
    """
    ELBO terms (ii) kappa_ig and (iv) KL_ig per subject and class, each class at its own
    m^(g). At the structured m^(g), their sum is the class's marginal negative
    log-likelihood of mu_i up to terms free of g.

    Args:
    - mu_phi, logvar_phi (Tensor): encoder means and log-variances. [B, L]
    - xS (Tensor): fixed-effect design. [B, dS]
    - xR (Tensor): random-effect design. [B, dR]
    - sid (Tensor): subject of each visit. [B]
    - V, m_g (Tensor): the posterior. [N, L, dR, dR], [G, N, L, dR]
    - B_g (Tensor): class coefficients. [G, L, dS]
    - D (Tensor): random-effect covariances. [L, dR, dR]
    - sigma_eps2 (float): latent residual variance.

    Returns:
    - kappa_ig (Tensor): [N, G]
    - kl_ig (Tensor): [N, G]
    """
    mbar = (torch.einsum('gls,bs->bgl', B_g, xS)
            + torch.einsum('gblr,br->bgl', m_g[:, sid], xR))            # [B, G, L]
    v_re = torch.einsum('br,blrq,bq->bl', xR, V[sid], xR)               # [B, L]
    kappa_ijg = kappa(mu_phi, logvar_phi, mbar, v_re, sigma_eps2).sum(-1)   # [B, G]
    kappa_ig = kappa_ijg.new_zeros(V.shape[0], kappa_ijg.shape[1]).index_add_(0, sid,
                                                                              kappa_ijg)
    return kappa_ig, kl_re(m_g, V, D)


def kl_re(m_g, V, D):
    """
    ELBO term (iv) KL_ig per subject and class, summed over l.

    Args:
    - m_g (Tensor): [G, N, L, dR]
    - V (Tensor): [N, L, dR, dR]
    - D (Tensor): [L, dR, dR]

    Returns:
    - kl_ig (Tensor): [N, G]
    """
    return torch.stack([kl_re_l(m_g[g], V, D).sum(-1)
                        for g in range(m_g.shape[0])], 1)


def gamma_star(score, pi):
    """
    Responsibilities: softmax of log pi_g - score, the exact maximiser of the bound in gamma.

    Args:
    - score (Tensor): kappa_ig + KL_ig, per subject and class. [N, G]
    - pi (Tensor): class prevalences. [G]

    Returns:
    - gamma (Tensor): responsibilities. [N, G]
    """
    logits = torch.log(pi.clamp_min(1e-30)).unsqueeze(0) - score
    return torch.softmax(logits, dim=-1)


def shared_mean(gamma, m_g):
    """
    Mean-field random-effect mean: the gamma-weighted average of m^(g), repeated on the
    class axis.

    Args:
    - gamma (Tensor): responsibilities. [N, G]
    - m_g (Tensor): class-conditional means, from class_posterior. [G, N, L, dR]

    Returns:
    - m (Tensor): the same mean in every class slice. [G, N, L, dR]
    """
    m = torch.einsum('ng,gnlr->nlr', gamma, m_g)
    return m.unsqueeze(0).expand(m_g.shape[0], -1, -1, -1).contiguous()


def latent_mean(B_g, gamma, m_g, xS, xR, sid):
    """
    Posterior latent trajectory sum_g gamma_ig (B^(g) x_S + m_i^(g) x_R) at design rows.

    Args:
    - B_g (Tensor): class coefficients. [G, L, dS]
    - gamma (Tensor): responsibilities. [N, G]
    - m_g (Tensor): random-effect means. [G, N, L, dR]
    - xS, xR (Tensor): design rows. [M, dS], [M, dR]
    - sid (Tensor): subject of each row. [M]

    Returns:
    - z (Tensor): [M, L]
    """
    z_g = (torch.einsum('gls,ms->mgl', B_g, xS)
           + torch.einsum('gmlr,mr->mgl', m_g[:, sid], xR))                 # [M, G, L]
    return torch.einsum('mg,mgl->ml', gamma[sid], z_g)


def e_step(cache, XtX_R, pi, B_g, D, sigma_eps2, posterior, gamma_start=None, sweeps=None):
    """
    E-step at a fixed prior. structured: gamma at m^(g), exact in one pass. meanfield:
    `sweeps` rounds of m = shared_mean(gamma, m^(g)) then gamma, from gamma_start.

    Args:
    - cache (dict): mu_phi, logvar_phi [B, L], xS [B, dS], xR [B, dR] and sid [B]; other
      keys are ignored.
    - XtX_R (Tensor): from xtx_per_subject. [N, dR, dR]
    - pi (Tensor): class prevalences. [G]
    - B_g (Tensor): class coefficients. [G, L, dS]
    - D (Tensor): random-effect covariances. [L, dR, dR]
    - sigma_eps2 (float): latent residual variance.
    - posterior (str): one of POSTERIORS.
    - gamma_start (Tensor): meanfield only, the responsibilities the sweeps start
      from. [N, G]
    - sweeps (int): meanfield only, number of E-step sweeps, at least 1.

    Returns:
    - gamma (Tensor): responsibilities. [N, G]
    - m_g (Tensor): random-effect means the score was evaluated at. [G, N, L, dR]
    - V (Tensor): random-effect covariances. [N, L, dR, dR]
    - score (Tensor): kappa_ig + KL_ig. [N, G]

    Raises:
    - RuntimeError: if gamma is not finite.
    """
    mu_phi, logvar_phi, xS, xR, sid = (cache[k] for k in ('mu_phi', 'logvar_phi', 'xS', 'xR',
                                                          'sid'))
    V, m_g = class_posterior(mu_phi, xS, xR, sid, XtX_R, B_g, D, sigma_eps2)
    if posterior == 'structured':
        kappa_ig, kl_ig = kappa_kl(mu_phi, logvar_phi, xS, xR, sid, V, m_g, B_g, D,
                                   sigma_eps2)
        score = kappa_ig + kl_ig
        gamma = gamma_star(score, pi)
    else:
        gamma = gamma_start
        for _ in range(sweeps):
            m = shared_mean(gamma, m_g)
            kappa_ig, kl_ig = kappa_kl(mu_phi, logvar_phi, xS, xR, sid, V, m, B_g, D,
                                       sigma_eps2)
            score = kappa_ig + kl_ig
            gamma = gamma_star(score, pi)
        m_g = m
    if not torch.isfinite(gamma).all():
        raise RuntimeError("gamma is not finite: the fit diverged")
    return gamma, m_g, V, score


def pi_star(gamma):
    """
    Class prevalences: mean responsibility.

    Args:
    - gamma (Tensor): responsibilities. [N, G]

    Returns:
    - pi (Tensor): [G]
    """
    return gamma.mean(dim=0)


def D_star(gamma, m_g, V):
    """
    Closed-form update of the random-effect covariance D.

    Args:
    - gamma (Tensor): [N, G]
    - m_g (Tensor): [G, N, L, dR]
    - V (Tensor): [N, L, dR, dR]

    Returns:
    - D (Tensor): [L, dR, dR]
    """
    return (torch.einsum('ng,gnlr,gnlq->lrq', gamma, m_g, m_g) + V.sum(0)) / V.shape[0]


def B_star(gamma_ijg, mu_phi, xS, xR, m_g_ijl, B_prev):
    """
    Closed-form B^(g), one weighted least-squares solve per class; a class with a
    near-singular system keeps B_prev.

    Args:
    - gamma_ijg (Tensor): gamma_ig gathered onto each of subject i's visits. [B, G]
    - mu_phi (Tensor): encoder means. [B, L]
    - xS (Tensor): fixed-effect design rows. [B, dS]
    - xR (Tensor): random-effect design rows. [B, dR]
    - m_g_ijl (Tensor): m^(g)_i,l gathered onto each of subject i's visits. [G, B, L, dR]
    - B_prev (Tensor): current coefficients. [G, L, dS]

    Returns:
    - B (Tensor): the solved coefficients. [G, L, dS]
    - empty (Tensor): bool, marking the classes that kept B_prev. [G]
    """
    B, empty = [], []
    for g in range(B_prev.shape[0]):
        weight = gamma_ijg[:, g]                                          # [B]
        resid = mu_phi - torch.einsum('blr,br->bl', m_g_ijl[g], xR)       # [B, L]
        XtX = torch.einsum('b,bs,bt->st', weight, xS, xS)                 # [dS, dS]
        RHS = torch.einsum('b,bl,bs->ls', weight, resid, xS)              # [L, dS]
        eigvals = torch.linalg.eigvalsh(XtX)
        bad = bool(eigvals[-1] <= 0) or bool(eigvals[0] <= 1e-6 * eigvals[-1].clamp_min(0))
        B.append(B_prev[g] if bad else
                 torch.linalg.solve(XtX, RHS.unsqueeze(-1)).squeeze(-1))
        empty.append(bad)
    return torch.stack(B), torch.tensor(empty)


def m_step(gamma, mu_phi, xS, xR, sid, m_g, V, B_prev):
    """
    Closed-form M-step at a fixed encoder: pi_star, D_star and B_star.

    Args:
    - gamma (Tensor): responsibilities. [N, G]
    - mu_phi (Tensor): encoder means. [B, L]
    - xS (Tensor): fixed-effect design. [B, dS]
    - xR (Tensor): random-effect design. [B, dR]
    - sid (Tensor): subject of each visit. [B]
    - m_g, V (Tensor): the E-step's posterior. [G, N, L, dR], [N, L, dR, dR]
    - B_prev (Tensor): current coefficients, kept by unsolvable classes. [G, L, dS]

    Returns:
    - pi (Tensor): [G]
    - D (Tensor): [L, dR, dR]
    - B (Tensor): [G, L, dS]
    - empty (Tensor): bool, the classes that kept B_prev. [G]
    """
    B, empty = B_star(gamma[sid], mu_phi, xS, xR, m_g[:, sid], B_prev)
    return pi_star(gamma), D_star(gamma, m_g, V), B, empty
