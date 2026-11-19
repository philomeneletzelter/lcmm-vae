"""Latent priors: the LCMM prior and the G=1 pretraining prior."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class LCMMPrior(nn.Module):
    """
    Latent-class mixed-model prior over the latent trajectory.

        c_i     ~ Cat(pi)
        a_R,i,l ~ N(0, D_l),  l = 1..L
        z_ij    = (A_S + Delta^(c_i)) x_S,ij + A_R,i x_R,ij + eps_ij
        eps_ij  ~ N(0, sigma_eps^2 I_L)
        sum_g pi_g Delta^(g) = 0

    A_S holds the shared coefficients and Delta^(g) the class departures from A_S. pi, D,
    A_S and Delta are buffers written by the closed-form updates, never by the optimiser.

    Args:
    - G (int): number of latent classes.
    - L (int): latent dimension.
    - dS (int): width of x_S.
    - dR (int): width of x_R; 0 disables random effects (D is [L, 0, 0]).
    - sigma_eps2 (float): latent residual variance, fixed.
    - d_init (float): initial D_l = d_init * I.
    - min_D_diag (float): min_D_diag^2 is added to the diagonal of D_l at every write.
    """

    def __init__(self, G, L, dS, dR,
                 sigma_eps2=1.0, d_init=1.0, min_D_diag=1e-3):
        super().__init__()
        self.G, self.L, self.dS, self.dR = G, L, dS, dR

        self.register_buffer('pi_logits', torch.zeros(G))

        self.register_buffer('A_S', torch.zeros(L, dS))
        self.register_buffer('Delta', torch.zeros(G, L, dS))

        self.register_buffer('min_D_diag', torch.tensor(float(min_D_diag)))
        self.register_buffer('_D', d_init * torch.eye(dR).repeat(L, 1, 1))

        self.register_buffer('sigma_eps2', torch.tensor(float(sigma_eps2)))

    @property
    def pi(self):
        """Class prevalences. [G]"""
        return F.softmax(self.pi_logits, dim=-1)

    @property
    def D(self):
        """Random-effect covariance. [L, dR, dR]"""
        return self._D

    def class_coefficients(self):
        """
        Returns:
        - pi (Tensor): class prevalences. [G]
        - B_g (Tensor): A_S + Delta^(g). [G, L, dS]
        - Delta (Tensor): [G, L, dS]
        """
        return self.pi, self.A_S.unsqueeze(0) + self.Delta, self.Delta

    @torch.no_grad()
    def set_pi_(self, pi_new):
        """
        Args:
        - pi_new (Tensor): new prevalences. [G]
        """
        self.pi_logits.copy_(torch.log(pi_new.clamp_min(1e-12)))

    @torch.no_grad()
    def set_D_(self, D_new):
        """
        Write D, symmetrised, plus min_D_diag^2 on the diagonal.

        Args:
        - D_new (Tensor): new covariances. [L, dR, dR]
        """
        D_new = 0.5 * (D_new + D_new.transpose(-1, -2))
        eye = torch.eye(self.dR, device=D_new.device, dtype=D_new.dtype)
        self._D.copy_(D_new + (self.min_D_diag ** 2) * eye)

    @torch.no_grad()
    def set_from_class_coefficients_(self, B):
        """
        Write A_S (pi-weighted mean of B^(g)) and Delta = B^(g) - A_S.

        Args:
        - B (Tensor): class coefficients. [G, L, dS]
        """
        pi64 = self.pi.double()
        pi64 = pi64 / pi64.sum()
        A_S_new = torch.einsum('g,gls->ls', pi64, B.double())
        Delta = B.double() - A_S_new.unsqueeze(0)
        A_S_new = A_S_new.to(self.A_S.dtype)
        Delta = Delta.to(self.Delta.dtype)
        self.A_S.copy_(A_S_new)
        self.Delta.copy_(Delta)


class LMMPrior(nn.Module):
    """
    G=1 pretraining prior without random effects: z ~ N(A x, I) with a mean-field Gaussian
    q(A) and N(0, I) prior on A; the linear prior of Ong et al. (2025) without their subject
    terms.

    Args:
    - latent_dim (int): latent dimension L.
    - condition_dim (int): width of the covariate row x.
    """

    def __init__(self, latent_dim, condition_dim):
        super().__init__()
        self.register_buffer('min_logvar', -8.0 * torch.ones(1))

        self.A_mean = nn.Linear(condition_dim, latent_dim, bias=False)
        self.A_log_var = nn.Linear(condition_dim, latent_dim, bias=False)

    def forward(self, x):
        """
        Args:
        - x (Tensor): covariate rows. [B, condition_dim]

        Returns:
        - A_mu (Tensor): mean of A. [condition_dim, latent_dim]
        - A_logvar (Tensor): log-variance of A, floored. [condition_dim, latent_dim]
        - batch_prior_mu (Tensor): prior mean per row. [B, latent_dim]
        - batch_prior_var (Tensor): prior variance per row. [B, latent_dim]
        """
        A_mu = self.A_mean.weight.T
        A_logvar = self.min_logvar + F.softplus(self.A_log_var.weight.T - self.min_logvar)

        batch_prior_mu = x @ A_mu
        batch_prior_var = (x ** 2) @ A_logvar.exp()

        return A_mu, A_logvar, batch_prior_mu, batch_prior_var
