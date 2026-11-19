"""Per-visit Gaussian encoder."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    """
    Per-visit diagonal-Gaussian encoder q_phi(z_ij | y_ij): MLP input_dim -> hidden_dims
    -> latent_dim, ELU.

    Args:
    - input_dim (int): number of markers K.
    - hidden_dims (list): hidden layer widths.
    - latent_dim (int): dimension of z.
    - min_logvar (float): softplus floor on the log-variance.
    """

    def __init__(self, input_dim, hidden_dims, latent_dim, min_logvar=-8.0):
        super().__init__()
        self.register_buffer('min_logvar', torch.tensor(min_logvar))

        layers = []
        in_features = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(in_features, h))
            layers.append(nn.ELU())
            in_features = h
        self.backbone = nn.Sequential(*layers)

        self.mu_head = nn.Linear(in_features, latent_dim)
        self.logvar_head = nn.Linear(in_features, latent_dim)

    def forward(self, y):
        """
        Args:
        - y (Tensor): standardised observations, hidden cells at 0. [B, input_dim]

        Returns:
        - mu (Tensor): posterior mean. [B, latent_dim]
        - logvar (Tensor): posterior log-variance, floored. [B, latent_dim]
        """
        h = self.backbone(y)
        mu = self.mu_head(h)
        logvar = self.min_logvar + F.softplus(self.logvar_head(h) - self.min_logvar)
        return mu, logvar
