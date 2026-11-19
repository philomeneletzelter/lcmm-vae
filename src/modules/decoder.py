"""Per-visit decoder f_theta."""
import torch.nn as nn


class Decoder(nn.Module):
    """
    Per-visit decoder mean: MLP latent_dim -> hidden_dims -> input_dim, ELU, linear output.

    Args:
    - input_dim (int): number of markers K.
    - latent_dim (int): dimension of z.
    - hidden_dims (list): hidden layer widths.
    """

    def __init__(self, input_dim, latent_dim, hidden_dims):
        super().__init__()
        layers = []
        in_features = latent_dim
        for h in hidden_dims:
            layers.append(nn.Linear(in_features, h))
            layers.append(nn.ELU())
            in_features = h
        self.backbone = nn.Sequential(*layers)
        self.output_head = nn.Linear(in_features, input_dim)

    def forward(self, z):
        """
        Args:
        - z (Tensor): latent codes. [B, latent_dim]

        Returns:
        - y_hat (Tensor): reconstructed observations. [B, input_dim]
        """
        return self.output_head(self.backbone(z))
