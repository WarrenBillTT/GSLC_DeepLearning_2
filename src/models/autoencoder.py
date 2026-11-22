"""Vanilla Autoencoder, Sparse Autoencoder, and Variational Autoencoder (MLP-based).

Encoder: 784 -> 256 -> latent   (same hidden layout as the MLP classifier 784-256-64,
so that the architectural similarity argued in Section 2 of the assignment is literal).
"""
import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, latent_dim: int = 32, hidden: int = 256, out_act: str = "none"):
        super().__init__()
        self.fc1 = nn.Linear(784, hidden)
        self.fc2 = nn.Linear(hidden, latent_dim)
        self.out_act = out_act  # "relu" for the sparse AE so the L1 penalty acts on a non-negative code

    def forward(self, x):
        h = torch.relu(self.fc1(x.flatten(1)))
        z = self.fc2(h)
        return torch.relu(z) if self.out_act == "relu" else z


class Decoder(nn.Module):
    def __init__(self, latent_dim: int = 32, hidden: int = 256):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim, hidden)
        self.fc2 = nn.Linear(hidden, 784)

    def forward(self, z):
        return torch.sigmoid(self.fc2(torch.relu(self.fc1(z)))).view(-1, 1, 28, 28)


class Autoencoder(nn.Module):
    """Deterministic AE trained with pixel-wise MSE."""

    def __init__(self, latent_dim: int = 32, hidden: int = 256):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = Encoder(latent_dim, hidden)
        self.decoder = Decoder(latent_dim, hidden)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z

    def loss(self, x, lam: float = 0.0):
        xh, z = self(x)
        rec = ((xh - x) ** 2).flatten(1).mean(1).mean()
        return rec, {"rec": rec.item()}


class SparseAutoencoder(Autoencoder):
    """AE + L1 penalty on the latent code:  L = MSE + lambda * ||h||_1.

    latent_dim=64  -> sparse *under*complete (bottleneck)
    latent_dim=256 -> sparse *over*complete   (sparsity is the only regularizer)
    """

    def __init__(self, latent_dim: int = 64, hidden: int = 256):
        super().__init__(latent_dim, hidden)
        self.encoder = Encoder(latent_dim, hidden, out_act="relu")

    def loss(self, x, lam: float = 1e-4):
        xh, z = self(x)
        rec = ((xh - x) ** 2).flatten(1).mean(1).mean()
        l1 = z.abs().sum(1).mean()
        return rec + lam * l1, {"rec": rec.item(), "l1": l1.item()}

    @torch.no_grad()
    def sparsity(self, loader, device, thr: float = 1e-3) -> float:
        """Fraction of (near-)zero latent units, averaged over a loader."""
        tot, n = 0.0, 0
        for x, _ in loader:
            z = self.encoder(x.to(device))
            tot += (z.abs() < thr).float().mean(1).sum().item()
            n += x.size(0)
        return tot / n


class VAE(nn.Module):
    """VAE with Gaussian posterior, N(0,I) prior, Bernoulli likelihood.

    L = BCE(x_hat, x) summed over pixels + beta * KL(q(z|x) || N(0,I)).
    """

    def __init__(self, latent_dim: int = 32, hidden: int = 256, beta: float = 1.0):
        super().__init__()
        self.latent_dim, self.beta = latent_dim, beta
        self.fc1 = nn.Linear(784, hidden)
        self.fc_mu = nn.Linear(hidden, latent_dim)
        self.fc_logvar = nn.Linear(hidden, latent_dim)
        self.decoder = Decoder(latent_dim, hidden)

    def encode(self, x):
        h = torch.relu(self.fc1(x.flatten(1)))
        return self.fc_mu(h), self.fc_logvar(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)  # reparameterization trick
        return self.decoder(z), mu, logvar

    def loss(self, x, lam: float = 0.0):
        xh, mu, logvar = self(x)
        bce = nn.functional.binary_cross_entropy(xh.flatten(1), x.flatten(1), reduction="none").sum(1).mean()
        kl = (-0.5 * (1 + logvar - mu ** 2 - logvar.exp()).sum(1)).mean()
        return bce + self.beta * kl, {"bce": bce.item(), "kl": kl.item()}

    @torch.no_grad()
    def reconstruct(self, x):
        """Deterministic reconstruction through the posterior mean."""
        mu, _ = self.encode(x)
        return self.decoder(mu)

    @torch.no_grad()
    def sample(self, n: int, device):
        return self.decoder(torch.randn(n, self.latent_dim, device=device))
