"""Vanilla GAN (fully-connected) and DCGAN (convolutional). Images live in [-1, 1] (tanh output).

Both use the non-saturating generator loss  L_G = -E[log D(G(z))]  and BCE-with-logits for D
(the standard remedy for vanishing gradients of the original minimax objective).
"""
import torch
import torch.nn as nn

NZ = 100  # prior dimension z ~ N(0, I)


# ---------------------------------------------------------------- Vanilla GAN (MLP)
class MLPGenerator(nn.Module):
    def __init__(self, nz: int = NZ):
        super().__init__()
        self.nz = nz
        self.net = nn.Sequential(
            nn.Linear(nz, 256), nn.LeakyReLU(0.2),
            nn.Linear(256, 512), nn.LeakyReLU(0.2),
            nn.Linear(512, 1024), nn.LeakyReLU(0.2),
            nn.Linear(1024, 784), nn.Tanh(),
        )

    def forward(self, z):
        return self.net(z).view(-1, 1, 28, 28)


class MLPDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(784, 1024), nn.LeakyReLU(0.2), nn.Dropout(0.3),
            nn.Linear(1024, 512), nn.LeakyReLU(0.2), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.LeakyReLU(0.2), nn.Dropout(0.3),
            nn.Linear(256, 1),  # logit
        )

    def forward(self, x):
        return self.net(x.flatten(1)).squeeze(1)


# ---------------------------------------------------------------- DCGAN
class DCGenerator(nn.Module):
    """z -> 128x7x7 -> ConvT(64, 14x14) -> ConvT(1, 28x28)."""

    def __init__(self, nz: int = NZ):
        super().__init__()
        self.nz = nz
        self.fc = nn.Sequential(nn.Linear(nz, 128 * 7 * 7, bias=False), nn.BatchNorm1d(128 * 7 * 7), nn.ReLU(True))
        self.net = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False), nn.BatchNorm2d(64), nn.ReLU(True),  # 14x14
            nn.ConvTranspose2d(64, 1, 4, 2, 1), nn.Tanh(),                                        # 28x28
        )

    def forward(self, z):
        return self.net(self.fc(z).view(-1, 128, 7, 7))


class DCDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 64, 4, 2, 1), nn.LeakyReLU(0.2, True),                          # 14x14
            nn.Conv2d(64, 128, 4, 2, 1, bias=False), nn.BatchNorm2d(128), nn.LeakyReLU(0.2, True),  # 7x7
            nn.Flatten(), nn.Linear(128 * 7 * 7, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(1)


def dcgan_init(m: nn.Module) -> None:
    """N(0, 0.02) weight init from the DCGAN paper."""
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        nn.init.normal_(m.weight, 0.0, 0.02)
    elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)


def build_gan(kind: str, nz: int = NZ):
    if kind == "gan":
        return MLPGenerator(nz), MLPDiscriminator()
    if kind == "dcgan":
        g, d = DCGenerator(nz), DCDiscriminator()
        g.apply(dcgan_init)
        d.apply(dcgan_init)
        return g, d
    raise ValueError(kind)
